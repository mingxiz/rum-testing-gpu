"""The RUM class: the distance from observed choice probabilities to the random utility model.

    from solver.rum_solver import RUM
    rum = RUM(menu, alternative, choice_probability)      # one row per observed (menu, alternative)
    rum = RUM.from_rows(cd, n)                            # the same from an (M, 3) array of rows
    res = rum.calculate_distance()                        # defaults: solver/defaults.py
    res = rum.calculate_distance(abs_tol=1e-6, rel_tol=0, method="bm_admm",
                                 method_args={"check": 100}, max_time=600)
    print(res)                                            # exit_flag, distance, lb, ub, gaps, times, ...

Input. Row i: menu[i] is the bitmask of the offered menu (bit a set = alternative a offered),
alternative[i] the chosen alternative (0-based), choice_probability[i] its observed probability.
A menu need not have a row for every alternative (partly observed menus are fine).

Distance. dist(p) = min ||q - p|| over the choice probabilities q of random utility models, on the
observed rows; ||x||^2 = x' vcov x when a weight matrix vcov is given. Rows of singleton menus are
taken out before solving (their probability must be 1) and added back as a constant.

Stopping. A call stops when ub - lb <= abs_tol or (ub - lb) / lb <= rel_tol, when max_time runs
out (setup included), or on the method's own limits. Whatever the reason, lb <= dist <= ub, and
distance = ub is the distance of the fitted probabilities p_fit.

Race. Given opt, a known optimum (the paper's references are in data/opt_refs.json), the call is
the paper's timing race instead: solver.solve(), the code behind the paper's race tables, runs
until ub <= max((1 + opt_rel_tol) * opt, opt_floor) (defaults: 1 %, no floor). A race computes no
lower bound (lb = nan) and takes no abs_tol or rel_tol: opt is a reference value, not a bound. ub
is the distance of the feasible point the race stopped at (the point stays inside the solver:
p_fit = None), and time_solve is the race time the tables report. Methods: bm_admm, flow_cg,
flow_admm; method_args are the race's own keywords (_RACE_ARGS).

    rum.calculate_distance(method="bm_admm", opt=9.4644623)             # stop at ub <= 1.01 opt
    rum.calculate_distance(method="flow_cg", opt=0.0, opt_floor=1e-3)   # opt = 0 needs a floor

Session. Many points with one plan (the bootstrap's solver): a session plans the method and its
storage format once, then solves any finite p (a bootstrap point may leave [0, 1]), with early
stops at a threshold and warm starts from a kept solve. calculate_distance does not use it.

    with rum.session() as s:
        r0 = s.solve(rum.choice_probability, abs_tol=5e-4, keep=True)      # keeps its state and p_fit
        rb = s.solve(p_b, abs_tol=5e-4, below=tau_lo, above=tau_hi, warm=True)

    exit_flag   1 abs_tol met    2 rel_tol met    3 both met    4 opt target reached (race)
                5 threshold reached (Session.solve: ub < below or lb >= above)
                0 time limit    -1 iteration limit    -2 stalled    -3 out of memory
               -4 interrupted   -5 solver error (message says which)

Problems found before solving (bad input, a named method that cannot run here) raise.
"""
import math
import time
import types
import warnings
from dataclasses import dataclass, field, fields

import numpy as np
import scipy.sparse as sp

from . import ALIASES, defaults, params
from . import machine as _machine

METHODS = ("bm_admm", "flow_cg", "flow_admm", "ip_cg", "ip_cg_heur", "bm_qp", "flow_qp")
GPU_METHODS = ("bm_admm", "flow_cg", "flow_admm")
RACE_METHODS = ("bm_admm", "flow_cg", "flow_admm")     # calculate_distance(opt=...): solver.solve's GPU races
EXACT_QP_MAX_N = 15            # the exact QPs build n 2^(n-1) variables
STATUS = {5: "threshold reached", 4: "opt target reached", 3: "abs_tol and rel_tol met", 1: "abs_tol met",
          2: "rel_tol met", 0: "time limit", -1: "iteration limit", -2: "stalled", -3: "out of memory",
          -4: "interrupted", -5: "solver error"}
_FLAG = {"threshold": 5, "target": 4, "time": 0, "iters": -1, "stalled": -2, "oom": -3, "interrupted": -4,
         "error": -5}
# The keywords method_args may set in a race (the race code's own names; the call sets the rest).
_RACE_ARGS = {"bm_admm": ("beta", "alpha", "precond_p", "check", "max_iter", "mode", "dtype", "ub_arm_immediately"),
              "flow_cg": ("max_iters", "layers", "pricing", "early_ub_check"),
              "flow_admm": ("check", "force") + tuple(params.AF_DEFAULTS)}
_RACE_STOP = {"time_limit": "time", "max_iter": "iters", "max_iters": "iters", "maxiter": "iters",
              "infeasible": "oom"}
_WARM = set()                  # race methods whose kernels this process has compiled
_UNIT = {"bm_admm": "ADMM iterations", "flow_admm": "ADMM iterations", "flow_cg": "columns", "ip_cg": "columns",
         "ip_cg_heur": "columns", "bm_qp": "QP solves", "flow_qp": "QP solves"}


@dataclass
class Result:
    exit_flag: int
    status: str
    message: str
    distance: float           # = ub, the distance of p_fit
    lb: float
    ub: float
    abs_gap: float
    rel_gap: float
    abs_tol: float
    rel_tol: float
    met_abs: bool
    met_rel: bool
    method: str
    auto_reason: str
    method_args: dict
    iterations: int
    iteration_unit: str
    time: float               # seconds, setup included
    time_setup: float
    time_solve: float
    seed: object
    p_fit: np.ndarray = None  # fitted probabilities, input row order (singleton rows = 1)
    trace: list = field(default_factory=list)      # (iteration, seconds since the call, ub, lb)
    machine: str = ""
    info: dict = field(default_factory=dict)
    mode: str = "certificate"  # "race" when opt was given: lb, the gaps and the tolerances are then nan
    opt: float = None          # race: the optimum given, the stop target, ub - opt and (ub - opt) / opt
    target: float = None
    abs_to_opt: float = None
    rel_to_opt: float = None

    def __repr__(self):
        rows = [("exit_flag", "%d (%s)" % (self.exit_flag, self.status)),
                ("distance", "%.10g" % self.distance)]
        if self.mode == "race":
            to_opt = "-" if self.abs_to_opt is None else "%.3g" % self.abs_to_opt
            if self.rel_to_opt is not None:
                to_opt += " (%+.3f %% of opt)" % (100 * self.rel_to_opt)
            rows += [("mode", "race to opt (no lower bound)"),
                     ("opt, target", "%.10g, %.10g" % (self.opt, self.target)),
                     ("ub - opt", to_opt)]
        else:
            rows += [("lb, ub", "%.10g, %.10g" % (self.lb, self.ub)),
                     ("abs_gap", "%.3g (abs_tol %.3g, %s)" % (self.abs_gap, self.abs_tol, "met" if self.met_abs else "not met")),
                     ("rel_gap", "%.3g (rel_tol %.3g, %s)" % (self.rel_gap, self.rel_tol, "met" if self.met_rel else "not met"))]
        rows += [("method", self.method + ("  [%s]" % self.auto_reason if self.auto_reason else "")),
                ("iterations", "%d %s" % (self.iterations, self.iteration_unit)),
                ("time", "%.3f s (setup %.3f s, solve %.3f s)" % (self.time, self.time_setup, self.time_solve)),
                ("seed", str(self.seed))]
        if self.message:
            rows.append(("message", self.message))
        rows.append(("method_args", str({k: v for k, v in self.method_args.items()})))
        rows.append(("p_fit", "None" if self.p_fit is None else "array of %d" % len(self.p_fit)))
        rows.append(("trace", "%d entries" % len(self.trace)))
        rows.append(("machine", self.machine))
        if self.info:
            rows.append(("info", str(self.info)))
        w = max(len(k) for k, _ in rows)
        return "\n".join("%s : %s" % (k.rjust(w), v) for k, v in rows)

    def to_dict(self, p_fit=False):
        """The fields as plain Python values, ready for json.dump; p_fit only when asked."""
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "p_fit"}
        d["trace"] = [list(row) for row in self.trace]
        if p_fit:
            d["p_fit"] = None if self.p_fit is None else self.p_fit.tolist()
        return d


class _TimeUp(Exception):
    pass


class _Decided(Exception):              # a Session threshold was crossed inside a runner's callback
    pass


class _Geometry:
    """The non-singleton rows, with the attributes the Gurobi builders (exact_qp, ip_cg) read and
    the caches they fill (flow_qp, bm_qp, cgip, cgipheuristic), plus Flow-ADMM's Mhat."""

    def __init__(self, menu, alternative, n):
        self.menu = np.asarray(menu, dtype=np.uint64)
        self.alternative = np.asarray(alternative, dtype=np.int64)
        self.num_alternatives = int(n)
        self.M = int(self.menu.size)
        self.flow_qp = self.bm_qp = self.cgip = self.cgipheuristic = None
        self.gurobi_threads = None
        self.mhat = None                  # (beta2, beta3, Mhat on the host), reused across calls

    def choice_data(self, p):
        """(M, 3) rows (menu, alternative, p) as the GPU solvers take them."""
        return np.column_stack([self.menu.astype(np.float64), self.alternative.astype(np.float64),
                                np.asarray(p, dtype=np.float64)])


class RUM:
    def __init__(self, menu, alternative, choice_probability, num_alternatives=None):
        menu = _as_int(menu, "menu")
        alternative = _as_int(alternative, "alternative")
        if menu.ndim != 1 or alternative.shape != menu.shape:
            raise ValueError("menu and alternative must be 1-D arrays of the same length")
        if menu.size == 0:
            raise ValueError("no observations")
        if (menu <= 0).any():
            raise ValueError("row %d: empty menu" % int(np.flatnonzero(menu <= 0)[0]))
        if (alternative < 0).any():
            raise ValueError("row %d: negative alternative" % int(np.flatnonzero(alternative < 0)[0]))
        need = max(int(menu.max()).bit_length(), int(alternative.max()) + 1)
        n = need if num_alternatives is None else int(num_alternatives)
        if n < need:
            raise ValueError("num_alternatives = %d, but the data use alternatives up to %d" % (n, need - 1))
        if n > 62:
            raise ValueError("menus are 64-bit masks: at most 62 alternatives")
        off = ((menu >> alternative) & 1) == 0
        if off.any():
            i = int(np.flatnonzero(off)[0])
            raise ValueError("row %d: alternative %d is not in its menu (mask %d)" % (i, alternative[i], menu[i]))
        _, first, counts = np.unique(np.column_stack([menu, alternative]), axis=0, return_index=True,
                                     return_counts=True)
        if (counts > 1).any():
            i = int(first[np.flatnonzero(counts > 1)[0]])
            raise ValueError("row %d: (menu %d, alternative %d) appears more than once"
                             % (i, menu[i], alternative[i]))
        self.n = n
        self.M = int(menu.size)
        self.menu = menu
        self.alternative = alternative
        self.singleton = (menu & (menu - 1)) == 0
        self._rest = np.flatnonzero(~self.singleton)
        self._single = np.flatnonzero(self.singleton)
        self._geom = _Geometry(menu[self._rest], alternative[self._rest], n)
        self._menu_of_row = np.unique(menu, return_inverse=True)[1]
        self.choice_probability = self._check_p(choice_probability)
        self._w_key = self._w_val = None

    @classmethod
    def from_rows(cls, rows, num_alternatives=None):
        """RUM from an (M, 3) array of rows (menu, alternative, choice probability): the layout
        of the data/ instances and of the engines' choice_data."""
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != 3:
            raise ValueError("rows must be an (M, 3) array: menu, alternative, choice probability")
        return cls(rows[:, 0], rows[:, 1], rows[:, 2], num_alternatives=num_alternatives)

    def __repr__(self):
        return "RUM(n=%d, %d rows, %d menus, %d singleton rows)" % (
            self.n, self.M, np.unique(self.menu).size, self._single.size)

    # ------------------------------------------------------------------ distance
    def calculate_distance(self, abs_tol=None, rel_tol=None, method=None, method_args=None,
                           max_time=None, p=None, vcov=None, seed=0, verbose=False, machine=None,
                           opt=None, opt_rel_tol=None, opt_floor=None):
        """dist(p) with a certified bracket, or with opt the paper's race; see the module docstring.

        abs_tol, rel_tol, max_time : None = solver/defaults.py.
        method      : None = automatic choice (choose_method), or one of METHODS (old names work).
        method_args : dict of that method's settings (names in defaults.METHOD_ARGS; in a race, the
                      race's keywords in _RACE_ARGS); needs a method.
        p           : probabilities to use instead of the stored ones (e.g. a bootstrap draw).
        vcov        : (M, M) weight matrix of the norm (dense or scipy.sparse); None = plain L2.
        seed        : seed of the random first columns; None = fresh (the drawn seed is reported).
        machine     : a machine profile (solver.machine.detect()); None = the saved profile.
        opt         : a known optimum: race to ub <= max((1 + opt_rel_tol) * opt, opt_floor) instead
                      of certifying (no lower bound; abs_tol and rel_tol are then not allowed).
        opt_rel_tol, opt_floor : the race target; None = solver/defaults.py (1 %, no floor). The
                      paper's floors: 1e-3 in the lambda study, 1e-5 in the convex-path race.
        """
        if opt is not None:
            return self._race(opt, opt_rel_tol, opt_floor, abs_tol, rel_tol, method, method_args, max_time,
                              p, vcov, seed, verbose, machine)
        if opt_rel_tol is not None or opt_floor is not None:
            raise ValueError("opt_rel_tol and opt_floor set a race target and need opt")
        t0 = time.perf_counter()
        abs_tol = float(defaults.ABS_TOL if abs_tol is None else abs_tol)
        rel_tol = float(defaults.REL_TOL if rel_tol is None else rel_tol)
        max_time = float(defaults.MAX_TIME if max_time is None else max_time)
        if abs_tol < 0 or rel_tol < 0 or not max_time > 0:
            raise ValueError("abs_tol and rel_tol must be >= 0 and max_time > 0")
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2 ** 31))
        p = self.choice_probability if p is None else self._check_p(p)
        W, const = self._weights(vcov, p)
        prof = _machine.load() if machine is None else machine
        geom, pr = self._geom, p[self._rest]

        if geom.M == 0:                       # only singleton menus: the distance is the constant
            d = math.sqrt(const)
            return self._result(t0, t0, "none", "only singleton menus", {}, abs_tol, rel_tol, seed,
                                dict(lb=d, ub=d, p_fit=np.zeros(0), iterations=0, trace=[], info={},
                                     setup=0.0), "tol", "", 0.0, prof)

        name, reason, args = self._plan(method, method_args, W is not None, prof)
        st = dict(lb=0.0, ub=math.inf, p_fit=None, iterations=0, trace=[], info={}, setup=0.0)
        task = types.SimpleNamespace(geom=geom, p=pr, W=W, args=args, abs_tol=abs_tol, rel_tol=rel_tol,
                                     deadline=t0 + max_time, seed=seed, verbose=verbose, t0=t0,
                                     state=st, const=const, ram=prof.get("ram_bytes"))
        t_run = time.perf_counter()
        message = ""
        try:
            stop = _RUN[name](task)
        except _TimeUp:
            stop = "time"
        except KeyboardInterrupt:
            stop, message = "interrupted", "interrupted by the user"
        except Exception as e:                # noqa: BLE001 -- reported through exit_flag
            stop = "oom" if _is_oom(e) else "error"
            message = "%s: %s" % (type(e).__name__, e)
        finally:
            if name in GPU_METHODS:
                _free_gpu()
        return self._result(t0, t_run, name, reason, args, abs_tol, rel_tol, seed, st, stop, message,
                            const, prof)

    # ------------------------------------------------------------------ session
    def session(self, method=None, method_args=None, vcov=None, machine=None, seed=0):
        """A Session: one plan (method, storage format, GPU budget), then many solves at any finite
        p, with threshold stops and warm starts. The bootstrap's solver; see Session."""
        return Session(self, method, method_args, vcov, machine, seed)

    # ------------------------------------------------------------------ bootstrap
    def bootstrap(self, n=None, panel=None, replies=None, B=None, weights=None, seed=None, r_n=None, eps_n=None,
                  method=None, method_args=None, abs_tol=None, rel_tol=None, refine=None, max_time=None,
                  draw_time=None, vcov=None, machine=None, checkpoint=None, verbose=False):
        """Hong and Li's bootstrap p-value and confidence interval for dist (solver/inference).

        The choices behind p_hat, one row per choice, as one of two tables (a pandas DataFrame or a
        dict of columns); rum's rows must be counted from it (solver.inference.cells_from_panel):
        panel   : DSR style, three columns: respondent (an id), menu (the bitmask, as in
                  RUM(menu, ...)) and chosen (the alternative picked), for people who each answered
                  several menus. Respondents are resampled; n is the number of respondents.
        replies : AI style, two columns, menu and chosen, for replies collected one by one (an LLM
                  asked each menu many times). Each menu's replies are resampled on their own; n is
                  the number of replies per menu.
        n       : the sample size above. None (the default) takes it from the data: the number of
                  respondents, or the most replies on any menu. A given n is checked against them.
                  r_n = sqrt(n) and eps_n = n^(-1/3) unless given.
        others  : None = solver/defaults.py BOOTSTRAP (weights applies to panel only); method and
                  method_args as in calculate_distance; checkpoint, a JSON path that saves and
                  resumes the run.

        Returns a Bootstrap that has solved nothing yet: boot.p_value(alpha=0.05) and
        boot.confidence_interval(alpha=0.05) do the work and share it."""
        from .inference.bootstrap import Bootstrap
        return Bootstrap(self, n, panel=panel, replies=replies, B=B, weights=weights, seed=seed, r_n=r_n, eps_n=eps_n,
                         method=method, method_args=method_args, abs_tol=abs_tol, rel_tol=rel_tol, refine=refine,
                         max_time=max_time, draw_time=draw_time, vcov=vcov, machine=machine, checkpoint=checkpoint,
                         verbose=verbose)

    # ------------------------------------------------------------------ race
    def _race(self, opt, opt_rel_tol, opt_floor, abs_tol, rel_tol, method, method_args, max_time, p,
              vcov, seed, verbose, machine):
        """calculate_distance(opt=...): the paper's race, solver.solve with the arguments
        validation/run_one.py passes, so iterations and ub match the race tables."""
        from . import solve as race, warmup
        t0 = time.perf_counter()
        if abs_tol is not None or rel_tol is not None:
            raise ValueError("with opt the call is a race to ub <= target and computes no lower bound, so "
                             "abs_tol and rel_tol do not apply (they belong to the certificate, opt=None); "
                             "opt_rel_tol and opt_floor set the target")
        opt = float(opt)
        r = float(defaults.OPT_REL_TOL if opt_rel_tol is None else opt_rel_tol)
        floor = float(defaults.OPT_FLOOR if opt_floor is None else opt_floor)
        max_time = float(defaults.MAX_TIME if max_time is None else max_time)
        if not (math.isfinite(opt) and opt >= 0 and r >= 0 and floor >= 0 and max_time > 0):
            raise ValueError("opt, opt_rel_tol and opt_floor must be finite and >= 0, and max_time > 0")
        target = max((1.0 + r) * opt, floor)
        if not target > 0:
            raise ValueError("opt = 0 makes the target (1 + opt_rel_tol) * opt = 0, which a race never "
                             "reaches: pass opt_floor > 0 (the paper: 1e-3 in the lambda study, 1e-5 on "
                             "the convex path)")
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2 ** 31))
        p = self.choice_probability if p is None else self._check_p(p)
        W, const = self._weights(vcov, p)
        if W is not None:
            raise ValueError("a race computes the plain L2 distance (vcov None or the identity)")
        if self._geom.M == 0 or target ** 2 <= const:
            raise ValueError("the singleton-menu rows alone are at distance %.10g, so a target of %.10g "
                             "leaves nothing to race" % (math.sqrt(const), target))
        prof = _machine.load() if machine is None else machine
        n, mp = self.n, self._geom.M
        if method is None:
            if method_args:
                raise ValueError("method_args needs a named method")
            name, reason = choose_method(n, mp, False, prof)
        else:
            name, reason = _method_name(method), ""
        if name not in RACE_METHODS:
            raise ValueError("%s has no race here; the races are %s (CGIP's: solver.solve). For a "
                             "certified distance leave opt out" % (name, ", ".join(RACE_METHODS)))
        bad = set(method_args or {}) - set(_RACE_ARGS[name])
        if bad:
            raise ValueError("unknown race method_args for %s: %s; allowed: %s"
                             % (name, sorted(bad), sorted(_RACE_ARGS[name])))
        _need_gpu(name, False, prof)
        kw = dict(method_args or {})
        floor_binds = floor > (1.0 + r) * opt
        if name == "bm_admm":
            if not 2 <= n <= 32:
                raise RuntimeError("bm_admm handles 2 <= n <= 32 (menus are 32-bit masks on the GPU)")
            if floor_binds:             # as run_one.py: the raw objective never nears a floor target,
                kw.setdefault("ub_arm_immediately", True)          # so the push runs from the first check
        elif name == "flow_cg":
            if "pricing" not in kw:     # the race's default is mask pricing; take what fits (mask up to n = 32)
                plan = dict(defaults.METHOD_ARGS["flow_cg"], layers=kw.get("layers", "auto"),
                            max_iters=kw.get("max_iters", params.CB_DEFAULTS["max_iters"]))
                kw["pricing"] = _plan_flow_cg(n, mp, False, prof, plan)["_engine"]
            kw["seed"] = seed
        else:
            _plan_flow_admm(n, mp, False, prof, dict(defaults.METHOD_ARGS["flow_admm"]))
        # The race stops at ub <= (1 + r) opt_ref on the non-singleton rows: opt itself, as the paper's
        # call passes it, unless a floor binds (run_one.py: floor / 1.01) or singleton rows add to dist^2.
        rest = math.sqrt(target ** 2 - const) if const else target
        opt_ref = opt if const == 0 and not floor_binds else rest / (1.0 + r)
        if name in ("bm_admm", "flow_cg") and name not in _WARM:
            warmup((name,))             # compile the kernels before the clock, as run_one.py does
            _WARM.add(name)
        out, stop, message = {}, "time", ""
        t_run = time.perf_counter()
        try:
            remaining = t0 + max_time - t_run
            if remaining > 0:
                out = race(n, self._geom.choice_data(p[self._rest]), method=name, criterion="race",
                           opt_ref=opt_ref, rel_tol=r, time_limit=remaining, verbose=verbose, **kw)
                stop = "target" if out.get("converged") else _RACE_STOP.get(out.get("stop"), "stalled")
        except KeyboardInterrupt:
            stop, message = "interrupted", "interrupted by the user"
        except Exception as e:                # noqa: BLE001 -- reported through exit_flag
            stop = "oom" if _is_oom(e) else "error"
            message = "%s: %s" % (type(e).__name__, e)
        finally:
            _free_gpu()
        t_end = time.perf_counter()

        def full(u):                    # the distance with the singleton rows added back
            return math.sqrt(float(u) ** 2 + const) if const else float(u)

        ub_rest = out.get("ub")
        ub = full(ub_rest) if ub_rest is not None and math.isfinite(ub_rest) else math.inf
        if not message:
            message = {"time": "max_time reached before the target",
                       "iters": "the race's iteration limit was reached before the target (method_args)",
                       "oom": out.get("note", ""),
                       "stalled": ("the race reached this data's optimum, %.10g, above the target: opt is "
                                   "below the optimum" % ub if out.get("stop") == "rc"
                                   else "the race stopped (%s) before the target" % out.get("stop"))}.get(stop, "")
        if ub < opt:
            message += ("; " if message else "") + ("ub is %.3g below opt: opt is above this data's optimum "
                                                    "(a reference can be, by up to its own gap)" % (opt - ub))
        trace, off = [], (t_run - t0) + float(out.get("build_s") or 0.0)
        for row in out.get("trace") or []:  # bm_admm (it, s, phi, ub or None); flow_cg (it, s, ub, lb, columns)
            it, s, u = (row[0], row[1], row[3]) if name == "bm_admm" else (row[4], row[1], row[2])
            if u is not None:
                trace.append((int(it), off + float(s), full(u), math.nan))
        info = {k: (v.item() if isinstance(v, np.generic) else v if isinstance(v, (int, float, str, bool, type(None)))
                    else str(v)) for k, v in out.items()
                if k not in ("method", "criterion", "ub", "lb", "iters", "wall_s", "converged", "trace",
                             "abs_to_opt", "rel_to_opt")}
        info["race_opt_ref"] = opt_ref
        solve_s = float(out.get("wall_s") or 0.0)
        flag, nan = _FLAG[stop], math.nan
        return Result(exit_flag=flag, status=STATUS[flag], message=message, distance=ub, lb=nan, ub=ub,
                      abs_gap=nan, rel_gap=nan, abs_tol=nan, rel_tol=nan, met_abs=False, met_rel=False,
                      method=name, auto_reason=reason, method_args={k: v for k, v in kw.items() if k != "seed"},
                      iterations=int(out.get("iters") or 0), iteration_unit=_UNIT[name], time=t_end - t0,
                      time_setup=(t_end - t0) - solve_s, time_solve=solve_s,
                      seed=seed if name == "flow_cg" else None, p_fit=None, trace=trace,
                      machine=_machine_line(prof), info=info, mode="race", opt=opt, target=target,
                      abs_to_opt=ub - opt if math.isfinite(ub) else None,
                      rel_to_opt=(ub - opt) / opt if math.isfinite(ub) and opt > 0 else None)

    # ------------------------------------------------------------------ helpers
    def _check_p(self, p):
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (self.M,):
            raise ValueError("choice probabilities: expected %d values, got shape %s" % (self.M, p.shape))
        if not np.isfinite(p).all():
            raise ValueError("choice probabilities must be finite")
        if (p < -1e-12).any() or (p > 1 + 1e-12).any():
            i = int(np.flatnonzero((p < -1e-12) | (p > 1 + 1e-12))[0])
            raise ValueError("row %d: probability %.6g is outside [0, 1]" % (i, p[i]))
        sums = np.bincount(self._menu_of_row, weights=p)
        if (sums > 1 + 1e-9).any():
            warnings.warn("%d menus have observed probabilities summing to more than 1 (largest %.6g)"
                          % (int((sums > 1 + 1e-9).sum()), sums.max()))
        return p

    def _finite_p(self, p):
        """Session.solve's p: any finite vector, one value per row (no [0, 1] check)."""
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (self.M,):
            raise ValueError("p: expected %d values, got shape %s" % (self.M, p.shape))
        if not np.isfinite(p).all():
            raise ValueError("p must be finite")
        return p

    def _weights(self, vcov, p):
        """(W on the non-singleton rows or None for L2, the singleton part of dist^2)."""
        r_s = 1.0 - p[self._single]
        if vcov is None:
            return None, float(r_s @ r_s)
        if vcov is not self._w_key:
            V = vcov.tocsr() if sp.issparse(vcov) else np.asarray(vcov, dtype=np.float64)
            if V.shape != (self.M, self.M):
                raise ValueError("vcov must be (%d, %d); got %s" % (self.M, self.M, V.shape))
            if _is_identity(V):
                self._w_val = (None, None)
            else:
                s, r = self._single, self._rest
                cross = V[s][:, r] if sp.issparse(V) else V[np.ix_(s, r)]
                if (cross.count_nonzero() if sp.issparse(cross) else np.count_nonzero(cross)):
                    raise ValueError("vcov couples singleton-menu rows with other rows; the singleton "
                                     "rows cannot then be taken out exactly (this is not supported)")
                Wss = V[s][:, s] if sp.issparse(V) else V[np.ix_(s, s)]
                Wrr = V[r][:, r] if sp.issparse(V) else V[np.ix_(r, r)]
                self._w_val = (Wrr, Wss)
            self._w_key = vcov
        Wrr, Wss = self._w_val
        if Wrr is None:
            return None, float(r_s @ r_s)
        return Wrr, float(r_s @ (Wss @ r_s))

    def _plan(self, method, method_args, weighted, prof):
        n, mp = self.n, self._geom.M
        if method is None:
            if method_args:
                raise ValueError("method_args needs a named method")
            name, reason = choose_method(n, mp, weighted, prof)
        else:
            name, reason = _method_name(method), ""
        args = dict(defaults.METHOD_ARGS[name])
        bad = set(method_args or {}) - set(args)
        if bad:
            raise ValueError("unknown method_args for %s: %s; allowed: %s"
                             % (name, sorted(bad), sorted(args)))
        args.update(method_args or {})
        return name, reason, _PLAN[name](n, mp, weighted, prof, args)

    def _result(self, t0, t_run, name, reason, args, abs_tol, rel_tol, seed, st, stop, message,
                const, prof):
        lb = math.sqrt(max(st["lb"], 0.0) ** 2 + const)
        ub = math.sqrt(st["ub"] ** 2 + const) if math.isfinite(st["ub"]) else math.inf
        abs_gap = ub - lb
        rel_gap = abs_gap / lb if lb > 0 else math.inf
        met_abs, met_rel = abs_gap <= abs_tol, lb > 0 and rel_gap <= rel_tol
        flag = 3 if (met_abs and met_rel) else 1 if met_abs else 2 if met_rel else _FLAG.get(stop, -2)
        message = message or st.get("message", "")
        if flag == -2 and not message and stop == "tol" and name == "flow_qp":
            message = ("Gurobi's point certifies only ub - lb = %.2g (relative %.2g), wider than the tolerances: "
                       "loosen abs_tol or rel_tol, or use a method that iterates to them" % (abs_gap, rel_gap))
        elif flag == -2 and not message:
            message = ("the method stopped on its own test, but the bracket does not meet the tolerances"
                       if stop == "tol" else "the method stopped without meeting the tolerances")
        elif flag == 0 and not message:
            message = "max_time reached"
        elif flag == -1 and not message:
            message = "the method's iteration limit was reached (see method_args)"
        notes, info, met = [], st["info"], flag in (1, 2, 3)
        if name == "bm_admm" and args.get("dtype") == "float32" and not met:
            notes.append("a float32 state's pushed upper bound usually stalls above the tolerances; "
                         "dtype='float64' certifies where it fits")
        if info.get("lb_rigorous") is False:
            m, u = float(info.get("pricing_margin") or 0.0), st["ub"]
            floor = u - math.sqrt(max(u * u - 2.0 * m, 0.0)) if m > 0 and math.isfinite(u) else 0.0
            notes.append("lb is the %s pricing bound, since no float64 bound fits the GPU at n = %d, so it does not "
                         "cover the float32 rounding inside the sweep%s"
                         % (info.get("lb_dtype"), self.n, "" if met or floor <= 0 else
                            "; its margin %.2g keeps ub - lb above about %.2g" % (m, floor)))
        if notes:
            message = "; ".join(([message] if message else []) + notes)
        p_fit = None
        if st["p_fit"] is not None:
            p_fit = np.ones(self.M)
            p_fit[self._rest] = st["p_fit"]
        t_end = time.perf_counter()
        setup = (t_run - t0) + st["setup"]
        return Result(exit_flag=flag, status=STATUS[flag], message=message, distance=ub, lb=lb, ub=ub,
                      abs_gap=abs_gap, rel_gap=rel_gap, abs_tol=abs_tol, rel_tol=rel_tol,
                      met_abs=bool(met_abs), met_rel=bool(met_rel), method=name, auto_reason=reason,
                      method_args={k: v for k, v in args.items() if not k.startswith("_")},
                      iterations=int(st["iterations"]), iteration_unit=_UNIT.get(name, ""),
                      time=t_end - t0, time_setup=setup, time_solve=(t_end - t0) - setup, seed=seed,
                      p_fit=p_fit, trace=st["trace"], machine=_machine_line(prof), info=st["info"])


# ---------------------------------------------------------------------- session
class Session:
    """One plan, many solves: what the bootstrap asks of the solver.

    The method, its storage format and its GPU budget are planned once, as calculate_distance plans
    them, and every solve uses that plan: a certificate and the draws compared with it share method
    and dtype. solve() takes any finite p, since a bootstrap point may leave [0, 1], and returns a
    Result like calculate_distance's.

        below, above   early stops on the distance: the solve may end once ub < below or lb >= above
                       (exit_flag 5). The caller decides from lb and ub; a stop is only a shortcut.
        keep           store this solve's state and its fitted point (s.fit, input row order)
        warm           start from the kept state: bm_admm its ADMM iterate (on the host above n = 22),
                       flow_cg its columns and weights; the other methods start as usual

    The GPU memory pool stays allocated between solves; close() or the end of a with block frees it.
    """

    def __init__(self, rum, method=None, method_args=None, vcov=None, machine=None, seed=0):
        t0 = time.perf_counter()
        self.rum, self.vcov, self.seed = rum, vcov, seed
        self.prof = _machine.load() if machine is None else machine
        weighted = rum._weights(vcov, rum.choice_probability)[0] is not None
        self.method, self.auto_reason, self._args = rum._plan(method, method_args, weighted, self.prof)
        self.method_args = {k: v for k, v in self._args.items() if not k.startswith("_")}
        self.fit = None
        self._kept = None
        self.time_plan = time.perf_counter() - t0

    def solve(self, p, abs_tol=None, rel_tol=None, max_time=None, below=None, above=None, warm=False,
              keep=False, verbose=False):
        """dist(p) with a certified bracket, by the session's plan (see the class docstring)."""
        rum, t0 = self.rum, time.perf_counter()
        abs_tol = float(defaults.ABS_TOL if abs_tol is None else abs_tol)
        rel_tol = float(defaults.REL_TOL if rel_tol is None else rel_tol)
        max_time = float(defaults.MAX_TIME if max_time is None else max_time)
        if abs_tol < 0 or rel_tol < 0 or not max_time > 0:
            raise ValueError("abs_tol and rel_tol must be >= 0 and max_time > 0")
        if warm and self._kept is None and self.method in ("bm_admm", "flow_cg"):
            raise ValueError("warm=True starts from a kept solve: call solve(..., keep=True) first")
        p = rum._finite_p(p)
        W, const = rum._weights(self.vcov, p)
        geom, pr = rum._geom, p[rum._rest]
        if geom.M == 0:                       # only singleton menus: the distance is the constant
            d = math.sqrt(const)
            return rum._result(t0, t0, "none", "only singleton menus", {}, abs_tol, rel_tol, self.seed,
                               dict(lb=d, ub=d, p_fit=np.zeros(0), iterations=0, trace=[], info={}, setup=0.0),
                               "tol", "", 0.0, self.prof)
        st = dict(lb=0.0, ub=math.inf, p_fit=None, iterations=0, trace=[], info={}, setup=0.0)
        task = types.SimpleNamespace(geom=geom, p=pr, W=W, args=self._args, abs_tol=abs_tol, rel_tol=rel_tol,
                                     deadline=t0 + max_time, seed=self.seed, verbose=verbose, t0=t0, state=st,
                                     const=const, ram=self.prof.get("ram_bytes"),
                                     below=_on_rest(below, const, False), above=_on_rest(above, const, True),
                                     warm=self._kept if warm else None, keep=bool(keep))
        t_run = time.perf_counter()
        message = ""
        try:
            stop = _RUN[self.method](task)
        except _TimeUp:
            stop = "time"
        except _Decided:
            stop = "threshold"
        except KeyboardInterrupt:
            stop, message = "interrupted", "interrupted by the user"
        except Exception as e:                # noqa: BLE001 -- reported through exit_flag
            stop = "oom" if _is_oom(e) else "error"
            message = "%s: %s" % (type(e).__name__, e)
        kept = st.pop("keep", None)
        res = rum._result(t0, t_run, self.method, self.auto_reason, self._args, abs_tol, rel_tol, self.seed,
                          st, stop, message, const, self.prof)
        if keep:
            self._kept, self.fit = kept, res.p_fit
        return res

    def close(self):
        """Drop the kept state and free the GPU memory pool."""
        self._kept = self.fit = None
        if self.method in GPU_METHODS:
            _free_gpu()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return "Session(%s%s, %s; kept state: %s)" % (
            self.method, "  [%s]" % self.auto_reason if self.auto_reason else "", self.method_args,
            "no" if self._kept is None else "yes")


def _on_rest(tau, const, above):
    """A threshold on dist as one on the non-singleton rows (dist^2 = rest^2 + const): None for a
    below-stop that cannot fire, 0 for an above-stop that holds from the start."""
    if tau is None:
        return None
    tau = float(tau)
    if not math.isfinite(tau):
        raise ValueError("below and above must be finite")
    if tau * tau <= const:
        return 0.0 if above else None
    return math.sqrt(tau * tau - const)


# ---------------------------------------------------------------------- choosing a method
def choose_method(n, mp, weighted, prof):
    """(method, reason) for an instance with n alternatives and mp non-singleton rows.

    With a usable GPU and the plain norm: bm_admm from the density defaults.AUTO sets for this n,
    flow_cg below it and whenever bm_admm does not fit the GPU. Without a GPU, with a weighted
    norm, or when neither fits: a Gurobi method."""
    A = defaults.AUTO
    density = mp / (n << (n - 1))
    why = ["n = %d, %d rows, density %.3g%%" % (n, mp, 100 * density)]
    if weighted:
        why.append("weighted norm: only the Gurobi methods take vcov")
    elif not _machine.can_use_gpu(prof):
        why.append("no usable GPU (%s)" % ((prof.get("gpu") or {}).get("reason") or "none found"
                                          if prof.get("use_gpu", True) else "use_gpu = False"))
    else:
        d_min = bm_admm_min_density(n)
        if d_min is None:
            why.append("n <= %d -> flow_cg at any density" % (min(A["bm_admm_min_density"]) - 1))
        elif density < d_min:
            why.append("density < %.2g%% at n = %d -> flow_cg" % (100 * d_min, n))
        else:
            try:
                _plan_bm_admm(n, mp, False, prof, dict(defaults.METHOD_ARGS["bm_admm"]))
                return "bm_admm", "; ".join(why + ["density >= %.2g%% at n = %d -> bm_admm" % (100 * d_min, n)])
            except RuntimeError as e:
                why.append(str(e))
        try:
            _plan_flow_cg(n, mp, False, prof, dict(defaults.METHOD_ARGS["flow_cg"]))
            return "flow_cg", "; ".join(why)
        except RuntimeError as e:
            why.append(str(e))
    if _machine.can_use_gurobi(prof):
        if n <= A["exact_qp_max_n"]:
            return "flow_qp", "; ".join(why + ["n <= %d -> exact QP (flow_qp)" % A["exact_qp_max_n"]])
        return "ip_cg_heur", "; ".join(why + ["n > %d -> ip_cg_heur" % A["exact_qp_max_n"]])
    raise RuntimeError("no method can run on this machine: " + "; ".join(why)
                       + "; Gurobi " + ("off" if prof.get("gurobi") else "not installed"))


def bm_admm_min_density(n):
    """The density from which the automatic choice takes bm_admm at n; None below the first n."""
    steps = [d for n0, d in sorted(defaults.AUTO["bm_admm_min_density"].items()) if n >= n0]
    return steps[-1] if steps else None


def _need_gpu(name, weighted, prof):
    if weighted:
        raise RuntimeError("%s computes the unweighted distance only (vcov None or identity); for a "
                           "weighted norm use ip_cg, ip_cg_heur, bm_qp or flow_qp" % name)
    if not _machine.can_use_gpu(prof):
        g = prof.get("gpu")
        raise RuntimeError("%s needs a supported NVIDIA GPU with CuPy: %s" % (
            name, "use_gpu = False in the machine profile" if g and not prof.get("use_gpu", True)
            else (g or {}).get("reason") or "none found (see solver.machine.setup())"))


# Peak device bytes per slab entry of one BM-ADMM call (14 state slabs, graph replay, the float64
# push and the copy for the lower bound), measured at n = 21-24 on an RTX 5090: {(itemsize, graph)}.
_BM_BYTES_PER_ENTRY = {(8, False): 141, (8, True): 145, (4, False): 82, (4, True): 92}


def _bm_admm_bytes(n, itemsize, graph):
    """Device memory of one BM-ADMM call: the measured peak per entry of n 2^(n-1), plus 10 %."""
    return int(1.1 * _BM_BYTES_PER_ENTRY[itemsize, bool(graph)] * (n << (n - 1)))


def _plan_bm_admm(n, mp, weighted, prof, a):
    _need_gpu("bm_admm", weighted, prof)
    if not 2 <= n <= 32:
        raise RuntimeError("bm_admm handles 2 <= n <= 32 (menus are 32-bit masks on the GPU)")
    for k in ("check", "lb_every", "max_iter"):
        if int(a[k]) < 1:
            raise ValueError("bm_admm %s must be >= 1" % k)
    if a["dtype"] not in ("auto", "float64", "float32"):
        raise ValueError("bm_admm dtype must be 'auto', 'float64' or 'float32'")
    budget = defaults.AUTO["gpu_memory_fraction"] * _machine.gpu_free_bytes()
    # "auto" is float64 only: from a float32 iterate the pushed upper bound stalls (n18_m39 at
    # rel 1e-4: float64 certifies in 41 s; float32 is still 2.7 % above the lower bound after 120 s)
    dt = "float64" if a["dtype"] == "auto" else a["dtype"]
    item = 8 if dt == "float64" else 4
    graphs = ((True, False) if n <= 22 else (False,)) if a["use_graph"] == "auto" else (bool(a["use_graph"]),)
    for g in graphs:
        need = _bm_admm_bytes(n, item, g)
        if need <= budget:
            a.update(dtype=dt, use_graph=g, _gpu_bytes=need)
            a["beta"] = params.bm_beta_default(mp) if a["beta"] is None else float(a["beta"])
            return a
    raise RuntimeError("bm_admm at n = %d in %s needs %.1f GB of GPU memory; %.1f GB may be used. Use flow_cg%s"
                       % (n, dt, _bm_admm_bytes(n, item, False) / 2 ** 30, budget / 2 ** 30,
                          " (dtype='float32' needs %.1f GB, but its upper bound rarely certifies)"
                          % (_bm_admm_bytes(n, 4, False) / 2 ** 30) if dt == "float64" else ""))


def _flow_cg_bytes(n, pricing, layers):
    N, cap = 1 << n, math.comb(n, n // 2)
    if pricing == "mask":
        stored = layers == "stored" or (layers == "auto" and n <= 28)
        return 5 * N + 4 * (N if stored else cap)
    k_pred = max(1, n - 10)
    item = {"stream_f32": 4, "stream_u24": 3, "stream_u16": 2}[pricing]
    return 2 * cap * item + sum(math.comb(n, k) for k in range(k_pred, n + 1)) + 5 * (1 << k_pred)


# flow_cg's (layout, dtype) -> the engine's pricing backend, most precise first: the automatic order
_CG_ENGINE = {("mask", "float32"): "mask", ("stream", "float32"): "stream_f32",
              ("stream", "uint24"): "stream_u24", ("stream", "uint16"): "stream_u16"}
_CG_OLD = {v: k for k, v in _CG_ENGINE.items() if v != "mask"}      # pricing="stream_u16" etc. still work


def _plan_flow_cg(n, mp, weighted, prof, a):
    _need_gpu("flow_cg", weighted, prof)
    k = int(a["max_iters"]) + 2
    host = 8 * k * k + 8 * k * ((mp + 63) // 64)          # Gram matrix + column bitsets on the host
    if prof.get("ram_bytes") and host > 0.5 * prof["ram_bytes"]:
        raise RuntimeError("flow_cg max_iters = %d needs %.1f GB of host memory for its Gram matrix"
                           % (a["max_iters"], host / 2 ** 30))
    layout, dt = a["pricing"], a["dtype"]
    if layout in _CG_OLD:
        if dt not in ("auto", _CG_OLD[layout][1]):
            raise ValueError("pricing=%r is %s storage; it contradicts dtype=%r" % (layout, _CG_OLD[layout][1], dt))
        layout, dt = _CG_OLD[layout]
    if layout not in ("auto", "mask", "stream"):
        raise ValueError("flow_cg pricing must be 'auto', 'mask' or 'stream' (old names: stream_f32, stream_u24, "
                         "stream_u16)")
    if dt not in ("auto", "float32", "uint24", "uint16"):
        raise ValueError("flow_cg dtype must be 'auto', 'float32', 'uint24' or 'uint16'")
    if layout == "mask" and dt not in ("auto", "float32"):
        raise ValueError("mask pricing stores float32 only; dtype=%r needs pricing='stream' or 'auto'" % dt)
    budget = defaults.AUTO["gpu_memory_fraction"] * _machine.gpu_free_bytes()
    tried = []
    for (lay, d), engine in _CG_ENGINE.items():
        if layout not in ("auto", lay) or dt not in ("auto", d) or (lay == "mask" and n > 32):
            continue
        need = _flow_cg_bytes(n, engine, a["layers"])
        if need <= budget:
            a.update(pricing=lay, dtype=d, _engine=engine, _gpu_bytes=need)
            return a
        tried.append("%s %s %.1f GB" % (lay, d, need / 2 ** 30))
    raise RuntimeError("flow_cg at n = %d does not fit the GPU: %.1f GB may be used (%s)"
                       % (n, budget / 2 ** 30, "; ".join(tried) or "mask needs n <= 32"))


def _flow_admm_bytes(n, mp):
    """Device memory of one Flow-ADMM call, from the peaks measured on an RTX 5090 (n = 18-22,
    m_p = 306-11939), plus 10 %: 55 bytes per m_p^2 (Mhat, the Schur matrix, its Cholesky factor
    and workspace, float64), 82 per entry of n 2^(n-1) (the float32 slabs, the push and the bound)
    and 0.25 GB of fixed workspace."""
    return int(1.1 * (55 * mp * mp + 82 * (n << (n - 1))) + 2.5e8)


def _plan_flow_admm(n, mp, weighted, prof, a):
    _need_gpu("flow_admm", weighted, prof)
    if not 2 <= n <= 32:
        raise RuntimeError("flow_admm handles 2 <= n <= 32")
    for k in ("check", "lb_every", "max_iter"):
        if int(a[k]) < 1:
            raise ValueError("flow_admm %s must be >= 1" % k)
    if a["dtype"] not in ("auto", "float32"):
        raise ValueError("flow_admm dtype must be 'auto' or 'float32': its ADMM kernels are float32 (the dense "
                         "Mhat and Schur factor are always float64); for a float64 state use bm_admm")
    a["dtype"] = "float32"
    budget = defaults.AUTO["gpu_memory_fraction"] * _machine.gpu_free_bytes()
    need = _flow_admm_bytes(n, mp)
    if need > budget:
        raise RuntimeError("flow_admm needs %.1f GB of GPU memory (dense m_p x m_p matrices, m_p = %d); "
                           "%.1f GB may be used. Use bm_admm or flow_cg" % (need / 2 ** 30, mp, budget / 2 ** 30))
    a["_gpu_bytes"] = need
    return a


def _plan_gurobi(name):
    def plan(n, mp, weighted, prof, a):
        if not _machine.can_use_gurobi(prof):
            raise RuntimeError("%s needs gurobipy and a Gurobi licence" % name)
        if name in ("bm_qp", "flow_qp") and n > EXACT_QP_MAX_N:
            raise RuntimeError("%s builds n 2^(n-1) = %d variables; it is meant for n <= %d"
                               % (name, n << (n - 1), EXACT_QP_MAX_N))
        if a.get("threads") is not None and int(a["threads"]) < 1:
            raise ValueError("threads must be >= 1 (or None for Gurobi's default)")
        return a
    return plan


_PLAN = {"bm_admm": _plan_bm_admm, "flow_cg": _plan_flow_cg, "flow_admm": _plan_flow_admm,
         "ip_cg": _plan_gurobi("ip_cg"),
         "ip_cg_heur": _plan_gurobi("ip_cg_heur"), "bm_qp": _plan_gurobi("bm_qp"),
         "flow_qp": _plan_gurobi("flow_qp")}


# ---------------------------------------------------------------------- running a method
# Each runner fills task.state (lb, ub, p_fit, iterations, trace, info, setup) as it goes, so an
# interrupt or an error still leaves the best bracket found, and returns its stop reason:
# "tol", "time", "iters" or "stalled". Trace rows are (iteration, seconds since the call, ub, lb).

def _run_bm_admm(t):
    import cupy as cp
    from .bm_admm.solver import solve_kadmm_v2
    from .common.transforms import build_obs_idx, fw_lb_rows_gpu, push_to_ub64
    g, a, st = t.geom, t.args, t.state
    n = g.num_alternatives
    g_obs = cp.asarray(build_obs_idx(n, g.menu.astype(np.int64), g.alternative))
    g_p = cp.asarray(t.p)
    checks = [0]

    def push(x2):                       # upper bound: the point pushed in float64 (best one kept)
        if time.perf_counter() >= t.deadline:
            raise _TimeUp
        if not checks[0]:               # setup plus the first check (exact once the call returns)
            st["setup"] = time.perf_counter() - t_call
        ub, x_ub = push_to_ub64(n, x2, g_obs, g_p)
        checks[0] += 1
        st["iterations"] = checks[0] * a["check"]
        if ub < st["ub"]:
            st["ub"] = float(ub)
            st["p_fit"] = cp.asnumpy(x_ub[g_obs])
        return ub

    def lower(x2):                      # lower bound: Frank-Wolfe sweep on the GPU, every lb_every checks
        if (checks[0] - 1) % a["lb_every"] == 0:
            st["lb"] = max(st["lb"], fw_lb_rows_gpu(n, g_obs, x2[g_obs], g_p))
        st["trace"].append((st["iterations"], time.perf_counter() - t.t0, st["ub"], st["lb"]))
        return st["lb"]

    st["info"].update(gpu_bytes_planned=a["_gpu_bytes"], lb_dtype="float64", lb_layout="slab", lb_rigorous=True)
    t_call = time.perf_counter()
    r = solve_kadmm_v2(n, g.choice_data(t.p), beta1=a["beta"], beta2=a["beta"], alpha_relax=a["alpha"],
                       precond_p=a["precond_p"], max_iter=int(a["max_iter"]), check=int(a["check"]),
                       push_fn=push, ub_arm_immediately=True, lb_fn=lower, ub_running_min=True,
                       gap_abs_tol=t.abs_tol, gap_rel_tol=t.rel_tol,
                       time_limit=max(t.deadline - time.perf_counter(), 0.0), dtype=a["dtype"],
                       use_graph=a["use_graph"], verbose=t.verbose, **_bm_session(t))
    st["iterations"], st["setup"] = int(r["iters"]), float(r["setup_s"])
    st["info"].update(per_iter_ms=round(r["per_iter_ms"], 3), push_s=round(r["push_s"], 3))
    if getattr(t, "keep", False):           # Session: the ADMM state, kept on the host above n = 22
        st["keep"] = {k: cp.asnumpy(v) if n > 22 else v for k, v in r["state"].items()}
    return {"gap_abs": "tol", "gap_rel": "tol", "time_limit": "time", "max_iter": "iters", "ub_target": "threshold",
            "lb_target": "threshold"}.get(r["stop_reason"], "stalled")


def _bm_session(t):
    """solve_kadmm_v2's threshold stops and warm start for a Session.solve; none from calculate_distance.
    The engine stops at ub <= ub_target (ub_eps 0) or lb >= lb_target."""
    kw = {}
    below, above, warm = getattr(t, "below", None), getattr(t, "above", None), getattr(t, "warm", None)
    if below is not None and below > 0:
        kw.update(ub_target=below, ub_eps=0.0)
    if above is not None:
        kw["lb_target"] = above
    if warm is not None:
        kw["init_state"] = warm
    return kw


def _flow_cg_lb(n, g, obs, p_fit, p, engine_lb, delta, dtype):
    """flow_cg's final lower bound in the most precise form that fits the GPU, as (lb, dtype, layout):
    the float64 bound from its slab (n <= 26 on 32 GB), else two layers at a time (n <= 33), else
    the engine's pricing bound lowered by delta (it does not cover the rounding inside the sweep)."""
    from .common.transforms import fw_lb_rows_gpu, fw_lb_stream64
    from .flow_cg.pricing_stream import sweep_max64_bytes
    if _fits_gpu(16 * (n << (n - 1)) + 16 * (1 << n)):    # the slab and the butterfly's copy; DP, layers
        return fw_lb_rows_gpu(n, obs, p_fit, p), "float64", "slab"
    if _fits_gpu(sweep_max64_bytes(n, g.M)):
        return fw_lb_stream64(n, g.menu, g.alternative, p_fit, p), "float64", "two layers"
    return engine_lb - delta, dtype, "pricing sweep"


def _run_flow_cg(t):
    from .flow_cg import solve_cgbfqp
    from .common.transforms import build_obs_idx
    g, a, st = t.geom, t.args, t.state
    n = g.num_alternatives
    # flow_cg solves for the float32-rounded p and prices in a.dtype (float32, uint24, uint16). The
    # class recomputes both bounds for the true p: ub = the exact distance of the fitted mixture, lb =
    # the float64 Frank-Wolfe bound at that mixture on the GPU (_flow_cg_lb; where no float64 form fits,
    # the engine's lb lowered by delta = ||p - float32(p)||, lb_rigorous False).
    # The engine gets abs_tol - 2 delta. A stop that the float64 bracket misses continues once from
    # the engine's own columns, with the tolerances tightened by the difference (1.5x for safety).
    delta = float(np.linalg.norm(t.p - t.p.astype(np.float32).astype(np.float64)))
    perm = np.argsort(g.alternative.astype(np.int32), kind="stable")     # flow_cg's row order
    obs = build_obs_idx(n, g.menu.astype(np.int64), g.alternative)
    rel, abs_e = t.rel_tol, max(t.abs_tol - 2 * delta, 0.0)
    # Session.solve: start from the kept columns, and stop the engine at a threshold moved inward by
    # 2 delta; the float64 bounds then confirm the decision (a below-decision needs no lower bound)
    below, above, kept = getattr(t, "below", None), getattr(t, "above", None), getattr(t, "warm", None)
    warm = {} if kept is None else dict(init_columns=kept["columns"], init_lambda=kept["lam"])
    stops = _cg_stops(below, above, 2 * delta)
    for attempt in range(2):
        remaining = t.deadline - time.perf_counter()
        if remaining <= 0:
            return "time"
        t_call = time.perf_counter()
        r = solve_cgbfqp(n, g.choice_data(t.p), criterion="cert", rel_tol=rel, abs_tol=abs_e,
                         time_limit=remaining, max_iters=int(a["max_iters"]), seed=t.seed,
                         pricing=a["_engine"], layers=a["layers"], return_state=True, **warm, **stops)
        p_fit = np.empty(g.M)
        p_fit[perm] = r["p_out"]
        st["p_fit"] = p_fit
        st["ub"] = float(np.linalg.norm(p_fit - t.p))        # exact distance of the fitted mixture
        lb_prev = st["lb"]
        if below is not None and st["ub"] < below:
            lb, lb_dt, lb_lay = lb_prev, None, "not computed (below the threshold)"
        else:
            lb, lb_dt, lb_lay = _flow_cg_lb(n, g, obs, p_fit, t.p, float(r["lb"]), delta, a["dtype"])
        st["lb"] = max(lb_prev, lb)
        st["iterations"] = int(r["iters"])
        st["setup"] += float(r["build_s"])
        off = (t_call - t.t0) + float(r["build_s"])
        st["trace"] += [(k, off + w, ub, max(lb_prev, lb - delta)) for _it, w, ub, lb, k in r["trace"]]
        if st["trace"]:                 # the last row is the final bracket
            st["trace"][-1] = st["trace"][-1][:2] + (st["ub"], st["lb"])
        st["info"].update(pricing=r["pricing"], layers=r["layers_mode"], gpu_bytes=r["device_bytes"],
                          p_float32_shift=delta, pricing_margin=r["margin_max"], restarts=attempt,
                          lb_dtype=lb_dt, lb_layout=lb_lay, lb_rigorous=lb_dt in ("float64", None))
        if getattr(t, "keep", False):
            st["keep"] = dict(columns=r["columns"], lam=r["lam"])
        if (below is not None and st["ub"] < below) or (above is not None and st["lb"] >= above):
            return "threshold"
        gap = st["ub"] - st["lb"]
        if (attempt or r["stop"] not in ("gap_rel", "gap_abs", "phi_ub", "lb_stop") or gap <= t.abs_tol
                or gap <= t.rel_tol * st["lb"]):
            break
        excess = 1.5 * max(gap - (float(r["ub"]) - float(r["lb"])), 2 * delta)
        rel = max(t.rel_tol - excess / st["lb"], 0.0) if st["lb"] > 0 else 0.0
        abs_e = max(t.abs_tol - 2 * delta - excess, 0.0)
        if rel <= 0 and abs_e <= 0 and not stops:
            break
        stops = _cg_stops(below, above, 2 * delta + excess)
        warm = dict(init_columns=r["columns"], init_lambda=r["lam"])
    return {"gap_rel": "tol", "gap_abs": "tol", "rc": "tol", "time_limit": "time",
            "max_iters": "iters"}.get(r["stop"], "stalled")


def _cg_stops(below, above, margin):
    """flow_cg's engine stops for a Session threshold, moved inward by margin: the engine's bounds
    are for the float32-rounded p, and the float64 ones confirm the decision afterwards."""
    kw = {}
    if below is not None:
        kw.update(ub_stop=max(below - margin, 0.0), early_ub_check=True)
    if above is not None:
        kw["lb_stop"] = above + margin
    return kw


def _run_flow_admm(t):
    import cupy as cp
    from .flow_admm.solver import solve_arcflow_fast
    from .common.arcflow_kernels import make_r_feasible_gpu64
    from .common.transforms import build_obs_idx, fw_lb_rows_gpu, layers_gpu, zeta, phi_at
    g, a, st = t.geom, t.args, t.state
    n = g.num_alternatives
    g_obs = cp.asarray(build_obs_idx(n, g.menu.astype(np.int64), g.alternative))
    g_p = cp.asarray(t.p)
    layers, bounds = layers_gpu(n)
    checks = [0]
    below, above = getattr(t, "below", None), getattr(t, "above", None)     # Session thresholds

    def tick():                         # max_time includes the Mhat build
        if time.perf_counter() >= t.deadline:
            raise _TimeUp

    def push(r):                        # upper bound: the arc flow r pushed to a feasible flow in float64
        tick()
        if not checks[0]:               # setup (exact once the call returns): Mhat, factorization
            st["setup"] = time.perf_counter() - t_call
        x_ub = zeta(make_r_feasible_gpu64(n, r, layers, bounds), n)
        ub = phi_at(x_ub, g_obs, g_p)
        checks[0] += 1
        st["iterations"] = max(1, (checks[0] - 1) * a["check"])         # checks at 1, check, 2 check, ...
        if ub < st["ub"]:
            st["ub"] = float(ub)
            st["p_fit"] = cp.asnumpy(x_ub[g_obs])
        if below is not None and st["ub"] < below:
            raise _Decided
        return ub

    def lower(r):                       # lower bound: Frank-Wolfe sweep on the GPU at the flow's probabilities
        if (checks[0] - 1) % a["lb_every"] == 0:
            st["lb"] = max(st["lb"], fw_lb_rows_gpu(n, g_obs, zeta(r, n)[g_obs], g_p))
        st["trace"].append((st["iterations"], time.perf_counter() - t.t0, st["ub"], st["lb"]))
        if above is not None and st["lb"] >= above:
            raise _Decided
        return st["lb"]

    key = (float(a["beta2"]), float(a["beta3"]))            # Mhat depends on them and the menus only
    mhat = g.mhat[2] if g.mhat is not None and g.mhat[:2] == key else None
    opts = dict(beta2=key[0], beta3=key[1], alpha_relax=float(a["alpha"]), adapt_on=bool(a["adaptive"]),
                maxiter=int(a["max_iter"]), log_every=int(a["check"]),
                time_limit=max(t.deadline - time.perf_counter(), 0.0), push_fn=push, lb_fn=lower,
                gap_abs_tol=t.abs_tol, gap_rel_tol=t.rel_tol, progress_fn=tick, verbose=t.verbose,
                keep_mhat=mhat is None and 8 * g.M ** 2 <= (t.ram or 0) // 8)   # cache up to 1/8 of RAM
    st["info"].update(mhat_reused=mhat is not None, gpu_bytes_planned=a["_gpu_bytes"], lb_dtype="float64",
                      lb_layout="slab", lb_rigorous=True)
    t_call = time.perf_counter()
    r = solve_arcflow_fast(n, g.choice_data(t.p), mhat=mhat, opts=opts)
    if "mhat" in r:
        g.mhat = key + (r["mhat"],)
    st["iterations"], st["setup"] = int(r["iters"]), float(r["setup_s"])
    st["info"].update(per_iter_ms=round(r["per_iter_ms"], 3), mhat_s=round(r["mhat_s"], 3))
    return {"gap_abs": "tol", "gap_rel": "tol", "time_limit": "time", "maxiter": "iters"}.get(
        r["stop_reason"], "stalled")


def _gurobi_threads(g, threads):
    """Gurobi threads for this geometry's models; cached models are rebuilt when it changes."""
    from .common.gurobi_env import _ENV
    want = 0 if threads is None else int(threads)          # 0 = Gurobi's automatic choice
    if g.gurobi_threads != want:
        _ENV.setParam("Threads", want)
        g.flow_qp = g.bm_qp = g.cgip = g.cgipheuristic = None
        g.gurobi_threads = want


def _run_exact_qp(which):
    def run(t):
        from .exact_qp import flow_bm_qp
        from .common.certificate import fw_lb_rows, push_flow, slab_index, zeta_rows
        g, a, st = t.geom, t.args, t.state
        _gurobi_threads(g, a["threads"])
        t_b = time.perf_counter()
        if which == "bm_qp":
            if g.bm_qp is None:
                flow_bm_qp.build_BM(g)
            g.bm_qp.model(t.W)                              # the persistent Gurobi model, built once
        else:
            if g.flow_qp is None:
                flow_bm_qp.build_arc_flow(g)
            g.flow_qp.model(g.M, t.W)
        st["setup"] = time.perf_counter() - t_b
        remaining = t.deadline - time.perf_counter()
        if remaining <= 0:
            return "time"
        solve = flow_bm_qp.solve_BM_qp if which == "bm_qp" else flow_bm_qp.solve_arc_flow_qp
        out = solve(g, t.p, vcov=t.W, time_limit=remaining, mem_limit=a["mem_limit"], method=a["method"],
                    verbose=t.verbose)
        st["iterations"] = 1
        st["info"]["gurobi_status"] = int(out["status"])
        if out.get("p_pred") is None:                                   # Gurobi returned no point
            return "time" if out["status"] == 9 else "stalled"
        # Gurobi's point meets the constraints only to Gurobi's tolerances, and its objective is an
        # estimate of dist, not a bound. Certify the point like any iterate: ub = the distance of
        # the point pushed to an exact unit flow, lb = the Frank-Wolfe bound at Gurobi's point.
        n = g.num_alternatives
        obs = slab_index(n, g.menu.astype(np.int64), g.alternative)
        flow = out["r"] if which == "flow_qp" else g.bm_qp.K @ out["p"]   # bm_qp: q = K p~
        y = zeta_rows(n, push_flow(n, flow), obs)
        d = y - t.p
        st["ub"] = math.sqrt(max(0.0, float(d @ (d if t.W is None else t.W @ d))))
        st["lb"] = fw_lb_rows(n, obs, np.asarray(out["p_pred"], dtype=np.float64), t.p, t.W)[0]
        st["p_fit"] = y
        st["trace"].append((1, time.perf_counter() - t.t0, st["ub"], st["lb"]))
        st["info"]["gurobi_distance"] = math.sqrt(max(0.0, float(out["obj"])) + t.const)   # not a bound
        return "time" if out["status"] == 9 else "tol"
    return run


def _run_cg(heur):
    def run(t):
        from .ip_cg import cgip
        g, a, st = t.geom, t.args, t.state
        _gurobi_threads(g, a["threads"])
        t_b = time.perf_counter()
        if heur and g.cgipheuristic is None:
            cgip.build_cgipheuristic(g)
        if not heur and g.cgip is None:
            cgip.build_cgip(g)
        st["setup"] = time.perf_counter() - t_b
        remaining = t.deadline - time.perf_counter()
        if remaining <= 0:
            return "time"

        def on_iter(it, UB, LB, orders, ip_calls, mode):      # after each new column: the trace
            st["iterations"] = it
            st["ub"], st["lb"] = min(st["ub"], float(UB)), max(st["lb"], float(LB))
            st["trace"].append((it, time.perf_counter() - t.t0, st["ub"], st["lb"]))

        old = cgip.HOOKS["on_iter"]
        cgip.HOOKS["on_iter"] = on_iter
        try:
            kw = dict(vcov=t.W, abs_tol=t.abs_tol, rel_tol=t.rel_tol, max_iters=int(a["max_iters"]),
                      seed=t.seed, verbose=t.verbose, qp_backend=a["qp_backend"], strict=True,
                      time_limit=remaining)
            if getattr(t, "below", None) is not None:          # Session thresholds: UB < ub_stop, LB >= lb_stop
                kw["ub_stop"] = t.below
            if getattr(t, "above", None) is not None:
                kw["lb_stop"] = t.above
            if heur:
                out = cgip.solve_cgipheuristic_rand(g, t.p, tol=a["tol"], n_restart=int(a["n_restart"]), **kw)
            else:
                out = cgip.solve_cgip(g, t.p, mode="ip", **kw)
        finally:
            cgip.HOOKS["on_iter"] = old
        st["ub"], st["lb"] = float(out["UB"]), float(out["LB"])
        st["p_fit"] = np.asarray(out["p_pred"], dtype=np.float64)
        st["iterations"] = int(out["iters"])
        if not st["trace"] or st["trace"][-1][0] < st["iterations"]:   # the hook skips the last column
            st["trace"].append((st["iterations"], time.perf_counter() - t.t0, st["ub"], st["lb"]))
        st["info"].update(columns=int(out["n_columns"]), ip_calls=int(out["ip_calls"]),
                          master_kkt=float(out["master_kkt"]))
        if out.get("decision") in ("below", "above"):
            return "threshold"
        if out["converged"]:
            return "tol"
        if out["stop_reason"] == "repeat":
            st["message"] = ("the pricing returned a column already in the master before the gap "
                             "closed (master QP accuracy %.1e; qp_backend 'native' is exact)"
                             % out["master_kkt"])
            return "stalled"
        return "time" if out["stop_reason"] == "time_limit" else "iters"
    return run


_RUN = {"bm_admm": _run_bm_admm, "flow_cg": _run_flow_cg, "flow_admm": _run_flow_admm, "ip_cg": _run_cg(False),
        "ip_cg_heur": _run_cg(True), "bm_qp": _run_exact_qp("bm_qp"), "flow_qp": _run_exact_qp("flow_qp")}


# ---------------------------------------------------------------------- small helpers
def _method_name(method):
    """The method's name, old names accepted (ALIASES)."""
    name = str(method).strip().lower().replace("-", "_")
    name = ALIASES.get(name, name)
    if name not in METHODS:
        raise ValueError("unknown method %r; choose from %s" % (method, METHODS))
    return name


def _as_int(x, what):
    a = np.asarray(x)
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or not np.all(a == np.floor(a)):
            raise ValueError("%s must hold integers" % what)
    elif a.dtype.kind not in "iu":
        raise ValueError("%s must hold integers" % what)
    return a.astype(np.int64)


def _is_identity(V):
    n = V.shape[0]
    if sp.issparse(V):
        return (V - sp.identity(n, dtype=V.dtype, format="csr")).count_nonzero() == 0
    return np.array_equal(V, np.eye(n))


def _is_oom(e):
    return isinstance(e, MemoryError) or "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()


def _free_gpu():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _fits_gpu(nbytes):
    """Whether nbytes fit the share of free GPU memory a method may use, once the pool is freed."""
    _free_gpu()
    return nbytes <= defaults.AUTO["gpu_memory_fraction"] * _machine.gpu_free_bytes()


def _machine_line(prof):
    g = prof.get("gpu")
    gpu = ("%s (%s)" % (g["name"], "used" if _machine.can_use_gpu(prof) else "not used")) if g else "no GPU"
    gu = prof.get("gurobi")
    return "%s; %s cores; %s" % (gpu, prof.get("cpu_count"),
                                 ("Gurobi %s" % gu["version"]) if gu else "no Gurobi")
