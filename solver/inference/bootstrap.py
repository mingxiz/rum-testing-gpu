"""Hong and Li's bootstrap for the RUM distance: a certified p-value and confidence interval.

    import pandas as pd
    from solver.rum_solver import RUM
    from solver.inference import cells_from_panel
    panel = pd.read_csv("final_data/Application_DSR/raw_data/overload_data_25_long.csv")
    menu, alt, p_hat = cells_from_panel(panel, 25, alternatives=[0])
    rum = RUM(menu, alt, p_hat, num_alternatives=25)
    boot = rum.bootstrap(panel=panel)              # n = 520 respondents; defaults: solver/defaults.py BOOTSTRAP
    print(boot.p_value())                          # alpha = 0.05
    print(boot.confidence_interval())              # one-sided: [lower, inf)
    print(boot.distance_interval())                # two-sided, the statistical supplement's
    boot.save("dsr_alt0_bootstrap.json")

    boot = rum.bootstrap(replies=replies)          # AI style, n = 100: each menu's replies resampled

The two inputs, the panel (respondent, menu, chosen) and the replies (menu, chosen), one row per
choice, and the draws made from them are in draws.py, the formulas and their certified forms in
hong_li.py. Nothing is solved until a result is asked for. The first p_value() then

    certifies p_hat   with a Session (one plan for every solve), keeping its state and fitted
                      point: dist(p_hat) in [L0, U0], and the band tau = s [L0, U0]
    screens           every draw at once: ||p_fit - P_b|| bounds dist(P_b) from above, no solve
    solves            the draws still open, warm from p_hat, each stopping once it clears the band
    refines           up to refine times at a tenfold tighter tolerance: p_hat first (its bracket
                      sets the band), then the draws still undecided

and confidence_interval() reuses every bracket, solving to tolerance only the draws that can move
its quantile. distance_interval() is the two-sided interval of the paper's statistical supplement:
the certified distance widened by the (1 - alpha)-quantile of ||p_hat_b - p_hat||, which the screen
computes for every draw, so it solves nothing. Every answer comes from certified brackets; an
engine's stop reason decides nothing.
"""
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp

from .. import defaults
from . import hong_li
from .draws import Draws, MenuDraws

VERSION = 1
NEVER = 127                     # exit_flag of a draw that was never solved
CHUNK_BYTES = 1 << 23           # perturbed points are rebuilt at most 8 MB at a time for solving


@dataclass
class PValue:
    """The bootstrap p-value of H0: dist = 0 (the data are rationalizable by a RUM)."""
    p: float                    # the reported p-value: p_hi, an upper bound on the exact Hong-Li p-value
    p_lo: float                 # what resolving every undecided draw could at best give
    p_hi: float
    alpha: float
    reject: bool                # p <= alpha
    certain: bool               # the decision holds for every value in [p_lo, p_hi]
    n_below: int
    n_above: int
    n_undecided: int            # counted 1, as the reference counts them
    n_screened: int             # decided below by the screen, with no solve
    n_solved: int               # draws solved at least once
    B: int
    tau: tuple                  # (tau_lo, tau_hi) = s (L0, U0)
    dist_hat: tuple             # (L0, U0): dist(p_hat) lies here
    T: tuple                    # r_n (L0, U0): the test statistic
    stopped_early: bool         # stop_when_decided ended the run with draws still open
    time: float                 # seconds of this call, solves included

    def __repr__(self):
        return _rows([("p", "%.6g  (bracket [%.6g, %.6g])" % (self.p, self.p_lo, self.p_hi)),
                      ("decision", "%s at alpha = %g%s" % ("reject H0" if self.reject else "do not reject H0",
                                                          self.alpha, "" if self.certain else " (not certain)")),
                      ("draws", "%d: %d below, %d above, %d undecided; %d screened, %d solved"
                       % (self.B, self.n_below, self.n_above, self.n_undecided, self.n_screened, self.n_solved)),
                      ("dist(p_hat)", "[%.10g, %.10g]" % self.dist_hat),
                      ("tau", "[%.10g, %.10g]" % self.tau), ("T", "[%.10g, %.10g]" % self.T),
                      ("time", "%.2f s%s" % (self.time, ", stopped once the decision was locked"
                                             if self.stopped_early else ""))])


@dataclass
class Interval:
    """The one-sided (1 - alpha) confidence interval [lower, inf) for dist."""
    lower: float                # conservative: the reference's arithmetic with L0 and the draws' upper bounds
    lower_bracket: tuple        # the exact lower end lies here
    lower_clipped: float        # max(0, lower), since dist >= 0
    upper: float                # inf
    alpha: float
    q: tuple                    # (q_lo, q_hi): the (1 - alpha)-quantile of (dist(P_b) - dist(p_hat)) / eps_n
    reject: bool                # lower > 0: 0 lies outside the interval (the reference's dual test)
    n_solved: int               # draws solved to tolerance by this call
    time: float

    def __repr__(self):
        return _rows([("interval", "[%.6g, inf)  at %g %%" % (self.lower, 100 * (1 - self.alpha))),
                      ("lower end", "in [%.6g, %.6g]; max(0, lower) = %.6g" % (self.lower_bracket + (self.lower_clipped,))),
                      ("q", "in [%.6g, %.6g]" % self.q),
                      ("H0: dist = 0", "rejected (0 is outside)" if self.reject else "not rejected"),
                      ("time", "%.2f s, %d draws solved to tolerance" % (self.time, self.n_solved))])


@dataclass
class DistanceInterval:
    """The two-sided (1 - alpha) interval for dist of the statistical supplement: [max(0, L0 - c), U0 + c]
    with c the (1 - alpha)-quantile of ||p_hat_b - p_hat|| over the draws (in the run's norm)."""
    lower: float                # max(0, L0 - c)
    upper: float                # U0 + c
    alpha: float
    half_width: float           # c
    dist_hat: tuple             # (L0, U0)
    reject: bool                # lower > 0: 0 lies outside the interval
    time: float

    def __repr__(self):
        return _rows([("interval", "[%.6g, %.6g]  at %g %%, two-sided" % (self.lower, self.upper, 100 * (1 - self.alpha))),
                      ("half-width", "%.6g, the (1 - alpha)-quantile of ||p_hat_b - p_hat||" % self.half_width),
                      ("dist(p_hat)", "[%.10g, %.10g]" % self.dist_hat),
                      ("H0: dist = 0", "rejected (0 is outside)" if self.reject else "not rejected"),
                      ("time", "%.2f s, no draw solved" % self.time)])


class Bootstrap:
    """rum.bootstrap(panel= or replies=, n=None, ...): see the module docstring; the defaults are defaults.BOOTSTRAP."""

    def __init__(self, rum, n=None, panel=None, replies=None, B=None, weights=None, seed=None, r_n=None, eps_n=None,
                 method=None, method_args=None, abs_tol=None, rel_tol=None, refine=None, max_time=None,
                 draw_time=None, vcov=None, machine=None, checkpoint=None, verbose=False):
        d = defaults.BOOTSTRAP
        pick = lambda value, key: d[key] if value is None else value      # noqa: E731
        self.rum, self.p_hat = rum, rum.choice_probability
        self.B = int(pick(B, "B"))
        self.abs_tol, self.rel_tol = float(pick(abs_tol, "abs_tol")), float(pick(rel_tol, "rel_tol"))
        self.refine, self.max_time = int(pick(refine, "refine")), float(pick(max_time, "max_time"))
        self.draw_time = float(pick(draw_time, "draw_time"))
        if self.abs_tol < 0 or self.rel_tol < 0 or self.abs_tol + self.rel_tol == 0 or self.refine < 0:
            raise ValueError("abs_tol and rel_tol must be >= 0 and not both 0, and refine >= 0")
        self.method, self.method_args, self.vcov, self.machine = method, method_args, vcov, machine
        self.verbose = verbose
        if (panel is None) == (replies is None):
            raise ValueError("give the choices behind p_hat once: panel= (respondent, menu, chosen; respondents are "
                             "resampled) or replies= (menu, chosen; each menu's replies are resampled)")
        if panel is not None:
            self.draws = Draws(panel, rum.menu, rum.alternative, self.p_hat, self.B, pick(weights, "weights"),
                               pick(seed, "seed"))
        elif weights not in (None, MenuDraws.weights):
            raise ValueError("weights = %r is for panel=; replies= are resampled menu by menu" % (weights,))
        else:
            self.draws = MenuDraws(replies, rum.menu, rum.alternative, self.p_hat, self.B, pick(seed, "seed"))
        if n is None:                               # the panel's respondents, or the most replies on a menu
            n = self.draws.default_n
        self.r_n, self.eps_n, self.s = hong_li.scaling(n, r_n, eps_n)
        self.n = int(n)
        self.draws.check_n(self.n)
        self.settings = dict(n=self.n, input=self.draws.input, B=self.B, weights=self.draws.weights,
                             seed=self.draws.seed, r_n=self.r_n, eps_n=self.eps_n, method=method,
                             method_args=method_args, abs_tol=self.abs_tol, rel_tol=self.rel_tol, refine=self.refine,
                             max_time=self.max_time, draw_time=self.draw_time, weighted=vcov is not None)
        self.fingerprints = {"data": _sha(np.column_stack([rum.menu.astype(np.float64),
                                                           rum.alternative.astype(np.float64), self.p_hat])),
                             self.draws.input: self.draws.input_fingerprint, "draws": None}
        B = self.B
        self.L, self.U = np.zeros(B), np.full(B, np.inf)       # every draw's bracket of dist(P_b)
        self.tol_stop = np.full(B, np.inf)                     # tightest abs_tol of any solve of the draw
        self.tol_full = np.full(B, np.inf)                     # ... of a solve without the band's stops
        self.flag = np.full(B, NEVER, dtype=np.int8)           # exit_flag of the draw's last solve
        self.L0 = self.U0 = None
        self.certificates, self.results = [], {"p_value": {}, "confidence_interval": {}, "distance_interval": {}}
        self.dev = None                                         # every draw's ||p_hat_b - p_hat|| (the screen)
        self.screened, self.n_screened = False, 0
        self.time = dict(certificate=0.0, screen=0.0, solve=0.0)
        self.checkpoint, self._saved, self._resumed_draws = checkpoint, time.perf_counter(), None
        self._session = None
        if checkpoint and os.path.exists(checkpoint):
            self._resume(checkpoint)

    # ------------------------------------------------------------------ results
    def p_value(self, alpha=0.05, stop_when_decided=False):
        """The certified p-value; stop_when_decided ends once the decision at alpha is locked."""
        alpha, t0 = hong_li.check_alpha(alpha), time.perf_counter()
        self._start()
        stopped = False
        if self.U0 > hong_li.ZERO:
            locked = (lambda: hong_li.locked(*self._above_open(), self.B, alpha)) if stop_when_decided else None
            tol_a, tol_r = self.abs_tol, self.rel_tol
            for rnd in range(self.refine + 1):
                if rnd:
                    if not self._undecided().any() or (locked and locked()):
                        break
                    tol_a, tol_r = tol_a / 10, tol_r / 10
                    if self.U0 - self.L0 > tol_a:              # the band is p_hat's bracket times s
                        self._certify(tol_a, tol_r)
                todo = np.flatnonzero(self._undecided() & (self.tol_stop > tol_a))
                self._solve(todo, tol_a, tol_r, stops=True, until=locked)
            stopped = bool(locked and locked() and self._undecided().any())
        dec = self._decisions()
        p_lo, p_hi = hong_li.p_bracket(dec)
        res = PValue(p=p_hi, p_lo=p_lo, p_hi=p_hi, alpha=alpha, reject=p_hi <= alpha,
                     certain=p_hi <= alpha or p_lo > alpha, n_below=int((dec == hong_li.BELOW).sum()),
                     n_above=int((dec == hong_li.ABOVE).sum()), n_undecided=int((dec == hong_li.UNDECIDED).sum()),
                     n_screened=self.n_screened, n_solved=int((self.flag != NEVER).sum()), B=self.B, tau=self.tau,
                     dist_hat=(self.L0, self.U0), T=(self.r_n * self.L0, self.r_n * self.U0), stopped_early=stopped,
                     time=time.perf_counter() - t0)
        self.results["p_value"][repr(alpha)] = asdict(res)
        self._save_checkpoint(force=True)
        return res

    def confidence_interval(self, alpha=0.05):
        """The certified one-sided interval [lower, inf): draws are solved to tolerance, largest upper
        bound first, only while their bracket overlaps the band [Q(L), Q(U)] of the quantile."""
        alpha, t0 = hong_li.check_alpha(alpha), time.perf_counter()
        self._start()
        n0 = int(np.isfinite(self.tol_full).sum())
        batch = max(1, math.ceil(alpha * self.B / 4))
        tol_a, tol_r = self.abs_tol, self.rel_tol
        for rnd in range(self.refine + 1):
            if rnd:
                low, high, _ = hong_li.ci_lower(self.L0, self.U0, self.L, self.U, alpha, self.r_n, self.eps_n)
                if high - low <= 2 * self.abs_tol:
                    break
                tol_a, tol_r = tol_a / 10, tol_r / 10
                if self.U0 - self.L0 > tol_a:
                    self._certify(tol_a, tol_r)
            while True:
                q_l, q_u = hong_li.quantile_band(self.L, self.U, alpha)
                hot = np.flatnonzero((self.U >= q_l) & (self.L <= q_u) & (self.tol_full > tol_a))
                if hot.size == 0:
                    break
                self._solve(np.sort(hot[np.argsort(-self.U[hot], kind="stable")[:batch]]), tol_a, tol_r, stops=False)
        low, high, q = hong_li.ci_lower(self.L0, self.U0, self.L, self.U, alpha, self.r_n, self.eps_n)
        res = Interval(lower=low, lower_bracket=(low, high), lower_clipped=max(0.0, low), upper=math.inf, alpha=alpha,
                       q=q, reject=low > 0, n_solved=int(np.isfinite(self.tol_full).sum()) - n0,
                       time=time.perf_counter() - t0)
        self.results["confidence_interval"][repr(alpha)] = asdict(res)
        self._save_checkpoint(force=True)
        return res

    def distance_interval(self, alpha=0.05):
        """The certified two-sided interval [max(0, L0 - c), U0 + c], c the (1 - alpha)-quantile of
        ||p_hat_b - p_hat|| (hong_li.distance_interval): the p_hat certificate and the screen's pass
        over the draws, no draw solved."""
        alpha, t0 = hong_li.check_alpha(alpha), time.perf_counter()
        self._start()
        if self.dev is None:
            self.dev = np.concatenate([self._norm(block - self.p_hat) for _lo, _hi, block in self.draws.blocks()])
        lower, upper, c = hong_li.distance_interval(self.L0, self.U0, self.dev, alpha)
        res = DistanceInterval(lower=lower, upper=upper, alpha=alpha, half_width=c, dist_hat=(self.L0, self.U0),
                               reject=lower > 0, time=time.perf_counter() - t0)
        self.results.setdefault("distance_interval", {})[repr(alpha)] = asdict(res)
        self._save_checkpoint(force=True)
        return res

    @property
    def tau(self):
        return (self.s * self.L0, self.s * self.U0)

    @property
    def session(self):
        if self._session is None:
            self._session = self.rum.session(method=self.method, method_args=self.method_args, vcov=self.vcov,
                                             machine=self.machine)
        return self._session

    # ------------------------------------------------------------------ the steps
    def _start(self):
        if self.session.fit is None:                # first call, or resumed from a checkpoint
            self._certify(self.abs_tol, self.rel_tol)
        if not self.screened:
            self._screen()

    def _certify(self, abs_tol, rel_tol):
        """Certify p_hat and keep its state; a second certificate tightens the first (both hold)."""
        t0 = time.perf_counter()
        r = self.session.solve(self.p_hat, abs_tol=abs_tol, rel_tol=rel_tol, max_time=self.max_time, keep=True,
                               verbose=self.verbose)
        if not math.isfinite(r.ub):
            raise RuntimeError("p_hat could not be certified: %s (%s)" % (r.status, r.message))
        self.L0 = r.lb if self.L0 is None else max(self.L0, r.lb)
        self.U0 = r.ub if self.U0 is None else min(self.U0, r.ub)
        self.certificates.append(dict(abs_tol=abs_tol, rel_tol=rel_tol, lb=r.lb, ub=r.ub, exit_flag=r.exit_flag,
                                      status=r.status, message=r.message, iterations=r.iterations, time=r.time,
                                      method=r.method, method_args=r.method_args, machine=r.machine))
        self.time["certificate"] += time.perf_counter() - t0
        self._say("p_hat: dist in [%.10g, %.10g], %s (%s, %.1f s)" % (r.lb, r.ub, r.status, r.method, r.time))

    def _screen(self):
        """U_b = ||p_fit - P_b|| for every draw: p_fit is a RUM point, so it bounds dist(P_b)."""
        t0, fit = time.perf_counter(), self.session.fit
        self.dev = np.empty(self.B)
        for lo, hi, block in self.draws.blocks():
            self.dev[lo:hi] = self._norm(block - self.p_hat)
            if fit is not None:
                ub = self._norm(hong_li.perturb(self.p_hat, block, self.r_n, self.eps_n) - fit)
                self.U[lo:hi] = np.minimum(self.U[lo:hi], ub)
        self.fingerprints["draws"] = self.draws.fingerprint
        if self._resumed_draws not in (None, self.draws.fingerprint):
            raise ValueError("the draws differ from the checkpoint's (fingerprint %s, checkpoint %s)"
                             % (self.draws.fingerprint, self._resumed_draws))
        self.screened, self.n_screened = True, int((self.U < self.tau[0]).sum())
        self.time["screen"] += time.perf_counter() - t0
        self._say("screen: %d of %d draws below tau_lo = %.10g with no solve (%.1f s)"
                  % (self.n_screened, self.B, self.tau[0], time.perf_counter() - t0))

    def _solve(self, idx, abs_tol, rel_tol, stops, until=None):
        """Solve the draws idx (in this order) at the tolerance, with the band's early stops or without."""
        t0, s = time.perf_counter(), self.session
        tau_lo, tau_hi = self.tau
        step = max(1, CHUNK_BYTES // (8 * self.draws.M))
        report = max(1, len(idx) // 10)
        for k, first in enumerate(range(0, len(idx), step)):
            chunk = idx[first:first + step]
            P = hong_li.perturb(self.p_hat, self.draws.rows(chunk), self.r_n, self.eps_n)
            for j, b in enumerate(chunk):
                r = s.solve(P[j], abs_tol=abs_tol, rel_tol=rel_tol, max_time=self.draw_time, warm=True,
                            below=tau_lo if stops else None, above=tau_hi if stops else None)
                if r.exit_flag == -4:
                    self._save_checkpoint(force=True)
                    raise KeyboardInterrupt("interrupted at draw %d; the checkpoint keeps every bracket" % b)
                self.L[b], self.U[b] = max(self.L[b], r.lb), min(self.U[b], r.ub)
                self.tol_stop[b] = min(self.tol_stop[b], abs_tol)
                if not stops:
                    self.tol_full[b] = min(self.tol_full[b], abs_tol)
                self.flag[b] = r.exit_flag
                done = first + j + 1
                if self.verbose and done % report == 0:
                    self._say("  %d / %d draws at abs_tol %g (%.0f s)" % (done, len(idx), abs_tol, time.perf_counter() - t0))
                self._save_checkpoint()
                if until is not None and done % 25 == 0 and until():
                    self.time["solve"] += time.perf_counter() - t0
                    return
        self.time["solve"] += time.perf_counter() - t0

    def _norm(self, D):
        if self.vcov is None:
            return np.sqrt(np.einsum("ij,ij->i", D, D))
        DV = (self.vcov @ D.T).T if sp.issparse(self.vcov) else D @ np.asarray(self.vcov)
        return np.sqrt(np.maximum(np.einsum("ij,ij->i", np.asarray(DV), D), 0.0))

    def _decisions(self):
        if self.U0 <= hong_li.ZERO:                 # rationalizable: tau = 0 and every draw counts 1
            return np.full(self.B, hong_li.ABOVE, dtype=np.int8)
        return hong_li.decide(self.L, self.U, *self.tau)

    def _undecided(self):
        return self._decisions() == hong_li.UNDECIDED

    def _above_open(self):
        dec = self._decisions()
        return int((dec == hong_li.ABOVE).sum()), int((dec == hong_li.UNDECIDED).sum())

    # ------------------------------------------------------------------ record, checkpoint
    def record(self):
        """Everything needed to report or resume the run, as plain Python values."""
        s = self._session
        return dict(version=VERSION, settings=self.settings, fingerprints=self.fingerprints,
                    plan=None if s is None else dict(method=s.method, method_args=s.method_args,
                                                    auto_reason=s.auto_reason),
                    p_hat=dict(lb=self.L0, ub=self.U0, certificates=self.certificates),
                    tau=None if self.L0 is None else list(self.tau), scaling=dict(r_n=self.r_n, eps_n=self.eps_n, s=self.s),
                    screen=dict(done=self.screened, n_below=self.n_screened),
                    **{self.draws.input: self.draws.summary()},
                    draws=dict(lb=self.L.tolist(), ub=self.U.tolist(), tol_stop=self.tol_stop.tolist(),
                               tol_full=self.tol_full.tolist(), exit_flag=self.flag.tolist()),
                    results=self.results, time=self.time, when=time.strftime("%Y-%m-%d %H:%M:%S"))

    def save(self, path):
        """Write record() as JSON (infinite bounds as Infinity)."""
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.record(), f, indent=1, default=_plain)
        os.replace(tmp, path)

    def _save_checkpoint(self, force=False):
        if self.checkpoint and (force or time.perf_counter() - self._saved >= defaults.BOOTSTRAP["checkpoint_every"]):
            self.save(self.checkpoint)
            self._saved = time.perf_counter()

    def _resume(self, path):
        rec = json.load(open(path))
        same = dict(version=rec.get("version") == VERSION,
                    settings=json.loads(json.dumps(self.settings, default=_plain)) == rec.get("settings"),
                    data=rec.get("fingerprints", {}).get("data") == self.fingerprints["data"],
                    **{self.draws.input: rec.get("fingerprints", {}).get(self.draws.input)
                       == self.fingerprints[self.draws.input]})
        if not all(same.values()):
            raise ValueError("checkpoint %s belongs to another run (differs in: %s); delete it or pass another path"
                             % (path, ", ".join(k for k, v in same.items() if not v)))
        dr = rec["draws"]
        self.L, self.U = np.array(dr["lb"], dtype=np.float64), np.array(dr["ub"], dtype=np.float64)
        self.tol_stop = np.array(dr["tol_stop"], dtype=np.float64)
        self.tol_full = np.array(dr["tol_full"], dtype=np.float64)
        self.flag = np.array(dr["exit_flag"], dtype=np.int8)
        self.L0, self.U0 = rec["p_hat"]["lb"], rec["p_hat"]["ub"]
        self.certificates, self.results, self.time = rec["p_hat"]["certificates"], rec["results"], rec["time"]
        self._resumed_draws = rec["fingerprints"]["draws"]
        self._say("resumed %s: %d draws solved before" % (path, int((self.flag != NEVER).sum())))

    # ------------------------------------------------------------------ the rest
    def close(self):
        if self._session is not None:
            self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _say(self, msg):
        if self.verbose:
            print("[bootstrap] " + msg, flush=True)

    def __repr__(self):
        rows = [("bootstrap", "n = %d, B = %d %s (seed %s); r_n = %.6g, eps_n = %.6g, s = %.6g"
                 % (self.n, self.B, self.draws.describe, self.draws.seed, self.r_n, self.eps_n, self.s))]
        if self._session is not None:
            s = self._session
            rows.append(("plan", "%s %s%s" % (s.method, s.method_args.get("dtype", ""),
                                              "  [%s]" % s.auto_reason if s.auto_reason else "")))
        if self.L0 is not None:
            rows.append(("dist(p_hat)", "[%.10g, %.10g]; tau = [%.10g, %.10g]" % ((self.L0, self.U0) + self.tau)))
            dec = self._decisions()
            rows.append(("draws", "%d screened below; %d solved; %d below, %d above, %d undecided"
                         % (self.n_screened, int((self.flag != NEVER).sum()), int((dec == 0).sum()),
                            int((dec == 1).sum()), int((dec == 2).sum()))))
        for a, r in self.results["p_value"].items():
            rows.append(("p_value", "%.6g at alpha %s (%s)" % (r["p"], a, "reject" if r["reject"] else "do not reject")))
        for a, r in self.results["confidence_interval"].items():
            rows.append(("interval", "[%.6g, inf) at alpha %s" % (r["lower"], a)))
        for a, r in self.results.get("distance_interval", {}).items():
            rows.append(("two-sided", "[%.6g, %.6g] at alpha %s" % (r["lower"], r["upper"], a)))
        rows.append(("time", ", ".join("%s %.1f s" % kv for kv in self.time.items())))
        return _rows(rows)


def _rows(rows):
    w = max(len(k) for k, _ in rows)
    return "\n".join("%s : %s" % (k.rjust(w), v) for k, v in rows)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _plain(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError("not JSON serializable: %r" % type(x))
