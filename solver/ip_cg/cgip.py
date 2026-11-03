"""Column-generation / integer-programming backends for the RUM projection: CGIP (exact IP
pricing, mode 'ip') and CGIP-heuristic (random-insertion pricing, mode 'rand', certified by the
IP when it stalls). Mode 'best' (exhaustive best-insertion) is the earlier heuristic, kept because
run scripts outside solver/ wrap CGIPBuild.heuristic_price.

Port of Code_CGFW_0114 (B_solve_rum_CG / B_IP_pricing / B_CG_heuristic_best /
B_QP), rewritten to operate directly on a solver.rum.RUM instance's
per-observation (menu, alternative) layout.

Idea
----
The RUM polytope, in observed coordinates, is the convex hull of the choice
vectors of the deterministic preference rankings: for a ranking r (a strict
order), its column is

    v_r[o] = 1{ the top-ranked member of menu[o] under r == alternative[o] }.

So phi(p) = min_{lambda>=0, sum=1} || V lambda - p ||_Omega, a projection onto
that hull. Column generation solves it without enumerating all n! rankings:

    RMP:      min_lambda ||V lambda - p||^2_Omega   over the current columns V
    pricing:  find the ranking r maximizing v_r . g,  g = Omega (p - V lambda)
    stop:     when the reduced cost  max_r v_r.g - (V lambda).g  < tol.

cgip prices with an exact Gurobi IP each iteration; the heuristic modes price with a
polynomial insertion heuristic and only call the IP to certify that no improving
column remains (so all of them return the exact projection distance).

The pricing IP's variables/constraints depend only on the menu geometry, so the
Gurobi model is built ONCE (cached) and re-solved per iteration with only the
objective (the residual g) changing -- same build-once contract as flow_bm_qp.
"""
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..common.gurobi_env import _ENV        # the one licensed Gurobi env, shared by every model
from ..common.weights import _as_weight     # weight-matrix normaliser (shared with exact_qp)
from ..common.solve_qp import solve_rmp     # shared master QP; looked up at call time (run scripts wrap it)

# Optional caller hooks for long runs (both default off; the loop is unchanged when unset):
#   HOOKS['on_iter'](it=, UB=, LB=, orders=, ip_calls=, mode=) is called once per CG iteration after the
#   termination tests, before the next pricing call; a truthy return value stops the solve with that
#   stop_reason. HOOKS['init_orders'] (a list of rankings) seeds the master instead of the random start
#   column and is consumed by the next _cg_solve call (checkpoint resume). Results carry 'orders'.
HOOKS = {"on_iter": None, "init_orders": None}

# Numba accelerates the best-insertion heuristic (pure arithmetic over arrays).
# The kernels below are written to run identically with or without numba, so if
# numba is missing `njit` becomes a no-op decorator and they execute as plain
# Python (same result, just slower).
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]                            # @njit used bare

        def _wrap(func):
            return func
        return _wrap                                 # @njit(cache=True) form


# --------------------------------------------------------------------------- #
#  Numba kernels: menu layout is passed as padded arrays (members_arr / sizes) #
#  so everything is fixed-shape numpy -- njit-able and bit-identical to the    #
#  pure-Python path (same o-ascending summation order).                        #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _partial_score_nb(pos, members_arr, menu_sizes, obs_alt, g, M):
    """v_r . g over observations whose menu is fully placed (pos[a] >= 0 for all
    members); menus with an unplaced member contribute nothing."""
    s = 0.0
    for o in range(M):
        top = -1
        best = 1 << 30
        ok = True
        for j in range(menu_sizes[o]):
            a = members_arr[o, j]
            r = pos[a]
            if r < 0:
                ok = False
                break
            if r < best:
                best = r
                top = a
        if ok and top == obs_alt[o]:
            s += g[o]
    return s


@njit(cache=True)
def _column_nb(pos, members_arr, menu_sizes, obs_alt, M):
    """Choice column for a complete ranking (all members placed)."""
    v = np.zeros(M, dtype=np.float64)
    for o in range(M):
        top = -1
        best = 1 << 30
        for j in range(menu_sizes[o]):
            a = members_arr[o, j]
            r = pos[a]
            if r < best:
                best = r
                top = a
        if top == obs_alt[o]:
            v[o] = 1.0
    return v


@njit(cache=True)
def _heuristic_price_nb(g, members_arr, menu_sizes, obs_alt, n, M, inner):
    """Best-insertion pricing over all n starts. For each start, greedily insert
    the ascending remaining alternatives at the score-maximizing position (ties
    broken to the first alt/position, matching the pure-Python loop). Returns the
    best full ranking's (order, column, reduced_cost)."""
    best_rc = -np.inf
    best_order = np.empty(n, dtype=np.int64)
    best_col = np.zeros(M, dtype=np.float64)
    order = np.empty(n, dtype=np.int64)
    pos_cand = np.empty(n, dtype=np.int64)
    placed = np.empty(n, dtype=np.bool_)
    for start in range(n):
        for a in range(n):
            placed[a] = False
        order[0] = start
        placed[start] = True
        L = 1
        while L < n:
            b_local = -np.inf
            b_alt = -1
            b_pos = -1
            for alt in range(n):                     # ascending unplaced alts
                if placed[alt]:
                    continue
                for p in range(L + 1):               # ascending insert positions
                    for a in range(n):
                        pos_cand[a] = -1
                    for kk in range(L):
                        e = order[kk]
                        pos_cand[e] = kk if kk < p else kk + 1
                    pos_cand[alt] = p
                    sc = _partial_score_nb(pos_cand, members_arr, menu_sizes,
                                           obs_alt, g, M)
                    if sc > b_local:
                        b_local = sc
                        b_alt = alt
                        b_pos = p
            for kk in range(L, b_pos, -1):           # shift right, insert b_alt
                order[kk] = order[kk - 1]
            order[b_pos] = b_alt
            placed[b_alt] = True
            L += 1
        for a in range(n):
            pos_cand[a] = -1
        for kk in range(n):
            pos_cand[order[kk]] = kk
        col = _column_nb(pos_cand, members_arr, menu_sizes, obs_alt, M)
        rc = 0.0
        for o in range(M):
            rc += col[o] * g[o]
        rc -= inner
        if rc > best_rc:
            best_rc = rc
            for kk in range(n):
                best_order[kk] = order[kk]
            for o in range(M):
                best_col[o] = col[o]
    return best_order, best_col, best_rc


@njit(cache=True)
def _rand_price_nb(g, members_arr, menu_sizes, obs_alt, n, M, inner, n_restart,
                   seed):
    """Random-insertion pricing (port of B_CG_heuristic_best_rand). Each restart
    begins at a RANDOM alternative and, at each step, inserts a RANDOM unplaced
    alternative at its best-scoring position (best-insertion only optimizes the
    POSITION, not which alt). Returns the best (order, column, reduced_cost) over
    n_restart random constructions. Cost O(n_restart * n^2 * M) -- a factor ~n^2
    cheaper per construction than the exhaustive best-insertion (which also picks
    WHICH alt each step), at lower column quality."""
    np.random.seed(seed)
    best_rc = -np.inf
    best_order = np.empty(n, dtype=np.int64)
    best_col = np.zeros(M, dtype=np.float64)
    order = np.empty(n, dtype=np.int64)
    pos_cand = np.empty(n, dtype=np.int64)
    remaining = np.empty(n, dtype=np.int64)
    for _rs in range(n_restart):
        start = np.random.randint(n)
        order[0] = start
        nr = 0
        for a in range(n):
            if a != start:
                remaining[nr] = a
                nr += 1
        L = 1
        while L < n:
            ridx = np.random.randint(nr)             # a RANDOM unplaced alt
            alt = remaining[ridx]
            b_pos = 0
            b_local = -np.inf
            for p in range(L + 1):                   # best position for that alt
                for a in range(n):
                    pos_cand[a] = -1
                for kk in range(L):
                    e = order[kk]
                    pos_cand[e] = kk if kk < p else kk + 1
                pos_cand[alt] = p
                sc = _partial_score_nb(pos_cand, members_arr, menu_sizes,
                                       obs_alt, g, M)
                if sc > b_local:
                    b_local = sc
                    b_pos = p
            for kk in range(L, b_pos, -1):
                order[kk] = order[kk - 1]
            order[b_pos] = alt
            remaining[ridx] = remaining[nr - 1]      # drop alt from remaining
            nr -= 1
            L += 1
        for a in range(n):
            pos_cand[a] = -1
        for kk in range(n):
            pos_cand[order[kk]] = kk
        col = _column_nb(pos_cand, members_arr, menu_sizes, obs_alt, M)
        rc = 0.0
        for o in range(M):
            rc += col[o] * g[o]
        rc -= inner
        if rc > best_rc:
            best_rc = rc
            for kk in range(n):
                best_order[kk] = order[kk]
            for o in range(M):
                best_col[o] = col[o]
    return best_order, best_col, best_rc


# --------------------------------------------------------------------------- #
#  Geometry-only build: members, ranking->column, persistent IP pricing model  #
# --------------------------------------------------------------------------- #
class CGIPBuild:
    """Per-observation menu members, the ranking->column map, and a persistent
    Gurobi IP pricing model (linear-ordering polytope + per-menu linking). Built
    once per RUM geometry; reused across CG iterations and bootstrap draws."""

    def __init__(self, rum):
        self.n = int(rum.num_alternatives)
        self.M = int(rum.M)
        menus = rum.menu.astype(np.int64)
        self.obs_alt = rum.alternative.astype(np.int64)
        self.obs_menu = menus
        # members[o] = sorted alternatives of observation o's menu
        self.members = [tuple(i for i in range(self.n) if (int(menus[o]) >> i) & 1)
                        for o in range(self.M)]
        # padded array form (M x max_menu_size, -1 filled) + sizes, for the njit
        # heuristic kernels.
        max_sz = max((len(mem) for mem in self.members), default=0)
        self.members_arr = np.full((self.M, max_sz), -1, dtype=np.int64)
        self.menu_sizes = np.empty(self.M, dtype=np.int64)
        for o, mem in enumerate(self.members):
            self.menu_sizes[o] = len(mem)
            for j, a in enumerate(mem):
                self.members_arr[o, j] = a
        self._build_pricing_model()

    # ---- ranking (order, best->worst) -> column over observations ---------- #
    def _pos(self, order):
        pos = np.empty(self.n, dtype=np.int64)
        for k, a in enumerate(order):
            pos[a] = k
        return pos

    def column(self, pos):
        """Choice column for the ranking whose position array is `pos`
        (pos[a] = rank, 0 = most preferred)."""
        v = np.zeros(self.M, dtype=np.float64)
        for o in range(self.M):
            mem = self.members[o]
            top = mem[0]
            best = pos[top]
            for a in mem[1:]:
                if pos[a] < best:
                    best = pos[a]
                    top = a
            if top == self.obs_alt[o]:
                v[o] = 1.0
        return v

    # ---- persistent IP pricing model -------------------------------------- #
    def _build_pricing_model(self):
        n = self.n
        m = gp.Model(env=_ENV)
        # pairwise preference vars x[i,j] = 1 iff i preferred to j (i != j)
        x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
             for i in range(n) for j in range(n) if i != j}
        for i in range(n):
            for j in range(i + 1, n):
                m.addConstr(x[i, j] + x[j, i] == 1)                 # antisymmetry
        for i in range(n):                                          # transitivity
            for j in range(i + 1, n):
                for k in range(i + 1, n):
                    if k == j:
                        continue
                    m.addConstr(x[i, j] + x[j, k] + x[k, i] <= 2)

        # for each observed menu of size >= 3, a "top" indicator p[mask,a]
        distinct = {}
        for o in range(self.M):
            if len(self.members[o]) >= 3:
                distinct[int(self.obs_menu[o])] = self.members[o]
        pvar = {}
        for mask, mem in distinct.items():
            for a in mem:
                pvar[mask, a] = m.addVar(lb=0.0, ub=1.0, name=f"p_{mask}_{a}")
            for a in mem:                       # p(D,a) <= x[a,b] for all b in D\a
                for b in mem:
                    if b != a:
                        m.addConstr(pvar[mask, a] <= x[a, b])
            m.addConstr(gp.quicksum(pvar[mask, a] for a in mem) == 1)   # simplex

        # observation o -> the model var whose value is v_r[o] (1 iff o's alt is
        # top of o's menu): x[a,b] for a binary menu {a,b}, else p[mask,a].
        obs_var = []
        for o in range(self.M):
            mem = self.members[o]
            a = int(self.obs_alt[o])
            if len(mem) == 2:
                b = mem[0] if mem[1] == a else mem[1]
                obs_var.append(x[a, b])
            else:
                obs_var.append(pvar[int(self.obs_menu[o]), a])
        m.update()
        self._model, self._x, self._obs_var = m, x, obs_var

    def ip_price(self, g):
        """IP pricing: return (order, bound) where order is the incumbent
        ranking (best->worst) for max v_r . g and bound is Gurobi's ObjBound,
        >= that maximum at any MIP gap (ObjVal is <= it, so an LB built from
        ObjVal can overstate the distance). bound == the maximum once the IP
        is closed (MIPGap = 0). order is None when a time limit stopped the IP
        before it found any ranking."""
        m = self._model
        m.setObjective(gp.quicksum(float(g[o]) * self._obs_var[o]
                                   for o in range(self.M)), GRB.MAXIMIZE)
        m.optimize()
        if m.SolCount == 0:
            return None, float(m.ObjBound)
        n = self.n
        score = np.zeros(n)                     # out-degree = # alternatives beaten
        for i in range(n):
            for j in range(n):
                if i != j and self._x[i, j].X > 0.5:
                    score[i] += 1
        order = list(np.argsort(-score, kind='stable'))   # best (beats most) first
        return order, float(m.ObjBound)

    # ---- best-insertion heuristic pricing (njit kernel) ------------------- #
    def heuristic_price(self, g, inner):
        """Best-insertion pricing over all n starts, delegated to the njit
        kernel. Returns (order, column, reduced_cost) of the best ranking found.
        Identical algorithm/result to a pure-Python best-insertion; just fast."""
        g = np.ascontiguousarray(g, dtype=np.float64)
        order, col, rc = _heuristic_price_nb(
            g, self.members_arr, self.menu_sizes, self.obs_alt,
            self.n, self.M, float(inner))
        return [int(a) for a in order], col, float(rc)

    # ---- random-insertion heuristic pricing (njit kernel) ----------------- #
    def rand_price(self, g, inner, n_restart, seed):
        """Random-insertion pricing (n_restart random constructions, best kept),
        delegated to the njit kernel. Stochastic in `seed`. Returns
        (order, column, reduced_cost)."""
        g = np.ascontiguousarray(g, dtype=np.float64)
        order, col, rc = _rand_price_nb(
            g, self.members_arr, self.menu_sizes, self.obs_alt,
            self.n, self.M, float(inner), int(n_restart), int(seed))
        return [int(a) for a in order], col, float(rc)


def build_cgip(rum):
    """Build and cache the CG-IP geometry on rum.cgip."""
    rum.cgip = CGIPBuild(rum)
    return rum.cgip


def build_cgipheuristic(rum):
    """Build and cache the CG-IP geometry on rum.cgipheuristic."""
    rum.cgipheuristic = CGIPBuild(rum)
    return rum.cgipheuristic


# --------------------------------------------------------------------------- #
#  Restricted master problem (projection onto the current columns)             #
# --------------------------------------------------------------------------- #
def _solve_rmp(V, p, Omega, qp_backend='gurobi', **warm):
    """min ||V lambda - p||^2_Omega  s.t. lambda >= 0, sum lambda = 1.

    Forms the lambda-space Gram (Q = V^T Omega V, h = V^T Omega p) and hands it to
    the shared restricted-master solver (solver.common.solve_qp.solve_rmp), so the same
    QP that cgbf uses can run here under either backend ('gurobi' or 'native').
    warm: native-backend options (A_init, max_outer); only strict solves pass them.
    Returns (lambda, pred = V lambda, residual = p - pred, error)."""
    # Fold the weight Omega and the point p into (Q, h); the QP is oblivious to V.
    OmegaV = V if Omega is None else Omega @ V              # (M, K)
    Q = V.T @ OmegaV                                        # (K, K), PSD
    h = V.T @ (p if Omega is None else np.asarray(Omega @ p).ravel())   # (K,)
    lam_v, _A = solve_rmp(Q, h, backend=qp_backend, **warm)
    pred = V @ lam_v
    residual = p - pred
    err = float(residual @ (residual if Omega is None else Omega @ residual))
    return lam_v, pred, residual, err


_IP_GAP_EPS = 1e-10      # Gurobi's default MIPGapAbs: a MIPGap = 0 re-solve cannot close more


def _ip_price(build, g, inner, repeated, would_stop, t_end=None):
    """One pricing IP with a valid bound. Returns (order, value, rc, solves, column): the
    incumbent ranking, value >= max_r v_r.g (Gurobi's ObjBound), the incumbent's own reduced
    cost, the number of IP solves, and the incumbent's column. The IP is re-solved once with
    MIPGap = 0 only when its own gap blocks a stop: its column is already in the master
    (repeated(order, column); the stall test needs the true maximiser), or would_stop(value)
    holds for the incumbent's value but not for the bound.
    t_end (a time.perf_counter() time): each solve gets the time left as Gurobi's TimeLimit.
    value stays a valid bound at the limit; order and column are None if the IP found no
    ranking by then."""
    prm = build._model.Params
    tl0 = prm.TimeLimit if t_end is not None else None

    def price():
        if t_end is not None:
            prm.TimeLimit = max(t_end - time.perf_counter(), 1e-3)
        return build.ip_price(g)
    try:
        order, bound = price()
        if order is None:
            return None, bound, -np.inf, 1, None
        col = build.column(build._pos(order))
        inc = float(col @ g)
        solves = 1
        if bound - inc > _IP_GAP_EPS and (repeated(order, col)
                                          or (would_stop(inc) and not would_stop(bound))):
            gap0 = prm.MIPGap
            prm.MIPGap = 0.0
            try:
                order2, bound2 = price()
            finally:
                prm.MIPGap = gap0
            solves = 2
            if order2 is not None:
                order, bound = order2, bound2
                col = build.column(build._pos(order))
                inc = float(col @ g)
            else:                                     # the time limit came first: keep the first incumbent
                bound = min(bound, bound2)
    finally:
        if t_end is not None:
            prm.TimeLimit = tl0
    return order, max(bound, inc), inc - inner, solves, col


# --------------------------------------------------------------------------- #
#  Column-generation solve                                                     #
# --------------------------------------------------------------------------- #
def _cg_solve(build, p, Omega, mode, tol, max_iters, seed, verbose,
              abs_tol=None, rel_tol=None, lb_stop=None, ub_stop=None,
              qp_backend='gurobi', n_restart=1, strict=False, time_limit=None):
    """Column generation with cgbf-style certified bounds on the distance.

    Every iteration brackets phi in [lb_best, UB]:
        UB = sqrt(error^2)                       (current fit; always valid)
        LB = sqrt(max(0, error^2 - 2*fw_gap))    (Frank-Wolfe / convexity bound)
        lb_best = running max of LB              (retained across iterations)
    with fw_gap >= max_v <v - pred, g>, the max reduced cost bounded by the IP's
    ObjBound (see _ip_price), so LB stays valid at any MIP gap.
    Termination fires on any of:
        * UB - lb_best <= abs_tol                (cgbf absolute gap)
        * (UB - lb_best)/lb_best <= rel_tol      (cgbf relative gap)
        * lb_best >= lb_stop -> phi provably ABOVE the threshold (decision='above')
        * UB < ub_stop       -> phi provably BELOW the threshold (decision='below')
        * a priced vertex repeats (stall) -> no improving column -> exact optimum
    There is NO reduced-cost convergence stop: with no abs_tol/rel_tol/threshold,
    ip mode runs to the exact optimum via the stall, matching the exact QP
    backends. `tol` is retained ONLY as the heuristic's IP-certification trigger
    (h_gap < tol), never as a convergence test.

    The lb_stop/ub_stop pair is the inference threshold decision -- pass
    lb_stop=ub_stop=tau and 1{tau <= phi} is decided as soon as [lb_best, UB]
    lands on one side of tau. Returned 'distance' is UB.

    In heuristic mode the IP is invoked ONLY when the heuristic stalls
    (h_gap < tol) -- never once per iteration -- so a certified LB exists at those
    points (and every iteration in ip mode). lb_best retains the tightest LB
    across the non-certified heuristic iterations, so the abs_tol/rel_tol/lb_stop
    tests can fire on any iteration (ub_stop needs only UB). Convergence stays
    exact: the stall is the terminator and the bracket [lb_best, UB] only
    tightens.

    strict=False keeps the behaviour of the published runs. strict=True (the RUM
    class) makes a stall honest near distance 0, where a master solved only to a
    tolerance leaves its own columns with a small positive reduced cost, so the
    pricing returns them again:
        * repeats are tested on the column (two rankings can share one); a repeated
          column is never added again and ends the solve with stop_reason 'repeat',
          converged only if the gap tests above hold;
        * the relative test needs lb_best > 0 (no 1e-8 floor);
        * the native master is warm-started from the previous support and may take
          2 K + 200 active-set steps (K columns) instead of 200;
        * the result adds master_kkt, the largest  max_j v_j.g - (V lambda).g  over
          the master's own columns (0 for an exactly solved master).
    time_limit (seconds, None = no limit) is checked before each pricing step and
    handed to the pricing IP as Gurobi's TimeLimit; stop_reason 'time_limit'."""
    t_end = None if time_limit is None else time.perf_counter() + float(time_limit)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(build.n))              # one random start column
    _init = HOOKS.get("init_orders")
    HOOKS["init_orders"] = None                        # consumed once
    if _init:                                           # resume from checkpointed columns
        orders = [list(int(x) for x in o) for o in _init]
        V = np.column_stack([build.column(build._pos(o)) for o in orders])
        seen = {tuple(o) for o in orders}
        order = orders[-1]
    else:
        orders = [list(int(x) for x in order)]
        V = build.column(build._pos(order)).reshape(-1, 1)
        seen = {tuple(order)}
    if strict:
        seen = {_col_key(c) for c in V.T}

        def repeated(o, col):
            return _col_key(col) in seen
    else:
        def repeated(o, col):
            return tuple(o) in seen
    support, master_kkt = None, 0.0
    _hook = HOOKS.get("on_iter")
    hooked = False
    stop_reason = None
    lam = pred = residual = None
    err = float('inf')
    best_score = float('inf')
    LB, UB, lb_best = None, float('inf'), 0.0    # lb_best: running max of the LB
    converged = False
    decision = None
    ip_calls = 0                                 # # exact-IP solves (pricing/cert)
    if verbose:
        print(f"Starting column generation | pricing = {mode} | tol = {tol:.1e}"
              f" | {build.M} obs, {build.n} alternatives | 1 initial column",
              flush=True)
    it = 0
    for it in range(1, max_iters + 1):
        if strict and qp_backend == 'native':
            warm = dict(A_init=support, max_outer=2 * V.shape[1] + 200)
        else:
            warm = {}
        lam, pred, residual, err = _solve_rmp(V, p, Omega, qp_backend=qp_backend, **warm)
        UB = float(np.sqrt(max(0.0, err)))
        if ub_stop is not None and UB < ub_stop:         # 'below': UB alone
            decision = 'below'
            break
        g = residual if Omega is None else np.asarray(Omega @ residual).ravel()
        inner = float(pred @ g)
        if strict:
            support = [int(j) for j in np.flatnonzero(lam > 1e-9)]
            master_kkt = max(master_kkt, float(np.max(V.T @ g)) - inner)
        if t_end is not None and time.perf_counter() >= t_end:
            stop_reason = 'time_limit'
            break

        def would_stop(value):        # the gap tests below, if the pricing maximum were `value`
            lb = max(lb_best, min(UB, float(np.sqrt(max(0.0, err - 2.0 * (value - inner))))))
            return ((lb_stop is not None and lb >= lb_stop)
                    or (abs_tol is not None and UB - lb <= abs_tol)
                    or (rel_tol is not None and (UB - lb <= rel_tol * lb if strict
                                                 else (UB - lb) / max(1e-8, lb) <= rel_tol)))

        # ---- pricing (+ certified fw_gap for LB when an LB-based test needs it) --
        if mode == 'ip':
            order, optim_value, _rc, k, add_col = _ip_price(build, g, inner, repeated, would_stop, t_end)
            ip_calls += k
            best_score = optim_value - inner             # certified fw_gap (bound)
            fw_gap = best_score
            add_order = order
        else:
            if mode == 'rand':                           # random-insertion
                h_order, h_col, h_gap = build.rand_price(g, inner, n_restart,
                                                         seed + it)
            else:                                        # 'best' (best-insertion)
                h_order, h_col, h_gap = build.heuristic_price(g, inner)
            # Certify with exact IP pricing when the heuristic is near-optimal
            # (h_gap < tol) OR when it STALLS by re-proposing a column already in
            # the basis. A heuristic repeat does NOT prove optimality (only exact
            # pricing does), so we follow the IP pricing to certify the gap and,
            # if a genuinely improving column exists, to find it.
            if h_gap < tol or repeated(h_order, h_col):
                if verbose:
                    print("           calling IP (certification)", flush=True)
                ip_order, optim_value, ip_rc, k, ip_col = _ip_price(build, g, inner, repeated,
                                                                    would_stop, t_end)
                ip_calls += k
                fw_gap = optim_value - inner             # certified (bound)
                if ip_rc > h_gap:                        # IP found a better column
                    add_order, add_col, best_score = ip_order, ip_col, ip_rc
                else:
                    add_order, add_col, best_score = h_order, h_col, h_gap
            else:
                fw_gap = None
                add_order, add_col, best_score = h_order, h_col, h_gap

        LB = (min(UB, float(np.sqrt(max(0.0, err - 2.0 * fw_gap))))
              if fw_gap is not None else None)
        if LB is not None:
            lb_best = max(lb_best, LB)            # running max (as in cgbf)

        if verbose:
            if LB is not None:
                print(f"  Iter {it:4d} | UB = {UB:.6f} | LB = {LB:.6f} | "
                      f"gap = {UB - LB:.4e} | cols = {V.shape[1]}", flush=True)
            else:
                print(f"  Iter {it:4d} | UB = {UB:.6f} | "
                      f"best_score = {best_score:.4e} | cols = {V.shape[1]}",
                      flush=True)

        # ---- termination (certified gap on the running-max lb_best) ----
        # NO reduced-cost stop: exact optimum is reached via the stall below.
        if lb_stop is not None and lb_best >= lb_stop:   # dist >= lb_best >= tau
            decision = 'above'
            break
        if abs_tol is not None and (UB - lb_best) <= abs_tol:
            converged = True
            break
        if rel_tol is not None and (UB - lb_best <= rel_tol * lb_best if strict
                                    else (UB - lb_best) / max(1e-8, lb_best) <= rel_tol):
            converged = True
            break
        if t_end is not None and (add_order is None or time.perf_counter() >= t_end):
            stop_reason = 'time_limit'
            break

        if strict:                                       # repeats by column: see the docstring
            key = _col_key(add_col)
            if key in seen:
                stop_reason = 'repeat'
                break
            seen.add(key)
            orders.append(list(int(x) for x in add_order))
            V = np.column_stack([V, add_col])
        else:
            key = tuple(add_order)
            # A repeated priced column proves the optimum ONLY under exact pricing
            # (mode 'ip'): there a stall means no improving column exists. Heuristic
            # modes ('best'/'rand') escalate to IP above on a stall, so they never rely
            # on this terminator (they stop via lb_stop/abs_tol once the IP
            # certification tightens lb_best) -- gating it to exact pricing prevents a
            # heuristic stall from being mis-reported as a converged optimum.
            if mode == 'ip' and key in seen:             # no new vertex -> done
                converged = True
                break
            seen.add(key)
            order = add_order
            orders.append(list(int(x) for x in add_order))
            V = np.column_stack([V, build.column(build._pos(add_order))])
        if _hook is not None:
            _r = _hook(it=it, UB=UB, LB=lb_best, orders=orders, ip_calls=ip_calls, mode=mode)
            if _r:
                stop_reason = str(_r)
                hooked = True
                break

    dist = float(np.sqrt(max(0.0, err)))
    if verbose:
        if decision == 'above':
            tag = f"Stopped: lb_best = {lb_best:.6f} >= lb_stop = {lb_stop} (phi above)"
        elif decision == 'below':
            tag = f"Stopped: UB = {UB:.6f} < ub_stop = {ub_stop} (phi below)"
        elif converged:
            tag = "Convergence achieved"
        elif hooked:
            tag = f"Stopped by the caller's hook: {stop_reason}"
        elif stop_reason == 'time_limit':
            tag = f"Stopped at time_limit = {time_limit:g} s"
        elif stop_reason == 'repeat':
            tag = "Stopped: the priced column is already in the master, gap not closed"
        else:
            tag = f"Stopped at max_iters = {max_iters}"
        print(f"{tag} | iters = {it} | columns = {V.shape[1]} | "
              f"distance = {dist:.6f}", flush=True)

    out = {'distance': dist, 'obj': err, 'p_pred': pred, 'lambda': lam,
           'n_support': int(np.sum(lam > 1e-8)), 'n_columns': V.shape[1],
           'iters': it, 'best_score': float(best_score), 'converged': converged,
           'LB': lb_best, 'UB': UB, 'decision': decision, 'ip_calls': ip_calls,
           'orders': orders, 'stop_reason': stop_reason}
    if strict:
        out['master_kkt'] = master_kkt
    return out


def _col_key(col):
    """A column's identity: its choice vector packed to bits."""
    return np.packbits(np.asarray(col) > 0.5).tobytes()


def solve_cgip(rum, p, vcov=None, mode='ip', tol=1e-6, abs_tol=None,
               rel_tol=None, lb_stop=None, ub_stop=None, max_iters=2000,
               seed=0, verbose=False, qp_backend='gurobi', strict=False,
               time_limit=None, **_ignored):
    """Exact CG-IP projection distance (mode='ip'). Reuses rum.cgip. vcov=Omega
    weights the norm (None => L2). abs_tol/rel_tol add cgbf-style gap stopping;
    lb_stop/ub_stop add the inference threshold early stop (the oracle passes
    lb_stop=ub_stop=tau). qp_backend selects the shared master-QP solver
    ('gurobi' default, or 'native' active-set+Cholesky). strict and time_limit:
    see _cg_solve (both off by default). **_ignored swallows other inference kwargs."""
    build = rum.cgip if mode == 'ip' else rum.cgipheuristic
    p = np.asarray(p, dtype=np.float64)
    Omega = _as_weight(vcov)
    return _cg_solve(build, p, Omega, mode, tol, max_iters, seed, verbose,
                     abs_tol=abs_tol, rel_tol=rel_tol,
                     lb_stop=lb_stop, ub_stop=ub_stop, qp_backend=qp_backend,
                     strict=strict, time_limit=time_limit)


def solve_cgipheuristic_rand(rum, p, vcov=None, tol=1e-6, abs_tol=None,
                             rel_tol=None, lb_stop=None, ub_stop=None,
                             max_iters=2000, seed=0, verbose=False,
                             qp_backend='gurobi', n_restart=1, strict=False,
                             time_limit=None, **_ignored):
    """Random-insertion CG pricing with IP-certified convergence. Reuses
    rum.cgipheuristic. Each pricing call runs `n_restart` random-insertion
    constructions (random start + random alt each step, best position only,
    O(n_restart * n^2 M)) and keeps the best; the exact IP is called only when
    the (weaker) heuristic stalls. Cheaper per call than best-insertion but lower
    column quality, so it may stall -> call the IP -> more often. Same certified
    distance. `seed` makes the run reproducible (per-iteration seed = seed+it).
    strict and time_limit: see _cg_solve (both off by default)."""
    p = np.asarray(p, dtype=np.float64)
    Omega = _as_weight(vcov)
    return _cg_solve(rum.cgipheuristic, p, Omega, 'rand', tol, max_iters,
                     seed, verbose, abs_tol=abs_tol, rel_tol=rel_tol,
                     lb_stop=lb_stop, ub_stop=ub_stop, qp_backend=qp_backend,
                     n_restart=n_restart, strict=strict, time_limit=time_limit)

