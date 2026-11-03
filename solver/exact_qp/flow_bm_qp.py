"""Exact Gurobi QP backends for the RUM projection: flow_qp and bm_qp.

Port of Code_Gurobi_QP/RUM_QP_Gurobi (numpy/scipy/gurobipy, plus the shared
Gurobi env and weight helper in solver/common). These are functions that operate
on a solver.rum.RUM instance -- they do NOT define a solver class. The built
formulation (geometry-only, shared across bootstrap samples) is cached on the
RUM instance:

    build_arc_flow(rum) -> stores a FlowBuild(A, b, Z, ...) on rum.flow_qp
    build_BM(rum)       -> stores a BMBuild(K, S, W, ...) on rum.bm_qp

    solve_arc_flow_qp(rum, p) -> min ||y - p||^2 s.t. y = Z r, A r = b, r >= 0
                                 (the memory-scalable "aux" flow QP)
    solve_BM_qp(rum, p)       -> min ||W p~ - p||^2 s.t. K p~ >= 0, S p~ = 1, p~>=0
                                 (the Block-Marschak / p-space dual)

The build depends only on (menu, alternative, n); only p changes across
bootstrap draws, so build once (cached on rum.flow_qp / rum.bm_qp) and re-solve.

Geometry: r is a flow on the Boolean-lattice digraph on {0..n-1}, length
m2 = n * 2^(n-1), in item-interleaved compressed-tail order (n blocks of
Nloc = 2^(n-1), one per chosen item). Tractable for moderate n (roughly n<=15);
the flow QP has m2 vars and 2^n equality constraints.
"""
import time

import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

from ..common.gurobi_env import _ENV        # one Gurobi env/license shared across all solves
from ..common.weights import _as_weight     # weight-matrix normaliser (shared with cgip)

# Geometry-only builds cached on the RUM instance (shared across bootstrap p's).
# Each also lazily builds ONE persistent Gurobi model whose constraints and
# quadratic objective are fixed; across bootstrap draws only p changes, and p
# enters solely through the linear objective term (updated via the `Obj`
# attribute), so the model is built once and re-solved -- not rebuilt per draw.
class FlowBuild:
    """Arc-flow geometry (A, b, Z) + a persistent Gurobi model for
    min ||y - p||^2_Omega s.t. y = Z r, A r = b, r >= 0. The model is (re)built
    only when Omega changes (compared by object identity); inference fixes
    Omega, so it is built once per run."""
    __slots__ = ('A', 'b', 'Z', 'm2', 'Nnode', 'Nloc',
                 '_model', '_r', '_y', '_omega')

    def __init__(self, A, b, Z, m2, Nnode, Nloc):
        self.A, self.b, self.Z = A, b, Z
        self.m2, self.Nnode, self.Nloc = m2, Nnode, Nloc
        self._model = None
        self._r = self._y = self._omega = None

    def model(self, M, omega):
        """(model, r, y), reusing the cached model unless Omega changed. The
        quadratic objective y'Omega y (Omega=None => y'y) is set here once; the
        linear term -2 (Omega p) . y is set per solve by the caller."""
        if self._model is None or self._omega is not omega:
            m = gp.Model(env=_ENV)
            r = m.addMVar(self.m2, lb=0.0, name="r")       # r >= 0
            y = m.addMVar(M, lb=0.0, name="y")             # y = predicted p
            m.addConstr(self.A @ r == self.b, name="flow")
            m.addConstr(y - self.Z @ r == 0.0, name="pred")
            quad = (y @ y) if omega is None else (y @ omega @ y)
            m.setObjective(quad, GRB.MINIMIZE)             # y'Omega y (fixed)
            self._model, self._r, self._y, self._omega = m, r, y, omega
        return self._model, self._r, self._y


class BMBuild:
    """Block-Marschak geometry (K, S, W) + a persistent Gurobi model for
    min ||W p~ - p||^2_Omega s.t. K p~ >= 0, S p~ = 1, p~ >= 0. Same
    build-once / re-solve contract as FlowBuild."""
    __slots__ = ('K', 'S', 'W', 'm2', 'num_menus', '_model', '_pv', '_omega')

    def __init__(self, K, S, W, m2, num_menus):
        self.K, self.S, self.W = K, S, W
        self.m2, self.num_menus = m2, num_menus
        self._model = None
        self._pv = self._omega = None

    def model(self, omega):
        """(model, p~), reusing the cached model unless Omega changed. The
        quadratic objective p~'W'Omega W p~ is set here once; the linear term
        -2 (W' Omega p) . p~ is set per solve by the caller."""
        if self._model is None or self._omega is not omega:
            m = gp.Model(env=_ENV)
            pv = m.addMVar(self.m2, lb=0.0, name="p")      # p~ >= 0
            m.addConstr(self.K @ pv >= 0, name="bm")       # K p~ >= 0
            m.addConstr(self.S @ pv == 1.0, name="simplex")  # S p~ = 1
            Wp = self.W @ pv
            quad = (Wp @ Wp) if omega is None else (Wp @ omega @ Wp)
            m.setObjective(quad, GRB.MINIMIZE)             # p~'W'Omega W p~
            self._model, self._pv, self._omega = m, pv, omega
        return self._model, self._pv


# --------------------------------------------------------------------------- #
#  Index helper                                                                #
# --------------------------------------------------------------------------- #
def _compress_tail(S, xb):
    """Compress full n-bit tail mask S (bit xb clear) to its (n-1)-bit
    block-local index by deleting bit position xb and packing."""
    low = S & ((1 << xb) - 1)
    high = S >> (xb + 1)
    return low | (high << xb)


# --------------------------------------------------------------------------- #
#  Arc-flow formulation:  A (node-arc incidence), b (divergence), Z (zeta)     #
# --------------------------------------------------------------------------- #
def _build_A_b(n, Nloc, m2, Nnode):
    """Closed-form node-arc incidence A (Nnode x m2) and divergence RHS b.
    Arc (x, S) goes tail S -> head S u {x}: -1 at node S, +1 at head."""
    rest = np.arange(Nloc, dtype=np.int64)             # compressed tail index t
    rows = np.empty(2 * m2, dtype=np.int64)
    cols = np.empty(2 * m2, dtype=np.int64)
    vals = np.empty(2 * m2, dtype=np.float64)

    ptr = 0
    for xb in range(n):                                # 0-based item
        low = rest & ((1 << xb) - 1)
        upper = (rest >> xb) << (xb + 1)
        S = low | upper                                # tail node (bit xb clear)
        head = S | (1 << xb)                           # head = S u {x}
        arc_cols = xb * Nloc + rest
        k = Nloc
        rows[ptr:ptr + k] = S;    cols[ptr:ptr + k] = arc_cols; vals[ptr:ptr + k] = -1.0; ptr += k
        rows[ptr:ptr + k] = head; cols[ptr:ptr + k] = arc_cols; vals[ptr:ptr + k] = +1.0; ptr += k

    A = sp.csr_matrix((vals, (rows, cols)), shape=(Nnode, m2))
    b = np.zeros(Nnode, dtype=np.float64)
    b[0] = -1.0
    b[Nnode - 1] = +1.0
    return A, b


def _build_Z(n, Nloc, m2, M, menu_masks, choice_alt):
    """Observed-coordinate zeta operator Z (M x m2): row i sums the flows over
    every superset of observation i's compressed tail in item x_i's block."""
    full_mask = Nloc - 1
    rows, cols = [], []
    for i in range(M):
        D = int(menu_masks[i])
        xb = int(choice_alt[i])
        S = D & ~(1 << xb)                             # tail = D \ {x}
        t0 = _compress_tail(S, xb)
        free = full_mask & ~t0                         # bits we may add
        base = xb * Nloc
        sub = free
        while True:                                    # enumerate submasks of free
            rows.append(i)
            cols.append(base + (t0 | sub))
            if sub == 0:
                break
            sub = (sub - 1) & free
    vals = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((vals, (np.asarray(rows), np.asarray(cols))),
                         shape=(M, m2))


def build_arc_flow(rum):
    """Build the arc-flow formulation (A, b, Z) for rum's geometry and cache it
    on rum.flow_qp. Depends only on (menu, alternative, num_alternatives)."""
    n = rum.num_alternatives
    Nnode = 1 << n
    Nloc = 1 << (n - 1)
    m2 = n * Nloc
    menu_masks = rum.menu.astype(np.int64)
    choice_alt = rum.alternative.astype(np.int64)
    A, b = _build_A_b(n, Nloc, m2, Nnode)
    Z = _build_Z(n, Nloc, m2, rum.M, menu_masks, choice_alt)
    rum.flow_qp = FlowBuild(A=A, b=b, Z=Z, m2=m2, Nnode=Nnode, Nloc=Nloc)
    return rum.flow_qp


# --------------------------------------------------------------------------- #
#  Block-Marschak formulation:  K (Mobius), S (simplex), W (selector)          #
# --------------------------------------------------------------------------- #
def _build_K(n):
    """Block-diagonal superset-Mobius (inverse-zeta) transform: n identical
    2^(n-1) x 2^(n-1) blocks, each the (n-1)-fold Kronecker power of [[1,-1],[0,1]].
    K p >= 0 <=> RUM-rational."""
    m = n - 1
    B = sp.csr_matrix(np.array([[1.0, -1.0], [0.0, 1.0]]))
    block = sp.identity(1, format="csr", dtype=np.float64)
    for _ in range(m):
        block = sp.kron(block, B, format="csr")
    return sp.kron(sp.identity(n, format="csr", dtype=np.float64), block,
                   format="csr")


def _build_simplex(n, Nloc, Nnode):
    """Per-menu simplex selector S (num_menus x m2): row for menu D sums
    p(x|D) over x in D."""
    num_menus = Nnode - 1
    m2 = n * Nloc
    rows, cols = [], []
    for D in range(1, Nnode):
        row = D - 1
        mm = D
        while mm:                                      # iterate set bits (items x)
            xb = (mm & -mm).bit_length() - 1
            S = D & ~(1 << xb)
            t0 = _compress_tail(S, xb)
            rows.append(row)
            cols.append(xb * Nloc + t0)
            mm &= mm - 1
    vals = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((vals, (np.asarray(rows), np.asarray(cols))),
                         shape=(num_menus, m2))


def _build_W(n, Nloc, m2, M, menu_masks, choice_alt):
    """Observed-coordinate selector W (M x m2): row i picks coordinate
    (D_i, x_i), so (W p)_i = p(x_i | D_i)."""
    cols = np.empty(M, dtype=np.int64)
    for i in range(M):
        D = int(menu_masks[i])
        xb = int(choice_alt[i])
        S = D & ~(1 << xb)
        cols[i] = xb * Nloc + _compress_tail(S, xb)
    rows = np.arange(M, dtype=np.int64)
    vals = np.ones(M, dtype=np.float64)
    return sp.csr_matrix((vals, (rows, cols)), shape=(M, m2))


def build_BM(rum):
    """Build the Block-Marschak formulation (K, S, W) for rum's geometry and
    cache it on rum.bm_qp. Depends only on (menu, alternative, num_alternatives)."""
    n = rum.num_alternatives
    Nnode = 1 << n
    Nloc = 1 << (n - 1)
    m2 = n * Nloc
    menu_masks = rum.menu.astype(np.int64)
    choice_alt = rum.alternative.astype(np.int64)
    K = _build_K(n)
    S = _build_simplex(n, Nloc, Nnode)
    W = _build_W(n, Nloc, m2, rum.M, menu_masks, choice_alt)
    rum.bm_qp = BMBuild(K=K, S=S, W=W, m2=m2, num_menus=Nnode - 1)
    return rum.bm_qp


# --------------------------------------------------------------------------- #
#  Solvers (build once, re-solve per p)                                        #
# --------------------------------------------------------------------------- #
def _apply_params(m, verbose, method, time_limit, mem_limit):
    if verbose:
        m.setParam("OutputFlag", 1)
    if method != -1:
        m.setParam("Method", method)
    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)
    if mem_limit is not None:
        m.setParam("MemLimit", mem_limit)


def _weighted_sq(diff, Omega):
    """||diff||^2  (Omega is None)  or  diff' Omega diff  (weighted).
    Omega @ diff works for both dense and sparse Omega."""
    diff = np.asarray(diff, dtype=np.float64)
    if Omega is None:
        return float(diff @ diff)
    return float(diff @ (Omega @ diff))


def _no_solution(m, solve_time, extra):
    out = {'distance': None, 'obj': None, 'p_pred': None,
           'status': int(m.Status), 'solve_time': solve_time,
           'n_support': 0, 'no_solution': True}
    out.update(extra)
    return out


def solve_arc_flow_qp(rum, p, vcov=None, time_limit=None, mem_limit=None,
                      verbose=False, method=-1, **_ignored):
    """Solve the aux flow QP  min ||y - p||^2_Omega s.t. y = Z r, A r = b, r >= 0,
    reusing the cached rum.flow_qp build AND its persistent Gurobi model: only
    the linear objective term changes with p, so the model (vars, constraints,
    quadratic objective) is built once and re-solved. vcov=Omega is the (M, M)
    weight matrix (None => unweighted L2). **_ignored swallows inference-only
    kwargs (lb_stop/ub_stop/...) -- the QP has no early-stop."""
    fb = rum.flow_qp
    p = np.asarray(p, dtype=np.float64)
    Omega = _as_weight(vcov)
    m, r, y = fb.model(rum.M, Omega)                   # persistent model
    _apply_params(m, verbose, method, time_limit, mem_limit)

    # objective = y'Omega y - 2 (Omega p) . y + const == ||y - p||^2_Omega;
    # the quadratic is fixed on the model, so only reset y's linear coeffs.
    y.Obj = -2.0 * (p if Omega is None else np.asarray(Omega @ p).ravel())

    t0 = time.perf_counter()
    m.optimize()
    solve_time = time.perf_counter() - t0
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"flow_qp not solved (status {m.Status})")
    if m.SolCount == 0:
        return _no_solution(m, solve_time, {'r': None})

    r_val = r.X
    p_pred = y.X
    obj = _weighted_sq(p_pred - p, Omega)
    return {'distance': float(np.sqrt(max(0.0, obj))), 'obj': obj,
            'p_pred': p_pred, 'r': r_val, 'status': int(m.Status),
            'solve_time': solve_time, 'n_support': int(np.sum(r_val > 1e-8))}


def solve_BM_qp(rum, p, vcov=None, time_limit=None, mem_limit=None,
                verbose=False, method=-1, **_ignored):
    """Solve the Block-Marschak QP  min ||W p~ - p||^2_Omega s.t. K p~ >= 0,
    S p~ = 1, p~ >= 0, reusing the cached rum.bm_qp build AND its persistent
    Gurobi model (built once; only the linear objective term changes with p).
    vcov=Omega is the (M, M) weight matrix (None => unweighted L2)."""
    bm = rum.bm_qp
    p = np.asarray(p, dtype=np.float64)
    Omega = _as_weight(vcov)
    m, pv = bm.model(Omega)                            # persistent model
    _apply_params(m, verbose, method, time_limit, mem_limit)

    # objective = p~'W'Omega W p~ - 2 (W' Omega p) . p~ + const; quadratic fixed,
    # so only reset p~'s linear coeffs. (W' Omega p) = W' (Omega p).
    Omega_p = p if Omega is None else np.asarray(Omega @ p).ravel()
    pv.Obj = -2.0 * np.asarray(bm.W.T @ Omega_p).ravel()

    t0 = time.perf_counter()
    m.optimize()
    solve_time = time.perf_counter() - t0
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"bm_qp not solved (status {m.Status})")
    if m.SolCount == 0:
        return _no_solution(m, solve_time, {'p': None})

    p_val = pv.X
    p_pred = bm.W @ p_val
    obj = _weighted_sq(p_pred - p, Omega)
    return {'distance': float(np.sqrt(max(0.0, obj))), 'obj': obj,
            'p_pred': p_pred, 'p': p_val, 'status': int(m.Status),
            'solve_time': solve_time, 'n_support': int(np.sum(p_val > 1e-8))}
