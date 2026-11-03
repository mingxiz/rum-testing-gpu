"""solver.common.qp_gurobi: the Gurobi restricted-master QP backend.

Solves the same (Q, h) simplex QP as qp_native

    min_lambda  1/2 lambda^T Q lambda - h^T lambda   s.t. lambda >= 0, sum = 1,

but with Gurobi. gurobipy and gurobi_env._ENV are imported LAZILY inside the
solve, so importing this module (and solve_qp) needs no Gurobi unless this
backend is actually selected. project_simplex is borrowed from qp_native to clean
tiny solver drift on the returned weights.

This is one of the two interchangeable RMP backends; the dispatcher that picks
between this and qp_native is solver.common.solve_qp.solve_rmp.

Guard against the singular Gram matrix (the same guard as solve_qp_gurobi_guarded in the
cluster scripts of the paper's IP-CG runs). Once the master holds more
columns than the rank of Q = V^T Omega V (about n(n-1)/2 + 1 on pairwise menus),
Q is singular, and with Gurobi's default NonConvex = -1 rounding can make it
look indefinite: Gurobi then switches to a global non-convex solve that does not
finish and cannot be interrupted (observed: 24 h and 112 GB on one pairwise cell).
So the objective uses Q + ridge * I with ridge = 1e-8 * max(diag Q), NonConvex = 0
(a non-PSD Q is an error, never a non-convex solve) and PSDTol = 1e-3. The
returned weights are still exact simplex weights; the ridge changes the master
objective by at most ridge * ||lam||^2 / 2 <= ridge / 2, and the callers evaluate
the true objective at the returned weights.
"""
import numpy as np

from .qp_native import project_simplex

RIDGE_REL = 1e-8          # ridge = RIDGE_REL * max(diag Q)


def solve_qp_gurobi(Q, h, **_ignored):
    """Gurobi QP for  min 1/2 lam^T Q lam - h^T lam  s.t. lam >= 0, sum(lam) = 1,
    solved on Q + ridge*I with NonConvex = 0 and PSDTol = 1e-3 (module docstring).
    Lazy-imports gurobipy and reuses gurobi_env._ENV so importing solve_qp needs no
    Gurobi unless this backend is actually selected. **_ignored swallows the
    native-only tuning kwargs (max_outer/tol/reg). Returns (lam, A_support)."""
    import gurobipy as gp
    from gurobipy import GRB
    from .gurobi_env import _ENV

    Q = np.ascontiguousarray(Q, dtype=np.float64)   # slices/views -> contiguous
    h = np.ascontiguousarray(h, dtype=np.float64)
    K = h.shape[0]
    m = gp.Model(env=_ENV)
    m.setParam("NonConvex", 0)
    m.setParam("PSDTol", 1e-3)
    lam = m.addMVar(K, lb=0.0, name="lam")
    m.addConstr(lam.sum() == 1.0, name="simplex")
    Qr = Q + RIDGE_REL * max(float(np.max(np.diag(Q))), 1e-300) * np.eye(K)
    m.setObjective(0.5 * (lam @ Qr @ lam) - (h @ lam), GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"master QP not optimal (status {m.Status})")
    lam_v = project_simplex(np.asarray(lam.X, dtype=np.float64))   # clean drift
    A_support = [int(i) for i in range(K) if lam_v[i] > 1e-9]
    return lam_v, A_support
