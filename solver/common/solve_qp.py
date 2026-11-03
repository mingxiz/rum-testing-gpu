"""solver.common.solve_qp: the shared restricted-master QP (RMP) entry point.

Both CG backends -- cgip (dense V, Gurobi-formed Q,h) and cgbf (bitset / weighted
Gram, incremental Q,h) -- solve the SAME restricted master:

    min_lambda  || V lambda - p ||^2_Omega     s.t.  lambda >= 0, sum(lambda) = 1.

Expanded, this is the simplex-constrained QP in lambda-space

    min_lambda  1/2 lambda^T Q lambda - h^T lambda   s.t. lambda >= 0, sum = 1,
        Q = V^T Omega V   (K x K, PSD),   h = V^T Omega p   (K,).

The weight Omega and the observed p enter ONLY through (Q, h); the solver is
oblivious to how they were formed. cgip forms (Q, h) from a dense V and Omega;
cgbf maintains (Q, h) incrementally and never materializes V. Because the argmin
is invariant to the constant p^T Omega p and to the overall factor of 2, the
(Q, h) minimizer equals the || V lambda - p ||^2_Omega minimizer.

solve_rmp dispatches to one of two interchangeable backends, each in its own
module:
    backend='native'  qp_native.solve_qp_native  (active-set + Cholesky, scipy;
                       warm-started on A_init)
    backend='gurobi'  qp_gurobi.solve_qp_gurobi   (Gurobi QP on (Q, h); lazy
                       import, reuses gurobi_env._ENV)
Both return (lam, A_support): the optimal weights and the support index list
(lam > 1e-9), usable as the next iteration's warm-start active set.

Usage
-----
    from ..common.solve_qp import solve_rmp         # solver/ip_cg/cgip.py

    lam, A = solve_rmp(Q, h, backend='native', A_init=A_prev)
"""
from .qp_native import solve_qp_native
from .qp_gurobi import solve_qp_gurobi

_BACKENDS = ('native', 'gurobi')


def solve_rmp(Q, h, backend='native', A_init=None, **kwargs):
    """Solve the restricted master  min 1/2 lam^T Q lam - h^T lam  s.t. lam >= 0,
    sum(lam) = 1  with the chosen backend.

    Parameters
    ----------
    Q : (K, K) PSD Gram matrix  V^T Omega V  (a view is fine; the gurobi backend
        copies to contiguous).
    h : (K,) linear term  V^T Omega p.
    backend : 'native' (active-set + Cholesky) or 'gurobi'.
    A_init : optional warm-start working set (native only; gurobi ignores it).
    kwargs : native tuning (max_outer / tol / reg); ignored by gurobi.

    Returns
    -------
    (lam, A_support) : optimal simplex weights and their support index list
        (lam > 1e-9), reusable as the next iteration's A_init.
    """
    b = str(backend).strip().lower()
    if b == 'native':
        return solve_qp_native(Q, h, A_init=A_init, **kwargs)
    if b == 'gurobi':
        return solve_qp_gurobi(Q, h, **kwargs)
    raise ValueError(f"unknown qp backend {backend!r}; choose from {_BACKENDS}")
