"""solver.common.qp_native: the native (scipy) restricted-master QP backend.

Active-set + Cholesky for the simplex-constrained QP

    min_lambda  1/2 lambda^T Q lambda - h^T lambda   s.t. lambda >= 0, sum = 1,

plus project_simplex, the Euclidean simplex projection (also reused by the gurobi
backend to clean solver drift). Pure numpy/scipy -- no Gurobi, no GPU.

This is one of the two interchangeable RMP backends; the dispatcher that picks
between this and qp_gurobi is solver.common.solve_qp.solve_rmp.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


# --------------------------------------------------------------------------- #
#  Simplex projection (Wang-Carreira-Perpinan / Condat, O(K log K))            #
# --------------------------------------------------------------------------- #
def project_simplex(y):
    """Euclidean projection of y onto {x: x>=0, sum(x)=1}. O(K log K) via sort:
    x_i = max(y_i - tau, 0) with tau the unique scalar making sum(x) = 1."""
    n = y.size
    u = np.sort(y)[::-1]                       # descending sort
    css = np.cumsum(u) - 1.0                    # u_1 + ... + u_k - 1
    cond = u - css / np.arange(1, n + 1) > 0
    if not cond.any():                          # (cannot happen for a sum-1 target)
        return np.ones(n) / n
    rho = int(np.where(cond)[0][-1] + 1)
    tau = css[rho - 1] / rho
    return np.maximum(y - tau, 0.0)


# --------------------------------------------------------------------------- #
#  Native backend: active-set + Cholesky on (Q, h)                             #
# --------------------------------------------------------------------------- #
def solve_qp_native(Q, h, A_init=None, max_outer=200, tol=1e-10, reg=1e-12):
    """Active-set + Cholesky for  min 1/2 lam^T Q lam - h^T lam  s.t. lam >= 0,
    sum(lam) = 1.

    Equality-constrained subproblem on a working set A solved analytically:
        lam_A = Q_AA^{-1} (h_A - mu 1_A),
        mu    = (1_A^T Q_AA^{-1} h_A - 1) / (1_A^T Q_AA^{-1} 1_A).
    Outer loop: drop the most-negative lam_A, else add the most-violated inactive
    column, else stop (KKT). A_init warm-starts the working set. K_active is small
    so the O(K_act^3) refactor per outer iter is sub-ms. Returns (lam, A_support).
    """
    K = len(h)
    if A_init is not None and len(A_init) > 0:
        A = [int(i) for i in A_init if 0 <= i < K]
        if not A:
            A = [int(np.argmax(h))]
    else:
        A = [int(np.argmax(h))]

    lam_A = None
    A_arr = None
    for _ in range(max_outer):
        A_arr = np.asarray(A, dtype=np.int64)
        k = len(A)
        ones_A = np.ones(k)
        Q_AA = Q[np.ix_(A_arr, A_arr)] + reg * np.eye(k)
        h_A = h[A_arr]
        try:
            cf = cho_factor(Q_AA, lower=True, check_finite=False)
            rhs = np.column_stack([h_A, ones_A])
            sol = cho_solve(cf, rhs, check_finite=False)
            Q_inv_h = sol[:, 0]
            Q_inv_e = sol[:, 1]
        except np.linalg.LinAlgError:
            Q_inv_h, *_ = np.linalg.lstsq(Q_AA, h_A, rcond=None)
            Q_inv_e, *_ = np.linalg.lstsq(Q_AA, ones_A, rcond=None)

        eQe = float(ones_A @ Q_inv_e)
        eQh = float(ones_A @ Q_inv_h)
        mu = (eQh - 1.0) / eQe if abs(eQe) > 1e-30 else 0.0
        lam_A = Q_inv_h - mu * Q_inv_e

        # Feasibility: drop most-negative column
        if k > 0 and float(lam_A.min()) < -tol:
            del A[int(np.argmin(lam_A))]
            continue

        # Optimality: add most-negative-gradient inactive column
        if K > k:
            mask = np.ones(K, dtype=bool)
            mask[A_arr] = False
            inactive = np.where(mask)[0]
            grad = Q[np.ix_(inactive, A_arr)] @ lam_A + mu - h[inactive]
            i_min = int(np.argmin(grad))
            if grad[i_min] < -tol:
                A.append(int(inactive[i_min]))
                continue
        break

    lam_full = np.zeros(K)
    lam_full[A_arr] = lam_A
    lam_full = project_simplex(lam_full)
    A_support = [int(i) for i in A if lam_full[i] > 1e-9]
    return lam_full, A_support
