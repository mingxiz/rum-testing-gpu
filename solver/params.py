"""Default parameters of every solver, and the one rule that varies with the
instance: the BM-ADMM penalty.

BM-ADMM (K formulation)
-----------------------
    alpha  = 1.95       over-relaxation (monotone gain up to 1.95; 1.99 is worse)
    p      = sqrt(2)    per-bit diagonal preconditioner of the inequality block
    beta   = f(mp)      penalty = dual step; the only parameter that must follow
                        the instance. Iteration counts are U-shaped in beta with
                        a flat basin (about a factor 3 wide), and the empirical
                        optimum grows with the number of observations mp:
                            beta*  ~=  0.033 * mp^0.51
                        (least squares on the per-cell best of a 3-point grid
                        over the 30 instances of the paper, n = 18-20, from
                        pairwise to 10% density; residual factor 1.6). Anchors:
                        mp = 306 (pairwise n=18) -> 0.6, mp = 24k -> 5,
                        mp = 1.05M -> 34, all menus at n=18 (2.4M) -> 58.
    beta="probe"        instead of the rule, race three candidates
                        {beta/3, beta, 3 beta} for PROBE_ITERS iterations and
                        keep the one with the lowest pushed upper bound; the
                        probe time is charged to the solve.
    Fixed for the whole run: the convergence theory covers fixed penalties.

Flow-ADMM (arc flow, prestored Mhat)
------------------------------------
    beta2 = 2, beta3 = 0.05, alpha = 1.8, adaptive penalties on (the
    configuration of the original implementation). Its m_p x m_p double
    precision matrices (Mhat, Schur complement, Cholesky factor) must fit in
    device memory: about 3 * 8 * mp^2 bytes.

CGBFQP-GPU
----------
    parameter-free. rc_tol = 0 disables the reduced-cost stop so the solver can
    only stop on the criterion of the experiment.

CGIP / CGIP-heuristic (Gurobi)
------------------------------
    the defaults of solver/ip_cg/cgip.py (tol = 1e-6, seed = 0); Gurobi Threads
    left to Gurobi (all cores) unless `threads` is given.
"""
import math

# ---- BM-ADMM ---------------------------------------------------------------
BM_ALPHA = 1.95
BM_PRECOND_P = math.sqrt(2.0)
BM_CHECK_EVERY = 50           # pushed-UB check period (race criterion)
BM_BETA_COEF, BM_BETA_EXP = 10 ** -1.48, 0.508
BM_BETA_MIN, BM_BETA_MAX = 0.3, 100.0
PROBE_ITERS = 300


def bm_beta_default(mp):
    """Penalty from the number of observed (menu, alternative) cells."""
    return float(min(BM_BETA_MAX, max(BM_BETA_MIN, BM_BETA_COEF * float(mp) ** BM_BETA_EXP)))


# ---- Flow-ADMM -------------------------------------------------------------
AF_DEFAULTS = dict(beta2=2.0, beta3=0.05, alpha_relax=1.80, adapt_on=True)
AF_CHECK_EVERY = 50


def af_device_bytes(mp):
    """Device high-water mark of Flow-ADMM, measured on the paper runs: about
    50 bytes per mp^2 (Mhat, its symmetrized copy, the Schur matrix, its
    Cholesky factor and the factorization workspace, all double precision)."""
    return 50 * int(mp) ** 2


# ---- CGBFQP-GPU ------------------------------------------------------------
CB_DEFAULTS = dict(rc_tol=0.0, max_iters=4000)


# ---- BM-ADMM device memory ---------------------------------------------------
def bm_device_bytes(n, hybrid=True):
    """Persistent slab buffers: 13 slabs of n*2^(n-1) entries in double
    precision, plus the same in single precision for the hybrid sprint."""
    m2 = n * (1 << (n - 1))
    return 13 * m2 * (8 + (4 if hybrid else 0)) + 4 * (1 << n)
