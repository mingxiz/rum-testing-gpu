"""BM-ADMM: matrix-free ADMM on the Block-Marschak (K) formulation, GPU.

    from solver.bm_admm import solve_bm_admm
    r = solve_bm_admm(n, cd, criterion="race", opt_ref=OPT)          # 1% of a known optimum

The paper's race: a hybrid solver (single-precision sprint, double-precision finish,
every stop re-verified in double precision); the pushed-feasible upper bound is
evaluated every `check` iterations and the run stops when (UB - opt_ref)/opt_ref <= rel_tol.
A certified distance comes from the RUM class (solver/rum_solver.py, method="bm_admm"),
which runs solve_kadmm_v2 once with the upper- and lower-bound hooks.
Iterations are ADMM iterations; wall_s is the solve only (the one-off butterfly
plan calibration and kernel compilation happen in warmup()).
"""
import time

import numpy as np

from .. import params
from ..common.transforms import build_obs_idx, push_to_ub
from ..common.instances import warmup_instance
from .solver import solve_kadmm_v2, solve_kadmm_hybrid


def _obs(n, cd):
    cd = np.asarray(cd, dtype=np.float64)
    om = cd[:, 0].astype(np.int64)
    ox = cd[:, 1].astype(np.int64)
    ph = cd[:, 2].astype(np.float64)
    return build_obs_idx(n, om, ox), ph


def warmup(n=10):
    """Compile kernels and calibrate the butterfly plan on a tiny instance."""
    solve_bm_admm(n, warmup_instance(n), criterion="race", opt_ref=1.0, rel_tol=1e9, max_iter=100)


def _resolve_beta(n, cd, beta, obs_idx, ph, alpha, p, verbose):
    mp = len(ph)
    b0 = params.bm_beta_default(mp)
    if beta is None:
        return b0, 0.0
    if beta != "probe":
        return float(beta), 0.0
    t0 = time.perf_counter()
    best = None
    for b in (b0 / 3, b0, 3 * b0):
        r = solve_kadmm_v2(n, cd, beta1=b, beta2=b, alpha_relax=alpha, precond_p=p,
                           max_iter=params.PROBE_ITERS, check=params.PROBE_ITERS,
                           dtype="float32", use_graph=True)
        ub = push_to_ub(n, r["state"]["x2"], obs_idx, ph)[0]
        if verbose:
            print("  probe beta=%.3g -> pushed UB %.6f after %d its" % (b, ub, params.PROBE_ITERS))
        if best is None or ub < best[0]:
            best = (ub, b)
    return best[1], time.perf_counter() - t0


def solve_bm_admm(n, cd, criterion="race", opt_ref=None, rel_tol=None, beta=None,
                  alpha=params.BM_ALPHA, precond_p=params.BM_PRECOND_P, check=params.BM_CHECK_EVERY,
                  max_iter=10 ** 7, time_limit=None, mode="hybrid", verbose=False, abs_tol=None,
                  dtype="float64", ub_arm_immediately=False):
    """The race to a known optimum opt_ref (criterion="race" is the only criterion here; for a
    certified distance use solver.rum_solver.RUM(...).calculate_distance(method="bm_admm")).

    ub_arm_immediately: the push that produces the feasible upper bound is normally armed
    once the raw objective approaches the target from below (it costs a few ms per check).
    When the target is an absolute floor far above the optimum (lambda = 1 cells: OPT = 0,
    target 1e-3) the raw objective starts near 0 and never reaches the arming threshold, so
    the push would never run and the race would hit its cap with a feasible point already
    far below the target; the runners set this flag exactly when the floor binds, and
    nothing changes for the paper's lambda = 0 cells.
    dtype applies to the non-hybrid race path only: "float64" (default) or "float32".
    float32 halves the slab (14 arrays of n*2^(n-1) elements), which is what makes the
    pairwise race at n = 25 fit a 32 GB card at all; the pushed UB it reports carries f32
    rounding, so use it for races (UB only) and never for a certificate."""
    if criterion != "race":
        raise ValueError("solve_bm_admm runs the race only; for a certified distance use "
                         "solver.rum_solver.RUM(...).calculate_distance(method='bm_admm')")
    obs_idx, ph = _obs(n, cd)
    if rel_tol is None:
        rel_tol = 1e-2
    b, probe_s = _resolve_beta(n, cd, beta, obs_idx, ph, alpha, precond_p, verbose)

    def push(x2):
        return push_to_ub(n, x2, obs_idx, ph)[0]

    t0 = time.perf_counter()
    if opt_ref is None:
        raise ValueError("criterion='race' needs opt_ref (a certified optimum)")
    if opt_ref == 0 and abs_tol is None:      # (UB - opt)/opt is undefined at OPT = 0
        raise ValueError("opt_ref = 0: pass abs_tol (the race stops at UB <= abs_tol)")
    if mode == "hybrid":
        r = solve_kadmm_hybrid(n, cd, beta1=b, beta2=b, alpha_relax=alpha, precond_p=precond_p,
                               check=check, ub_target=opt_ref, ub_eps=rel_tol,
                               time_limit=time_limit, push_fn=push, verbose=verbose,
                               ub_arm_immediately=ub_arm_immediately, ub_abs=abs_tol)
    else:
        r = solve_kadmm_v2(n, cd, beta1=b, beta2=b, alpha_relax=alpha, precond_p=precond_p,
                           max_iter=max_iter, check=check, ub_target=opt_ref, ub_eps=rel_tol,
                           time_limit=time_limit, push_fn=push, dtype=dtype,
                           use_graph=True, verbose=verbose, ub_arm_immediately=ub_arm_immediately,
                           ub_abs=abs_tol)
    ub = float(r["ub"]) if np.isfinite(r["ub"]) else push(r["x2"])
    return dict(method="bm_admm", criterion="race", ub=ub, lb=None, iters=int(r["iters"]),
                wall_s=time.perf_counter() - t0, build_s=probe_s, stop=r["stop_reason"],
                beta=b, alpha=alpha, precond_p=precond_p, mode=mode, dtype=dtype,
                iters32=r.get("iters32"), iters64=r.get("iters64"), converged=r["stop_reason"] == "ub_target",
                trace=r.get("trace"))
