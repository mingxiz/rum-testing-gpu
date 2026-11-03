"""Flow-ADMM: arc-flow ADMM with a prestored dense Mhat (GPU).

    from solver.flow_admm import solve_flow_admm
    r = solve_flow_admm(n, cd, criterion="race", opt_ref=OPT)

Only the race criterion is implemented (the solver has no dual bound of its
own). wall_s is the iteration loop; build_s is the one-off construction of the
m_p x m_p double-precision Mhat and its Schur/Cholesky factor, reported
separately (the amortized regime in which Mhat is reused across many p-hat).
The run is refused up front ("infeasible") when the factorization cannot fit
in device memory (params.af_device_bytes vs. free memory, with headroom) —
these are the "--" cells of the paper's tables.
"""
import hashlib
import time

import numpy as np

from .. import params
from .solver import solve_arcflow_fast

AF_FIT_FRACTION = 2.0     # allow the WDDM host-spill regime the paper reports (up to ~2x device memory)


def device_fits(mp):
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return params.af_device_bytes(mp) <= AF_FIT_FRACTION * total, params.af_device_bytes(mp), total
    except Exception:
        return False, params.af_device_bytes(mp), 0


def solve_flow_admm(n, cd, criterion="race", opt_ref=None, rel_tol=1e-2, time_limit=900.0,
                    check=params.AF_CHECK_EVERY, verbose=False, force=False, **overrides):
    if criterion != "race":
        raise NotImplementedError("flow_admm has no certificate; use criterion='race'")
    if opt_ref is None:
        raise ValueError("criterion='race' needs opt_ref (a certified optimum)")
    if opt_ref == 0 and overrides.get("abs_tol") is None:     # (UB - opt)/opt is undefined at OPT = 0
        raise ValueError("opt_ref = 0: pass abs_tol (the race stops at UB <= abs_tol)")
    mp = int(np.asarray(cd).shape[0])
    ok, need, total = device_fits(mp)
    if not ok and not force:
        return dict(method="flow_admm", criterion="race", ub=None, lb=None, iters=0, wall_s=None,
                    build_s=None, stop="infeasible", converged=False,
                    note="Mhat/Schur/Cholesky need %.1f GB > %.1fx device memory (%.1f GB)"
                         % (need / 2 ** 30, AF_FIT_FRACTION, total / 2 ** 30))
    opts = dict(params.AF_DEFAULTS)
    opts.update(maxiter=10 ** 9, opt_ref=opt_ref, ub_rel_tol=rel_tol, log_every=check,
                time_limit=time_limit)
    opts.update(overrides)
    t0 = time.perf_counter()
    # cache tag from the menu pattern only (masks and alternatives), so every p_hat on the
    # same design, including the lambda variants of a cell, reuses one Mhat build
    pattern = np.ascontiguousarray(np.asarray(cd)[:, :2].astype(np.int64))
    tag = "pat" + hashlib.sha256(pattern.tobytes()).hexdigest()[:16]
    r = solve_arcflow_fast(n, np.asarray(cd, dtype=np.float64), mhat_tag=tag, opts=opts)
    return dict(method="flow_admm", criterion="race", ub=float(r["fit"]), lb=None, iters=int(r["iters"]),
                wall_s=float(r["loop_s"]), build_s=float(r["mhat_s"]), setup_s=float(r["setup_s"]),
                total_s=time.perf_counter() - t0, stop=r["stop_reason"],
                converged=r["stop_reason"] == "ub_rel", mhat_from_cache=r.get("mhat_from_cache"),
                **{k: opts[k] for k in params.AF_DEFAULTS})
