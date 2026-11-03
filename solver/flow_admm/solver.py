r"""Flow-ADMM: ADMM on the arc-flow formulation with a prestored dense Mhat (GPU).

The per-iteration transforms (A / A', the Woodbury solve with H0, the zeta operators C / C' and
the feasibility rounding) are the float32 kernels of common/arcflow_kernels.py. The m_p x m_p
matrix Mhat = C H0^{-1} C' and the Cholesky factor of its Schur complement stay in float64 (in
float32 the solve diverges). Mhat depends only on the menu geometry, so it is cached on disk
(data/cache_mhat, keyed by the caller's tag) and its build is reported separately (mhat_s).

Returns loop_s (the solve loop), setup_s (Mhat load or build + Schur Cholesky), iters,
per_iter_ms, fit (the pushed-feasible upper bound) and stop_reason.

Certified bracket (the RUM class; every key unset on the paper path, which is then unchanged):
    opts push_fn(r), lb_fn(r)       an upper and a lower bound on the distance from the arc flow r,
                                    called at every check (every log_every iterations)
    opts gap_abs_tol, gap_rel_tol   stop once ub - lb <= gap_abs_tol or (ub - lb)/lb <= gap_rel_tol
                                    (best ub and lb so far); stop_reason 'gap_abs' / 'gap_rel'
    opts progress_fn()              called every 256 columns of the Mhat build (time limits)
    opts keep_mhat                  also return mhat (host, float64) and gamma, for reuse as mhat=
"""
import os
import time

import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpla

from ..common import arcflow_kernels as AF
from ..common.transforms import build_obs_idx

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cache_mhat')

_F32 = cp.float32
MHAT_CACHE_MAX_BYTES = 16 * 2 ** 30      # cache Mhat on disk up to 16 GiB (dense float64 m_p x m_p)


def _cache_path(n, gamma, tag):
    d = CACHE_DIR
    return os.path.join(d, f"Mhat_py_n{n}_gamma_{gamma:.12e}_{tag}.npy")


def solve_arcflow_fast(n, choice_data, mhat=None, mhat_tag=None, opts=None):
    if opts is None:
        opts = {}
    Nnode = 1 << n
    Nloc = 1 << (n - 1)
    m2 = n * Nloc
    beta2 = opts.get("beta2", 2.0)
    beta3 = opts.get("beta3", 0.05)
    alpha = opts.get("alpha_relax", 1.80)
    obj_scale = opts.get("obj_scale", 2.0 * 2 ** (-(n - 4)))
    maxiter = opts.get("maxiter", 4000)
    log_every = opts.get("log_every", 25)
    adapt_on = opts.get("adapt_on", True)
    opt_ref = opts.get("opt_ref", None)
    ub_rel_tol = opts.get("ub_rel_tol", 0.0)
    ub_abs_tol = opts.get("abs_tol", None)       # race target when opt_ref == 0 (OPT = 0)
    time_limit = opts.get("time_limit", 600)
    push_fn, lb_fn = opts.get("push_fn"), opts.get("lb_fn")          # certified bracket: see above
    gap_abs_tol, gap_rel_tol = opts.get("gap_abs_tol"), opts.get("gap_rel_tol")
    progress_fn = opts.get("progress_fn")

    wA = float(1.0 / np.sqrt(max(1, n)))            # python float (avoid f64 upcast)
    gamma = float((beta2 * wA ** 2) / beta3)

    phat = np.ascontiguousarray(choice_data[:, 2]).astype(np.float32)
    obs_mask = choice_data[:, 0].astype(np.int64)
    obs_x0 = choice_data[:, 1].astype(np.int64)
    obs_idx = cp.asarray(build_obs_idx(n, obs_mask, obs_x0))
    mp = len(phat)
    g_phat = cp.asarray(phat)

    t_setup = time.perf_counter()
    layers, bounds = AF.build_layers(n)
    g_b = cp.zeros(Nnode, dtype=_F32)
    g_b[0] = -1.0
    g_b[Nnode - 1] = 1.0
    pc = cp.zeros(Nnode, dtype=_F32)
    ar = cp.arange(Nnode, dtype=cp.int64)
    for bit in range(n):
        pc += ((ar >> bit) & 1).astype(_F32)
    g_denom = (1.0 + gamma * 2.0 * pc).astype(_F32)

    def C_mul(x):
        return AF.zeta_superset_blk(x, n, Nloc)[obs_idx]

    def Ct_mul(v):
        z = cp.zeros(m2, dtype=_F32)
        z[obs_idx] = v
        return AF.zeta_subset_blk(z, n, Nloc)

    def H0inv_1(v):                                     # beta3=1 (Mhat normalization)
        return AF.H0inv_woodbury(n, v, 1.0, gamma, g_denom, Nnode, Nloc)

    # ---- Mhat in float64 (the Schur path must be f64: float32 Cholesky/solve of
    # the mp x mp system DIVERGES the ADMM). Transforms stay float32 (the fast
    # kernels = the per-iter bottleneck). Mhat built once with fast f32 transforms
    # then promoted, OR loaded from the float64 cache. Build EXCLUDED from loop_s.
    t_mhat = time.perf_counter()
    # Mhat depends only on (n, menus, alternatives), never on p_hat, so it is cached on disk
    # under a tag the caller derives from the menu pattern; the lambda variants of a cell and
    # every bootstrap draw share one build. Its build/load time is reported separately
    # (mhat_s) and excluded from loop_s, the paper convention for Flow-ADMM.
    cache_file = _cache_path(n, gamma, mhat_tag) if (mhat is None and mhat_tag is not None) else None
    mhat_from_cache = False
    if cache_file is not None and os.path.exists(cache_file):
        mhat = np.load(cache_file, mmap_mode="r")
        mhat_from_cache = True
    if mhat is not None:
        g_Mhat = cp.asarray(mhat, dtype=cp.float64)
    else:
        g_Mhat = cp.empty((mp, mp), dtype=cp.float64)
        e = cp.zeros(mp, dtype=_F32)
        prev = 0
        for j in range(mp):
            if progress_fn is not None and j % 256 == 0:
                progress_fn()
            e[prev] = 0.0
            e[j] = 1.0
            prev = j
            g_Mhat[:, j] = C_mul(H0inv_1(Ct_mul(e))).astype(cp.float64)
    g_Mhat = (g_Mhat + g_Mhat.T) * 0.5
    cp.cuda.Stream.null.synchronize()
    mhat_s = time.perf_counter() - t_mhat
    if cache_file is not None and not mhat_from_cache and mp * mp * 8 <= MHAT_CACHE_MAX_BYTES:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, cp.asnumpy(g_Mhat))

    def chol_factor(b3):
        S = cp.eye(mp, dtype=cp.float64) + (2.0 * obj_scale / b3) * g_Mhat
        S = (S + S.T) * 0.5 + 1e-10 * cp.eye(mp, dtype=cp.float64)
        return cp.linalg.cholesky(S)

    g_L = chol_factor(beta3)
    g_WtP2 = _F32(2.0 * obj_scale) * Ct_mul(g_phat)
    cp.cuda.Stream.null.synchronize()
    setup_s = time.perf_counter() - t_setup

    def chol_solve(L, rhs_f32):                          # f64 triangular solve
        rhs = rhs_f32.astype(cp.float64)
        y = cpla.solve_triangular(L, rhs, lower=True)
        y = cpla.solve_triangular(L.T, y, lower=False)
        return y.astype(_F32)

    g_r = cp.zeros(m2, dtype=_F32)
    g_s = cp.zeros(m2, dtype=_F32)
    g_u2 = cp.zeros(Nnode, dtype=_F32)
    g_u3 = cp.zeros(m2, dtype=_F32)
    g_s_prev = g_s.copy()
    g_Ar_prev = cp.zeros(Nnode, dtype=_F32)
    stop = "maxiter"
    fit = float("nan")
    ub = float("nan")
    ub_best, lb_best = float("inf"), 0.0
    k = 0
    cp.cuda.Stream.null.synchronize()
    t_loop = time.perf_counter()
    for k in range(1, maxiter + 1):
        if time.perf_counter() - t_loop > time_limit:
            stop = "time_limit"
            break
        cp.copyto(g_s_prev, g_s)
        At_rhs = AF.at_mul((_F32(wA) * g_b - g_u2).astype(_F32, copy=False),
                           n, Nloc)
        rhs = g_WtP2 + _F32(beta2 * wA) * At_rhs + _F32(beta3) * (g_s - g_u3)
        t_vec = AF.H0inv_woodbury(n, rhs, beta3, gamma, g_denom, Nnode, Nloc)
        y = chol_solve(g_L, _F32(2.0 * obj_scale) * C_mul(t_vec))
        corr = AF.H0inv_woodbury(n, Ct_mul(y), beta3, gamma, g_denom, Nnode, Nloc)
        g_r = t_vec - corr

        r_hat = _F32(alpha) * g_r + _F32(1 - alpha) * g_s
        g_s = cp.maximum(0, r_hat + g_u3)
        g_u3 = g_u3 + (r_hat - g_s)

        g_Ar = AF.a_mul(g_r, n, Nnode, Nloc)
        g_Ar_hat = _F32(alpha) * g_Ar + _F32(1 - alpha) * g_b
        g_u2 = g_u2 + _F32(wA) * (g_Ar_hat - g_b)

        if adapt_on and k >= 25 and k % 25 == 0:
            prA = float(cp.max(cp.abs(g_Ar - g_b)))
            prS = float(cp.max(cp.abs(g_r - g_s)))
            drA = beta2 * wA * float(cp.max(cp.abs(g_Ar - g_Ar_prev)))
            drS = beta3 * float(cp.max(cp.abs(g_s - g_s_prev)))
            f2 = 2.0 if prA > 5 * drA else (0.5 if drA > 5 * prA else 1.0)
            f3 = 2.0 if prS > 5 * drS else (0.5 if drS > 5 * prS else 1.0)
            fc = float(np.sqrt(f2 * f3))
            if fc != 1.0:
                b2o, b3o = beta2, beta3
                beta2 = max(1e-10, min(1e10, beta2 * fc))
                beta3 = max(1e-10, min(1e10, beta3 * fc))
                g_u2 *= _F32(b2o / beta2)
                g_u3 *= _F32(b3o / beta3)
                g_L = chol_factor(beta3)        # refactor (gamma & Mhat unchanged)
        cp.copyto(g_Ar_prev, g_Ar)

        if k % log_every == 0 or k == 1:
            g_Cr = C_mul(g_r)
            err = float(cp.sum((g_Cr - g_phat) ** 2))
            fit = float(np.sqrt(max(0.0, err)))
            if opts.get("verbose", False) and (k <= 10 or k % (log_every * 8) == 0):
                resA = float(cp.max(cp.abs(g_Ar - g_b)))
                resRS = float(cp.max(cp.abs(g_r - g_s)))
                rn = float(cp.max(cp.abs(g_r)))
                print(f"    k={k:5d} |r|={rn:.3e} fit_raw={fit:.5f} Arb={resA:.2e} "
                      f"rs={resRS:.2e} b3={beta3:.2e}", flush=True)
            if ub_rel_tol > 0 and opt_ref is not None:
                rf = AF.make_r_feasible_gpu(n, g_r, layers, bounds)
                ub = float(cp.sqrt(cp.sum((C_mul(rf) - g_phat) ** 2)))
                if ((ub - opt_ref) / opt_ref <= ub_rel_tol if opt_ref != 0
                        else ub_abs_tol is not None and ub - opt_ref <= ub_abs_tol):   # OPT = 0: absolute only
                    stop = "ub_rel"
                    break
            if push_fn is not None and lb_fn is not None:
                ub_best = min(ub_best, float(push_fn(g_r)))
                lb_best = max(lb_best, float(lb_fn(g_r)))
                gap = ub_best - lb_best
                if gap_abs_tol is not None and gap <= gap_abs_tol:
                    stop = "gap_abs"
                    break
                if gap_rel_tol is not None and lb_best > 0 and gap / lb_best <= gap_rel_tol:
                    stop = "gap_rel"
                    break
    cp.cuda.Stream.null.synchronize()
    loop_s = time.perf_counter() - t_loop
    rf = AF.make_r_feasible_gpu(n, g_r, layers, bounds)
    ub_final = float(cp.sqrt(cp.sum((C_mul(rf) - g_phat) ** 2)))
    out = {"fit": ub_final, "fit_raw_iterate": fit, "iters": k,
           "loop_s": loop_s, "setup_s": setup_s, "mhat_s": mhat_s,
           "per_iter_ms": loop_s / max(1, k) * 1e3, "stop_reason": stop,
           "mp": mp, "beta2": beta2, "beta3": beta3, "mhat_from_cache": mhat_from_cache}
    if push_fn is not None and lb_fn is not None:
        out.update(ub=ub_best, lb=lb_best)
    if opts.get("keep_mhat"):
        out.update(mhat=cp.asnumpy(g_Mhat), gamma=gamma)
    return out
