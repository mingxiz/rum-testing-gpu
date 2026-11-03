r"""BM-ADMM: ADMM on the Block-Marschak (K) formulation, one GPU iteration as a few fused kernels.

The iteration is the tuned configuration of the paper (fixed penalties; adaptive penalties,
Nesterov steps and relative tau were tried and dropped). Per-iteration cost is cut by
  [fusion]   the ~14 elementwise CuPy ops (one full-slab pass + temporaries each) become 5 fused
             RawKernels with no temporaries:
               fuse_w     w    = y - b1*s
               fuse_rhs   rhs  = (z + b2*x2 - AT) * ib1
               fuse_xk    xk   = a*x + (1-a)*x2 - ib2*z
               fuse_z     z   -= b2*(a*x + (1-a)*x2 - x2n)
               fuse_sy    Axor = -a*Ax - (1-a)*s; s' = max(ib1*y - Axor, 0);
                          y   -= b1*(Axor + s')          (one kernel)
  [prox1]    the per-cardinality prox (n-1 launches) is ONE launch over all menus; k = __popc(menu)
             in-kernel (k == 1 -> value 1).
  [bfly]     the Mobius / eigen butterflies (Bfly2, butterfly2.py) write into PREALLOCATED ping-pong
             buffers, with the stage plan measured on this GPU at setup; every plan gives
             bit-identical output.
  [phi_obs]  phi by an observation-index gather (mp elements) instead of a full-slab reduction.
  [graph]    optional: the whole iteration captured as ONE CUDA graph, launched per iteration
             (removes the per-kernel launch latency of WDDM).
  [dtype]    float64 by default; float32 for the hybrid sprint and where float64 does not fit.
"""
import time

import numpy as np
import cupy as cp

from .kernels import _SCATTER_DT, _grid, _BLK, _F64, _F32
from .butterfly2 import Bfly2
from ..common.layers import generate_layer
from ..common.transforms import build_obs_idx

_EW_SRC = {
    "fuse_w": ("const real* y, const real* s, real b1, real* w",
               "w[i] = y[i] - b1*s[i];"),
    # AT/Ax arrive UN-negated (raw butterfly); the -1 of A_ineq is folded in.
    "fuse_rhs": ("const real* AT, const real* z, const real* x2,"
                 " real b2, real ib1, real* rhs",
                 "rhs[i] = (z[i] + b2*x2[i] - AT[i]) * ib1;"),
    "fuse_xk": ("const real* x, const real* x2, const real* z,"
                " real a, real ib2, real* xk",
                "xk[i] = a*x[i] + ((real)1-a)*x2[i] - ib2*z[i];"),
    "fuse_z": ("const real* x, const real* x2, const real* x2n,"
               " real a, real b2, real* z",
               "real xo = a*x[i] + ((real)1-a)*x2[i];"
               " z[i] -= b2*(xo - x2n[i]);"),
    "fuse_sy": ("const real* Ax, real a, real b1, real ib1,"
                " real* s, real* y",
                "real axor = -a*Ax[i] - ((real)1-a)*s[i];"
                " real sn = ib1*y[i] - axor; sn = sn > (real)0 ? sn : (real)0;"
                " y[i] -= b1*(axor + sn); s[i] = sn;"),
    "diag_mul": ("const real* g, long long Nloc, real* v",
                 "v[i] *= g[i % Nloc];"),
}


def _mk_ew(name, dt):
    args, body = _EW_SRC[name]
    src = (f"typedef {'float' if dt == _F32 else 'double'} real;\n"
           f'extern "C" __global__ void {name}({args}, long long m2){{\n'
           f"  long long i = blockIdx.x*(long long)blockDim.x + threadIdx.x;\n"
           f"  if (i >= m2) return;\n  {body}\n}}")
    return cp.RawKernel(src, name)


_PROX1_SRC = r'''
extern "C" __global__ void prox_all(
    const real* __restrict__ slab_in, real* __restrict__ slab_out,
    const real* __restrict__ xhat, const real* __restrict__ wmask,
    const int* __restrict__ menu_masks,
    long long Nloc, real beta2, int n_menus){
  int tt = blockIdx.x*blockDim.x + threadIdx.x;
  if (tt >= n_menus) return;
  unsigned int S = (unsigned int)menu_masks[tt];
  int k = __popc(S);
  real vals[32]; long long pos_arr[32];
  unsigned int SS = S; int kk = 0;
  real w0 = (real)0; bool mixed = false;
  while (SS){
    unsigned int bit = SS & (0u - SS);
    int i = __ffs(bit) - 1;
    unsigned int rm = S & ~(1u << i);
    unsigned int low = rm & ((1u << i) - 1u);
    unsigned int high = rm >> (i + 1);
    unsigned int rest = low | (high << i);
    long long pos = (long long)i*Nloc + rest;
    if (k == 1){ slab_out[pos] = (real)1; return; }
    real v = slab_in[pos], w = wmask[pos], xh = xhat[pos];
    if (kk == 0) w0 = w; else if (w != w0) mixed = true;
    vals[kk] = (beta2*v + (real)2*w*xh) / (beta2 + (real)2*w);
    pos_arr[kk] = pos;
    SS ^= bit; kk++;
  }
  double srt[32];
  if (mixed){
    // Menu with observed AND unobserved members: curvatures c_i = beta2 + 2 w_i differ, and
    // the exact prox is u_i = max((t_i - mu)/c_i, 0), t_i = beta2 v_i + 2 w_i xh_i, with mu
    // set by sum u_i = 1 (support = the largest t_i). The common threshold on t_i/c_i below
    // is exact only for equal c_i, so fully observed / unobserved menus keep that path.
    for (int a2=0; a2<k; ++a2){
      long long q = pos_arr[a2];
      srt[a2] = (double)beta2*(double)slab_in[q] + 2.0*(double)wmask[q]*(double)xhat[q];
    }
    for (int a2=1; a2<k; ++a2){                      // sort (t, pos) by t, descending
      double key = srt[a2]; long long kp = pos_arr[a2]; int b = a2-1;
      while (b>=0 && srt[b]<key){ srt[b+1]=srt[b]; pos_arr[b+1]=pos_arr[b]; --b; }
      srt[b+1] = key; pos_arr[b+1] = kp;
    }
    double stc = 0.0, sic = 0.0, mu = 0.0;
    for (int a2=0; a2<k; ++a2){
      double c = (double)beta2 + 2.0*(double)wmask[pos_arr[a2]];
      stc += srt[a2]/c; sic += 1.0/c;
      double th = (stc - 1.0) / sic;
      if (srt[a2] - th > 0.0) mu = th;
    }
    for (int a2=0; a2<k; ++a2){
      double c = (double)beta2 + 2.0*(double)wmask[pos_arr[a2]];
      double yv = (srt[a2] - mu) / c;
      slab_out[pos_arr[a2]] = yv > 0.0 ? (real)yv : (real)0.0;
    }
    return;
  }
  for (int a2=0; a2<k; ++a2) srt[a2] = (double)vals[a2];
  for (int a2=1; a2<k; ++a2){
    double key = srt[a2]; int b = a2-1;
    while (b>=0 && srt[b]<key){ srt[b+1]=srt[b]; --b; }
    srt[b+1] = key;
  }
  double css = 0.0, tau = 0.0;
  for (int a2=0; a2<k; ++a2){
    css += srt[a2];
    double th = (css - 1.0) / (double)(a2+1);
    if (srt[a2] - th > 0.0) tau = th;
  }
  for (int a2=0; a2<k; ++a2){
    double yv = (double)vals[a2] - tau;
    slab_out[pos_arr[a2]] = yv > 0.0 ? (real)yv : (real)0.0;
  }
}'''


def _mk_prox1(dt):
    src = ("typedef %s real;\n" % ("float" if dt == _F32 else "double")) \
        + _PROX1_SRC
    return cp.RawKernel(src, "prox_all")


def solve_kadmm_v2(n, choice_data, beta1=1.5, beta2=6.0, alpha_relax=1.95,
                   precond_p=1.41421356, max_iter=10 ** 9, check=50,
                   ub_target=None, ub_eps=1e-2, time_limit=None,
                   push_fn=None, dtype=None, use_graph=False, verbose=False,
                   init_state=None, stall_checks=0, stall_rtol=1e-3,
                   ub_arm_immediately=False, lb_fn=None, lb_target=None,
                   gap_abs_tol=None, gap_rel_tol=None, ub_running_min=False, ub_abs=None):
    """ub_abs: race target when ub_target == 0 (OPT = 0), where the relative test
    (ub - ub_target)/ub_target <= ub_eps is undefined: stop at ub - ub_target <= ub_abs.
    Unused for ub_target != 0, so the paper's race path is unchanged.

    Bootstrap hooks (used by the RUM class, solver/rum_solver.py; inert by default):
      ub_arm_immediately  compute the pushed upper bound from the FIRST check instead of
            waiting until the raw objective approaches ub_target. The default arming rule
            assumes ub_target sits near the optimum (a race to 1 % of OPT); in a bootstrap
            threshold test the target lies far ABOVE the draw's distance, so the raw
            objective never reaches it and the push would never fire.
      lb_fn, lb_target  optional certified lower bound evaluated at each check; the loop
            stops with stop_reason 'lb_target' once lb >= lb_target ('provably above').
      ub_running_min  report the BEST pushed upper bound seen so far instead of the most
            recent one. The push is a nonlinear rounding, so its value is NOT monotone in the
            iteration count: measured on the paper's own race cells the raw sequence dips and
            rises again on 4 of 5. Default False keeps the semantics the paper's recorded
            race numbers were produced with (the stop tests the LATEST push). The bootstrap
            decider sets it, because there the push is armed from the first check, where
            non-monotonicity is the rule rather than the exception.
      gap_abs_tol, gap_rel_tol  with both push_fn and lb_fn given, certify in ONE call:
            stop once the running bracket satisfies ub - lb <= gap_abs_tol or
            (ub - lb)/lb <= gap_rel_tol. Certifying by calling this function repeatedly in
            chunks instead re-allocates the whole state (14 slabs) per chunk, which at n = 25
            is 22 GiB per chunk and thrashes; one call allocates once.

    init_state: optional dict {x2,s,y,z} (any float dtype) to resume from —
    used by the hybrid f32->f64 driver. stall_checks>0: stop with
    stop_reason='stall' after that many consecutive UB/phi checks without a
    relative improvement > stall_rtol (progress-plateau exit for the f32 leg)."""
    t_setup = time.perf_counter()
    rdt = np.dtype(dtype) if dtype is not None else _F64
    Nloc = 1 << (n - 1)
    m2 = n * Nloc
    om = np.ascontiguousarray(choice_data[:, 0]).astype(np.uint32)
    ox0 = np.ascontiguousarray(choice_data[:, 1]).astype(np.int32)
    ph = np.ascontiguousarray(choice_data[:, 2]).astype(rdt)
    M = len(ph)
    scat = _SCATTER_DT[rdt]
    g_ox = cp.asarray(ox0); g_om = cp.asarray(om); g_ph = cp.asarray(ph)
    xhat = cp.zeros(m2, rdt); wmask = cp.zeros(m2, rdt)
    scat(_grid(M), (_BLK,), (g_ox, g_om, g_ph, xhat, np.int32(M), np.int32(Nloc)))
    scat(_grid(M), (_BLK,), (g_ox, g_om, cp.ones(M, rdt), wmask,
                             np.int32(M), np.int32(Nloc)))
    obs_idx = cp.asarray(build_obs_idx(n, om, ox0))   # gather index for phi (mp elements, not the slab)

    p = float(precond_p)
    alpha = rdt.type(alpha_relax)
    b1 = rdt.type(beta1); b2 = rdt.type(beta2)
    ib1 = rdt.type(1.0 / beta1); ib2 = rdt.type(1.0 / beta2)
    gamma = beta2 / beta1

    # eigen data (as in ExactSolveGPU, but stored for the buffer-stable path)
    Ghat = np.array([[1.0, -1.0], [-1.0, 1.0 + p * p]])
    evals, Q = np.linalg.eigh(Ghat)
    lam = np.ones(Nloc)
    for bit in range(n - 1):
        b1s = ((np.arange(Nloc) >> bit) & 1).astype(np.float64)
        lam *= np.where(b1s > 0, evals[1], evals[0])
    inv_gl = cp.asarray((1.0 / (gamma + lam)).astype(rdt))
    Qt = Q.T

    # all menus (k>=1) in one mask array for the single-launch prox
    masks = [generate_layer(n, k).astype(cp.int32) for k in range(1, n + 1)]
    all_masks = cp.concatenate(masks)
    n_menus = int(all_masks.size)

    ew = {k: _mk_ew(k, rdt) for k in _EW_SRC}
    prox1 = _mk_prox1(rdt)
    bf = Bfly2(n, rdt)

    # persistent buffers (pointer-stable: required for graph capture)
    x2 = cp.zeros(m2, rdt); s = cp.zeros(m2, rdt)
    y = cp.zeros(m2, rdt); z = cp.zeros(m2, rdt)
    w = cp.empty(m2, rdt); AT = cp.empty(m2, rdt)
    rhs = cp.empty(m2, rdt); x = cp.empty(m2, rdt)
    Ax = cp.empty(m2, rdt); xk = cp.empty(m2, rdt); x2n = cp.empty(m2, rdt)
    m2i = np.int64(m2)
    g1 = _grid(m2)

    # Calibrate the butterfly plan on this GPU using two scratch slabs (AT/rhs
    # are pure scratch until one_iter runs). Must precede any one_iter call so
    # the plan is fixed before graph capture. Cached per (device, dtype, n), so
    # only the first solve in a process pays.
    bf.calibrate(AT, rhs)

    def one_iter():
        ew["fuse_w"](g1, (_BLK,), (y, s, b1, w, m2i))
        bf.apply(w, AT, 1.0, 0.0, -1.0, p)                    # raw A' (sign folded)
        ew["fuse_rhs"](g1, (_BLK,), (AT, z, x2, b2, ib1, rhs, m2i))
        bf.apply(rhs, x, Qt[0, 0], Qt[0, 1], Qt[1, 0], Qt[1, 1])
        ew["diag_mul"](g1, (_BLK,), (inv_gl, np.int64(Nloc), x, m2i))
        bf.apply(x, x, Q[0, 0], Q[0, 1], Q[1, 0], Q[1, 1])
        bf.apply(x, Ax, 1.0, -1.0, 0.0, p)                    # raw A (sign folded)
        ew["fuse_xk"](g1, (_BLK,), (x, x2, z, alpha, ib2, xk, m2i))
        prox1(_grid(n_menus), (_BLK,),
              (xk, x2n, xhat, wmask, all_masks, np.int64(Nloc), b2,
               np.int32(n_menus)))
        ew["fuse_z"](g1, (_BLK,), (x, x2, x2n, alpha, b2, z, m2i))
        ew["fuse_sy"](g1, (_BLK,), (Ax, alpha, b1, ib1, s, y, m2i))
        cp.copyto(x2, x2n)

    if init_state is not None:
        for name, buf in (("x2", x2), ("s", s), ("y", y), ("z", z)):
            v = init_state[name]
            if isinstance(v, cp.ndarray):
                cp.copyto(buf, v.astype(rdt))
            else:            # host state (a state saved on the host): no device temp
                buf.set(np.ascontiguousarray(v, dtype=rdt))
    else:
        # warm start: prox of xhat (matches baseline)
        prox1(_grid(n_menus), (_BLK,),
              (xhat, x2, xhat, wmask, all_masks, np.int64(Nloc), b2,
               np.int32(n_menus)))

    graph = None
    if use_graph:
        # warmup one_iter compiles/allocs BUT advances state -> snapshot and
        # restore so the graph trajectory aligns exactly with eager (the UB
        # race samples every `check` iters; a 1-iter phase shift changes
        # which threshold crossing gets sampled).
        snap = [buf.copy() for buf in (x2, s, y, z)]
        st = cp.cuda.Stream(non_blocking=True)
        # st is non-blocking: it does NOT wait for the null-stream setup work
        # above (scatter/memset/warm-start prox, all_masks concat). Without
        # this fence the warm-up one_iter can read half-built index arrays and
        # fault (cudaErrorIllegalAddress) -- seen on the 2nd+ solve of a
        # process, where the cached butterfly plan skips calibration's syncs.
        cp.cuda.Device().synchronize()
        with st:
            one_iter()                        # warm compile/alloc outside capture
            cp.cuda.Device().synchronize()
            st.begin_capture()
            one_iter()
            graph = st.end_capture()
        cp.cuda.Device().synchronize()
        for buf, sv in zip((x2, s, y, z), snap):
            cp.copyto(buf, sv)
        del snap

    ub_last = float("nan"); push_s = 0.0; n_push = 0
    ub_armed = False
    phi = float("nan")
    stop = "max_iter"
    best_prog = -float("inf"); n_flat = 0        # stall detection (phi rises)
    stall_was_armed = False
    if ub_arm_immediately:
        ub_armed = True
    lb_last = 0.0
    cp.cuda.Stream.null.synchronize()
    setup_s = time.perf_counter() - t_setup
    t0 = time.perf_counter()
    it = 0
    trace = []            # per check: [iteration, wall_s, phi(iterate), pushed UB or None]
    while it < max_iter:
        if graph is not None:
            with st:
                for _ in range(check):
                    graph.launch(st)
            st.synchronize()
            it += check
        else:
            for _ in range(check):
                one_iter()
            it += check
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            stop = "time_limit"; break
        phi = float(cp.sqrt(cp.sum((x2[obs_idx] - g_ph) ** 2)).get())
        rec = [it, time.perf_counter() - t0, phi, None]
        trace.append(rec)
        if push_fn is not None:
            if not ub_armed:
                # no target (a certification call) -> the bound is wanted from the first check
                ub_armed = ub_target is None or phi >= ub_target * (1.0 - 3.0 * ub_eps)
            if ub_armed:
                t_p = time.perf_counter()
                _ub = float(push_fn(x2))
                # min() only when asked: see ub_running_min above. ub_last starts as NaN and
                # min(nan, x) is nan, so the first value must replace it outright.
                ub_last = _ub if (not ub_running_min or np.isnan(ub_last)) else min(ub_last, _ub)
                push_s += time.perf_counter() - t_p
                n_push += 1
                rec[1] = time.perf_counter() - t0
                rec[3] = ub_last
                if ub_target is not None and (
                        (ub_last - ub_target) / ub_target <= ub_eps if ub_target != 0
                        else ub_abs is not None and ub_last - ub_target <= ub_abs):   # OPT = 0: absolute only
                    stop = "ub_target"; break
        if lb_fn is not None and (lb_target is not None or gap_abs_tol is not None
                                  or gap_rel_tol is not None):
            lb_last = max(lb_last, float(lb_fn(x2)))
            if lb_target is not None and lb_last >= lb_target:
                stop = "lb_target"; break
            if not np.isnan(ub_last):
                gap = ub_last - lb_last
                if gap_abs_tol is not None and gap <= gap_abs_tol:
                    stop = "gap_abs"; break
                if gap_rel_tol is not None and lb_last > 0 and gap / lb_last <= gap_rel_tol:
                    stop = "gap_rel"; break
        if stall_checks > 0:
            # progress = phi (rises to OPT) and, once armed, -ub (falls)
            if ub_armed != stall_was_armed:      # metric switch -> reset
                best_prog = -float("inf"); n_flat = 0
                stall_was_armed = ub_armed
            prog = phi if not ub_armed else -ub_last
            if prog > best_prog * (1.0 + np.sign(best_prog) * stall_rtol) \
                    or best_prog == -float("inf"):
                best_prog = prog; n_flat = 0
            else:
                n_flat += 1
                if n_flat >= stall_checks:
                    stop = "stall"; break
        if verbose and it % (check * 20) == 0:
            print(f"   it={it} phi={phi:.6f} ub={ub_last:.6f}", flush=True)
    cp.cuda.Stream.null.synchronize()
    wall_s = time.perf_counter() - t0
    phi = float(cp.sqrt(cp.sum((x2[obs_idx] - g_ph) ** 2)).get())
    return {"distance": phi, "iters": it, "wall_s": wall_s, "setup_s": setup_s,
            "per_iter_ms": wall_s / max(1, it) * 1e3, "stop_reason": stop,
            "x2": x2, "ub": ub_last, "lb": lb_last, "push_s": push_s, "n_push": n_push,
            "trace": trace,
            "dtype": str(rdt), "use_graph": bool(use_graph),
            "bfly": "v2", "bfly_plan": bf.plan,
            "state": {"x2": x2, "s": s, "y": y, "z": z}}


def solve_kadmm_hybrid(n, choice_data, beta1=1.5, beta2=6.0,
                       alpha_relax=1.95, precond_p=1.41421356, check=50,
                       ub_target=None, ub_eps=1e-2, time_limit=None,
                       push_fn=None, use_graph=True, stall_checks=3,
                       stall_rtol=1e-3, verbose=False, ub_arm_immediately=False,
                       ub_abs=None):
    """f32 sprint until the pushed-UB/phi progress plateaus (its precision
    floor), then hand the state to the f64 solver to close the race.
    Returns the f64 result with combined iters/wall and leg breakdown.
    ub_abs: absolute race target used only when ub_target == 0 (see solve_kadmm_v2)."""
    t0 = time.perf_counter()
    r32 = solve_kadmm_v2(n, choice_data, beta1=beta1, beta2=beta2,
                         alpha_relax=alpha_relax, precond_p=precond_p,
                         check=check, ub_target=ub_target, ub_eps=ub_eps,
                         time_limit=time_limit, push_fn=push_fn, ub_arm_immediately=ub_arm_immediately,
                         dtype="float32", use_graph=use_graph,
                         stall_checks=stall_checks, stall_rtol=stall_rtol,
                         verbose=verbose, ub_abs=ub_abs)
    if r32["stop_reason"] == "ub_target":            # f32 alone finished
        # confirm the stop in f64 (review P2b): recompute the feasibility
        # push from the f64-cast iterate; accept only if the criterion
        # still holds — otherwise fall through to the f64 leg.
        ub64 = push_fn(r32["state"]["x2"].astype(cp.float64))
        if ((ub64 - ub_target) / ub_target <= ub_eps if ub_target != 0
                else ub_abs is not None and ub64 - ub_target <= ub_abs):   # OPT = 0: absolute only
            r32["ub"] = ub64                          # f64-confirmed UB
            r32["iters32"] = r32["iters"]; r32["iters64"] = 0
            r32["wall32_s"] = r32["wall_s"]; r32["wall64_s"] = 0.0
            r32["f64_confirmed"] = True
            return r32
    rem = None if time_limit is None else \
        max(1.0, time_limit - (time.perf_counter() - t0))
    r64 = solve_kadmm_v2(n, choice_data, beta1=beta1, beta2=beta2,
                         alpha_relax=alpha_relax, precond_p=precond_p,
                         check=check, ub_target=ub_target, ub_eps=ub_eps,
                         time_limit=rem, push_fn=push_fn, ub_arm_immediately=ub_arm_immediately,
                         dtype="float64", use_graph=use_graph,
                         init_state=r32["state"], verbose=verbose, ub_abs=ub_abs)
    r64["iters32"] = r32["iters"]; r64["iters64"] = r64["iters"]
    r64["wall32_s"] = r32["wall_s"]; r64["wall64_s"] = r64["wall_s"]
    r64["trace"] = r32["trace"] + [[it + r32["iters"], w + r32["wall_s"], phi, ub]
                                   for it, w, phi, ub in r64["trace"]]
    r64["iters"] = r32["iters"] + r64["iters"]
    r64["wall_s"] = time.perf_counter() - t0
    r64["push_s"] = r32["push_s"] + r64["push_s"]
    return r64
