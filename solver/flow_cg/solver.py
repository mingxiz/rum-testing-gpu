"""The CGBFQP-GPU column-generation loop (entry point: solve_cgbfqp in __init__.py).

Each iteration
  1. solves the restricted master  min 1/2 lam'Q lam - h'lam  over the simplex with the
     active-set Cholesky solver (common/qp_native.solve_qp_native), on Q = V'V and h = V'phat
     kept incrementally from packed column bitsets (qp_bitset_incremental.py);
  2. reads the error and the inner product off Q, h and lam (the M x K matrix V is never formed);
  3. prices on the GPU: a Bellman-Ford sweep over the subset lattice, one kernel launch per
     cardinality layer (bf_pricing.py), or the rank-indexed streamed sweep for large n
     (pricing_stream.py);
  4. tests the certified gap: LB = sqrt(max(UB^2 - 2 fw_gap, 0)), kept as a running max;
  5. adds the priced ranking's 0/1 column.
"""
import time

import numpy as np
import cupy as cp

from .cg_core import CGData
from .bf_pricing import bf as bf_kernel
from .qp_bitset_incremental import BitsetGramMaintainer, PoutMaintainer
from ..common.qp_native import solve_qp_native


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------
def solve_cgbfqp_cholbitset(n, choice_data, rc_tol=0.01, max_iters=1000,
                              time_limit_s=0.0, seed=0, verbose=False,
                              phi_ub_target=None, gap_rel_tol=None, gap_abs_tol=None, layers="auto",
                              pricing="mask", init_columns=None, init_lambda=None,
                              early_ub_check=False, lb_stop=None):
    """CGBFQP-CholBitset: GPU BF pricing + bitset Q + AS-Chol Q,h-only.

    Args:
        n: number of alternatives
        choice_data: (M, 3) np array [menu_mask, chosen_alt_0based, prob]
        rc_tol: stop when reduced cost fw_gap = V_max − ⟨u, p_current⟩ < rc_tol
        max_iters: hard cap on CG iterations
        time_limit_s: optional wall-clock cap (0 = no cap)
        seed: seed of the random initial ranking (a local generator; NumPy's global
            random state is left untouched)
        verbose: print per-iter progress
        gap_rel_tol: optional CERTIFIED relative-gap stop in distance space:
            LB = sqrt(max(UB^2 - 2*fw_gap, 0)) (FW duality, f* >= f - gap),
            tracked as a running max; stop when (UB - LB)/LB <= gap_rel_tol.
        pricing: "mask" (one value per subset, 5 B/node; n <= 32 on 32 GB),
            "stream_f32" or "stream_u16" (rank-indexed, two live layers; see
            pricing_stream.py). stream_f32 is bit-identical to mask; stream_u16
            adds its quantization margin to Vmax before the FW gap.

    Returns dict with: algorithm, wall_s, setup_s, iters, n_cols, rc_final,
        distance, stop_reason, qp_total_s, kernel_total_s, backtrace_total_s,
        qp_pct, kernel_pct, bt_pct.
    """
    cp.cuda.Device().synchronize()

    t_setup = time.perf_counter()
    model = CGData(choice_data, n, layers=(layers if pricing == "mask" else "none"))
    M = len(model.choice_probabilities)
    pricer = None
    if pricing != "mask":
        from .pricing_stream import StreamPricer
        pricer = StreamPricer(n, model.choice_set, model.choice_alternative, model.offsets,
                              mode={"stream_f32": "f32", "stream_u16": "u16", "stream_u24": "u24"}[pricing])
    cp.cuda.Device().synchronize()
    setup_s = time.perf_counter() - t_setup

    # GPU pricing scaffolding (mask-indexed path only)
    Nnode = 1 << n
    if pricer is None:
        V_dev = cp.zeros(Nnode, dtype=cp.float32)
        pred = cp.zeros(Nnode, dtype=cp.uint8)
        menu_masks_dev = cp.asarray(model.choice_set, dtype=cp.uint32)
    else:
        V_dev = pred = menu_masks_dev = None
    offsets = cp.asarray(model.offsets, dtype=cp.uint64)
    margins = []

    # Use the SAME float64-cast phat the maintainer uses internally
    p_obs = model.choice_probabilities.astype(np.float64)
    p_obs_sq = float(p_obs @ p_obs)
    lb_best = 0.0            # running max of the certified FW lower bound (distance)
    gap_rel = float("inf")

    # First column from a random ranking (local generator: the caller's np.random state is untouched)
    ranking_initial = np.random.RandomState(seed).permutation(n)
    col0_u8 = model.map_ranking_to_choice_probability(ranking_initial)

    # Maintainers
    n_init = len(init_columns) if init_columns is not None else 0
    maintainer = BitsetGramMaintainer(M, p_obs, max_K=max_iters + 2 + n_init)
    pout_mtr = PoutMaintainer(M, maintainer)
    if n_init:
        # Warm restart: seed the master with stored 0/1 ranking columns and their weights, so
        # the solve continues from a previous point instead of from one random ranking.
        cols_init = [np.ascontiguousarray(np.asarray(c, dtype=np.uint8)) for c in init_columns]
        for c in cols_init:
            maintainer.add_column(c.astype(np.float64))
            pout_mtr.add_column(c)
        K = n_init
        lam = (np.asarray(init_lambda, dtype=np.float64).copy() if init_lambda is not None
               else np.full(K, 1.0 / K))
        lam = np.maximum(lam, 0.0)
        lam /= lam.sum()
        pout_mtr.apply_delta(lam)
        A_warm = [int(i) for i in np.where(lam > 1e-12)[0]] or [0]
        keys_list = [bytes(c) for c in cols_init]
    else:
        maintainer.add_column(col0_u8.astype(np.float64))
        pout_mtr.add_column(col0_u8)
        # State
        K = 1
        lam = np.array([1.0])
        pout_mtr.apply_delta(np.array([1.0]))
        A_warm = [0]
        keys_list = [bytes(col0_u8)]

    qp_ms_total = 0.0
    kernel_ms_total = 0.0
    bt_ms_total = 0.0
    rc_final = float("nan")
    UB = float("inf")
    stop_reason = "max_iters"

    cp.cuda.Device().synchronize()
    t_total = time.perf_counter()
    iters = 0
    trace = []            # (column, wall_s, UB, LB_best, n_cols) after every pricing sweep

    for it in range(1, max_iters + 1):
        iters = it
        if time_limit_s > 0 and (time.perf_counter() - t_total) > time_limit_s:
            stop_reason = "time_limit"
            break

        # ---- master QP on (Q, h) ----
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        Q_view, h_view = maintainer.view()
        lam_new, A_warm = solve_qp_native(Q_view, h_view, A_init=A_warm)

        # Incremental p_out update
        delta = lam_new - lam
        pout_mtr.apply_delta(delta)
        lam = lam_new

        # Scalars from Q, h, lam — no dense V
        Q_lam = Q_view @ lam
        lam_Q_lam = float(lam @ Q_lam)
        h_lam = float(h_view @ lam)
        error_squared = lam_Q_lam - 2.0 * h_lam + p_obs_sq
        inner_product = h_lam - lam_Q_lam

        if early_ub_check and phi_ub_target is not None:
            # The master alone may already clear the target (a warm restart usually does),
            # in which case the pricing sweep for this iteration is not needed at all.
            UB = float(np.sqrt(max(0.0, error_squared)))
            if UB <= phi_ub_target:
                stop_reason = "phi_ub"
                break
        qp_ms = (time.perf_counter() - t) * 1e3
        qp_ms_total += qp_ms

        residual = p_obs - pout_mtr.p_out

        # ---- pricing (GPU) ----
        margin = 0.0
        if pricer is None:
            menu_scores = cp.asarray(residual.astype(np.float32))
            V_dev.fill(0)
            pred.fill(0)
            cp.cuda.Device().synchronize()
            ev_s = cp.cuda.Event()
            ev_e = cp.cuda.Event()
            ev_s.record()
            threads = 256
            for c in range(1, n + 1):
                states = model.layer(c)
                ns = int(states.size)
                if ns == 0:
                    continue
                grid = (ns + threads - 1) // threads
                bf_kernel((grid,), (threads,),
                          (V_dev, pred, menu_masks_dev, menu_scores, offsets, states, ns))
            ev_e.record()
            ev_e.synchronize()
            kernel_ms = cp.cuda.get_elapsed_time(ev_s, ev_e)
            kernel_ms_total += kernel_ms

            # ---- backtrace (host) ----
            t = time.perf_counter()
            D = (1 << n) - 1
            Vmax = float(V_dev[D])
            ranking_out = np.zeros(n, dtype=np.int32)
            for i in range(n):
                curr = int(pred[D])
                ranking_out[curr] = i
                D ^= (1 << curr)
            bt_ms_total += (time.perf_counter() - t) * 1e3
        else:
            t = time.perf_counter()
            Vmax, ranking_out, margin = pricer.price(residual.astype(np.float32))
            kernel_ms_total += (time.perf_counter() - t) * 1e3      # sweep + backtrace + sub-lattice
            margins.append(margin)

        # ---- gap test (Vmax + margin is a valid upper bound on the true pricing maximum) ----
        fw_gap = (Vmax + margin) - inner_product
        rc_final = fw_gap
        UB = float(np.sqrt(max(0.0, error_squared)))
        # certified LB (distance): FW duality f* >= f - 2*rc in objective space
        lb_now = float(np.sqrt(max(0.0, error_squared - 2.0 * fw_gap)))
        lb_best = max(lb_best, lb_now)
        gap_rel = (UB - lb_best) / lb_best if lb_best > 0 else float("inf")
        trace.append((it, time.perf_counter() - t_total, UB, lb_best, K))

        if verbose and (it <= 5 or it % 25 == 0):
            print(f"[CholBitset iter {it:4d}] K={K:4d} |A|={len(A_warm):3d} "
                  f"dist={UB:.5f} rc={fw_gap:.5f} gap={gap_rel:.2e}")

        if phi_ub_target is not None and UB <= phi_ub_target:
            stop_reason = "phi_ub"
            break
        if lb_stop is not None and lb_best >= lb_stop:   # bootstrap: provably ABOVE the threshold
            stop_reason = "lb_stop"
            break
        if gap_rel_tol is not None and gap_rel <= gap_rel_tol:
            stop_reason = "gap_rel"
            break
        if gap_abs_tol is not None and (UB - lb_best) <= gap_abs_tol:   # absolute floor: distance ~ 0 cells
            stop_reason = "gap_abs"
            break
        if fw_gap < rc_tol:
            stop_reason = "rc"
            break

        # ---- add new column ----
        new_col_u8 = model.map_ranking_to_choice_probability(ranking_out)
        keys_list.append(bytes(new_col_u8))
        maintainer.add_column(new_col_u8.astype(np.float64))
        pout_mtr.add_column(new_col_u8)
        K += 1
        lam = np.concatenate([lam, [0.0]])

    cp.cuda.Device().synchronize()
    wall_s = time.perf_counter() - t_total

    # Cleanup GPU
    layers_mode = model.layers_mode
    device_bytes = pricer.device_bytes if pricer is not None else model.device_bytes()
    del V_dev, pred, menu_masks_dev, offsets, model, pricer
    cp.get_default_memory_pool().free_all_blocks()

    return {
        "algorithm": "CGBFQP-CholBitset",
        "wall_s": wall_s,
        "setup_s": setup_s,
        "trace": trace,
        "pricing": pricing,
        "margins": margins,
        "layers_mode": layers_mode,
        "device_bytes": device_bytes,
        "iters": iters,
        "n_cols": K,
        "lambda": lam,
        "p_out": pout_mtr.p_out.copy(),          # the mixture V lam, in the model's x-major order
        "columns": [np.frombuffer(k, dtype=np.uint8).copy() for k in keys_list],
        "rc_final": rc_final,
        "distance": UB,
        "lb": lb_best,
        "gap_rel": gap_rel,
        "stop_reason": stop_reason,
        "qp_total_s": qp_ms_total / 1e3,
        "kernel_total_s": kernel_ms_total / 1e3,
        "backtrace_total_s": bt_ms_total / 1e3,
        "qp_pct": qp_ms_total / 1e3 / wall_s * 100 if wall_s > 0 else 0,
        "kernel_pct": kernel_ms_total / 1e3 / wall_s * 100 if wall_s > 0 else 0,
        "bt_pct": bt_ms_total / 1e3 / wall_s * 100 if wall_s > 0 else 0,
    }
