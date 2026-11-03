"""CGBFQP-GPU: column generation with Bellman-Ford pricing on the GPU (one kernel
launch per cardinality layer of the 2^n lattice), a bitset/popcount Gram
update and an active-set Cholesky master QP on the CPU.

    from solver.flow_cg import solve_cgbfqp
    r = solve_cgbfqp(n, cd, criterion="race", opt_ref=OPT)      # stop at UB <= (1+rel_tol)*OPT
    r = solve_cgbfqp(n, cd, criterion="cert", rel_tol=1e-3)     # stop at (UB-LB)/LB <= rel_tol

Parameter-free. Iterations = columns generated. The native reduced-cost stop is
disabled (rc_tol = 0) so the solver stops only on the criterion asked for.
wall_s excludes the one-off layer precomputation (setup_s).
"""
import time

import numpy as np

from .. import params
from ..common.instances import warmup_instance
from .solver import solve_cgbfqp_cholbitset


def warmup(n=10):
    """Compile the pricing kernels on a tiny instance."""
    solve_cgbfqp_cholbitset(n, warmup_instance(n), rc_tol=0.0, max_iters=20)


def solve_cgbfqp(n, cd, criterion="race", opt_ref=None, rel_tol=None, time_limit=None,
                 max_iters=params.CB_DEFAULTS["max_iters"], seed=0, verbose=False, layers="auto",
                 pricing="mask", abs_tol=None, init_columns=None, init_lambda=None,
                 early_ub_check=False, return_state=False, ub_stop=None, lb_stop=None):
    """init_columns / init_lambda: warm-restart the master from stored 0/1 ranking columns and
    their weights (e.g. the columns of a previous certified solve), instead of one random
    ranking. early_ub_check tests the master's own upper bound against the race target before
    each pricing sweep, so a warm restart that already clears the target costs no sweep at all.
    return_state=True adds `columns`, `lam` and `p_out` (the mixture) to the result so a later
    call can restart from them. It is off by default: those arrays are large (one byte per
    observation per column) and are not JSON-serialisable, so result files stay small."""
    if rel_tol is None:
        rel_tol = 1e-2 if criterion == "race" else 1e-3
    kw = dict(rc_tol=params.CB_DEFAULTS["rc_tol"], max_iters=max_iters, seed=seed, verbose=verbose,
              time_limit_s=float(time_limit) if time_limit else 0.0, layers=layers, pricing=pricing,
              init_columns=init_columns, init_lambda=init_lambda, early_ub_check=early_ub_check)
    if criterion == "race":
        if opt_ref is None:
            raise ValueError("criterion='race' needs opt_ref (a certified optimum)")
        if opt_ref == 0 and abs_tol is None:      # (1 + rel_tol) * 0 = 0 is never reached
            raise ValueError("opt_ref = 0: pass abs_tol (the race stops at UB <= abs_tol)")
        kw["phi_ub_target"] = (1.0 + rel_tol) * opt_ref if opt_ref != 0 else float(abs_tol)   # OPT = 0: absolute only
    else:
        kw["gap_rel_tol"] = rel_tol
        if abs_tol:
            kw["gap_abs_tol"] = float(abs_tol)     # certificate also closes at UB - LB <= abs_tol
    if ub_stop is not None:          # bootstrap threshold test: below / above / undecided
        kw["phi_ub_target"] = ub_stop
    if lb_stop is not None:
        kw["lb_stop"] = lb_stop
    t0 = time.perf_counter()
    r = solve_cgbfqp_cholbitset(n, np.asarray(cd, dtype=np.float64), **kw)
    stop = r["stop_reason"]
    return dict(method="flow_cg", criterion=criterion, ub=float(r["distance"]), lb=float(r["lb"]),
                iters=int(r["n_cols"]), wall_s=float(r["wall_s"]), build_s=float(r["setup_s"]),
                total_s=time.perf_counter() - t0, stop=stop, gap_rel=float(r["gap_rel"]),
                converged=stop in ("phi_ub", "gap_rel", "gap_abs", "lb_stop"),
                decision={"phi_ub": "below", "lb_stop": "above"}.get(stop), trace=r["trace"],
                layers_mode=r["layers_mode"], device_bytes=int(r["device_bytes"]),
                pricing=r["pricing"], margin_max=(max(r["margins"]) if r["margins"] else 0.0),
                **(dict(columns=r.get("columns"), lam=r.get("lambda"), p_out=r.get("p_out"))
                   if return_state else {}))
