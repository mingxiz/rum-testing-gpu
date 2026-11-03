"""One feasibility-study measurement in a fresh process.

    python validation/feasibility/feas_one.py --key n30_sparse --solver cgbfqp --criterion cert --out DIR
    python validation/feasibility/feas_one.py --key n30_sparse --solver cgbfqp --criterion race --out DIR

cert : certify (UB-LB)/LB <= 1e-3 through the RUM class (solver/rum_solver.py), with the
       chosen solver's upper and lower bounds. If data/opt_refs.json
       has no entry for the cell, the result is written there as its OPT reference
       (opt = UB, lb, gap, method, mp, cd_hash). If an entry exists it is NOT
       overwritten; the new bracket is compared with it (consistent = the two
       brackets [LB, UB] overlap, so both certify the same optimum) and recorded.
race : time to (UB - OPT)/OPT <= 1% against data/opt_refs.json, with the full
       per-check trajectory (column or iteration, wall seconds, UB, LB) saved.
Writes DIR/<criterion>_<solver>_<key>.json with the result, the trajectory and
what the machine did (cores busy, peak host RSS, device memory).
"""
import argparse
import json
import os
import platform
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "validation"))

import numpy as np                                     # noqa: E402
from gen_instances import get_instance, fingerprint    # noqa: E402
from run_one import peak_rss_mb, gpu_info, certify     # noqa: E402
import solver                                          # noqa: E402

REFS = os.path.join(ROOT, "data", "opt_refs.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--solver", required=True, choices=["bm_admm", "cgbfqp"])
    ap.add_argument("--criterion", default="race", choices=["race", "cert"])
    ap.add_argument("--cap", type=float, default=3600.0)
    ap.add_argument("--cert-every", type=int, default=200,
                    help="bm_admm cert: iterations between upper and lower bounds (the class's check)")
    ap.add_argument("--max-cols", type=int, default=4000, help="cgbfqp: column cap")
    ap.add_argument("--pricing", default="mask", choices=["mask", "stream_f32", "stream_u16", "stream_u24"],
                    help="cgbfqp pricing backend (see solver/flow_cg/pricing_stream.py)")
    ap.add_argument("--tag", default=None, help="suffix for the result file name (default: the pricing backend if not mask)")
    ap.add_argument("--race-floor", type=float, default=0.0,
                    help="race: absolute floor on the stop target, used where OPT is ~0 and the "
                         "relative criterion (UB-OPT)/OPT is undefined; target = max((1+rel_tol)*OPT, floor)")
    ap.add_argument("--bm-mode", default=None, choices=[None, "hybrid", "v2"],
                    help="bm_admm race: 'hybrid' (f32 sprint + f64 finish, default) or 'v2' (single precision leg)")
    ap.add_argument("--bm-dtype", default=None, choices=[None, "float64", "float32"],
                    help="bm_admm race with --bm-mode v2: the working precision")
    ap.add_argument("--abs-tol", type=float, default=0.0,
                    help="cgbfqp cert: also accept UB - LB <= this (for cells whose distance is ~0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    n, cat = int(a.key.split("_")[0][1:]), a.key.split("_", 1)[1]
    t = time.perf_counter()
    cd, mp = get_instance(n, cat)
    gen_s = time.perf_counter() - t
    fp = fingerprint(cd)
    refs = json.load(open(REFS))
    is_lam0 = a.key.endswith("_lam0")              # the convex-path lambda = 0 cell IS the pairwise cell
    ref = refs.get(a.key[:-5] if is_lam0 else a.key)
    if ref is not None:
        assert ref["cd_hash"] == fp, "instance does not match the stored OPT reference (%s vs %s)" % (fp, ref["cd_hash"])
    out = dict(key=a.key, n=n, cat=cat, mp=int(mp), density_pct=100.0 * mp / (n * (1 << (n - 1))),
               cd_hash=fp, gen_s=gen_s, solver=a.solver, criterion=a.criterion, cap_s=a.cap,
               hostname=socket.gethostname(), cpu_model=platform.processor(), logical_cores=os.cpu_count(),
               python=platform.python_version(), when=time.strftime("%Y-%m-%d %H:%M:%S"))
    solver.warmup((a.solver,))
    tag = a.tag if a.tag is not None else ("" if a.pricing == "mask" else a.pricing)
    c0, w0 = time.process_time(), time.perf_counter()
    if a.criterion == "cert":
        # the RUM class certifies: BM-ADMM computes both bounds every --cert-every iterations,
        # CGBFQP at every column; both also stop at UB - LB <= --abs-tol
        args = (dict(check=a.cert_every, lb_every=1) if a.solver == "bm_admm"
                else dict(max_iters=a.max_cols, pricing=a.pricing))
        r = certify(n, cd, a.solver, rel_tol=1e-3, abs_tol=a.abs_tol, max_time=a.cap,
                    method_args=args, verbose=a.verbose)
    else:
        if ref is None:
            sys.exit("no OPT reference for %s: run --criterion cert first" % a.key)
        kw = {}
        if a.solver == "bm_admm":
            if a.bm_mode:
                kw["mode"] = a.bm_mode
            if a.bm_dtype:
                kw["dtype"] = a.bm_dtype
        else:
            kw["max_iters"] = a.max_cols
            kw["pricing"] = a.pricing
        opt_ref = float(ref["opt"])
        # target = max(1.01*OPT, race_floor): on the convex path the large-lambda cells have
        # OPT ~ 0, where a relative target is undefined (and exactly 0 at lambda = 1).
        if a.race_floor > 0 and 1.01 * opt_ref < a.race_floor:
            opt_ref_eff = a.race_floor / 1.01
            out["race_floor_applied"] = a.race_floor
        else:
            opt_ref_eff = opt_ref
        out["race_target"] = 1.01 * opt_ref_eff
        if a.solver == "bm_admm" and opt_ref_eff != opt_ref:
            kw["ub_arm_immediately"] = True    # floor cell: arm the feasibility push from the first check
        r = solver.solve(n, cd, method=a.solver, criterion="race", opt_ref=opt_ref_eff, rel_tol=1e-2,
                         time_limit=a.cap, verbose=a.verbose, **kw)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    r = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v)) for k, v in r.items()}
    out.update(r)
    out.update(process_cpu_s=cpu, process_wall_s=wall, avg_cores=cpu / max(wall, 1e-9),
               peak_host_rss_mb=peak_rss_mb(), **gpu_info())

    if a.criterion == "cert":
        ub, lb = float(out["ub"]), float(out["lb"])
        gap = (ub - lb) / max(lb, 1e-12)
        by_abs = bool(a.abs_tol) and (ub - lb) <= a.abs_tol
        method = {"cgbfqp": "CB-GPU %s" % out.get("stop") + ("" if a.pricing == "mask" else " [%s]" % a.pricing),
                  "bm_admm": "K-ADMM cert (beta=%.3g)" % out.get("beta", float("nan"))}[a.solver]
        if out.get("converged") and (gap <= 1e-3 or by_abs):
            if ref is None:
                rec = dict(opt=ub, lb=lb, gap=(gap if lb > 0 else None), gap_abs=ub - lb,
                           criterion=("gap_rel" if (lb > 0 and gap <= 1e-3) else "gap_abs<=%g" % a.abs_tol),
                           method=method, mp=int(mp),
                           n_cols=int(out["iters"]) if a.solver == "cgbfqp" else None,
                           iters=int(out["iters"]), wall_s=float(out["wall_s"]), cd_hash=fp,
                           when=out["when"])
                if "_lam" in cat:
                    rec.update(**{"lambda": int(cat.split("_lam")[1]) / 10.0,
                                  "base_key": "n%d_%s" % (n, cat.split("_lam")[0])})
                refs[a.key] = rec
                refs = dict(sorted(refs.items(), key=lambda kv: (int(kv[0].split("_")[0][1:]), kv[0])))
                with open(REFS, "w") as f:
                    json.dump(refs, f, indent=2)
                out["ref_action"] = "written"
            else:
                # Two certificates are consistent when their brackets overlap: both UBs
                # are objective values of feasible points and both LBs are valid, so the
                # true optimum lies in [max(LBs), min(UBs)]; a UB may be tighter than the
                # stored opt by up to the 1e-3 certificate tolerance.
                lo, hi = max(lb, ref["lb"]), min(ub, ref["opt"])
                agree = lo <= hi * (1 + 1e-9)
                if is_lam0:
                    # lambda = 0 is the base cell itself. CGBFQP is deterministic: the same columns and
                    # the same UB, up to the float32 rounding of p_hat that the class corrects the UB
                    # for (p_float32_shift). BM-ADMM: the two brackets must overlap.
                    same = agree
                    if a.solver == "cgbfqp":
                        shift = out["info"].get("p_float32_shift", 0.0)
                        same = (agree and int(out["iters"]) == ref.get("n_cols")
                                and abs(ub - ref["opt"]) <= 1e-9 + shift)
                    out["lam0_check"] = bool(same)
                    if not same:
                        print("LAMBDA-0 CHECK FAILED: %s vs base %s (ub %.9f vs %.9f, cols %s vs %s)" % (
                            a.key, a.key[:-5], ub, ref["opt"], out["iters"], ref.get("n_cols")), flush=True)
                out["ref_action"] = "checked"
                out["ref_check"] = dict(existing_opt=ref["opt"], existing_lb=ref["lb"], existing_method=ref["method"],
                                        new_ub=ub, new_lb=lb, rel_diff_ub=(ub - ref["opt"]) / ref["opt"],
                                        intersection=[lo, hi], brackets_consistent=bool(agree))
        else:
            out["ref_action"] = "not certified (stop=%s, gap=%.2e)" % (out.get("stop"), gap)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "%s_%s%s_%s.json" % (a.criterion, a.solver, ("-" + tag) if tag else "", a.key))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    ub = out.get("ub")
    extra = ""
    if a.criterion == "cert":
        extra = " ref:%s" % out["ref_action"] + (" consistent=%s dUB=%+.2e" % (out["ref_check"]["brackets_consistent"], out["ref_check"]["rel_diff_ub"]) if out.get("ref_check") else "")
    elif ref is not None and ub is not None:
        if ref["opt"] > 0:
            extra = " rel=%+.3f%%" % (100 * (ub - ref["opt"]) / ref["opt"])
        else:
            extra = " abs=%.2e (OPT ~ 0, relative criterion undefined)" % ub
    print("[%s] %s %s: %s UB=%s LB=%s iters=%s wall=%.2fs cores=%.2f dev=%s%s -> %s"
          % (a.key, a.solver, a.criterion, out.get("stop"), "%.6f" % ub if ub is not None else "-",
             "%.6f" % out["lb"] if out.get("lb") is not None else "-", out.get("iters"), out.get("wall_s") or -1,
             out["avg_cores"], ("%.1f GB" % out["gpu_pool_peak_gb"]) if out.get("gpu_pool_peak_gb") is not None else "-",
             extra, os.path.basename(path)), flush=True)


if __name__ == "__main__":
    main()
