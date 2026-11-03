"""One (cell, solver, criterion) measurement in a fresh process.

    python validation/run_one.py --key n18_m39 --solver bm_admm [--criterion race] [--cap 900] --out DIR

Loads the instance from data/ (regenerated from its seeds and checked against
the fingerprint stored with its OPT), warms the solver up on a tiny instance
(kernel compilation / calibration are not timed), solves once (race: solver.solve;
cert: the RUM class, solver/rum_solver.py), and writes
DIR/<solver>_<key>.json with the result plus what the machine did: GPU name,
CPU model, logical cores, average cores busy (process CPU time / wall), peak
host memory, peak device-pool memory.
"""
import argparse
import json
import os
import platform
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))

import numpy as np                                     # noqa: E402
from gen_instances import get_instance, fingerprint    # noqa: E402
import solver                                          # noqa: E402
from solver.rum_solver import RUM                      # noqa: E402


def peak_rss_mb():
    try:
        if os.name == "nt":
            import ctypes
            import ctypes.wintypes as w

            class PMC(ctypes.Structure):
                _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = w.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(PMC), w.DWORD]
            pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
            psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.PeakWorkingSetSize / 2 ** 20
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def gpu_info():
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        return dict(gpu=cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode(),
                    gpu_mem_total_gb=cp.cuda.runtime.memGetInfo()[1] / 2 ** 30,
                    gpu_pool_peak_gb=cp.get_default_memory_pool().total_bytes() / 2 ** 30)
    except Exception:
        return dict(gpu=None)


_STOP = {0: "time_limit", -1: "max_iter", -2: "stalled", -3: "out_of_memory", -4: "interrupted", -5: "error"}


def certify(n, cd, method, rel_tol, abs_tol, max_time, method_args=None, verbose=False):
    """One certificate through the RUM class, as a record: the Result's fields plus the keys the
    result tables read (ub, lb, iters, wall_s, build_s, stop, converged; beta for BM-ADMM). method
    may be a driver name (cgbfqp, cgip, cgip_heur): the class accepts the old names."""
    res = RUM.from_rows(cd, n).calculate_distance(method=method, abs_tol=abs_tol, rel_tol=rel_tol,
                                                  max_time=max_time, method_args=method_args,
                                                  verbose=verbose)
    rec = res.to_dict()
    rec.update(ub=res.ub, lb=res.lb, iters=res.iterations, wall_s=res.time, build_s=res.time_setup,
               converged=res.exit_flag > 0,
               stop="gap_rel" if res.met_rel else "gap_abs" if res.met_abs else _STOP[res.exit_flag])
    if "beta" in res.method_args:              # BM-ADMM's penalty, as the old records had it
        rec["beta"] = res.method_args["beta"]
    if "pricing" in res.info:                  # CGBFQP's backend (mask, stream_f32, ...): the tables sort by it
        rec["pricing"] = res.info["pricing"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--solver", required=True, choices=["bm_admm", "cgbfqp", "flow_admm", "cgip", "cgip_heur"])
    ap.add_argument("--criterion", default="race", choices=["race", "cert"])
    ap.add_argument("--rel-tol", type=float, default=None)
    ap.add_argument("--max-cols", type=int, default=None,
                    help="cgbfqp: column cap (default: the solver default, 4000). The lambda study passes 20000 "
                         "so that only the wall cap can stop a run")
    ap.add_argument("--race-floor", type=float, default=0.0,
                    help="race: absolute floor on the stop target where OPT is ~0 (lambda = 1 cells): "
                         "target = max((1 + rel_tol) * OPT, floor)")
    ap.add_argument("--cap", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=None, help="Gurobi threads (cgip*); default: Gurobi's choice")
    ap.add_argument("--qp-backend", default="gurobi", choices=["gurobi", "native"],
                    help="cgip*: restricted-master QP solver (gurobi = as shipped; native = active-set Cholesky)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    refs = json.load(open(os.path.join(ROOT, "data", "opt_refs.json")))
    ref = refs[a.key]
    n, cat = int(a.key.split("_")[0][1:]), a.key.split("_", 1)[1]
    cd, mp = get_instance(n, cat)
    assert fingerprint(cd) == ref["cd_hash"], "instance does not match the stored OPT reference"
    opt = float(ref["opt"])
    opt_eff = opt
    if a.criterion == "race" and a.race_floor > 0 and 1.01 * opt < a.race_floor:
        opt_eff = a.race_floor / 1.01          # the solvers stop at UB <= 1.01 * opt_ref
    out = dict(key=a.key, n=n, cat=cat, mp=int(mp), density_pct=100.0 * mp / (n * (1 << (n - 1))),
               opt=opt, opt_lb=ref["lb"], solver=a.solver, criterion=a.criterion, cap_s=a.cap,
               race_target=(1.01 * opt_eff if a.criterion == "race" else None),
               race_floor_applied=(a.race_floor if opt_eff != opt else None),
               hostname=socket.gethostname(), cpu_model=platform.processor(), logical_cores=os.cpu_count(),
               python=platform.python_version(), when=time.strftime("%Y-%m-%d %H:%M:%S"))
    if a.solver in ("bm_admm", "cgbfqp"):
        solver.warmup((a.solver,))
    kw, args = {}, {}          # race keywords for solver.solve; certificate settings for the RUM class
    if a.criterion == "race" and opt_eff != opt and a.solver == "bm_admm":
        kw["ub_arm_immediately"] = True        # floor cell: arm the feasibility push from the first check
    if a.solver == "cgbfqp" and a.max_cols:
        kw["max_iters"] = args["max_iters"] = a.max_cols
    if a.solver in ("cgip", "cgip_heur"):
        kw["qp_backend"] = args["qp_backend"] = a.qp_backend
        if a.threads is not None:
            kw["threads"] = args["threads"] = a.threads
    c0, w0 = time.process_time(), time.perf_counter()
    try:
        if a.criterion == "cert":
            r = certify(n, cd, a.solver, rel_tol=1e-3 if a.rel_tol is None else a.rel_tol, abs_tol=0.0,
                        max_time=a.cap, method_args=args, verbose=a.verbose)
            r.update(abs_to_opt=r["ub"] - opt, rel_to_opt=(r["ub"] - opt) / opt if opt != 0 else None)
        else:
            r = solver.solve(n, cd, method=a.solver, criterion="race", opt_ref=opt_eff, rel_tol=a.rel_tol,
                             time_limit=a.cap, verbose=a.verbose, **kw)
        r = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v)) for k, v in r.items()}
        out.update(r)
    except Exception as e:                      # keep the row; the table shows "error"
        out.update(stop="error", error="%s: %s" % (type(e).__name__, str(e)[:500]), ub=None, iters=None, wall_s=None)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    out.update(process_cpu_s=cpu, process_wall_s=wall, avg_cores=cpu / max(wall, 1e-9),
               peak_host_rss_mb=peak_rss_mb(), **gpu_info())
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "%s_%s_%s.json" % (a.solver, a.criterion, a.key))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    ub = out.get("ub")
    print("[%s] %s %s: %s  UB=%s rel=%s iters=%s wall=%s build=%s cores=%.2f -> %s"
          % (a.key, a.solver, a.criterion, out.get("stop"),
             "%.6f" % ub if ub is not None else "-",
             ("%+.3f%%" % (100 * (ub - opt) / opt) if opt > 0 else "abs=%.2e" % ub) if ub is not None else "-",
             out.get("iters"), "%.2fs" % out["wall_s"] if out.get("wall_s") is not None else "-",
             "%.2fs" % out["build_s"] if out.get("build_s") is not None else "-",
             out["avg_cores"], os.path.basename(path)), flush=True)


if __name__ == "__main__":
    main()
