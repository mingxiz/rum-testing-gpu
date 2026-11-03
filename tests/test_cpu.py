"""CPU checks of the RUM class: input checks, the CPU methods (IP-CG, IP-H-CG, the exact QPs), certificates,
weights, the documented errors, and the race mode with stubbed GPU engines. Gurobi with one thread; the GPU is
switched off in the machine profile, so nothing runs on it. A few minutes.

    python tests/test_cpu.py
"""
import io
import contextlib
import math
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
import numpy as np                                    # noqa: E402
import scipy.sparse as sp                             # noqa: E402
from gen_instances import gen_pairwise, gen_medium, p_inside   # noqa: E402
from solver import defaults, machine                  # noqa: E402
from solver.rum_solver import RUM, choose_method      # noqa: E402
from solver.rum import RUM as OldRUM                  # noqa: E402

for m in ("ip_cg", "ip_cg_heur", "bm_qp", "flow_qp"):     # one Gurobi thread
    defaults.METHOD_ARGS[m]["threads"] = 1
PROF = machine.detect()
PROF["use_gpu"] = False                                    # never touch the GPU in this test
FAIL = []


def check(name, ok, detail=""):
    print("%-58s %s %s" % (name, "ok  " if ok else "FAIL", detail), flush=True)
    if not ok:
        FAIL.append(name)


def old_exact(cd, n, vcov=None):
    with contextlib.redirect_stdout(io.StringIO()):
        r = OldRUM(cd[:, 0].astype(np.uint64), cd[:, 1].astype(np.int64), cd[:, 2], num_alternatives=n)
        return float(r.calculate_distance(vcov=vcov, method="bm_qp")["distance"])


def raises(fn, exc, text):
    try:
        fn()
    except exc as e:
        return text in str(e), str(e)
    except Exception as e:                       # noqa: BLE001
        return False, "%s: %s" % (type(e).__name__, e)
    return False, "no error"


print(machine.summary(PROF), "\n", flush=True)

# ---- instances: pairwise n = 6, random menus n = 7, an inside point (distance 0) at n = 6
cases = []
cd, _ = gen_pairwise(6, phat_seed=3); cases.append(("pairwise n6", 6, cd))
cd, _ = gen_medium(7, mult=3, menu_seed=5, phat_seed=5); cases.append(("random menus n7", 7, cd))
cd, _ = gen_pairwise(6, phat_seed=4); cd = p_inside(cd); cases.append(("inside point n6", 6, cd))

for label, n, cd in cases:
    ref = old_exact(cd, n)
    rum = RUM(cd[:, 0].astype(np.int64), cd[:, 1].astype(np.int64), cd[:, 2], num_alternatives=n)
    for m in ("bm_qp", "flow_qp", "ip_cg", "ip_cg_heur"):
        r = rum.calculate_distance(method=m, machine=PROF)
        ok = r.exit_flag > 0 and r.lb <= ref + 1e-7 and r.ub >= ref - 1e-7 and \
            (r.ub - ref <= max(r.abs_tol, r.rel_tol * ref) + 1e-9)
        check("%s / %s" % (label, m), ok, "exit %d, [%.8f, %.8f], ref %.8f, %.3fs"
              % (r.exit_flag, r.lb, r.ub, ref, r.time))
        if m == "bm_qp":
            pf_ok = r.p_fit is not None and abs(np.linalg.norm(r.p_fit - cd[:, 2]) - r.distance) < 1e-9
            check("%s / p_fit reproduces the distance" % label, pf_ok)

# ---- singleton menus: rows with p = 0.9 add 0.1^2 each to dist^2
n, cd = cases[0][1], cases[0][2]
extra = np.array([[1 << 0, 0, 0.9], [1 << 3, 3, 0.75]])
cds = np.vstack([cd, extra])
ref_s = old_exact(cds, n)
rum_s = RUM(cds[:, 0].astype(np.int64), cds[:, 1].astype(np.int64), cds[:, 2], num_alternatives=n)
d0 = old_exact(cd, n)
for m in ("bm_qp", "ip_cg_heur"):
    r = rum_s.calculate_distance(method=m, machine=PROF)
    check("singletons / %s" % m, r.exit_flag > 0 and abs(r.distance - ref_s) < 2e-5,
          "%.8f vs old-class %.8f (sqrt(d^2 + 0.01 + 0.0625) = %.8f); p_fit singleton rows %s"
          % (r.distance, ref_s, math.sqrt(d0 ** 2 + 0.0725), r.p_fit[-2:]))
only = RUM([1, 2, 4], [0, 1, 2], [1.0, 0.5, 0.8])
r = only.calculate_distance(machine=PROF)
check("only singleton menus", r.exit_flag == 3 and abs(r.distance - math.sqrt(0.25 + 0.04)) < 1e-12,
      "distance %.6f, method %s" % (r.distance, r.method))

# ---- weighted norm: a diagonal vcov, against the old class
rng = np.random.RandomState(0)
w = rng.uniform(0.5, 2.0, size=len(cd))
ref_w = old_exact(cd, n, vcov=sp.diags(w).tocsr())
rum = RUM(cd[:, 0].astype(np.int64), cd[:, 1].astype(np.int64), cd[:, 2], num_alternatives=n)
for m in ("bm_qp", "flow_qp", "ip_cg"):
    r = rum.calculate_distance(method=m, vcov=sp.diags(w).tocsr(), machine=PROF)
    check("diagonal vcov / %s" % m, r.exit_flag > 0 and abs(r.distance - ref_w) < 2e-5
          and r.lb <= ref_w + 1e-7 and r.ub >= ref_w - 1e-7,
          "exit %d, [%.8f, %.8f] vs old class %.8f" % (r.exit_flag, r.lb, r.ub, ref_w))
r = rum.calculate_distance(method="bm_qp", vcov=np.eye(len(cd)), machine=PROF)
r0 = rum.calculate_distance(method="bm_qp", machine=PROF)
check("identity vcov = no weights", (r.lb, r.ub) == (r0.lb, r0.ub), "[%.10f, %.10f]" % (r.lb, r.ub))
V = np.eye(len(cds)); V[-1, 0] = V[0, -1] = 0.1
ok, msg = raises(lambda: rum_s.calculate_distance(method="bm_qp", vcov=V, machine=PROF), ValueError, "couples")
check("vcov coupling a singleton row is refused", ok, msg)

# ---- pre-run errors
rum = RUM(cd[:, 0].astype(np.int64), cd[:, 1].astype(np.int64), cd[:, 2], num_alternatives=n)
for label, fn, exc, text in [
        ("named GPU method without a GPU", lambda: rum.calculate_distance(method="bm_admm", machine=PROF), RuntimeError, "needs a supported NVIDIA GPU"),
        ("GPU method with weights", lambda: rum.calculate_distance(method="flow_cg", vcov=sp.diags(w).tocsr(), machine=PROF), RuntimeError, "unweighted"),
        ("unknown method_args name", lambda: rum.calculate_distance(method="ip_cg", method_args={"thread": 1}, machine=PROF), ValueError, "unknown method_args"),
        ("method_args without a method", lambda: rum.calculate_distance(method_args={"threads": 1}, machine=PROF), ValueError, "needs a named method"),
        ("unknown method", lambda: rum.calculate_distance(method="newton", machine=PROF), ValueError, "unknown method"),
        ("named flow_admm without a GPU", lambda: rum.calculate_distance(method="flow_admm", machine=PROF), RuntimeError, "flow_admm needs a supported NVIDIA GPU"),
        ("exact QP at large n", lambda: RUM([2 ** 20 - 1, 2 ** 20 - 1], [0, 1], [0.5, 0.5]).calculate_distance(method="bm_qp", machine=PROF), RuntimeError, "meant for n <="),
        ("alternative not in its menu", lambda: RUM([3, 5], [0, 1], [0.5, 0.5]), ValueError, "not in its menu"),
        ("duplicate row", lambda: RUM([3, 3], [0, 0], [0.5, 0.5]), ValueError, "more than once"),
        ("probability outside [0, 1]", lambda: RUM([3, 3], [0, 1], [0.5, 1.5]), ValueError, "outside [0, 1]"),
        ("wrong length p", lambda: rum.calculate_distance(p=np.ones(3), machine=PROF), ValueError, "expected"),
        ("negative tolerance", lambda: rum.calculate_distance(abs_tol=-1, machine=PROF), ValueError, ">= 0"),
        ("num_alternatives too small", lambda: RUM([7], [2], [1.0], num_alternatives=2), ValueError, "num_alternatives")]:
    ok, msg = raises(fn, exc, text)
    check("error: " + label, ok, msg[:90])
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    RUM([3, 3], [0, 1], [0.7, 0.6])
check("warning: menu probabilities summing above 1", any("summing" in str(x.message) for x in wl))

# ---- exit flags other than success
cd8, _ = gen_medium(8, mult=4, menu_seed=2, phat_seed=2)
rum8 = RUM(cd8[:, 0].astype(np.int64), cd8[:, 1].astype(np.int64), cd8[:, 2], num_alternatives=8)
r = rum8.calculate_distance(method="ip_cg", method_args={"max_iters": 2}, abs_tol=0, rel_tol=0, machine=PROF)
check("iteration limit -> exit -1", r.exit_flag == -1 and r.lb <= r.ub, "exit %d [%.6f, %.6f] %s" % (r.exit_flag, r.lb, r.ub, r.message))
r = rum8.calculate_distance(method="ip_cg_heur", abs_tol=0, rel_tol=0, max_time=0.05, machine=PROF)
check("time limit -> exit 0 (within one column of max_time)", r.exit_flag == 0 and r.time < 5,
      "exit %d after %.3fs, [%.6f, %.6f]" % (r.exit_flag, r.time, r.lb, r.ub))
ref8 = old_exact(cd8, 8)
for m in ("bm_qp", "flow_qp"):
    r = rum8.calculate_distance(method=m, abs_tol=0, rel_tol=0, machine=PROF)
    want = ("loosen abs_tol or rel_tol, or use a method that iterates to them" if m == "flow_qp"
            else "the method stopped on its own test, but the bracket does not meet the tolerances")
    check("%s with zero tolerances -> exit -2 and its message" % m,
          r.exit_flag == -2 and r.abs_gap > 0 and r.lb <= ref8 + 1e-7 and r.ub >= ref8 - 1e-7
          and r.message.endswith(want) and ("certifies only" in r.message) == (m == "flow_qp"),
          "exit %d: %s" % (r.exit_flag, r.message))

# ---- exact QPs: Gurobi's point certified (ub = pushed point, lb = Frank-Wolfe at Gurobi's point)
from solver.exact_qp import flow_bm_qp                # noqa: E402
cdp = cases[0][2]                                     # pairwise n = 6: every menu fully observed
rum_p = RUM(cdp[:, 0].astype(np.int64), cdp[:, 1].astype(np.int64), cdp[:, 2], num_alternatives=6)
for m in ("bm_qp", "flow_qp"):
    r = rum_p.calculate_distance(method=m, machine=PROF)
    sums = np.bincount(np.unique(cdp[:, 0], return_inverse=True)[1], weights=r.p_fit)
    check("%s: lb < ub, p_fit an exact RUM point (pair sums = 1)" % m,
          r.exit_flag == 3 and r.lb < r.ub and np.abs(sums - 1).max() < 1e-12
          and abs(np.linalg.norm(r.p_fit - cdp[:, 2]) - r.ub) < 1e-12,
          "[%.10f, %.10f], max |sum - 1| %.1e" % (r.lb, r.ub, np.abs(sums - 1).max()))
    check("%s: Gurobi's own value reported in info" % m,
          math.isfinite(r.info.get("gurobi_distance", math.nan)),
          "%.10f" % r.info.get("gurobi_distance", math.nan))
for m, fn in (("flow_qp", "solve_arc_flow_qp"), ("bm_qp", "solve_BM_qp")):
    real = getattr(flow_bm_qp, fn)

    def poor(*a, _real=real, _m=m, **k):              # Gurobi stopped early: a point off the optimum
        out = _real(*a, **k)
        noise = np.random.RandomState(1)
        key = "r" if _m == "flow_qp" else "p"
        out[key] = out[key] + 0.02 * noise.rand(out[key].size)
        out["p_pred"] = out["p_pred"] + 0.02 * noise.randn(out["p_pred"].size)
        out["status"] = 9
        return out
    setattr(flow_bm_qp, fn, poor)
    try:
        r = rum8.calculate_distance(method=m, machine=PROF)
    finally:
        setattr(flow_bm_qp, fn, real)
    check("%s at Gurobi's time limit, a poor point -> exit 0, valid bracket" % m,
          r.exit_flag == 0 and r.lb <= ref8 + 1e-7 and r.ub >= ref8 - 1e-7 and r.abs_gap > 1e-5,
          "exit %d, [%.6f, %.6f], ref %.6f" % (r.exit_flag, r.lb, r.ub, ref8))
r = rum8.calculate_distance(method="ip_cg", abs_tol=1e-3, rel_tol=0, machine=PROF)
check("abs_tol only -> exit 1 or 3", r.exit_flag in (1, 3), "exit %d (%s)" % (r.exit_flag, r.status))

# ---- CGIP at distance 0 (fix 7): a mixture of 5 rankings on random menus, n = 7
from solver.ip_cg.cgip import CGIPBuild              # noqa: E402
cdq = np.asarray(gen_medium(7, mult=3)[0], dtype=np.float64)
rq = RUM(cdq[:, 0].astype(np.int64), cdq[:, 1].astype(np.int64), cdq[:, 2], num_alternatives=7)
bq = CGIPBuild(rq._geom)
gq = np.random.default_rng(1)
pq = np.zeros(len(cdq))
for wq in gq.dirichlet(np.ones(5)):
    pq += wq * bq.column(bq._pos(list(gq.permutation(7))))
for m in ("ip_cg", "ip_cg_heur"):
    r = rq.calculate_distance(method=m, p=pq, abs_tol=1e-6, rel_tol=0, machine=PROF)
    check("distance 0 / %s (native master) -> certified" % m, r.exit_flag in (1, 3) and r.ub < 1e-6,
          "exit %d, [%.2e, %.2e], %d columns, master kkt %.1e" % (r.exit_flag, r.lb, r.ub, r.info["columns"],
                                                                  r.info["master_kkt"]))
    r = rq.calculate_distance(method=m, p=pq, abs_tol=1e-6, rel_tol=0, machine=PROF,
                              method_args={"qp_backend": "gurobi"})
    check("distance 0 / %s (Gurobi master) -> honest stall" % m, r.exit_flag in (1, 3) or
          (r.exit_flag == -2 and "already in the master" in r.message),
          "exit %d, [%.2e, %.2e], %s" % (r.exit_flag, r.lb, r.ub, r.message))

# ---- automatic choice without a GPU
for nn, want in ((6, "flow_qp"), (10, "flow_qp"), (11, "ip_cg_heur"), (12, "ip_cg_heur")):
    m, why = choose_method(nn, 30, False, PROF)
    check("auto, no GPU, n = %d -> %s" % (nn, want), m == want, why)
m, why = choose_method(20, 10 ** 6, True, dict(PROF, use_gpu=True))
check("auto, weighted norm -> a Gurobi method", m in ("ip_cg_heur", "flow_qp"), why)
m, why = choose_method(8, 100, True, dict(PROF, use_gpu=True))
check("auto, weighted norm, n = 8, with a GPU -> flow_qp", m == "flow_qp", why)

# ---- automatic choice with a GPU: a stub profile and 30 GB free (no CUDA call is made)
GPU_PROF = dict(PROF, use_gpu=True, gpu=dict(name="stub GPU", supported=True, reason="stub"))
real_free = machine.gpu_free_bytes
machine.gpu_free_bytes = lambda: 30 * 2 ** 30
try:
    for nn, dens, want in ((10, 0.0176, "flow_cg"), (10, 0.99, "flow_cg"), (11, 0.0199, "flow_cg"),
                           (11, 0.0201, "bm_admm"), (12, 0.10, "bm_admm"), (13, 0.0199, "flow_cg"),
                           (13, 0.0201, "bm_admm"), (14, 0.0099, "flow_cg"), (14, 0.0101, "bm_admm"),
                           (15, 0.10, "bm_admm"), (17, 0.0099, "flow_cg"),
                           (17, 0.0101, "bm_admm"), (18, 0.0049, "flow_cg"), (18, 0.0051, "bm_admm"),
                           (20, 0.01, "bm_admm"), (23, 0.01, "bm_admm"), (24, 0.01, "flow_cg")):
        mp = int(round(dens * (nn << (nn - 1))))
        m, why = choose_method(nn, mp, False, GPU_PROF)
        check("auto, GPU, n = %d at %.2f%% -> %s" % (nn, 100 * dens, want), m == want, why)
finally:
    machine.gpu_free_bytes = real_free

# ---- storage formats (dtype): what the planners choose on a stub GPU (no CUDA call is made)
import solver.rum_solver as RS                                # noqa: E402
_PLANNED = {RS._plan_flow_cg: "flow_cg", RS._plan_flow_admm: "flow_admm", RS._plan_bm_admm: "bm_admm"}


def plan(fn, nn, free=30, **kw):
    machine.gpu_free_bytes = lambda: free * 2 ** 30
    return fn(nn, 500, False, GPU_PROF, dict(defaults.METHOD_ARGS[_PLANNED[fn]], **kw))


try:
    for nn, free, kw, want in (
            (20, 30, {}, ("mask", "float32", "mask")),
            (33, 30, {}, ("stream", "float32", "stream_f32")),
            (35, 30, {}, ("stream", "uint24", "stream_u24")),
            (35, 28, {}, ("stream", "uint16", "stream_u16")),
            (20, 30, {"dtype": "uint16"}, ("stream", "uint16", "stream_u16")),
            (20, 30, {"dtype": "float32", "pricing": "stream"}, ("stream", "float32", "stream_f32")),
            (20, 30, {"pricing": "stream_u24"}, ("stream", "uint24", "stream_u24")),
            (20, 30, {"pricing": "stream_f32", "dtype": "float32"}, ("stream", "float32", "stream_f32"))):
        a = plan(RS._plan_flow_cg, nn, free, **kw)
        got = (a["pricing"], a["dtype"], a["_engine"])
        check("dtype: flow_cg n = %d, %d GB free, %s -> %s" % (nn, free, kw or "auto", want[2]), got == want, str(got))
    for label, fn, nn, kw, exc, text in (
            ("dtype: flow_cg pricing and dtype contradict", RS._plan_flow_cg, 20,
             {"pricing": "stream_u16", "dtype": "float32"}, ValueError, "contradicts"),
            ("dtype: flow_cg mask with uint16", RS._plan_flow_cg, 20, {"pricing": "mask", "dtype": "uint16"},
             ValueError, "float32 only"),
            ("dtype: flow_cg float64 is not a pricing dtype", RS._plan_flow_cg, 20, {"dtype": "float64"}, ValueError,
             "dtype must be"),
            ("dtype: flow_cg unknown pricing", RS._plan_flow_cg, 20, {"pricing": "lattice"}, ValueError,
             "pricing must be"),
            ("dtype: flow_cg uint16 at n = 36 does not fit", RS._plan_flow_cg, 36, {"dtype": "uint16"}, RuntimeError,
             "stream uint16 34.5 GB"),
            ("dtype: flow_cg float32 at n = 35 does not fit", RS._plan_flow_cg, 35, {"dtype": "float32"}, RuntimeError,
             "stream float32 34.2 GB"),
            ("dtype: flow_admm float64 refused", RS._plan_flow_admm, 18, {"dtype": "float64"}, ValueError,
             "kernels are float32"),
            ("dtype: bm_admm auto at n = 24 -> error naming float32", RS._plan_bm_admm, 24, {}, RuntimeError,
             "dtype='float32' needs 16.9 GB")):
        ok, msg = raises(lambda: plan(fn, nn, **kw), exc, text)
        check(label, ok, msg[:100])
    check("dtype: flow_admm auto -> float32", plan(RS._plan_flow_admm, 18)["dtype"] == "float32")
    a = plan(RS._plan_bm_admm, 24, dtype="float32")
    check("dtype: bm_admm float32 at n = 24 fits", a["dtype"] == "float32", "%.1f GB" % (a["_gpu_bytes"] / 2 ** 30))
    check("dtype: bm_admm auto at n = 23 -> float64", plan(RS._plan_bm_admm, 23)["dtype"] == "float64")
finally:
    machine.gpu_free_bytes = real_free
r = rum.calculate_distance(machine=PROF)
check("auto end to end (n = 6 -> flow_qp)", r.method == "flow_qp" and r.exit_flag > 0, r.auto_reason)

# ---- RUM.from_rows, Result.to_dict and the trace layout (iteration, seconds, ub, lb)
import json                                                   # noqa: E402
ra = rum8.calculate_distance(method="ip_cg", machine=PROF)
rb = RUM.from_rows(cd8, 8).calculate_distance(method="ip_cg", machine=PROF)
check("RUM.from_rows gives the same result", (ra.lb, ra.ub, ra.iterations) == (rb.lb, rb.ub, rb.iterations),
      "[%.8f, %.8f] %d columns" % (rb.lb, rb.ub, rb.iterations))
tr = rb.trace
ok = (len(tr) == rb.iterations and all(len(row) == 4 for row in tr) and [row[0] for row in tr] == list(range(1, len(tr) + 1))
      and all(t1 <= t2 for (_, t1, _, _), (_, t2, _, _) in zip(tr, tr[1:])) and abs(tr[-1][2] - rb.ub) < 1e-12
      and all(row[3] <= row[2] + 1e-12 for row in tr))
check("trace rows are (iteration, seconds, ub, lb)", ok, str(tr[-1]))
d = rb.to_dict()
ok = "p_fit" not in d and json.loads(json.dumps(d))["ub"] == rb.ub and len(rb.to_dict(p_fit=True)["p_fit"]) == len(cd8)
check("Result.to_dict is JSON-ready (p_fit only when asked)", ok, ", ".join(sorted(d)[:6]) + ", ...")

# ---- seeds and bootstrap-style reuse
r1 = rum8.calculate_distance(method="ip_cg_heur", seed=None, machine=PROF)
check("seed=None draws and reports a seed", isinstance(r1.seed, int), "seed %s" % r1.seed)
t = time.perf_counter()
pb = np.clip(cd8[:, 2] + 0.01 * np.random.RandomState(1).randn(len(cd8)), 0, 1)
r2 = rum8.calculate_distance(method="bm_qp", p=pb, machine=PROF)
r3 = rum8.calculate_distance(method="bm_qp", p=pb, machine=PROF)
check("second call reuses the Gurobi build", r3.time_setup < max(0.05, 0.5 * r2.time_setup) or r2.time_setup < 0.05,
      "setup %.3fs then %.3fs" % (r2.time_setup, r3.time_setup))

check("a certificate reports mode 'certificate' and no opt", r3.mode == "certificate" and r3.opt is None, r3.mode)

# ---- race mode (calculate_distance with opt): the argument rules and the call made to solver.solve.
#      solver.solve and solver.warmup are stubbed, so no CUDA call is made; race_via_class_gpu.py
#      checks the real races against solver.solve on the GPU.
import solver                                                 # noqa: E402
CALLS, STUB_STOP = [], ["ub_target"]


def stub_solve(n, cd, method=None, criterion="race", opt_ref=None, rel_tol=None, time_limit=None,
               verbose=False, **kw):
    CALLS.append(dict(n=n, mp=len(cd), method=method, opt_ref=opt_ref, rel_tol=rel_tol, time_limit=time_limit, kw=kw))
    ub = (1 + rel_tol) * opt_ref * 0.999
    return dict(method=method, criterion="race", ub=ub, lb=None, iters=77, wall_s=0.5, build_s=0.1, stop=STUB_STOP[0],
                converged=STUB_STOP[0] in ("ub_target", "phi_ub", "ub_rel"),
                trace=[[50, 0.2, 1.0, None], [77, 0.5, 1.0, ub]] if method == "bm_admm" else [])


cd_r, _ = gen_pairwise(8, phat_seed=7)
rum_r = RUM.from_rows(cd_r, 8)
opt = 1.25
real = (solver.solve, solver.warmup, machine.gpu_free_bytes)
solver.solve, solver.warmup, machine.gpu_free_bytes = stub_solve, (lambda methods=(): None), (lambda: 30 * 2 ** 30)
try:
    for label, fn, exc, text in (
            ("race: abs_tol with opt", lambda: rum_r.calculate_distance(method="bm_admm", opt=opt, abs_tol=1e-5, machine=GPU_PROF), ValueError, "do not apply"),
            ("race: rel_tol with opt", lambda: rum_r.calculate_distance(method="bm_admm", opt=opt, rel_tol=1e-3, machine=GPU_PROF), ValueError, "do not apply"),
            ("race: opt_floor without opt", lambda: rum_r.calculate_distance(method="bm_admm", opt_floor=1e-3, machine=GPU_PROF), ValueError, "need opt"),
            ("race: opt = 0 without a floor", lambda: rum_r.calculate_distance(method="bm_admm", opt=0.0, machine=GPU_PROF), ValueError, "opt_floor > 0"),
            ("race: negative opt", lambda: rum_r.calculate_distance(method="bm_admm", opt=-1.0, machine=GPU_PROF), ValueError, ">= 0"),
            ("race: weighted norm", lambda: rum_r.calculate_distance(method="bm_admm", opt=opt, vcov=sp.diags(np.full(len(cd_r), 2.0)).tocsr(), machine=GPU_PROF), ValueError, "plain L2"),
            ("race: an exact QP has no race", lambda: rum_r.calculate_distance(method="bm_qp", opt=opt, machine=GPU_PROF), ValueError, "has no race"),
            ("race: CGIP has no race here", lambda: rum_r.calculate_distance(method="cgip", opt=opt, machine=GPU_PROF), ValueError, "has no race"),
            ("race: a certificate-only method_arg", lambda: rum_r.calculate_distance(method="bm_admm", opt=opt, method_args={"lb_every": 2}, machine=GPU_PROF), ValueError, "unknown race method_args"),
            ("race: a GPU method without a GPU", lambda: rum_r.calculate_distance(method="flow_cg", opt=opt, machine=PROF), RuntimeError, "needs a supported NVIDIA GPU"),
            ("race: automatic choice without a GPU (flow_qp)", lambda: rum_r.calculate_distance(opt=opt, machine=PROF), ValueError, "has no race")):
        ok, msg = raises(fn, exc, text)
        check(label, ok, msg[:70])
    r = rum_r.calculate_distance(method="bm_admm", opt=opt, machine=GPU_PROF)
    c = CALLS[-1]
    check("race: the paper's call (opt_ref = opt, rel_tol 0.01, no keywords)",
          c["opt_ref"] == opt and c["rel_tol"] == 0.01 and c["kw"] == {} and c["method"] == "bm_admm" and c["mp"] == len(cd_r),
          "opt_ref %r rel_tol %r kw %s" % (c["opt_ref"], c["rel_tol"], c["kw"]))
    ub_want = 1.01 * opt * 0.999
    check("race: Result (exit 4, lb nan, ub, target, ub - opt, race time)",
          r.exit_flag == 4 and r.status == "opt target reached" and r.mode == "race" and math.isnan(r.lb)
          and r.ub == ub_want and r.distance == ub_want and r.target == 1.01 * opt and r.abs_to_opt == ub_want - opt
          and r.rel_to_opt == (ub_want - opt) / opt and r.time_solve == 0.5 and r.iterations == 77 and r.p_fit is None
          and math.isnan(r.abs_tol) and math.isnan(r.rel_tol) and not r.met_abs and not r.met_rel,
          "ub %.6g target %.6g" % (r.ub, r.target))
    check("race: trace keeps the checks with a pushed ub, lb nan",
          len(r.trace) == 1 and r.trace[0][0] == 77 and r.trace[0][2] == ub_want and math.isnan(r.trace[0][3]), str(r.trace))
    d = json.loads(json.dumps(r.to_dict()))
    check("race: to_dict is JSON-ready and printing works", d["mode"] == "race" and math.isnan(d["lb"])
          and "race to opt" in str(r) and "abs_to_opt" not in r.info, ", ".join(k for k in d if k in ("opt", "target", "mode")))
    r = rum_r.calculate_distance(method="bm_admm", opt=0.0, opt_floor=1e-3, machine=GPU_PROF)
    c = CALLS[-1]
    check("race: floor cell (opt_ref = floor / 1.01, push armed at once)",
          c["opt_ref"] == 1e-3 / 1.01 and c["kw"] == {"ub_arm_immediately": True} and r.target == 1e-3
          and r.rel_to_opt is None and r.exit_flag == 4, "opt_ref %r kw %s" % (c["opt_ref"], c["kw"]))
    r = rum_r.calculate_distance(method="cgbfqp", opt=opt, machine=GPU_PROF)
    c = CALLS[-1]
    check("race: flow_cg (old name) gets mask pricing and seed 0",
          r.method == "flow_cg" and c["kw"] == {"pricing": "mask", "seed": 0} and r.seed == 0
          and r.method_args == {"pricing": "mask"}, str(c["kw"]))
    r = rum_r.calculate_distance(method="flow_cg", opt=opt, method_args={"max_iters": 20000, "pricing": "stream_f32"},
                                 machine=GPU_PROF)
    c = CALLS[-1]
    check("race: method_args reach the race unchanged",
          c["kw"] == {"max_iters": 20000, "pricing": "stream_f32", "seed": 0}, str(c["kw"]))
    cd_s = np.vstack([cd_r, [1.0, 0.0, 0.9]])            # singleton menu {0} at p = 0.9: dist^2 + (1 - 0.9)^2
    const = (1 - 0.9) ** 2
    r = RUM.from_rows(cd_s, 8).calculate_distance(method="bm_admm", opt=opt, machine=GPU_PROF)
    c = CALLS[-1]
    want = math.sqrt((1.01 * opt) ** 2 - const) / 1.01
    ub_rest = 1.01 * want * 0.999
    check("race: singleton rows move the target onto the other rows",
          c["mp"] == len(cd_r) and abs(c["opt_ref"] - want) < 1e-15 and abs(r.ub - math.sqrt(ub_rest ** 2 + const)) < 1e-15
          and r.ub <= r.target, "opt_ref %.12g (want %.12g), ub %.12g, target %.12g" % (c["opt_ref"], want, r.ub, r.target))
    for stop, m, flag in (("time_limit", "bm_admm", 0), ("max_iter", "bm_admm", -1), ("infeasible", "flow_admm", -3),
                          ("rc", "flow_cg", -2), ("something else", "bm_admm", -2)):
        STUB_STOP[0] = stop
        r = rum_r.calculate_distance(method=m, opt=opt, machine=GPU_PROF)
        check("race: %s stop '%s' -> exit %d" % (m, stop, flag), r.exit_flag == flag, r.message[:70])
    STUB_STOP[0] = "ub_target"
    r = rum_r.calculate_distance(method="bm_admm", opt=opt, max_time=1e-9, machine=GPU_PROF)
    check("race: max_time spent before the race -> exit 0", r.exit_flag == 0 and math.isinf(r.ub), r.message)
finally:
    solver.solve, solver.warmup, machine.gpu_free_bytes = real

print("\n%s" % r3)
print("\n%d checks failed: %s" % (len(FAIL), FAIL))

sys.exit(1 if FAIL else 0)
