"""GPU checks: about two minutes on an RTX 5090. Run them alone, with nothing else on the GPU.

    python tests/test_gpu.py

A  the automatic choice on nine small instances (n = 10-17): the method the rule in solver/defaults.py picks,
   and a certified bracket (exit flag > 0, lb <= ub).
B  five of the paper's stored race records, one per GPU method and table, re-run through the replication drivers
   with the records' own settings: the iterations, the upper bound and the stop reason must be identical. The
   times are printed next to the stored ones; they depend on the machine.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import numpy as np                                   # noqa: E402
from gen_instances import get_instance               # noqa: E402
from run_paper_table import wait_for_idle_gpu        # noqa: E402
from solver import machine                           # noqa: E402
from solver.rum_solver import RUM                    # noqa: E402

PROF = machine.detect()
FAIL = []


def check(name, ok, detail=""):
    print("%-4s %s  %s" % ("ok" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAIL.append(name)


if not machine.can_use_gpu(PROF):
    print("No supported NVIDIA GPU:\n" + machine.summary(PROF))
    sys.exit(1)

# ---------------------------------------------------------------- A: the automatic choice
AUTO = ((10, "dense", "flow_cg"), (11, "m2", "flow_cg"), (11, "m10", "bm_admm"), (12, "m4", "bm_admm"),
        (13, "m2", "flow_cg"), (13, "m7", "bm_admm"), (15, "m6", "flow_cg"), (15, "m23", "bm_admm"),
        (17, "m410", "bm_admm"))
for n, cat, want in AUTO:
    cd = get_instance(n, cat)[0]
    rum = RUM(cd[:, 0].astype(np.int64), cd[:, 1].astype(np.int64), cd[:, 2], num_alternatives=n)
    r = rum.calculate_distance(machine=PROF, max_time=120)
    check("A n%d %s -> %s" % (n, cat, want), r.method == want and r.exit_flag > 0 and r.lb <= r.ub,
          "%.2f %% of the cells: %s, exit %d, [%.8f, %.8f], %.2f s"
          % (100 * len(cd) / (n << (n - 1)), r.method, r.exit_flag, r.lb, r.ub, r.time))

# ---------------------------------------------------------------- B: stored races
RACES = (("run_one", "replication/density_grid/results/cgbfqp_race_n18_sparse.json"),            # tab:density18, Flow-CG
         ("run_one", "replication/density_grid/results/bm_admm_race_n18_m771.json"),             # tab:density18, BM-ADMM
         ("run_one", "replication/density_grid/results/flow_admm_race_n18_sparse.json"),         # tab:density18, Flow-ADMM
         ("feas_one", "replication/feasibility_in_n/results/race_cgbfqp_n25_sparse.json"),       # tab:binary, Flow-CG
         ("feas_one", "replication/feasibility_in_density/results/race_bm_admm_n20_d1pct.json"))  # tab:dense, BM-ADMM
FIELDS = ("iters", "iters32", "iters64", "ub", "stop")
OUT = tempfile.mkdtemp(prefix="rum_test_gpu_")
for driver, rec in RACES:
    ref = json.load(open(os.path.join(ROOT, rec)))
    name, solver = os.path.basename(rec), ref["solver"]
    floor = ["--race-floor", "0.001"] if ref.get("race_floor_applied") else []
    if driver == "feas_one":
        cmd = [sys.executable, os.path.join(ROOT, "validation", "feasibility", "feas_one.py")]
        extra = ["--pricing", ref.get("pricing") or "mask", "--max-cols", "20000"] if solver == "cgbfqp" else []
    else:
        cmd = [sys.executable, os.path.join(ROOT, "validation", "run_one.py")]
        extra = ["--max-cols", "20000"] if solver == "cgbfqp" and floor else []
    cmd += ["--key", ref["key"], "--solver", solver, "--criterion", "race", "--cap", str(ref["cap_s"]),
            "--out", OUT] + floor + extra
    wait_for_idle_gpu(lambda m: print("   " + m, flush=True))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    path = os.path.join(OUT, name)
    if p.returncode != 0 or not os.path.exists(path):
        check("B %s" % name, False, "the run failed: %s" % (p.stderr or p.stdout)[-400:])
        continue
    new = json.load(open(path))
    diff = {f: (ref.get(f), new.get(f)) for f in FIELDS if ref.get(f) != new.get(f)}
    check("B %s" % name, not diff, "%s iterations, ub %.10f, stop %s; %.2f s (stored %.2f s)%s"
          % (new.get("iters"), new.get("ub"), new.get("stop"), new.get("wall_s"), ref.get("wall_s"),
             "" if not diff else "; differs: %s" % diff))

print("\n%d failed%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
sys.exit(1 if FAIL else 0)
