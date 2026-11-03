"""The lambda study: every timed cell of the paper at lambda = 0, 0.5, 1.

p(lambda) = lambda * p_inside + (1 - lambda) * p_hat. Stopping rule, one rule everywhere:
the first check with UB <= max(1.01 * OPT, 1e-3). For lambda = 0 and 0.5 the point is outside
the polytope and the relative term binds ((UB - OPT)/OPT <= 1 %); for lambda = 1 the point is
inside, OPT = 0, and the absolute term binds (UB <= 1e-3).

    task 1  pairwise design, CGBFQP-GPU, n = 25, 27, 30, 32, 35
    task 2  n = 20 density axis, BM-ADMM, density 0.5, 1, 5, 10, 100 %
    task 3  the n = 18, 19, 20 grid, BM-ADMM / CGBFQP-GPU / Flow-ADMM

    python validation/run_lambda_study.py --task 1 2 3          # certs, then races, then tables
    python validation/run_lambda_study.py --task 3 --phase race
    python validation/run_lambda_study.py --tables

Protocol: reference optima first (untimed, not gated), then races - one fresh process per
(cell, solver), the GPU idle before each (validation/run_paper_table.wait_for_idle_gpu),
strictly sequential. Cells that already have a record are skipped, so the run resumes.
Results: validation/results/lambda_study/, copied into the replication stores; tables by
validation/make_lambda_tables.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
from run_paper_table import wait_for_idle_gpu      # noqa: E402
from result_store import find                       # noqa: E402
from gen_instances import CELLS                     # noqa: E402

PY = sys.executable
OUT = os.path.join(ROOT, "validation", "results", "lambda_study")
REFS = os.path.join(ROOT, "data", "opt_refs.json")
FEAS_ONE = os.path.join(HERE, "feasibility", "feas_one.py")
RUN_ONE = os.path.join(HERE, "run_one.py")
FLOOR = 1e-3
STORE = {1: os.path.join(ROOT, "replication", "feasibility_in_n", "results"),
         2: os.path.join(ROOT, "replication", "feasibility_in_density", "results"),
         3: os.path.join(ROOT, "replication", "density_grid", "results")}
PAIRWISE_N = [25, 27, 30, 32, 35]
DENSITY_CATS = ["m138", "d1pct", "m1380", "m2759", "dense"]      # 0.5, 1, 5, 10, 100 %
GRID_SOLVERS = ["bm_admm", "cgbfqp", "flow_admm"]


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def refs():
    return json.load(open(REFS))


def pricing_for(n):
    return "mask" if n <= 32 else ("stream_f32" if n <= 34 else "stream_u24")


def run(cmd, timeout):
    log("$ " + " ".join(os.path.basename(c) if c.endswith(".py") else c for c in cmd[1:]))
    try:
        subprocess.run(cmd, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        log("   killed after %.0f s" % timeout)


# ------------------------------------------------------------------ references
def cert(key, solver, cap, n, abs_tol=None, cert_every=200):
    if key in refs():
        return True
    cmd = [PY, FEAS_ONE, "--key", key, "--solver", solver, "--criterion", "cert", "--cap", str(cap),
           "--out", OUT, "--cert-every", str(cert_every)]
    if solver == "cgbfqp":
        cmd += ["--pricing", pricing_for(n), "--max-cols", "20000"]
    if abs_tol:
        cmd += ["--abs-tol", str(abs_tol)]
    run(cmd, cap + 1800)
    return key in refs()


def cert_with_fallback(key, n, cap):
    """CGBFQP first (it certified every lambda = 0 grid reference), BM-ADMM if it caps."""
    if cert(key, "cgbfqp", cap, n):
        return True
    log("   %s: CGBFQP did not certify within %.0f s -> BM-ADMM" % (key, cap))
    return cert(key, "bm_admm", cap, n)


def lambda1_refs():
    run([PY, os.path.join(HERE, "write_lambda1_refs.py")], 3600)


# ------------------------------------------------------------------ races
def race_feas(key, solver, n, cap, gate=True):
    """feas_one race (tasks 1 and 2): file race_<solver>[-<pricing>]_<key>.json."""
    pr = pricing_for(n) if solver == "cgbfqp" else None
    name = "race_%s%s_%s.json" % (solver, ("-" + pr) if (pr and pr != "mask") else "", key)
    if os.path.exists(os.path.join(OUT, name)):
        log("skip (exists) " + name)
        return
    if key not in refs():
        log("skip (no OPT reference) " + key)
        return
    cmd = [PY, FEAS_ONE, "--key", key, "--solver", solver, "--criterion", "race", "--cap", str(cap),
           "--race-floor", str(FLOOR), "--out", OUT]
    if solver == "cgbfqp":
        cmd += ["--pricing", pr, "--max-cols", "20000"]
    if gate:
        wait_for_idle_gpu(lambda m: log("   " + m))
    run(cmd, cap + 1800)


def race_grid(key, solver, cap, gate=True):
    """run_one race (task 3): file <solver>_race_<key>.json."""
    name = "%s_race_%s.json" % (solver, key)
    if os.path.exists(os.path.join(OUT, name)):
        log("skip (exists) " + name)
        return
    if key not in refs():
        log("skip (no OPT reference) " + key)
        return
    cmd = [PY, RUN_ONE, "--key", key, "--solver", solver, "--criterion", "race", "--cap", str(cap),
           "--race-floor", str(FLOOR), "--out", OUT]
    if solver == "cgbfqp":
        cmd += ["--max-cols", "20000"]        # only the wall cap may stop a lambda race
    if gate:
        wait_for_idle_gpu(lambda m: log("   " + m))
    run(cmd, cap + 1800)


# ------------------------------------------------------------------ tasks
def task1(phase):
    if phase in ("cert", "all"):
        log("== task 1: references (n = 27)")
        subprocess.run([PY, os.path.join(ROOT, "data", "gen_instances.py"), "--feas", "--no-csv"], check=False, cwd=os.path.join(ROOT, "data"))
        cert("n27_sparse", "cgbfqp", 7200, 27)
        cert("n27_sparse_lam5", "cgbfqp", 7200, 27)
        lambda1_refs()
        subprocess.run([PY, os.path.join(ROOT, "data", "gen_instances.py"), "--path", "--no-csv", "--path-n"] + [str(n) for n in [25, 27, 30, 32, 33, 34, 35]],
                       check=False, cwd=os.path.join(ROOT, "data"))
    if phase in ("race", "all"):
        log("== task 1: races (n = 27 at lambda 0, 0.5, 1; lambda = 1 re-measured under the 1e-3 floor at every n)")
        for key in ("n27_sparse", "n27_sparse_lam5", "n27_sparse_lam10"):
            race_feas(key, "cgbfqp", 27, 14400)
        for n in PAIRWISE_N:
            race_feas("n%d_sparse_lam10" % n, "cgbfqp", n, 14400)
    store(1)


def task2(phase):
    if phase in ("cert", "all"):
        log("== task 2: references (0.5 % cell at lambda 0.5; lambda = 1 analytic)")
        cert_with_fallback("n20_m138_lam5", 20, 7200)
        lambda1_refs()
    if phase in ("race", "all"):
        log("== task 2: BM-ADMM races, five density cells x lambda 0.5, 1")
        for k in (5, 10):
            for cat in DENSITY_CATS:
                race_feas("n20_%s_lam%d" % (cat, k), "bm_admm", 20, 3600)
    store(2)


def task3(phase):
    if phase in ("cert", "all"):
        log("== task 3: references (grid cells at lambda 0.5; lambda = 1 analytic)")
        for n in (18, 19, 20):
            for cat in CELLS[n]:
                cert_with_fallback("n%d_%s_lam5" % (n, cat), n, 3600)
        lambda1_refs()
        subprocess.run([PY, os.path.join(ROOT, "data", "gen_instances.py"), "--path-grid", "--no-csv"], check=False, cwd=os.path.join(ROOT, "data"))
    if phase in ("race", "all"):
        log("== task 3: races, 30 cells x 3 GPU solvers x lambda 0.5, 1")
        for k in (5, 10):
            for n in (18, 19, 20):
                for cat in CELLS[n]:
                    for s in GRID_SOLVERS:
                        race_grid("n%d_%s_lam%d" % (n, cat, k), s, 900)
    store(3)


def store(task):
    """Copy this task's records into its replication store (never overwrite a newer working copy
    with an older stored one: the working directory is the source of truth here)."""
    os.makedirs(STORE[task], exist_ok=True)
    convex = os.path.join(ROOT, "replication", "convex_path", "results")
    n = 0
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json"):
            continue
        if task == 1:
            ok = "_n27_sparse" in f or ("_sparse_lam10.json" in f and "race_" in f)
        elif task == 2:
            ok = "_n20_" in f and "_lam" in f and ("bm_admm" in f or "m138" in f)
        else:
            ok = "_lam" in f and (("_race_n" in f) or
                                  (f.startswith("cert_") and ("_n18_" in f or "_n19_" in f or ("_n20_" in f and "m138" not in f))))
        if not ok:
            continue
        # pairwise lambda variants follow the convention of the earlier paths: convex_path/results
        dest = convex if (task == 1 and "_lam" in f) else STORE[task]
        shutil.copy2(os.path.join(OUT, f), os.path.join(dest, f))
        n += 1
    log("stored %d records -> %s (pairwise lambda variants -> convex_path/results)" % (n, os.path.relpath(STORE[task], ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, nargs="*", default=[])
    ap.add_argument("--phase", default="all", choices=["cert", "race", "all"])
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for t in a.task:
        {1: task1, 2: task2, 3: task3}[t](a.phase)
    if a.tables or a.task:
        subprocess.run([PY, os.path.join(HERE, "make_lambda_tables.py")], check=False)
    log("done")


if __name__ == "__main__":
    main()
