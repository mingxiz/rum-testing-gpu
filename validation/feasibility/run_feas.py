"""Feasibility study: (1) pairwise design with growing n (CGBFQP-GPU), (2) n = 20
with growing density (BM-ADMM). One command, fresh process per measurement, GPU
verified idle before each run.

    python validation/feasibility/run_feas.py                      # both parts, cert + race
    python validation/feasibility/run_feas.py --part pairwise --n-max 32
    python validation/feasibility/run_feas.py --part density --stage race

Stages: data (regenerate + fingerprints) -> cert (OPT references; cells that
already have one are re-certified as a consistency check; their time is not
reported, so they run without the idle gate) -> race (timed runs with
trajectories; each waits for an idle GPU) -> tables (results/feasibility/*.md, paper/tables/tab_feas_*.tex).
The CPU baselines (CGIP, CGIP-heur) are run elsewhere on the same instances
(data/instances/*.csv.gz + data/opt_refs.json) and merged by make_feas_tables.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
sys.path.insert(0, os.path.join(ROOT, "data"))
from run_paper_table import wait_for_idle_gpu     # noqa: E402
from gen_instances import FEAS_PAIRWISE, FEAS_DENSITY20   # noqa: E402

PY = sys.executable
OUT = os.path.join(ROOT, "validation", "results", "feasibility")


def pairwise_cells(n_max):
    return ["n20_sparse"] + ["n%d_%s" % (n, c) for n in sorted(FEAS_PAIRWISE) for c in FEAS_PAIRWISE[n] if n <= n_max]


def density_cells():
    return ["n20_d1pct", "n20_m1380", "n20_m2759", "n20_m6899", "n20_m13797", "n20_dense"]


def run(key, solver, criterion, cap, extra=()):
    path = os.path.join(OUT, "%s_%s_%s.json" % (criterion, solver, key))
    if criterion == "race":          # wall time is reported: the machine must be idle (no other GPU work)
        wait_for_idle_gpu(lambda m: print(m, flush=True))
    cmd = [PY, os.path.join(HERE, "feas_one.py"), "--key", key, "--solver", solver,
           "--criterion", criterion, "--cap", str(cap), "--out", OUT] + list(extra)
    try:
        subprocess.run(cmd, check=False, timeout=cap + 3600)
    except subprocess.TimeoutExpired:
        json.dump(dict(key=key, solver=solver, criterion=criterion, stop="killed", cap_s=cap), open(path, "w"))
        print("[%s] %s %s killed" % (key, solver, criterion), flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "pairwise", "density"])
    ap.add_argument("--stage", default="all", choices=["all", "data", "cert", "race", "tables"])
    ap.add_argument("--n-max", type=int, default=32, help="largest pairwise n to run (32 = mask-indexed limit on 32 GB)")
    ap.add_argument("--cap", type=float, default=3600.0)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stages = ["data", "cert", "race", "tables"] if a.stage == "all" else [a.stage]
    parts = ["pairwise", "density"] if a.part == "all" else [a.part]

    if "data" in stages:
        print("== data: regenerate feasibility cells + fingerprint check", flush=True)
        subprocess.run([PY, os.path.join(ROOT, "data", "gen_instances.py"), "--feas"], check=True)

    plan = []
    if "pairwise" in parts:
        plan += [(k, "cgbfqp") for k in pairwise_cells(a.n_max)]
    if "density" in parts:
        plan += [(k, "bm_admm") for k in density_cells()]

    for stage in ("cert", "race"):
        if stage not in stages:
            continue
        print("== %s: %d runs" % (stage, len(plan)), flush=True)
        for key, s in plan:
            path = os.path.join(OUT, "%s_%s_%s.json" % (stage, s, key))
            if a.skip_existing and os.path.exists(path):
                continue
            t0 = time.perf_counter()
            run(key, s, stage, a.cap)
            print("   (%.0f s)" % (time.perf_counter() - t0), flush=True)

    if "tables" in stages:
        subprocess.run([PY, os.path.join(HERE, "make_feas_tables.py")], check=True)


if __name__ == "__main__":
    main()
