"""Reproduce the paper's n = 18-20 tables with one command.

    python validation/run_paper_table.py                      # GPU solvers, all 30 cells, race criterion
    python validation/run_paper_table.py --solvers bm_admm,cgbfqp,flow_admm,cgip,cgip_heur
    python validation/run_paper_table.py --n 18 --cells n18_sparse,n18_m39 --criterion cert

Steps: (1) regenerate the instances from their seeds and verify every
fingerprint against data/opt_refs.json; (2) run every (cell, solver) in its own
fresh process, nothing else running (a run that hits --cap is reported as CAP;
a Flow-ADMM cell whose factorization cannot fit is reported as "--");
(3) write results/<tag>/*.json, then the LaTeX tables paper/tables/tab_n*.tex
and results/<tag>/summary.md, and a side-by-side comparison with the reference
results committed from the paper machine (replication/density_grid/results/).

Fairness conventions (the paper's measurement protocol): every solver gets the whole
machine; no thread caps are imposed; the cores actually used are measured and
recorded per run (avg_cores); the GPU solvers use one CPU thread plus one GPU.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
GPU_SOLVERS = ("bm_admm", "cgbfqp", "flow_admm")
IDLE_UTIL_PCT, IDLE_MEM_MB, IDLE_SAMPLES, IDLE_PERIOD_S = 10, 4500, 2, 15   # desktop apps hold ~3 GB on WDDM


def gpu_busy():
    """(util %, used MiB) from nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        u, m = out.strip().split("\n")[0].split(",")
        return int(u), int(m)
    except Exception:
        return None


def wait_for_idle_gpu(log=print):
    """Every run must have the machine to itself: block while another process
    keeps the GPU busy (utilization or memory above the idle baseline)."""
    quiet, waited = 0, 0
    while quiet < IDLE_SAMPLES:
        s = gpu_busy()
        if s is None or (s[0] <= IDLE_UTIL_PCT and s[1] <= IDLE_MEM_MB):
            quiet += 1
        else:
            quiet = 0
            if waited % 300 == 0:
                log("   GPU busy (util %d%%, %d MiB used by others) - waiting" % s)
            waited += IDLE_PERIOD_S
        if quiet < IDLE_SAMPLES:
            time.sleep(IDLE_PERIOD_S)
    if waited:
        log("   GPU idle after %d s of waiting" % waited)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", default=",".join(GPU_SOLVERS))
    ap.add_argument("--n", default="18,19,20")
    ap.add_argument("--cells", default=None, help="comma-separated keys (default: all grid cells)")
    ap.add_argument("--criterion", default="race", choices=["race", "cert"])
    ap.add_argument("--cap", type=float, default=900.0)
    ap.add_argument("--tag", default=None, help="results subfolder (default: hostname_date)")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi threads for cgip* (default: Gurobi's)")
    ap.add_argument("--qp-backend", default="gurobi", choices=["gurobi", "native"],
                    help="cgip*: master-QP backend (gurobi as shipped, or the native active-set Cholesky)")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    # (1) data
    print("== data: regenerate + fingerprint check", flush=True)
    subprocess.run([PY, os.path.join(ROOT, "data", "gen_instances.py"), "--check"], check=True)
    manifest = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    ns = {int(x) for x in a.n.split(",")}
    cells = a.cells.split(",") if a.cells else [k for k, r in manifest.items() if r["n"] in ns]
    solvers = a.solvers.split(",")
    tag = a.tag or "%s_%s" % (os.environ.get("COMPUTERNAME") or os.uname().nodename, time.strftime("%Y%m%d"))
    out = os.path.join(HERE, "results", tag)
    os.makedirs(out, exist_ok=True)
    print("== %d cells x %d solvers -> %s" % (len(cells), len(solvers), out), flush=True)

    # (2) runs, one fresh process each
    for key in cells:
        for s in solvers:
            path = os.path.join(out, "%s_%s_%s.json" % (s, a.criterion, key))
            if a.skip_existing and os.path.exists(path):
                continue
            cmd = [PY, os.path.join(HERE, "run_one.py"), "--key", key, "--solver", s,
                   "--criterion", a.criterion, "--cap", str(a.cap), "--out", out]
            if s.startswith("cgip"):
                cmd += ["--qp-backend", a.qp_backend]
                if a.threads is not None:
                    cmd += ["--threads", str(a.threads)]
            wait_for_idle_gpu(lambda msg: print(msg, flush=True))
            try:
                subprocess.run(cmd, check=False, timeout=a.cap + 1800)
            except subprocess.TimeoutExpired:
                json.dump(dict(key=key, solver=s, criterion=a.criterion, stop="killed", cap_s=a.cap),
                          open(path, "w"), indent=2)
                print("[%s] %s killed after %.0fs" % (key, s, a.cap + 1800), flush=True)

    # (3) tables + comparison
    subprocess.run([PY, os.path.join(HERE, "make_tables.py"), "--tag", tag, "--criterion", a.criterion], check=True)


if __name__ == "__main__":
    main()
