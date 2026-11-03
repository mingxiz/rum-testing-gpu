"""Timed races along the convex path, pairwise design: CGBFQP-GPU to (UB - OPT)/OPT <= 1 %
against the stored reference optima, for n = 25, 30, 32, 33, 34, 35 and lambda = 0, 0.1, ..., 1.0.

These are the GPU side of the comparison whose CPU side (CGIP / CGIP-heur) runs on a cluster,
so they are wall-clock measurements and the protocol matters:

  * one fresh process per (cell) - no state carried between runs;
  * before each run the GPU must be idle (<= 10 % utilisation, <= 4.5 GB in use, twice in a
    row), enforced by validation/run_paper_table.wait_for_idle_gpu;
  * runs are strictly sequential, never two at once.

The race numbers can also be read off the certificate trajectories, but those were partly
recorded on a shared GPU; this script re-measures every cell on its own.

Target: 1.01 x OPT, with an absolute floor of 1e-5 for the large-lambda cells, whose optimum is
~0 (inside the polytope) so that a relative target is undefined - the floor is the same absolute
tolerance their certificate used.

    python validation/feasibility/run_path_race.py                         # all six n, lambda 0..1
    python validation/feasibility/run_path_race.py --n 25 30 --skip-existing
    python validation/feasibility/run_path_race.py --table-only

Results: validation/results/feasibility/race_cgbfqp[-<backend>]_n{n}_sparse[_lam{k}].json and
the table validation/results/feasibility/PATH_RACE.md.
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
from run_paper_table import wait_for_idle_gpu    # noqa: E402
from result_store import find                    # noqa: E402

PY = sys.executable
OUT = os.path.join(ROOT, "validation", "results", "feasibility")
REFS = os.path.join(ROOT, "data", "opt_refs.json")
NS = [25, 30, 32, 33, 34, 35]
RACE_FLOOR = 1e-5


def pricing_for(n):
    return "mask" if n <= 32 else ("stream_f32" if n <= 34 else "stream_u24")


def key_for(n, k):
    return "n%d_sparse" % n if k == 0 else "n%d_sparse_lam%d" % (n, k)


def result_path(n, k):
    backend = pricing_for(n)
    tag = "" if backend == "mask" else "-" + backend
    return os.path.join(OUT, "race_cgbfqp%s_%s.json" % (tag, key_for(n, k)))


def run_cell(n, k, cap, skip_existing, no_gate):
    key = key_for(n, k)
    path = result_path(n, k)
    if skip_existing and os.path.exists(path):
        print("skip (result exists)", os.path.basename(path), flush=True)
        return
    if key not in json.load(open(REFS)):
        print("skip (no OPT reference)", key, flush=True)
        return
    if not no_gate:
        wait_for_idle_gpu(lambda m: print("   " + m, flush=True))
    t0 = time.perf_counter()
    subprocess.run([PY, os.path.join(HERE, "feas_one.py"), "--key", key, "--solver", "cgbfqp",
                    "--criterion", "race", "--cap", str(cap), "--pricing", pricing_for(n),
                    "--race-floor", str(RACE_FLOOR), "--max-cols", "20000", "--out", OUT], check=False)
    print("   (%.0f s)" % (time.perf_counter() - t0), flush=True)


def table(ns):
    refs = json.load(open(REFS))
    lines = ["# Convex-path races, pairwise design: CGBFQP-GPU to (UB - OPT)/OPT <= 1 %\n",
             "One fresh process per cell on an idle GPU, strictly sequential; the GPU side of the comparison "
             "with the cluster CPU baselines. Target 1.01 x OPT, floored at %g for the large-lambda cells whose "
             "optimum is ~0. Backends: mask-indexed lattice at n <= 32, two-layer float32 at n = 33, 34, two-layer "
             "uint24 at n = 35.\n" % RACE_FLOOR,
             "## Wall time in seconds (columns generated)\n",
             "| lambda | " + " | ".join("n = %d" % n for n in ns) + " |", "|---|" + "---|" * len(ns)]
    grid = {}
    for k in range(0, 11):
        row = []
        for n in ns:
            p = find(os.path.basename(result_path(n, k)), first=[OUT])
            if p is None:
                row.append("")
                continue
            r = json.load(open(p))
            grid[(n, k)] = r
            if r.get("stop") != "phi_ub":
                row.append("%s (%s)" % (r.get("stop"), r.get("iters")))
                continue
            t = r["wall_s"]
            row.append("%s (%d)" % (("%.2f" % t) if t < 10 else ("%.1f" % t), r["iters"]))
        lines.append("| %.1f | " % (k / 10.0) + " | ".join(row) + " |")
    lines += ["", "## Reference optimum and the achieved upper bound\n",
              "| n | lambda | OPT | UB at the stop | (UB-OPT)/OPT | columns | wall s | backend |",
              "|---|---|---|---|---|---|---|---|"]
    for n in ns:
        for k in range(0, 11):
            r = grid.get((n, k))
            if r is None:
                continue
            opt = refs[key_for(n, k)]["opt"]
            rel = ("%.4f %%" % (100.0 * (r["ub"] - opt) / opt)) if opt > 0 else "OPT ~ 0 (absolute floor)"
            lines.append("| %d | %.1f | %.6f | %.6f | %s | %d | %.2f | %s |" % (
                n, k / 10.0, opt, r["ub"], rel, r["iters"], r["wall_s"], r.get("pricing", "mask")))
    lines.append("")
    lines.append("Every run reports `avg_cores` ~ 1.0: the GPU solvers use one CPU thread and the whole device.")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "PATH_RACE.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=NS)
    ap.add_argument("--k", type=int, nargs="+", default=list(range(0, 11)))
    ap.add_argument("--cap", type=float, default=14400.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-gate", action="store_true", help="skip the idle-GPU wait (NOT for reported timings)")
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    if not a.table_only:
        os.makedirs(OUT, exist_ok=True)
        for n in a.n:
            for k in a.k:
                run_cell(n, k, a.cap, a.skip_existing, a.no_gate)
    table(a.n)


if __name__ == "__main__":
    main()
