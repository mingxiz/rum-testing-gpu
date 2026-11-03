"""OPT references along the convex path p(lambda) = lambda * p_inside + (1 - lambda) * p_hat
on the pairwise instances (data/gen_instances.py: gen_convex_path, keys n{n}_sparse_lam{k},
lambda = k/10), certified with CGBFQP-GPU in a fresh process per cell:

    python validation/feasibility/run_path.py --n 25 30 32 --check-base     # lambda 0 check at the first n, then k = 1..10
    python validation/feasibility/run_path.py --n 33 34 35 --skip-existing  # stream backends (f32 at 33/34, u16 at 35)

Certificate: (UB - LB)/LB <= 1e-3, or UB - LB <= abs_tol (default 1e-5) for cells whose
distance is ~0 (large lambda: the point is inside the polytope). lambda = 0 is the stored
pairwise reference and is only re-run as a consistency check (--check-base, first n).
Results: data/opt_refs.json (new keys), validation/results/feasibility/cert_cgbfqp*_n{n}_sparse_lam{k}.json,
and the table validation/results/feasibility/PATH_REFS.md (+ data/CELLS_PATH.md via gen_instances --path).
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
sys.path.insert(0, os.path.join(ROOT, "validation"))
from result_store import find, glob_all, DENSITY_GRID    # noqa: E402
OUT = os.path.join(ROOT, "validation", "results", "feasibility")
REFS = os.path.join(ROOT, "data", "opt_refs.json")


def pricing_for(n):
    return "mask" if n <= 32 else ("stream_f32" if n <= 34 else "stream_u24")   # u24: the u16 margin floors the certificate on small distances


def run_cell(key, n, cap, abs_tol, skip_existing):
    refs = json.load(open(REFS))
    if skip_existing and key in refs:
        print("skip (reference exists)", key, flush=True)
        return
    t0 = time.perf_counter()
    cmd = [PY, os.path.join(HERE, "feas_one.py"), "--key", key, "--solver", "cgbfqp", "--criterion", "cert",
           "--cap", str(cap), "--pricing", pricing_for(n), "--abs-tol", str(abs_tol), "--out", OUT]
    subprocess.run(cmd, check=False)
    print("   (%.0f s)" % (time.perf_counter() - t0), flush=True)


def table(ns):
    refs = json.load(open(REFS))
    lines = ["# OPT references along the convex path (CGBFQP-GPU certificates)\n",
             "p(lambda) = lambda * p_inside + (1 - lambda) * p_hat on the pairwise design; p_inside = 1/|D| on every observed "
             "menu (0.5 / 0.5). lambda = 0 is the stored pairwise reference n{n}_sparse. Certificate: (UB - LB)/LB <= 1e-3, or "
             "UB - LB <= 1e-5 (absolute) where the distance is ~0. OPT = UB of the certified feasible point.\n",
             "| n | lambda | key | OPT (UB) | OPT LB | gap rel | gap abs | criterion | columns | wall s | method |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ns:
        for k in range(0, 11):
            key = "n%d_sparse" % n if k == 0 else "n%d_sparse_lam%d" % (n, k)
            r = refs.get(key)
            if r is None:
                lines.append("| %d | %.1f | %s | (not run) | | | | | | | |" % (n, k / 10.0, key))
                continue
            g = r.get("gap")
            lines.append("| %d | %.1f | %s | %.6f | %.6f | %s | %.1e | %s | %s | %.0f | %s |" % (
                n, k / 10.0, key, r["opt"], r["lb"], ("%.1e" % g) if g is not None else "-",
                r.get("gap_abs", r["opt"] - r["lb"]), r.get("criterion", "gap_rel"), r.get("n_cols"), r.get("wall_s", 0),
                r.get("method")))
    # columns to the race criterion, read off the certificate trajectories (a race run is
    # the same loop stopped at that column): first column with (UB - OPT)/OPT <= 1e-2
    lines += ["", "## Columns to (UB - OPT)/OPT <= 1e-2 along the path (from the certificate trajectories)", "",
              "| lambda | " + " | ".join("n = %d" % n for n in ns) + " |", "|---|" + "---|" * len(ns)]
    for k in range(0, 7):
        cells = []
        for n in ns:
            key = "n%d_sparse" % n if k == 0 else "n%d_sparse_lam%d" % (n, k)
            r = refs.get(key)
            f = find("cert_cgbfqp_%s.json" % key, first=[OUT])
            if r is None or f is None or r["opt"] <= 0:
                cells.append("-")
                continue
            tr = json.load(open(f))["trace"]
            hit = next((e for e in tr if (e[2] - r["opt"]) / r["opt"] <= 1e-2), None)
            cells.append("%d (%.1f s)" % (hit[0], hit[1]) if hit else "-")
        lines.append("| %.1f | " % (k / 10.0) + " | ".join(cells) + " |")
    lines.append("")
    lines.append("For lambda >= 0.7 the optimum is 0 and the relative criterion is undefined. The 1 % target is relative, "
                 "so it is ten times tighter in absolute terms at lambda = 0.5 than at lambda = 0.")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "PATH_REFS.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[25, 30, 32])
    ap.add_argument("--k", type=int, nargs="+", default=list(range(1, 11)), help="lambda = k/10")
    ap.add_argument("--cap", type=float, default=14400.0)
    ap.add_argument("--abs-tol", type=float, default=1e-5)
    ap.add_argument("--check-base", action="store_true", help="re-run lambda = 0 at the first n and compare with the stored reference")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    if not a.table_only:
        os.makedirs(OUT, exist_ok=True)
        if a.check_base:
            run_cell("n%d_sparse_lam0" % a.n[0], a.n[0], a.cap, a.abs_tol, False)
        for n in a.n:
            for k in a.k:
                run_cell("n%d_sparse_lam%d" % (n, k), n, a.cap, a.abs_tol, a.skip_existing)
        subprocess.run([PY, os.path.join(ROOT, "data", "gen_instances.py"), "--path"], check=False, cwd=os.path.join(ROOT, "data"))
    table(a.n)


if __name__ == "__main__":
    main()
