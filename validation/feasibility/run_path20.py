"""OPT references along the convex path on the n = 20 DENSITY axis:
p(lambda) = lambda * p_inside + (1 - lambda) * p_hat on each of the six feasibility density
cells (1, 5, 10, 25, 50, 100 %), lambda = k/10 (data/gen_instances.py --path20, keys
n20_{cat}_lam{k}; k = 0 is the stored reference n20_{cat}).

Certifier = the one that certified the base cell in the feasibility study: CGBFQP-GPU at 1 %,
BM-ADMM (K-ADMM, explicit certificate every 200 iterations) at 5-100 %. Certificate:
(UB - LB)/LB <= 1e-3, or UB - LB <= 1e-5 * OPT(lambda = 0) of that density once the point is
inside the polytope (distance ~0). If BM-ADMM does not certify within the cap and the cell is
<= 10 % dense, CGBFQP-GPU is tried as a fallback. One fresh process per cell; certificates only
(no timing claims), so the GPU need not be idle.

    python validation/feasibility/run_path20.py --check-base          # lambda-0 checks (1 %: exact; 5 %: bracket) then k = 1..10
    python validation/feasibility/run_path20.py --cats d1pct m1380 --skip-existing
    python validation/feasibility/run_path20.py --table-only

Results: data/opt_refs.json, validation/results/feasibility/cert_*_n20_{cat}_lam{k}.json,
validation/results/feasibility/PATH20_REFS.md (+ data/CELLS_PATH20.md).
"""
import argparse
import glob
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
CATS = ["d1pct", "m1380", "m2759", "m6899", "m13797", "dense"]
DENSITY = {"d1pct": "1 %", "m1380": "5 %", "m2759": "10 %", "m6899": "25 %", "m13797": "50 %", "dense": "100 %"}
PRIMARY = {"d1pct": "cgbfqp", "m1380": "bm_admm", "m2759": "bm_admm", "m6899": "bm_admm", "m13797": "bm_admm", "dense": "bm_admm"}
FALLBACK_CGBFQP = {"m1380", "m2759"}          # M = 0.5M / 1.0M: column generation still feasible


def refs():
    return json.load(open(REFS))


def floor_for(cat, rel=1e-5):
    return rel * refs()["n20_%s" % cat]["opt"]


def run_cell(key, solver, cap, abs_tol, cert_every):
    t0 = time.perf_counter()
    cmd = [PY, os.path.join(HERE, "feas_one.py"), "--key", key, "--solver", solver, "--criterion", "cert",
           "--cap", str(cap), "--abs-tol", "%.6g" % abs_tol, "--cert-every", str(cert_every), "--out", OUT]
    subprocess.run(cmd, check=False)
    print("   (%s, %.0f s)" % (solver, time.perf_counter() - t0), flush=True)


def certify(cat, k, cap, cert_every, skip_existing, solver=None):
    key = "n20_%s_lam%d" % (cat, k)
    if skip_existing and key in refs():
        print("skip (reference exists)", key, flush=True)
        return
    primary = solver or PRIMARY[cat]
    run_cell(key, primary, cap, floor_for(cat), cert_every)
    if key not in refs() and solver is None and primary == "bm_admm" and cat in FALLBACK_CGBFQP:
        print("   %s not certified by bm_admm within the cap -> fallback cgbfqp" % key, flush=True)
        run_cell(key, "cgbfqp", cap, floor_for(cat), cert_every)


def load_certs(key, converged_only=True):
    """Every certificate run of a cell: {'cgbfqp': record, 'bm_admm': record} (a cell may have
    been certified by both; the reference is the first, the other is a consistency check).
    converged_only=False also returns runs that stopped at a cap: their UB trajectory is still
    a sequence of feasible objectives, usable for the columns-to-1 % table."""
    out = {}
    for solver in ("cgbfqp", "bm_admm"):
        p = find("cert_%s_%s.json" % (solver, key), first=[OUT])
        if p is not None:
            r = json.load(open(p))
            if r.get("converged") or not converged_only:
                out[solver] = r
    return out


def count(r):
    return "%s cols" % r["iters"] if r["solver"] == "cgbfqp" else "%s its" % r["iters"]


def table(cats):
    R = refs()
    lines = ["# OPT references along the convex path, n = 20 density axis\n",
             "p(lambda) = lambda * p_inside + (1 - lambda) * p_hat on each density cell's menus; p_inside = 1/|D| on every "
             "observed cell. lambda = 0 is the stored reference n20_{cat}. Certificate: (UB - LB)/LB <= 1e-3, or "
             "UB - LB <= 1e-5 * OPT(lambda = 0) (absolute) where the distance is ~0. Certifier: CGBFQP-GPU at 1 %, "
             "BM-ADMM at 5-100 % (iterations = ADMM iterations, certificate every 200); a cell certified by the fallback is "
             "marked.\n",
             "| density | lambda | key | OPT (UB) | OPT LB | gap rel | gap abs | criterion | certifier | iterations / columns | wall s | other certifier (check) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cat in cats:
        for k in range(0, 11):
            key = "n20_%s" % cat if k == 0 else "n20_%s_lam%d" % (cat, k)
            r = R.get(key)
            if r is None:
                lines.append("| %s | %.1f | %s | (not certified) | | | | | | | | |" % (DENSITY[cat], k / 10.0, key))
                continue
            g = r.get("gap")
            its = r.get("n_cols") if r.get("n_cols") else r.get("iters")
            certs = load_certs(key)
            other = [c for sname, c in certs.items() if abs(c["ub"] - r["opt"]) > 1e-12 or c["iters"] != its]
            other_txt = "; ".join("%s: [%.5f, %.5f] %s" % (c["solver"], c["lb"], c["ub"], count(c)) for c in other) or ""
            lines.append("| %s | %.1f | %s | %.6f | %.6f | %s | %.1e | %s | %s | %s | %.0f | %s |" % (
                DENSITY[cat], k / 10.0, key, r["opt"], r["lb"], ("%.1e" % g) if g is not None else "-",
                r.get("gap_abs", r["opt"] - r["lb"]), r.get("criterion", "gap_rel"), r.get("method"), its, r.get("wall_s") or 0, other_txt))
    # iterations / columns to (UB - OPT)/OPT <= 1e-2 from the certificate trajectories
    lines += ["", "## Iterations (BM-ADMM, 200-iteration checks) or columns (CGBFQP) to (UB - OPT)/OPT <= 1e-2, from the certificate trajectories", "",
              "| lambda | " + " | ".join(DENSITY[c] for c in cats) + " |", "|---|" + "---|" * len(cats)]
    for k in range(0, 11):
        cells = []
        for cat in cats:
            key = "n20_%s" % cat if k == 0 else "n20_%s_lam%d" % (cat, k)
            r = R.get(key)
            certs = load_certs(key, converged_only=False)
            if r is None or not certs or r["opt"] <= 0:
                cells.append("-")
                continue
            parts = []
            for sname, c in certs.items():
                tr = c.get("trace") or []
                hit = next((e for e in tr if e[2] is not None and (e[2] - r["opt"]) / r["opt"] <= 1e-2), None)
                parts.append("%s %s" % (hit[0], "cols" if sname == "cgbfqp" else "its") if hit else "%s: -" % sname)
            cells.append("; ".join(parts))
        lines.append("| %.1f | " % (k / 10.0) + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells with OPT = 0 (inside the polytope) have no relative target; the 1 % target is relative and therefore "
                 "tighter in absolute terms as lambda grows.")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "PATH20_REFS.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="+", default=CATS)
    ap.add_argument("--k", type=int, nargs="+", default=list(range(1, 11)), help="lambda = k/10")
    ap.add_argument("--cap", type=float, default=3600.0)
    ap.add_argument("--cert-every", type=int, default=200)
    ap.add_argument("--solver", default=None, choices=[None, "cgbfqp", "bm_admm"], help="override the per-density certifier")
    ap.add_argument("--check-base", action="store_true",
                    help="lambda = 0 checks: 1 %% with CGBFQP must reproduce the stored reference exactly; 5 %% with BM-ADMM must agree by bracket")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    if not a.table_only:
        os.makedirs(OUT, exist_ok=True)
        if a.check_base:
            run_cell("n20_d1pct_lam0", "cgbfqp", a.cap, floor_for("d1pct"), a.cert_every)
            run_cell("n20_m1380_lam0", "bm_admm", a.cap, floor_for("m1380"), a.cert_every)
        for cat in a.cats:
            for k in a.k:
                certify(cat, k, a.cap, a.cert_every, a.skip_existing, a.solver)
        subprocess.run([PY, os.path.join(ROOT, "data", "gen_instances.py"), "--path20"], check=False, cwd=os.path.join(ROOT, "data"))
    table(a.cats)


if __name__ == "__main__":
    main()
