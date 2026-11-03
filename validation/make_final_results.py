"""final_results/: the numbers behind the paper tables as CSV, laid out like final_data/.

    final_results/Numerical1/results_n18.csv, results_n19.csv, results_n20.csv
        tab_race_n{18,19,20}_l{0,05,1}: BM-ADMM, CGBFQP-GPU, Flow-ADMM on the density grid
    final_results/Numerical2/results_pairwise.csv
        tab_fes_pairwise_l{0,05,1}: CGBFQP-GPU, pairwise design, n = 25, 27, 30, 32, 35
    final_results/Numerical3/results_density.csv
        tab_fes_density_l{0,05,1}: BM-ADMM, n = 20, density 0.1, 0.5, 1, 5 %

One row per (instance, lambda, solver). The records and the cell rule are the ones
make_lambda_tables.py uses, so `table_entry` is exactly what the .tex prints and `wall_s` is the
unrounded time behind it. `instance` is the key of the instance; with lambda = 0 it is the file
stem in final_data/<experiment>/raw_data.

CGIP and CGIP-heur come from the CPU cluster runs, made with an earlier version of the CGIP code, in
replication/cgip/cluster_records/<experiment>/summary.csv, one row per run there (every cell that was
submitted, solved or not). Their time is build + solve on 24 Gurobi threads. CGIP-heur is the
random-insertion heuristic with R = 10 restarts (method cgip_heur_rand_R10); the summaries' R = 1 rows
(cgip_heur_rand) are left out here.
"""
import csv
import os
import re

import make_lambda_tables as T

FINAL = os.path.join(T.ROOT, "final_results")
SOLVER_NAME = {"bm_admm": "BM-ADMM", "cgbfqp": "CGBFQP-GPU", "flow_admm": "Flow-ADMM"}
COUNT_UNIT = {"bm_admm": "iterations", "cgbfqp": "columns", "flow_admm": "iterations"}
STATUS = {"ok": "ok", "cap": "CAP", "infeasible": "not run (device memory)",
          "err": "failed", None: "missing"}
IP = os.path.join(T.ROOT, "replication", "cgip", "cluster_records")
IP_NAME = {"cgip": "CGIP", "cgip_heur_rand_R10": "CGIP-heur"}
FIELDS = ["n", "m2", "design", "density_pct", "m_p", "lambda", "instance", "opt", "opt_lb",
          "target_ub", "solver", "table_entry", "status", "fastest_gpu", "wall_s", "count",
          "count_unit", "ub_at_stop", "stop", "cap_s", "pricing_backend", "note"]


def row(n, cat, k, lam, solver, r, fastest=None):
    key = T.key_of(n, cat, k)
    ref = T.REFS[key]
    m2 = n * (1 << (n - 1))
    st = T.status(r)
    ran = st in ("ok", "cap")
    r = r or {}
    return {
        "n": n, "m2": m2, "design": "pairwise" if cat == "sparse" else "random_menu",
        "density_pct": 100.0 * ref["mp"] / m2, "m_p": ref["mp"], "lambda": lam, "instance": key,
        "opt": ref["opt"], "opt_lb": ref["lb"], "target_ub": max(1.01 * ref["opt"], 1e-3),
        "solver": SOLVER_NAME[solver], "table_entry": T.cell(r if st else None),
        "status": STATUS[st], "fastest_gpu": "" if fastest is None else int(fastest == solver),
        "wall_s": r.get("wall_s") if ran else None, "count": r.get("iters") if ran else None,
        "count_unit": COUNT_UNIT[solver], "ub_at_stop": r.get("ub"), "stop": r.get("stop"),
        "cap_s": r.get("cap_s"), "pricing_backend": r.get("pricing", "mask") if solver == "cgbfqp" and r else "",
        "note": r.get("note", ""),
    }


def slurm_s(x):
    """Slurm elapsed '[D-]HH:MM:SS' in seconds."""
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", x or "")
    return None if m is None else int(m[1] or 0) * 86400 + int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4])


def ip_summary(experiment):
    """{(instance, method): row} from replication/cgip/cluster_records/<experiment>/summary.csv."""
    p = os.path.join(IP, experiment, "summary.csv")
    if not os.path.exists(p):
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {(r["key"], r.get("method") or "cgip"): r for r in csv.DictReader(f)}


def f_or_none(x):
    return None if x in (None, "") else float(x)


def ip_row(n, cat, k, lam, solver, s, cap_s):
    """One CGIP / CGIP-heur row from a cluster summary line s."""
    key = T.key_of(n, cat, k)
    ref = T.REFS[key]
    m2 = n * (1 << (n - 1))
    st = s["status"]
    if st == "solved":
        status, wall = "ok", float(s["total_s"])
        entry = T.t(wall)
    elif st == "timeout":
        # killed by Slurm at the wall-time limit
        status, wall = "CAP", slurm_s(s["elapsed"])
        entry = "CAP"
    else:
        status, wall, entry = "out of memory (%s GB)" % s["peak_mem_gb"].split(".")[0], None, "--"
    note = s["note"]
    if st == "solved" and float(s["LB"]) == 0.0 and ref["opt"] > 0:
        note = ("no lower bound of its own (LB = 0, no IP call): it stopped on UB <= target_ub, which "
                "uses the reference optimum")
    return {
        "n": n, "m2": m2, "design": "pairwise" if cat == "sparse" else "random_menu",
        "density_pct": 100.0 * ref["mp"] / m2, "m_p": ref["mp"], "lambda": lam, "instance": key,
        "opt": ref["opt"], "opt_lb": ref["lb"], "target_ub": max(1.01 * ref["opt"], 1e-3),
        "solver": solver, "table_entry": entry, "status": status, "fastest_gpu": "",
        "wall_s": wall, "count": s["iters"] or None,
        "count_unit": "columns",
        "ub_at_stop": f_or_none(s["UB"]),
        "stop": {"solved": "ub_target", "timeout": "wall_cap", "out_of_memory": "out_of_memory"}[st],
        "cap_s": cap_s, "pricing_backend": "", "note": note,
    }


def ip_rows(summary, n, cat, k, lam, names, cap_s):
    key = T.key_of(n, cat, k)
    return [ip_row(n, cat, k, lam, name, summary[(key, m)], cap_s)
            for m, name in names.items() if (key, m) in summary]


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print("wrote %s (%d rows)" % (os.path.relpath(path, T.ROOT), len(rows)))


def main():
    day, week = 86400.0, 7 * 86400.0
    ip1 = ip_summary("Numerical1")
    ip2, ip3 = ip_summary("Numerical2"), ip_summary("Numerical3")
    # Numerical1: grid races, one file per n; fastest_gpu marks the bold cell (GPU solvers only)
    for n in (18, 19, 20):
        rows = []
        for k, _, lam in T.LAMS:
            for cat in T.CELLS[n]:
                recs = {s: T.grid_rec(n, cat, k, s) for s in SOLVER_NAME}
                ok = {s: r["wall_s"] for s, r in recs.items() if T.status(r) == "ok"}
                best = min(ok, key=ok.get) if ok else ""
                rows += [row(n, cat, k, lam, s, recs[s], best) for s in SOLVER_NAME]
                rows += ip_rows(ip1, n, cat, k, lam, IP_NAME, day)
        write(os.path.join(FINAL, "Numerical1", "results_n%d.csv" % n), rows)
    # Numerical2: pairwise design, n = 25 ... 35
    # (CGIP limits: 24 h for n <= 30, 7 d for n = 32, 35)
    rows = [z for k, _, lam in T.LAMS for n in T.PAIRWISE_N
            for z in [row(n, "sparse", k, lam, "cgbfqp", T.pairwise_rec(n, k))]
            + ip_rows(ip2, n, "sparse", k, lam, IP_NAME, week if n >= 32 else day)]
    write(os.path.join(FINAL, "Numerical2", "results_pairwise.csv"), rows)
    # Numerical3: density at n = 20
    rows = [z for k, _, lam in T.LAMS for cat, _ in T.DENSITY
            for z in [row(20, cat, k, lam, "bm_admm", T.density_rec(cat, k))]
            + ip_rows(ip3, 20, cat, k, lam, IP_NAME, day)]
    write(os.path.join(FINAL, "Numerical3", "results_density.csv"), rows)


if __name__ == "__main__":
    main()
