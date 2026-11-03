"""final_results/Application_DSR and final_results/Application_AI: the two empirical applications as CSV,
one row per complete run, plus the CGIP / CGIP-heur cluster runs.

    final_results/Application_DSR/results_dsr.csv   Dean, Ravindran and Stoye (2022), n = 25 (alt0 and full)
    final_results/Application_AI/results_ai.csv     LLM choice data, n = 18 (gpt-5.6-luna, deepseek-v4-flash,
                                                    gpt-6 luna without reasoning, GMI Cloud)

Sources are the stored records of this repository:
    replication/empirical_{dsr_n25,ai_n18}/results/bootstrap_*.json    (run_bootstrap.py, rum.bootstrap: the GPU rows)
    replication/cgip/cluster_records/Application_{DSR,AI}/summary.csv  (CGIP / CGIP-heur on the CPU cluster)

The AI data sets are gpt-5.6-luna and deepseek-v4-flash (100 replies per menu), gpt-6 luna without reasoning and
GMI Cloud (2,000).

The cluster runs were made with an earlier version of the CGIP code (replication/cgip/run_cgip.py re-runs them with
this release). The cluster DSR runs are the ones marked report = yes (one uninterrupted job on an Ice Lake node);
the chained day jobs on Cascade Lake reproduce them on slower hardware and are left out. They draw B = 10,000 at
abs 5e-4, then tighten dist(p_hat) and every undecided draw to abs 1e-7, so dist_lb / dist_ub and the tau band
are the tightened ones. CGIP-heur is reported with R = 10 random-insertion restarts; the summaries' R = 1 rows are
left out here.

Every run is the Hong-Li numerical delta-method p-value with a single per-solve tolerance abs_tol (rel 0) that
governs both the dist(p_hat) certificate and every draw that the threshold tests do not decide. All runs start from
abs 5e-4. The GPU rows then refine within the run: up to two tenfold tighter rounds (5e-5, 5e-6) for p_hat and the
draws still undecided; the cluster CGIP rows settle the undecided draws at abs 1e-7. The final runs use B = 10,000
multinomial_reject draws. The GPU rows also carry the one-sided 95 % confidence interval where it was computed.

The AI data also have one BM-ADMM row per data set with menu-by-menu draws (run_bootstrap.py --input replies: each
menu's replies resampled on their own, multinomial weights). Those rows carry the two-sided 95 % interval for dist,
[max(0, dist_lb - c), dist_ub + c] with c the 95 % quantile of ||p_hat_b - p_hat|| (Bootstrap.distance_interval,
no draw solved), the interval of the paper's application section.
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DSR = os.path.join(ROOT, "replication", "empirical_dsr_n25", "results")
AI = os.path.join(ROOT, "replication", "empirical_ai_n18", "results")
FINAL = os.path.join(ROOT, "final_results")
IP = os.path.join(ROOT, "replication", "cgip", "cluster_records")

FIELDS = ["application", "data", "n", "M", "units", "B", "bootstrap", "seed", "solver", "variant", "abs_tol", "rel_tol",
          "dist_lb", "dist_ub", "dist_gap", "tau_lo", "tau_hi", "cert_s", "cert_count", "count_unit", "loop_s", "total_s",
          "draw_mean_s", "draw_count_mean", "n_below", "n_above", "n_undecided", "pval_lo", "pval_hi", "reject_5pct",
          "status", "draws_done", "boot_redraws", "data_fingerprint", "draws_fingerprint", "hardware", "record", "when", "note",
          "ci_lower", "ci_lower_hi", "ci_s", "ci_draws", "dci_lower", "dci_upper", "dci_half_width"]
IP_SOLVER = {"cgip": ("CGIP", "exact IP pricing"), "cgip_rand": ("CGIP-heur", "random insertion, R = 10, IP at stalls")}
HEUR_R = "10"                     # the cluster CGIP-heur runs reported here; the R = 1 runs are left out


def reported(s):
    """A cluster summary line to report: CGIP, or CGIP-heur with R = HEUR_R."""
    return s["method"] != "cgip_rand" or s["n_restart"] == HEUR_R
AI_DATA = {"gpt": "gpt-5.6-luna", "deepseek": "deepseek-v4-flash"}

# the GPU rows: rum.bootstrap records (data, record); BM-ADMM with the published settings and CGBFQP-GPU on AI,
# CGBFQP-GPU (the automatic choice) on DSR at B = 10,000 and 1,000
DSR_BOOT = [("alt0", "bootstrap_dsr_alt0.json"), ("full", "bootstrap_dsr_full.json"),
            ("alt0", "bootstrap_dsr_alt0_B1000.json"), ("full", "bootstrap_dsr_full_B1000.json")]
AI_BOOT = [("gpt-5.6-luna", "bootstrap_ai_luna.json"), ("gpt-5.6-luna", "bootstrap_ai_luna_flow_cg.json"),
           ("deepseek-v4-flash", "bootstrap_ai_deepseek.json"), ("deepseek-v4-flash", "bootstrap_ai_deepseek_flow_cg.json"),
           ("gpt-6-luna-noreason", "bootstrap_ai_gpt6luna_noreason.json"),
           ("gpt-6-luna-noreason", "bootstrap_ai_gpt6luna_noreason_flow_cg.json"),
           ("gmi-cloud", "bootstrap_ai_gmicloud.json"),
           ("gmi-cloud", "bootstrap_ai_gmicloud_flow_cg.json")]
AI_BOOT += [(data, "bootstrap_ai_%s_replies.json" % model)           # menu-by-menu draws, the two-sided interval
            for data, model in (("gpt-5.6-luna", "luna"), ("deepseek-v4-flash", "deepseek"),
                                ("gpt-6-luna-noreason", "gpt6luna_noreason"), ("gmi-cloud", "gmicloud"))]
AI_ORDER = {"gpt-5.6-luna": 0, "deepseek-v4-flash": 1, "gpt-6-luna-noreason": 2, "gmi-cloud": 3}   # in order
SHAPE = {"alt0": (25, 301), "full": (25, 901), "gpt-5.6-luna": (18, 73492), "deepseek-v4-flash": (18, 73492),
         "gpt-6-luna-noreason": (18, 73492), "gmi-cloud": (18, 73492)}   # n, M
BOOT_SOLVER = {"flow_cg": ("CGBFQP-GPU", "columns"), "bm_admm": ("BM-ADMM", "iterations")}


def r6(x):
    return None if x is None else round(float(x), 6)


def fx(x):
    return None if x is None else float(x)


def boot_row(app, data, folder, name):
    """A GPU row from a rum.bootstrap record (replication/empirical_*/run_bootstrap.py), or None if it is missing."""
    p = os.path.join(folder, name)
    if not os.path.exists(p):
        print("   missing record:", name)
        return None
    r = json.load(open(p))
    st, pv = r["settings"], r["results"]["p_value"]["0.05"]
    ci = r["results"]["confidence_interval"].get("0.05") or {}
    dci = r["results"].get("distance_interval", {}).get("0.05") or {}
    menu_draws = st.get("input") == "replies"
    solver, unit = BOOT_SOLVER[r["plan"]["method"]]
    cert = r["p_hat"]["certificates"]
    cert_s = r["time"]["certificate"]
    n, M = SHAPE[data]
    return {
        "application": app, "data": data, "n": n, "M": M, "units": st["n"], "B": st["B"], "bootstrap": st["weights"],
        "seed": st["seed"], "solver": solver,
        "variant": "restart from p_hat, refine %d%s" % (st["refine"], "; menu-by-menu draws" if menu_draws else ""),
        "abs_tol": st["abs_tol"], "rel_tol": st["rel_tol"], "dist_lb": r["p_hat"]["lb"], "dist_ub": r["p_hat"]["ub"],
        "dist_gap": r["p_hat"]["ub"] - r["p_hat"]["lb"], "tau_lo": r["tau"][0], "tau_hi": r["tau"][1],
        "cert_s": r6(cert_s), "cert_count": cert[0]["iterations"], "count_unit": unit, "loop_s": r6(pv["time"] - cert_s),
        "total_s": r6(pv["time"]), "draw_mean_s": r6((pv["time"] - cert_s) / st["B"]), "draw_count_mean": None,
        "n_below": pv["n_below"], "n_above": pv["n_above"], "n_undecided": pv["n_undecided"], "pval_lo": r6(pv["p_lo"]),
        "pval_hi": r6(pv["p_hi"]), "reject_5pct": int(bool(pv["reject"])),
        "status": "complete" if not pv["stopped_early"] else "stopped once the decision was locked",
        "draws_done": st["B"], "boot_redraws": None if menu_draws else r["panel"]["redraws"],
        "data_fingerprint": r["fingerprints"]["data"],
        "draws_fingerprint": r["fingerprints"]["draws"],
        "hardware": cert[0]["machine"].split(" (used)")[0].replace("NVIDIA GeForce ", ""),
        "record": os.path.relpath(p, ROOT).replace(os.sep, "/"), "when": r["when"],
        "note": "rum.bootstrap: %d draws screened below with no solve, %d solved; p_hat certified %d time%s"
                % (pv["n_screened"], pv["n_solved"], len(cert), "" if len(cert) == 1 else "s"),
        "ci_lower": ci.get("lower"), "ci_lower_hi": (ci.get("lower_bracket") or [None, None])[1],
        "ci_s": r6(ci.get("time")), "ci_draws": ci.get("n_solved"),
        "dci_lower": dci.get("lower"), "dci_upper": dci.get("upper"), "dci_half_width": dci.get("half_width")}


def ip_summary(app):
    p = os.path.join(IP, app, "summary.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def slurm_s(x):
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", x or "")
    return None if m is None else int(m[1] or 0) * 86400 + int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4])


def dsr_cluster_rows():
    res = os.path.join(IP, "Application_DSR")
    by_job = {}                       # last_job -> run record (record names differ by tag: '', _day, _R10_week, ...)
    for name in os.listdir(res):
        if name.endswith(".json") and "_dist" not in name and "_firstpass" not in name:
            by_job[json.load(open(os.path.join(res, name))).get("last_job")] = name
    rows = []
    for s in ip_summary("Application_DSR"):
        if s["report"] != "yes" or not reported(s):
            continue
        name = by_job[s["last_job"]]
        r = json.load(open(os.path.join(res, name)))
        solver, variant = IP_SOLVER[s["method"]]
        B = int(s["B"])
        rows.append({
            "application": "DSR", "data": s["obs"], "n": 25, "M": int(s["M"]), "units": 520, "B": B,
            "bootstrap": "multinomial_reject", "seed": r["seed"], "solver": solver, "variant": variant,
            "abs_tol": float(s["abs_tol"]), "rel_tol": 0.0, "dist_lb": fx(s["dist_refined_lb"]), "dist_ub": fx(s["dist_refined_ub"]),
            "dist_gap": fx(s["dist_refined_ub"]) - fx(s["dist_refined_lb"]), "tau_lo": fx(s["tau_lo"]), "tau_hi": fx(s["tau_hi"]),
            "cert_s": r6(s["dist_s"]), "cert_count": (r.get("dist") or {}).get("n_columns"), "count_unit": "columns",
            "loop_s": r6(s["draws_s"]), "total_s": r6(s["total_s"]), "draw_mean_s": r6(float(s["draws_s"]) / B),
            "draw_count_mean": None, "n_below": int(s["n_below"]), "n_above": int(s["n_above"]),
            "n_undecided": int(s["n_undecided"]), "pval_lo": r6(s["pval_lo"]), "pval_hi": r6(s["pval_hi"]),
            "reject_5pct": int(s["reject_5pct"] == "True"), "status": s["status"], "draws_done": B,
            "boot_redraws": r.get("n_redrawn"), "data_fingerprint": None, "draws_fingerprint": None,
            "hardware": "%s, Gurobi %s threads" % (s["cpu"], s["threads"]),
            "record": "replication/cgip/cluster_records/Application_DSR/" + name, "when": "",
            "note": "%s draws undecided at abs %s, settled at abs %g (%.0f s, included in total_s)"
                    % (s["n_undecided_first_pass"], s["abs_tol"], r["refine_tol"], float(s["refine_s"]))})
    return rows


def dsr_rows():
    rows = [z for z in (boot_row("DSR", data, DSR, name) for data, name in DSR_BOOT) if z]
    rows += dsr_cluster_rows()
    order = {"CGBFQP-GPU": 0, "CGIP": 1, "CGIP-heur": 2}
    rows.sort(key=lambda z: (z["data"], order[z["solver"]], z["B"], -z["abs_tol"], z["variant"]))
    return rows


def ai_rows():
    rows = [z for z in (boot_row("AI", data, AI, name) for data, name in AI_BOOT) if z]
    for s in ip_summary("Application_AI"):
        if not reported(s) or AI_DATA[s["model"]] not in AI_ORDER:
            continue
        solver, variant = IP_SOLVER[s["method"]]
        rows.append({
            "application": "AI", "data": AI_DATA[s["model"]], "n": 18, "M": int(s["M"]), "units": int(s["n"]),
            "B": int(s["B"]), "bootstrap": "multinomial_reject", "seed": 0, "solver": solver, "variant": variant,
            "abs_tol": float(s["abs_tol"]), "rel_tol": 0.0, "dist_lb": fx(s["dist_lb"] or None), "dist_ub": fx(s["dist_ub"] or None),
            "count_unit": "columns", "total_s": slurm_s(s["elapsed"]) if s["status"] == "timeout" else None,
            "status": "capped (24 h, during the dist(p_hat) certificate)" if s["status"] == "timeout" else s["status"],
            "draws_done": int(s["draws_done"] or 0), "hardware": "CPU cluster node, Gurobi %s threads" % s["threads"],
            "record": "replication/cgip/cluster_records/Application_AI/summary.csv",
            "note": "%s; best UB %s" % (s["note"], s["best_UB"])})
    rows.sort(key=lambda z: (AI_ORDER[z["data"]], -z["abs_tol"], z["solver"], z["variant"]))
    return rows


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in FIELDS})
    print("wrote %s (%d rows)" % (os.path.relpath(path, ROOT), len(rows)))


def main():
    write(os.path.join(FINAL, "Application_DSR", "results_dsr.csv"), dsr_rows())
    write(os.path.join(FINAL, "Application_AI", "results_ai.csv"), ai_rows())


if __name__ == "__main__":
    main()
