"""IP-CG and IP-H-CG, the paper's CGIP columns, through the RUM class.

    python replication/cgip/run_cgip.py race --set Numerical2 --key n25_sparse --method ip_cg --threads 24
    python replication/cgip/run_cgip.py race --set Numerical3 --key n20_m28 --method ip_cg_heur --threads 24
    python replication/cgip/run_cgip.py dsr --obs alt0 --method ip_cg --threads 24
    python replication/cgip/run_cgip.py ai --model gpt6luna_noreason --method ip_cg_heur --threads 24

The printed IP-CG and IP-H-CG numbers come from cluster runs made with an earlier version of the CGIP
code; their result files are in cluster_records/ (see its README.md). This file runs the same experiments
with this release's solver, RUM(...) from solver/rum_solver.py: method "ip_cg" is IP-CG (exact IP pricing),
"ip_cg_heur" IP-H-CG (random-insertion pricing, the IP when it stalls, --restarts 10 per pricing call as
in the paper). Gurobi on the CPU; no GPU.

race  tab:density18-20 (Numerical1, IP-CG only), tab:binary (Numerical2), tab:dense (Numerical3): one cell
      of final_data/<set>/ by rum.session(method).solve(p_hat, below=target), which stops once
      UB < target = OPT + max(1e-2 OPT, 1e-3), the paper's criterion, or at the cap: 7 d for Numerical2
      with n >= 32, else 24 h, the cluster's limits. A capped cell is the table's ">= 24 h" or ">= 7 d".
dsr   tab:dsg: rum.bootstrap on the DSR panel, as empirical_dsr_n25/run_bootstrap.py.
ai    tab:ai: rum.bootstrap on the AI table, panel input, as empirical_ai_n18/run_bootstrap.py.
      dsr and ai certify p_hat first, within --cert-hours, and stop there if it does not close: on the
      cluster no AI certificate closed in 24 h, which is the table's ">= 24 h".

Everything else is the class's default (solver/defaults.py): the native master QP, 100,000 columns, and
for the bootstrap B = 10,000 draws, seed 0, abs 5e-4, rel 0, and two tenfold tighter rounds for the draws
still undecided. Writes results/<run>.json. A bootstrap run keeps a checkpoint next to it and, started
again with the same arguments, continues from it.
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "replication", "empirical_ai_n18"))
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, ROOT)
from solver.rum_solver import RUM                    # noqa: E402
from solver.inference import cells_from_panel        # noqa: E402
from gen_instances import fingerprint                # noqa: E402
from replies import load_table                       # noqa: E402

DATA = os.path.join(ROOT, "final_data")
DSR = os.path.join(DATA, "Application_DSR", "raw_data", "overload_data_25_long.csv")
N_DSR = 520             # the DSR sample size: respondents
AI = {"gpt6luna_noreason": "gpt6luna_noreason.npz",     # tab:ai: GPT-6 Luna
      "gmicloud": "gmicloud.npz",                       # tab:ai: DeepSeek V4 Flash
      "luna": "gpt5.6-luna.npz",                        # the 100-reply sets of the cluster's AI runs
      "deepseek": "deepseekv4-flash.npz"}
REL, ABS = 1e-2, 1e-3   # race target: UB < OPT + max(REL * OPT, ABS)


def method_args(a):
    args = dict(threads=a.threads)
    if a.method == "ip_cg_heur":
        args["n_restart"] = a.restarts
    return args


def output(a, stem):
    heur = "_R%d" % a.restarts if a.method == "ip_cg_heur" else ""
    return os.path.join(HERE, "results", "%s_%s%s%s.json" % (stem, a.method, heur, a.tag))


def race(a):
    refs = json.load(open(os.path.join(DATA, a.set, "opt_ref.json")))
    if a.key not in refs:
        raise SystemExit("%s has no cell %r; its cells: %s" % (a.set, a.key, " ".join(refs)))
    ref = refs[a.key]
    cd = np.loadtxt(os.path.join(DATA, a.set, ref["file"]), delimiter=",", skiprows=1)
    if fingerprint(cd) != ref["fingerprint"]:
        raise SystemExit("%s %s does not match the fingerprint stored with its OPT" % (a.set, a.key))
    n, opt = int(ref["n"]), float(ref["opt"])
    rum = RUM(cd[:, 0].astype(np.uint64), cd[:, 1].astype(np.int64), cd[:, 2], num_alternatives=n)
    target = opt + max(REL * opt, ABS)
    hours = a.cap_hours or (168.0 if a.set == "Numerical2" and n >= 32 else 24.0)
    print("[%s %s] n = %d, m_p = %d, OPT = %.6f: stop at UB < %.6f or after %g h"
          % (a.set, a.key, n, rum.M, opt, target, hours), flush=True)
    c0 = time.process_time()
    with rum.session(method=a.method, method_args=method_args(a)) as s:
        r = s.solve(rum.choice_probability, abs_tol=0.0, rel_tol=0.0, max_time=3600.0 * hours, below=target,
                    verbose=a.verbose)
    reached = r.ub < target
    rec = dict(set=a.set, key=a.key, n=n, mp=rum.M, opt=opt, target=target, cap_h=hours, reached=bool(reached),
               table="%.2f" % r.time if reached else ">= %g h" % hours, process_cpu_s=time.process_time() - c0,
               host=socket.gethostname(), when=time.strftime("%Y-%m-%d %H:%M:%S"), result=r.to_dict())
    out = output(a, "%s_%s" % (a.set, a.key))
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print("[%s %s] %s: UB = %.6f after %d columns, %.1f s (model build %.1f s) -> table entry %s"
          % (a.set, a.key, r.status, r.ub, r.iterations, r.time, r.time_setup, rec["table"]), flush=True)
    print("wrote", out)


def bootstrap(a, rum, stem, **data):
    out = output(a, stem)
    with rum.bootstrap(B=a.B, seed=a.seed, abs_tol=a.abs_tol, rel_tol=a.rel_tol, refine=a.refine,
                       max_time=3600.0 * a.cert_hours, draw_time=3600.0 * a.draw_hours, method=a.method,
                       method_args=method_args(a), checkpoint=out + ".checkpoint", verbose=True, **data) as boot:
        print(boot.distance_interval(alpha=a.alpha), flush=True)   # certifies p_hat and screens; solves no draw
        cert = boot.certificates[-1]
        if cert["exit_flag"] > 0:
            print(boot.p_value(alpha=a.alpha), flush=True)
        else:
            print("p_hat not certified to abs_tol %g within %g h (%s; %s): the table's '>= %g h'"
                  % (boot.abs_tol, a.cert_hours, cert["status"], cert["message"], a.cert_hours), flush=True)
        print(boot, flush=True)
        boot.save(out)
    if os.path.exists(out + ".checkpoint"):
        os.remove(out + ".checkpoint")
    print("wrote", out)


def dsr(a):
    panel = pd.read_csv(DSR)
    menu, alt, p_hat = cells_from_panel(panel, 25, alternatives=[0] if a.obs == "alt0" else None)
    bootstrap(a, RUM(menu, alt, p_hat, num_alternatives=25), "dsr_" + a.obs, n=N_DSR, panel=panel)


def ai(a):
    table, dropped = load_table(os.path.join(DATA, "Application_AI", "raw_data", AI[a.model]))
    print("%s: %d replies kept, %d dropped" % (AI[a.model], len(table), dropped), flush=True)
    menu, alt, p_hat = cells_from_panel(table, 18)
    bootstrap(a, RUM(menu, alt, p_hat, num_alternatives=18), "ai_" + a.model, panel=table)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="run", required=True)
    r = sub.add_parser("race", help="one synthetic cell: tab:density18-20, tab:binary, tab:dense")
    r.add_argument("--set", required=True, choices=["Numerical1", "Numerical2", "Numerical3"])
    r.add_argument("--key", required=True, help="a cell of final_data/<set>/opt_ref.json, e.g. n25_sparse")
    r.add_argument("--cap-hours", type=float, help="default: 168 for Numerical2 with n >= 32, else 24")
    r.add_argument("--verbose", action="store_true", help="one line per column")
    d = sub.add_parser("dsr", help="tab:dsg")
    d.add_argument("--obs", required=True, choices=["alt0", "full"])
    i = sub.add_parser("ai", help="tab:ai")
    i.add_argument("--model", required=True, choices=sorted(AI))
    for p in (d, i):
        p.add_argument("--B", type=int)
        p.add_argument("--seed", type=int)
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--abs-tol", type=float)
        p.add_argument("--rel-tol", type=float)
        p.add_argument("--refine", type=int)
        p.add_argument("--cert-hours", type=float, default=24.0, help="cap on each certificate of p_hat")
        p.add_argument("--draw-hours", type=float, default=24.0,
                       help="cap on each solve of a draw; a draw left undecided counts 1")
    for p in (r, d, i):
        p.add_argument("--method", required=True, choices=["ip_cg", "ip_cg_heur"],
                       help="ip_cg = IP-CG, ip_cg_heur = IP-H-CG")
        p.add_argument("--restarts", type=int, default=10,
                       help="ip_cg_heur: random-insertion restarts per pricing call (the paper: 10)")
        p.add_argument("--threads", type=int, help="Gurobi threads (the paper: 24); default: Gurobi's choice")
        p.add_argument("--tag", default="", help="suffix of the output name")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    {"race": race, "dsr": dsr, "ai": ai}[a.run](a)


if __name__ == "__main__":
    main()
