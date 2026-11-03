"""The DSR application through rum.bootstrap: the Hong-Li p-value and confidence interval.

    python replication/empirical_dsr_n25/run_bootstrap.py --obs alt0
    python replication/empirical_dsr_n25/run_bootstrap.py --obs full --no-ci

Data: final_data/Application_DSR/raw_data/overload_data_25_long.csv, 520 respondents x 10 menus,
already a panel (respondent, menu, chosen). alt0 keeps the reference option's share of every menu
(M = 301), full every alternative (M = 901). Defaults: solver/defaults.py BOOTSTRAP (B = 10,000
multinomial_reject draws, seed 0, abs 5e-4, rel 0) and the solver's automatic method. Writes
results/bootstrap_dsr_<obs><tag>.json; while it runs, a checkpoint next to it resumes a killed run.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from solver.rum_solver import RUM                    # noqa: E402
from solver.inference import cells_from_panel        # noqa: E402

DATA = os.path.join(ROOT, "final_data", "Application_DSR", "raw_data", "overload_data_25_long.csv")
N = 520                 # the sample size: respondents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=["alt0", "full"], default="alt0")
    ap.add_argument("--B", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--abs-tol", type=float)
    ap.add_argument("--rel-tol", type=float)
    ap.add_argument("--refine", type=int)
    ap.add_argument("--draw-time", type=float)
    ap.add_argument("--method", help="a solver method instead of the automatic choice")
    ap.add_argument("--no-ci", action="store_true", help="the p-value only")
    ap.add_argument("--tag", default="", help="suffix of the output name")
    a = ap.parse_args()
    panel = pd.read_csv(DATA)
    menu, alt, p_hat = cells_from_panel(panel, 25, alternatives=[0] if a.obs == "alt0" else None)
    rum = RUM(menu, alt, p_hat, num_alternatives=25)
    out = os.path.join(HERE, "results", "bootstrap_dsr_%s%s.json" % (a.obs, a.tag))
    with rum.bootstrap(n=N, panel=panel, B=a.B, seed=a.seed, abs_tol=a.abs_tol, rel_tol=a.rel_tol, refine=a.refine,
                       draw_time=a.draw_time, method=a.method, checkpoint=out + ".checkpoint", verbose=True) as boot:
        print(boot.p_value(alpha=a.alpha), flush=True)
        if not a.no_ci:
            print(boot.confidence_interval(alpha=a.alpha), flush=True)
        print(boot, flush=True)
        boot.save(out)
    os.remove(out + ".checkpoint")
    print("wrote", out)


if __name__ == "__main__":
    main()
