"""The AI application through rum.bootstrap: the Hong-Li p-value and confidence interval.

    python replication/empirical_ai_n18/run_bootstrap.py --model luna
    python replication/empirical_ai_n18/run_bootstrap.py --model deepseek --no-ci
    python replication/empirical_ai_n18/run_bootstrap.py --model luna --method flow_cg --no-ci --tag _flow_cg
    python replication/empirical_ai_n18/run_bootstrap.py --model luna --input replies --no-ci
    python replication/empirical_ai_n18/run_bootstrap.py --model gpt6luna_noreason --no-ci
    python replication/empirical_ai_n18/run_bootstrap.py --model gpt6luna_noreason --input replies --no-ci --distance-interval

Data: final_data/Application_AI/raw_data/, the 8,807 menus of 18 products put to each model 100
times (gpt-5.6-luna, deepseek-v4-flash: gpt5.6-luna.npz, deepseekv4-flash.npz) or 2,000 times
(gpt-6 luna without reasoning, and GMI Cloud: gpt6luna_noreason.npz, gmicloud.npz): the tables
replies.py reads from the models' jsonl replies, saved by replies.save_table. Unparseable replies
are dropped. Every member alternative is a row:
M = 73,492. --input panel (the default) passes them as a panel with the replicate as the
respondent, as the published runs did; --input replies passes them as replies, each menu's
replies resampled on their own. n is taken from the data: the replicates (100 or 2,000), which
are also the replies per menu. Defaults:
solver/defaults.py BOOTSTRAP (B = 10,000 draws, multinomial_reject for the panel, seed 0, abs
5e-4, rel 0) and BM-ADMM with the published runs' settings (beta 5, an upper bound every 25
iterations, a lower bound every 100); --method auto takes the solver's automatic choice, and any
other method name that method with the class's settings.
Writes results/bootstrap_ai_<model><tag>.json (<model>_replies<tag> with --input replies).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from solver.rum_solver import RUM                    # noqa: E402
from solver.inference import cells_from_panel        # noqa: E402
from replies import load_table                       # noqa: E402

FILES = {"luna": "gpt5.6-luna.npz", "deepseek": "deepseekv4-flash.npz",
         "gpt6luna_noreason": "gpt6luna_noreason.npz", "gmicloud": "gmicloud.npz"}
PUBLISHED = dict(beta=5.0, check=25, lb_every=4)     # the published BM-ADMM settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(FILES), default="luna")
    ap.add_argument("--B", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--abs-tol", type=float)
    ap.add_argument("--rel-tol", type=float)
    ap.add_argument("--refine", type=int)
    ap.add_argument("--draw-time", type=float)
    ap.add_argument("--method", default="published",
                    help="published (BM-ADMM, the published settings), auto, or a method name")
    ap.add_argument("--input", choices=("panel", "replies"), default="panel",
                    help="panel: the replicate as the respondent (the published runs); replies: each menu's replies "
                         "resampled on their own")
    ap.add_argument("--no-ci", action="store_true", help="no one-sided Hong-Li interval (the p-value only)")
    ap.add_argument("--distance-interval", action="store_true",
                    help="also the two-sided interval of the statistical supplement (no draw solved)")
    ap.add_argument("--tag", default="", help="suffix of the output name")
    a = ap.parse_args()
    table, dropped = load_table(os.path.join(ROOT, "final_data", "Application_AI", "raw_data", FILES[a.model]))
    print("%s: %d replies kept, %d dropped" % (FILES[a.model], len(table), dropped), flush=True)
    menu, alt, p_hat = cells_from_panel(table, 18)
    rum = RUM(menu, alt, p_hat, num_alternatives=18)
    out = os.path.join(HERE, "results", "bootstrap_ai_%s%s%s.json"
                       % (a.model, "_replies" if a.input == "replies" else "", a.tag))
    method, args = {"published": ("bm_admm", PUBLISHED), "auto": (None, None)}.get(a.method, (a.method, None))
    with rum.bootstrap(**{a.input: table}, B=a.B, seed=a.seed, abs_tol=a.abs_tol, rel_tol=a.rel_tol,
                       refine=a.refine, draw_time=a.draw_time, method=method, method_args=args,
                       checkpoint=out + ".checkpoint", verbose=True) as boot:
        print(boot.p_value(alpha=a.alpha), flush=True)
        if not a.no_ci:
            print(boot.confidence_interval(alpha=a.alpha), flush=True)
        if a.distance_interval:
            print(boot.distance_interval(alpha=a.alpha), flush=True)
        print(boot, flush=True)
        boot.save(out)
    os.remove(out + ".checkpoint")
    print("wrote", out)


if __name__ == "__main__":
    main()
