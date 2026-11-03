"""Bootstrap.distance_interval, the two-sided interval for dist (CPU; Gurobi flow_qp on toys, one thread).

    python tests/test_intervals.py

A  on a toy (replies and panel input): the interval equals [max(0, L0 - c), U0 + c] with c the 95 % quantile of
   ||p_hat_b - p_hat|| computed here from the draws directly; it holds [d - c, d + c] for d in [L0, U0]; the record
   carries it; a checkpoint resume gives the same interval; alpha changes c the right way.
B  vcov: with a diagonal weight, c is the quantile of the weighted norm.
C  the AI data (the paper's two intervals): c from the draws of gpt6luna_noreason and gmicloud as replies equals the
   independent computation (1.4472 and 1.7673), and the interval reproduces the paper's
   [11.942, 14.836] and [5.453, 8.988] to 1e-3 (distance bracket from the stored certificate).
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "replication", "empirical_ai_n18"))
from solver import defaults                                   # noqa: E402
from solver.rum_solver import RUM                             # noqa: E402
from solver.inference import MenuDraws, cells_from_panel, hong_li   # noqa: E402
from replies import load_table                                # noqa: E402
from run_bootstrap import FILES                               # noqa: E402

for m in ("flow_qp", "bm_qp"):
    defaults.METHOD_ARGS[m]["threads"] = 1
FAIL = []


def check(name, ok, detail=""):
    print("%-4s %s  %s" % ("ok" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAIL.append(name)


def toy(seed, per_menu=12, hi=1.6, lo=0.7):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, 32):
        items = [a for a in range(5) if (m >> a) & 1]
        if len(items) < 2:
            continue
        w = np.ones(len(items))
        if 0 in items and 1 in items:
            w[items.index(0)] = hi if 2 in items else lo
        for r in range(per_menu):
            rows.append((r, m, int(rng.choice(items, p=w / w.sum()))))
    return pd.DataFrame(rows, columns=["respondent", "menu", "chosen"])


# ---------------------------------------------------------------- A
tab = toy(7)
menu, alt, p_hat = cells_from_panel(tab, 5)
rum = RUM(menu, alt, p_hat, num_alternatives=5)
for style in ("replies", "panel"):
    with rum.bootstrap(**{style: tab}, B=300, method="flow_qp") as b:
        di = b.distance_interval()
        dev = np.concatenate([np.sqrt(((blk - p_hat) ** 2).sum(1)) for _lo, _hi, blk in b.draws.blocks()])
        c = float(np.quantile(dev, 0.95))
        rec = json.loads(json.dumps(b.record(), default=float))
        wide = b.distance_interval(alpha=0.2)
        check("A %s formula" % style, abs(di.half_width - c) < 1e-12 and abs(di.lower - max(0.0, b.L0 - c)) < 1e-12
              and abs(di.upper - (b.U0 + c)) < 1e-12 and di.lower <= max(0.0, b.L0 - c) + 1e-15 and di.lower >= 0.0,
              "c = %.6f, interval [%.6f, %.6f], dist(p_hat) in [%.6f, %.6f]" % (c, di.lower, di.upper, b.L0, b.U0))
        check("A %s record" % style, rec["results"]["distance_interval"]["0.05"]["upper"] == di.upper
              and wide.half_width < di.half_width, "alpha 0.2 half-width %.6f < %.6f" % (wide.half_width, di.half_width))
ck = os.path.join(tempfile.mkdtemp(), "t.json")
with rum.bootstrap(replies=tab, B=300, method="flow_qp", checkpoint=ck) as b1:
    d1 = b1.distance_interval()
with rum.bootstrap(replies=tab, B=300, method="flow_qp", checkpoint=ck) as b2:
    d2 = b2.distance_interval()
check("A resume", abs(d1.half_width - d2.half_width) < 1e-12 and abs(d1.upper - d2.upper) < 1e-6,
      "[%.6f, %.6f] then [%.6f, %.6f]" % (d1.lower, d1.upper, d2.lower, d2.upper))

# ---------------------------------------------------------------- B
w = np.random.default_rng(1).uniform(0.5, 2.0, menu.size)
with rum.bootstrap(replies=tab, B=300, method="flow_qp", vcov=sp.diags(w).tocsr()) as b:
    di = b.distance_interval()
    dev = np.concatenate([np.sqrt((((blk - p_hat) ** 2) * w).sum(1)) for _lo, _hi, blk in b.draws.blocks()])
    check("B vcov", abs(di.half_width - float(np.quantile(dev, 0.95))) < 1e-12, "weighted c = %.6f" % di.half_width)

# ---------------------------------------------------------------- C
PAPER = {"gpt6luna_noreason": ((11.942, 14.836), 1.4472), "gmicloud": ((5.453, 8.988), 1.7673)}
for model, ((plo, phi), cref) in PAPER.items():
    table, _ = load_table(os.path.join(W, "final_data", "Application_AI", "raw_data", FILES[model]))
    m, a, p = cells_from_panel(table, 18)
    D = MenuDraws(table, m, a, p, B=10000, seed=0)
    dev = np.concatenate([np.sqrt(((blk - p) ** 2).sum(1)) for _lo, _hi, blk in D.blocks()])
    rec = json.load(open(os.path.join(W, "replication", "empirical_ai_n18", "results", "bootstrap_ai_%s.json" % model)))
    lo, hi, c = hong_li.distance_interval(rec["p_hat"]["lb"], rec["p_hat"]["ub"], dev, 0.05)
    check("C %s" % model, abs(c - cref) < 1e-4 and abs(lo - plo) < 1e-3 and abs(hi - phi) < 1.5e-3,
          "c = %.4f; [%.4f, %.4f] against the paper's [%.3f, %.3f]" % (c, lo, hi, plo, phi))

print("\n%d failed%s" % (len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))

sys.exit(1 if FAIL else 0)
