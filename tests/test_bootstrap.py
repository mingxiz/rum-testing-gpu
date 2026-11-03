"""rum.bootstrap with both inputs, panel= (DSR style) and replies= (AI style). CPU only: Gurobi flow_qp on toys,
one thread; a few minutes.

    python tests/test_bootstrap.py

A  MenuDraws on a toy (5 alternatives, all 26 menus of 2+, 60 replies each): every draw resamples
   each menu (shares sum to 1, counts are integers), each row's mean and variance are those of
   Multinomial(N_D, p_hat)/N_D, rows of different menus are uncorrelated; the same seed gives the
   same draws through blocks(), block rebuilds and rows(idx); another seed other draws; a rum with
   alternative 0 only works.
B  the documented errors.
C  rum.bootstrap(n, replies=...) end to end (flow_qp) on a toy near the threshold (12 replies per
   menu, a mild decoy): p-value and interval, the record (input, fingerprints, the replies summary),
   and a resume from the checkpoint that reproduces both and solves nothing again; the panel input
   on the same toy (replicate as respondent) still records as before.
D  the 100-reply AI data as replies: every draw below tau_lo by the no-solve bound U0 + eps_n r_n ||p_b - p_hat||
   with the stored run's p_hat certificate (p_hat, hence [L0, U0] and tau, do not depend on the input).
E  n from the data when not given: the panel's unique respondents (toy 12, DSR 520, AI 100), the most replies
   on any menu (toy 12, also when one menu keeps 10; AI 100, DeepSeek's menus 94 to 100); n positional still
   works, and a given n is still checked.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "replication", "empirical_ai_n18"))
from solver import defaults                                                  # noqa: E402
from solver.rum_solver import RUM                                            # noqa: E402
from solver.inference import MenuDraws, cells_from_panel, draws as dr        # noqa: E402
from replies import load_table                                              # noqa: E402

for m in ("flow_qp", "bm_qp"):
    defaults.METHOD_ARGS[m]["threads"] = 1
OUT, FAIL = {}, []


def check(name, ok, detail=""):
    print("%-4s %s  %s" % ("ok" if ok else "FAIL", name, detail), flush=True)
    OUT[name] = dict(ok=bool(ok), detail=str(detail))
    if not ok:
        FAIL.append(name)


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        return text in str(e), str(e)
    return False, "no error"


def toy_replies(seed, per_menu=60, hi=3.0, lo=0.4):
    """Every menu of 2+ of 5 alternatives asked per_menu times, with a decoy effect (not a RUM):
    alternative 0 has weight hi next to 1 and 2, lo next to 1 alone."""
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
    return pd.DataFrame(rows, columns=["replicate", "menu", "chosen"])


# ---------------------------------------------------------------- A: the draws
rep = toy_replies(7)
menu, alt, p_hat = cells_from_panel(rep, 5)
M, B = menu.size, 4000
D = MenuDraws(rep, menu, alt, p_hat, B=B, seed=0)
X = np.vstack([blk for _, _, blk in D.blocks()])
N = D.N[np.unique(menu, return_inverse=True)[1]]
share_sum = np.zeros((B, D.N.size))
np.add.at(share_sum.T, np.unique(menu, return_inverse=True)[1], X.T)
check("A shares sum to 1", np.abs(share_sum - 1).max() < 1e-12, "max |sum - 1| = %.1e over %d draws x %d menus"
      % (np.abs(share_sum - 1).max(), B, D.N.size))
check("A integer counts", np.abs(X * N - np.round(X * N)).max() < 1e-9, "N_D p_b integral")
sd = np.sqrt(p_hat * (1 - p_hat) / N)
z = np.abs(X.mean(0) - p_hat) / np.maximum(sd / np.sqrt(B), 1e-300)
check("A mean", (z[sd > 0] < 4.5).all(), "max |mean - p_hat| / se = %.2f" % z[sd > 0].max())
big = p_hat * (1 - p_hat) > 0.05
rel = np.abs(X.var(0)[big] / sd[big] ** 2 - 1)
check("A variance", rel.max() < 0.12, "max |var / (p(1-p)/N) - 1| = %.3f over %d rows" % (rel.max(), big.sum()))
C = np.corrcoef(X[:, sd > 0].T)
mi = np.unique(menu, return_inverse=True)[1][sd > 0]
cross = np.abs(C[mi[:, None] != mi[None, :]])
check("A menus independent", cross.max() < 0.09, "max |corr| across menus %.3f (B = %d)" % (cross.max(), B))
saved = dr.BLOCK_BYTES
dr.BLOCK_BYTES = 8 * M * 300                        # 300 draws per block
Db = MenuDraws(rep, menu, alt, p_hat, B=1000, seed=3)
Xb = np.vstack([blk for _, _, blk in Db.blocks()])
again = MenuDraws(rep, menu, alt, p_hat, B=1000, seed=3)
_ = [blk for _, _, blk in again.blocks()]
b1, b0 = Db.block(1).copy(), Db.block(0).copy()
idx = np.array([999, 3, 450, 301])
check("A blocks", Db.n_blocks == 4 and np.array_equal(b0, Xb[:300]) and np.array_equal(b1, Xb[300:600])
      and np.array_equal(Db.rows(idx), Xb[idx]) and again.fingerprint == Db.fingerprint,
      "4 blocks of 300; rebuilt blocks and rows(idx) equal the pass; same seed, same fingerprint %s" % Db.fingerprint)
other = MenuDraws(rep, menu, alt, p_hat, B=1000, seed=4)
_ = [blk for _, _, blk in other.blocks()]
check("A seed", other.fingerprint != Db.fingerprint, "seed 4 fingerprint %s" % other.fingerprint)
dr.BLOCK_BYTES = saved
m0, a0, p0 = cells_from_panel(rep, 5, alternatives=[0])
D0 = MenuDraws(rep, m0, a0, p0, B=B, seed=0)
X0 = np.vstack([blk for _, _, blk in D0.blocks()])
z0 = np.abs(X0.mean(0) - p0) / (np.sqrt(p0 * (1 - p0) / D0.N) / np.sqrt(B))
check("A alternative 0 only", D0.M == 15 and (z0 < 4.5).all() and D0.rows_unused == 11 * 60,
      "M = %d, max z %.2f, %d replies on menus without alternative 0 unused" % (D0.M, z0.max(), D0.rows_unused))

# ---------------------------------------------------------------- B: errors
rum = RUM(menu, alt, p_hat, num_alternatives=5)
panel = rep.rename(columns={"replicate": "respondent"})
bad_row = int(np.flatnonzero(rep["menu"] == 3)[0])      # menu {0, 1}: alternative 4 is not in it
cases = {
    "missing column": (lambda: MenuDraws(rep.drop(columns=["chosen"]), menu, alt, p_hat, B=10),
                       "the replies table needs the columns menu and chosen; missing: chosen"),
    "chosen not in menu": (lambda: cells_from_panel(rep.drop(columns=["replicate"]).assign(
        chosen=np.where(np.arange(len(rep)) == bad_row, 4, rep["chosen"])), 5),
        "replies table row %d: chosen 4 is not in its menu (mask 3)" % bad_row),
    "menu not in replies": (lambda: MenuDraws(rep[rep["menu"] != menu[0]], menu, alt, p_hat, B=10),
                            "rum row 0 (menu %d): the menu is not in the replies" % menu[0]),
    "p_hat mismatch": (lambda: MenuDraws(rep, menu, alt, np.where(np.arange(M) == 4, 0.7, p_hat), B=10),
                       "rum row 4 (menu %d, alternative %d): p_hat = 0.7, but the replies give" % (menu[4], alt[4])),
    "n out of range": (lambda: rum.bootstrap(n=59, replies=rep), "n = 59, but the menus have 60 to 60 replies"),
    "both inputs": (lambda: rum.bootstrap(n=60, panel=panel, replies=rep), "give the choices behind p_hat once"),
    "no input": (lambda: rum.bootstrap(n=60), "give the choices behind p_hat once"),
    "weights": (lambda: rum.bootstrap(n=60, replies=rep, weights="exponential"),
                "weights = 'exponential' is for panel="),
}
for name, (fn, text) in cases.items():
    ok, msg = raises(fn, text)
    check("B %s" % name, ok, msg)

# ---------------------------------------------------------------- C: end to end
rep12 = toy_replies(7, per_menu=12, hi=1.6, lo=0.7)
m12, a12, p12 = cells_from_panel(rep12, 5)
rum12 = RUM(m12, a12, p12, num_alternatives=5)
t0 = time.perf_counter()
with rum12.bootstrap(n=12, replies=rep12, B=300, method="flow_qp") as boot:
    pv, ci = boot.p_value(), boot.confidence_interval()
    rec = json.loads(json.dumps(boot.record(), default=float))
print(pv, "\n", ci, flush=True)
check("C p-value and interval", 0 < pv.p < 1 and pv.n_solved > 0 and ci.lower_bracket[0] <= ci.lower_bracket[1],
      "p = %.4f (%d/%d/%d; %d screened, %d solved), interval [%.5f, inf) with %d more solved, %.1f s"
      % (pv.p, pv.n_below, pv.n_above, pv.n_undecided, pv.n_screened, pv.n_solved, ci.lower, ci.n_solved,
         time.perf_counter() - t0))
check("C record", rec["settings"]["input"] == "replies" and rec["settings"]["weights"] == "multinomial"
      and set(rec["fingerprints"]) == {"data", "replies", "draws"} and "panel" not in rec
      and rec["replies"] == dict(replies=26 * 12, replies_per_menu=[12, 12], rows_unused=0),
      "settings.input %s, fingerprints %s, replies %s" % (rec["settings"]["input"], sorted(rec["fingerprints"]),
                                                          rec["replies"]))
ck = os.path.join(tempfile.mkdtemp(), "toy.json")
with rum12.bootstrap(n=12, replies=rep12, B=300, method="flow_qp", checkpoint=ck) as b1:
    p1, c1 = b1.p_value(), b1.confidence_interval()
    solved1 = int((b1.flag != 127).sum())
with rum12.bootstrap(n=12, replies=rep12, B=300, method="flow_qp", checkpoint=ck) as b2:
    resumed = int((b2.flag != 127).sum())
    p2, c2 = b2.p_value(), b2.confidence_interval()
    solved2 = int((b2.flag != 127).sum())
check("C resume", (p1.p, c1.lower) == (p2.p, c2.lower) == (pv.p, ci.lower) and resumed == solved1 == solved2 > 0,
      "p %.4f and lower end %.5f again after resuming with %d solved draws, none solved again"
      % (p2.p, c2.lower, resumed))
panel12 = rep12.rename(columns={"replicate": "respondent"})
with rum12.bootstrap(n=12, panel=panel12, B=300, method="flow_qp") as bp:
    pp = bp.p_value()
    recp = json.loads(json.dumps(bp.record(), default=float))
check("C panel input", recp["settings"]["input"] == "panel" and set(recp["fingerprints"]) == {"data", "panel", "draws"}
      and recp["panel"]["respondents"] == 12 and recp["fingerprints"]["data"] == rec["fingerprints"]["data"],
      "p = %.4f with the replicate as respondent (replies: %.4f); panel %s" % (pp.p, pv.p, recp["panel"]))

# ---------------------------------------------------------------- D: the AI data
RAW = os.path.join(W, "final_data", "Application_AI", "raw_data")
for model, fname in (("luna", "gpt5.6-luna.npz"), ("deepseek", "deepseekv4-flash.npz")):
    t0 = time.perf_counter()
    table, dropped = load_table(os.path.join(RAW, fname))
    menu, alt, p_hat = cells_from_panel(table, 18)
    pub = json.load(open(os.path.join(W, "replication", "empirical_ai_n18", "results", "bootstrap_ai_%s.json" % model)))
    er = pub["settings"]["eps_n"] * pub["settings"]["r_n"]
    U0, tau_lo = pub["p_hat"]["ub"], pub["tau"][0]
    t1 = time.perf_counter()
    Dm = MenuDraws(table, menu, alt, p_hat, B=10000, seed=0)
    Dm.check_n(100)
    d = np.empty(Dm.B)
    for lo, hi, blk in Dm.blocks():
        d[lo:hi] = np.sqrt(np.einsum("ij,ij->i", blk - p_hat, blk - p_hat))
    t2 = time.perf_counter()
    bound = (U0 + er * d) / tau_lo
    check("D %s below" % model, (bound < 1).all(),
          "all %d draws below tau_lo with no solve, max bound / tau_lo %.3f; %s; draws %.0f s (fingerprint %s)"
          % (Dm.B, bound.max(), Dm.summary(), t2 - t1, Dm.fingerprint))
    rum_ai = RUM(menu, alt, p_hat, num_alternatives=18)
    with rum_ai.bootstrap(replies=table, B=10) as br, rum_ai.bootstrap(panel=table, B=10) as bp:
        check("E %s n" % model, br.n == bp.n == 100 and br.settings["n"] == 100,
              "n = %d from the replies (menus %d to %d), %d from the panel" % (br.n, br.draws.N.min(), br.draws.N.max(),
                                                                           bp.n))

# ---------------------------------------------------------------- E: n from the data (toys and DSR)
with rum12.bootstrap(panel=panel12, B=50) as b:
    e_panel = (b.n, b.settings["n"], b.r_n, b.eps_n) == (12, 12, np.sqrt(12), 12 ** (-1 / 3))
with rum12.bootstrap(replies=rep12, B=50) as b:
    e_replies = b.n == 12
uneven = rep12[~((rep12["menu"] == m12[0]) & (rep12["replicate"] < 2))]
mu, au, pu = cells_from_panel(uneven, 5)
with RUM(mu, au, pu, num_alternatives=5).bootstrap(replies=uneven, B=50) as b:
    e_uneven = (b.n, int(b.draws.N.min())) == (12, 10)
with rum12.bootstrap(12, panel12, B=50) as b:
    e_pos = b.n == 12
e_given, msg = raises(lambda: rum12.bootstrap(n=11, panel=panel12, B=50), "n = 11, but the panel has 12 respondents")
check("E toy n", e_panel and e_replies and e_uneven and e_pos and e_given,
      "panel 12 (r_n, eps_n from it), replies 12, one menu at 10 replies still 12, positional n works; %s" % msg)
dsr = pd.read_csv(os.path.join(W, "final_data", "Application_DSR", "raw_data", "overload_data_25_long.csv"))
md, ad, pdh = cells_from_panel(dsr, 25, alternatives=[0])
with RUM(md, ad, pdh, num_alternatives=25).bootstrap(panel=dsr, B=50) as b:
    check("E DSR n", b.n == 520, "n = %d from the panel" % b.n)

print("\n%d checks, %d failed%s" % (len(OUT), len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))

sys.exit(1 if FAIL else 0)
