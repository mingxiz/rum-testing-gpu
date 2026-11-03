# tab:dsr: the choice-overload data of Dean, Ravindran and Stoye (2022), n = 25

The Hong-Li bootstrap test of RUM on the DSR data, through `rum.bootstrap` (`solver/inference/`): the p-value,
the certified dist(p&#770;) and the Flow-CG time of tab:dsr. The IP-CG and IP-H-CG columns are cluster runs; see
[`../cgip/`](../cgip/).

## The data

520 respondents choosing over **n = 25** lotteries, 10 questions each. Alternative 0 is the safe reference
option and appears in every menu: 24 binary menus {0, j}, 276 ternary menus {0, i, j} and the grand menu of all
25 (the overload test). The small menus are thin, seen by about 15 respondents each; the grand menu by all 520.

Two observation sets: **alt0** (tab:dsr's Default Only), only the reference option's frequency in each menu,
M = 301 cells; and **full** (All Alternatives), every alternative of every menu, M = 901.

`final_data/Application_DSR/raw_data/overload_data_25_long.csv` has one row per respondent and question
(`respondent`, `question`, `menu` as a bitmask, `chosen`, `menu_size`, `menu_items`), so respondents can be
resampled; `overload_data_25.csv` is the same data, one row per respondent.

## Run it

    python replication/empirical_dsr_n25/run_bootstrap.py --obs alt0                        # about 11.5 min, + 4 min for the interval
    python replication/empirical_dsr_n25/run_bootstrap.py --obs full                        # about 25 min, + 11 min for the interval
    python replication/empirical_dsr_n25/run_bootstrap.py --obs alt0 --B 1000 --tag _B1000   # and full

`--no-ci` skips the one-sided interval. The panel is the CSV itself. The settings are the class's
defaults (`solver/defaults.py`):

- B = 10,000 **multinomial_reject** draws, seed 0: the respondents are resampled with replacement, and a resample
  that leaves a menu unseen is drawn again (72 of 10,000 here), so every menu is seen in every draw;
- n = 520 respondents, so r_n = 22.80, eps_n = 0.1244 and scale = 3.8357;
- dist(p&#770;) certified to abs 5e-4 (rel 0); the threshold band is tau = scale x [LB, UB];
- every draw is first screened by its distance to the RUM point fitted at p&#770;; the rest restart from the
  p&#770; solution and stop as *below* once UB < tau_lo or *above* once LB >= tau_hi; up to two tenfold tighter
  rounds (5e-5, 5e-6) settle the draws still undecided;
- the solver is the class's automatic choice, Flow-CG (`flow_cg`, float32 pricing).

A run writes `results/bootstrap_dsr_<obs><tag>.json`; while it runs, a checkpoint next to it lets the same
command resume a killed run.

## Results

RTX 5090; `CGBFQP-GPU` is the records' name for Flow-CG.

| obs | M | B | dist(p&#770;) in [LB, UB] | below / above / undecided | p-value | time | 95 % interval |
|---|---|---|---|---|---|---|---|
| alt0 | 301 | 10,000 | [1.489091, 1.489091] | 8,917 / 1,083 / 0 | 0.1083 | 688 s | [-0.0476, &infin;) |
| full | 901 | 10,000 | [2.298668, 2.298669] | 8,608 / 1,392 / 0 | 0.1392 | 1,499 s | [-0.0902, &infin;) |
| alt0 | 301 | 1,000 | [1.489091, 1.489091] | 892 / 108 / 0 | 0.1080 | 69.8 s | [-0.0441, &infin;) |
| full | 901 | 1,000 | [2.298204, 2.298678] | 857 / 143 / 0 | 0.1430 | 152 s | [-0.0963, &infin;) |

RUM is not rejected at 5 % on either set. Every draw is decided in one run; without the two tighter rounds
2 draws (alt0) and 13 (full) of the 10,000 stay undecided at 5e-4. The intervals contain 0, consistent with
p > 0.05. The IP-CG and IP-H-CG runs on the cluster reach the same decision on every draw
(`../cgip/cluster_records/Application_DSR/`).

BM-ADMM is not in tab:dsr: at n = 25 it has to run in float32, and on alt0 it did not certify dist(p&#770;)
within a 4 h cap, where Flow-CG needs 0.8 s. M = 301 cells against a lattice of 4.2 x 10^8 is the far sparse
end of the paper's crossover.

`final_results/Application_DSR/` lists these runs with every column.

## Files

| file | what it is |
|---|---|
| `run_bootstrap.py` | the test through `rum.bootstrap`: the p-value and the one-sided interval |
| `results/bootstrap_dsr_{alt0,full}.json` | the B = 10,000 records: settings, fingerprints, the p&#770; certificates, every draw's bracket |
| `results/bootstrap_dsr_{alt0,full}_B1000.json` | the same at B = 1,000 |
