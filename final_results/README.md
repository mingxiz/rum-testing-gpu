# final_results

The numbers behind the paper's numerical tables, one row per (instance, lambda, solver). The
layout follows `final_data/`. Regenerate with `python validation/make_final_results.py`; it reads
the same records as `validation/make_lambda_tables.py`, so `table_entry` is exactly what the .tex
tables print.

| file | paper tables | solvers |
|---|---|---|
| `Numerical1/results_n18.csv`, `results_n19.csv`, `results_n20.csv` | tab:density18, tab:density19, tab:density20 (`tab_race_n{18,19,20}_l{0,05,1}.tex`) | BM-ADMM, CGBFQP-GPU, Flow-ADMM; CGIP (lambda = 0) |
| `Numerical2/results_pairwise.csv` | tab:binary (`tab_fes_pairwise_l{0,05,1}.tex`), n = 25, 27, 30, 32, 35 | CGBFQP-GPU, CGIP, CGIP-heur |
| `Numerical3/results_density.csv` | tab:dense (`tab_fes_density_l{0,05,1}.tex`), n = 20, density 0.1, 0.5, 1, 5 % | BM-ADMM, CGIP, CGIP-heur |
| `Application_DSR/results_dsr.csv` | tab:dsg: the DSR application, n = 25, alt0 and full (see `Application_DSR/README.md`) | CGBFQP-GPU (B = 1,000 and 10,000); CGIP, CGIP-heur (cluster, B = 10,000) |
| `Application_AI/results_ai.csv` | tab:ai and the two AI intervals: the LLM application, n = 18: gpt-5.6-luna and deepseek-v4-flash (100 replies per menu), gpt-6 luna without reasoning and GMI Cloud (2,000) (see `Application_AI/README.md`) | BM-ADMM, CGBFQP-GPU; CGIP, CGIP-heur (capped at 24 h) |

The CGIP and CGIP-heur rows are CPU cluster runs made with an earlier version of the CGIP code. Their result files
are in `replication/cgip/cluster_records/<experiment>/` (24 Gurobi threads per run, one run per node; time = build +
solve), and `replication/cgip/run_cgip.py` re-runs them with this release. Every submitted cell has a row, solved or
not. CGIP-heur is the random-insertion heuristic with R = 10 restarts (the summaries' R = 1 rows are left out). Not
run on the cluster, so absent here: CGIP in Numerical1 at lambda = 0.5 and 1, and CGIP-heur in Numerical1. The .tex
tables written by `validation/make_lambda_tables.py` leave the CGIP columns blank.

The two application files come from `python validation/make_final_results_apps.py` (records in
`replication/empirical_dsr_n25/results` and `replication/empirical_ai_n18/results`); their READMEs from
`python validation/write_final_results_readmes.py`.

## Stopping criteria of the two applications

Both applications run the Hong&ndash;Li p-value with one per-solve tolerance, abs_tol with rel 0. It governs the dist(p&#770;)
certificate, and so the threshold band &tau; = scale &times; [LB, UB], and every draw that the threshold tests leave open.
Both draw with the multinomial_reject respondent bootstrap, seed 0, start from **abs 5e-4**, and run their GPU rows
through `rum.bootstrap`, which screens every draw with the p&#770; fit before solving and then refines within the run:
up to two tenfold tighter rounds (5e-5, 5e-6) for p&#770; and the draws still undecided.

| application | data | draws B | abs_tol | refinement |
|---|---|---|---|---|
| DSR | alt0 and full | 1,000 and 10,000 | **5e-4**, refined to 5e-5 and 5e-6 | some draws sit close to their threshold; the rounds decide every one of the 10,000 draws within one run (without them, three complete runs are needed, at 5e-4, 1e-4 and 1e-5) |
| AI | gpt-5.6-luna, deepseek-v4-flash, gpt-6 luna without reasoning, GMI Cloud | 10,000 | **5e-4** | none needed: every draw is far below its threshold, and the screen decides all 10,000 |

The cluster CGIP rows draw at abs 5e-4 and then settle dist(p&#770;) and every undecided draw at abs 1e-7. Relative to
the distance, the AI tolerance (3.7e-5 to 6.9e-5 of dist(p&#770;)) is about as tight as 1e-4 on DSR (6.7e-5 for alt0,
4.4e-5 for full), because the AI distances are 3 to 9 times larger.

## Columns

- `n`, `m2` = n 2^(n-1), `m_p` (observed cells), `density_pct` = 100 m_p / m2, `design` (pairwise or random_menu)
- `lambda`: p(lambda) = lambda p_inside + (1 - lambda) p_hat, with values 0, 0.5 and 1
- `instance`: the instance key. For lambda = 0 it is the file stem in `final_data/<experiment>/raw_data`; the suffixes `_lam5` and `_lam10` mean lambda = 0.5 and 1
- `opt`, `opt_lb`: the certified reference optimum and its lower bound. The true optimum lies in [opt_lb, opt]; opt = 0 at lambda = 1 is analytic
- `target_ub` = max(1.01 opt, 1e-3): the stopping rule is UB <= target_ub
- `table_entry`: the cell as printed. `CAP` = hit the wall cap. `--` = no measurement
- `status`: `ok`, `CAP`, `not run (device memory)`, `failed` or `out of memory (N GB)` (CPU runs killed at the node's memory limit). The last three print as `--`
- `fastest_gpu` (Numerical1 only): 1 = the bold (fastest) cell among the three GPU solvers
- `wall_s`: unrounded seconds to the criterion, or to the cap. `count` with `count_unit`: ADMM iterations or generated columns
- `ub_at_stop`, `stop` (the solver's stop reason), `cap_s` (the wall cap). CGIP: 24 h, or 7 d for pairwise n = 32, 35; `wall_s` of a capped CGIP run is the Slurm elapsed time
- `pricing_backend` (CGBFQP): `mask` or the two-layer `stream_u24` (n = 35)
- `note`: why a cell was not run, for example the Flow-ADMM memory requirement, or what happened to a CPU cluster run

### Application files (`results_dsr.csv`, `results_ai.csv`)

- `application` (DSR or AI), `data` (alt0 / full, or the LLM data set), `n` alternatives, `M` observed cells, `units` (respondents, or the LLM replicates: 100 or 2,000), `B` draws, `bootstrap` weights, `seed`
- `solver`, `variant`: for the GPU rows `restart from p_hat, refine 2` (`rum.bootstrap`: the screen, then each remaining draw restarts from the p_hat solution, with two tenfold tighter rounds for the draws still undecided); for CGIP the pricing (exact IP, or random insertion with R restarts and the IP at stalls)
- `abs_tol`, `rel_tol`: the per-solve tolerance for the certificate and for undecided draws
- `dist_lb`, `dist_ub`, `dist_gap`: the certified bracket on dist(p_hat); `tau_lo`, `tau_hi`: the threshold band, scale times that bracket
- `cert_s`, `cert_count`, `count_unit`: certificate time (for the GPU rows, every certificate of p_hat in the run: one, or one more per refinement round that needed it) and the first certificate's columns or ADMM iterations; `loop_s`: the B draws (building and screening them, and the solves); `total_s`: the whole test (certificate + draws, plus the one-off build where the solver has one)
- `draw_mean_s`, `draw_count_mean`: per draw. The GPU rows record no per-draw counts, so their `draw_count_mean` is empty
- `n_below`, `n_above`, `n_undecided`, `pval_lo`, `pval_hi` (undecided draws counted as 1), `reject_5pct`
- `status` (`complete`, or `capped (...)` for the AI CGIP runs that did not finish the certificate), `draws_done`
- `boot_redraws`: resamples redrawn by multinomial_reject because they emptied a menu (DSR: 10 at B = 1,000 and 72 at B = 10,000; AI: 0)
- `data_fingerprint`, `draws_fingerprint`: hashes of the data and of the bootstrap draws; runs with the same draws fingerprint saw identical draws. The GPU rows hash (menu, alternative, p_hat)
- `hardware`, `record` (the source file), `when`
- `note`: for the GPU rows, how many draws the screen decided with no solve and how many were solved; for the cluster DSR runs, how many draws were undecided at abs 5e-4 and settled at abs 1e-7 (dist_lb, dist_ub and the tau band are the 1e-7 ones); for the AI CGIP runs, where they stopped and the best UB
- `ci_lower`, `ci_lower_hi`, `ci_s`, `ci_draws` (GPU rows): the one-sided 95 % confidence interval [ci_lower, &infin;) for dist, ci_lower being the conservative end of a certified bracket whose upper end is ci_lower_hi; its time and the number of draws it solved to tolerance. Empty where the interval was not computed
- `dci_lower`, `dci_upper`, `dci_half_width` (the AI rows with `variant` ending in `menu-by-menu draws`): the two-sided 95 % interval [dci_lower, dci_upper] = [max(0, dist_lb &minus; c), dist_ub + c] for dist, c = `dci_half_width` the 95 % quantile of &Vert;p&#770;<sup>b</sup> &minus; p&#770;&Vert; over the draws (`Bootstrap.distance_interval`, no draw solved). Those rows draw each menu's replies on their own (`bootstrap` = multinomial, `boot_redraws` empty)
