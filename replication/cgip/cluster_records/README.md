# The paper's IP-CG and IP-H-CG runs

The IP-CG (`CGIP` in the CSVs) and IP-H-CG (`CGIP-heur`, 10 random-insertion restarts) numbers in the paper come
from runs on a CPU cluster: one run per node, 24 Gurobi threads, scheduled by Slurm. They were made with an
earlier version of the CGIP code, not the one in `solver/ip_cg/`. These are their result files, as the runs
wrote them; `validation/make_final_results.py` and `validation/make_final_results_apps.py` read them into
`final_results/`. To run the same experiments with this release's solver, use [`../run_cgip.py`](../run_cgip.py)
([`../README.md`](../README.md) lists the commands and how the two versions differ).

| folder | paper | files |
|---|---|---|
| `Numerical1/` | tab:density18-20, the IP-CG entries | `summary.csv`: the 30 cells at lambda = 0, of which the three pairwise cells were solved (`n18_sparse.json`, `n19_sparse.json`, `n20_sparse.json`) |
| `Numerical2/` | tab:binary | `summary.csv`; one JSON per solved cell: `cgip_<cell>.json` (IP-CG), `cgip_heur_rand_R10_<cell>.json` (IP-H-CG) |
| `Numerical3/` | tab:dense | the same |
| `Application_DSR/` | tab:dsr | `summary.csv` and the four reported runs (alt0 and full, IP-CG and IP-H-CG): `<run>.json`, the dist(p&#770;) certificate before and after refinement (`<run>_dist.json`, `<run>_dist_refined.json`), every draw (`<run>_draws.csv`) and the draws settled at abs 1e-7 (`<run>_refined.csv`) |
| `Application_AI/` | tab:ai | `summary.csv`: the runs on the 100-reply data (`final_data/Application_AI/raw_data/gpt5.6-luna.npz`, `deepseekv4-flash.npz`), all stopped by the 24 h limit during the dist(p&#770;) certificate |

- `summary.csv` has one row per submitted run, solved or not, with the Slurm job id, elapsed time, peak memory and
  the node it ran on. Rows with method `cgip_heur_rand` (or `cgip_rand`) and `n_restart` 1 are IP-H-CG with a
  single restart per pricing call; `final_results/` leaves them out.
- Times are build + solve in seconds (`total_s`). The wall-time limit was 24 h, and 7 days for tab:binary at
  n = 32 and 35; a run stopped by it has status `timeout`.
- The DSR runs draw B = 10,000 at abs 5e-4, then settle dist(p&#770;) and every undecided draw at abs 1e-7. They
  decide every draw as this release's Flow-CG runs do
  (`replication/empirical_dsr_n25/results/bootstrap_dsr_{alt0,full}.json`).
- The DSR rows marked `report` = no are chained 24 h jobs on slower nodes that reproduce the reported runs; their
  per-run files are not included.
