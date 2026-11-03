# IP-CG and IP-H-CG through the RUM class

`run_cgip.py` re-runs every IP-CG and IP-H-CG number in the paper with this release's solver: the RUM class
(`solver/rum_solver.py`), methods `ip_cg` (IP-CG, exact IP pricing) and `ip_cg_heur` (IP-H-CG, random-insertion
pricing with 10 restarts per call, the IP when it stalls). The printed numbers come from cluster runs made with an
earlier version of the CGIP code; their result files are in [`cluster_records/`](cluster_records/).

`run_cgip.py` has not been run for the paper; `results/` appears with the first run.

| paper | runs | command (`python replication/cgip/run_cgip.py ... --threads 24`) |
|---|---|---|
| tab:density18-20, the IP-CG row | 30 cells | `race --set Numerical1 --key <cell> --method ip_cg` |
| tab:binary | 15 cells x 2 methods | `race --set Numerical2 --key <cell> --method ip_cg` (or `ip_cg_heur`) |
| tab:dense | 12 cells x 2 methods | `race --set Numerical3 --key <cell> --method ip_cg` (or `ip_cg_heur`) |
| tab:dsr | alt0, full x 2 methods | `dsr --obs alt0 --method ip_cg` (or `--obs full`, `ip_cg_heur`) |
| tab:ai | 2 models x 2 methods | `ai --model gpt6luna_noreason --method ip_cg` (GPT-6 Luna; `gmicloud` = DeepSeek V4 Flash) |

The cells are the keys of `final_data/<set>/opt_ref.json`. The cluster's numbers to compare with are the CGIP and
CGIP-heur rows of `final_results/` (from `cluster_records/*/summary.csv`). tab:dsr prints 84,747 s and
28,449 s (alt0), 348,381 s and 204,779 s (full); tab:ai prints "≥ 24 h" in all four cells.

## How a run stops

- **race**: once UB < OPT + max(1e-2 OPT, 1e-3), the paper's termination criterion, or at the cap: 7 days for
  Numerical2 with n ≥ 32, 24 h otherwise (`--cap-hours`), the cluster's limits. The record's `table` field is the
  time in seconds or "≥ 24 h". The time is the class's: the Gurobi pricing model's build plus the solve, as the
  cluster's build_s + solve_s.
- **dsr, ai**: `rum.bootstrap` with the class's defaults (B = 10,000, seed 0, abs 5e-4, rel 0, two tenfold
  tighter rounds for the draws still undecided). p̂ is certified first; if that does not close within
  `--cert-hours` (24) the run stops there and reports "≥ 24 h". Each draw may take `--draw-hours` (24). The time
  is the record's `time`: certificate, screen and solve.

## Where it differs from the cluster runs

- Master QP: the class's exact active-set QP (`qp_backend` native); the cluster's scripts swap in Gurobi's QP
  with a 1e-8 ridge.
- Draws left undecided at 5e-4: the class's rounds at 5e-5 and 5e-6, where the pricing IP is re-solved with
  MIPGap 0 when its own gap blocks a stop; the cluster re-solves them once at 1e-7 with MIPGap 0.
- Column cap: 100,000; the cluster's bootstrap runs used 20,000.
- tab:ai data: the cluster's AI runs used the 100-reply data (`--model luna`, `deepseek`); tab:ai's other entries
  use the 2,000-reply data (`gpt6luna_noreason`, `gmicloud`).
- The draws are the same: the class's Flow-CG run (`../empirical_dsr_n25/run_bootstrap.py`) decides all 10,000 DSR
  draws as the cluster's IP-CG and IP-H-CG runs do (`cluster_records/Application_DSR/*_draws.csv`).

## Running it

- Python 3.12 with numpy, scipy, pandas, numba and gurobipy, and a Gurobi licence the node can reach. No GPU.
- One run per node and nothing else on it: the times are the result. The paper's runs had 24 cores and 24
  Gurobi threads.
- A bootstrap run stopped by a wall limit keeps `results/<run>.json.checkpoint` (written every 60 s); the same
  command continues it, certifying p̂ again first. A race writes its record only at the end, so give the job a
  wall limit above the cap, or pass e.g. `--cap-hours 23.9` on a 24 h partition.
