# Replication

Every table in the paper's numerical and application sections is regenerated from this folder. Each
subfolder has a `README.md` (what it holds and what the result should show) and the script that runs it;
the subfolders are entry points into `validation/`, not copies of it. Their `results/` folders hold the
stored records from the paper machine, every JSON with its full trajectory. A fresh run writes to
`validation/results/...`; the table builders read that first and fall back to the stores, so a clean clone
regenerates every table from the committed records (`validation/result_store.py`).

Everything runs through this repository's solver (`solver/`): the timed races through `solver.solve`, the
certificates and the bootstrap through the RUM class (`solver/rum_solver.py`).
[`docs/replication.html`](../docs/replication.html) shows the same map as a page.

## The paper's tables

| paper | command | stored records | `final_results/` |
|---|---|---|---|
| tab:density18, tab:density19, tab:density20 | `python replication/density_grid/reproduce.py` (lambda = 0); `python validation/run_lambda_study.py --task 3` (lambda = 0.5, 1) | `density_grid/results/` | `Numerical1/results_n{18,19,20}.csv` |
| tab:binary (pairwise design, n = 25-35) | `python validation/run_lambda_study.py --task 1` | `feasibility_in_n/results/`, `convex_path/results/` | `Numerical2/results_pairwise.csv` |
| tab:dense (n = 20, density 0.1-5 %) | `python validation/run_lambda_study.py --task 2 3` | `feasibility_in_density/results/`, `density_grid/results/` | `Numerical3/results_density.csv` |
| tab:dsr (DSR data, n = 25) | `python replication/empirical_dsr_n25/run_bootstrap.py --obs alt0` (and `--obs full`) | `empirical_dsr_n25/results/` | `Application_DSR/results_dsr.csv` |
| tab:ai and the two AI intervals (LLM data, n = 18) | `python replication/empirical_ai_n18/run_bootstrap.py --model gpt6luna_noreason` (and `gmicloud`; see its README) | `empirical_ai_n18/results/` | `Application_AI/results_ai.csv` |
| the IP-CG and IP-H-CG entries of all of the above | `python replication/cgip/run_cgip.py ...` (see its README) | `cgip/cluster_records/`: the paper's cluster runs | the CGIP and CGIP-heur rows |

After the runs: `python validation/make_lambda_tables.py` (the LaTeX tables, in `paper/tables/`),
`python validation/make_final_results.py`, `python validation/make_final_results_apps.py` and
`python validation/write_final_results_readmes.py` (`final_results/`).

Names: the paper's Flow-CG is method `flow_cg` (`CGBFQP-GPU` in the CSVs and the records), IP-CG is
`ip_cg` (`CGIP`), IP-H-CG is `ip_cg_heur` with 10 random-insertion restarts (`CGIP-heur`); BM-ADMM is
`bm_admm` and Flow-ADMM `flow_admm`.

## The other folders

| folder | what it holds |
|---|---|
| [`feasibility_in_n/`](feasibility_in_n/) | the pairwise design from n = 20 to 35: certified optima and races, n = 33-35 with the two-layer pricing |
| [`feasibility_in_density/`](feasibility_in_density/) | n = 20 from 1 % to 100 % density: certified optima and BM-ADMM races |
| [`convex_path/`](convex_path/) | the reference optima along p(lambda) = lambda p_inside + (1 - lambda) p_hat, which the lambda = 0.5 and 1 cells use |
| [`pricing_backends/`](pricing_backends/) | the lattice layouts of the pricing sweep (mask, two-layer float32 / uint16 / uint24) on identical cells |
| [`qp_matrix_memory/`](qp_matrix_memory/) | the measured size of the explicit QP matrices: why the projection is not handed to a QP solver at n = 18-20 |

## Instances and reference optima

**Instances come from seeds.** `data/gen_instances.py` rebuilds every synthetic instance and checks a sha256
fingerprint against the one stored beside its certified optimum in `data/opt_refs.json`; a run refuses to start
if a fingerprint does not match, so a result can never be attributed to the wrong instance. The same instances
are also in `final_data/Numerical1-3/raw_data/` as CSV files (rows: menu bitmask, chosen alternative, p_hat),
with their optima in `final_data/<set>/opt_ref.json`. The two empirical data sets are observed data, in
`final_data/Application_DSR/` and `final_data/Application_AI/`.

**Reference optima are certified, not assumed.** `OPT` is the objective of a *feasible* point whose certified
bracket satisfies (UB - LB)/LB <= 1e-3, so the true optimum lies in [LB, OPT] and OPT overstates it by at most
0.1 %. Cells whose distance is about 0 use an absolute criterion instead; the record's `criterion` field says
which was used.

## Measurement protocol for anything with a time in it

- one fresh process per (cell, solver), so nothing is carried between runs;
- the GPU must be idle before a timed run starts (<= 10 % utilisation, <= 4.5 GB in use, two consecutive
  samples) - `validation/run_paper_table.py::wait_for_idle_gpu`;
- runs are strictly sequential, never two at once;
- no thread caps: every solver gets the whole machine, and the cores it actually used are measured and recorded
  per run (`avg_cores`; the GPU solvers report about 1.0, one CPU thread plus the whole device).

Certification runs do not report times, so they are not idle-gated.

## Hardware of the stored results

GPU columns: AMD Ryzen 9 9950X (16 cores / 32 threads), 64 GB DDR5, one NVIDIA GeForce RTX 5090 (32 GB),
Windows 11 (WDDM), CUDA 13.3, Python 3.12, CuPy 14. IP-CG and IP-H-CG: one CPU cluster node per run with 24
Gurobi threads (`cgip/cluster_records/`). Keep the stored records to diff a new run against them.
