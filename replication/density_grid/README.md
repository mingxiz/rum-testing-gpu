# tab:density18-20: the n = 18, 19, 20 density grid

Reproduces `tab:density18`, `tab:density19`, `tab:density20` of the section
"Algorithm Comparison in Synthetic Data".

## What it is

Five solvers on identical instances, identical stopping rule, identical machine: BM-ADMM,
Flow-ADMM and CGBFQP-GPU (this paper, GPU), plus CGIP and CGIP-heur (CPU column generation
pricing with a Gurobi integer program).

Thirty cells: n = 18, 19, 20 crossed with ten observation densities. The pairwise design
(every binary menu, both alternatives, m_p = n(n-1)) plus nine random-menu targets, the same
at every n: 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1 and 10 % of m_2 = n 2^(n-1).

Each solver is timed until it holds a **feasible** point with (UB - OPT)/OPT <= 1 %, checked
every 50 iterations for the ADMMs (through the feasibility push, since their iterates are
infeasible at finite iteration counts) and after every column for the column-generation
methods, whose iterates are feasible by construction.

## Run it

    python replication/density_grid/reproduce.py
    python replication/density_grid/reproduce.py --solvers bm_admm,cgbfqp,flow_admm,cgip,cgip_heur
    python replication/density_grid/reproduce.py --n 18 --cells n18_sparse,n18_m39

Writes `validation/results/<tag>/*.json`, the LaTeX tables `paper/tables/tab_n*.tex`, a
`summary.md`, and a side-by-side diff against the reference results committed from the paper
machine (`results/` here, which also holds the lambda = 0.5 and 1 races of
`validation/run_lambda_study.py --task 3`).

The two CPU baselines need a Gurobi licence. Without one the GPU columns still reproduce and
the CGIP columns stay blank.

## What the result should show

Three regimes. CGBFQP-GPU dominates from the pairwise design through 0.3 % density, taking
0.04-0.05 s on pairwise against 3-42 s for BM-ADMM. The crossover sits at 0.5 % density at
every n, and beyond it the BM-ADMM lead grows with both density and n, reaching 90x, 68x and
238x at 10 % density for n = 18, 19, 20. Flow-ADMM needs the fewest iterations (100-400) but
its memory grows as m_p squared, so it is confined to the sparse cells and is reported as
"--" where the factorization cannot fit.
