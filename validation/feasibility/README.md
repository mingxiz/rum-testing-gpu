# Feasibility study: number of alternatives, and observation density

Two experiments that turn the speed-ups of the density tables into absolute
time ("an hour versus days"):

1. **Pairwise design, growing n** — n = 20, 25, 30, 32 with the mask-indexed lattice, and
   n = 33, 34, 35 with the rank-indexed two-layer pricing backend (`pricing=stream_f32` /
   `stream_u16`, `solver/flow_cg/pricing_stream.py`; n = 35 needs the uint16 variant on
   32 GB). m_p = n(n−1). GPU solver: CGBFQP-GPU. CPU baselines (CGIP, CGIP-heur) on the same
   instances, run elsewhere. `run_pricing.py` compares the backends on n = 20–32 (same instances,
   same OPT) and runs n = 33–35; `compare_pricing.py` makes the table.
2. **n = 20, growing density** — 1 %, 5 %, 10 %, 25 %, 50 %, 100 % of the
   m₂ = 10,485,760 (menu, alternative) cells. GPU solver: BM-ADMM. Same CPU
   baselines.

## Rules

- **Reproducible instances.** `data/gen_instances.py --feas` regenerates every
  cell from the same generators and seeds as the density grid (`menu_seed=42`,
  `phat_seed=100`); the cells shared with the grid (`n20_sparse`, `n20_d1pct`,
  `n20_m2759`) are byte-identical to it and checked against the stored
  fingerprints. 100 % density is `gen_dense(20)` (every menu of size ≥ 2). CSVs
  for the CPU runs: `data/instances/<key>.csv.gz` (`menu,alt,phat`).
- **OPT reference**: certified (UB − LB)/LB ≤ 1e-3 — by CGBFQP-GPU for the
  pairwise cells, by BM-ADMM (pushed UB + Frank–Wolfe/Bellman–Ford LB) for the
  n = 20 density cells — stored in `data/opt_refs.json` with the instance
  fingerprint. Cells that already had a reference (n20_sparse from CGBFQP;
  n20_d1pct from CGBFQP; n20_m2759 and n20_dense from BM-ADMM) are re-certified
  and the brackets compared (`ref_check` in the cert JSON); the stored value is
  kept.
- **Race**: time to a feasible point with (UB − OPT)/OPT ≤ 1 %, checked after
  every column (CGBFQP-GPU) or every 50 iterations (BM-ADMM, pushed iterate).
- **One measurement per process, machine otherwise idle**: `run_feas.py`
  launches each run in a fresh process and blocks until the GPU shows ≤ 10 %
  utilisation and ≤ 4.5 GB in use (desktop baseline) for two consecutive
  samples. Cores busy, peak host RSS and device-pool high-water are recorded.
- **Trajectory**: every run keeps `trace` — per column `[column, wall_s, UB,
  LB, n_cols]` for CGBFQP-GPU; per 50-iteration check `[iteration, wall_s,
  phi(iterate), pushed UB or null]` for BM-ADMM races and `[iteration, wall_s,
  UB, LB]` for BM-ADMM certifications.

## Commands

```
python validation/feasibility/run_feas.py                     # data -> cert -> race -> tables, both parts
python validation/feasibility/run_feas.py --part pairwise --n-max 32
python validation/feasibility/run_feas.py --stage race --skip-existing
python validation/feasibility/feas_one.py --key n30_sparse --solver cgbfqp --criterion race --out validation/results/feasibility
python validation/feasibility/make_feas_tables.py             # FEASIBILITY.md + paper/tables/tab_feas_*.tex
python validation/feasibility/run_pricing.py --part compare    # pricing backends on n = 20-32 (mask / stream_f32 / stream_u16)
python validation/feasibility/run_pricing.py --part extend     # n = 33, 34 (f32 + u16) and n = 35 (u16): cert + race
python validation/feasibility/run_pricing.py --part certcmp    # optional: certify n = 20-32 with the stream backends too (not run for the paper)
python validation/feasibility/run_path.py --n 25 30 32 --check-base --skip-existing   # convex-path OPT references (data/README.md (d)), table PATH_REFS.md
python validation/feasibility/run_path20.py --check-base --skip-existing              # the same path on the n = 20 density axis (six densities), table PATH20_REFS.md
python validation/feasibility/feas_one.py --key n33_sparse --solver cgbfqp --criterion race --pricing stream_u16 --out validation/results/feasibility
```

Results: `validation/results/feasibility/<criterion>_<solver>_<key>.json`, with
`<solver>` = `cgbfqp-stream_f32` / `cgbfqp-stream_u16` for the two-layer backends
(committed copies under `replication/feasibility_in_n/results/`, `replication/feasibility_in_density/results/`, `replication/pricing_backends/results/` and `replication/convex_path/results/`). Backend comparison and
the n = 33–35 numbers: `PRICING_COMPARISON.md` there.

## For the CPU baselines

IP-CG and IP-H-CG run on the same instances (`data/instances/<key>.csv.gz`) with the same OPT
(`data/opt_refs.json`) and the race criterion `ub_stop = 1.01·OPT`:
`solver.solve(..., method="ip_cg", criterion="race", opt_ref=OPT)` (or `"ip_cg_heur"`), or through the
RUM class with `replication/cgip/run_cgip.py`.
Drop the result JSONs as `race_cgip_<key>.json` / `race_cgip_heur_<key>.json`
into `validation/results/feasibility/` and rerun `make_feas_tables.py`. Note
that on the pairwise design CGIP's pricing IP is a linear-ordering problem in
n(n−1)/2 binaries, whose cost does not grow as 2ⁿ, whereas the density
experiment gives it an integer program with one auxiliary binary per observed
cell (up to 10.5 M at 100 %).
