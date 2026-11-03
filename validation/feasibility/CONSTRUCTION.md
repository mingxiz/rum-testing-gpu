# How the feasibility study was constructed

A record of the decisions and the code behind `validation/feasibility/`, in the order they
were taken, so that the study can be audited and extended (e.g. to n = 35 or to other GPUs).

## 1. Aim

Turn the relative speed-ups of the density tables into absolute wall time on two axes:
the number of alternatives n on the pairwise design (CGBFQP-GPU vs the CPU baselines), and the
observation density at n = 20 (BM-ADMM vs the CPU baselines). GPU solvers run on the paper
machine (RTX 5090); CGIP / CGIP-heur run on a CPU cluster from the same instances and the same
OPT references.

## 2. Two rules and how the code enforces them

1. **Exact replication elsewhere.** Instances are regenerated from seeds by `data/gen_instances.py`
   (`--feas`), never edited by hand; each cell carries a sha256 fingerprint stored next to its
   OPT in `data/opt_refs.json`; `feas_one.py` refuses to run a cell whose regenerated fingerprint
   differs from the stored one. Cells shared with the density grid (`n20_sparse`, `n20_d1pct`,
   `n20_m2759`, `n20_dense`) are byte-identical to it — checked, not assumed.
2. **One measurement, an otherwise idle machine.** `run_feas.py` runs every (cell, solver,
   criterion) in its own fresh process and, before each, blocks until the GPU reports ≤ 10 %
   utilisation and ≤ 4.5 GB in use for two consecutive samples
   (`validation/run_paper_table.py: wait_for_idle_gpu`). Each result
   records cores busy (CPU time / wall), peak host RSS and the device-pool high-water mark.

## 3. Feasibility of CGBFQP-GPU in n (Task 1)

- Measured first (`bf_sweep_scaling.py`): one Bellman–Ford sweep on the pairwise design takes
  0.4 ms (n = 20), 39 ms (n = 28), 170 ms (n = 30): it scales as 2ⁿ·n. Time is not the limit.
- Memory is: the original layout keeps 9 bytes per lattice node (V float32, pred uint8, and
  every layer's masks as int32), 36 GB at n = 32.
- **Memory trick implemented** (`solver/flow_cg/cg_core.py`, `layers="stream"`): layer masks
  are no longer stored; each layer is generated on the GPU right before its kernel launch by a
  colex-unranking kernel into one reusable buffer of C(n, n/2) masks. 5 bytes per node + the
  buffer: 22 GB at n = 32 — fits. The unranking order equals Gosper's order, so results are
  bit-identical to the stored layout (`cg_core.selftest()`; verified end-to-end: same columns,
  same UB to 1e-12 on two cells). Automatic for n ≥ 29.
- n = 33 and beyond need a different scheme. **Implemented second** (`pricing_stream.py`):
  subsets addressed by colex rank within their layer, only layers k−1 and k resident, predecessors
  stored for the top 10 layers and the rest of the path recovered by an exact DP on the
  2^(n−10) sub-lattice of the node reached there, and an optional uint16 block-scaled storage
  (per-1,024-rank offset/step) whose bounded quantization error is added to the Frank–Wolfe
  gap so the certificate stays valid. `stream_f32` is bit-identical to the mask kernel
  (`test_stream_pricer.py`); `stream_u16` halves the memory again. Device memory: n = 33
  8.7 / 4.4 GiB, n = 34 17.4 / 8.7 GiB, n = 35 33.8 / 16.9 GiB (float32 / uint16), so n = 35
  runs only with uint16 on 32 GB. The study covers n = 20, 25, 30, 32 with the mask kernel
  (and, for comparison, with both stream backends) and n = 33, 34, 35 with the stream backends.
- **Outcome** (`run_pricing.py`, September 2026; `PRICING_COMPARISON.md`): on n = 20–32 all three
  backends generate the same columns and the same UB, with the stream backends within a few
  percent of the mask kernel's time (67.0 / 67.6 / 69.4 s at n = 32). n = 33: 146 s (f32) and
  155 s (u16), 90 columns; n = 34: 325 s and 334 s, 103 columns; n = 35: 708 s (u16 only,
  17.4 GiB), 103 columns. The n = 33–35 OPTs were certified by the stream backends (f32 at 33,
  34 with u16 re-certification agreeing to 1e-5; u16 at 35) and added to `data/opt_refs.json`;
  the pairwise table marks these rows. uint16 is 3–6 % slower per column than float32 and
  needs 3–4 % more columns to certify; it is the memory option, not the speed option.
  The n = 20–32 certifications with the stream backends were not run (race comparison only,
  by decision); `run_pricing.py --part certcmp` would add them.

## 4. Instances (Tasks 2 / 1')

`data/gen_instances.py`: `FEAS_PAIRWISE = {25, 30, 32: sparse}` (pairwise = every pair, both
alternatives, p̂ from `RandomState(100)`), `FEAS_DENSITY20 = [m1380, m6899, m13797, dense]` for
5 %, 25 %, 50 % (random-menu family, `mult = round(d·m2/(n(n−1)))`, `RandomState(42)` for menus,
`RandomState(100)` for p̂) and 100 % (`gen_dense`). Shared cells come from the grid. Output:
`data/instances/<key>.csv.gz`, `data/manifest_feas.json`.

## 5. OPT references (Tasks 3 / 2')

`feas_one.py --criterion cert`: CGBFQP-GPU with its Frank–Wolfe bound for the pairwise cells,
BM-ADMM with the explicit certificate (feasibility-pushed UB + Frank–Wolfe/Bellman–Ford LB every
200 iterations) for the density cells; stop at (UB − LB)/LB ≤ 1e-3; the result is written to
`data/opt_refs.json` as `opt` (= UB), `lb`, `gap`, `method`, `mp`, `cd_hash` when the cell has
no reference. Cells that already have one are re-certified and the brackets compared
(`ref_check`): the stored value is kept, the agreement is recorded.

## 6. Timed races with trajectories (Tasks 4 / 3')

`feas_one.py --criterion race`: (UB − OPT)/OPT ≤ 1 % against the stored OPT; the solvers were
instrumented to return `trace` — per column `[column, wall_s, UB, LB, n_cols]` (CGBFQP-GPU,
`solver/flow_cg/solver.py`) and per 50-iteration check `[iteration, wall_s, phi, pushed UB]`
(BM-ADMM, `solver/bm_admm/solver.py`, merged across the single- and double-precision legs of
the hybrid). The final wall time and the trace are in the race JSON.

## 7. Tables

`make_feas_tables.py` → `validation/results/feasibility/FEASIBILITY.md` and
`paper/tables/tab_feas_pairwise.tex`, `tab_feas_density.tex`, with a "human units" column
(seconds / minutes / hours / days) for the message the study is meant to deliver. CPU-baseline
results are merged from `race_cgip_<key>.json` / `race_cgip_heur_<key>.json` when they arrive.

## 8. What to expect on the two axes

On the density axis at n = 20, BM-ADMM's per-iteration cost is O(n2ⁿ) whatever m_p is, while
CGIP's pricing IP grows with the number of observed cells (up to 10.5 M binaries at 100 %).
On the pairwise n-axis, CGBFQP-GPU's sweep doubles per unit of n; CGIP's pricing IP is a
linear-ordering problem in n(n−1)/2 binaries whose cost grows with n through branch-and-bound.
The runs decide which grows faster; both tables are built to show it.
