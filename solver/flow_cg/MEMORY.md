# CGBFQP-GPU: memory versus n, and the two pricing backends

The Bellman–Ford pricing sweep walks the subset lattice {0..n−1} layer by layer
(layer k = all k-subsets). What has to be resident per node decides how far a
32 GB card goes. Two backends are implemented (`pricing=` in `solve_cgbfqp`):

| | `mask` (original, default) | `stream_f32` / `stream_u16` (`pricing_stream.py`) |
|---|---|---|
| addressing | one slot per subset, indexed by the mask (2ⁿ slots) | colex rank within the layer; only layers k−1 and k resident |
| value V | float32, 4 B × 2ⁿ | float32 (4 B) or uint16 block-scaled (2 B) × 2·C(n,⌊n/2⌋) |
| predecessor | uint8, 1 B × 2ⁿ | uint8 for the top 10 layers only; the rest of the path from an exact DP on the 2^(n−10) sub-lattice of the node reached there |
| layer masks | generated on the fly into one buffer of C(n,⌊n/2⌋) masks (`layers="stream"`, default for n ≥ 29) | unranked inside the kernel, no buffer |
| results | reference | `stream_f32` bit-identical to `mask` (values, ties, columns); `stream_u16` within a certified margin |

| n | `mask` | `stream_f32` | `stream_u16` | fits a 32 GB card (≈ 29 GiB usable)? |
|---|---|---|---|---|
| 30 | 5.6 GiB | 1.2 GiB | 0.7 GiB | all |
| 32 | 22.2 GiB | 4.5 GiB | 2.3 GiB | all |
| 33 | 44 GiB | 8.7 GiB | 4.4 GiB | stream only |
| 34 | 89 GiB | 17.4 GiB | 8.7 GiB | stream only |
| 35 | 177 GiB | 33.8 GiB | 16.9 GiB | `stream_u16` only |

(`stream_*` figures include the top-layer predecessors and the sub-lattice scaffolding.)

## How the stream backend works

- **Unranking.** Thread r of layer k rebuilds its subset from its colex rank with the
  combinatorial number system (largest b with C(b,k) ≤ r is the top element, recurse), O(n)
  table lookups; the order equals ascending mask order, i.e. Gosper's order.
- **Predecessor ranks in O(k).** For sorted elements b₀ < … < b_{k−1}, rank(D) = Σ C(b_t, t+1);
  removing b_j gives Σ_{t<j} C(b_t, t+1) + Σ_{t>j} C(b_t, t), computed by one backward pass with a
  running prefix and suffix. The predecessors are visited from the highest element down with a
  ≥ update, which reproduces the mask kernel's "lowest alternative wins ties".
- **Weights** w(x, D) are the same scan of alternative x's menu list, in the same order, so the
  float32 sums are identical to the mask kernel's.
- **uint16 block scaling.** A thread block owns 1,024 consecutive ranks: it computes them in
  float32 registers, reduces min and max in shared memory, stores `q = round((V − min)/step)` with
  `step = (max − min)/65535` and the two scalars per block (8.8 MB for the widest layer at
  n = 35). Reads decode `min + q·step`. Addition and maximum are 1-Lipschitz, so each layer adds
  at most step/2 of error: the sweep returns `margin = Σ_k max_block step_k / 2`, and the solver
  uses `Vmax + margin` in the Frank–Wolfe gap. The lower-bound certificate therefore stays valid
  (slightly looser); the returned column's reduced cost is recomputed exactly in float64 by the
  master, so the race criterion is unaffected. Measured margins: 7e-3 at the first column of
  n = 20, 1e-3 late; the certification gap at n = 32–35 is about 0.013.
- **uint24 (`stream_u24`).** The same block scaling with a 24-bit index stored as a uint16 high
  part plus a uint8 low part (3 B per rank): `step = (max − min)/(2^24 − 1)`, so the margin is 128×
  below uint16 (n = 20 pairwise: 5.4e-5 against 6.9e-3 at the first column) at 25.4 GiB for two
  layers at n = 35, which still fits. The index is formed in float32, exact only up to 2^24, so the
  per-layer bound is taken as one full step rather than step/2. Why it exists: the margin m floors
  the certificate at UB − LB ≈ m/UB, so uint16 cannot close (UB − LB)/LB ≤ 1e-3 once UB² < 1000·m;
  on the n = 32 convex path at λ = 0.6 (distance 0.073) the uint16 bracket plateaued at 3.6e-3
  while the exact reference closed at 1e-3. Use uint16 for races and for certificates of
  distances above a few units; use uint24 for certificates on small distances.
- **Backtrace.** Descend through the stored predecessors from the full set to layer n−10 (host
  computes the colex rank of each node; one byte read per layer), then relabel the surviving
  node's n−10 elements to 0..n−11, restrict the menus to that set and run the mask-indexed DP on
  2^(n−10) nodes (0.1 % of a sweep); the path is identical because V[S] depends only on subsets of S.

## Measured sweep time (pairwise design, RTX 5090)

Mask kernel, one sweep: 0.4 ms (n = 20), 39 ms (28), 170 ms (30), ≈ 0.7 s per column at n = 32;
it scales as 2ⁿ·n (×2.1 per unit of n). The stream backends pay for unranking and rank
arithmetic, and `stream_u16` additionally for the per-block min/max reduction and the decode on
every gather; neither is bandwidth-bound on the value arrays, so halving the element size buys
memory, not time. Measured per column in the races (same instances, same columns, idle GPU):

| n | `mask` | `stream_f32` | `stream_u16` | device memory mask / f32 / u16 |
|---|---|---|---|---|
| 32 | 0.70 s | 0.70 s | 0.72 s | 22.2 / 4.6 / 2.4 GiB |
| 33 | — | 1.62 s | 1.72 s | — / 8.9 / 4.5 GiB |
| 34 | — | 3.16 s | 3.24 s | — / 17.7 / 9.0 GiB |
| 35 | — | — | 6.88 s | — / — / 17.4 GiB |

`stream_u16` is 3–6 % slower per column than `stream_f32`. Its certificates need 3–4 % more
columns (355 vs 345 at n = 33, 400 vs 387 at n = 34) because the margin, about 1e-3 to 3e-3, is a
sizeable fraction of the absolute certification gap (≈ 4e-3 at these n) and the run must close
the true gap by that much more; the races, which do not use the lower bound, generated identical
columns at every n. A smaller `BLOCK` would shrink the margin at some cost per sweep. Full
tables: `validation/results/feasibility/PRICING_COMPARISON.md` (generated by
`validation/feasibility/compare_pricing.py`). Correctness tests: `validation/feasibility/test_stream_pricer.py`
(bit-identity of `stream_f32` vs `mask` for n = 6–20 on pairwise and general menus, brute force at
n = 6 and 8, `stream_u16` within its margin).

The backtrace recompute is not a cost worth removing: the sub-lattice DP has 2^(n−10) nodes,
0.1 % of the sweep, a few milliseconds against seconds per column; storing every predecessor
instead would add 2ⁿ bytes (8 GiB at n = 33, 16 GiB at n = 34) and a byte write per node.

## Beyond n = 35 on one card

Two uint16 layers reach 33.8 GiB at n = 36 (C(36,18) = 9.1 × 10⁹): the same 5 GiB overshoot as
float32 at n = 35, so n = 36 needs either host staging of the three middle transitions or 8-bit
storage, and n = 37 (67 GiB) needs a larger device. On a 141 GB H200 the uint16 backend reaches
n = 38 (C(38,19) = 3.5 × 10¹⁰, 131 GiB) and float32 reaches n = 37.
