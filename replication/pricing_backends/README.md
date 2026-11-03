# The lattice layouts, compared on identical instances

The evidence for how n = 33 to 35 fit a 32 GB card. `solver/flow_cg/MEMORY.md` explains the
layouts and what each one holds in memory.

## What it is

The same pairwise cells solved with different storage layouts for the Bellman-Ford sweep:

| layout | value storage | device memory at n = 35 |
|---|---|---|
| `mask` | one slot per subset, float32, 2^n slots | 288 GiB (does not fit) |
| `stream_f32` | two live layers, rank-indexed, float32 | 34.3 GiB (does not fit) |
| `stream_u24` | the same, 24-bit block-scaled | 25.8 GiB |
| `stream_u16` | the same, 16-bit block-scaled | 17.4 GiB |

Stored results: `results/` (the same cells under each backend, the uint16 / uint24 diagnostics, `PRICING_COMPARISON.md`).

## Run it

    python replication/pricing_backends/reproduce.py

It runs the correctness test first, then the comparison. The test is the load-bearing part:
`stream_f32` must be bit-identical to the mask kernel in values, tie-breaking and the returned
ranking for n = 6 to 20; the quantized backends must stay inside the margin they report; and
all of them must agree with brute force at n = 6 and 8.

## What the result should show

On the cells all layouts can hold they generate the same columns and the same upper bound,
with the two-layer backends within a few percent of the mask kernel time: 67.0, 67.6 and
69.4 s at n = 32.

Quantized storage buys memory, not speed. uint16 is 3 to 6 % *slower* per column than float32,
never faster, because the sweep is bound by unranking and scattered gathers rather than by the
bytes of the value array. It also costs certificate tightness: the quantization margin floors
the bracket at roughly margin divided by UB, so uint16 cannot close a 1e-3 certificate once
the distance is small. Measured at n = 32 on the convex-path cell with optimum 0.0727, uint16
stalls at a gap of 3.6e-3 while uint24 certifies at 9.7e-4. That is why n = 35 uses uint24.
