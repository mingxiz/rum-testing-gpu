# Feasibility in the number of alternatives

How far the pairwise design can be pushed in n, and what it costs, from n = 20 to n = 35. Its
races at n = 25-35 are the lambda = 0 entries of tab:binary; `validation/run_lambda_study.py --task 1`
runs the whole table.

## What it is

The pairwise design (m_p = n(n-1), every binary menu observed) at n = 20, 25, 30, 32 with the
mask-indexed lattice, and at n = 33, 34, 35 with the two-layer pricing backends that make
those sizes fit a 32 GB card at all. The lattice grows from 10^6 to 3.4 x 10^10 nodes while
the number of observations stays under 1,200, so this axis isolates the lattice cost from the
observation cost.

Two measurements per cell: the certified optimum, (UB - LB)/LB <= 1e-3, which defines OPT and
is not timed in the paper, and the race to (UB - OPT)/OPT <= 1 %, which is.

Stored results: `results/` (pairwise cells n = 20 to 35, certificates and races, plus `FEASIBILITY.md`).

## Run it

    python replication/feasibility_in_n/reproduce.py

Races are idle-gated and strictly sequential; see `../README.md` for the protocol.
Certification runs are not timed and so are not gated.

## What the result should show

CGBFQP-GPU reaches the criterion in 0.05 s at n = 20, 0.29 s at 25, 14.6 s at 30, 67 s at 32,
146 s at 33, 325 s at 34 and 708 s at 35, using 63-103 columns throughout. The column count
barely grows with n; the time is the Bellman-Ford sweep, which doubles per unit of n.

BM-ADMM is not competitive on this axis and cannot run past n = 25 at all: its state is about
fourteen slabs of n 2^(n-1) numbers, which is 44 GiB at n = 25 in double precision and 840 GiB
at n = 30. That memory wall, far earlier than the one CGBFQP hits, is itself a result of this
table, and it is the reason the pairwise axis is a column-generation regime.
