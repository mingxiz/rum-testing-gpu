# Feasibility in observation density

n fixed at 20, density raised from 1 % to 100 % of m_2 = 10,485,760 cells. tab:dense reads these
records together with the n = 20 cells of the density grid (`validation/run_lambda_study.py --task 2 3`).

## What it is

Six cells at n = 20: 1 %, 5 %, 10 %, 25 %, 50 % and 100 % (every menu of size at least two,
m_p = 10,485,740). This axis isolates the observation cost from the lattice cost, and it is
where the column-generation side breaks down, because pricing costs O(m_p) per column.

Stored results: `results/` (the six n = 20 density cells, certificates and races).

## Run it

    python replication/feasibility_in_density/reproduce.py

## What the result should show

The BM-ADMM per-iteration cost is O(n 2^n) whatever m_p is, and its iteration count *falls* as
observations accumulate and constrain the solution more tightly. The race takes 1.9 s at 1 %
density (700 iterations) and 0.4 to 0.6 s from 10 % to 100 % (100 iterations), using at most
2 GB of device memory.

CGBFQP-GPU certifies the 1 % cell but stalls at the dense end, where BM-ADMM is the only
practical solver. That is why the dense reference optima in `data/opt_refs.json` are certified
with BM-ADMM rather than by column generation.
