# Can the projection be handed to a QP solver as explicit matrices?

The question behind the CPU baselines: at the paper's sizes, n = 18, 19, 20, can the RUM
projection min ||p_D - p-hat||^2 be given to a commercial QP solver (Gurobi, CPLEX, MOSEK) as an
explicit sparse quadratic program? A solver needs the constraint matrices in memory in sparse
form, so the first test is simply their size. Measured here with scipy on the machine the
paper's GPU runs use, which has **64 GB** of host memory.

    python validation/qp_matrix_memory.py        # rebuilds MATRIX_MEMORY.md

## The two exact formulations

Both are the ones shipped with the CPU baseline (`solver/exact_qp/flow_bm_qp.py`),
with m_2 = n 2^(n-1) the number of (menu, alternative) cells.

| form | variables | constraints | nonzeros |
|---|---|---|---|
| Block-Marschak (BM) | p~ in R^{m_2}, a probability per cell | K p~ >= 0 (the Mobius transform is nonnegative), S p~ = 1 (one simplex per menu), p~ >= 0 | nnz(K) = n 3^(n-1), nnz(S) = m_2 |
| arc-flow (AF) | r in R^{m_2}, a flow on the arcs of the subset lattice | A r = b (flow conservation at the 2^n nodes), r >= 0 | nnz(A) = n 2^n |

## Measured storage, one CSR copy (float64 values, int32 indices)

Built and measured up to n = 16 for K and up to n = 20 for A; the closed forms reproduce the
measured nonzero counts exactly at every n built and are used above that. A solver holds at
least two copies of the model (the original and the presolved one) plus its factorization
workspace, so these are floors.

| n | m_2 | K, one copy | K with int64 indices | S | A |
|---|---|---|---|---|---|
| 12 | 24,576 | 24 MB | 33 MB | 0.3 MB | 0.6 MB |
| 15 | 245,760 | 0.8 GB | 1.1 GB | 2.9 MB | 5.8 MB |
| 16 | 524,288 | 2.6 GB | 3.4 GB | 6.3 MB | 12 MB |
| 17 | 1,114,112 | 8.2 GB | 10.9 GB | 13 MB | 26 MB |
| 18 | 2,359,296 | **26.0 GB** | 34.7 GB | 28 MB | 55 MB |
| 19 | 4,980,736 | **82.3 GB** | 109.7 GB | 59 MB | 116 MB |
| 20 | 10,485,760 | **259.8 GB** | 346.5 GB | 124 MB | 244 MB |

Full table with build times in `MATRIX_MEMORY.md`.

## What it means

**Block-Marschak form: not feasible at n >= 18 on this machine.** The Mobius matrix K grows as
3^n. One copy exceeds the 64 GB at n = 19 (82 GB) and n = 20 (260 GB) outright. At n = 18 a
single copy is 26 to 35 GB, and a solver's second copy plus the factorization of the KKT
system takes it past 64 GB. The threshold is n = 18, not lower: at n = 15 one copy is 0.8 GB and
at n = 16 it is 2.6 GB, both easily held.

**Arc-flow form: the matrix fits, the solve does not.** A has only n 2^n nonzeros, 244 MB at
n = 20, so storage is not the obstacle here. The obstacle is what an interior-point method must
factor: a system on the 2^n-node subset lattice. That graph's treewidth is of order 2^n / sqrt(n),
so a sparse Cholesky factor of the normal equations holds at least about 2^n x 2^n / (2 sqrt(n))
nonzeros: roughly 65 GB at n = 18, 250 GB at n = 19 and 1 TB at n = 20, at 8 bytes each. That is
a bound from the graph structure rather than a measurement (the size-limited Gurobi licence on
this machine cannot build models above 2,000 variables), and the package's own docstring puts
the practical limit of this QP at about n = 15.

**So neither exact formulation can be handed to a general QP solver at n = 18 to 20 here.** That
is why the CPU baselines (CGIP, CGIP-heur) use column generation with an integer-programming
pricing subproblem rather than the full QP, and why the GPU solvers keep the lattice implicit:
BM-ADMM stores only vectors of length m_2 and applies K through the butterfly transform, never
as a matrix; CGBFQP-GPU never forms a constraint matrix at all.
