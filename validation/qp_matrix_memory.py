"""Memory to hand the RUM projection to a QP solver as explicit sparse matrices.

A commercial QP solver (Gurobi, CPLEX, ...) needs the constraint matrices in memory in sparse
form. Two exact formulations exist for min ||p_D - p_hat||^2 over the RUM polytope:

  Block-Marschak (BM) form    variables p~ in R^{m2}, m2 = n 2^(n-1) (a probability for every
                              (menu, alternative)); constraints K p~ >= 0 (the Mobius transform is
                              nonnegative), S p~ = 1 (one simplex per menu), p~ >= 0.
                              nnz(K) = n 3^(n-1),  nnz(S) = m2.
  arc-flow (AF) form          variables r in R^{m2} (a flow on the arcs of the subset lattice);
                              constraints A r = b (flow conservation at the 2^n nodes), r >= 0.
                              nnz(A) = n 2^n.

This script builds both matrices in scipy for the n it can hold, measures the bytes of the
compressed-sparse storage, checks the measured nonzero counts against the closed forms, and
extrapolates by those forms to n = 18, 19, 20. Bytes are for CSR with float64 values and
int32 indices (12 B per nonzero + row pointers); solvers keep at least one such copy and
typically two or three (presolved copy, factorization workspace), so the numbers are floors.

    python validation/qp_matrix_memory.py [--max-n-k 16] [--max-n-a 20]
Writes replication/qp_matrix_memory/MATRIX_MEMORY.md.
"""
import argparse
import os
import time

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "replication", "qp_matrix_memory")


def slab_index(D, x, n):
    """(menu mask D, alternative x in D) -> index in the n 2^(n-1) slab (compressed tail)."""
    low = D & ((np.int64(1) << x) - 1)
    high = (D >> (x + 1)) << x
    return x * (np.int64(1) << (n - 1)) + (low | high)


def build_K(n):
    """K p~ >= 0: for every (D, x) with x in D, sum over E subset of D with x in E of
    (-1)^{|D| - |E|} p~(x | E)  >= 0.  Rows and columns in slab order."""
    N = 1 << n
    rows, cols, vals = [], [], []
    masks = np.arange(N, dtype=np.int64)
    pop = np.array([bin(m).count("1") for m in range(N)], dtype=np.int8)
    for E in range(1, N):
        comp = (N - 1) ^ E
        # all T subset of comp: enumerate submasks
        sub = masks[(masks & ~comp) == 0]              # T with T subset of comp
        D = E | sub                                    # supersets of E
        sign = np.where(pop[sub] % 2 == 0, 1, -1).astype(np.int8)
        xs = [x for x in range(n) if (E >> x) & 1]
        for x in xs:
            rows.append(slab_index(D, x, n)); cols.append(np.full(D.size, slab_index(np.int64(E), x, n), dtype=np.int64)); vals.append(sign)
    r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
    m2 = n * (1 << (n - 1))
    return sp.csr_matrix((v.astype(np.float64), (r, c)), shape=(m2, m2))


def build_S(n):
    """S p~ = 1: one row per nonempty menu D, entries at (D, x) for x in D."""
    N = 1 << n
    rows, cols = [], []
    for x in range(n):
        D = np.arange(N, dtype=np.int64)
        D = D[(D >> x) & 1 == 1]
        rows.append(D); cols.append(slab_index(D, x, n))
    r = np.concatenate(rows); c = np.concatenate(cols)
    return sp.csr_matrix((np.ones(r.size), (r, c)), shape=(N, n * (1 << (n - 1))))


def build_A(n):
    """A r = b: flow conservation at every node D of the subset lattice; the arc (D, x)
    leaves D (+1) and enters D \\ {x} (-1)."""
    N = 1 << n
    rows, cols, vals = [], [], []
    for x in range(n):
        D = np.arange(N, dtype=np.int64)
        D = D[(D >> x) & 1 == 1]
        a = slab_index(D, x, n)
        rows += [D, D ^ (np.int64(1) << x)]; cols += [a, a]; vals += [np.ones(D.size), -np.ones(D.size)]
    r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
    return sp.csr_matrix((v, (r, c)), shape=(N, n * (1 << (n - 1))))


def csr_bytes(M, index_bytes=4):
    return M.nnz * (8 + index_bytes) + (M.shape[0] + 1) * index_bytes


def fmt_b(b):
    for unit, div in (("TB", 2 ** 40), ("GB", 2 ** 30), ("MB", 2 ** 20), ("KB", 2 ** 10)):
        if b >= div:
            return "%.1f %s" % (b / div, unit)
    return "%d B" % b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n-k", type=int, default=16)
    ap.add_argument("--max-n-a", type=int, default=20)
    a = ap.parse_args()
    lines = ["# Memory to hand the RUM projection to a QP solver as explicit sparse matrices\n",
             "Measured with scipy CSR (float64 values, int32 indices: 12 B per nonzero plus row pointers); the closed "
             "forms are checked against the measured nonzero counts and used above the largest n built here. A QP "
             "solver holds at least one such copy, typically two or three (presolved model, factorization "
             "workspace), so these are floors. This machine has 64 GB of host memory.\n",
             "## Block-Marschak form: K p~ >= 0 (nnz = n 3^(n-1)), S p~ = 1 (nnz = n 2^(n-1))\n",
             "| n | m2 = n 2^(n-1) | nnz(K) measured | nnz(K) = n 3^(n-1) | K bytes (CSR, int32) | K bytes (int64 idx) | S bytes | build s |",
             "|---|---|---|---|---|---|---|---|"]
    rowsK = []
    for n in range(8, a.max_n_k + 1):
        t0 = time.perf_counter()
        K = build_K(n); S = build_S(n)
        dt = time.perf_counter() - t0
        m2 = n * (1 << (n - 1))
        assert K.nnz == n * 3 ** (n - 1), (n, K.nnz)
        assert S.nnz == m2
        rowsK.append((n, m2, K.nnz, csr_bytes(K), csr_bytes(K, 8), csr_bytes(S), dt))
        lines.append("| %d | %s | %s | %s | %s | %s | %s | %.1f |" % (n, format(m2, ","), format(K.nnz, ","), format(n * 3 ** (n - 1), ","),
                                                                     fmt_b(csr_bytes(K)), fmt_b(csr_bytes(K, 8)), fmt_b(csr_bytes(S)), dt))
        print("K n=%d nnz=%s bytes=%s (%.1fs)" % (n, format(K.nnz, ","), fmt_b(csr_bytes(K)), dt), flush=True)
        del K, S
    for n in range(a.max_n_k + 1, 21):
        m2 = n * (1 << (n - 1)); nnz = n * 3 ** (n - 1)
        lines.append("| %d | %s | (formula) | %s | %s | %s | %s | - |" % (n, format(m2, ","), format(nnz, ","),
                                                                        fmt_b(nnz * 12 + (m2 + 1) * 4), fmt_b(nnz * 16 + (m2 + 1) * 8), fmt_b(m2 * 12 + ((1 << n) + 1) * 4)))
    lines += ["", "## Arc-flow form: A r = b (nnz = n 2^n), 2^n rows, n 2^(n-1) columns\n",
              "| n | rows 2^n | cols m2 | nnz(A) measured | nnz = n 2^n | A bytes (CSR, int32) | build s |", "|---|---|---|---|---|---|---|"]
    for n in range(8, a.max_n_a + 1):
        t0 = time.perf_counter()
        A = build_A(n)
        dt = time.perf_counter() - t0
        assert A.nnz == n * (1 << n), (n, A.nnz)
        lines.append("| %d | %s | %s | %s | %s | %s | %.1f |" % (n, format(1 << n, ","), format(n * (1 << (n - 1)), ","), format(A.nnz, ","),
                                                             format(n * (1 << n), ","), fmt_b(csr_bytes(A)), dt))
        print("A n=%d nnz=%s bytes=%s (%.1fs)" % (n, format(A.nnz, ","), fmt_b(csr_bytes(A)), dt), flush=True)
        del A
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "MATRIX_MEMORY.md")
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
