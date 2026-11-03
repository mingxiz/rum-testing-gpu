"""Cardinality layers of the subset lattice: every k-subset bitmask of {0..n-1}, ascending."""
import math

import numpy as np
import cupy as cp
from numba import njit


@njit(cache=True)
def _gosper_fill(out, n, k):
    """Fill `out` with the k-subsets of {0..n-1} in ascending order (Gosper's next permutation)."""
    x = (1 << k) - 1
    limit = 1 << n
    i = 0
    while x < limit:
        out[i] = x
        i += 1
        u = x & (-x)
        v = x + u
        x = v + (((v ^ x) // u) >> 2)


def generate_layer(n, k):
    """All cardinality-k subset masks of {0..n-1}, ascending, as a CuPy int32 array."""
    if k == 0:
        return cp.zeros(1, dtype=cp.int32)              # just the empty set, mask 0
    if k > n:
        return cp.empty(0, dtype=cp.int32)
    out = np.empty(math.comb(n, k), dtype=np.int32)     # exact count, preallocated
    _gosper_fill(out, n, k)
    return cp.asarray(out)                              # single host->device transfer


def precompute_layers(n):
    """Return a list `layers` with layers[k] = every k-subset bitmask of {0..n-1}
    as a device int32 array (k = 0..n). Reused on every pricing sweep."""
    return [generate_layer(n, k) for k in range(n + 1)]
