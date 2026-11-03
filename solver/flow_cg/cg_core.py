"""Observation layout for the GPU column generation: observations grouped by
chosen alternative (the layout the pricing kernel expects), the cardinality
layers of the subset lattice, and the deterministic choice column of a
ranking. No master-QP dependency (the master lives in solver.py).

Layers (all k-subset masks of {0..n-1}, k = 1..n) are what the Bellman-Ford
sweep iterates over. Two ways to provide them:

  layers="stored"  precompute every layer once on the device (4 B per lattice
                   node in total, i.e. 4 * 2^n bytes) -- the original layout.
  layers="stream"  generate each layer on the fly, just before its kernel
                   launch, into ONE reusable buffer of C(n, n//2) masks, with a
                   colex-unranking kernel (thread r writes the r-th k-subset in
                   ascending mask order, the same order Gosper's hack produces).
                   Device memory drops from 9 to 5 bytes per node (V float32 +
                   pred uint8) plus the buffer, which is what makes n = 31, 32
                   fit on a 32 GB card (see MEMORY.md in this folder).
  layers="auto"    stored for n <= 28, stream above.
"""
import math

import numpy as np
import cupy as cp
from numba import njit

from ..common.layers import precompute_layers

_UNRANK_SRC = r'''
extern "C" __global__ void unrank_layer(unsigned int* __restrict__ out,
                                        const unsigned long long* __restrict__ binom,
                                        const int n, const int k,
                                        const unsigned long long count)
{
    unsigned long long r = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;
    unsigned long long rr = r;
    unsigned int mask = 0u;
    int b = n - 1;
    // colex unranking: the largest element b_{k-1} is the largest b with C(b,k) <= r,
    // then recurse on r - C(b,k) with k-1 over elements below b.
    for (int t = k; t >= 1; --t) {
        while (binom[b * (n + 1) + t] > rr) --b;
        mask |= (1u << b);
        rr -= binom[b * (n + 1) + t];
        --b;
    }
    out[r] = mask;
}
'''
_UNRANK = cp.RawKernel(_UNRANK_SRC, "unrank_layer")


def binom_table(n):
    """C(b, t) for 0 <= b, t <= n, flattened row-major: entry b*(n+1) + t (0 if t > b), uint64.
    The layout the unranking kernels index."""
    tab = np.zeros((n + 1) * (n + 1), dtype=np.uint64)
    for b in range(n + 1):
        for t in range(b + 1):
            tab[b * (n + 1) + t] = math.comb(b, t)
    return tab


@njit(cache=True)
def map_ranking_to_choice(choice_set, choice_alternative, rank_pos, n):
    """Numba-compiled loop form of the deterministic choice vector.
    CGData.map_ranking_to_choice_probability delegates to it (numba cannot compile a method)."""
    M = choice_set.shape[0]
    out = np.empty(M, dtype=np.uint8)
    for s in range(M):
        mask = choice_set[s]
        best_alt = -1
        best_rank = n + 1
        for a in range(n):
            if (mask >> a) & 1:                # alternative a is in the menu
                r = rank_pos[a]
                if r < best_rank:              # most-preferred member so far
                    best_rank = r
                    best_alt = a
        out[s] = 1 if best_alt == choice_alternative[s] else 0
    return out


class CGData:
    def __init__(self, choice_data, num_alternatives, layers="auto"):
        self.n = int(num_alternatives)
        if self.n > 32 and layers != "none":
            raise ValueError("flow_cg mask-indexed lattice works for num_alternatives <= 32; "
                             "use pricing='stream_f32' or 'stream_u16' for n >= 33 (see MEMORY.md)")
        if self.n > 63:
            raise ValueError("menus are 64-bit masks: num_alternatives <= 63")
        (self.choice_set, self.choice_alternative,
         self.choice_probabilities, self.offsets) = self.prepare_choice_data(choice_data, self.n)
        if layers == "auto":
            layers = "stored" if self.n <= 28 else "stream"
        self.layers_mode = layers
        if layers == "stored":
            self.layers = precompute_layers(self.n)     # layers[k]: all k-subset masks (device int32)
            self._buf = None
        elif layers == "none":                          # rank-indexed pricer: no mask arrays at all
            self.layers = None
            self._buf = None
        else:
            self.layers = None
            self._binom = cp.asarray(binom_table(self.n))
            self._buf = cp.empty(math.comb(self.n, self.n // 2), dtype=cp.uint32)
        cp.cuda.Device().synchronize()

    def layer(self, k):
        """Masks of all k-subsets, ascending, as a device array (a view into the
        reusable buffer in stream mode -- consume before requesting the next layer)."""
        if self.layers is not None:
            return self.layers[k]
        if self._buf is None:
            raise RuntimeError("CGData(layers='none') has no layer masks; use the stream pricer")
        count = math.comb(self.n, k)
        if count == 0:
            return self._buf[:0]
        view = self._buf[:count]
        threads = 256
        grid = (count + threads - 1) // threads
        _UNRANK((grid,), (threads,), (view, self._binom, np.int32(self.n), np.int32(k), np.uint64(count)))
        return view

    def device_bytes(self):
        """Persistent device memory of the pricing scaffolding for this n and mode."""
        Nnode = 1 << self.n
        if self.layers_mode == "none":
            return 0
        base = 5 * Nnode                                     # V float32 + pred uint8
        if self.layers is not None:
            return base + 4 * Nnode
        return base + 4 * math.comb(self.n, self.n // 2)

    @staticmethod
    def prepare_choice_data(choice_data, n):
        """x-major layout: offsets[x]:offsets[x+1] is alternative x's block."""
        choice_set = choice_data[:, 0].astype(np.int64).astype(np.uint32 if n <= 32 else np.uint64)
        choice_alternative = choice_data[:, 1].astype(np.int32)
        choice_probabilities = choice_data[:, 2].astype(np.float32)
        perm = np.argsort(choice_alternative, kind="stable")
        choice_set, choice_alternative = choice_set[perm], choice_alternative[perm]
        choice_probabilities = choice_probabilities[perm]
        counts = np.bincount(choice_alternative, minlength=n)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return choice_set, choice_alternative, choice_probabilities, offsets

    def map_ranking_to_choice_probability(self, rank_pos):
        """0/1 column of the ranking (rank_pos[a] = position of alternative a)."""
        return map_ranking_to_choice(self.choice_set, self.choice_alternative, rank_pos, self.n)


def selftest(n=12):
    """Streamed layers must equal the stored (Gosper) layers exactly."""
    cd = np.array([[3.0, 0.0, 0.5], [3.0, 1.0, 0.5]])
    a = CGData(cd, n, layers="stored")
    b = CGData(cd, n, layers="stream")
    for k in range(1, n + 1):
        la = cp.asnumpy(a.layer(k)).astype(np.uint32)
        lb = cp.asnumpy(b.layer(k))
        assert la.shape == lb.shape and np.array_equal(la, lb), ("layer mismatch", k)
    return True
