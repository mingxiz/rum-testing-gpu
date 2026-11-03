"""Column bookkeeping for the CGBFQP master, without ever forming the M x K matrix V of 0/1
ranking columns:

    BitsetGramMaintainer   Q = V'V and h = V'phat, grown one column at a time. A column is stored
                           as ceil(M/64) packed uint64 words, and Q[a, b] = popcount(bits_a & bits_b).
    PoutMaintainer         the mixture p_out = V lambda, updated by the change in lambda.
"""
import numpy as np


def pack_bits(col):
    """Pack a 0/1 numpy column (length M) into uint64 words (ceil(M/64) of them).
    Uses np.packbits — vectorized, 50× faster than a Python loop."""
    M = len(col)
    nwords = (M + 63) // 64
    # Pad to multiple of 64
    padded = np.zeros(nwords * 64, dtype=np.uint8)
    padded[:M] = (col > 0.5).astype(np.uint8)
    packed_bytes = np.packbits(padded, bitorder='little')
    return packed_bytes.view(np.uint64)


def popcount64(x):
    """popcount of a uint64 array, returning a uint64 array of counts."""
    x = x.astype(np.uint64)                     # numpy bit-trick popcount (on a copy)
    M55 = np.uint64(0x5555555555555555)
    M33 = np.uint64(0x3333333333333333)
    M0F = np.uint64(0x0F0F0F0F0F0F0F0F)
    M01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & M55)
    x = (x & M33) + ((x >> np.uint64(2)) & M33)
    x = (x + (x >> np.uint64(4))) & M0F
    return (x * M01) >> np.uint64(56)


class BitsetGramMaintainer:
    """Incrementally maintain G = V^T V and h = V^T phat as columns are added."""

    def __init__(self, M, phat, max_K=2000):
        self.M = int(M)
        self.W = (self.M + 63) // 64   # number of uint64 words per column
        self.phat = phat.astype(np.float64)
        self.K = 0
        # Bitset storage: K_max × W uint64
        self.cols_bits = np.zeros((max_K, self.W), dtype=np.uint64)
        # Q (Gram matrix) and h
        self.Q = np.zeros((max_K, max_K), dtype=np.float64)
        self.h = np.zeros(max_K, dtype=np.float64)
        self.max_K = max_K

    def add_column(self, col):
        """Add column V[:, K] = col (a 0/1 numpy vector of length M)."""
        k = self.K
        if k >= self.max_K:
            raise RuntimeError(f"max_K={self.max_K} exceeded")
        bits = pack_bits(col)
        self.cols_bits[k] = bits

        # Compute G[k, j] for j = 0..k via popcount(bits & cols_bits[j])
        # Vectorize: bits broadcast against all 0..k columns
        if k > 0:
            # AND with all previous bitsets, then popcount per row
            and_results = self.cols_bits[:k] & bits       # (k, W)
            pop_per_word = popcount64(and_results)         # (k, W)
            counts = pop_per_word.sum(axis=1)              # (k,)
            self.Q[k, :k] = counts.astype(np.float64)
            self.Q[:k, k] = counts.astype(np.float64)
        # G[k, k] = popcount(bits)
        self.Q[k, k] = float(popcount64(bits).sum())

        # h[k] = sum of phat[i] where bit i is set in col
        # Iterate set bits via np.where on the original col (faster than bit unpacking)
        nz = np.where(col > 0.5)[0]
        self.h[k] = float(self.phat[nz].sum())

        self.K += 1
        return k

    def view(self):
        """Return current (Q, h) views (no copy) sized to K × K and K."""
        K = self.K
        return self.Q[:K, :K], self.h[:K]


# ---------------------------------------------------------------------------
# Incremental p_output maintainer using bitset columns
# ---------------------------------------------------------------------------
# Threshold below which we cache per-column set-bit positions (int64) on
# add_column. Above it, we go lazy: unpack the bitset on demand each time
# apply_delta touches a column. The cached-positions path is faster for
# sparse menus (typical pair/L30/L40 at n<=24), but its memory footprint is
# 8 * popcount * K_active bytes; at M >= 1e6 with dense menus, K=800 active
# columns, popcount ~ M/2, this becomes multi-GB. The lazy path is slower
# per call but uses O(M) scratch only (no per-col storage beyond the bitsets
# already kept by BitsetGramMaintainer).
_POUT_POSITION_CACHE_THRESHOLD = 1_000_000


class PoutMaintainer:
    """Maintain p_out = Σ_a λ_a · V[:, a] incrementally as λ changes.

    Two modes selected at construction by M:

      M < threshold (default 1e6):
        On add_column, precompute and cache int64 set-bit positions.
        On apply_delta, do sparse indexed add: p_out[positions[a]] += Δλ_a.
        This matches the previous implementation for pair/L30/L40 perf.

      M >= threshold:
        Lazy mode. add_column is a no-op. On apply_delta, unpack the column
        bitset from the shared BitsetGramMaintainer and do dense add:
        p_out += Δλ_a * unpacked. Memory stays O(M).
    """
    def __init__(self, M, gram_maintainer):
        self.M = M
        self.gram = gram_maintainer
        self.p_out = np.zeros(M, dtype=np.float64)
        self._cache_positions = (M < _POUT_POSITION_CACHE_THRESHOLD)
        self._positions = []   # used only when _cache_positions is True

    def add_column(self, uint8_col):
        if self._cache_positions:
            positions = np.where(uint8_col.astype(bool))[0].astype(np.int64)
            self._positions.append(positions)
        # else: bitset is already stored in self.gram.cols_bits — nothing to do

    def apply_delta(self, delta_lam):
        nz = np.where(np.abs(delta_lam) > 1e-12)[0]
        if self._cache_positions:
            for a in nz:
                a_int = int(a)
                self.p_out[self._positions[a_int]] += float(delta_lam[a])
        else:
            # Lazy: unpack on demand. Dense add (p_out += delta * unpacked) is
            # numpy-vectorized and avoids storing per-col positions for big M.
            for a in nz:
                a_int = int(a)
                bits_a = self.gram.cols_bits[a_int]
                unpacked_full = np.unpackbits(bits_a.view(np.uint8),
                                                bitorder='little')
                self.p_out += float(delta_lam[a]) * unpacked_full[:self.M]
