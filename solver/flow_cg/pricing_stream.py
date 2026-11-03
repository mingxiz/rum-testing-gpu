"""Rank-indexed, two-live-layer Bellman-Ford pricing for CGBFQP-GPU.

The subset lattice is swept layer by layer (layer k = all k-subsets, colex
order = ascending mask order, the order the mask-indexed kernel uses too).
Only layers k-1 and k are resident:

    V_k[r] = max over x in D(r) of  V_{k-1}[rank_{k-1}(D \\ {x})] + w(x, D),
    w(x, D) = sum of scores of the stored menus D' with x in D', D' subset of D,

with every subset addressed by its colex rank within its layer. Thread r
unranks its subset (combinatorial number system), derives the k predecessor
ranks in O(k) from prefix/suffix sums of binomials over the set bits, and
scans the menus of each alternative exactly like the mask kernel, so in
float32 mode the values, ties and the returned ranking are bit-identical to
`bf_update` (ties -> lowest alternative).

Storage modes for the two live layers
    f32 : float32 per rank.
    u16 : uint16 mantissa per rank plus a float32 (offset, step) per block of
          BLOCK consecutive ranks -- block floating point. Every stored value is
          within step/2 of the computed one; the recursion is 1-Lipschitz in each
          input, so the returned V[full] is within margin = sum over layers of
          (largest step in the layer)/2 of its float32 counterpart. The margin is
          returned and the solver adds it to Vmax before forming the Frank-Wolfe
          gap, which keeps the lower-bound certificate valid.
    u24 : the same with a 24-bit mantissa (uint16 high + uint8 low arrays, 3 B per
          rank): step = range / (2^24 - 1), 256x smaller than u16. The quantized
          index is formed in float32, which is exact only up to 2^24, so the
          per-layer bound is taken as one full step (not step/2): margin = sum of
          the largest steps. Needed where the certificate must close on a small
          distance (the u16 margin floors the bracket at about margin / UB).

Predecessors are stored only for layers k >= k_pred (uint8 per rank, a few
hundred MB at n = 35); the backtrace descends through them and then re-runs the
mask-indexed DP on the sub-lattice of the node reached at layer k_pred (2^k_pred
nodes, e.g. 2^25 at n = 35, 0.1 % of a sweep), which gives exactly the same
values and path because V[S] depends only on subsets of S.

Device memory: two layers of C(n, n//2) ranks -> 4 B (f32) or 2 B (u16) each:
n = 33: 8.7 / 4.4 GiB, n = 34: 17.4 / 8.7 GiB, n = 35: 33.8 / 16.9 GiB.

sweep_max64 is the same sweep in float64 that returns only V[full] (no predecessors, no
quantization): the RUM class's lower bound where the float64 slab does not fit (n >= 27 on
32 GB), 8 B per rank, so n <= 33 on 32 GB. It is a separate module, so the pricing kernels
above compile exactly as before.
"""
import math

import numpy as np
import cupy as cp

from ..common.layers import precompute_layers
from .bf_pricing import bf as bf_mask_kernel
from .cg_core import binom_table

BLOCK = 1024                  # ranks per quantization block (= THREADS * VPT)
THREADS = 256
VPT = BLOCK // THREADS
MAXN = 40

_SRC = r'''
#define BLOCK %(BLOCK)d
#define THREADS %(THREADS)d
#define VPT %(VPT)d
#define LOG2BLOCK %(LOG2BLOCK)d
#define MAXN %(MAXN)d
typedef unsigned long long ull;

#define LOAD_F32(rp) (vprev[(rp)])
#define LOAD_U16(rp) (mprev[(rp) >> LOG2BLOCK] + (float)vprev[(rp)] * eprev[(rp) >> LOG2BLOCK])
#define LOAD_U24(rp) (mprev[(rp) >> LOG2BLOCK] + (float)((((unsigned int)vprev[(rp)]) << 8) | (unsigned int)lprev[(rp)]) * eprev[(rp) >> LOG2BLOCK])

#define BODY(LOADPREV)                                                                               \
    const ull base = (ull)blockIdx.x * BLOCK;                                                        \
    float vals[VPT]; unsigned char preds[VPT];                                                       \
    float tmin = 3.0e38f, tmax = -3.0e38f;                                                           \
    for (int j = 0; j < VPT; ++j) {                                                                  \
        ull r = base + (ull)j * THREADS + threadIdx.x;                                               \
        vals[j] = 0.0f; preds[j] = 0;                                                                \
        if (r < count) {                                                                             \
            ull rr = r; int b = n - 1; int elems[MAXN]; ull D = 0ULL;                                \
            for (int t = k; t >= 1; --t) {                                                           \
                while (binom[b * (n + 1) + t] > rr) --b;                                             \
                elems[t - 1] = b; D |= (1ULL << b); rr -= binom[b * (n + 1) + t]; --b;               \
            }                                                                                        \
            float best = 0.0f; bool seen = false; int bestx = 0;                                     \
            ull suf = 0ULL, remain = r;                                                              \
            for (int t = k - 1; t >= 0; --t) {                                                       \
                int x = elems[t];                                                                    \
                remain -= binom[x * (n + 1) + (t + 1)];                                              \
                ull rp = remain + suf;                                                               \
                suf += binom[x * (n + 1) + t];                                                       \
                float vp = LOADPREV(rp);                                                             \
                float w = 0.0f;                                                                      \
                ull lo = offsets[x], hi = offsets[x + 1];                                            \
                for (ull i = lo; i < hi; ++i) { ull Dp = menu_masks[i]; if ((Dp & D) == Dp) w += menu_scores[i]; } \
                float cand = vp + w;                                                                 \
                if (!seen || cand >= best) { best = cand; seen = true; bestx = x; }                  \
            }                                                                                        \
            vals[j] = best; preds[j] = (unsigned char)bestx;                                         \
            if (best < tmin) tmin = best; if (best > tmax) tmax = best;                              \
        }                                                                                            \
    }

extern "C" __global__ void bf_stream_f32(
    const ull* __restrict__ binom, const int n, const int k, const ull count,
    const float* __restrict__ vprev, const float* __restrict__ mprev, const float* __restrict__ eprev,
    float* __restrict__ vcur, float* __restrict__ mcur, float* __restrict__ ecur,
    unsigned char* __restrict__ pred, const int store_pred,
    const ull* __restrict__ menu_masks, const float* __restrict__ menu_scores, const ull* __restrict__ offsets)
{
    BODY(LOAD_F32)
    for (int j = 0; j < VPT; ++j) {
        ull r = base + (ull)j * THREADS + threadIdx.x;
        if (r < count) { vcur[r] = vals[j]; if (store_pred) pred[r] = preds[j]; }
    }
}

extern "C" __global__ void bf_stream_u16(
    const ull* __restrict__ binom, const int n, const int k, const ull count,
    const unsigned short* __restrict__ vprev, const float* __restrict__ mprev, const float* __restrict__ eprev,
    unsigned short* __restrict__ vcur, float* __restrict__ mcur, float* __restrict__ ecur,
    unsigned char* __restrict__ pred, const int store_pred,
    const ull* __restrict__ menu_masks, const float* __restrict__ menu_scores, const ull* __restrict__ offsets)
{
    BODY(LOAD_U16)
    __shared__ float smin[THREADS / 32], smax[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) {
        tmin = fminf(tmin, __shfl_xor_sync(0xffffffffu, tmin, o));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
    }
    if ((threadIdx.x & 31) == 0) { smin[threadIdx.x >> 5] = tmin; smax[threadIdx.x >> 5] = tmax; }
    __syncthreads();
    float bmin = smin[0], bmax = smax[0];
    for (int i = 1; i < THREADS / 32; ++i) { bmin = fminf(bmin, smin[i]); bmax = fmaxf(bmax, smax[i]); }
    float step = (bmax - bmin) / 65535.0f;          // 0 when the block is constant: q = 0, no error
    if (threadIdx.x == 0) { mcur[blockIdx.x] = bmin; ecur[blockIdx.x] = step; }
    for (int j = 0; j < VPT; ++j) {
        ull r = base + (ull)j * THREADS + threadIdx.x;
        if (r < count) {
            float q = (step > 0.0f) ? rintf((vals[j] - bmin) / step) : 0.0f;
            q = fminf(fmaxf(q, 0.0f), 65535.0f);
            vcur[r] = (unsigned short)q;
            if (store_pred) pred[r] = preds[j];
        }
    }
}

extern "C" __global__ void bf_stream_u24(
    const ull* __restrict__ binom, const int n, const int k, const ull count,
    const unsigned short* __restrict__ vprev, const unsigned char* __restrict__ lprev,
    const float* __restrict__ mprev, const float* __restrict__ eprev,
    unsigned short* __restrict__ vcur, unsigned char* __restrict__ lcur,
    float* __restrict__ mcur, float* __restrict__ ecur,
    unsigned char* __restrict__ pred, const int store_pred,
    const ull* __restrict__ menu_masks, const float* __restrict__ menu_scores, const ull* __restrict__ offsets)
{
    BODY(LOAD_U24)
    __shared__ float smin[THREADS / 32], smax[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) {
        tmin = fminf(tmin, __shfl_xor_sync(0xffffffffu, tmin, o));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
    }
    if ((threadIdx.x & 31) == 0) { smin[threadIdx.x >> 5] = tmin; smax[threadIdx.x >> 5] = tmax; }
    __syncthreads();
    float bmin = smin[0], bmax = smax[0];
    for (int i = 1; i < THREADS / 32; ++i) { bmin = fminf(bmin, smin[i]); bmax = fmaxf(bmax, smax[i]); }
    float step = (bmax - bmin) / 16777215.0f;       // 2^24 - 1 levels; 0 when the block is constant
    if (threadIdx.x == 0) { mcur[blockIdx.x] = bmin; ecur[blockIdx.x] = step; }
    for (int j = 0; j < VPT; ++j) {
        ull r = base + (ull)j * THREADS + threadIdx.x;
        if (r < count) {
            float q = (step > 0.0f) ? rintf((vals[j] - bmin) / step) : 0.0f;
            q = fminf(fmaxf(q, 0.0f), 16777215.0f);
            unsigned int qi = (unsigned int)q;
            vcur[r] = (unsigned short)(qi >> 8);
            lcur[r] = (unsigned char)(qi & 255u);
            if (store_pred) pred[r] = preds[j];
        }
    }
}
''' % dict(BLOCK=BLOCK, THREADS=THREADS, VPT=VPT, LOG2BLOCK=int(math.log2(BLOCK)), MAXN=MAXN)

_MODULE = cp.RawModule(code=_SRC, options=("--std=c++11",))
_K_F32 = _MODULE.get_function("bf_stream_f32")
_K_U16 = _MODULE.get_function("bf_stream_u16")
_K_U24 = _MODULE.get_function("bf_stream_u24")

# float64, maximum only (sweep_max64): BODY's unranking and predecessor ranks, one rank per thread
_SRC64 = r'''
typedef unsigned long long ull;
extern "C" __global__ void bf_stream_max64(
    const ull* __restrict__ binom, const int n, const int k, const ull count,
    const double* __restrict__ vprev, double* __restrict__ vcur,
    const ull* __restrict__ menu_masks, const double* __restrict__ menu_scores, const ull* __restrict__ offsets)
{
    ull r = (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;
    ull rr = r; int b = n - 1; int elems[64]; ull D = 0ULL;
    for (int t = k; t >= 1; --t) {
        while (binom[b * (n + 1) + t] > rr) --b;
        elems[t - 1] = b; D |= (1ULL << b); rr -= binom[b * (n + 1) + t]; --b;
    }
    double best = 0.0; bool seen = false;
    ull suf = 0ULL, remain = r;
    for (int t = k - 1; t >= 0; --t) {
        int x = elems[t];
        remain -= binom[x * (n + 1) + (t + 1)];
        ull rp = remain + suf;
        suf += binom[x * (n + 1) + t];
        double w = 0.0;
        for (ull i = offsets[x], hi = offsets[x + 1]; i < hi; ++i) { ull Dp = menu_masks[i]; if ((Dp & D) == Dp) w += menu_scores[i]; }
        double cand = vprev[rp] + w;
        if (!seen || cand > best) { best = cand; seen = true; }
    }
    vcur[r] = best;
}
'''
_K_MAX64 = []                 # compiled on first use


def sweep_max64_bytes(n, rows):
    """Device bytes of sweep_max64: two float64 layers of C(n, n//2) ranks, the rows, the tables, and
    4 KiB for the memory pool's rounding of each allocation to 512-byte blocks."""
    return 2 * 8 * math.comb(n, n // 2) + 16 * int(rows) + 8 * (n + 1) * (n + 2) + 4096


def sweep_max64(n, menu_masks, menu_scores, offsets):
    """max over rankings of the summed scores (V[full] of the pricing sweep), in float64 with two live
    layers. Rows in x-major order: offsets[x]:offsets[x+1] are alternative x's rows, menu_masks their
    menus (bitmasks), menu_scores their scores (float64)."""
    if not _K_MAX64:
        _K_MAX64.append(cp.RawModule(code=_SRC64, options=("--std=c++11",)).get_function("bf_stream_max64"))
    n = int(n)
    if n > 62:
        raise ValueError("sweep_max64 handles n <= 62")
    counts = [math.comb(n, k) for k in range(n + 1)]
    buf = [cp.zeros(max(counts), dtype=cp.float64), cp.zeros(max(counts), dtype=cp.float64)]
    masks = cp.asarray(np.asarray(menu_masks, dtype=np.uint64))
    scores = cp.asarray(np.asarray(menu_scores, dtype=np.float64))
    offs = cp.asarray(np.asarray(offsets, dtype=np.uint64))
    binom = cp.asarray(binom_table(n))
    prev, cur = 0, 1                                        # layer 0: V[empty set] = 0
    for k in range(1, n + 1):
        _K_MAX64[0](((counts[k] + 255) // 256,), (256,),
                    (binom, np.int32(n), np.int32(k), np.uint64(counts[k]), buf[prev], buf[cur], masks, scores, offs))
        prev, cur = cur, prev
    return float(buf[prev][0].get())


_LAYER_CACHE = {}


def colex_rank(mask):
    """Colex rank of a subset (int mask) within its cardinality layer."""
    r, t = 0, 0
    b = 0
    while mask:
        if mask & 1:
            t += 1
            r += math.comb(b, t)
        mask >>= 1
        b += 1
    return r


class StreamPricer:
    """Prices with two live layers. price(scores) -> (Vmax, ranking, margin)."""

    def __init__(self, n, choice_set, choice_alternative, offsets, mode="u16", k_pred=None):
        self.n = int(n)
        if self.n > MAXN - 1:
            raise ValueError("n too large for the stream kernel (MAXN=%d)" % MAXN)
        self.mode = mode
        self.k_pred = (self.n - 10) if k_pred is None else int(k_pred)
        self.k_pred = max(1, min(self.k_pred, self.n))
        self.menu_masks = cp.asarray(np.asarray(choice_set, dtype=np.uint64))
        self.alts = np.asarray(choice_alternative, dtype=np.int32)
        self.menu_masks_host = np.asarray(choice_set, dtype=np.uint64)
        self.offsets = cp.asarray(np.asarray(offsets, dtype=np.uint64))
        self.binom = cp.asarray(binom_table(self.n))
        self.counts = [math.comb(self.n, k) for k in range(self.n + 1)]
        cap = max(self.counts)
        nblk = (cap + BLOCK - 1) // BLOCK
        if mode not in ("f32", "u16", "u24"):
            raise ValueError("mode must be f32, u16 or u24")
        vt = cp.float32 if mode == "f32" else cp.uint16
        self.vbuf = [cp.zeros(cap, dtype=vt), cp.zeros(cap, dtype=vt)]
        self.lbuf = [cp.zeros(cap, dtype=cp.uint8), cp.zeros(cap, dtype=cp.uint8)] if mode == "u24" else None
        self.mbuf = [cp.zeros(nblk, dtype=cp.float32), cp.zeros(nblk, dtype=cp.float32)]
        self.ebuf = [cp.zeros(nblk, dtype=cp.float32), cp.zeros(nblk, dtype=cp.float32)]
        self.layer_step = cp.zeros(self.n + 1, dtype=cp.float32)     # largest quantization step per layer
        self.pred = {k: cp.zeros(self.counts[k], dtype=cp.uint8) for k in range(self.k_pred, self.n + 1)}
        self._dummy_pred = cp.zeros(1, dtype=cp.uint8)
        self.kernel = {"f32": _K_F32, "u16": _K_U16, "u24": _K_U24}[mode]
        self.device_bytes = (2 * cap * {"f32": 4, "u16": 2, "u24": 3}[mode] + 2 * nblk * 8
                             + sum(self.counts[k] for k in range(self.k_pred, self.n + 1)) + 5 * (1 << self.k_pred))
        # sub-lattice scaffolding (mask-indexed, 2^k_pred nodes), built lazily
        self._sub = None

    # ------------------------------------------------------------------ sweep
    def sweep(self, scores):
        """Forward pass. scores: float32 device array aligned with the menus.
        Returns (Vmax, margin)."""
        n = self.n
        cur, prev = 0, 1
        # layer 0: one node, V = 0 (u16: offset 0, step 0, q 0)
        self.vbuf[prev][:1] = 0
        if self.lbuf is not None:
            self.lbuf[prev][:1] = 0
        self.mbuf[prev][:1] = 0.0
        self.ebuf[prev][:1] = 0.0
        for k in range(1, n + 1):
            count = self.counts[k]
            grid = (count + BLOCK - 1) // BLOCK
            store = 1 if k >= self.k_pred else 0
            pred = self.pred[k] if store else self._dummy_pred
            if self.mode == "u24":
                self.kernel((grid,), (THREADS,),
                            (self.binom, np.int32(n), np.int32(k), np.uint64(count),
                             self.vbuf[prev], self.lbuf[prev], self.mbuf[prev], self.ebuf[prev],
                             self.vbuf[cur], self.lbuf[cur], self.mbuf[cur], self.ebuf[cur],
                             pred, np.int32(store), self.menu_masks, scores, self.offsets))
            else:
                self.kernel((grid,), (THREADS,),
                            (self.binom, np.int32(n), np.int32(k), np.uint64(count),
                             self.vbuf[prev], self.mbuf[prev], self.ebuf[prev],
                             self.vbuf[cur], self.mbuf[cur], self.ebuf[cur],
                             pred, np.int32(store), self.menu_masks, scores, self.offsets))
            if self.mode != "f32":
                cp.max(self.ebuf[cur][:grid], out=self.layer_step[k])         # 0-d view: no host sync inside the sweep
            cur, prev = prev, cur
        cp.cuda.Device().synchronize()
        if self.mode == "f32":
            vmax = float(self.vbuf[prev][0].get())
            margin = 0.0
        elif self.mode == "u16":
            vmax = float(self.mbuf[prev][0].get()) + float(self.vbuf[prev][0].get()) * float(self.ebuf[prev][0].get())
            margin = 0.5 * float(cp.sum(self.layer_step[1:n + 1]).get())    # each layer adds at most step/2
        else:
            q = (int(self.vbuf[prev][0].get()) << 8) | int(self.lbuf[prev][0].get())
            vmax = float(self.mbuf[prev][0].get()) + q * float(self.ebuf[prev][0].get())
            margin = float(cp.sum(self.layer_step[1:n + 1]).get())          # one full step per layer (f32 index rounding)
        self._last_top = prev
        return vmax, margin

    # -------------------------------------------------------------- backtrace
    def backtrace(self, scores_host):
        """Descend through the stored predecessor layers, then solve the
        sub-lattice of the node reached at layer k_pred exactly. Returns the
        ranking as rank positions (rank_pos[a] = position of alternative a,
        0 = most preferred), like the mask kernel."""
        n = self.n
        rank_pos = np.zeros(n, dtype=np.int32)
        D = (1 << n) - 1
        pos = 0
        for k in range(n, self.k_pred, -1):
            r = colex_rank(D)
            x = int(self.pred[k][r].get())
            rank_pos[x] = pos
            pos += 1
            D ^= (1 << x)
        # sub-lattice of D (|D| = k_pred): relabel to 0..K-1 and run the mask-indexed DP
        K = self.k_pred
        elems = [b for b in range(n) if (D >> b) & 1]
        assert len(elems) == K
        sub_order = self._solve_sublattice(elems, scores_host)      # rank_pos over 0..K-1
        for j, e in enumerate(elems):
            rank_pos[e] = pos + int(sub_order[j])
        return rank_pos

    def _solve_sublattice(self, elems, scores_host):
        K = len(elems)
        emask = 0
        for e in elems:
            emask |= 1 << e
        inside = (self.menu_masks_host & np.uint64(emask)) == self.menu_masks_host
        idx = np.nonzero(inside)[0]
        sub_masks = np.zeros(idx.size, dtype=np.uint32)
        for j, e in enumerate(elems):
            sub_masks |= (((self.menu_masks_host[idx] >> np.uint64(e)) & np.uint64(1)).astype(np.uint32) << np.uint32(j))
        pos_of = {e: j for j, e in enumerate(elems)}
        sub_alts = np.array([pos_of[int(a)] for a in self.alts[idx]], dtype=np.int32)
        sub_scores = np.asarray(scores_host, dtype=np.float32)[idx]
        perm = np.argsort(sub_alts, kind="stable")           # x-major layout, as CGData.prepare_choice_data
        sub_masks, sub_alts, sub_scores = sub_masks[perm], sub_alts[perm], sub_scores[perm]
        offs = np.concatenate(([0], np.cumsum(np.bincount(sub_alts, minlength=K)))).astype(np.uint64)
        if self._sub is None or self._sub["K"] != K:
            if K not in _LAYER_CACHE:
                _LAYER_CACHE[K] = precompute_layers(K)
            self._sub = dict(K=K, V=cp.zeros(1 << K, dtype=cp.float32), pred=cp.zeros(1 << K, dtype=cp.uint8),
                             layers=_LAYER_CACHE[K])
        V, pred, layers = self._sub["V"], self._sub["pred"], self._sub["layers"]
        V.fill(0); pred.fill(0)
        g_masks = cp.asarray(sub_masks); g_scores = cp.asarray(sub_scores); g_offs = cp.asarray(offs)
        for c in range(1, K + 1):
            st = layers[c]; ns = int(st.size)
            bf_mask_kernel(((ns + 255) // 256,), (256,), (V, pred, g_masks, g_scores, g_offs, st, ns))
        cp.cuda.Device().synchronize()
        Dm = (1 << K) - 1
        order = np.zeros(K, dtype=np.int32)
        pred_h = cp.asnumpy(pred)
        for i in range(K):
            x = int(pred_h[Dm]); order[x] = i; Dm ^= (1 << x)
        return order

    def price(self, scores_host):
        """One pricing call: returns (Vmax, rank_pos ranking, margin)."""
        g = cp.asarray(np.asarray(scores_host, dtype=np.float32))
        vmax, margin = self.sweep(g)
        ranking = self.backtrace(scores_host)
        return vmax, ranking, margin
