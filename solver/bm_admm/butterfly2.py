r"""Butterfly v2 for BM-ADMM: a pair-indexed blocked kernel plus a stage plan measured on this GPU.

Why (measured on an RTX 4060 Ti):

  bfly_blk wastes half its threads. Its inner loop guards with
  `if(((l>>j)&1)==0)`, so half the block idles at every one of the B barriers,
  and for j>=5 those are WHOLE idle warps. Cost therefore RISES with B despite
  identical traffic (f64 n=20, s=0: 121 GB/s at B=5 -> 85 GB/s at B=9).

  bfly_blk2 (here) uses 2^(B-1) threads with pair indexing
      lo = ((t>>j)<<(j+1)) | (t & ((1<<j)-1)),   hi = lo | (1<<j)
  so EVERY thread works at EVERY stage. Same FLOPs, same traffic, no
  divergence, half the block size. f64 n=20 s=0: 203 GB/s at B=6 (2.4x blk).

  Neither blocked kernel fixes the STRIDED global access at s>0 (consecutive
  threads land 2^s apart). There it goes transaction-bound: f64 and f32 cost
  the SAME wall time per stage (0.860 vs 0.850 ms at n=20, s=9, B=6) because
  the sector count, not the byte count, is the limit. bfly_co exists for that
  case and wins it in f32; in f64 it is bandwidth-bound and blk2 still edges
  it. So the right kernel is n- AND dtype-dependent.

Hence: measure the stage cost table, then shortest-path DP over consumed-bit
count for the cheapest decomposition. Stage costs are additive (the measured
back-to-back gap is +0.004 ms at n=20), so the DP searches ALL decompositions
exactly rather than spot-checking a few.

SAFETY: any decomposition applies the same bit-stages in the same ascending
order, so every plan produces BIT-IDENTICAL output (verified rel err 0.0e+00
vs butterfly_fast for every kernel and plan). Plan choice therefore cannot
perturb the ADMM trajectory — only wall time.
"""
import numpy as np
import cupy as cp

from ..common.butterfly import _BLK_K, _CO_K

_BLK2_SRC = r'''
extern "C" __global__ void bfly_blk2(const real* __restrict__ IN,
                                     real* __restrict__ OUT,
                                     long long Nloc, int s, int B,
                                     real a, real b, real c, real d){
  extern __shared__ __align__(16) unsigned char _smem[];
  real* sh = reinterpret_cast<real*>(_smem);
  long long tilesPerItem = Nloc >> B;
  long long x     = blockIdx.x / tilesPerItem;
  long long outer = blockIdx.x % tilesPerItem;
  int half = 1 << (B - 1);
  int t = threadIdx.x;                                  // 0 .. 2^(B-1)-1
  long long low  = outer & ((1LL<<s)-1);
  long long high = (outer >> s) << (s+B);
  long long base = x*Nloc | low | high;
  long long g0 = base | ((long long)t << s);
  long long g1 = base | ((long long)(t + half) << s);
  sh[t]        = IN[g0];
  sh[t + half] = IN[g1];
  __syncthreads();
  for(int j=0;j<B;j++){
    int lo = ((t >> j) << (j+1)) | (t & ((1<<j)-1));
    int hi = lo | (1<<j);
    real X = sh[lo], Y = sh[hi];
    sh[lo] = a*X + b*Y;
    sh[hi] = c*X + d*Y;
    __syncthreads();
  }
  OUT[g0] = sh[t];
  OUT[g1] = sh[t + half];
}'''


def _compile2():
    out = {}
    for dt, ty in ((np.float32, "float"), (np.float64, "double")):
        k = cp.RawKernel("typedef %s real;\n" % ty + _BLK2_SRC, "bfly_blk2")
        try:
            k.max_dynamic_shared_size_bytes = 1 << 16
        except Exception:
            pass
        out[np.dtype(dt)] = k
    return out


_BLK2_K = _compile2()
_KERNELS = {"blk": _BLK_K, "blk2": _BLK2_K, "co": _CO_K}

_PLAN_CACHE = {}          # (device, dtype, nbits) -> plan
_B_FLOOR = 4              # smallest tile width worth calibrating (see below)


def feasible(kind, s, B, nbits, isize):
    """Launch-shape limits, straight from the kernel sources."""
    if B < 1 or s + B > nbits:
        return False
    if kind == "co":
        # needs 32 contiguous low-values below bit s, npp = 2^B/32 >= 1
        return s >= 5 and B >= 5 and (1 << B) * 32 * isize <= (1 << 16)
    if kind == "blk":
        return B <= 10 and (1 << B) * isize <= (1 << 16)     # block = 2^B
    if kind == "blk2":
        return B <= 11 and (1 << B) * isize <= (1 << 16)     # block = 2^(B-1)
    raise ValueError(kind)


def make_launch(kind, s, B, n, Nloc, nbits, dt, src, out, coeffs):
    """Thunk for one stage. Aliasing is the caller's job (src only for s==0)."""
    isize = dt.itemsize
    cast = np.float32 if isize == 4 else np.float64
    af, bf, cf, df = (cast(c) for c in coeffs)
    if kind == "co":
        nlowblk = (1 << s) >> 5
        nhigh = 1 << (nbits - s - B)
        grid, blk = (int(n * nhigh * nlowblk),), (1024,)
        shm = (1 << B) * 32 * isize
        args = (src, out, np.int64(Nloc), np.int32(s), np.int32(B),
                np.int64(nlowblk), np.int64(nhigh), af, bf, cf, df)
    else:
        grid = (int(n * (Nloc >> B)),)
        blk = ((1 << B),) if kind == "blk" else ((1 << (B - 1)),)
        shm = (1 << B) * isize
        args = (src, out, np.int64(Nloc), np.int32(s), np.int32(B),
                af, bf, cf, df)
    kern = _KERNELS[kind][dt]
    return lambda: kern(grid, blk, args, shared_mem=shm)


def _dp(nbits, cost):
    """Shortest path over consumed-bit count. cost(kind,s,B) -> ms or None."""
    INF = float("inf")
    best = [INF] * (nbits + 1)
    back = [None] * (nbits + 1)
    best[0] = 0.0
    for s in range(nbits):
        if best[s] == INF:
            continue
        for kind in ("blk", "blk2", "co"):
            for B in range(1, 12):
                c = cost(kind, s, B)
                if c is None:
                    continue
                if best[s] + c < best[s + B]:
                    best[s + B] = best[s] + c
                    back[s + B] = (kind, s, B)
    plan, s = [], nbits
    while s > 0:
        step = back[s]
        if step is None:
            raise RuntimeError("no butterfly plan reaches %d bits" % nbits)
        plan.append(step)
        s -= step[2]
    return best[nbits], list(reversed(plan))


def _time(launch, reps, warmup):
    for _ in range(warmup):
        launch()
    cp.cuda.Device().synchronize()
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    ev0.record()
    for _ in range(reps):
        launch()
    ev1.record()
    ev1.synchronize()
    return cp.cuda.get_elapsed_time(ev0, ev1) / reps


def plan_calibrated(n, dt, src, out, reps=5, warmup=3):
    """Measure THIS GPU's stage costs on the caller's slab buffers, then DP.
    Costs at s>0 are s-independent in the measurements (s=8/9/10 agree to ~1%),
    so each B is probed at a single representative s."""
    dt = np.dtype(dt)
    nbits, isize = n - 1, dt.itemsize
    Nloc = 1 << (n - 1)
    coeffs = (0.5, 0.8660254037844386, -0.8660254037844386, 0.5)  # orthogonal
    src.fill(1.0)                       # avoid denormals/NaN in raw scratch
    tab = {"blk": {0: {}, 1: {}}, "blk2": {0: {}, 1: {}}, "co": {0: {}, 1: {}}}
    # B<4 is never optimal (it needs >=5 stages) yet those probes are the most
    # expensive ones in the table -- B=1 costs ~3.4x B=6. Skipping them is most
    # of the calibration cost. Below n = 5 the whole lattice is narrower than
    # _B_FLOOR bits, so the probes start at nbits (one stage covers it).
    for kind in ("blk", "blk2", "co"):
        for B in range(min(_B_FLOOR, nbits), 12):
            for strided in (0, 1):
                s = 0 if strided == 0 else max(5, (nbits - B) // 2)
                if not feasible(kind, s, B, nbits, isize):
                    continue
                inp = src if strided == 0 else out
                launch = make_launch(kind, s, B, n, Nloc, nbits, dt,
                                     inp, out, coeffs)
                tab[kind][strided][B] = _time(launch, reps, warmup)

    def cost(kind, s, B):
        if not feasible(kind, s, B, nbits, isize):
            return None
        return tab[kind][0 if s == 0 else 1].get(B)

    return _dp(nbits, cost)[1], tab


def _device_key():
    try:
        d = cp.cuda.Device()
        return cp.cuda.runtime.getDeviceProperties(d.id)["name"].decode()
    except Exception:
        return "unknown"


class Bfly2:
    """The (a,b,c,d) butterfly over all n-1 rest-bit stages with a fixed stage plan per
    (device, dtype, n). apply(src, out, a,b,c,d) leaves the result in `out` and allocates
    nothing (CUDA-graph safe).

    The plan is a DP over stage costs measured by calibrate() on the caller's scratch buffers
    (cached per process), unless one is passed in. Because every plan is bit-identical,
    calibrating never changes results -- only wall time.
    """

    def __init__(self, n, dt, plan=None, calibrate=True):
        self.n = int(n)
        self.dt = np.dtype(dt)
        self.Nloc = 1 << (n - 1)
        self.isize = self.dt.itemsize
        self.nbits = n - 1
        self.calib_table = None
        self._key = (_device_key(), self.dt.str, self.nbits)
        self.plan = list(plan) if plan is not None else _PLAN_CACHE.get(self._key)
        if self.plan is None and not calibrate:
            raise ValueError("Bfly2: no stage plan; pass plan= or calibrate on scratch buffers")
        self._want_calib = self.plan is None

    def calibrate(self, src, out, reps=5, warmup=3):
        """Time stages on the caller's slab-sized scratch buffers (`src` is
        overwritten). No-op once a plan for this (device, dtype, n) is known."""
        if not self._want_calib:
            return self.plan
        self.plan, self.calib_table = plan_calibrated(
            self.n, self.dt, src, out, reps=reps, warmup=warmup)
        _PLAN_CACHE[self._key] = self.plan
        self._want_calib = False
        return self.plan

    def apply(self, src, out, a, b, c, d):
        n, Nloc, nbits, dt = self.n, self.Nloc, self.nbits, self.dt
        coeffs = (a, b, c, d)
        cur = src
        for (kind, s, B) in self.plan:
            make_launch(kind, s, B, n, Nloc, nbits, dt, cur, out, coeffs)()
            cur = out                                   # later stages in place
        return out
