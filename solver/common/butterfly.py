r"""Blocked + coalesced, dtype-parametric (float32/float64) generalized 2x2
butterfly for the K-formulation. Generalizes a blocked/coalesced
zeta (_ZSUP_BLK / _ZSUP_CO) to an arbitrary 2x2 kernel (a,b,c,d), so ONE op
serves the Mobius transforms (A_ineq/A_ineq') AND the eigen-solve butterflies.

Why: the naive butterfly does n-1 SEPARATE global passes; at high bit-stages the
stride (2^s) is uncoalesced and thrashes cache -> per-stage cost collapses at
n=22 (one butterfly 0.5ms@n20 -> 8.85ms@n22). Blocking does ~10 bit-stages per
shared-memory tile (n-1 passes -> ~2) and coalesces the high-bit pass.

float64 is needed at n>=18: the Block-Marschak Mobius sums 2^(n-1) alternating
terms (lambda-range 2.618^(n-1) crosses float32's ~1e7 at n~17).

Transform is per-item-column over the rest-axis (Nloc=2^(n-1) per item, n items);
flat slab index = item*Nloc + rest. Within a 2x2 stage at bit `j`: pair (l, l|2^j),
new_low = a*X + b*Y, new_high = c*X + d*Y (the low thread writes both, race-free).
"""
import numpy as np
import cupy as cp

# --- blocked kernel: B bit-stages of the window [s, s+B) in shared memory ------
_BLK_SRC = r'''
extern "C" __global__ void bfly_blk(const real* __restrict__ IN, real* __restrict__ OUT,
                                    long long Nloc, int s, int B,
                                    real a, real b, real c, real d){
  extern __shared__ __align__(16) unsigned char _smem[];
  real* sh = reinterpret_cast<real*>(_smem);
  long long tilesPerItem = Nloc >> B;
  long long x     = blockIdx.x / tilesPerItem;        // item
  long long outer = blockIdx.x % tilesPerItem;
  int l = threadIdx.x;                                 // 0 .. 2^B-1
  long long low  = outer & ((1LL<<s)-1);
  long long high = (outer >> s) << (s+B);
  long long rest = low | ((long long)l << s) | high;
  long long gidx = x*Nloc + rest;
  sh[l] = IN[gidx];
  __syncthreads();
  for(int j=0;j<B;j++){
    if(((l>>j)&1)==0){
      int p = l | (1<<j);
      real X = sh[l], Y = sh[p];
      sh[l] = a*X + b*Y;
      sh[p] = c*X + d*Y;
    }
    __syncthreads();
  }
  OUT[gidx] = sh[l];
}'''

# --- coalesced kernel for high-bit passes (s>=5): W=32 contiguous low-values ----
_CO_SRC = r'''
extern "C" __global__ void bfly_co(const real* __restrict__ IN, real* __restrict__ OUT,
                                   long long Nloc, int s, int B,
                                   long long nlowblk, long long nhigh,
                                   real a, real b, real c, real d){
  extern __shared__ __align__(16) unsigned char _smem[];
  real* sh = reinterpret_cast<real*>(_smem);          // (1<<B) * 32 reals
  long long blk = blockIdx.x;
  long long lowblk = blk % nlowblk; blk /= nlowblk;
  long long high_i = blk % nhigh;   blk /= nhigh;
  long long base = blk * Nloc;                         // blk == item
  long long low0 = lowblk * 32;
  long long high = high_i << (s + B);
  int w = threadIdx.x & 31, g = threadIdx.x >> 5, npp = (1 << B) >> 5;
  for(int li=0; li<npp; ++li){ int lp = g + li*32;
    long long rest = (low0 + w) | ((long long)lp << s) | high;
    sh[lp*32 + w] = IN[base + rest]; }
  __syncthreads();
  for(int j=0;j<B;++j){
    for(int li=0; li<npp; ++li){ int lp = g + li*32;
      if(((lp>>j)&1)==0){
        int pp = lp | (1<<j);
        real X = sh[lp*32+w], Y = sh[pp*32+w];
        sh[lp*32+w] = a*X + b*Y;
        sh[pp*32+w] = c*X + d*Y;
      } }
    __syncthreads(); }
  for(int li=0; li<npp; ++li){ int lp = g + li*32;
    long long rest = (low0 + w) | ((long long)lp << s) | high;
    OUT[base + rest] = sh[lp*32 + w]; }
}'''


def _compile(src, name):
    k32 = cp.RawKernel('typedef float real;\n' + src, name)
    k64 = cp.RawKernel('typedef double real;\n' + src, name)
    for k in (k32, k64):
        try:
            k.max_dynamic_shared_size_bytes = 1 << 16     # opt-in 64KB
        except Exception:
            pass
    return {np.dtype(np.float32): k32, np.dtype(np.float64): k64}


_BLK_K = _compile(_BLK_SRC, 'bfly_blk')
_CO_K = _compile(_CO_SRC, 'bfly_co')


def butterfly_fast(r, n, Nloc, a, b, c, d):
    """Apply the (a,b,c,d) 2x2 butterfly over all n-1 rest-bit stages.
    Returns a NEW array (same dtype as r). float32 or float64 by r.dtype."""
    dt = r.dtype
    blk_k = _BLK_K[dt]; co_k = _CO_K[dt]
    isize = dt.itemsize
    m2 = n * Nloc
    nbits = n - 1
    if nbits == 0:
        return r.copy()
    Bmax = 10 if isize == 4 else 9                       # 2^B reals <= 8KB shared
    Bco = 9 if isize == 4 else 8                         # 2^B*32 reals <= 64KB
    cast = np.float32 if isize == 4 else np.float64
    af, bf, cf, df = cast(a), cast(b), cast(c), cast(d)
    R = cp.empty(m2, dtype=dt)
    src = r
    s = 0
    first = True
    while s < nbits:
        rem = nbits - s
        if first:
            B, use_co, first = min(Bmax, rem), False, False
        else:
            B = rem if rem <= Bco else (rem - 5 if rem - Bco < 5 else Bco)
            use_co = (B >= 5 and s >= 5)
        if use_co:
            nlowblk = (1 << s) >> 5
            nhigh = 1 << (nbits - s - B)
            grid = (int(n * nhigh * nlowblk),)
            co_k(grid, (1024,),
                 (src, R, np.int64(Nloc), np.int32(s), np.int32(B),
                  np.int64(nlowblk), np.int64(nhigh), af, bf, cf, df),
                 shared_mem=(1 << B) * 32 * isize)
        else:
            tiles = Nloc >> B
            grid = (int(n * tiles),)
            blk_k(grid, (1 << B,),
                  (src, R, np.int64(Nloc), np.int32(s), np.int32(B), af, bf, cf, df),
                  shared_mem=(1 << B) * isize)
        src = R
        s += B
    return R
