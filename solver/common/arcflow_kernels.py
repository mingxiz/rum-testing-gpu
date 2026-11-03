"""GPU primitives of the arc-flow formulation, float32, used by Flow-ADMM (solver/flow_admm) and by
the feasibility push behind every ADMM upper bound (common/transforms.py):

    fwht                       unnormalized Walsh-Hadamard transform, natural (Sylvester) order
    a_mul, at_mul              node-arc incidence A (divergence of a flow) and its transpose
    zeta_superset_blk          superset / subset zeta transforms along the rest axis of the slab,
    zeta_subset_blk            several bit-stages per shared-memory pass
    H0inv_woodbury             H0^{-1} v for H0 = beta3 (I + gamma A'A), by Woodbury (A A' is diagonal
                               in the Walsh-Hadamard basis)
    build_layers               nodes of the subset lattice sorted by cardinality
    make_r_feasible_gpu        clip a flow to >= 0 and push unit mass forward through the layers:
                               an exact unit flow, i.e. a feasible RUM point

The RUM class certifies with float64 twins (the races keep the float32 kernels above):

    make_r_feasible_gpu64      the push in float64, so the flow is conserved to float64 rounding
    longest_path_gpu64         the best ranking's total of a slab of arc values, layer by layer: the
                               Frank-Wolfe lower bound's sweep (certificate._longest_path on the GPU)
    build_layers_gpu           build_layers on the GPU, the same arrays

Each primitive is one launch, or one launch per bit-stage or lattice layer; no reshapes, no atomics.
"""
import os
import sys
import numpy as np

os.environ.setdefault('CONDA_PREFIX', sys.prefix)
import cupy as cp

_BLK = 256


def _grid(nthreads):
    return ((int(nthreads) + _BLK - 1) // _BLK,)


# ---------------------------------------------------------------------------
#  In-place FWHT: one kernel per stage, N/2 threads, no copy
#  Natural (Sylvester) ordering: H[i,j] = (-1)^popcount(i&j).
# ---------------------------------------------------------------------------
_FWHT = cp.RawKernel(r'''
extern "C" __global__ void fwht_stage(float* x, long long N, long long h){
  long long t = blockIdx.x*(long long)blockDim.x + threadIdx.x;
  if(t >= (N>>1)) return;
  long long group = t / h, pos = t % h;
  long long i = group*2*h + pos, j = i + h;
  float a = x[i], b = x[j];
  x[i] = a + b; x[j] = a - b;
}''', 'fwht_stage')


def fwht(x):
    """Unnormalized natural-order FWHT of 1-D float32 x (returns new array)."""
    y = x.astype(cp.float32, copy=True)
    N = y.size
    g = _grid(N // 2)
    h = 1
    while h < N:
        _FWHT(g, (_BLK,), (y, np.int64(N), np.int64(h)))
        h <<= 1
    return y


# ---------------------------------------------------------------------------
#  A / At incidence operators: node- / arc-parallel, single launch, no atomics
# ---------------------------------------------------------------------------
_AMUL = cp.RawKernel(r'''
extern "C" __global__ void amul(const float* r, float* div, int n, long long Nloc){
  long long D = blockIdx.x*(long long)blockDim.x + threadIdx.x;
  long long Nnode = 1LL<<n; if(D>=Nnode) return;
  float v = 0.f;
  for(int x=0; x<n; ++x){
    long long bit = 1LL<<x;
    if(D & bit){                       // x in D: +r[arc(x, compress(D^bit,x))]
      long long S = D ^ bit;
      long long rest = (S & (bit-1)) | ((S>>(x+1))<<x);
      v += r[(long long)x*Nloc + rest];
    } else {                           // x not in D: -r[arc(x, compress(D,x))]
      long long rest = (D & (bit-1)) | ((D>>(x+1))<<x);
      v -= r[(long long)x*Nloc + rest];
    }
  }
  div[D] = v;
}''', 'amul')

_ATMUL = cp.RawKernel(r'''
extern "C" __global__ void atmul(const float* lam, float* z, int n, long long Nloc){
  long long a = blockIdx.x*(long long)blockDim.x + threadIdx.x;
  long long m2 = (long long)n*Nloc; if(a>=m2) return;
  long long x = a/Nloc, rest = a - x*Nloc;
  long long base = (rest & ((1LL<<x)-1)) | ((rest>>x)<<(x+1));
  z[a] = lam[base | (1LL<<x)] - lam[base];
}''', 'atmul')


def a_mul(r, n, Nnode, Nloc):
    div = cp.empty(Nnode, dtype=cp.float32)
    _AMUL(_grid(Nnode), (_BLK,), (r, div, np.int32(n), np.int64(Nloc)))
    return div


def at_mul(lam, n, Nloc):
    m2 = n * Nloc
    z = cp.empty(m2, dtype=cp.float32)
    _ATMUL(_grid(m2), (_BLK,), (lam, z, np.int32(n), np.int64(Nloc)))
    return z


# ---------------------------------------------------------------------------
#  Shared-memory BLOCKED zeta: B bit-stages per global pass (one block = one
#  2^B tile). Cuts (n-1) global passes down to ceil((n-1)/B) (e.g. 19 -> 2).
#  Bits of the SOS transform commute, so processing a contiguous window
#  [s, s+B) inside shared memory and tiling the rest is exact.
#    superset: sh[l] += sh[l | (1<<j)]   (local bit j == 0 accumulates)
#    subset:   sh[l] += sh[l ^ (1<<j)]   (local bit j == 1 accumulates)
#  Tile<->global map: rest = low(outer) | (l<<s) | (high(outer)<<(s+B)).
# ---------------------------------------------------------------------------
_ZSUP_BLK = cp.RawKernel(r'''
extern "C" __global__ void zsup_blk(const float* IN, float* OUT, long long Nloc, int s, int B){
  extern __shared__ float sh[];
  long long tilesPerItem = Nloc >> B;
  long long x = blockIdx.x / tilesPerItem;
  long long outer = blockIdx.x % tilesPerItem;
  int l = threadIdx.x;
  long long low  = outer & ((1LL<<s)-1);
  long long high = (outer >> s) << (s+B);
  long long rest = low | ((long long)l << s) | high;
  long long gidx = x*Nloc + rest;
  sh[l] = IN[gidx];
  __syncthreads();
  for(int j=0;j<B;j++){
    if(((l>>j)&1)==0) sh[l] += sh[l | (1<<j)];
    __syncthreads();
  }
  OUT[gidx] = sh[l];
}''', 'zsup_blk')

_ZSUB_BLK = cp.RawKernel(r'''
extern "C" __global__ void zsub_blk(const float* IN, float* OUT, long long Nloc, int s, int B){
  extern __shared__ float sh[];
  long long tilesPerItem = Nloc >> B;
  long long x = blockIdx.x / tilesPerItem;
  long long outer = blockIdx.x % tilesPerItem;
  int l = threadIdx.x;
  long long low  = outer & ((1LL<<s)-1);
  long long high = (outer >> s) << (s+B);
  long long rest = low | ((long long)l << s) | high;
  long long gidx = x*Nloc + rest;
  sh[l] = IN[gidx];
  __syncthreads();
  for(int j=0;j<B;j++){
    if(((l>>j)&1)==1) sh[l] += sh[l ^ (1<<j)];
    __syncthreads();
  }
  OUT[gidx] = sh[l];
}''', 'zsub_blk')


# Coalesced pass for s>=5: batch W=32 contiguous low-value columns per block so
# each warp's 32 lanes load 32 consecutive global addresses (fully coalesced),
# while the strided window-bit butterfly runs in 64KB shared memory.
_ZSUP_CO = cp.RawKernel(r'''
extern "C" __global__ void zsup_co(const float* IN, float* OUT, long long Nloc,
                                   int s, int B, long long nlowblk, long long nhigh){
  extern __shared__ float sh[];                 // (1<<B) * 32 floats
  long long blk = blockIdx.x;
  long long lowblk = blk % nlowblk; blk /= nlowblk;
  long long high_i = blk % nhigh;   blk /= nhigh;
  long long base = blk * Nloc;                  // blk == x
  long long low0 = lowblk * 32;
  long long high = high_i << (s + B);
  int w = threadIdx.x & 31, g = threadIdx.x >> 5, npp = (1 << B) >> 5;
  for(int li=0; li<npp; ++li){ int lp = g + li*32;
    long long rest = (low0 + w) | ((long long)lp << s) | high;
    sh[lp*32 + w] = IN[base + rest]; }
  __syncthreads();
  for(int j=0;j<B;++j){
    for(int li=0; li<npp; ++li){ int lp = g + li*32;
      if(((lp>>j)&1)==0) sh[lp*32+w] += sh[(lp|(1<<j))*32 + w]; }
    __syncthreads(); }
  for(int li=0; li<npp; ++li){ int lp = g + li*32;
    long long rest = (low0 + w) | ((long long)lp << s) | high;
    OUT[base + rest] = sh[lp*32 + w]; }
}''', 'zsup_co')

_ZSUB_CO = cp.RawKernel(r'''
extern "C" __global__ void zsub_co(const float* IN, float* OUT, long long Nloc,
                                   int s, int B, long long nlowblk, long long nhigh){
  extern __shared__ float sh[];
  long long blk = blockIdx.x;
  long long lowblk = blk % nlowblk; blk /= nlowblk;
  long long high_i = blk % nhigh;   blk /= nhigh;
  long long base = blk * Nloc;
  long long low0 = lowblk * 32;
  long long high = high_i << (s + B);
  int w = threadIdx.x & 31, g = threadIdx.x >> 5, npp = (1 << B) >> 5;
  for(int li=0; li<npp; ++li){ int lp = g + li*32;
    long long rest = (low0 + w) | ((long long)lp << s) | high;
    sh[lp*32 + w] = IN[base + rest]; }
  __syncthreads();
  for(int j=0;j<B;++j){
    for(int li=0; li<npp; ++li){ int lp = g + li*32;
      if(((lp>>j)&1)==1) sh[lp*32+w] += sh[(lp^(1<<j))*32 + w]; }
    __syncthreads(); }
  for(int li=0; li<npp; ++li){ int lp = g + li*32;
    long long rest = (low0 + w) | ((long long)lp << s) | high;
    OUT[base + rest] = sh[lp*32 + w]; }
}''', 'zsub_co')

for _k in (_ZSUP_CO, _ZSUB_CO):
    try:
        _k.max_dynamic_shared_size_bytes = 1 << 16     # opt-in 64KB (B up to 9)
    except Exception:
        pass


def _zeta_blocked(r, n, Nloc, kern_blk, kern_co, Bmax=10):
    # No pre-copy: first pass reads `r` -> fresh R; later passes in-place on R.
    m2 = n * Nloc
    R = cp.empty(m2, dtype=cp.float32)
    src = r if r.dtype == cp.float32 else r.astype(cp.float32)
    nbits = n - 1
    s = 0
    first = True
    while s < nbits:
        rem = nbits - s
        if first:                                      # pass 1: contiguous (s=0)
            B, use_co, first = min(Bmax, rem), False, False
        else:
            # coalesced pass: B in [5,9], never leaving a leftover < 5
            B = rem if rem <= 9 else (rem - 5 if rem - 9 < 5 else 9)
            use_co = (B >= 5 and s >= 5)
        if use_co:                                     # W=32 coalesced (<=64KB)
            nlowblk = (1 << s) >> 5
            nhigh = 1 << (nbits - s - B)
            grid = (int(n * nhigh * nlowblk),)
            kern_co(grid, (1024,),
                    (src, R, np.int64(Nloc), np.int32(s), np.int32(B),
                     np.int64(nlowblk), np.int64(nhigh)),
                    shared_mem=(1 << B) * 32 * 4)
        else:                                          # contiguous blocked pass
            tiles = (Nloc >> B)
            grid = (int(n * tiles),)
            kern_blk(grid, (1 << B,),
                     (src, R, np.int64(Nloc), np.int32(s), np.int32(B)),
                     shared_mem=(1 << B) * 4)
        src = R
        s += B
    return R


def zeta_superset_blk(r, n, Nloc):
    return _zeta_blocked(r, n, Nloc, _ZSUP_BLK, _ZSUP_CO)


def zeta_subset_blk(z, n, Nloc):
    return _zeta_blocked(z, n, Nloc, _ZSUB_BLK, _ZSUB_CO)


# ---------------------------------------------------------------------------
#  Woodbury H0^{-1} v = (1/beta3)(v - gamma A'(I+gamma AA')^{-1} A v)
#  using the optimized fwht / a_mul / at_mul (single-vector hot path).
# ---------------------------------------------------------------------------
# fused elementwise: (v - gamma*Atw)/beta3 in one pass
_H0FIN = cp.ElementwiseKernel(
    'float32 v, float32 Atw, float32 g, float32 ib3', 'float32 out',
    'out = (v - g*Atw) * ib3', 'h0fin')



def H0inv_woodbury(n, v, beta3, gamma, g_denom, Nnode, Nloc):
    Av = a_mul(v, n, Nnode, Nloc)
    fAv = fwht(Av)
    w = fwht(fAv / g_denom) / Nnode
    Atw = at_mul(w, n, Nloc)
    return _H0FIN(v, Atw, cp.float32(gamma), cp.float32(1.0 / beta3))


# ---------------------------------------------------------------------------
#  make_r_feasible: forward unit-mass propagation, layer-by-layer gather kernel
# ---------------------------------------------------------------------------
_WMAX = cp.ElementwiseKernel('float32 r, float32 eps', 'float32 w',
                             'w = (r > 0.f ? r : 0.f) + eps', 'wmax')

_OUTWT = cp.RawKernel(r'''
extern "C" __global__ void outwt(const float* w, float* out, int n, long long Nloc){
  long long D = blockIdx.x*(long long)blockDim.x + threadIdx.x;
  long long Nnode = 1LL<<n; if(D>=Nnode) return;
  float s = 0.f;
  for(int x=0; x<n; ++x){ long long bit=1LL<<x;
    if(!(D & bit)){ long long rest=(D&(bit-1))|((D>>(x+1))<<x); s += w[(long long)x*Nloc+rest]; } }
  out[D] = s;
}''', 'outwt')

_MASSG = cp.RawKernel(r'''
extern "C" __global__ void massg(const float* w, const float* outw, float* mass,
                                 const long long* nodes, long long ns, int n, long long Nloc){
  long long i = blockIdx.x*(long long)blockDim.x + threadIdx.x; if(i>=ns) return;
  long long C = nodes[i]; float s = 0.f;
  for(int x=0; x<n; ++x){ long long bit=1LL<<x;
    if(C & bit){ long long P=C^bit; long long rest=(P&(bit-1))|((P>>(x+1))<<x);
      float ow=outw[P]; if(ow>0.f) s += mass[P]*w[(long long)x*Nloc+rest]/ow; } }
  mass[C] = s;
}''', 'massg')

_EXTRACT = cp.RawKernel(r'''
extern "C" __global__ void extract(const float* w, const float* outw, const float* mass,
                                   float* rf, int n, long long Nloc){
  long long a = blockIdx.x*(long long)blockDim.x + threadIdx.x;
  long long m2=(long long)n*Nloc; if(a>=m2) return;
  long long x=a/Nloc, rest=a-x*Nloc;
  long long T=(rest&((1LL<<x)-1))|((rest>>x)<<(x+1));
  float ow=outw[T]; rf[a] = (ow>0.f) ? mass[T]*w[a]/ow : 0.f;
}''', 'extract')


def build_layers(n):
    """Node ids sorted by popcount, and the start of each cardinality layer in that order."""
    Nnode = 1 << n
    v = np.arange(Nnode, dtype=np.uint32)
    pc = np.zeros(Nnode, dtype=np.int32)
    for b in range(n):
        pc += ((v >> b) & 1).astype(np.int32)
    order = np.argsort(pc, kind='stable')
    bounds = np.searchsorted(pc[order], np.arange(n + 2))
    return cp.asarray(order.astype(np.int64)), bounds


def make_r_feasible_gpu(n, g_r, layers, bounds, eps=1e-12):
    """GPU forward-propagation feasibility rounding. g_r float32 (m2,) on device."""
    Nnode = 1 << n
    Nloc = 1 << (n - 1)
    m2 = n * Nloc
    w = cp.empty(m2, dtype=cp.float32)
    _WMAX(g_r, np.float32(eps), w)
    out = cp.empty(Nnode, dtype=cp.float32)
    _OUTWT(_grid(Nnode), (_BLK,), (w, out, np.int32(n), np.int64(Nloc)))
    mass = cp.zeros(Nnode, dtype=cp.float32)
    mass[0:1] = 1.0
    for c in range(1, n + 1):
        lo, hi = int(bounds[c]), int(bounds[c + 1]); ns = hi - lo
        if ns <= 0:
            continue
        _MASSG(_grid(ns), (_BLK,), (w, out, mass, layers[lo:hi], np.int64(ns),
                                    np.int32(n), np.int64(Nloc)))
    rf = cp.empty(m2, dtype=cp.float32)
    _EXTRACT(_grid(m2), (_BLK,), (w, out, mass, rf, np.int32(n), np.int64(Nloc)))
    return rf


# ---------------------------------------------------------------------------
#  float64 twins for the RUM class's certificates: the push, and the longest path of the FW bound
# ---------------------------------------------------------------------------
_WMAX64 = cp.ElementwiseKernel('float64 r, float64 eps', 'float64 w',
                               'w = (r > 0.0 ? r : 0.0) + eps', 'wmax64')

_OUTWT64 = cp.RawKernel(r'''
extern "C" __global__ void outwt64(const double* w, double* out, int n, long long Nloc){
  long long D = blockIdx.x*(long long)blockDim.x + threadIdx.x;
  long long Nnode = 1LL<<n; if(D>=Nnode) return;
  double s = 0.0;
  for(int x=0; x<n; ++x){ long long bit=1LL<<x;
    if(!(D & bit)){ long long rest=(D&(bit-1))|((D>>(x+1))<<x); s += w[(long long)x*Nloc+rest]; } }
  out[D] = s;
}''', 'outwt64')

_MASSG64 = cp.RawKernel(r'''
extern "C" __global__ void massg64(const double* w, const double* outw, double* mass,
                                   const long long* nodes, long long ns, int n, long long Nloc){
  long long i = blockIdx.x*(long long)blockDim.x + threadIdx.x; if(i>=ns) return;
  long long C = nodes[i]; double s = 0.0;
  for(int x=0; x<n; ++x){ long long bit=1LL<<x;
    if(C & bit){ long long P=C^bit; long long rest=(P&(bit-1))|((P>>(x+1))<<x);
      double ow=outw[P]; if(ow>0.0) s += mass[P]*w[(long long)x*Nloc+rest]/ow; } }
  mass[C] = s;
}''', 'massg64')

_EXTRACT64 = cp.RawKernel(r'''
extern "C" __global__ void extract64(const double* w, const double* outw, const double* mass,
                                     double* rf, int n, long long Nloc){
  long long a = blockIdx.x*(long long)blockDim.x + threadIdx.x;
  long long m2=(long long)n*Nloc; if(a>=m2) return;
  long long x=a/Nloc, rest=a-x*Nloc;
  long long T=(rest&((1LL<<x)-1))|((rest>>x)<<(x+1));
  double ow=outw[T]; rf[a] = (ow>0.0) ? mass[T]*w[a]/ow : 0.0;
}''', 'extract64')

_LPATH64 = cp.RawKernel(r'''
extern "C" __global__ void lpath64(const double* R, double* dp, const long long* nodes, long long ns,
                                   int n, long long Nloc){
  long long i = blockIdx.x*(long long)blockDim.x + threadIdx.x; if(i>=ns) return;
  long long C = nodes[i]; double best = -1e300;
  for(int x=0; x<n; ++x){ long long bit=1LL<<x;
    if(C & bit){ long long P=C^bit; long long rest=(P&(bit-1))|((P>>(x+1))<<x);
      double v = dp[P] + R[(long long)x*Nloc+rest]; if(v>best) best=v; } }
  dp[C] = best;
}''', 'lpath64')

_POPC = cp.ElementwiseKernel('uint32 v', 'int32 c', 'c = __popc(v)', 'popc32')


def build_layers_gpu(n):
    """build_layers on the GPU: the same two arrays (ids ascending within a layer, as the stable sort
    gives), without numpy's sort of 2^n keys (2.4 s at n = 25)."""
    pc = _POPC(cp.arange(1 << n, dtype=cp.uint32))
    order = cp.empty(1 << n, dtype=cp.int64)
    bounds = np.zeros(n + 2, dtype=np.intp)
    for c in range(n + 1):
        ids = cp.flatnonzero(pc == c)
        bounds[c + 1] = bounds[c] + ids.size
        order[bounds[c]:bounds[c + 1]] = ids
    return order, bounds


def make_r_feasible_gpu64(n, g_r, layers, bounds, eps=1e-12):
    """make_r_feasible_gpu in float64: g_r (m2,) on device, any float dtype."""
    Nnode = 1 << n
    Nloc = 1 << (n - 1)
    m2 = n * Nloc
    w = cp.empty(m2, dtype=cp.float64)
    _WMAX64(g_r.astype(cp.float64, copy=False), np.float64(eps), w)
    out = cp.empty(Nnode, dtype=cp.float64)
    _OUTWT64(_grid(Nnode), (_BLK,), (w, out, np.int32(n), np.int64(Nloc)))
    mass = cp.zeros(Nnode, dtype=cp.float64)
    mass[0:1] = 1.0
    for c in range(1, n + 1):
        lo, hi = int(bounds[c]), int(bounds[c + 1]); ns = hi - lo
        if ns <= 0:
            continue
        _MASSG64(_grid(ns), (_BLK,), (w, out, mass, layers[lo:hi], np.int64(ns),
                                      np.int32(n), np.int64(Nloc)))
    rf = cp.empty(m2, dtype=cp.float64)
    _EXTRACT64(_grid(m2), (_BLK,), (w, out, mass, rf, np.int32(n), np.int64(Nloc)))
    return rf


def longest_path_gpu64(n, R, layers, bounds):
    """Value of the best path from the empty set to the full set, where the arc S -> S + {x} is worth
    R[x*2^(n-1) + compress(S, x)] (R float64 (m2,) on device), one launch per cardinality layer."""
    Nnode = 1 << n
    dp = cp.empty(Nnode, dtype=cp.float64)
    dp[0:1] = 0.0
    for c in range(1, n + 1):
        lo, hi = int(bounds[c]), int(bounds[c + 1]); ns = hi - lo
        if ns <= 0:
            continue
        _LPATH64(_grid(ns), (_BLK,), (R, dp, layers[lo:hi], np.int64(ns), np.int32(n), np.int64(1 << (n - 1))))
    return float(dp[Nnode - 1].get())
