"""Small CUDA helpers shared by the BM-ADMM kernels: launch geometry, the
dtype-parametric RawKernel compiler and the observation-scatter kernel that
places (menu, alternative, value) triples into the item-major Block-Marschak
slab (position = alt * 2^(n-1) + compress_bit(menu minus alt, alt))."""
import numpy as np
import cupy as cp

_BLK = 256
_F32 = np.dtype(np.float32)
_F64 = np.dtype(np.float64)


def _grid(nt):
    return ((int(nt) + _BLK - 1) // _BLK,)


def _compile_dt(src, name):
    """Compile a RawKernel in float32 and float64 (uses the `real` typedef)."""
    return {_F32: cp.RawKernel('typedef float real;\n' + src, name),
            _F64: cp.RawKernel('typedef double real;\n' + src, name)}


_SCATTER_DT = _compile_dt(r'''
extern "C" __global__ void scatter_obs(const int* ox0, const unsigned int* om,
                                       const real* vals, real* slab,
                                       int M, int Nloc){
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if (j >= M) return;
  int x = ox0[j];
  unsigned int mask = om[j] & ~(1u << x);
  unsigned int low  = mask & ((1u << x) - 1u);
  unsigned int high = mask >> (x + 1);
  unsigned int rest = low | (high << x);
  slab[(long long)x*Nloc + rest] = vals[j];
}''', 'scatter_obs')
