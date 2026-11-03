"""Mask-indexed Bellman-Ford pricing kernel for CGBFQP-GPU: thread D computes
    V[D] = max over x in D of  V[D minus x] + w(x, D),   w(x, D) = sum of the scores of the observed
                                                       menus D' with x in D', D' subset of D,
for every subset D of one cardinality layer; the sweep (one launch per layer) is in solver.py."""
import cupy as cp

_BF_UPDATE_SRC = r'''
extern "C" __global__ void bf_update(
    float* V,
    unsigned char* pred,
    const unsigned int*       __restrict__ menu_masks,
    const float*              __restrict__ menu_scores,
    const unsigned long long* __restrict__ offsets,
    const int*                __restrict__ states,
    const int n_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;            // bounds guard

    unsigned int D = (unsigned int) states[idx];   // this thread's subset mask

    float best = 0.0f;
    bool  seen = false;                     // avoids -INFINITY (NVRTC has no <cfloat>)

    unsigned int S = D;                     // bits still to visit
    while (S) {
        unsigned int bit  = S & (-S);       // isolate lowest set bit
        int          x    = __ffs(bit) - 1; // its 0-based index
        unsigned int prev = D ^ bit;        // D with alternative x removed

        // w(x, D) = sum_{ x in D',  D' subset of D } s(x, D')
        float w = 0.0f;
        unsigned long long lo = offsets[x];
        unsigned long long hi = offsets[x + 1];
        for (unsigned long long j = lo; j < hi; ++j) {
            unsigned int Dp = menu_masks[j];          // a stored menu containing x
            if ((Dp & D) == Dp) w += menu_scores[j];  // keep only D' subset of D; arc weights are calculated on the fly
        }

        float cand = V[prev] + w;
        if (!seen || cand > best) { best = cand; seen = true; pred[D] = (unsigned char) x; }

        S ^= bit;                           // clear lowest set bit
    }

    V[D] = best;                            // in-place update of this layer's node
}
'''

bf = cp.RawModule(code=_BF_UPDATE_SRC).get_function("bf_update")
