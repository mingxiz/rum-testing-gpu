"""How does one Bellman-Ford pricing sweep scale with n on pairwise menus, with
the CURRENT kernel and memory layout (V float32 2^n, pred uint8 2^n, all layer
masks int32 = 4 B * 2^n)?  Prints per-sweep time and device memory per n.
    python validation/feasibility/bf_sweep_scaling.py 20 22 24 26 28
"""
import os, sys, time
import numpy as np, cupy as cp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "data"))
from gen_instances import gen_pairwise
from solver.flow_cg.cg_core import CGData
from solver.flow_cg.bf_pricing import bf as bf_kernel

for n in [int(a) for a in sys.argv[1:]]:
    cd, mp = gen_pairwise(n)
    pool = cp.get_default_memory_pool(); pool.free_all_blocks()
    t0 = time.perf_counter(); model = CGData(cd, n); cp.cuda.Device().synchronize(); t_layers = time.perf_counter() - t0
    Nnode = 1 << n
    V = cp.zeros(Nnode, dtype=cp.float32); pred = cp.zeros(Nnode, dtype=cp.uint8)
    masks = cp.asarray(model.choice_set, dtype=cp.uint32); offs = cp.asarray(model.offsets, dtype=cp.uint64)
    rng = np.random.RandomState(0); scores = cp.asarray(rng.randn(mp).astype(np.float32))
    def sweep():
        V.fill(0); pred.fill(0)
        for c in range(1, n + 1):
            st = model.layers[c]; ns = int(st.size); grid = (ns + 255) // 256
            bf_kernel((grid,), (256,), (V, pred, masks, scores, offs, st, ns))
        cp.cuda.Device().synchronize()
    sweep()
    reps = 3 if n >= 26 else 10
    t0 = time.perf_counter()
    for _ in range(reps): sweep()
    t = (time.perf_counter() - t0) / reps
    used = pool.used_bytes() / 2**30
    print("n=%2d  mp=%5d  sweep=%8.3f s  (2^n*n=%.2e)  device used=%.2f GB  (V+pred %.2f GB, layers %.2f GB)  layer precompute %.1fs"
          % (n, mp, t, Nnode * n, used, 5 * Nnode / 2**30, 4 * Nnode / 2**30, t_layers), flush=True)
    del V, pred, model; pool.free_all_blocks()
