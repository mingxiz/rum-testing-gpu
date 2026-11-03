"""The tiny instance the GPU warm-ups (bm_admm.warmup, flow_cg.warmup) compile kernels on."""
import numpy as np


def warmup_instance(n=10, seed=0):
    """All n(n-1)/2 binary menus, both alternatives observed, random choice probabilities.
    Rows (menu mask, chosen alternative, p), as float64."""
    rng = np.random.RandomState(seed)
    om, ox, ph = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            g = rng.rand(2); g /= g.sum()
            om += [(1 << i) | (1 << j)] * 2; ox += [i, j]; ph += [g[0], g[1]]
    return np.column_stack([np.array(om, float), np.array(ox, float), np.array(ph, float)])
