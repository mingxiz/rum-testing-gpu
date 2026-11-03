r"""Slab transforms and the feasibility push that turns any iterate into an upper bound.

A slab holds one value per (alternative i, menu S containing i), at position
i*2^(n-1) + compress_bit(S minus i, i). The push maps an iterate x to an exactly feasible RUM point:

    q = superset_Mobius(x)          the Block-Marschak measure (= the arc flow r, same layout)
    q_feas = make_r_feasible(q)     clip negatives, then push unit mass forward through the subset
                                    lattice in proportion -> an exact unit flow
    x_ub = superset_zeta(q_feas)

q_feas >= 0 and A q_feas = b (flow conservation) put x_ub inside the RUM polytope (menu sums 1,
Block-Marschak polynomials >= 0), so phi_ub = ||x_ub[obs] - phat||_2 is a valid upper bound.

push_to_ub runs make_r_feasible in float32, as the paper's races do. The RUM class certifies with
push_to_ub64 (the same push in float64, so q_feas is conserved to float64 rounding) and with
fw_lb_rows_gpu, the Frank-Wolfe lower bound of common/certificate.py computed on the GPU. Both take
the lattice layers from layers_gpu, which builds the same arrays as layers_for on the GPU. Where
the bound's float64 slab does not fit, fw_lb_stream64 computes the same bound two lattice layers at
a time (flow_cg's final lower bound from n = 27 on 32 GB).
"""
import numpy as np
import cupy as cp

from .butterfly import butterfly_fast
from .arcflow_kernels import (build_layers, build_layers_gpu, make_r_feasible_gpu, make_r_feasible_gpu64,
                              longest_path_gpu64, a_mul)

_LAYERS = {}
_LAYERS_GPU = {}


def layers_for(n):
    """Cardinality layers + bounds of the subset lattice, built once per n."""
    if n not in _LAYERS:
        _LAYERS[n] = build_layers(n)
    return _LAYERS[n]


def layers_gpu(n):
    """layers_for with the layers built on the GPU (the same arrays); the RUM class's float64 bounds
    use it, the races keep layers_for."""
    if n not in _LAYERS_GPU:
        _LAYERS_GPU[n] = build_layers_gpu(n)
    return _LAYERS_GPU[n]


def mobius(x, n):
    """q = superset_Mobius(x) (per item slab, contiguous rest-axis halves)."""
    Nloc = 1 << (n - 1)
    return butterfly_fast(cp.asarray(x, dtype=cp.float64), n, Nloc, 1.0, -1.0, 0.0, 1.0)


def zeta(q, n):
    """x = superset_zeta(q) = Mobius^{-1}(q) (per-stage [[1,1],[0,1]])."""
    Nloc = 1 << (n - 1)
    return butterfly_fast(cp.asarray(q, dtype=cp.float64), n, Nloc, 1.0, 1.0, 0.0, 1.0)


def build_obs_idx(n, obs_mask, obs_x0):
    """Slab indices for observations (obs_x0 0-based). Vectorized compress_bit."""
    Nloc = 1 << (n - 1)
    m = obs_mask.astype(np.int64) & ~(1 << obs_x0.astype(np.int64))
    x = obs_x0.astype(np.int64)
    rest = (m & ((1 << x) - 1)) | ((m >> (x + 1)) << x)
    return x * Nloc + rest


def phi_at(x, obs_idx, phat):
    """phi = ||x[obs] - phat||_2 (the projection distance at point x)."""
    xg = cp.asarray(x, dtype=cp.float64)
    return float(cp.sqrt(cp.sum((xg[cp.asarray(obs_idx)] - cp.asarray(phat)) ** 2)).get())


def push_to_ub(n, x, obs_idx, phat, check=False):
    """Feasibility push x -> exactly feasible RUM point -> (phi_ub, info).
    x is a slab iterate (device or host). check=True also reports the flow-conservation
    residual and the smallest flow of the pushed point."""
    Nnode = 1 << n
    Nloc = 1 << (n - 1)
    layers, bounds = layers_for(n)
    q = mobius(x, n)
    g_qf = make_r_feasible_gpu(n, q.astype(cp.float32), layers, bounds).astype(cp.float64)
    x_ub = zeta(g_qf, n)
    phi_ub = phi_at(x_ub, obs_idx, phat)
    if not check:
        return phi_ub, {"x_ub": x_ub}
    b = cp.zeros(Nnode, dtype=cp.float64)
    b[0] = -1.0
    b[Nnode - 1] = 1.0
    cons = float(cp.max(cp.abs(a_mul(g_qf, n, Nnode, Nloc) - b)).get())
    return phi_ub, {"cons_resid": cons, "q_min": float(g_qf.min().get()), "x_ub": x_ub}


def push_to_ub64(n, x, obs_idx, phat):
    """push_to_ub with the push itself in float64: x (slab iterate) -> (phi_ub, x_ub)."""
    layers, bounds = layers_gpu(n)
    x_ub = zeta(make_r_feasible_gpu64(n, mobius(x, n), layers, bounds), n)
    return phi_at(x_ub, obs_idx, phat), x_ub


def fw_lb_rows_gpu(n, obs_idx, y, phat):
    """certificate.fw_lb_rows (plain L2) on the GPU in float64: the lower bound at any point, from
    its values y at the observed slab positions obs_idx. Host or device inputs; returns lb_dist."""
    Nloc = 1 << (n - 1)
    layers, bounds = layers_gpu(n)                          # first: built while the slab is not held
    obs = cp.asarray(obs_idx)
    y = cp.asarray(y).astype(cp.float64, copy=False)
    r = cp.asarray(phat, dtype=cp.float64) - y
    f = float(r @ r)
    g = cp.zeros(n * Nloc, dtype=cp.float64)
    g[obs] = r
    g = butterfly_fast(g, n, Nloc, 1.0, 0.0, 1.0, 1.0)    # sum over subsets of the rest bits, per item
    rc = longest_path_gpu64(n, g, layers, bounds) - float(r @ y)
    return float(np.sqrt(max(0.0, f - 2.0 * rc)))


def fw_lb_stream64(n, menu, alternative, y, phat):
    """fw_lb_rows_gpu where its slab does not fit: the same float64 bound, with the lattice swept two
    layers at a time (flow_cg/pricing_stream.sweep_max64; 16 B per rank of the widest layer). Rows
    (menu bitmask, chosen alternative, the point's value y, phat) in any order; returns lb_dist."""
    from ..flow_cg.pricing_stream import sweep_max64
    alternative = np.asarray(alternative, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    r = np.asarray(phat, dtype=np.float64) - y
    perm = np.argsort(alternative, kind="stable")          # x-major rows, as the pricing kernels read them
    offsets = np.concatenate(([0], np.cumsum(np.bincount(alternative, minlength=n)))).astype(np.uint64)
    vmax = sweep_max64(n, np.asarray(menu, dtype=np.uint64)[perm], r[perm], offsets)
    rc = vmax - float(r @ y)
    return float(np.sqrt(max(0.0, float(r @ r) - 2.0 * rc)))
