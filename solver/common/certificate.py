r"""Frank-Wolfe lower bound on the projection distance, from any slab iterate, and the CPU push that
gives the exact QPs their upper bound.

For f(x) = ||x - phat||^2 over the observed coordinates and ANY point x (feasible or not),
convexity gives

    f* >= f(x) - 2 rc,   rc = max over rankings sigma of <r, v_sigma> - <r, x>,   r = phat - x,

with r zero off the observed coordinates. The maximum over rankings is a longest path through
the subset lattice: sum the residual over subsets, then a DP over the 2^n nodes. The slab layout
(item i at i*2^(n-1) + compress_bit(S minus i, i)) is already the layout the DP indexes, so the
residual is reshaped, never gathered per observation (which matters at mp ~ 1e6). With a weight
matrix W on the observed rows, f(x) = r' W r and the same bound holds with W r in place of r
(fw_lb_rows).

The exact QPs (bm_qp, flow_qp) run without a GPU, and Gurobi's point meets the constraints only
to Gurobi's tolerances. push_flow, the float64 CPU twin of arcflow_kernels.make_r_feasible_gpu,
turns its arc flow into an exact unit flow, and zeta_rows reads that flow's choice probabilities
at the observed rows: a feasible point, so its distance is an upper bound.
"""
import numpy as np
from numba import njit

_ARCS = {}


@njit(cache=True)
def _longest_path(R, n):
    """Value of the best path from the empty set to the full set, where the arc S -> S + {x} is
    worth R[compress(S, x), x]. Index order is a topological order (adding a bit raises it)."""
    Nnode = 1 << n
    dp = np.empty(Nnode)
    for i in range(Nnode):
        dp[i] = -1e300
    dp[0] = 0.0
    for S in range(Nnode):
        if dp[S] < -1e299:
            continue
        v = dp[S]
        for x in range(n):
            if (S >> x) & 1:
                continue
            head = S | (1 << x)
            val = v + R[(S & ((1 << x) - 1)) | ((S >> (x + 1)) << x), x]
            if val > dp[head]:
                dp[head] = val
    return dp[Nnode - 1]


def _best_ranking(n, g_slab):
    """max over rankings sigma of <g, v_sigma>, for a slab vector g (zero off the observed coords)."""
    Nloc = 1 << (n - 1)
    R = g_slab.reshape(n, Nloc).T.copy()                # (Nloc, n), C-contiguous
    for bit in range(n - 1):                            # sum over subsets, one rest bit at a time
        s = 1 << bit
        R = R.reshape(Nloc // (2 * s), 2 * s, n)
        R[:, s:2 * s, :] += R[:, :s, :]
        R = R.reshape(Nloc, n)
    return float(_longest_path(R, n))


def fw_lb_slab(n, x2_slab, xhat_slab, wmask_slab):
    """Lower bound at a slab iterate. Inputs are 1-D float64 arrays of length n*2^(n-1):
    xhat_slab holds phat at the observed positions and wmask_slab is 1 there, 0 elsewhere.
    Returns (lb_dist, rc, f_raw)."""
    residual = (xhat_slab - x2_slab) * wmask_slab       # 0 at unobserved coords
    f_raw = float(np.dot(residual, residual))
    inner = float(np.dot(residual, x2_slab))
    rc = _best_ranking(n, residual) - inner
    lb_obj = f_raw - 2.0 * rc
    return float(np.sqrt(max(0.0, lb_obj))), float(rc), f_raw


def fw_lb_rows(n, obs_idx, y, phat, W=None):
    """Lower bound at any point, given by its values y at the observed slab positions obs_idx (y need
    not be feasible). W is the weight matrix of the norm on those rows, dense or sparse (None = L2).
    Returns (lb_dist, rc, f)."""
    r = phat - y
    Wr = r if W is None else 0.5 * np.asarray(W @ r + W.T @ r, dtype=np.float64).ravel()
    f = float(r @ Wr)
    g = np.zeros(n << (n - 1))
    g[obs_idx] = Wr
    rc = _best_ranking(n, g) - float(Wr @ y)
    lb_obj = f - 2.0 * rc
    return float(np.sqrt(max(0.0, lb_obj))), float(rc), f


def slab_index(n, obs_mask, obs_x0):
    """Slab position of each observation (obs_x0 0-based): transforms.build_obs_idx without CuPy."""
    Nloc = 1 << (n - 1)
    m = obs_mask.astype(np.int64) & ~(1 << obs_x0.astype(np.int64))
    x = obs_x0.astype(np.int64)
    rest = (m & ((1 << x) - 1)) | ((m >> (x + 1)) << x)
    return x * Nloc + rest


def _arcs(n):
    """Tail and head node of every arc x*2^(n-1) + rest (tail = rest with bit x put back, head = tail
    plus x), and the arcs ordered by the size of their tail with the start of each size. Once per n."""
    if n not in _ARCS:
        Nloc = 1 << (n - 1)
        rest = np.arange(Nloc, dtype=np.int64)
        tail = np.concatenate([(rest & ((1 << x) - 1)) | ((rest >> x) << (x + 1)) for x in range(n)])
        head = tail | np.repeat(np.int64(1) << np.arange(n, dtype=np.int64), Nloc)
        size = np.zeros_like(tail)
        for b in range(n):
            size += (tail >> b) & 1
        order = np.argsort(size, kind="stable")
        _ARCS[n] = (tail, head, order, np.searchsorted(size[order], np.arange(n + 1)))
    return _ARCS[n]


def push_flow(n, q, eps=1e-12):
    """make_r_feasible_gpu in float64: clip the arc flow q at 0 (plus eps), then push unit mass
    forward from the empty set, each node splitting its mass over its out-arcs in proportion.
    The result is an exact unit flow: >= 0 and conserved at every node, up to rounding."""
    tail, head, order, bounds = _arcs(n)
    w = np.maximum(np.asarray(q, dtype=np.float64), 0.0) + eps
    outw = np.bincount(tail, weights=w, minlength=1 << n)
    mass = np.zeros(1 << n)
    mass[0] = 1.0
    rf = np.empty_like(w)
    for c in range(n):                                  # the arcs leaving the sets of size c
        a = order[bounds[c]:bounds[c + 1]]
        rf[a] = mass[tail[a]] * w[a] / outw[tail[a]]
        mass += np.bincount(head[a], weights=rf[a], minlength=1 << n)
    return rf


def zeta_rows(n, q, obs_idx):
    """The choice probabilities of the arc flow q at the observed slab positions obs_idx: P(x | D) is
    q summed over the arcs that add x to a set containing D minus x (superset sums in x's block)."""
    Nloc = 1 << (n - 1)
    X = np.array(q, dtype=np.float64).reshape(n, Nloc)
    for bit in range(n - 1):                            # sum over supersets, one rest bit at a time
        s = 1 << bit
        V = X.reshape(n, Nloc // (2 * s), 2, s)
        V[:, :, 0, :] += V[:, :, 1, :]
    return X.reshape(-1)[obs_idx]
