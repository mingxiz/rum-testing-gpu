"""solver.rum: the RUM facade.

dist(p) = min_{p~ in P_RUM} || p - p~ ||   is the distance from a per-observation
choice-probability vector p to the RUM polytope. RUM holds the fixed menu
geometry once and evaluates dist under interchangeable solver backends.

Build sharing across bootstrap samples
--------------------------------------
The menu geometry -- (menu, alternative, num_alternatives) -- is FIXED; a
bootstrap resample
only changes the choice probabilities. Every backend's expensive setup (matrix
factorizations, lattice layers, Gurobi model, ...) depends only on the geometry,
so it is built ONCE and cached on a per-backend attribute:

    self.cgbf, self.flow_qp, self.bm_qp, self.flow_admm, self.bm_admm,
    self.cgip, self.cgipheuristic

`calculate_distance(p, vcov, method)` builds the backend for `method` on first
use (cached on the attribute above), then reuses it for every subsequent p -- so
a whole bootstrap loop pays the build cost once. Each backend lives in its own
solver folder (ip_cg, ip_cg_heur -> solver/ip_cg; flow_qp, bm_qp ->
solver/exact_qp) and is imported on first use. flow_cg, flow_admm and bm_admm
are not wired in yet. Old method names are accepted (solver.ALIASES).
"""
import numpy as np
import scipy.sparse as sp

from . import ALIASES as _ALIASES

# Recognized backend names (old names map through solver.ALIASES).
_METHODS = ('flow_cg', 'flow_admm', 'bm_admm', 'flow_qp', 'bm_qp',
            'ip_cg', 'ip_cg_heur')


def _is_identity(Omega):
    """True if Omega is exactly the identity matrix (dense or sparse). Used to
    take the plain resid @ resid objective instead of resid @ Omega @ resid."""
    n = Omega.shape[0]
    if Omega.shape[1] != n:
        return False
    if sp.issparse(Omega):
        return (Omega - sp.identity(n, dtype=Omega.dtype, format='csr')
                ).count_nonzero() == 0
    return np.array_equal(Omega, np.eye(n, dtype=Omega.dtype))


class RUM:
    """RUM-projection distance for a fixed menu geometry.

    Parameters
    ----------
    menu : (M,) array of menu bitmasks over {0..n-1} (the chosen bit is set).
    alternative : (M,) array of chosen alternatives, 0-based.
    choice_probability : (M,) observed choice probabilities p_hat. These are the
        default probabilities; pass a resampled vector to calculate_distance(p=)
        for each bootstrap draw.
    num_alternatives : int, number of alternatives.
    sample_size : optional int, the number of sampled units behind p_hat. This
        is the delta-method sample size used by rum.inference (r_n =
        sqrt(sample_size), eps_n = sample_size^{-1/3}) -- NOT the same as
        num_alternatives. May be left None and supplied per-call instead.

    The geometry (menu, alternative, num_alternatives) is fixed; solver builds
    that depend only on it are cached on self.<method> and shared across all
    bootstrap samples.
    """

    def __init__(self, menu, alternative, choice_probability, num_alternatives,
                 sample_size=None):
        menu = np.asarray(menu, dtype=np.uint64)
        alternative = np.asarray(alternative, dtype=np.int64)
        p = np.asarray(choice_probability, dtype=np.float64)
        M = menu.shape[0]
        if not (alternative.shape[0] == p.shape[0] == M):
            raise ValueError(
                f"menu, alternative, choice_probability must share length M; "
                f"got {M}, {alternative.shape[0]}, {p.shape[0]}")

        # chosen alternative must be a member of its menu
        in_menu = (menu.astype(np.int64) >> alternative) & 1
        if not np.all(in_menu == 1):
            bad = int(np.where(in_menu == 0)[0][0])
            raise ValueError(
                f"observation {bad}: chosen alt {int(alternative[bad])} "
                f"not in menu mask {int(menu[bad])}")

        # ---- fixed menu geometry (shared across all bootstrap samples) ----
        self.menu = menu
        self.alternative = alternative
        self.num_alternatives = int(num_alternatives)
        self.M = M
        # delta-method sample size (# units behind p_hat), distinct from
        # num_alternatives; used by rum.inference. May be None here and passed
        # to the inference calls instead.
        self.sample_size = None if sample_size is None else int(sample_size)
        # default choice probabilities; a bootstrap sample overrides these.
        self.choice_probability = p
        # variance-covariance / weight matrix for the distance norm; set on the
        # first calculate_distance call (defaults to identity if not provided).
        self.vcov = None
        # distance of the observed dataset, dist(choice_probability): cached
        # whenever calculate_distance runs on the default p (p=None). Bootstrap
        # calls pass an explicit p and leave these attributes untouched.
        #   distance_lb / distance_ub : the certified [LB, UB] bracket on the
        #   dataset distance. The exact QP backends (flow_qp/bm_qp) fill all
        #   three with the exact value; the bracketed CG backends (ip_cg/
        #   ip_cg_heur) fill only distance_lb/distance_ub and leave
        #   distance = None (there is no exact point estimate).
        self.distance = None
        self.distance_lb = None
        self.distance_ub = None

        # ---- cached geometry-only builds (reused across bootstrap resamples) --
        # None until the corresponding backend is first used; see _ensure_build.
        self.cgbf = None
        self.flow_qp = None
        self.bm_qp = None
        self.flow_admm = None
        self.bm_admm = None
        self.cgip = None
        self.cgipheuristic = None

    # ------------------------------------------------------------------ #
    #  Distance                                                            #
    # ------------------------------------------------------------------ #
    def calculate_distance(self, p=None, vcov=None, method='flow_cg', **kwargs):
        """Evaluate the projection distance dist(p) with the chosen backend.

        Parameters
        ----------
        p : (M,) probability vector; defaults to self.choice_probability. Pass a
            resampled vector for a bootstrap draw.
        vcov : optional (M, M) variance-covariance / weighting matrix Omega
            defining the distance norm ||x||_Omega = sqrt(x' Omega x). May be a
            scipy.sparse matrix or a dense array. If None, a sparse identity is
            used (unweighted L2). The matrix actually used is stored on self.vcov.
        method : one of 'flow_cg', 'flow_admm', 'bm_admm', 'flow_qp', 'bm_qp',
            'ip_cg', 'ip_cg_heur' (case-insensitive; old names accepted).

        Each backend builds its geometry-only solver once (cached on
        self) and reuses it for every p. Returns the backend result
        dict (at least 'distance' and 'method')."""
        is_dataset = p is None          # default p => the observed dataset itself
        p = self.choice_probability if p is None else np.asarray(p, dtype=np.float64)
        if p.shape[0] != self.M:
            raise ValueError(f"p has length {p.shape[0]}, expected M={self.M}")
        if vcov is None:
            print("vcov not provided; using an identity matrix instead")
            vcov = sp.identity(p.shape[0], format='csr', dtype=np.float64)
            omega = None                          # identity => unweighted fast path
        else:
            if not sp.issparse(vcov):            # keep sparse sparse; densify lists
                vcov = np.asarray(vcov, dtype=np.float64)
            if vcov.shape != (self.M, self.M):
                raise ValueError(
                    f"vcov must be ({self.M}, {self.M}); got {vcov.shape}")
            omega = None if _is_identity(vcov) else vcov
        self.vcov = vcov          # weight matrix (sparse when it makes sense)
        # `omega` is what the QP objective uses: None (=> resid @ resid) when the
        # weight is the identity, else the weight matrix (=> resid @ omega @ resid).
        method = str(method).strip().lower().replace('-', '_')
        method = _ALIASES.get(method, method)

        if method == 'flow_cg':
            raise NotImplementedError("flow_cg is not wired into RUM yet; CGBFQP-GPU runs "
                                      "from (n, cd): solver.flow_cg.solve_cgbfqp")

        elif method == 'flow_admm':
            raise NotImplementedError("flow_admm is not wired into RUM yet; it runs "
                                      "from (n, cd): solver.flow_admm.solve_flow_admm")

        elif method == 'bm_admm':
            raise NotImplementedError("bm_admm is not wired into RUM yet; it runs "
                                      "from (n, cd): solver.bm_admm.solve_bm_admm")

        elif method == 'flow_qp':
            from .exact_qp import flow_bm_qp
            if self.flow_qp is None:                 # 1. build formulation once
                flow_bm_qp.build_arc_flow(self)
            out = flow_bm_qp.solve_arc_flow_qp(       # 2. solve (omega=None => resid@resid)
                self, p, vcov=omega, **kwargs)

        elif method == 'bm_qp':
            from .exact_qp import flow_bm_qp
            if self.bm_qp is None:                   # 1. build formulation once
                flow_bm_qp.build_BM(self)
            out = flow_bm_qp.solve_BM_qp(             # 2. solve (omega=None => resid@resid)
                self, p, vcov=omega, **kwargs)

        elif method == 'ip_cg':
            from .ip_cg import cgip
            if self.cgip is None:                    # 1. build geometry once
                cgip.build_cgip(self)
            out = cgip.solve_cgip(                    # 2. exact IP-pricing CG
                self, p, vcov=omega, mode='ip', **kwargs)

        elif method == 'ip_cg_heur':
            from .ip_cg import cgip
            if self.cgipheuristic is None:           # 1. build geometry once
                cgip.build_cgipheuristic(self)
            out = cgip.solve_cgipheuristic_rand(     # 2. random-insertion CG (IP-certified)
                self, p, vcov=omega, **kwargs)

        else:
            raise ValueError(f"unknown method {method!r}; choose from {_METHODS}")

        out = dict(out)
        out.setdefault('method', method)
        if is_dataset:                  # cache the observed-dataset distance bracket
            if method in ('flow_qp', 'bm_qp'):
                # exact backends: all three attributes = the exact distance.
                d = float(out['distance'])
                self.distance = d
                self.distance_lb = d
                self.distance_ub = d
            elif method in ('ip_cg', 'ip_cg_heur', 'flow_cg'):
                # bracketed backends: cache the certified [LB, UB]; leave
                # self.distance = None (no exact point estimate).
                self.distance_lb = float(out['LB'])
                self.distance_ub = float(out['UB'])
        return out

    # ------------------------------------------------------------------ #
    #  Numerical delta-method inference (Hong & Li 2018)                   #
    # ------------------------------------------------------------------ #
    def inference(self, boot_p, method='flow_qp', mode='ci', n=None, **kwargs):
        """Numerical delta-method inference on dist(p_hat). mode is
        'ci' | 'p_value' | 'test'; boot_p is (B, M) bootstrap draws of p_hat;
        n is the sample size (defaults to self.sample_size). Ensures
        self.distance / self.vcov and the built formulation once, then loops.
        See rum.inference.inference on main; not part of this package yet."""
        raise NotImplementedError("RUM.inference is not wired in yet (rum.inference on main "
                                  "was not brought into this package)")
