"""The weight matrix Omega of the distance norm ||x||_Omega = sqrt(x' Omega x), in the form the CPU
solvers (solver/ip_cg, solver/exact_qp) take it; vcov=None means the plain L2 norm."""
import numpy as np
import scipy.sparse as sp


def _as_weight(vcov):
    """Normalize a weight matrix: None stays None, scipy sparse stays sparse,
    anything else becomes a dense float array. (Do NOT np.asarray a sparse
    matrix -- that yields a 0-d object array.)"""
    if vcov is None or sp.issparse(vcov):
        return vcov
    return np.asarray(vcov, dtype=np.float64)
