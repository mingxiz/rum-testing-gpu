"""Hong and Li's numerical delta method on the RUM distance, from certified brackets.

The formulas are the reference's (rum/inference/inference.py on main):

    r_n = sqrt(n),  eps_n = n^(-1/3),  s = 1 + eps_n r_n       the defaults; both can be set
    P_b = p_hat + eps_n r_n (p_hat_b - p_hat)                   the b-th perturbed point
    p   = #{b : dist(P_b) >= tau} / B,   tau = s dist(p_hat)    p-value of H0: dist = 0
    CI  = [dist(p_hat) - q / r_n, inf)                          q = the (1 - alpha)-quantile of
                                                                    (dist(P_b) - dist(p_hat)) / eps_n

and the two-sided interval of the paper's statistical supplement, which needs no draw solved:

    [max(0, dist(p_hat) - c), dist(p_hat) + c]                  c = the (1 - alpha)-quantile of
                                                                    ||p_hat_b - p_hat||

The solver brackets every distance, dist(p_hat) in [L0, U0] and dist(P_b) in [L_b, U_b], so each
answer here is a bracket too, and the reported value is its conservative end. A draw is

    below      if U_b <  tau_lo = s L0     and counts 0
    above      if L_b >= tau_hi = s U0     and counts 1
    undecided  otherwise                   and counts 1, so p is an upper bound on the exact p-value

Pure numpy: nothing here calls a solver.
"""
import math

import numpy as np

BELOW, ABOVE, UNDECIDED = 0, 1, 2
ZERO = 1e-9                 # U0 at or below this: the data are rationalizable and p = 1 (the reference's short cut)


def scaling(n, r_n=None, eps_n=None):
    """(r_n, eps_n, s): sqrt(n) and n^(-1/3) unless given, and s = 1 + eps_n r_n."""
    if int(n) != n or n < 1:
        raise ValueError("n, the sample size, must be a positive integer; got %r" % (n,))
    n = int(n)
    r_n = math.sqrt(n) if r_n is None else float(r_n)
    eps_n = n ** (-1.0 / 3.0) if eps_n is None else float(eps_n)
    if not (math.isfinite(r_n) and r_n > 0 and math.isfinite(eps_n) and eps_n > 0):
        raise ValueError("r_n and eps_n must be finite and > 0")
    return r_n, eps_n, 1.0 + eps_n * r_n


def perturb(p_hat, p_b, r_n, eps_n):
    """P_b = p_hat + eps_n r_n (p_b - p_hat); p_b is one draw or a (k, M) block of them."""
    return p_hat + (eps_n * r_n) * (p_b - p_hat)


def check_alpha(alpha):
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1); got %r" % alpha)
    return alpha


def decide(L, U, tau_lo, tau_hi):
    """Each draw's side of tau from its bracket: BELOW, ABOVE or UNDECIDED (an int8 array)."""
    L, U = np.asarray(L, dtype=np.float64), np.asarray(U, dtype=np.float64)
    d = np.full(U.shape, UNDECIDED, dtype=np.int8)
    d[L >= tau_hi] = ABOVE
    d[U < tau_lo] = BELOW
    return d


def p_bracket(decision):
    """(p_lo, p_hi) = (#above / B, (#above + #undecided) / B). p_hi is the reported p-value."""
    decision = np.asarray(decision)
    above = int((decision == ABOVE).sum())
    undecided = int((decision == UNDECIDED).sum())
    return above / decision.size, (above + undecided) / decision.size


def locked(n_above, n_open, B, alpha):
    """Whether the decision at alpha can no longer change while n_open draws are still undecided
    or unsolved: True if counting all of them 1 still gives p <= alpha (reject), or if the draws
    certified above already give p > alpha (no rejection)."""
    return (n_above + n_open) / B <= alpha or n_above / B > alpha


def ci_lower(L0, U0, L, U, alpha, r_n, eps_n):
    """The lower end of the one-sided interval as (value, its bracket's upper end, (q_lo, q_hi)).

    value = L0 - q_hi / r_n with q_hi the (1 - alpha)-quantile of (U_b - L0) / eps_n: the
    reference's arithmetic with dist(p_hat) = L0 and dist(P_b) = U_b. The exact lower end lies in
    [value, U0 - q_lo / r_n], q_lo the same quantile of (L_b - U0) / eps_n, since the lower end
    increases with dist(p_hat) and decreases with every dist(P_b)."""
    q_hi = float(np.quantile((np.asarray(U, dtype=np.float64) - L0) / eps_n, 1.0 - alpha))
    q_lo = float(np.quantile((np.asarray(L, dtype=np.float64) - U0) / eps_n, 1.0 - alpha))
    return L0 - q_hi / r_n, U0 - q_lo / r_n, (q_lo, q_hi)


def distance_interval(L0, U0, dev, alpha):
    """The two-sided interval as (lower, upper, c): c is the (1 - alpha)-quantile of the draws'
    ||p_hat_b - p_hat|| (dev), and [max(0, L0 - c), U0 + c] holds [max(0, d - c), d + c] for every
    d = dist(p_hat) in [L0, U0]. It is valid because |dist(p) - dist(p0)| <= ||p - p0|| for the
    distance to any convex set, so it needs no draw solved."""
    c = float(np.quantile(np.asarray(dev, dtype=np.float64), 1.0 - alpha))
    return max(0.0, L0 - c), U0 + c, c


def quantile_band(L, U, alpha):
    """(Q(L), Q(U)): the (1 - alpha)-quantile of the draws' distances lies between these. Only a
    draw whose bracket overlaps the band can move either end."""
    return (float(np.quantile(np.asarray(L, dtype=np.float64), 1.0 - alpha)),
            float(np.quantile(np.asarray(U, dtype=np.float64), 1.0 - alpha)))
