"""RUM projection solvers.

A certified distance comes from the RUM class (solver/rum_solver.py): it checks the input,
chooses the method when none is named, enforces max_time and returns lb <= dist <= ub.

    from solver.rum_solver import RUM
    res = RUM.from_rows(cd, n).calculate_distance()          # tolerances: solver/defaults.py

solve() is the paper's timing race: one named method runs until its feasible upper bound is
within rel_tol of a known optimum.

    import solver
    r = solver.solve(n, cd, method="bm_admm", opt_ref=OPT)    # stop at (UB - OPT)/OPT <= 1e-2

Input
-----
n  : number of alternatives
cd : (mp, 3) array — rows (menu bitmask, chosen alternative 0-based, p_hat);
     every observed (menu, alternative) cell is one row (data/README.md).

Methods
-------
    "bm_admm"     Block-Marschak ADMM, GPU (paper: BM-ADMM)
    "flow_admm"   arc-flow ADMM with prestored Mhat, GPU (paper: Flow-ADMM)
    "flow_cg"     column generation, Bellman-Ford GPU pricing (paper: CGBFQP-GPU)
    "ip_cg"       column generation, exact IP pricing, CPU + Gurobi (paper: CGIP)
    "ip_cg_heur"  column generation, randomized best-insertion pricing certified
                  by the IP, CPU + Gurobi (paper: CGIP-heuristic)
Old names are accepted (ALIASES): cgbfqp / cgbf, cgip, cgip_heur / cgipheuristic_rand.
The exact QPs bm_qp / flow_qp run through the RUM class.

The race
--------
Stops when (UB - opt_ref)/opt_ref <= rel_tol (default 1e-2), UB = the objective of a feasible
point and opt_ref = a certified optimum. With opt_ref = 0 the relative target is undefined: the
race then needs abs_tol and stops at UB - opt_ref <= abs_tol.

Result: dict with method, criterion, ub, iters, wall_s, build_s, stop, converged, plus
solver-specific fields. `iters` counts ADMM iterations for the ADMMs and generated columns for
the column-generation methods. wall_s is the solve loop; build_s is one-off preparation reported
separately (Flow-ADMM's Mhat, Gurobi's pricing model, BM-ADMM's optional beta probe).
"""
from . import params  # noqa: F401

METHODS = ("bm_admm", "flow_admm", "flow_cg", "ip_cg", "ip_cg_heur")
ALIASES = {"cgbfqp": "flow_cg", "cgbf": "flow_cg", "cgip": "ip_cg",
           "cgip_heur": "ip_cg_heur", "cgipheuristic_rand": "ip_cg_heur"}


def solve(n, cd, method=None, criterion="race", opt_ref=None, rel_tol=None,
          time_limit=None, verbose=False, **kw):
    if criterion != "race":
        raise ValueError("solver.solve runs the race to a known optimum only (criterion='race'); "
                         "for a certified distance use solver.rum_solver.RUM(...).calculate_distance()")
    if method is None or method == "auto":
        raise ValueError("solver.solve needs a named method; the automatic choice is "
                         "solver.rum_solver.RUM(...).calculate_distance(method=None)")
    method = ALIASES.get(method, method)
    if method not in METHODS:
        raise ValueError("method must be one of %s (or an old name: %s)" % (METHODS, tuple(ALIASES)))
    if opt_ref == 0 and kw.get("abs_tol") is None:
        raise ValueError("opt_ref = 0: the relative race target (UB - opt)/opt is undefined; "
                         "pass abs_tol (the race then stops at UB - opt_ref <= abs_tol)")
    if method == "bm_admm":
        from .bm_admm import solve_bm_admm
        r = solve_bm_admm(n, cd, criterion=criterion, opt_ref=opt_ref, rel_tol=rel_tol,
                          time_limit=time_limit, verbose=verbose, **kw)
    elif method == "flow_cg":
        from .flow_cg import solve_cgbfqp
        r = solve_cgbfqp(n, cd, criterion=criterion, opt_ref=opt_ref, rel_tol=rel_tol,
                         time_limit=time_limit, verbose=verbose, **kw)
    elif method == "flow_admm":
        from .flow_admm import solve_flow_admm
        r = solve_flow_admm(n, cd, criterion=criterion, opt_ref=opt_ref,
                            rel_tol=1e-2 if rel_tol is None else rel_tol,
                            time_limit=900.0 if time_limit is None else time_limit,
                            verbose=verbose, **kw)
    else:
        from .ip_cg import solve_cgip_paper
        r = solve_cgip_paper(n, cd, variant="ip" if method == "ip_cg" else "heur", criterion=criterion,
                             opt_ref=opt_ref, rel_tol=rel_tol, verbose=verbose, **kw)
    if opt_ref is not None and r.get("ub") is not None:
        r["abs_to_opt"] = r["ub"] - opt_ref
        r["rel_to_opt"] = (r["ub"] - opt_ref) / opt_ref if opt_ref != 0 else None   # OPT = 0: absolute only
    return r


def warmup(methods=("bm_admm", "flow_cg")):
    """Compile kernels / calibrate on a tiny instance so timed runs are clean."""
    for m in methods:
        m = ALIASES.get(m, m)
        if m == "bm_admm":
            from .bm_admm import warmup as w
        elif m == "flow_cg":
            from .flow_cg import warmup as w
        else:
            continue
        w()
