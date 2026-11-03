"""CGIP and CGIP-heuristic: column generation with integer-programming pricing
(cgip.py, from rum/algorithms/cgip.py on the main branch). CPU + Gurobi.

    from solver.ip_cg import solve_cgip_paper
    r = solve_cgip_paper(n, cd, variant="ip",   criterion="race", opt_ref=OPT)
    r = solve_cgip_paper(n, cd, variant="heur", criterion="cert", rel_tol=1e-3)

variant="ip"   : exact Gurobi IP pricing every iteration (solve_cgip, mode='ip')
variant="heur" : cgipheuristic_rand — randomized best-insertion pricing,
                 IP-certified (solve_cgipheuristic_rand)
race : stop at UB < (1 + rel_tol) * opt_ref (ub_stop), or UB < abs_tol when opt_ref = 0;
cert : (UB-LB)/LB <= rel_tol, or UB - LB <= abs_tol when abs_tol is given.
Iterations = columns generated. build_s = one-off Gurobi pricing-model build.
No wall cap is enforced inside (Gurobi cannot be interrupted safely from a
library call); run under a process timeout for capped experiments.
"""
import io
import re
import sys
import time

import numpy as np

_ITER = re.compile(r"Iter\s+(\d+)\s*\|\s*UB = ([-\d.eE+]+)(?:\s*\|\s*LB = ([-\d.eE+]+))?.*?cols = (\d+)")


class _Tee(io.TextIOBase):
    """Captures cgip.py's per-iteration verbose lines into a trace
    [column, wall_s, UB, LB or None, n_cols] (and forwards them when echo=True)."""

    def __init__(self, echo, real):
        self.echo, self.real, self.t0, self.buf, self.trace = echo, real, time.perf_counter(), "", []

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            m = _ITER.search(line)
            if m:
                self.trace.append([int(m.group(1)), time.perf_counter() - self.t0, float(m.group(2)),
                                   float(m.group(3)) if m.group(3) else None, int(m.group(4))])
            if self.echo:
                self.real.write(line + "\n")
        return len(s)

    def flush(self):
        if self.echo:
            self.real.flush()


def _import():
    try:
        from ..rum import RUM
        from . import cgip as cgip_mod
        from ..common.gurobi_env import _ENV
    except ImportError as e:   # gurobipy missing
        raise ImportError("CGIP needs gurobipy (and a Gurobi licence): %s" % e)
    return RUM, cgip_mod, _ENV


def solve_cgip_paper(n, cd, variant="ip", criterion="race", opt_ref=None, rel_tol=None, abs_tol=None,
                     max_iters=100000, threads=None, seed=0, qp_backend="gurobi", verbose=False):
    RUM, cgip_mod, ENV = _import()
    if threads is not None:
        ENV.setParam("Threads", int(threads))
    if rel_tol is None:
        rel_tol = 1e-2 if criterion == "race" else 1e-3
    cd = np.asarray(cd, dtype=np.float64)
    r = RUM(cd[:, 0].astype(np.uint64), cd[:, 1].astype(np.int64), cd[:, 2], num_alternatives=n)
    t0 = time.perf_counter()
    if variant == "ip":
        cgip_mod.build_cgip(r)
    else:
        cgip_mod.build_cgipheuristic(r)
    build_s = time.perf_counter() - t0
    target = None
    if criterion == "race":
        if opt_ref is None:
            raise ValueError("criterion='race' needs opt_ref")
        if opt_ref == 0 and abs_tol is None:      # (1 + rel_tol) * 0 = 0 is never reached
            raise ValueError("opt_ref = 0: pass abs_tol (the race stops at UB < abs_tol)")
        target = (1.0 + rel_tol) * opt_ref if opt_ref != 0 else float(abs_tol)   # OPT = 0: absolute only
        kw = dict(ub_stop=target)
    else:
        kw = dict(rel_tol=rel_tol, abs_tol=abs_tol)
    tee = _Tee(echo=verbose, real=sys.stdout)
    sys.stdout = tee                       # the solver prints one line per column when verbose=True
    t0 = time.perf_counter()
    try:
        if variant == "ip":
            res = cgip_mod.solve_cgip(r, cd[:, 2], vcov=None, mode="ip", max_iters=max_iters, seed=seed,
                                      verbose=True, qp_backend=qp_backend, **kw)
        else:
            res = cgip_mod.solve_cgipheuristic_rand(r, cd[:, 2], vcov=None, max_iters=max_iters, seed=seed,
                                                    verbose=True, qp_backend=qp_backend, **kw)
    finally:
        sys.stdout = tee.real
    wall = time.perf_counter() - t0
    ub, lb = float(res["UB"]), float(res["LB"])
    return dict(method="ip_cg" if variant == "ip" else "ip_cg_heur", criterion=criterion, ub=ub, lb=lb,
                iters=int(res["n_columns"]), wall_s=wall, build_s=build_s,
                stop=res.get("decision") or ("converged" if res.get("converged") else "max_iters"),
                converged=bool(res.get("converged")) or (criterion == "race" and ub < target),
                ip_calls=int(res.get("ip_calls", 0)), n_support=int(res.get("n_support", 0)), threads=threads,
                qp_backend=qp_backend, trace=tee.trace)
