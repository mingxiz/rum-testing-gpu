"""IP-CG and IP-H-CG (solver/ip_cg/cgip.py) at distance 0, against the exact QP, and on their time limits.
CPU, one Gurobi thread, under a minute.

    python tests/test_cgip.py

The first instance has n = 7, random menus and p = a mixture of 5 rankings, so the distance is exactly 0:
with strict stalls both pricing modes must certify it.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "data")]
import numpy as np                                   # noqa: E402
from solver.common.gurobi_env import _ENV            # noqa: E402
from solver.rum import RUM as OldRUM                 # noqa: E402
from solver.ip_cg import cgip                        # noqa: E402
from gen_instances import gen_medium                 # noqa: E402

_ENV.setParam("Threads", 1)
FAIL = []


def check(name, ok, detail=""):
    print("%-62s %s %s" % (name, "ok  " if ok else "FAIL", detail), flush=True)
    if not ok:
        FAIL.append(name)


def geom(cd, n):
    r = OldRUM(cd[:, 0].astype(np.uint64), cd[:, 1].astype(np.int64), cd[:, 2], num_alternatives=n)
    cgip.build_cgip(r)
    cgip.build_cgipheuristic(r)
    return r


n = 7
cd = np.asarray(gen_medium(n, mult=3)[0], dtype=np.float64)
rng = np.random.default_rng(1)
rum = geom(cd, n)
b = rum.cgip
p0 = np.zeros(len(cd))
for w in rng.dirichlet(np.ones(5)):
    p0 += w * b.column(b._pos(list(rng.permutation(n))))


def ncols_distinct(out, build):
    cols = {cgip._col_key(build.column(build._pos(o))) for o in out["orders"]}
    return len(cols), len(out["orders"])


print("distance-0 instance: n = %d, %d rows" % (n, len(cd)))
for strict in (False, True):
    for qpb in ("gurobi", "native"):
        for mode in ("ip", "rand"):
            kw = dict(abs_tol=1e-6, qp_backend=qpb, max_iters=60, strict=strict)
            t = time.perf_counter()
            if mode == "ip":
                out = cgip.solve_cgip(rum, p0, mode="ip", **kw)
            else:
                out = cgip.solve_cgipheuristic_rand(rum, p0, **kw)
            dt = time.perf_counter() - t
            nd, no = ncols_distinct(out, rum.cgip if mode == "ip" else rum.cgipheuristic)
            line = ("strict=%-5s master=%-6s mode=%-4s" % (strict, qpb, mode))
            info = ("converged=%s stop=%s UB=%.2e LB=%.2e iters=%d columns %d (distinct %d) kkt=%s %.2fs"
                    % (out["converged"], out["stop_reason"], out["UB"], out["LB"], out["iters"], no, nd,
                       "%.1e" % out["master_kkt"] if "master_kkt" in out else "-", dt))
            if strict:
                honest = (out["converged"] and out["UB"] - out["LB"] <= 1e-6) or \
                         (not out["converged"] and out["stop_reason"] == "repeat")
                ok = honest and nd == no and (qpb == "gurobi" or (out["converged"] and out["UB"] < 1e-6))
                check(line, ok, info)
            else:
                print("%-62s (old behaviour) %s" % (line, info), flush=True)

# ---- strict on an ordinary instance: same distance as the exact QP
for k, (nn, mult, seed) in enumerate([(7, 4, 2), (8, 3, 5)]):
    cdk = np.asarray(gen_medium(nn, mult=mult, menu_seed=seed, phat_seed=seed)[0], dtype=np.float64)
    rk = geom(cdk, nn)
    ref = float(OldRUM(cdk[:, 0].astype(np.uint64), cdk[:, 1].astype(np.int64), cdk[:, 2],
                       num_alternatives=nn).calculate_distance(method="bm_qp")["distance"])
    for qpb in ("native", "gurobi"):
        for mode in ("ip", "rand"):
            kw = dict(abs_tol=1e-7, rel_tol=0.0, qp_backend=qpb, max_iters=3000, strict=True)
            out = (cgip.solve_cgip(rk, cdk[:, 2], mode="ip", **kw) if mode == "ip"
                   else cgip.solve_cgipheuristic_rand(rk, cdk[:, 2], **kw))
            ok = out["LB"] <= ref + 1e-7 and out["UB"] >= ref - 1e-7 and \
                (out["converged"] or out["stop_reason"] == "repeat")
            check("n=%d random menus, strict, %s, %s" % (nn, qpb, mode), ok,
                  "[%.9f, %.9f] ref %.9f stop=%s conv=%s cols %d kkt %.1e"
                  % (out["LB"], out["UB"], ref, out["stop_reason"], out["converged"], out["n_columns"],
                     out["master_kkt"]))

# ---- time limit: an instance whose IP pricing takes a while
nn = 12
cdt = np.asarray(gen_medium(nn, mult=6, menu_seed=3, phat_seed=3)[0], dtype=np.float64)
rt = geom(cdt, nn)
for mode in ("ip", "rand"):
    for tl in (0.3, 2.0):
        t = time.perf_counter()
        kw = dict(abs_tol=0.0, rel_tol=0.0, qp_backend="native", max_iters=100000, strict=True, time_limit=tl)
        out = (cgip.solve_cgip(rt, cdt[:, 2], mode="ip", **kw) if mode == "ip"
               else cgip.solve_cgipheuristic_rand(rt, cdt[:, 2], **kw))
        dt = time.perf_counter() - t
        check("time_limit %.1f s, n = %d, %s" % (tl, nn, mode),
              out["stop_reason"] == "time_limit" and dt < tl + 1.0 and out["LB"] <= out["UB"],
              "stop=%s after %.2fs, [%.6f, %.6f], %d columns, %d IP solves"
              % (out["stop_reason"], dt, out["LB"], out["UB"], out["n_columns"], out["ip_calls"]))

# ---- an IP stopped by its time limit before any incumbent: the bound is still returned
bt = rt.cgip
g = np.random.default_rng(0).standard_normal(bt.M)
prm = bt._model.Params
tl0 = prm.TimeLimit
prm.TimeLimit = 0.0
try:
    order, bound = bt.ip_price(g)
    status, sols = bt._model.Status, bt._model.SolCount
finally:
    prm.TimeLimit = tl0
check("ip_price at TimeLimit with no incumbent -> (None, bound)",
      (order is None and bound >= 0) or sols > 0, "status %d, SolCount %d, order %s, bound %.4g"
      % (status, sols, "None" if order is None else "found", bound))

print("\n%d checks failed: %s" % (len(FAIL), FAIL))

sys.exit(1 if FAIL else 0)
