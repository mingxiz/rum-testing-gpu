"""The one Gurobi environment (licence check-out) shared by every Gurobi model in the package:
CGIP's pricing IP (solver/ip_cg), the Gurobi master QP (common/qp_gurobi.py) and the exact QPs
(solver/exact_qp). It is started on first import, so importing this module needs gurobipy and a
licence. Parameters set on it, e.g. _ENV.setParam("Threads", 16), apply to models built after."""
import gurobipy as gp

# One Gurobi env/license shared across all solves (avoids per-solve overhead).
_ENV = gp.Env(empty=True)
_ENV.setParam("OutputFlag", 0)
_ENV.start()
