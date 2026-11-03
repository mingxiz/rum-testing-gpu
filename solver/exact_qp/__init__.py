"""Flow-QP and BM-QP: the RUM projection written out as one explicit quadratic program and
solved by Gurobi (CPU). Two formulations, both in flow_bm_qp.py:

    flow_qp   min ||y - p||^2    s.t. y = Z r, A r = b, r >= 0          (arc flow on the subset lattice)
    bm_qp     min ||W p~ - p||^2 s.t. K p~ >= 0, S p~ = 1, p~ >= 0      (Block-Marschak)

They work on a solver.rum.RUM instance, through RUM.calculate_distance(method="flow_qp" or
"bm_qp") or directly:

    from solver.exact_qp import flow_bm_qp
    flow_bm_qp.build_arc_flow(rum)
    out = flow_bm_qp.solve_arc_flow_qp(rum, p)

The matrices have n 2^(n-1) columns, so this is for small n (roughly n <= 15; see
replication/qp_matrix_memory); solver.solve() does not offer it. Import the submodule
explicitly; it needs gurobipy and a licence.
"""
