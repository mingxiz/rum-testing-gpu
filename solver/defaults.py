"""Default settings of the RUM class (solver/rum_solver.py). Edit here, or override per call:

    rum.calculate_distance(abs_tol=1e-6, method="bm_admm", method_args={"check": 100})

A call stops as soon as UB - LB <= abs_tol OR (UB - LB) / LB <= rel_tol, or when max_time
(seconds, setup included) runs out. method_args may only use the names listed for that method
below; anything else is an error. Given opt, a call is the paper's race instead and stops at the
OPT_REL_TOL / OPT_FLOOR target (no lower bound).

Storage. Each GPU method takes dtype in method_args (bm_admm: float64, float32; flow_cg: float32,
uint24, uint16; flow_admm: float32). "auto" takes the most precise format that fits the GPU: the
tolerance never lowers precision, and a format that cannot reach it still runs and says so in the
result's message. An explicit dtype that does not fit raises, with the sizes. The bounds are not a
choice: float64 wherever a float64 form fits (info lb_dtype, lb_layout; lb_rigorous False when the
lower bound had to come from the pricing sweep). Races keep their own settings.
"""
import math

ABS_TOL = 1e-5        # absolute gap on the distance
REL_TOL = 1e-4        # relative gap (UB - LB) / LB
MAX_TIME = 3600.0     # hard stop in seconds, setup included
SEED = 0              # seed of the random first columns (flow_cg, ip_cg, ip_cg_heur); None = fresh
OPT_REL_TOL = 1e-2    # race (calculate_distance with opt): stop at UB <= max((1 + OPT_REL_TOL) opt, OPT_FLOOR)
OPT_FLOOR = 0.0       # race: absolute floor of that target, needed when opt = 0 (the paper: 1e-3 in the
                      # lambda study, 1e-5 on the convex path)

METHOD_ARGS = {
    "bm_admm": dict(            # Block-Marschak ADMM, GPU
        beta=None,              # penalty; None = 0.033 * mp^0.51 (params.bm_beta_default)
        alpha=1.95,             # over-relaxation
        precond_p=math.sqrt(2.0),
        check=50,               # iterations between upper bounds (feasibility push)
        lb_every=1,             # upper-bound checks between lower bounds (the LB is a GPU sweep)
        dtype="auto",           # "float64" (auto, never float32); "float32" halves the memory, but its pushed
                                # upper bound stalls, so it rarely certifies (a valid bracket, often wide)
        use_graph="auto",       # replay one iteration as a CUDA graph; auto = n <= 22
        max_iter=10 ** 7,
    ),
    "flow_admm": dict(          # arc-flow ADMM with a dense m_p x m_p matrix, GPU (never chosen automatically)
        dtype="auto",           # "float32" (auto; its ADMM kernels; the dense Mhat and Schur factor are float64)
        beta2=2.0,              # penalties of the flow and the bound constraints (start values)
        beta3=0.05,
        alpha=1.8,              # over-relaxation
        adaptive=True,          # rebalance the two penalties every 25 iterations
        check=50,               # iterations between upper bounds (feasibility push)
        lb_every=1,             # upper-bound checks between lower bounds (the LB is a GPU sweep)
        max_iter=10 ** 7,
    ),
    "flow_cg": dict(            # column generation, Bellman-Ford pricing on the GPU
        dtype="auto",           # pricing values: "float32", "uint24" or "uint16" (block-scaled); auto = the first
                                # that fits, in that order. They save memory, not time (uint16 is 3-6 % slower)
        pricing="auto",         # layout: "mask" (one value per subset, 5 * 2^n bytes, n <= 32) or "stream" (two
                                # live layers); auto = mask where it fits. Old names stream_f32/_u24/_u16 work
        layers="auto",          # mask pricing: stored lattice layers for n <= 28, streamed above
        max_iters=4000,         # columns; also sizes the host Gram matrix (max_iters^2 doubles)
    ),
    "ip_cg": dict(              # column generation, exact Gurobi IP pricing (CPU)
        qp_backend="native",    # master QP: "native" (active set, exact) or "gurobi" (barrier, solves
                                # to ~1e-7 and so cannot certify near distance 0)
        threads=None,           # Gurobi threads; None = Gurobi's default (all cores)
        max_iters=100000,       # columns
    ),
    "ip_cg_heur": dict(         # random-insertion pricing, certified by the IP when it stalls
        qp_backend="native",
        threads=None,
        max_iters=100000,
        n_restart=1,            # random constructions per pricing call
        tol=1e-6,               # heuristic gain below this triggers an IP certification
    ),
    "bm_qp": dict(              # exact Block-Marschak QP in Gurobi (small n)
        threads=None,
        method=-1,              # Gurobi Method (-1 = automatic)
        mem_limit=None,         # Gurobi MemLimit in GB
    ),
    "flow_qp": dict(            # exact arc-flow QP in Gurobi (small n)
        threads=None,
        method=-1,
        mem_limit=None,
    ),
}

# Automatic choice when method=None (solver/rum_solver.py: choose_method). With a GPU it is bm_admm
# or flow_cg: bm_admm from the density set for its n, flow_cg below it, for n <= 10, and whenever
# bm_admm does not fit the GPU. Measured on an RTX 5090 at the default tolerances: n = 11-13,
# flow_cg is faster below ~2 % and bm_admm from ~2 % (n = 13, every menu: 0.16 s vs 32.5 s);
# n = 14-17, flow_cg is faster at 0.5 % and bm_admm at 2 %; n = 18-20, the crossover is 0.3-0.5 %.
# n <= 10 is flow_cg at every density: at most 0.55 s slower (n = 10, every menu: 0.66 s vs 0.11 s).
AUTO = dict(
    bm_admm_min_density={11: 0.02, 14: 0.01, 18: 0.005},   # {from n: density (observed cells / n 2^(n-1))}
    exact_qp_max_n=10,          # without a GPU or with vcov: the exact QP (flow_qp) up to this n, CGIP-heuristic
                                # above. flow_qp, not bm_qp: on every n <= 10 design but the full one it is faster,
                                # and at n = 10 its certified bracket is 15-200x narrower
    gpu_memory_fraction=0.9,    # share of free GPU memory a method may plan to use
)

# rum.bootstrap(n, panel, ...): Hong and Li's p-value and confidence interval (solver/inference).
# r_n = sqrt(n) and eps_n = n^(-1/3) unless given; alpha (0.05) is an argument of the two results.
BOOTSTRAP = dict(
    B=10000,                    # draws: the final DSR and AI tables
    weights="multinomial_reject",   # respondent weights for panel= (replies= are resampled menu by menu)
    seed=0,                     # numpy.random.default_rng(seed) draws the weights
    abs_tol=5e-4,               # per solve, p_hat and every draw: the applications' criterion
    rel_tol=0.0,
    refine=2,                   # up to this many tenfold tighter rounds for what is still undecided
    max_time=3600.0,            # seconds for each certificate of p_hat
    draw_time=120.0,            # seconds for each solve of a draw; a draw left undecided counts 1
    checkpoint_every=60.0,      # seconds between checkpoint writes (rum.bootstrap(..., checkpoint=path))
)
