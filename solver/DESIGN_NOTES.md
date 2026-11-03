# solver/: design

This package computes the distance from observed choice probabilities to the random utility
model (RUM), with a certified bracket, tests RUM by a bootstrap built on that distance, and runs
the timed races reported in the paper. This note
explains how it is built and why. How to call it is in the repository's `README.md`; every
setting is in `defaults.py`; `docs/solver.html` (open it in a browser) is an illustrated
tour of the same material, and `docs/bootstrap.html` one of the bootstrap.

Measurements quoted below were taken on an RTX 5090 (32 GB) at the default tolerances unless
stated otherwise.

## The problem

The data are rows (menu D, chosen alternative x, observed probability p̂(x | D)) over n
alternatives, with each menu a bitmask. The distance is

    dist(p̂) = min ||q - p̂||   over the choice probabilities q of random utility models,

taken on the observed rows; with a weight matrix `vcov`, ||v||² = v' vcov v. A menu need not have a
row for every member. A singleton menu's RUM probability is 1, so its rows are taken out before
solving and add (1 - p̂)² to dist² as a constant.

Every certified answer is a bracket lb <= dist <= ub. The upper bound is the distance of an
explicit RUM point (`p_fit`); the lower bound comes from a duality argument. Both ends are valid
whatever the reason a run stopped.

## Two entry points, one job each

| job | entry point | returns |
|---|---|---|
| a certified distance | `RUM(...).calculate_distance(...)` in `rum_solver.py` | a `Result`: the bracket, the fitted point, the method, times, a trace |
| a test of RUM: the bootstrap p-value and confidence interval | `RUM(...).bootstrap(panel=...)` or `bootstrap(replies=...)`, code in `inference/` | a `Bootstrap`; its `p_value()`, `confidence_interval()` and `distance_interval()` return `PValue`, `Interval` and `DistanceInterval` |
| the paper's timing race to a known optimum | `solver.solve(n, cd, method, opt_ref=OPT)` in `__init__.py`, or `calculate_distance(opt=OPT)` | the race record: ub, iterations, the time to ub <= 1.01 OPT |

The race code is frozen: it reproduces the paper's stored race results bit for bit, and
`calculate_distance(opt=...)` calls it with the arguments the paper's scripts pass (same
iterations and ub on every cell checked). Everything below about certificates concerns the class.

## Layout

| path | content |
|---|---|
| `rum_solver.py` | the RUM class: input checks, method choice, memory planning, the runners, `Result`; `Session`, the bootstrap's solver |
| `inference/` | the bootstrap: `hong_li.py` (the formulas, numpy only), `draws.py` (the two inputs, panel and replies, and the draws made from them), `bootstrap.py` (the steps, the results, the checkpoint) |
| `defaults.py` | every user-adjustable default: tolerances, time limit, each method's settings, the automatic-choice thresholds |
| `machine.py` | detects the GPU, Gurobi, CPU and RAM; `machine.setup()` saves a profile to `~/.rum/machine.json` |
| `params.py` | the constants of the paper's runs (BM-ADMM's over-relaxation, preconditioner and penalty rule, Flow-ADMM's defaults) |
| `__init__.py` | `solve()`, the race entry point, and `warmup()` |
| `bm_admm/` | BM-ADMM: the GPU engine, its kernels and butterflies |
| `flow_admm/` | Flow-ADMM |
| `flow_cg/` | CGBFQP-GPU: the column-generation loop, the pricing kernels, the master's Gram matrix |
| `ip_cg/` | CGIP and CGIP-heuristic (Gurobi) |
| `exact_qp/` | the exact QPs bm_qp and flow_qp (Gurobi) |
| `common/` | shared pieces: the slab transforms and the push (`transforms.py`), the arc-flow kernels, the CPU certificate (`certificate.py`), the butterflies, the active-set QP, the Gurobi environment |
| `rum.py` | the previous RUM class, kept while the cluster scripts still call it |

## Methods

| method | formulation and algorithm | runs on | paper name | old names |
|---|---|---|---|---|
| `bm_admm` | Block-Marschak (K) formulation, ADMM | GPU | BM-ADMM | |
| `flow_admm` | arc-flow formulation, ADMM with a dense m_p x m_p matrix | GPU | Flow-ADMM | |
| `flow_cg` | column generation over rankings, Bellman-Ford pricing | GPU, master on the CPU | CGBFQP-GPU | `cgbfqp`, `cgbf` |
| `ip_cg` | column generation, exact IP pricing | Gurobi | CGIP | `cgip` |
| `ip_cg_heur` | column generation, random-insertion pricing certified by the IP | Gurobi | CGIP-heuristic | `cgip_heur`, `cgipheuristic_rand` |
| `bm_qp` | the Block-Marschak QP, solved directly | Gurobi, n <= 15 | | |
| `flow_qp` | the arc-flow QP, solved directly | Gurobi, n <= 15 | | |

Names follow `<formulation or pricing oracle>_<algorithm>`; the old names are accepted as aliases.

**BM-ADMM.** The variables are one value per (alternative i, menu S containing i), stored as a slab
of n 2^(n-1) entries with i's block at i 2^(n-1) + compress(S \ {i}); the Block-Marschak terms are
the superset Möbius transform of the slab. The ADMM is matrix-free: an iteration is a few fused
elementwise kernels, one prox launch over all menus, and the Möbius and eigen butterflies, whose
stage plan is timed on the GPU in use, once per process. The prox is exact for every menu, including menus where only
some members are observed (their members have different curvatures beta + 2 w_i). For n <= 22 an
iteration replays as one CUDA graph. Settings: over-relaxation 1.95, a sqrt(2) diagonal
preconditioner, and the penalty beta = 0.033 m_p^0.51, clipped to [0.3, 100]. The penalty is the
only setting that must follow the instance: iteration counts are U-shaped in beta with a basin
about a factor 3 wide, and the rule is a least-squares fit to the best of a three-point grid on the
paper's 30 instances (n = 18-20, pairwise to 10 % density).

**Flow-ADMM.** The unknown is a unit flow r from the empty set to the full set on the subset
lattice (n 2^(n-1) arcs, the slab's layout), with A r = b, r >= 0, and choice probabilities
y = zeta(r) on the observed rows. The ADMM needs Mhat = C H0^-1 C' (m_p x m_p) and the Cholesky
factor of its Schur complement, in float64: in float32 the solve diverges. The rest runs in float32
kernels. Mhat depends only on the menus and two penalties, so the class keeps it on the RUM object
(up to 1/8 of RAM) and later calls skip its build.

**CGBFQP-GPU (`flow_cg`).** dist² is the minimum over mixtures lambda of ranking columns of
||V lambda - p̂||², where a ranking's column is 1 on each observed row whose alternative that
ranking puts first in the menu. Each iteration solves the restricted master over the simplex with
an active-set Cholesky on Q = V'V and h = V'p̂ (kept up to date from packed 1-bit columns, so V is
never formed), then prices: the best new ranking is a longest path through the subset lattice,
computed one cardinality layer at a time on the GPU. The pricing has two layouts: `mask` keeps one
value per subset (2^n of them, up to n = 32); `stream` keeps only two adjacent layers, addressed by
their colex rank, which takes n to 35 on a 32 GB card.

**CGIP and CGIP-heuristic.** The same column generation with Gurobi: the pricing problem as an
integer program, or random-insertion rankings with the IP called when the heuristic stops
improving. The master is the native active-set solver by default; Gurobi's barrier is optional.

**Exact QPs.** For small n, the whole QP in Gurobi: bm_qp over the Block-Marschak terms, flow_qp
over the arc flow. They build n 2^(n-1) variables, so they are refused above n = 15 and chosen
automatically only up to n = 10.

## Calling the class

`calculate_distance(abs_tol, rel_tol, method, method_args, max_time, p, vcov, seed, verbose,
machine, opt, opt_rel_tol, opt_floor)`:

- `method=None` chooses automatically (below). A named method that cannot run on this machine
  raises before solving, with the reason.
- `method_args` may only use the names listed for that method in `defaults.METHOD_ARGS`; any other
  name raises.
- `p` replaces the stored probabilities for one call (a bootstrap draw, for example). Gurobi's
  models and Flow-ADMM's Mhat depend only on the menus and are reused across such calls.
- `seed` (default 0) seeds the random first columns; `None` draws a seed and reports it.

Each method has two functions in `rum_solver.py`. The planner checks the machine and the
arguments, estimates device memory against 90 % of the free GPU memory, and resolves every "auto"
setting before anything is solved. The runner calls the engine and records the bracket as it
improves, so an interrupt or an error still leaves the best bracket found.

**Stopping.** A call stops as soon as ub - lb <= abs_tol or (ub - lb)/lb <= rel_tol (the relative
test needs lb > 0), when max_time runs out (setup included), or on the method's own limits.
Defaults: abs_tol 1e-5, rel_tol 1e-4, one hour.

| exit_flag | meaning |
|---|---|
| 3, 1, 2 | both tolerances met, abs_tol met, rel_tol met |
| 4 | race mode: the target was reached |
| 5 | a session's threshold was reached: ub < below or lb >= above (see Bootstrap inference) |
| 0 | max_time reached |
| -1 | the method's iteration limit |
| -2 | stalled: the method stopped without meeting the tolerances (the message says why) |
| -3, -4, -5 | out of memory, interrupted, solver error (the message holds the exception) |

**Result.** The bracket (`lb`, `ub`, `distance` = ub, the gaps and which tolerance holds), the
method and the reason for an automatic choice, `method_args` as resolved, iterations and their
unit, time (setup and solve), the seed, a line about the machine, `p_fit` (input row order), a
trace of (iteration, seconds, ub, lb) rows whose last row is the final bracket, and `info`
(method-specific detail, such as the GPU bytes planned or the pricing backend).

## Automatic method choice

With a GPU and the plain norm, the choice is bm_admm or flow_cg, from n and the density (observed
cells / n 2^(n-1)): bm_admm from 2 % density at n = 11-13, from 1 % at n = 14-17 and from 0.5 %
at n >= 18; flow_cg below that, at every density for n <= 10, and whenever bm_admm does not fit
the GPU. Without a GPU, or with `vcov`, it is flow_qp up to n = 10 and ip_cg_heur above.
flow_admm is never chosen: it was dominated in every measured cell, and its dense matrices need
about 55 m_p² bytes.

The split follows the two cost models. A BM-ADMM iteration costs the same whatever the number of
observations, and dense designs need few iterations; CGBFQP-GPU pays per observation, and sparse
designs need few columns. The crossover, measured per n:

- n <= 10: flow_cg at every density. It took at most 0.66 s on every design, at most 0.55 s more
  than bm_admm (n = 10, every menu: 0.66 s against 0.11 s), while bm_admm needed 114,350
  iterations (4.1 s) on pairwise n = 8.
- n = 11-13 (seconds, flow_cg against bm_admm):

  | n | pairwise | 2 % | 10 % | every menu |
  |---|---|---|---|---|
  | 11 | 0.03 / 0.71 | 0.10 / 0.13 | 0.19 / 0.05 | 2.00 / 0.14 |
  | 12 | 0.04 / 0.70 | 0.12 / 0.11 | 0.66 / 0.07 | 10.2 / 0.14 |
  | 13 | 0.04 / 14.9 | 0.33 / 0.15 | 1.71 / 0.09 | 32.5 / 0.16 |

  The 2 % threshold picks the faster method in all 21 cells measured at n = 11-13.
- n = 13-17 (60 s cap): flow_cg is faster on pairwise and ~0.5 % designs, bm_admm on ~2 % and
  ~10 % designs, at every n. At n = 17: at 0.5 %, flow_cg 4.8 s against bm_admm 11 s; at 2 %,
  bm_admm 6.8 s against flow_cg 29 s; at 10 %, bm_admm 3.9 s against more than 60 s.
- n = 18-20: the crossover lies at 0.3-0.5 % density at all three n, so the rule is stated in
  density (the number of observations at the crossover doubles with n).

**Why sparse designs go to flow_cg.** On pairwise designs (n = 8, 10, 13) the BM-ADMM iterate
itself converges fast: its distance is within 1e-4 of the optimum, and its lower bound close, after
450-2,000 iterations. More than 99 % of the iterations then wait for the pushed upper bound, whose
error follows the negative Block-Marschak terms of the iterate on the unobserved part of the
lattice, which ADMM removes slowly. A float64 push gives the same bound. A polished upper bound (the
exact QP over the 500 widest paths of the pushed flow) reaches the optimum early on those small
designs, but not on the Dean-Stoye-Ravindran data (n = 25, 301 observations), where the iterate
itself is far off; the density rule already sends such designs to flow_cg, so it was not adopted.

**Without a GPU.** flow_qp rather than bm_qp: on the n = 6-10 grid flow_qp was faster on every
design but the full one (every menu observed), and at n = 10 its certified bracket was 15-200x
narrower. On the full design bm_qp is a little better (n = 10, one Gurobi thread: 1.34 s against
1.86 s). From n = 10 on, flow_cg is faster than both exact QPs on every design.

## The certificate

**Upper bound: a feasible point.**

- bm_admm, flow_admm: every `check` iterations (default 50), the iterate is pushed to an exactly
  feasible RUM point: Möbius transform to the arc flow, then clip the negative flows and push unit
  mass forward through the lattice one layer at a time, in proportion, then zeta. The push runs in
  float64 on the GPU, so the point conserves flow to float64 rounding; ub is its distance, and the
  best point found is kept.
- flow_cg: the master's mixture of rankings. The engine works with p̂ rounded to float32, so the
  class recomputes the mixture's distance exactly in float64.
- ip_cg, ip_cg_heur: the master's mixture of rankings, whose distance the master computes in
  float64 on the CPU.
- bm_qp, flow_qp: Gurobi's point meets the constraints only to Gurobi's tolerances, and its
  objective is an estimate, not a bound (on the n = 6-13 grid it was above flow_cg's certified
  upper bound in more than half the runs). So the class pushes Gurobi's arc flow (flow_qp: r;
  bm_qp: K p̃) to an exact unit flow on the CPU and takes the distance of that point;
  `info["gurobi_distance"]` keeps Gurobi's own value.

**Lower bound: Frank-Wolfe.** For any point y, convexity gives f(q) >= f(y) + ∇f(y)'(q - y) for
every RUM point q. With r = p̂ - y,

    dist² >= ||r||² - 2 rc(y),    rc(y) = max over rankings a of r'a - r'y,

and the maximum over rankings is a longest path through the subset lattice, the same computation
as pricing (with `vcov`, W r takes the place of r). The bound holds at any y, so each method
evaluates it wherever it is cheap:

- bm_admm, flow_admm: at the iterate's choice probabilities every `lb_every` checks, in float64 on
  the GPU (subset sums by a butterfly over the slab, then one kernel launch per lattice layer).
  One bound takes 1.3 ms at n = 20 and 22 ms at n = 24.
- flow_cg: the engine's own bound comes from its pricing sweep, computed for p̂ rounded to float32.
  At the end the class recomputes the bound in float64 at the mixture's point (see Storage
  formats). The engine is asked for abs_tol - 2 delta, with delta = ||p̂ - float32(p̂)||; if the
  float64 bracket then misses both tolerances after the engine stopped on its gap, the class
  resumes once from the engine's own columns with both tolerances tightened by the miss (1.5x).
- ip_cg, ip_cg_heur: from the pricing IP's bound, Gurobi's ObjBound, which stays valid when the IP
  stops early; the IP is re-solved at MIPGap 0 only when its own gap blocks a stop.
- bm_qp, flow_qp: the bound at Gurobi's point, on the CPU, weighted norms included. On the n = 6-12
  grid every bracket was valid; flow_qp's was at most 8e-8 wide.

## Storage formats and memory

Each GPU method keeps its working values in one format, set by `method_args["dtype"]`. Without it,
the class takes the most precise format that fits 90 % of the free GPU memory. The tolerance never
lowers precision: a format that cannot reach the tolerance still runs, and the result's message
says so. An explicit dtype that does not fit raises before solving, with the sizes.

| method | dtype a user may pass | without dtype |
|---|---|---|
| bm_admm | float64, float32 | float64. When it does not fit, auto takes flow_cg; an explicit bm_admm raises, naming float32's size |
| flow_cg | float32, uint24, uint16 (the pricing values) | the first that fits: float32 (mask layout, else stream), uint24, uint16 |
| flow_admm | float32 | float32 (its kernels; Mhat and the Schur factor are always float64) |
| Gurobi methods | - | float64 |

uint24 and uint16 are block-scaled integers: each block of 1,024 values stores an offset and a
step, so a stored value is within one quantization step of the computed one; the sweep returns the
resulting margin, which the engine adds to the pricing maximum before it forms its lower bound.
They save memory, not time (uint16 is 3-6 % slower per column than float32). flow_cg's `pricing` sets the layout only (`mask` or
`stream`); the older names `stream_f32`, `stream_u24` and `stream_u16` set both.

**The bounds are not a choice.** Both are float64 wherever a float64 form fits. bm_admm and
flow_admm take the lower bound from the slab. flow_cg's final lower bound takes the first form that
fits, once the pricing buffers are freed: from the slab (16 B per slab entry); else two lattice
layers at a time (16 B per rank of the widest layer); else the pricing sweep's own bound lowered by
delta. That last form does not cover the float32 rounding inside the sweep (for uint24 and uint16,
the margin covers the rounding of the stored values, not the float32 arithmetic), so
`info["lb_rigorous"]` is False and the message says so. `info["lb_dtype"]` and `info["lb_layout"]`
name the form used.

**Where each format fits** (the planners' formulas, 90 % of the free memory):

| format | largest n, 32 GB card | H200 (141 GB) |
|---|---|---|
| BM-ADMM state, float64 / float32 | 23 / 24 | 25 / 26 |
| flow_cg pricing, mask (float32) | 32 | 32 (32-bit menu masks) |
| flow_cg pricing, two layers: float32 / uint24 / uint16 | 34 / 35 / 35 | 36 / 37 / 37 |
| float64 lower bound from the slab | 26 | 29 |
| float64 lower bound, two layers at a time | 33 | 35 |

So on a 32 GB card flow_cg's lower bound is float64 up to n = 33 and comes from the pricing sweep
only at n = 34-35. At n = 35 auto takes uint24 when 28.6 GiB are free and uint16 otherwise; uint16
is only 2 % faster there (708 s against 721 s, the same 103 columns).

**Memory models.** Each planner compares a measured model, plus 10 %, with 90 % of the free memory,
so a run never relies on spilling to host memory:

- BM-ADMM: 145 B per slab entry in float64 with CUDA-graph replay, 141 B without (92 / 82 B in
  float32): 1.6 GB at n = 20, 13.9 GB at n = 23, 29.1 GB at n = 24.
- Flow-ADMM: 55 B per m_p² (Mhat, the Schur matrix, its factor and workspace) + 82 B per slab entry
  + 0.25 GB.
- flow_cg, mask pricing: 5 B per subset + 4 B per subset for the stored layers up to n = 28, else
  4 B per rank of the widest layer: 5.6 GB at n = 30, 22 GB at n = 32. Two-layer pricing: 2 x
  (4, 3 or 2 B) per rank of the widest layer, plus 1 B per predecessor for the top 10 layers. On
  the host, the master's Gram matrix takes 8 K² bytes for K = max_iters + 2 columns.

**Choices not made, and why.**

- Auto never picks a float32 BM-ADMM state. It does not certify: on n18_m39 at rel 1e-4, float64
  certifies in 22.5 s, while float32 is still 2.6 % above its lower bound after 120 s. In the
  races, the float32 leg stalled before 1 % of the optimum in 88 of 90 cells. It would only matter
  at one n per card (24 on 32 GB, 26 on the H200), where flow_cg fits.
- No rounded-up float32 bound for n = 34 on 32 GB: it would cover one n on one card (the H200 does
  n = 34 in float64, two layers at a time), and its looseness is unmeasured.
- No float64 pricing: with the final bound in float64, pricing precision only changes which column
  is picked, because the master recomputes each column's reduced cost in float64. It would double
  the memory, at 1/64 of the single-precision rate on GeForce cards.
- No float64 Flow-ADMM state: it is never chosen automatically, and every kernel of its loop would
  need a float64 version.

## Bootstrap inference

`rum.bootstrap(panel=...)` (or `replies=...`) tests H0: dist = 0 (a RUM generated the data) by Hong
and Li's numerical delta method, and gives the one-sided confidence interval for dist. The formulas
follow Hong and Li; here every number comes from certified brackets. `docs/bootstrap.html`
illustrates this section.

**The method.** With n the sample size (the input below), r_n = √n and ε_n = n^(-1/3) (both can
be set) and s = 1 + ε_n r_n,

    P_b = p̂ + ε_n r_n (p̂_b - p̂)                      the b-th perturbed point, b = 1..B
    p   = #{b : dist(P_b) >= s dist(p̂)} / B            the p-value
    CI  = [dist(p̂) - q / r_n, inf)                     q = the (1 - alpha)-quantile of (dist(P_b) - dist(p̂)) / ε_n

The solver gives dist(p̂) in [L0, U0] and dist(P_b) in [L_b, U_b], so the threshold becomes the
band tau = s [L0, U0]. A draw is below if U_b < tau_lo (it counts 0), above if L_b >= tau_hi (it
counts 1), and undecided otherwise (it counts 1, which keeps p conservative). The reported p is
(#above + #undecided) / B, an upper bound on the exact p; #above / B is the other end of its
bracket. The reported lower end of the interval uses L0 and the U_b, so it is conservative; the
exact lower end lies in a bracket whose other end uses U0 and the L_b.

**The input.** The draws redraw the data, so they need the choices before counting: one row per
choice, as one of two tables, picked by how the data were collected.

| input | columns | the data | resampled | n |
|---|---|---|---|---|
| `panel=` (DSR style) | respondent, menu, chosen | people who each answered several menus | respondents, each with all their rows | the number of respondents (the default) |
| `replies=` (AI style) | menu, chosen | replies collected one by one (an LLM asked each menu many times) | each menu's replies, on their own | the replies per menu (default: the most on any menu) |

| column | content | DSR rows |
|---|---|---|
| respondent | who chose (an integer or a string); a respondent's rows are resampled together | 0, 0, 0 |
| menu | the menu offered, as a bitmask (bit a set = alternative a offered), the encoding of `RUM(menu, ...)` | 2081 (= {0, 5, 11}), 1572865, 4195329 |
| chosen | the alternative picked, a member of its menu | 5, 20, 0 |

A pandas DataFrame or a dict of columns; other columns are ignored. `cells_from_panel(table,
num_alternatives, alternatives=None)` counts rum's rows from either table: p̂(D, a) is the share of menu D's rows
that chose a (DSR alt0 keeps alternative 0 only: `alternatives=[0]`). The bootstrap raises if
rum's p̂ is not the table's counts, or if a given n does not fit it (a panel: the number of
respondents; replies: between the fewest and the most replies on a menu); without n it takes the
number of respondents, or the most replies on any menu. p̂ alone cannot give the draws: it
does not say how often each menu was seen, nor which choices came from one respondent. On DSR the
menus were seen 6 to 520 times (median 15); treating every menu as seen n = 520 times makes the
draws about six times too tight (median ||p̂_b - p̂|| 0.33, against 2.03 from the panel).

The AI data are replies: each of the 8,807 menus was put to the model 100 times, each time in a
separate call with its own random display order (gpt-5.6-luna: 100 replies on every menu;
deepseek-v4-flash: 94 to 100, after 165 unreadable replies are dropped). The paper's runs pass
them as a panel with the replicate number as the respondent (`replication/empirical_ai_n18/replies.py`).
That is valid, since the 100 replicates are independent copies of the whole design, but within a
draw it ties all 8,807 menus to the same 100 picks. `replies=` resamples each menu on its own, as
the data were made. Each menu's spread is the same either way (identical on gpt-5.6-luna, where
every replicate answered every menu once), but ||p̂_b - p̂|| varies 13 to 16 times more from draw
to draw with the replicate grouping (gpt-5.6-luna: mean 6.86 and 6.87, standard deviation 0.49
and 0.038). Both inputs give p = 0. A panel must never be passed as replies: a person's answers
are linked.

**The draws.** Draw b gives respondent i a weight w_bi and counts again,
p̂_b(D, a) = Σ_i w_bi #{i's rows: menu D, chosen a} / Σ_i w_bi #{i's rows: menu D}. The default
weights, multinomial_reject, are multinomial counts of n picks with replacement; a resample that
leaves a menu unseen is drawn again (DSR: 72 redraws for 10,000 draws; AI: none). The weights are
drawn once, from `np.random.default_rng(seed)` in a fixed order, so the draws
rebuild bit for bit. The B x M draws are rebuilt from the weights 128 MB at
a time and never held whole (the AI's 10,000 x 73,492 would be 5.9 GB). With integer weights a
row's weighted count of views is its menu's, so it is computed once per menu (8,807 on AI, not
73,492 rows); with 2,000 respondents that keeps a full pass at minutes. Other weights:
multinomial, exponential, poisson, mammen. With replies, draw b redraws N_D replies with
replacement for each menu D on its own: menu D's counts are Multinomial(N_D, D's shares). Block k
of the draws comes from `np.random.default_rng([seed, k])`, so a block is rebuilt alone, and its
size is set by M. A full pass over the AI's 10,000 draws takes 28 s (gpt-5.6-luna) and 39 s (deepseek-v4-flash) on one core, against about 9 s
for the panel's (one weight matrix product per block).

**The steps.** `p_value(alpha=0.05)` runs, on one session (below):

1. certify p̂ to abs_tol, keeping the solver's state and its fitted RUM point p_fit;
2. screen every draw at once: U_b = ||p_fit - P_b|| bounds dist(P_b), since p_fit is a RUM point
   (DSR alt0: 2,378 of 10,000 draws below tau_lo with no solve; AI: all 10,000);
3. solve the other draws warm from p̂, each stopping once its bracket clears the band;
4. refine, up to `refine` times (default 2) at a tenfold tighter tolerance: re-certify p̂ if its
   bracket is wider than the new tolerance (it sets the band), then the draws still undecided.

`distance_interval(alpha=0.05)` is the two-sided interval of the paper's application section,

    [max(0, L0 - c), U0 + c]                           c = the (1 - alpha)-quantile of ||p̂_b - p̂||

It holds [max(0, d - c), d + c] for every d = dist(p̂) in [L0, U0], and it is valid because
|dist(p) - dist(p̂)| <= ||p - p̂|| for the distance to any convex set (in the run's norm, so with
`vcov` as well). The screen computes every draw's ||p̂_b - p̂|| as it passes over the draws, so the
interval solves nothing; called first, it makes that pass itself.

`confidence_interval(alpha=0.05)` reuses every bracket. The quantile of the draws' distances lies
in [Q(L), Q(U)], the quantiles of the lower and upper bounds; only a draw whose bracket overlaps
that band can move it. Those draws are solved to tolerance without stops, largest U_b first, in
batches of alpha B / 4, until the lower end's bracket is at most 2 abs_tol wide. Decisions come
from the brackets, never from an engine's stop reason. `p_value(stop_when_decided=True)` ends as
soon as the decision at alpha cannot change, with p known only to its bracket.

**Session.** `rum.session(method, method_args, vcov)` is the bootstrap's solver. It plans the
method, its storage format and its GPU budget once, as calculate_distance plans them, so the
certificate and every draw share method and dtype. `solve(p, abs_tol, rel_tol, max_time, below,
above, warm, keep)` takes any finite p: calculate_distance requires p in [0, 1], and every DSR P_b
leaves it (10,000 of 10,000 draws, alt0 and full). `below` and `above` end a solve once ub < below
or lb >= above (exit flag 5); `keep` stores the solve's state and p_fit (`s.fit`); `warm` starts
from them.

| method | below, above | warm |
|---|---|---|
| bm_admm | the engine's ub and lb targets | the ADMM iterate (on the host above n = 22) |
| flow_cg | the engine's stops, moved inward by 2 delta for float32 p̂; the float64 bounds confirm the decision | the kept columns and their weights |
| flow_admm | tested at each push and lower bound | - |
| ip_cg, ip_cg_heur | the pricing loop's stops | - |
| bm_qp, flow_qp | none (one Gurobi solve) | - |

calculate_distance does not use a session.

**The applications** (B = 10,000 multinomial_reject draws, seed 0, abs 5e-4, rel 0, refine 2;
`replication/empirical_dsr_n25/run_bootstrap.py`, `replication/empirical_ai_n18/run_bootstrap.py`):

| data | n | M | method | p | below / above / undecided | p time | 95 % interval | interval time |
|---|---|---|---|---|---|---|---|---|
| DSR alt0 | 520 | 301 | flow_cg, float32 (auto) | 0.1083 | 8,917 / 1,083 / 0 | 688 s | [-0.0476, inf) | 237 s, 592 draws |
| DSR full | 520 | 901 | flow_cg, float32 (auto) | 0.1392 | 8,608 / 1,392 / 0 | 1,499 s | [-0.0902, inf) | 680 s, 649 draws |
| AI gpt-5.6-luna | 100 | 73,492 | bm_admm, beta 5, check 25 | 0 | 10,000 / 0 / 0 | 21.9 s | not computed | |
| AI deepseek-v4-flash | 100 | 73,492 | bm_admm, beta 5, check 25 | 0 | 10,000 / 0 / 0 | 21.5 s | not computed | |
| AI gpt-6 luna, no reasoning | 2,000 | 73,492 | bm_admm, beta 5, check 25 | 0 | 10,000 / 0 / 0 | 110.5 s | not computed | |
| AI GMI Cloud | 2,000 | 73,492 | bm_admm, beta 5, check 25 | 0 | 10,000 / 0 / 0 | 130.9 s | not computed | |

These are the p-values the paper reports; every AI draw is below in every run. Without refine,
deciding every DSR draw takes three complete runs (abs 5e-4, 1e-4, 1e-5); refine settles the
last draws within one run. Both DSR intervals contain
0, consistent with p > 0.05. They were cross-checked by solving the 1,000 draws with the largest
upper bounds again to abs 5e-6: every tight bracket overlaps the stored one, the quantile is pinned
to 1e-7, and the exact lower end lies inside the reported bracket, within 1e-6 of the reported
value. The AI interval needs certified lower bounds on the upper tail: on
gpt-5.6-luna, 80 draws solved to abs 5e-4 took 5-13 s each (median 7.3 s), so the about 630 draws
it needs would take about 1.3 h.

**Checkpoint.** With `checkpoint=path`, every bracket is saved every 60 s and at the end of each
call; a new bootstrap given the same path resumes once settings, data and input match, and an
interrupt (exit flag -4) saves before it raises. `boot.save(path)` writes the record: settings
(with `input`, panel or replies), fingerprints (data, panel or replies, draws), the plan, the p̂ certificates, every draw's bracket and exit
flag, the results and the times.

## The race path

`solver.solve(n, cd, method, opt_ref=OPT)` runs one named method until its feasible upper bound is
within 1 % of a certified optimum: ub <= 1.01 OPT, tested every `check` iterations for the ADMMs
and after every column for CGBFQP-GPU. When OPT is (near) 0 the target is max(1.01 OPT, floor);
the paper uses a floor of 1e-3 in the lambda study and 1e-5 on the convex path. It refuses
`criterion="cert"` and `method="auto"`: certified distances come from the class. The race keeps
the paper's settings:

- BM-ADMM runs a float32 sprint until its upper bound stops improving, then continues in float64
  from the same state; a stop reached in float32 is confirmed by a push from the float64-cast
  iterate. Its push runs in float32.
- CGBFQP-GPU uses mask pricing where it fits, takes its first column from seed 0, and stops at
  4,000 columns unless a script raises the cap.
- Flow-ADMM caches Mhat on disk (`data/cache_mhat`).

`calculate_distance(opt=...)` runs the same race with `run_one.py`'s arguments; there it computes
no lower bound (`lb` is NaN) and takes no abs_tol or rel_tol. The validation drivers
(`validation/run_one.py`, `validation/feasibility/feas_one.py`) take races from `solve()` and
certificates from the class.

## Randomness

Only flow_cg and the two CGIP methods draw random numbers: the first column (one random ranking),
and CGIP-heuristic's random insertions. They change the path (iterations, time, the mixture found),
never the validity of the bracket. The ADMMs and the exact QPs draw none, and Gurobi's seed stays at
its default. No method touches NumPy's global random state: flow_cg's first ranking comes from
`np.random.RandomState(seed)`, the permutation the global `np.random.seed(seed)` used to give. The
bootstrap draws its respondent weights once, from `np.random.default_rng(seed)`, and the replies'
draws block by block, block k from `np.random.default_rng([seed, k])`; a session's solves
take the session's seed (default 0), and a warm flow_cg solve starts from the kept columns instead
of a random first ranking.

## Gurobi settings

- The restricted master in Gurobi: a ridge of 1e-8 x max diag(Q), NonConvex = 0, PSDTol = 1e-3.
- Pricing IP: the lower bound uses ObjBound, not ObjVal. With every IP stopped at a 5 % gap, a
  bound from ObjVal certified a wrong value (1.3e-3 above the exact distance on pairwise n = 10).
- The class runs CGIP with `strict=True`: a repeat is tested on the column, not the ranking; a
  repeated column ends the solve as "repeat" and is never added; the relative test needs lb > 0;
  and the native master is warm-started with a cap of 2K + 200 steps. Near distance 0 this matters:
  Gurobi's barrier solves the master to about 1e-7 on the squared distance, so columns already in
  the master keep a positive reduced cost and the pricing IP returns them again. The native master
  certifies such a case at 2e-13 in 10 columns; with the Gurobi master the solve stops honestly as
  "repeat".
- `time_limit` is checked before each pricing step and passed to the pricing IP as Gurobi's
  TimeLimit; an IP stopped before any incumbent returns its bound.

## Compatibility

- The old method names work as aliases; stored result files and the replication drivers keep their
  old tags.
- Some names are kept because code outside `solver/` wraps or patches them: CGIP's
  `CGIPBuild.ip_price`, `rand_price`, `heuristic_price`, `column` and `_pos`, `cgip.solve_rmp`,
  `cgip._solve_rmp`, `build_cgip*` and `solve_qp.solve_qp_gurobi` (the cluster scripts of the
  paper's IP-CG runs); `bm_admm.push_to_ub`; and `cg_core.selftest`
  (`validation/feasibility/CONSTRUCTION.md`).

## Known limitations

- flow_cg at n = 34-35 on a 32 GB card: no float64 form of the lower bound fits, so it comes from
  the pricing sweep, with `lb_rigorous = False`.
- The bootstrap's one-sided interval costs a solve to tolerance for every upper-tail draw that can
  move its quantile: 592 and 649 draws on DSR (4 and 11 min), about 630 on the AI data (about 1.3 h
  each, estimated). The p-value is cheaper wherever the screen decides most draws; the two-sided
  `distance_interval()` solves nothing.
- The replies' draws depend on the block size, which follows M and `BLOCK_BYTES` in `draws.py`:
  changing that constant changes them (the panel's do not). The record's draws fingerprint pins
  them. Replies take multinomial resampling only.
- `vcov` is accepted by the session and the bootstrap (the screen uses the same norm), but no
  application uses one. On a toy panel with flow_qp, the identity gives the unweighted p-value,
  decisions and interval exactly, and a diagonal weight keeps every lower bound below its screen
  bound.
- Flow-ADMM converges slowly at tight tolerances: at rel 1e-4, n18_sparse is still at 2.2e-4 after
  600 s (at rel 1e-3 it certifies n18_sparse, n20_sparse and n18_m39 in 2, 26 and 7 s).
- flow_cg and bm_admm rebuild their GPU setup on every call; only Gurobi's models and Flow-ADMM's
  Mhat are cached on the RUM object. The rebuild takes at most 0.24 s for flow_cg (n = 35) and
  7 ms for bm_admm (n <= 13).
- `push_to_ub(check=True)` reports a meaningless conservation residual (it passes the float64 flow
  to a float32 kernel). The upper bound itself is right, and nothing in the repository calls it
  with check=True.
- Calling CGIP directly on data with singleton menus raises a KeyError while it builds the pricing
  model. The class removes singleton rows first, so it never gets there.
- `rum.py`, the previous class, stays until the cluster scripts move to the new one.
