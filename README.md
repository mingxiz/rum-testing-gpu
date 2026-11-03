# rum-testing-gpu

GPU algorithms for testing discrete-choice random utility models (RUM): the code of the paper *Alternative
Computational Formulations for Testing Discrete-Choice Random Utility Models*.

Given observed choice frequencies p&#770; on a collection of menus, the package computes the distance from p&#770;
to the RUM polytope with a certified bracket, lb &le; dist &le; ub, and runs the bootstrap test of H0: dist = 0
(Hong and Li's numerical delta method) on top of it. `replication/` regenerates every table of the paper's
numerical and application sections, and `final_results/` holds the numbers behind them.

Illustrated overviews (open in a browser): [`docs/index.html`](docs/index.html),
[`docs/solver.html`](docs/solver.html) (the solver), [`docs/bootstrap.html`](docs/bootstrap.html) (the bootstrap)
and [`docs/replication.html`](docs/replication.html) (which script and data give which table).

## Methods

| paper | `method` | runs on | what it is |
|---|---|---|---|
| BM-ADMM | `bm_admm` | GPU | ADMM on the Block-Marschak (half-space) representation, matrix-free |
| Flow-ADMM | `flow_admm` | GPU | ADMM on the network-flow representation, with a prestored dense matrix |
| Flow-CG | `flow_cg` | GPU | column generation with Bellman-Ford pricing on the subset lattice |
| IP-CG | `ip_cg` | CPU, Gurobi | column generation with exact integer-programming pricing |
| IP-H-CG | `ip_cg_heur` | CPU, Gurobi | column generation with random-insertion pricing, the IP when it stalls |
| | `flow_qp`, `bm_qp` | CPU, Gurobi | the projection as one explicit QP, for small n |

The stored records and the CSVs use the earlier names `CGBFQP-GPU` (Flow-CG), `CGIP` (IP-CG) and `CGIP-heur`
(IP-H-CG); the solver accepts them as aliases (`cgbfqp`, `cgip`, `cgip_heur`).

## Install

Python 3.10 or newer (the paper's runs used 3.12).

    pip install -r requirements.txt

- The GPU methods need an NVIDIA GPU and CuPy matching the CUDA driver: `requirements.txt` installs
  `cupy-cuda13x`; use `cupy-cuda12x` for a CUDA 12 driver.
- The CPU methods need `gurobipy` and a Gurobi licence (the pip package alone is limited to small models).
- Without a GPU the automatic choice takes `flow_qp` up to n = 10 and `ip_cg_heur` above, so it needs Gurobi.

## Use

```python
import pandas as pd
from solver.rum_solver import RUM

# cd ("choice data") holds the observed choice probabilities, one row per (menu, alternative) pair:
#   menu  the offered menu, a bitmask over the alternatives 0..n-1 (bit a set = alternative a offered)
#   alt   an alternative in the menu
#   phat  the observed probability of choosing alt from the menu
n = 18                                         # the number of alternatives
cd = pd.read_csv("final_data/Numerical1/raw_data/n18_d1pct.csv").to_numpy()
rum = RUM.from_rows(cd, n)
res = rum.calculate_distance()                 # automatic method; stops at abs 1e-5 or rel 1e-4; 1 h cap
print(res)                                     # exit_flag, distance, lb, ub, gaps, times, ...
res = rum.calculate_distance(method="bm_admm", rel_tol=1e-3, method_args={"check": 100})
res = rum.calculate_distance(method="flow_cg", opt=OPT)   # the paper's timing race: stop at ub <= 1.01 OPT
```

A menu does not need a row for every alternative in it. `RUM(menu, alt, p_hat, num_alternatives=n)` takes the
three columns of `cd` separately. With `opt`, a call is the paper's timing race rather than a certificate: it
stops once the upper bound reaches max(1.01 OPT, `opt_floor`) and computes no lower bound. Each GPU method
takes `dtype` in `method_args` (BM-ADMM float64 or float32, Flow-CG float32, uint24 or uint16 for its pricing
values, Flow-ADMM float32); without it the class takes the most precise format that fits the GPU.

`method=None` chooses the method from n, the density of the observed cells and the GPU's free memory:
BM-ADMM from 2 % density at n = 11-13, from 1 % at n = 14-17 and from 0.5 % at n >= 18, Flow-CG below
that and at every density for n <= 10. The rule, its measurements and every default are in
`solver/defaults.py` and `solver/DESIGN_NOTES.md`.

## Test RUM: the bootstrap

`rum.bootstrap(panel=...)` or `rum.bootstrap(replies=...)` runs the test from certified brackets throughout. It
needs the choices behind p&#770;, one row per choice:

| input | columns | the data | resampled |
|---|---|---|---|
| `panel=` | respondent, menu, chosen | people who each answered several menus (DSR) | respondents, each with all their rows |
| `replies=` | menu, chosen | replies collected one by one, such as an LLM asked each menu many times (AI) | each menu's replies, on their own |

```python
import pandas as pd
from solver.rum_solver import RUM
from solver.inference import cells_from_panel

panel = pd.read_csv("final_data/Application_DSR/raw_data/overload_data_25_long.csv")
menu, alt, p_hat = cells_from_panel(panel, 25, alternatives=[0])   # DSR alt0: 301 cells
rum = RUM(menu, alt, p_hat, num_alternatives=25)
with rum.bootstrap(panel=panel) as boot:           # n = 520 respondents; B = 10,000 draws, abs 5e-4
    print(boot.p_value())                          # p = 0.1083, every draw decided
    print(boot.confidence_interval())              # [-0.0476, inf) at 95 %
    print(boot.distance_interval())                # two-sided [lower, upper] at 95 %, no draw solved
    boot.save("dsr_alt0_bootstrap.json")           # every draw's bracket, the plan, the times
```

`n` defaults to the number of respondents (panel) or the most replies on any menu (replies); r_n = sqrt(n)
and eps_n = n^(-1/3). The other defaults are in `solver/defaults.py` (`BOOTSTRAP`): B = 10,000, seed 0,
abs 5e-4 and rel 0 per solve, two tenfold tighter rounds for draws still undecided. `checkpoint=path` saves a
run and resumes it after an interrupt. How the steps work is in `solver/DESIGN_NOTES.md` (Bootstrap inference).

## Layout

```
solver/         the package: the RUM class (rum_solver.py), the methods (one folder each), the bootstrap
                (inference/), defaults.py, DESIGN_NOTES.md
data/           seeded instance generator (gen_instances.py) and the certified optima (opt_refs.json)
final_data/     the data: the synthetic instances of Numerical1-3 as CSV, the DSR data, the LLM data
final_results/  the numbers behind the paper's tables, one CSV per experiment
replication/    one folder per experiment: README, script, stored records (see replication/README.md)
validation/     the pipelines the replication folders run, and the table builders
tests/          quick checks of the package (see tests/README.md)
docs/           the illustrated overviews
```

## Hardware of the paper's runs

AMD Ryzen 9 9950X (16 cores), 64 GB, NVIDIA GeForce RTX 5090 (32 GB), Windows 11, CUDA 13.3, Python 3.12,
CuPy 14, Gurobi 13. The IP-CG and IP-H-CG runs: one CPU cluster node per run, 24 Gurobi threads. Every timed
run had the machine to itself: one process at a time, no thread caps, the GPU idle before it started.
