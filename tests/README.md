# tests

Quick checks of the package, each a script run from the repository root. They print one line per check and
exit with status 1 if any check fails.

| script | needs | time | what it checks |
|---|---|---|---|
| `test_cpu.py` | Gurobi | a few minutes | the RUM class on the CPU: input checks, IP-CG, IP-H-CG and the exact QPs, certificates, weights, the documented errors, the race mode (GPU engines stubbed) |
| `test_cgip.py` | Gurobi | under a minute | IP-CG and IP-H-CG at distance 0, against the exact QP, and on their time limits |
| `test_bootstrap.py` | Gurobi | a few minutes | `rum.bootstrap` with panel and replies input on toy data (the draws, the p-value and interval, the record, checkpoint resume, n from the data, the errors) and the 100-reply AI data (every draw below the threshold with no solve) |
| `test_intervals.py` | Gurobi | a few minutes | `Bootstrap.distance_interval` on toy data, with weights, and the paper's two AI intervals from the shipped data |
| `test_gpu.py` | NVIDIA GPU | about two minutes | the automatic method choice at n = 10-17, and five of the paper's stored races re-run with identical iterations, upper bound and stop reason |

    python tests/test_cpu.py
    python tests/test_cgip.py
    python tests/test_bootstrap.py
    python tests/test_intervals.py
    python tests/test_gpu.py

The CPU tests run Gurobi with one thread and never use the GPU. Run `test_gpu.py` alone: it waits until the GPU
is idle before each timed race (`validation/run_paper_table.py::wait_for_idle_gpu`).
