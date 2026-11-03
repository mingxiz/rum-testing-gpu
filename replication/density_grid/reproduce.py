"""tab:density18-20: the n = 18, 19, 20 density grid.

Reproduces: tab:density18, tab:density19, tab:density20 (section 'Algorithm Comparison in Synthetic Data')
Runs: every (cell, solver) of the 30-cell grid, timed to (UB-OPT)/OPT <= 1 %, then the LaTeX tables

    python replication/density_grid/reproduce.py [extra args passed through]

This is an entry point, not an implementation: it execs validation/run_paper_table.py, which is the single
copy of this pipeline. Add --solvers cgip,cgip_heur to include the Gurobi CPU baselines (a licence is needed).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CMD = [['validation/run_paper_table.py']]


def main():
    for step in CMD:
        full = [sys.executable] + step + sys.argv[1:]
        print("")
        print("$ " + " ".join(full[1:]), flush=True)
        rc = subprocess.run(full, cwd=ROOT).returncode
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
