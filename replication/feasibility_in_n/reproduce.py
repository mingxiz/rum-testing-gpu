"""Feasibility in the number of alternatives: the pairwise design from n = 20 to 35.

Reproduces: the pairwise design, n = 20 to 35 (its n = 25-35 races are tab:binary at lambda = 0)
Runs: certify OPT, then race each pairwise cell; n >= 33 uses the two-layer pricing backends

    python replication/feasibility_in_n/reproduce.py [extra args passed through]

This is an entry point, not an implementation: it execs validation/feasibility/run_feas.py and run_pricing.py, which is the single
copy of this pipeline. n = 33, 34 use two-layer float32; n = 35 uses uint24 (see solver/flow_cg/MEMORY.md).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CMD = [['validation/feasibility/run_feas.py', '--part', 'pairwise', '--skip-existing'], ['validation/feasibility/make_feas_tables.py']]


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
