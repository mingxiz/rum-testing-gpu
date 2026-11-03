"""Feasibility in observation density: n = 20 from 1 % to 100 %.

Reproduces: the n = 20 density axis (records that tab:dense reads with the n = 20 grid cells)
Runs: certify OPT, then race each density cell at n = 20

    python replication/feasibility_in_density/reproduce.py [extra args passed through]

This is an entry point, not an implementation: it execs validation/feasibility/run_feas.py, which is the single
copy of this pipeline. BM-ADMM is the certifier from 5 % up; CGBFQP-GPU at 1 %.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CMD = [['validation/feasibility/run_feas.py', '--part', 'density', '--skip-existing'], ['validation/feasibility/make_feas_tables.py']]


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
