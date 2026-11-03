"""Convex-path reference optima: p(lambda) = lambda p_inside + (1 - lambda) p_hat.

Reproduces: the lambda study (reference optima in data/opt_refs.json; tables PATH_REFS.md, PATH20_REFS.md)
Runs: certify OPT for both axes (pairwise n = 25..35, and the n = 20 density cells), then race the pairwise axis

    python replication/convex_path/reproduce.py [extra args passed through]

This is an entry point, not an implementation: it execs validation/feasibility/run_path.py, run_path20.py, run_path_race.py, which is the single
copy of this pipeline. p_inside is the uniform choice rule 1/|D| on the observed menus, which is rationalizable; lambda = 0 is the stored instance and is only re-run as a consistency check.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CMD = [['validation/feasibility/run_path.py', '--skip-existing'], ['validation/feasibility/run_path20.py', '--skip-existing'], ['validation/feasibility/run_path_race.py', '--skip-existing']]


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
