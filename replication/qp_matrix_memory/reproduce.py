"""Constraint-matrix memory of the two exact QP formulations.

Reproduces: MATRIX_MEMORY.md (the storage table behind the note on why the projection cannot be handed to a QP solver at n = 18-20)
Runs: builds K, S (Block-Marschak) and A (arc-flow) in scipy up to the n the host can hold, checks the closed forms, extrapolates

    python replication/qp_matrix_memory/reproduce.py [extra args passed through]

This is an entry point, not an implementation: it execs validation/qp_matrix_memory.py, which is the single
copy of this pipeline. CPU and host memory only (about 8 GB at n = 16); no GPU.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CMD = [["validation/qp_matrix_memory.py", "--max-n-k", "16", "--max-n-a", "20"]]


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
