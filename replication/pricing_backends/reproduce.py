"""The lattice layouts compared on identical instances.

Reproduces: the pricing-layout comparison (how n = 33-35 fit a 32 GB card)
Runs: the same pairwise cells under the mask-indexed lattice and the two-layer float32 / uint16 backends

    python replication/pricing_backends/reproduce.py [extra args passed through]

This is an entry point, not an implementation: it execs validation/feasibility/run_pricing.py, which is the single
copy of this pipeline. Correctness first: validation/feasibility/test_stream_pricer.py checks float32 is bit-identical to the mask kernel and that the quantized backends stay inside their reported margin.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CMD = [['validation/feasibility/test_stream_pricer.py', '.'], ['validation/feasibility/run_pricing.py', '--part', 'compare', '--skip-existing']]


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
