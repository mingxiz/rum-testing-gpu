"""Pricing-backend comparison runs on the pairwise design, one fresh process per
measurement, races gated on an idle GPU.

    python validation/feasibility/run_pricing.py --part compare      # n = 20, 25, 30, 32: race with mask / stream_f32 / stream_u16
    python validation/feasibility/run_pricing.py --part extend       # n = 33, 34: cert + race with f32 and u16; n = 35: u16 only
    python validation/feasibility/run_pricing.py --part certcmp      # n = 20, 25, 30, 32: certification with stream_f32 / stream_u16 (mask certs come from run_feas.py)
    python validation/feasibility/run_pricing.py --part all

The first certification of a new cell writes its OPT to data/opt_refs.json; the
second backend then re-certifies and is compared by bracket overlap.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "validation"))
from run_paper_table import wait_for_idle_gpu   # noqa: E402

PY = sys.executable
OUT = os.path.join(ROOT, "validation", "results", "feasibility")


def run(key, criterion, pricing, cap, skip_existing):
    tag = "" if pricing == "mask" else "-" + pricing
    path = os.path.join(OUT, "%s_cgbfqp%s_%s.json" % (criterion, tag, key))
    if skip_existing and os.path.exists(path):
        print("skip", os.path.basename(path), flush=True)
        return
    if criterion == "race":
        wait_for_idle_gpu(lambda m: print(m, flush=True))
    t0 = time.perf_counter()
    subprocess.run([PY, os.path.join(HERE, "feas_one.py"), "--key", key, "--solver", "cgbfqp",
                    "--criterion", criterion, "--cap", str(cap), "--pricing", pricing, "--out", OUT], check=False)
    print("   (%.0f s)" % (time.perf_counter() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "compare", "extend", "certcmp"])
    ap.add_argument("--cap", type=float, default=7200.0)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.part in ("all", "compare"):
        for n in (20, 25, 30, 32):
            for pricing in ("mask", "stream_f32", "stream_u16"):
                run("n%d_sparse" % n, "race", pricing, a.cap, a.skip_existing)
    if a.part in ("all", "certcmp"):
        for n in (20, 25, 30, 32):
            for pricing in ("stream_f32", "stream_u16"):
                run("n%d_sparse" % n, "cert", pricing, a.cap, a.skip_existing)
    if a.part in ("all", "extend"):
        for n, modes in ((33, ("stream_f32", "stream_u16")), (34, ("stream_f32", "stream_u16")), (35, ("stream_u16",))):
            for pricing in modes:
                run("n%d_sparse" % n, "cert", pricing, a.cap, a.skip_existing)
            for pricing in modes:
                run("n%d_sparse" % n, "race", pricing, a.cap, a.skip_existing)
    subprocess.run([PY, os.path.join(HERE, "compare_pricing.py")], check=False)


if __name__ == "__main__":
    main()
