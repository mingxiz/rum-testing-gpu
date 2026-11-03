"""Reference optima for the lambda = 1 cells: OPT = 0, written analytically.

p(lambda = 1) = p_inside, the uniform choice rule p(x|D) = 1/|D| on the observed menus. That is
the choice probability of the uniform mixture over all n! rankings, so it is rationalizable and
its distance to the RUM polytope is exactly 0. The certificate machinery cannot express 0 as a
relative gap, so these entries are written directly, uniformly for every lambda = 1 cell:
opt = lb = 0, method "analytic". Where a numerically certified record already exists (the
pairwise and n = 20 density paths were certified to an absolute floor: UB 0 to 3.6e-4), that
record is kept verbatim under "certified" so nothing is lost.

    python validation/write_lambda1_refs.py            # all lambda = 1 cells of the three paths
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
from gen_instances import get_instance, fingerprint, CELLS, PATH_N, PATH20_CATS    # noqa: E402

REFS = os.path.join(ROOT, "data", "opt_refs.json")
METHOD = "analytic: OPT = 0 (p_inside is the uniform mixture of all rankings)"


def keys():
    out = [(n, "sparse") for n in sorted(set(PATH_N) | {33, 34, 35})]
    out += [(20, c) for c in PATH20_CATS]
    out += [(n, c) for n in CELLS for c in CELLS[n]]
    return [(n, c) for n, c in out]


def main():
    refs = json.load(open(REFS))
    written, kept = 0, 0
    for n, cat in keys():
        key = "n%d_%s_lam10" % (n, cat)
        base = "n%d_%s" % (n, cat)
        cd, mp = get_instance(n, cat + "_lam10")
        fp = fingerprint(cd)
        old = refs.get(key)
        rec = dict(opt=0.0, lb=0.0, gap=None, gap_abs=0.0, criterion="analytic", method=METHOD,
                   mp=int(mp), n_cols=None, iters=None, wall_s=None, cd_hash=fp,
                   when=time.strftime("%Y-%m-%d %H:%M:%S"), **{"lambda": 1.0}, base_key=base)
        if old is not None and old.get("method") != METHOD:
            rec["certified"] = {k: old[k] for k in ("opt", "lb", "gap_abs", "criterion", "method", "n_cols",
                                                     "iters", "wall_s", "when") if k in old}
            kept += 1
        elif old is not None and "certified" in old:
            rec["certified"] = old["certified"]
            kept += 1
        refs[key] = rec
        written += 1
    json.dump(refs, open(REFS, "w"), indent=2)
    print("lambda = 1 references written: %d (certified records preserved under 'certified': %d) -> %s"
          % (written, kept, REFS))


if __name__ == "__main__":
    main()
