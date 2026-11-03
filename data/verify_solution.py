"""Independent evaluator for a CG-type solution: a mixture of rankings.

    python verify_solution.py --key n18_m39 --solution solutions/n18_m39.json

solution JSON: {"rankings": [[best, ..., worst], ...]   (0-based permutations of 0..n-1)
                "weights":  [w_1, ..., w_R]}            (w >= 0, sum w = 1)
Prints the objective sqrt(sum_obs (x - phat)^2) of the mixture, the relative gap
to the certified OPT, and whether the 1% race criterion is met. Uses only numpy,
so it does not depend on anybody's solver.
"""
import argparse
import json
import os

import numpy as np

from gen_instances import read_csv, get_instance, fingerprint

HERE = os.path.dirname(os.path.abspath(__file__))


def load_instance(key):
    path = os.path.join(HERE, "instances", key + ".csv.gz")
    if os.path.exists(path):
        return read_csv(path)
    n, cat = key.split("_", 1)                     # regenerate on the fly
    cd, _ = get_instance(int(n[1:]), cat)
    return cd


def mixture_probabilities(n, menu, alt, rankings, weights):
    """x[o] = sum_r w_r * 1{ top-ranked member of menu[o] under r == alt[o] }."""
    umenu, inv = np.unique(menu, return_inverse=True)
    memb = ((umenu[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)   # (U, n)
    x = np.zeros(len(menu))
    for r, w in zip(rankings, weights):
        pos = np.empty(n, dtype=np.int64)
        pos[np.asarray(r, dtype=np.int64)] = np.arange(n)    # rank position of each alt
        best = np.argmin(np.where(memb, pos[None, :], n), axis=1)   # top member per menu
        x += w * (best[inv] == alt)
    return x


def evaluate(key, rankings, weights, opt=None):
    cd = load_instance(key)
    n = int(key.split("_")[0][1:])
    menu = cd[:, 0].astype(np.int64)
    alt = cd[:, 1].astype(np.int64)
    ph = cd[:, 2]
    for r in rankings:
        if sorted(int(i) for i in r) != list(range(n)):
            raise ValueError("a ranking is not a permutation of 0..n-1")
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(rankings) or np.any(w < -1e-12):
        raise ValueError("weights must be >= 0 with one weight per ranking")
    x = mixture_probabilities(n, menu, alt, rankings, w / w.sum())
    dist = float(np.sqrt(np.sum((x - ph) ** 2)))
    out = dict(key=key, fingerprint=fingerprint(cd), n_rankings=len(rankings),
               weight_sum=float(w.sum()), distance=dist)
    if opt is not None:
        rel = (dist - opt) / opt
        out.update(opt=opt, rel_gap=rel, race_1pct_met=bool(rel <= 0.01))
    return out


def selftest():
    """Vectorized evaluator == naive per-observation loop on random mixtures."""
    rng = np.random.RandomState(0)
    for key in ("n18_sparse", "n18_m8"):
        cd = load_instance(key)
        n = 18
        menu = cd[:, 0].astype(np.int64)
        alt = cd[:, 1].astype(np.int64)
        rks = [list(rng.permutation(n)) for _ in range(7)]
        w = rng.rand(7)
        w /= w.sum()
        x = mixture_probabilities(n, menu, alt, rks, w)
        idx = rng.choice(len(menu), min(400, len(menu)), replace=False)
        for o in idx:
            items = [i for i in range(n) if (menu[o] >> i) & 1]
            xo = sum(wr for r, wr in zip(rks, w)
                     if min(items, key=lambda i: r.index(i)) == alt[o])
            assert abs(xo - x[o]) < 1e-12, (key, o, xo, x[o])
        print("selftest %s: OK (%d observations cross-checked)" % (key, len(idx)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", help="instance key, e.g. n18_m231")
    ap.add_argument("--solution", help="JSON with rankings + weights")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    sol = json.load(open(a.solution))
    refs = json.load(open(os.path.join(HERE, "opt_refs.json")))
    opt = refs.get(a.key, {}).get("opt")
    print(json.dumps(evaluate(a.key, sol["rankings"], sol["weights"], opt), indent=2))


if __name__ == "__main__":
    main()
