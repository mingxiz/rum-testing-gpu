"""Regenerate the n = 18-20 synthetic instances used in the K / CB-GPU / AF density
tables (K_param_select/PHASE_DIAGRAM.md) and write them as CSV for other solvers.

Self-contained: numpy only, no GPU. The generators are copied VERBATIM from
paper_sim_3cat/gen_3cat.py (legacy np.random.RandomState streams, which numpy
guarantees stable across versions/platforms), so every instance is bit-identical
to the one its OPT reference was certified on -- and this script proves it by
checking the sha256 fingerprint stored next to each OPT in opt_refs.json.

    python gen_instances.py            # the 30 race-grid cells: instances/*.csv.gz + manifest.json + CELLS.md
    python gen_instances.py --feas     # the 10 feasibility cells: instances/*.csv.gz + manifest_feas.json + CELLS_FEAS.md
    python gen_instances.py --extras   # + the 4 off-grid cells (medium tuning cells, n18 3%)
    python gen_instances.py --check    # regenerate only, verify fingerprints, write nothing
    python gen_instances.py --n 18 --cat m39    # one cell

Instance format (rows = observed (menu, alternative) cells; ALL alternatives of
every sampled menu are observed):
    menu  : integer bitmask over alternatives 0..n-1 (bit i set <=> alternative i in menu)
    alt   : chosen alternative, 0-based, always a member of menu
    phat  : observed choice probability; phat sums to 1 within each menu
"""
import argparse
import gzip
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.join(HERE, "instances")


# ----------------------------------------------------------------------------
# generators: verbatim copies of paper_sim_3cat/gen_3cat.py
# ----------------------------------------------------------------------------
def gen_pairwise(n, phat_seed=100):
    rng = np.random.RandomState(phat_seed)
    om, ox, ph = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            mask = (1 << i) | (1 << j)
            g = rng.rand(2)
            g = g / g.sum()
            om += [mask, mask]
            ox += [i, j]
            ph += [float(g[0]), float(g[1])]
    cd = np.column_stack([np.array(om, np.float64), np.array(ox, np.float64),
                          np.array(ph, np.float64)])
    return cd, len(ph)


def p_inside(cd):
    """The uniform choice rule on the OBSERVED menus: p(x|D) = 1/|D| for every observed
    cell (0.5 / 0.5 on a pair). It is the choice probability of the uniform mixture over
    all rankings, hence RUM-rationalizable and interior."""
    cd = np.array(cd, dtype=np.float64, copy=True)
    sizes = np.array([bin(int(m)).count("1") for m in cd[:, 0]], dtype=np.float64)
    cd[:, 2] = 1.0 / sizes
    return cd


def gen_convex_path(n, k, base_cat="sparse"):
    """Convex path between a stored instance p_hat (category base_cat: sparse = pairwise,
    d1pct, m<mult>, dense) and p_inside on the same menus:
    p(lambda) = lambda * p_inside + (1 - lambda) * p_hat,  lambda = k / 10, k = 0..10.
    k = 0 is the base instance itself (same bytes, same OPT reference); k = 10 is the
    uniform rule (distance 0)."""
    cd, mp = get_instance(n, base_cat)
    if k == 0:
        return cd, mp
    lam = k / 10.0
    pin = p_inside(cd)
    out = cd.copy()
    out[:, 2] = lam * pin[:, 2] + (1.0 - lam) * cd[:, 2]
    return out, mp


def gen_medium(n, mult=20, menu_seed=42, phat_seed=100):
    """Uniform random subsets (size>=2, no repeats) until mp >= mult*n*(n-1)."""
    target = mult * n * (n - 1)
    rng = np.random.RandomState(menu_seed)
    seen, menus, mp = set(), [], 0
    while mp < target:
        m = int(rng.randint(0, 1 << n))
        if m in seen:
            continue
        k = bin(m).count("1")
        if k < 2:
            continue
        seen.add(m)
        menus.append(m)
        mp += k
    prng = np.random.RandomState(phat_seed)
    om, ox, ph = [], [], []
    for m in menus:
        items = [i for i in range(n) if (m >> i) & 1]
        g = prng.rand(len(items))
        g = g / g.sum()
        for kk, x in enumerate(items):
            om.append(m)
            ox.append(x)
            ph.append(float(g[kk]))
    cd = np.column_stack([np.array(om, np.float64), np.array(ox, np.float64),
                          np.array(ph, np.float64)])
    return cd, len(ph)


def gen_dense(n, phat_seed=100):
    """ALL menus of size>=2 (mp = n*2^(n-1) - n rows). Not part of the CGIP
    cells (CB-GPU is already infeasible there); kept for completeness."""
    Nnode = 1 << n
    masks_all = np.arange(Nnode, dtype=np.int64)
    pop = np.zeros(Nnode, dtype=np.int64)
    for b in range(n):
        pop += (masks_all >> b) & 1
    keep = pop >= 2
    om_parts, ox_parts = [], []
    for i in range(n):
        has_i = ((masks_all >> i) & 1).astype(bool) & keep
        mi = masks_all[has_i]
        om_parts.append(mi)
        ox_parts.append(np.full(len(mi), i, dtype=np.int64))
    om = np.concatenate(om_parts)
    ox = np.concatenate(ox_parts)
    rng = np.random.RandomState(phat_seed)
    g = rng.rand(len(om))
    sums = np.bincount(om, weights=g, minlength=Nnode)
    ph = g / sums[om]
    cd = np.column_stack([om.astype(np.float64), ox.astype(np.float64), ph])
    return cd, len(ph)


# ----------------------------------------------------------------------------
# category names, exactly as in K_param_select/kps_harness.py
# ----------------------------------------------------------------------------
def get_instance(n, cat):
    if cat == "sparse":                       # pairwise: every {i,j}, both alts
        return gen_pairwise(n)
    if cat == "medium":                       # the tuning cell (mult=20)
        return gen_medium(n, mult=20)
    if cat == "dense":
        return gen_dense(n)
    if cat == "d1pct":                        # ~1% of the n*2^(n-1) slab
        m2 = n * (1 << (n - 1))
        return gen_medium(n, mult=round(0.01 * m2 / (n * (n - 1))))
    if cat.startswith("m") and cat[1:].isdigit():   # density grid: m<mult>
        return gen_medium(n, mult=int(cat[1:]))
    if "_lam" in cat and cat.split("_lam")[1].isdigit():     # convex path, lambda = k/10, any base
        base, k = cat.split("_lam")
        return gen_convex_path(n, int(k), base_cat=base)
    raise ValueError(cat)


# The density grid: pairwise + the SAME nine density targets at every n
#   sparse regime, x2 steps      : 0.025, 0.05, 0.1, 0.2 %
#   crossover regime, ~x1.5 steps: 0.3, 0.5, 0.7, 1 %   (K overtakes CB in 0.35-0.7 %)
#   dense regime, one decade     : 10 %
# mult = round(target * m2 / (n(n-1))), so the cell names differ per n while the
# realized densities agree to rounding (see CELLS.md).
CELLS = {
    18: ["sparse", "m2", "m4", "m8", "m15", "m23", "m39", "m54", "d1pct", "m771"],
    19: ["sparse", "m4", "m7", "m15", "m29", "m44", "m73", "m102", "d1pct", "m1456"],
    20: ["sparse", "m8", "m14", "m28", "m55", "m83", "m138", "m193", "d1pct", "m2759"],
}
# Off-grid instances that also have certified OPTs: the K-parameter tuning cells
# (fixed mult=20, so their density drifts with n) and the n=18 3% twin of the
# LLM dataset. Available via --n/--cat or --extras; not part of the ladder.
EXTRAS = {18: ["medium", "m231"], 19: ["medium"], 20: ["medium"]}

# Feasibility study (paper section "feasibility in n and in density"):
#   (1) pairwise design, growing n: n = 20 (grid cell n20_sparse), 25, 30, 32
#       (32 = the largest n whose mask-indexed lattice fits a 32 GB GPU; n = 35
#       would need the layer-streaming kernel, see solver/flow_cg/MEMORY.md,
#       and is left out for now -- gen_pairwise(35) still generates it on request)
#   (2) n = 20, growing density: 1 % (grid cell n20_d1pct), 5 %, 10 % (grid cell
#       n20_m2759), 25 %, 50 % (random-menu family, mult = round(d*m2/(n(n-1)))),
#       100 % (`dense` = every menu of size >= 2, gen_dense).
# Same generators and seeds as the grid, so the shared cells are byte-identical.
FEAS_PAIRWISE = {25: ["sparse"], 27: ["sparse"], 30: ["sparse"], 32: ["sparse"]}
# extension run with the rank-indexed two-layer pricer (pricing="stream_f32"/"stream_u16"):
FEAS_PAIRWISE_EXT = {33: ["sparse"], 34: ["sparse"], 35: ["sparse"]}
FEAS_DENSITY20 = ["m1380", "m6899", "m13797", "dense"]
FEAS_SHARED = ["n20_sparse", "n20_d1pct", "n20_m2759"]      # grid cells reused by the study
# Convex-path OPT references (--path): p(lambda) = lambda*p_inside + (1-lambda)*p_hat on the
# pairwise instances, lambda = k/10; k = 0 is the stored pairwise reference (n{n}_sparse), so it is
# only regenerated for the consistency check at n = 25. Keys: n{n}_sparse_lam{k}.
# n = 33-35 are possible with the stream pricing backends but were left out (run time).
PATH_N = [25, 27, 30, 32]
PATH_LAMBDAS = list(range(0, 11))
# The same path on the n = 20 density axis (--path20): base cells = the six feasibility density
# cells (1, 5, 10, 25, 50, 100 %), keys n20_{cat}_lam{k}. k = 0 is the stored reference n20_{cat}.
PATH20_CATS = ["d1pct", "m1380", "m2759", "m6899", "m13797", "dense"]


def fingerprint(cd):
    """Same as kps_harness._cd_hash: sha256 of the C-contiguous float64 (M,3) array."""
    a = np.ascontiguousarray(cd, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def write_csv(cd, path):
    with gzip.open(path, "wt", newline="\n") as f:
        f.write("menu,alt,phat\n")
        for m, x, p in cd:
            f.write("%d,%d,%.17g\n" % (int(m), int(x), p))


def read_csv(path):
    """Loader for the CSV written above (round-trips float64 exactly)."""
    with gzip.open(path, "rt") as f:
        a = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--cat")
    ap.add_argument("--check", action="store_true", help="verify fingerprints only")
    ap.add_argument("--no-csv", action="store_true", help="refresh the manifest and cell table without rewriting the instance CSVs")
    ap.add_argument("--extras", action="store_true", help="also the off-grid cells (EXTRAS)")
    ap.add_argument("--feas", action="store_true",
                    help="the feasibility-study cells only (FEAS_PAIRWISE + FEAS_DENSITY20 + shared grid cells)")
    ap.add_argument("--path", action="store_true",
                    help="the convex-path cells n{n}_sparse_lam{k} (PATH_N x PATH_LAMBDAS)")
    ap.add_argument("--path-n", type=int, nargs="*", default=None, help="restrict --path to these n")
    ap.add_argument("--path-grid", action="store_true",
                    help="the density-grid convex-path cells n{n}_{cat}_lam{k} for k in (5, 10) (CELLS x {0.5, 1.0})")
    ap.add_argument("--path20", action="store_true",
                    help="the n = 20 density-axis convex-path cells n20_{cat}_lam{k} (PATH20_CATS x PATH_LAMBDAS)")
    a = ap.parse_args()
    refs = json.load(open(os.path.join(HERE, "opt_refs.json")))
    if a.cat:
        cells = [(a.n, a.cat)]
    elif a.path:
        cells = [(n, "sparse_lam%d" % k) for n in (a.path_n or PATH_N) for k in PATH_LAMBDAS]
    elif a.path20:
        cells = [(20, "%s_lam%d" % (c, k)) for c in PATH20_CATS for k in PATH_LAMBDAS]
    elif a.path_grid:
        cells = [(n, "%s_lam%d" % (c, k)) for n in CELLS for c in CELLS[n] for k in (5, 10)]
    elif a.feas:
        cells = [(int(k.split("_")[0][1:]), k.split("_", 1)[1]) for k in FEAS_SHARED]
        cells += [(n, c) for n in FEAS_PAIRWISE for c in FEAS_PAIRWISE[n]]
        cells += [(n, c) for n in FEAS_PAIRWISE_EXT for c in FEAS_PAIRWISE_EXT[n]]
        cells += [(20, c) for c in FEAS_DENSITY20]
    else:
        cells = [(n, c) for n in CELLS for c in CELLS[n]]
    if a.extras and not a.cat:
        cells += [(n, c) for n in EXTRAS for c in EXTRAS[n]]
    os.makedirs(INST, exist_ok=True)
    manifest, rows, all_ok = {}, [], True
    for n, cat in cells:
        key = "n%d_%s" % (n, cat)
        cd, mp = get_instance(n, cat)
        fp = fingerprint(cd)
        m2 = n * (1 << (n - 1))
        ref = refs.get(key if not key.endswith("_lam0") else key[:-5])     # lambda = 0 IS the pairwise cell
        ok = ref is not None and ref["cd_hash"] == fp and ref["mp"] == mp
        all_ok &= ok
        n_menus = len(np.unique(cd[:, 0]))
        rec = dict(n=n, cat=cat, mp=int(mp), n_menus=int(n_menus), m2=m2,
                   density_pct=100.0 * mp / m2, fingerprint=fp,
                   **({"lambda": int(cat.split("_lam")[1]) / 10.0, "base_key": "n%d_%s" % (n, cat.split("_lam")[0])} if "_lam" in cat else {}),
                   fingerprint_matches_opt_ref=bool(ok),
                   file="instances/%s.csv.gz" % key)
        if ref:
            rec.update(opt=ref["opt"], opt_lb=ref["lb"], opt_gap_rel=ref.get("gap"), opt_gap_abs=ref.get("gap_abs"),
                       opt_method=ref["method"], opt_cb_columns=ref.get("n_cols"),
                       opt_wall_s=ref.get("wall_s"))
        manifest[key] = rec
        if not a.check and not a.no_csv:
            path = os.path.join(INST, "%s.csv.gz" % key)
            write_csv(cd, path)
            back = read_csv(path)
            assert fingerprint(back) == fp, "%s: CSV round-trip changed the data" % key
        print("%-12s mp=%9d menus=%7d density=%7.3f%%  fp=%s %s"
              % (key, mp, n_menus, rec["density_pct"], fp,
                 "MATCHES opt_refs" if ok else ("NO REF YET" if ref is None else "MISMATCHED REF")), flush=True)
        rows.append(rec)
    if not a.cat and not a.check:
        anypath = a.path or a.path20 or a.path_grid
        mname = ("manifest_path20.json" if a.path20 else ("manifest_path_grid.json" if a.path_grid else "manifest_path.json")) if anypath else ("manifest_feas.json" if a.feas else "manifest.json")
        cname = ("CELLS_PATH20.md" if a.path20 else ("CELLS_PATH_GRID.md" if a.path_grid else "CELLS_PATH.md")) if anypath else ("CELLS_FEAS.md" if a.feas else "CELLS.md")
        title = ("Convex-path cells, n = 20 density axis" if a.path20 else "Convex-path cells") if anypath else ("Feasibility-study cells" if a.feas else "Race-grid cells")
        flag = (" --path20" if a.path20 else " --path") if anypath else (" --feas" if a.feas else "")
        json.dump(manifest, open(os.path.join(HERE, mname), "w"), indent=2)
        with open(os.path.join(HERE, cname), "w") as f:
            f.write("# %s (auto-generated by gen_instances.py%s)\n\n"
                    % (title, flag) +
                    ("p(lambda) = lambda * p_inside + (1 - lambda) * p_hat on the base cell's menus; p_inside = 1/|D| on "
                     "every observed cell. lambda = 0 is the stored base reference. Cells whose distance is ~0 are "
                     "certified by an absolute criterion (see the `criterion` field of the reference).\n\n" if anypath else "") +
                    "OPT = certified projection distance sqrt(sum_obs (x - phat)^2); "
                    "the bracket is [OPT LB, OPT] with (OPT - OPT LB)/OPT LB <= 1e-3. "
                    "Race target = 1.01 * OPT.\n\n"
                    "| key | n | mp | menus | density | OPT (UB) | OPT LB | gap "
                    "| certified by | CB cols |\n"
                    "|---|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                g = r.get("opt_gap_rel")
                gtxt = ("%.1e" % g) if isinstance(g, (int, float)) and g == g else (
                    ("abs %.1e" % r["opt_gap_abs"]) if r.get("opt_gap_abs") is not None else "nan")
                f.write("| n%d_%s | %d | %s | %s | %.3f%% | %.6f | %.6f | %s | %s | %s |\n"
                        % (r["n"], r["cat"], r["n"], format(r["mp"], ","),
                           format(r["n_menus"], ","), r["density_pct"],
                           r.get("opt", float("nan")), r.get("opt_lb", float("nan")),
                           gtxt, r.get("opt_method", "-"),
                           r.get("opt_cb_columns", "-")))
    print("ALL FINGERPRINTS MATCH opt_refs.json:", all_ok,
          "" if all_ok else "(cells without a reference yet are expected for a fresh feasibility run)")


if __name__ == "__main__":
    main()
