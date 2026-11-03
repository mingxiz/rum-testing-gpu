"""Pricing-backend comparison on the pairwise design: mask-indexed (today) vs the
rank-indexed two-layer pricer in float32 and uint16 block-scaled storage.

Reads validation/results/feasibility/{race,cert}_cgbfqp[-<pricing>]_n*_sparse.json
and writes results/feasibility/PRICING_COMPARISON.md + paper/tables/tab_feas_pricing.tex.
Same instances (fingerprint-checked), same OPT, same stopping rules, one fresh
process per run on an idle GPU. Two criteria are tabulated: the race (first
column whose pushed UB is within 1 % of OPT) and the certification
((UB - LB)/LB <= 1e-3 with the Frank-Wolfe bound). Every CGBFQP iteration adds
one column, so "columns" is the iteration count.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "validation", "results", "feasibility")
MODES = [("", "mask (today)"), ("-stream_f32", "two layers, float32"), ("-stream_u16", "two layers, uint16 block-scaled")]
FITS = {"": 32, "-stream_f32": 34, "-stream_u16": 35}          # largest n each backend fits on the 32 GB card


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load(kind):
    d = {}
    for tag, _ in MODES:
        for p in glob.glob(os.path.join(RES, "%s_cgbfqp%s_n*_sparse.json" % (kind, tag))):
            r = read_json(p)
            d[(tag, r["key"])] = r
    return d


def cell(r, n, tag):
    """'time (columns)' for a finished run, the stop reason otherwise, '--' when it cannot fit, '' when not run."""
    if r is None:
        return "--" if n > FITS[tag] else ""
    if r.get("wall_s") is None:
        return r.get("stop", "err")
    if r.get("stop") not in ("phi_ub", "gap_rel"):
        return "%s (%d)" % (r.get("stop", "err"), r.get("iters", 0))
    t = r["wall_s"]
    return "%s (%d)" % (("%.2f" % t) if t < 10 else ("%.1f" % t), r["iters"])


def gib(r, tex=False):
    if r is None or not r.get("device_bytes"):
        return "--"
    x = r["device_bytes"] / 2 ** 30
    return ("%.1f" % x) if x >= 1 else (("%.2f" % x) if x >= 0.01 else ("$<$0.01" if tex else "<0.01"))


def margin(r):
    return ("%.1e" % r["margin_max"]) if r and r.get("margin_max") is not None else ""


def main():
    race, cert = load("race"), load("cert")
    refs = read_json(os.path.join(ROOT, "data", "opt_refs.json"))
    keys = sorted({k for (_, k) in race} | {k for (_, k) in cert}, key=lambda k: int(k.split("_")[0][1:]))

    def block(title, runs, extra_hdr, extra_fn):
        out = ["## " + title + "\n",
               "| n | mp | OPT | " + " | ".join("%s: s (columns)" % lbl for _, lbl in MODES) +
               " | u16 margin | device memory GiB mask / f32 / u16 |" + extra_hdr,
               "|---|---|---|" + "---|" * len(MODES) + "---|---|" + ("---|" if extra_hdr else "")]
        for k in keys:
            n = int(k.split("_")[0][1:])
            rs = [runs.get((tag, k)) for tag, _ in MODES]
            if all(r is None for r in rs):
                continue
            opt = refs.get(k, {}).get("opt")
            out.append("| %d | %s | %s | %s | %s | %s |%s" % (
                n, format(n * (n - 1), ","), ("%.6f" % opt) if opt else "",
                " | ".join(cell(r, n, tag) for r, (tag, _) in zip(rs, MODES)),
                margin(rs[2]), " / ".join(gib(r) for r in rs), extra_fn(rs)))
        return out

    def brackets(rs):
        # certified [LB, UB] per backend; the reference is the first backend that certified the cell
        parts = []
        for r in rs:
            parts.append("[%.5f, %.5f]" % (r["lb"], r["ub"]) if r and r.get("lb") is not None else "-")
        return " " + " / ".join(parts) + " |"

    md = ["# Pricing backends on the pairwise design\n",
          "Same instances (`data/gen_instances.py --feas`, fingerprint-checked), same OPT reference, same stopping rules; "
          "one fresh process per run on an otherwise idle RTX 5090 (32 GB). Every CGBFQP iteration adds one column, so the "
          "number in parentheses is the iteration count. `--` = does not fit in device memory; blank = not run.\n"]
    md += block("Race: first column whose feasibility-pushed UB satisfies (UB - OPT)/OPT <= 1 %", race, "", lambda rs: "")
    md.append("")
    md += block("Certification: (UB - LB)/LB <= 1e-3 with the Frank-Wolfe lower bound", cert,
                " certified [LB, UB] mask / f32 / u16 |", brackets)
    md.append("\nDevice memory: mask = 5 B per subset (V float32 + pred uint8) + one layer buffer of C(n, n/2) masks; "
              "two layers = 2 x C(n, n/2) x (4 or 2) B + uint8 predecessors for the top 10 layers + the 2^(n-10) sub-lattice "
              "scaffolding. The uint16 margin (0.5 x sum over layers of the largest block step) is added to Vmax before the "
              "Frank-Wolfe gap, so the uint16 lower bound is valid, only looser by the margin; the columns generated can differ "
              "from the float32 run only where two rankings tie within the margin.")
    write_text(os.path.join(RES, "PRICING_COMPARISON.md"), "\n".join(md) + "\n")

    tex = ["% generated by validation/feasibility/compare_pricing.py", "\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{CGBFQP-GPU pricing backends on the pairwise design: wall time in seconds (columns generated) to the race "
           "criterion, $(\\mathrm{UB}-\\mathrm{OPT})/\\mathrm{OPT}\\le 1\\%$, and to the certificate "
           "$(\\mathrm{UB}-\\mathrm{LB})/\\mathrm{LB}\\le 10^{-3}$, for the mask-indexed lattice and the rank-indexed two-layer "
           "lattice in float32 and uint16 block-scaled storage; same instances, same reference optimum; one RTX 5090 (32~GB). "
           "``--'' = does not fit in device memory.}",
           "\\label{tab:feas_pricing}", "\\begin{tabular}{rrrrrrrrrl}", "\\toprule",
           " & & & \\multicolumn{3}{c}{race to 1\\% of OPT} & \\multicolumn{3}{c}{certificate $10^{-3}$} & memory (GiB) \\\\",
           "\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}",
           "$n$ & $m_p$ & OPT & mask & f32 & u16 & mask & f32 & u16 & mask / f32 / u16 \\\\", "\\midrule"]
    for k in keys:
        n = int(k.split("_")[0][1:])
        rr = [race.get((tag, k)) for tag, _ in MODES]
        cc = [cert.get((tag, k)) for tag, _ in MODES]
        opt = refs.get(k, {}).get("opt")
        anym = [r if r else c for r, c in zip(rr, cc)]
        tex.append("%d & %s & %s & %s & %s & %s \\\\" % (
            n, format(n * (n - 1), ",").replace(",", "{,}"), ("%.4f" % opt) if opt else "",
            " & ".join(cell(r, n, tag) for r, (tag, _) in zip(rr, MODES)),
            " & ".join(cell(c, n, tag) for c, (tag, _) in zip(cc, MODES)),
            " / ".join(gib(r, tex=True) for r in anym)))
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    os.makedirs(os.path.join(ROOT, "paper", "tables"), exist_ok=True)
    write_text(os.path.join(ROOT, "paper", "tables", "tab_feas_pricing.tex"), "\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
