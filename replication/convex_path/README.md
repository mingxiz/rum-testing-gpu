# Convex-path reference optima and races

Reproduces the lambda study: the reference optima stored in `data/opt_refs.json` under the
keys `n{n}_sparse_lam{k}` and `n20_{cat}_lam{k}`, and the tables `PATH_REFS.md`,
`PATH20_REFS.md` and `PATH_RACE.md` in `results/` here.

## What it is

A family of instances interpolating between real data and a point known to be rationalizable:

    p(lambda) = lambda * p_inside + (1 - lambda) * p_hat,    lambda = k/10, k = 0 .. 10

`p_inside` is the uniform choice rule, p(x|D) = 1/|D| on every observed menu. Those are the
choice probabilities of the uniform mixture over all rankings, so the point is rationalizable
and sits inside the RUM polytope. lambda = 0 is the stored instance, byte for byte, and is
re-run only as a consistency check; lambda = 1 is the uniform rule, whose distance is 0.

Two axes, 120 reference optima in total: the pairwise design at n = 25, 30, 32, 33, 34, 35,
and the n = 20 density cells from 1 % to 100 %.

Certification uses (UB - LB)/LB <= 1e-3, or an absolute criterion once the distance is
essentially zero, where a relative one is undefined. Each record says which was used in its
`criterion` field, and carries `lambda` and `base_key`.

## Run it

    python replication/convex_path/reproduce.py

Three stages: certify the pairwise axis, certify the density axis, then race the pairwise
axis to (UB - OPT)/OPT <= 1 %. The races are idle-gated and sequential; certification is not
timed. Cells that already have a reference are skipped, so the run resumes cleanly.

## What the result should show

**The two axes behave differently.** On the pairwise design the distance falls roughly
linearly in lambda and reaches zero between 0.6 and 0.7: the path enters the polytope. On the
n = 20 density axis it falls linearly all the way to lambda = 1 and is never zero before the
endpoint, because there the polytope is a low-dimensional set in the space of all observed
cells, and a generic segment meets it only at p_inside.

**Certifying gets much harder near the boundary; racing does not.** The criterion is relative,
so as OPT shrinks the required absolute gap shrinks with it, and for the certificate it
shrinks with OPT squared, because the lower bound comes from a reduced cost. At n = 20, 1 %
density, the certificate needs 7,800 ADMM iterations at lambda = 0 and 288,400 at lambda = 0.9,
while the race to 1 % of OPT needs 600 and 3,800.

Part of that is the stopping rule and part is real. Measured at a fixed absolute accuracy the
work barely changes along the path, which isolates the rule. What is genuine is that as the
point approaches the polytope its projection lands on a wider face, so the optimal mixture
needs more rankings: at n = 25 the support grows from 159 rankings at lambda = 0 to 266 at
lambda = 0.5.
