"""Correctness of the stream pricer against the mask kernel and brute force."""
import itertools
import math
import os
import sys
import time

import numpy as np
import cupy as cp

C = sys.argv[1]
sys.path.insert(0, C); sys.path.insert(0, os.path.join(C, "data"))
from gen_instances import gen_pairwise, gen_medium
from solver.flow_cg.cg_core import CGData
from solver.flow_cg.bf_pricing import bf as bf_kernel
from solver.flow_cg.pricing_stream import StreamPricer, colex_rank


def mask_price(model, scores):
    n = model.n; N = 1 << n
    V = cp.zeros(N, dtype=cp.float32); pred = cp.zeros(N, dtype=cp.uint8)
    masks = cp.asarray(model.choice_set, dtype=cp.uint32); offs = cp.asarray(model.offsets, dtype=cp.uint64)
    g = cp.asarray(scores.astype(np.float32))
    for c in range(1, n + 1):
        st = model.layer(c); ns = int(st.size)
        bf_kernel(((ns + 255) // 256,), (256,), (V, pred, masks, g, offs, st, ns))
    cp.cuda.Device().synchronize()
    D = N - 1; rank = np.zeros(n, dtype=np.int32); ph = cp.asnumpy(pred)
    for i in range(n):
        x = int(ph[D]); rank[x] = i; D ^= 1 << x
    return float(V[N - 1].get()), rank


def brute(n, masks, alts, scores):
    best = -1e9
    for perm in itertools.permutations(range(n)):
        pos = {a: i for i, a in enumerate(perm)}
        val = sum(float(scores[i]) for i in range(len(masks))
                  if min((a for a in range(n) if (int(masks[i]) >> a) & 1), key=lambda a: pos[a]) == alts[i])
        best = max(best, val)
    return best


def col_value(n, masks, alts, scores, rank_pos):
    return sum(float(scores[i]) for i in range(len(masks))
               if min((a for a in range(n) if (int(masks[i]) >> a) & 1), key=lambda a: rank_pos[a]) == alts[i])


# colex rank helper vs unranking order (ascending masks)
for n in (5, 9):
    for k in range(0, n + 1):
        masks = sorted(m for m in range(1 << n) if bin(m).count("1") == k)
        assert all(colex_rank(m) == i for i, m in enumerate(masks)), (n, k)
print("colex_rank matches ascending mask order")

rng = np.random.RandomState(1)
for n, gen in ((6, "pair"), (8, "pair"), (12, "med"), (16, "pair"), (18, "med"), (20, "pair")):
    cd, mp = gen_pairwise(n) if gen == "pair" else gen_medium(n, mult=3)
    model = CGData(cd, n, layers="stored")
    for trial in range(3):
        scores = rng.randn(mp).astype(np.float32) * (2.0 if trial == 0 else 0.3)
        vm, rm = mask_price(model, scores)
        pf = StreamPricer(n, model.choice_set, model.choice_alternative, model.offsets, mode="f32", k_pred=max(1, n - 4))
        vf, rf, mf = pf.price(scores)
        pu = StreamPricer(n, model.choice_set, model.choice_alternative, model.offsets, mode="u16", k_pred=max(1, n - 4))
        vu, ru, mu = pu.price(scores)
        p24 = StreamPricer(n, model.choice_set, model.choice_alternative, model.offsets, mode="u24", k_pred=max(1, n - 4))
        v24, r24, m24 = p24.price(scores)
        vf_col = col_value(n, model.choice_set, model.choice_alternative, scores, rf)
        vu_col = col_value(n, model.choice_set, model.choice_alternative, scores, ru)
        v24_col = col_value(n, model.choice_set, model.choice_alternative, scores, r24)
        ok_f = (vf == vm) and np.array_equal(rf, rm)
        ok_u = abs(vu - vm) <= mu + 1e-5 and abs(vu_col - vm) <= mu + 1e-4
        ok_24 = abs(v24 - vm) <= m24 + 1e-5 and abs(v24_col - vm) <= m24 + 1e-4
        line = "n=%2d %s trial %d: mask V=%.6f | f32 V=%.6f identical=%s | u16 V=%.6f margin=%.2e |V-Vmask|=%.2e col(u16)=%.6f ok=%s | u24 margin=%.2e |V-Vmask|=%.2e col(u24)=%.6f ok=%s" % (
            n, gen, trial, vm, vf, ok_f, vu, mu, abs(vu - vm), vu_col, ok_u, m24, abs(v24 - vm), v24_col, ok_24)
        if n <= 8 and trial == 0:
            vb = brute(n, model.choice_set, model.choice_alternative, scores)
            line += " | brute=%.6f match=%s" % (vb, abs(vb - vm) < 1e-5)
        print(line, flush=True)
        assert ok_f, "f32 stream must be bit-identical to the mask kernel"
        assert ok_u, "u16 value outside its margin"
        assert ok_24, "u24 value outside its margin"
print("ALL OK")
