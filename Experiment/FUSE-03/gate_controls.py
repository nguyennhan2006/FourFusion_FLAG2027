"""Controls for the FUSE-03 gate (log 41): is the g/Bn gain about duration, or just re-weighting 003c vs r2_mix?

Same OOF protocol as fuse03.py / dur02.py: 5 folds over pseudo-identities, gain = EER(0.5/0.5) - EER(candidate)
on the held-out fold. Controls: static weight on three grids, real duration, duration shuffled within the cell,
and a gate on model disagreement |rank_003c - rank_r2mix|.
"""
import itertools, numpy as np
from common import cell_data, eer, group_folds

sig = lambda x: 1 / (1 + np.exp(-x))
G2 = list(itertools.product(np.arange(-3, 3.01, .5), np.arange(-4, 4.01, .5)))      # (a, b) of dur02.py


def oof_gate(A, B, F, y, folds, feat, grid):
    gains = []
    for va in folds:
        tr = ~va
        a, b = min(grid, key=lambda p: eer(sig(p[0] * feat[tr] + p[1]) * A[tr] + (1 - sig(p[0] * feat[tr] + p[1])) * B[tr], y[tr]))
        w = sig(a * feat[va] + b)
        gains.append(eer(F[va], y[va]) - eer(w * A[va] + (1 - w) * B[va], y[va]))
    return np.mean(gains), sum(g > 0 for g in gains)


def oof_static(A, B, F, y, folds, W):
    gains = []
    for va in folds:
        tr = ~va
        w = min(W, key=lambda w: eer(w * A[tr] + (1 - w) * B[tr], y[tr]))
        gains.append(eer(F[va], y[va]) - eer(w * A[va] + (1 - w) * B[va], y[va]))
    return np.mean(gains), sum(g > 0 for g in gains)


for k in [("gender", "Bangla"), ("no_gender", "Bangla")]:
    D = cell_data(k, ["003c", "r2_mix"]); kn = D["known"]
    y, fc, d = D["y"][kn], D["fc"][kn], D["dur"][kn]
    A, B = D["R"]["003c"][kn], D["R"]["r2_mix"][kn]; F = (A + B) / 2
    folds = group_folds(fc)
    logd = np.log(np.maximum(d, .5))
    dis = np.abs(A - B); dis = (dis - dis.mean()) / dis.std()
    res = {"static, simplex step 0.05 (fuse03.py)": oof_static(A, B, F, y, folds, np.round(np.arange(0, 1.0001, .05), 3)),
           "static, simplex step 0.02": oof_static(A, B, F, y, folds, np.round(np.arange(0, 1.0001, .02), 3)),
           "static, sigmoid(b) b step 0.25": oof_static(A, B, F, y, folds, sig(np.arange(-4, 4.01, .25))),
           "duration gate (2 param, FUSE-03)": oof_gate(A, B, F, y, folds, logd, G2)}
    sh = [oof_gate(A, B, F, y, folds, np.random.RandomState(s).permutation(logd), G2) for s in range(5)]
    res["duration SHUFFLED, mean of 5 perms"] = (np.mean([g for g, _ in sh]), [int(n) for _, n in sh])
    res["disagreement gate (2 param)"] = oof_gate(A, B, F, y, folds, dis, G2)
    print("/".join(k), f"n={len(y)}")
    for n, (g, f) in res.items():
        print(f"   {n:40s} OOF gain vs 0.5/0.5 {g:+.2f}   folds improved {f}")
