"""DUR-02 — does r2_mix (Bangla-length crops) win specifically on SHORT Bangla clips?
Then a 2-parameter duration gate w(d) = sigmoid(a log d + b) on r2_mix vs 003c, scored out-of-fold.
Gate to keep: gain concentrated in short bins AND OOF gain >= 0.25 with >= 4/5 folds.
"""
import itertools, numpy as np, pandas as pd
from common import cell_data, eer, group_folds
BINS = [0, 3, 5, 8, 12, 999]
rows = []
for k in [("no_gender", "Bangla"), ("gender", "Bangla"), ("no_gender", "English"), ("gender", "English")]:
    mem = ["003c", "r2_mix"] if k[1] == "Bangla" else ["007", "010B"]
    D = cell_data(k, mem); kn = D["known"]
    y, fc, d = D["y"][kn], D["fc"][kn], D["dur"][kn]
    A, B = D["R"][mem[0]][kn], D["R"][mem[1]][kn]; F = (A + B) / 2
    b = np.digitize(d, BINS[1:-1])
    for i in range(len(BINS) - 1):
        m = b == i
        if m.sum() < 60 or y[m].min() == y[m].max(): continue
        rows.append(dict(cell="/".join(k), bin=f"{BINS[i]}-{BINS[i+1]}s", n=int(m.sum()),
                         **{mem[0]: round(eer(A[m], y[m]), 2), mem[1]: round(eer(B[m], y[m]), 2)},
                         fused=round(eer(F[m], y[m]), 2)))
    if k[1] == "Bangla":
        grid = list(itertools.product(np.arange(-3, 3.01, .5), np.arange(-4, 4.01, .5)))
        gate = lambda a, bb, idx: (1 / (1 + np.exp(-(a * np.log(np.maximum(d[idx], .5)) + bb))))
        score = lambda a, bb, idx: (1 - gate(a, bb, idx)) * B[idx] + gate(a, bb, idx) * A[idx]  # gate = weight on 003c
        gains = []
        for va in group_folds(fc):
            tr = np.where(~va)[0]; vi = np.where(va)[0]
            a, bb = min(grid, key=lambda p: eer(score(*p, tr), y[tr]))
            gains.append(eer(F[vi], y[vi]) - eer(score(a, bb, vi), y[vi]))
        allidx = np.arange(len(y)); a, bb = min(grid, key=lambda p: eer(score(*p, allidx), y))
        w = gate(a, bb, allidx)
        print(f"{'/'.join(k)} duration gate: OOF gain {np.mean(gains):+.2f}, folds {sum(g > 0 for g in gains)}/5 {np.round(gains, 2)} | "
              f"full-data a={a} b={bb}: weight on 003c at 2s {1/(1+np.exp(-(a*np.log(2)+bb))):.2f}, 5s {1/(1+np.exp(-(a*np.log(5)+bb))):.2f}, 12s {1/(1+np.exp(-(a*np.log(12)+bb))):.2f}")
T = pd.DataFrame(rows); T.to_csv("dur02_bins.csv", index=False)
print(); print(T.to_string(index=False))
