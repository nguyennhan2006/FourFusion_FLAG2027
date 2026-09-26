"""FUSE-03 — constrained static rank weights per cell, chosen on pseudo-labels, scored OUT-OF-FOLD.

Per cell the members are pre-registered (<= 3). Weights live on the simplex (step 0.05).
For each of 5 identity-disjoint folds: pick weights on the other 4 folds, evaluate on the held-out fold,
compare with the CURRENT submitted fusion (equal weights of the FUSE-02 members) on the same fold.
Keep only if mean OOF gain >= 0.25 EER and >= 4/5 folds improve.
"""
import itertools, numpy as np, pandas as pd
from common import cell_data, eer, group_folds, CELLS
MEMBERS = {("no_gender", "English"): (["007", "010B", "003c"], {"007": .5, "010B": .5}),
           ("gender", "English"): (["007", "010B", "003c"], {"007": .5, "010B": .5}),
           ("no_gender", "Bangla"): (["003c", "r2_mix", "r2_192"], {"003c": .5, "r2_mix": .5}),
           ("gender", "Bangla"): (["003c", "r2_mix"], {"003c": .5, "r2_mix": .5})}
STEP = 0.05


def simplex(n):
    g = np.round(np.arange(0, 1 + 1e-9, STEP), 2)
    return [w for w in itertools.product(g, repeat=n) if abs(sum(w) - 1) < 1e-6]


rows = []
for k, (mem, cur) in MEMBERS.items():
    D = cell_data(k, mem); kn = D["known"]
    y, fc = D["y"][kn], D["fc"][kn]; R = np.stack([D["R"][m][kn] for m in mem])
    cw = np.array([cur.get(m, 0.0) for m in mem])
    W = simplex(len(mem))
    folds = group_folds(fc)
    gains, chosen = [], []
    for va in folds:
        tr = ~va
        best = min(W, key=lambda w: eer(np.dot(w, R[:, tr]), y[tr]))
        chosen.append(best)
        gains.append(eer(cw @ R[:, va], y[va]) - eer(np.dot(best, R[:, va]), y[va]))
    wfull = min(W, key=lambda w: eer(np.dot(w, R), y))
    rows.append(dict(cell="/".join(k), members="+".join(mem), current=dict(zip(mem, cw)),
                     full_best=dict(zip(mem, wfull)), eer_current=round(eer(cw @ R, y), 2),
                     eer_full_best_insample=round(eer(np.dot(wfull, R), y), 2),
                     oof_gain_mean=round(float(np.mean(gains)), 2), folds_improved=f"{sum(g > 0 for g in gains)}/5",
                     fold_weights=[dict(zip(mem, w)) for w in chosen]))
    r = rows[-1]
    keep = r["oof_gain_mean"] >= 0.25 and sum(g > 0 for g in gains) >= 4
    print(f"\n{r['cell']}: current {r['eer_current']} | in-sample best {r['eer_full_best_insample']} w={r['full_best']}")
    print(f"   OOF gain {r['oof_gain_mean']:+.2f}, folds improved {r['folds_improved']}, per-fold {np.round(gains, 2)} -> {'KEEP' if keep else 'drop'}")
    print("   fold weights:", [tuple(round(v, 2) for v in w) for w in chosen])
pd.DataFrame(rows).to_json("fuse03.json", orient="records", indent=1)
