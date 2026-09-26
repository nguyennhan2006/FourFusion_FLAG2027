"""EDA-3b: CCA capacity sweep (k, reg, pca) — how much shared identity subspace generalises to unseen speakers?
Also writes a linear-CCA submission zip (dev scored with +d^2, lower = same) as a non-deep reference."""
import json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"
Xf, Xv, yf, _, txt, gmap = load_train()
SEEDS = [1, 2, 3]

# PCA once per seed (expensive part), then vary k / reg cheaply
rows = []
for seed in SEEDS:
    tr, va = speaker_split(yf, seed)
    idx = np.where(va)[0]
    trials = {sg: build_trials(yf[idx], gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg) for sg in [False, True]}
    for pca_x in [128, 256]:
        base = RidgeCCA(k=1, reg=1e-1, pca_x=pca_x)
        base.mx, base.sx, base.Px = base._prep(Xf[tr], pca_x)
        base.my, base.sy, base.Py = base._prep(Xv[tr], None)
        A = ((Xf - base.mx) / base.sx) @ base.Px
        B = ((Xv - base.my) / base.sy) @ base.Py
        for reg in [1e-2, 1e-1, 1.0, 10.0]:
            m = RidgeCCA(k=64, reg=reg, pca_x=None, pca_y=None).fit(A[tr], B[tr])
            for k in [4, 8, 16, 24, 32, 48, 64]:
                Ef = l2n(((A - m.mx) / m.sx) @ m.Wx[:, :k]); Ev = l2n(((B - m.my) / m.sy) @ m.Wy[:, :k])
                for sg, (fi, vj, lab) in trials.items():
                    r = eval_trials(Ef[idx], Ev[idx], fi, vj, lab)
                    rows.append(dict(seed=seed, pca_x=pca_x, reg=reg, k=k, protocol="gender" if sg else "no_gender", eer=r["eer"], auc=r["auc"]))
df = pd.DataFrame(rows)
df.to_csv(OUT / "eda3b_sweep.csv", index=False)
piv = df.groupby(["pca_x", "reg", "k", "protocol"]).eer.mean().unstack("protocol").round(2)
piv["mean"] = piv.mean(1).round(2)
print(piv.to_string())
best = piv["mean"].idxmin()
print("BEST (pca_x, reg, k):", best, piv.loc[best].to_dict())
json.dump(dict(best=dict(pca_x=int(best[0]), reg=float(best[1]), k=int(best[2]), **piv.loc[best].to_dict()),
               table={f"{a}|{b}|{c}": v for (a, b, c), v in piv.to_dict("index").items()}),
          open(OUT / "eda_3b.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(7, 3.6))
sub = df[(df.pca_x == best[0]) & (df.reg == best[1])].groupby(["k", "protocol"]).eer.mean().unstack()
ax.plot(sub.index, sub["no_gender"], "-o", color=PALETTE["blue"], lw=2, ms=5, label="no_gender")
ax.plot(sub.index, sub["gender"], "-o", color=PALETTE["orange"], lw=2, ms=5, label="gender")
ax.axhline(36.31, color=PALETTE["muted"], ls=":", lw=1); ax.text(64, 36.6, "FOP CB ng/En 36.31", fontsize=7, ha="right", color=PALETTE["muted"])
ax.axhline(43.38, color=PALETTE["muted"], ls=":", lw=1); ax.text(64, 43.7, "FOP CB g/En 43.38", fontsize=7, ha="right", color=PALETTE["muted"])
ax.set_xlabel("CCA dims k"); ax.set_ylabel("internal EER (%) — 3 speaker-disjoint seeds")
ax.set_title(f"EDA-3b  Linear CCA capacity (pca_x={best[0]}, reg={best[1]})", fontsize=10)
ax.legend(frameon=False, fontsize=8); ax.grid(color=PALETTE["grid"]); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "eda3b_k_sweep.png", dpi=130); plt.close()

# ------------------------------------------------------- linear reference submission (fit on all 70 speakers)
pca_x, reg, k = int(best[0]), float(best[1]), int(best[2])
m = RidgeCCA(k=k, reg=reg, pca_x=pca_x).fit(Xf, Xv)
sub_dir = OUT / "submission_cca"
names = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
with zipfile.ZipFile(OUT / "submission_cca_linear.zip", "w") as zf:
    for (prot, lang), fn in names.items():
        a, b, t = load_dev(prot, lang)
        d2 = 2 - 2 * np.sum(m.transform_x(a) * m.transform_y(b), 1)   # +d^2, lower = same
        lines = "\n".join(f"{pid} {s:.6f}" for pid, s in zip(t.pair_id, d2)) + "\n"
        zf.writestr(fn, lines)
        print(fn, len(d2), "mean d2 %.3f" % d2.mean())
print("wrote", OUT / "submission_cca_linear.zip", "config", dict(pca_x=pca_x, reg=reg, k=k))
