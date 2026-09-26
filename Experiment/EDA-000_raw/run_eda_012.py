"""EDA-0 integrity, EDA-1 speaker distribution, EDA-2 feature geometry (train)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
R = {}

Xf, Xv, yf, yv, txt, gmap = load_train()

# ---------------------------------------------------------------- EDA-0
e0 = {}
e0["face_shape"] = list(Xf.shape); e0["voice_shape"] = list(Xv.shape)
e0["label_aligned"] = bool((yf == yv).all())
e0["label_matches_txt"] = True  # asserted in load_train (csv int label == txt speaker int)
e0["n_speakers"] = int(len(set(yf)))
for name, X in [("face", Xf), ("voice", Xv)]:
    e0[f"{name}_nan"] = int(np.isnan(X).sum()); e0[f"{name}_inf"] = int(np.isinf(X).sum())
    var = X.var(0)
    e0[f"{name}_const_dims"] = int((var < 1e-8).sum())
    e0[f"{name}_nearzero_var_dims(<1e-3*median)"] = int((var < 1e-3 * np.median(var)).sum())
    nrm = np.linalg.norm(X, axis=1)
    e0[f"{name}_zero_vectors"] = int((nrm < 1e-6).sum())
    e0[f"{name}_dup_rows"] = int(len(X) - len(np.unique(np.round(X, 4), axis=0)))
    e0[f"{name}_norm_mean/std/min/max"] = [float(nrm.mean()), float(nrm.std()), float(nrm.min()), float(nrm.max())]
    e0[f"{name}_value_min/max"] = [float(X.min()), float(X.max())]
    e0[f"{name}_frac_zero_values"] = float((X == 0).mean())
    e0[f"{name}_frac_negative"] = float((X < 0).mean())
    e0[f"{name}_abs>100"] = int((np.abs(X) > 100).sum())
both = np.hstack([np.round(Xf, 4), np.round(Xv, 4)])
e0["pair_dup_rows"] = int(len(both) - len(np.unique(both, axis=0)))
# duplicate face rows: are they within the same speaker?
_, inv, cnt = np.unique(np.round(Xf, 4), axis=0, return_inverse=True, return_counts=True)
dup_groups = [np.where(inv == g)[0] for g in np.where(cnt > 1)[0]]
e0["face_dup_groups"] = len(dup_groups)
e0["face_dup_cross_speaker_groups"] = int(sum(len(set(yf[g])) > 1 for g in dup_groups))
e0["face_dup_max_group"] = int(max([len(g) for g in dup_groups], default=0))
e0["txt_label_col_unique"] = sorted(txt.label.unique().tolist())
_, vgrp, vcnt = np.unique(np.round(Xv, 4), axis=0, return_inverse=True, return_counts=True)
_, fgrp, fcnt = np.unique(np.round(Xf, 4), axis=0, return_inverse=True, return_counts=True)
e0["voice_unique_vectors"] = int(len(vcnt)); e0["face_unique_vectors"] = int(len(fcnt))
e0["voice_dup_group_size_hist"] = np.bincount(vcnt).tolist()
e0["voice_dup_groups_cross_speaker"] = int(sum(len(set(yf[vgrp == g])) > 1 for g in np.where(vcnt > 1)[0]))
np.save(OUT / "voice_group.npy", vgrp); np.save(OUT / "face_group.npy", fgrp)
R["EDA0"] = e0
print("EDA-0", json.dumps(e0, indent=1))

# ---------------------------------------------------------------- EDA-1
e1 = {}
cnt = pd.Series(yf).value_counts().sort_index()
g = pd.Series({s: gmap[s] for s in cnt.index})
e1["samples_per_speaker"] = dict(min=int(cnt.min()), max=int(cnt.max()), median=float(cnt.median()),
                                 mean=float(cnt.mean()), std=float(cnt.std()), ratio_max_min=float(cnt.max() / cnt.min()))
e1["gender_counts_speakers"] = g.value_counts().to_dict()
e1["gender_counts_samples"] = pd.Series([gmap[s] for s in yf]).value_counts().to_dict()
e1["top5"] = cnt.sort_values(ascending=False).head(5).to_dict()
e1["bottom5"] = cnt.sort_values().head(5).to_dict()
uvps = pd.Series(vgrp).groupby(yf).nunique()
e1["unique_voice_vectors_per_speaker"] = dict(min=int(uvps.min()), max=int(uvps.max()), median=float(uvps.median()),
                                              speakers_with_le5=[str(k) for k in uvps[uvps <= 5].index])
ufps = pd.Series(fgrp).groupby(yf).nunique()
e1["unique_face_vectors_per_speaker"] = dict(min=int(ufps.min()), max=int(ufps.max()), median=float(ufps.median()))

# raw unimodal identity signal (cosine), all 70 speakers, no fitting -> no leakage
Ff, Fv = l2n(Xf), l2n(Xv)
Fzf = l2n(Xf - Xf.mean(0)); Fzv = l2n(Xv - Xv.mean(0))   # mean-centred variant
for sg in [False, True]:
    tag = "gender" if sg else "no_gender"
    fi, vj, lab = build_trials(yf, gmap, seed=1, n_pos=5000, n_neg=5000, same_gender=sg, group=fgrp)
    e1[f"face_face_raw_{tag}"] = eval_trials(Ff, Ff, fi, vj, lab)
    e1[f"face_face_centred_{tag}"] = eval_trials(Fzf, Fzf, fi, vj, lab)
    fi, vj, lab = build_trials(yf, gmap, seed=1, n_pos=5000, n_neg=5000, same_gender=sg, group=vgrp)
    e1[f"voice_voice_raw_{tag}"] = eval_trials(Fv, Fv, fi, vj, lab)
    e1[f"voice_voice_centred_{tag}"] = eval_trials(Fzv, Fzv, fi, vj, lab)
    fi, vj, lab = build_trials(yf, gmap, seed=1, n_pos=5000, n_neg=5000, same_gender=sg)  # leaky (same clip allowed)
    e1[f"voice_voice_centred_{tag}_LEAKY_sameclip_allowed"] = eval_trials(Fzv, Fzv, fi, vj, lab)
# same-clip vs different-clip positives: face similarity when the voice clip is shared
fi, vj, lab = build_trials(yf, gmap, seed=1, n_pos=8000, n_neg=10, same_gender=False)
p = lab == 1
same_clip = vgrp[fi[p]] == vgrp[vj[p]]
e1["pos_frac_same_voice_clip"] = float(same_clip.mean())
cos = np.sum(Fzf[fi[p]] * Fzf[vj[p]], axis=1)
e1["face_pos_cos_same_clip"] = float(cos[same_clip].mean()); e1["face_pos_cos_diff_clip"] = float(cos[~same_clip].mean())
# per-speaker intra-class compactness
intra = {}
for name, E in [("face", Fzf), ("voice", Fzv)]:
    vals = []
    for s in cnt.index:
        idx = np.where(yf == s)[0]
        c = l2n(E[idx].mean(0, keepdims=True))
        vals.append(float((E[idx] @ c.T).mean()))
    intra[name] = vals
e1["mean_cos_to_speaker_centroid"] = {k: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in intra.items()}
R["EDA1"] = e1
print("EDA-1", json.dumps(e1, indent=1, default=float))

# plot speaker histogram
fig, ax = plt.subplots(figsize=(10, 3.2))
cols = [PALETTE["blue"] if g[s] == "m" else PALETTE["orange"] for s in cnt.index]
ax.bar(range(len(cnt)), cnt.values, color=cols, width=0.8)
ax.set_xticks(range(0, 70, 5)); ax.set_xticklabels(cnt.index[::5], rotation=90, fontsize=7)
ax.set_ylabel("samples"); ax.set_title("EDA-1  Samples per speaker (blue = male, orange = female)", fontsize=10)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y", color=PALETTE["grid"]); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(OUT / "eda1_speaker_hist.png", dpi=130); plt.close()

# ---------------------------------------------------------------- EDA-2
e2 = {}
for name, X in [("face", Xf), ("voice", Xv)]:
    var = X.var(0)
    Z = (X - X.mean(0))
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    cum = np.cumsum(ev)
    e2[f"{name}_var_pct"] = {p: float(np.percentile(var, p)) for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]}
    e2[f"{name}_top1_var_share"] = float(ev[0])
    e2[f"{name}_pca_cum"] = {k: float(cum[k - 1]) for k in [8, 16, 32, 50, 64, 128, 192, 256, 512, 1024] if k <= len(cum)}
    e2[f"{name}_dims_for_90/95/99"] = [int(np.searchsorted(cum, q) + 1) for q in [.9, .95, .99]]
    # participation ratio = effective dimensionality
    e2[f"{name}_effective_dim(PR)"] = float((S ** 2).sum() ** 2 / (S ** 4).sum())
    # mean abs pairwise correlation on a random subset of dims
    rng = np.random.RandomState(0); sub = rng.choice(X.shape[1], min(300, X.shape[1]), replace=False)
    C = np.corrcoef(X[:, sub].T); C = np.nan_to_num(C)
    off = C[np.triu_indices_from(C, 1)]
    e2[f"{name}_mean_abs_corr"] = float(np.abs(off).mean()); e2[f"{name}_frac_|corr|>0.5"] = float((np.abs(off) > .5).mean())
    # anisotropy: mean cosine between random pairs (raw vs centred)
    a, b = rng.randint(len(X), size=3000), rng.randint(len(X), size=3000)
    e2[f"{name}_mean_cos_random_pairs_raw"] = float(np.sum(l2n(X)[a] * l2n(X)[b], 1).mean())
    e2[f"{name}_mean_cos_random_pairs_centred"] = float(np.sum(l2n(Z)[a] * l2n(Z)[b], 1).mean())
    # per-dim mean magnitude vs std: does one dim dominate?
    e2[f"{name}_top5_var_dims"] = [int(i) for i in np.argsort(var)[-5:][::-1]]
    e2[f"{name}_top5_var_values"] = [float(var[i]) for i in np.argsort(var)[-5:][::-1]]
    np.save(OUT / f"pca_cum_{name}.npy", cum)
R["EDA2"] = e2
print("EDA-2", json.dumps(e2, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, name in zip(axes, ["face", "voice"]):
    cum = np.load(OUT / f"pca_cum_{name}.npy")
    ax.plot(np.arange(1, len(cum) + 1), cum, color=PALETTE["blue"], lw=2)
    for k in [50, 128, 256]:
        if k <= len(cum):
            ax.axvline(k, color=PALETTE["muted"], lw=0.8, ls=":")
            ax.text(k, 0.05, f"{k}: {cum[k-1]:.2f}", fontsize=7, rotation=90, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_title(f"EDA-2  PCA cumulative explained variance — {name}", fontsize=10)
    ax.set_xlabel("components (log)"); ax.grid(color=PALETTE["grid"]); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "eda2_pca.png", dpi=130); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, (name, X) in zip(axes, [("face", Xf), ("voice", Xv)]):
    ax.hist(np.linalg.norm(X, axis=1), bins=60, color=PALETTE["blue"])
    ax.set_title(f"EDA-2  L2 norm — {name}", fontsize=10); ax.grid(color=PALETTE["grid"]); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "eda2_norms.png", dpi=130); plt.close()

json.dump(R, open(OUT / "eda_012.json", "w"), indent=1, default=float)
print("saved", OUT)
