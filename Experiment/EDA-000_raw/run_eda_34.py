"""EDA-3 cross-modal alignment (linear CCA ceiling, speaker-disjoint) + EDA-4 gender anatomy."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"
R = {}
Xf, Xv, yf, _, txt, gmap = load_train()
vgrp = np.load(OUT / "voice_group.npy")
gender = np.array([gmap[s] for s in yf])
SEEDS = [1, 2, 3]

# ---------------------------------------------------------------- EDA-3
e3 = {"config": {}, "per_seed": {}, "summary": {}}
configs = {
    "cca_k32": dict(k=32, reg=1e-1, pca_x=256),
    "cca_k64": dict(k=64, reg=1e-1, pca_x=256),
    "cca_k128": dict(k=128, reg=1e-1, pca_x=512, pca_y=None),
    "cca_k64_pca512": dict(k=64, reg=1e-1, pca_x=512),
}
rows = []
dist_store = {}
for seed in SEEDS:
    tr, va = speaker_split(yf, seed)
    for cname, cfg in configs.items():
        m = RidgeCCA(**cfg).fit(Xf[tr], Xv[tr])
        Ef, Ev = m.transform_x(Xf), m.transform_y(Xv)
        for split_name, mask in [("val", va), ("train", tr)]:
            idx = np.where(mask)[0]
            for sg in [False, True]:
                tag = "gender" if sg else "no_gender"
                fi, vj, lab = build_trials(yf[idx], gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg)
                r = eval_trials(Ef[idx], Ev[idx], fi, vj, lab)
                rows.append(dict(seed=seed, cfg=cname, split=split_name, protocol=tag, **r))
                if cname == "cca_k64" and split_name == "val":
                    cos = np.sum(Ef[idx][fi] * Ev[idx][vj], 1)
                    dist_store[(seed, tag)] = (2 - 2 * cos, lab, gender[idx][fi], gender[idx][vj])
        if seed == 1:
            e3["config"][cname] = dict(cfg, canon_corr_top5=[float(c) for c in m.canon_corr[:5]],
                                       canon_corr_mean=float(m.canon_corr.mean()))
df = pd.DataFrame(rows)
df.to_csv(OUT / "eda3_cca_trials.csv", index=False)
summ = df.groupby(["cfg", "split", "protocol"]).agg(eer_mean=("eer", "mean"), eer_std=("eer", "std"),
                                                    auc=("auc", "mean"), margin=("margin", "mean"),
                                                    cohen_d=("cohen_d", "mean")).round(2)
print("EDA-3 CCA linear baseline (mean over 3 speaker-disjoint seeds)\n", summ.to_string())
e3["summary"] = {f"{a}|{b}|{c}": v for (a, b, c), v in summ.to_dict("index").items()}
e3["per_seed"] = df[df.cfg == "cca_k64"].round(2).to_dict("records")

# chance sanity: shuffled speakers
tr, va = speaker_split(yf, 1)
m = RidgeCCA(k=64, reg=1e-1, pca_x=256).fit(Xf[tr], Xv[tr])
idx = np.where(va)[0]
perm = np.random.RandomState(0).permutation(yf[idx])
fi, vj, lab = build_trials(perm, gmap, seed=1, n_pos=3000, n_neg=3000)
e3["sanity_shuffled_labels_eer"] = eval_trials(m.transform_x(Xf[idx]), m.transform_y(Xv[idx]), fi, vj, lab)["eer"]
R["EDA3"] = e3

# distance distributions plot (seed 1, val, k64)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, tag in zip(axes, ["no_gender", "gender"]):
    d, lab, gi, gj = dist_store[(1, tag)]
    bins = np.linspace(0, 2.5, 60)
    ax.hist(d[lab == 1], bins, alpha=.85, color=PALETTE["blue"], label="positive (same id)")
    ax.hist(d[lab == 0], bins, alpha=.6, color=PALETTE["orange"], label="negative")
    ax.set_title(f"EDA-3  CCA-64 val distance — {tag}", fontsize=10); ax.set_xlabel("2 − 2·cos")
    ax.legend(fontsize=8, frameon=False); ax.grid(color=PALETTE["grid"]); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "eda3_cca_dist.png", dpi=130); plt.close()

# ---------------------------------------------------------------- EDA-4
e4 = {}
# 4a. negative anatomy on CCA val space: same-gender vs cross-gender negatives
neg_rows = []
for seed in SEEDS:
    d, lab, gi, gj = dist_store[(seed, "no_gender")]
    pos = d[lab == 1]
    neg_same = d[(lab == 0) & (gi == gj)]
    neg_cross = d[(lab == 0) & (gi != gj)]
    # EER restricted to same-gender negatives vs cross-gender negatives
    def eer_sub(negs):
        s = np.concatenate([-pos, -negs]); l = np.concatenate([np.ones(len(pos)), np.zeros(len(negs))])
        return eer_from_scores(s, l)
    neg_rows.append(dict(seed=seed, pos_mean=pos.mean(), neg_same_mean=neg_same.mean(), neg_cross_mean=neg_cross.mean(),
                         eer_same_gender_neg=eer_sub(neg_same), eer_cross_gender_neg=eer_sub(neg_cross),
                         frac_neg_same_gender=float(((gi == gj) & (lab == 0)).sum() / (lab == 0).sum())))
nd = pd.DataFrame(neg_rows)
e4["cca_negative_anatomy_mean"] = nd.mean().round(3).to_dict()
e4["cca_negative_anatomy_per_seed"] = nd.round(3).to_dict("records")
print("EDA-4 negative anatomy (CCA-64 val)\n", nd.round(3).to_string())

# 4b. gender probe: logistic regression, speaker-disjoint (3 seeds), on PCA-256/192 standardized
probe = {}
for name, X in [("face", Xf), ("voice", Xv)]:
    accs = []
    for seed in SEEDS:
        tr, va = speaker_split(yf, seed)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        p = PCA(n_components=min(128, X.shape[1] - 1), random_state=0).fit(Z[tr])
        A = p.transform(Z)
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(A[tr], gender[tr])
        accs.append(float((clf.predict(A[va]) == gender[va]).mean()))
    probe[name] = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)), per_seed=accs)
# gender probe in CCA space (k=64) — does the shared space carry gender?
accs_f, accs_v = [], []
for seed in SEEDS:
    tr, va = speaker_split(yf, seed)
    m = RidgeCCA(k=64, reg=1e-1, pca_x=256).fit(Xf[tr], Xv[tr])
    Ef, Ev = m.transform_x(Xf), m.transform_y(Xv)
    accs_f.append(float((LogisticRegression(max_iter=2000).fit(Ef[tr], gender[tr]).predict(Ef[va]) == gender[va]).mean()))
    accs_v.append(float((LogisticRegression(max_iter=2000).fit(Ev[tr], gender[tr]).predict(Ev[va]) == gender[va]).mean()))
probe["cca64_face"] = dict(acc_mean=float(np.mean(accs_f)), per_seed=accs_f)
probe["cca64_voice"] = dict(acc_mean=float(np.mean(accs_v)), per_seed=accs_v)
# how much of CCA variance is gender? cosine between gender-centroid direction and canonical dims
e4["gender_probe"] = probe
print("EDA-4 gender probe", json.dumps(probe, indent=1))

# 4c. gender-direction projection: remove gender direction from CCA space, re-evaluate
proj_rows = []
for seed in SEEDS:
    tr, va = speaker_split(yf, seed)
    m = RidgeCCA(k=64, reg=1e-1, pca_x=256).fit(Xf[tr], Xv[tr])
    Ef, Ev = m.transform_x(Xf), m.transform_y(Xv)
    idx = np.where(va)[0]
    for name, E in [("face", Ef), ("voice", Ev)]:
        pass
    # project out the male-female centroid direction (fit on train) in each view
    def deb(E):
        dvec = E[tr][gender[tr] == "m"].mean(0) - E[tr][gender[tr] == "f"].mean(0)
        dvec /= np.linalg.norm(dvec)
        return l2n(E - np.outer(E @ dvec, dvec))
    Ef2, Ev2 = deb(Ef), deb(Ev)
    for sg in [False, True]:
        tag = "gender" if sg else "no_gender"
        fi, vj, lab = build_trials(yf[idx], gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg)
        proj_rows.append(dict(seed=seed, protocol=tag, variant="cca64",
                              eer=eval_trials(Ef[idx], Ev[idx], fi, vj, lab)["eer"]))
        proj_rows.append(dict(seed=seed, protocol=tag, variant="cca64_gender_projected_out",
                              eer=eval_trials(Ef2[idx], Ev2[idx], fi, vj, lab)["eer"]))
pdf = pd.DataFrame(proj_rows).groupby(["variant", "protocol"]).eer.agg(["mean", "std"]).round(2)
print("EDA-4 gender projection-out\n", pdf.to_string())
e4["gender_projection_out"] = {f"{a}|{b}": v for (a, b), v in pdf.to_dict("index").items()}

# 4d. how much of unimodal identity separation is gender? (train, raw centred cosine)
Fzf = l2n(Xf - Xf.mean(0)); Fzv = l2n(Xv - Xv.mean(0))
for name, E in [("face", Fzf), ("voice", Fzv)]:
    rng = np.random.RandomState(0)
    a, b = rng.randint(len(E), size=20000), rng.randint(len(E), size=20000)
    keep = yf[a] != yf[b]
    a, b = a[keep], b[keep]
    cos = np.sum(E[a] * E[b], 1)
    same_g = gender[a] == gender[b]
    e4[f"{name}_raw_neg_cos_same_gender"] = float(cos[same_g].mean())
    e4[f"{name}_raw_neg_cos_cross_gender"] = float(cos[~same_g].mean())
    e4[f"{name}_raw_gender_gap_cohen_d"] = float(cohens_d(cos[same_g], cos[~same_g]))
R["EDA4"] = e4

fig, ax = plt.subplots(figsize=(6.5, 3.4))
d, lab, gi, gj = dist_store[(1, "no_gender")]
bins = np.linspace(0, 2.5, 60)
ax.hist(d[lab == 1], bins, alpha=.85, color=PALETTE["blue"], label="positive")
ax.hist(d[(lab == 0) & (gi == gj)], bins, alpha=.6, color=PALETTE["orange"], label="neg · same gender")
ax.hist(d[(lab == 0) & (gi != gj)], bins, alpha=.6, color=PALETTE["aqua"], label="neg · cross gender")
ax.set_title("EDA-4  CCA-64 val: negatives split by gender (seed 1)", fontsize=10); ax.set_xlabel("2 − 2·cos")
ax.legend(fontsize=8, frameon=False); ax.grid(color=PALETTE["grid"]); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "eda4_gender_neg.png", dpi=130); plt.close()

json.dump(R, open(OUT / "eda_34.json", "w"), indent=1, default=float)
print("saved")
