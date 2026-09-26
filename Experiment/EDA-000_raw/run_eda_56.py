"""EDA-5 language shift (dev English vs Bangla) + EDA-6 train->dev shift. All unsupervised (dev has no labels)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"
R = {}
Xf, Xv, yf, _, txt, gmap = load_train()
gender = np.array([gmap[s] for s in yf])

dev = {}
for prot in ["no_gender", "gender"]:
    for lang in ["English", "Bangla"]:
        dev[(prot, lang)] = load_dev(prot, lang)
        print(prot, lang, dev[(prot, lang)][0].shape, dev[(prot, lang)][1].shape, len(dev[(prot, lang)][2]))

sets = {"train": (Xf, Xv)}
for (p, l), (a, b, _) in dev.items():
    sets[f"{p}/{l}"] = (a, b)

# ---------------------------------------------------------------- integrity of dev
e0 = {}
for k, (a, b) in sets.items():
    if k == "train": continue
    e0[k] = dict(face_shape=list(a.shape), voice_shape=list(b.shape),
                 nan=int(np.isnan(a).sum() + np.isnan(b).sum()),
                 face_unique=int(len(np.unique(np.round(a, 4), axis=0))),
                 voice_unique=int(len(np.unique(np.round(b, 4), axis=0))))
# overlap between protocol files (same underlying pool?) and between languages
def overlap(A, B):
    sa = set(map(tuple, np.round(A, 3))); sb = set(map(tuple, np.round(B, 3)))
    return len(sa & sb), len(sa), len(sb)
for mod, i in [("face", 0), ("voice", 1)]:
    e0[f"{mod}_overlap_nogender_vs_gender_English"] = overlap(dev[("no_gender", "English")][i], dev[("gender", "English")][i])
    e0[f"{mod}_overlap_nogender_vs_gender_Bangla"] = overlap(dev[("no_gender", "Bangla")][i], dev[("gender", "Bangla")][i])
    e0[f"{mod}_overlap_English_vs_Bangla_nogender"] = overlap(dev[("no_gender", "English")][i], dev[("no_gender", "Bangla")][i])
    e0[f"{mod}_overlap_dev_vs_train_nogender_English"] = overlap(dev[("no_gender", "English")][i], sets["train"][i])
R["DEV_INTEGRITY"] = e0
print(json.dumps(e0, indent=1))

# ---------------------------------------------------------------- shared PCA basis (fit on train only)
stats = {}
pcas = {}
for mod, i in [("face", 0), ("voice", 1)]:
    Xtr = sets["train"][i]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    p = PCA(n_components=64, random_state=0).fit((Xtr - mu) / sd)
    pcas[mod] = (mu, sd, p)
    Ztr = p.transform((Xtr - mu) / sd)
    cov_inv = np.linalg.inv(np.cov(Ztr.T) + 1e-3 * np.eye(64))
    ctr = Ztr.mean(0)
    for k, XY in sets.items():
        X = XY[i]
        Z = p.transform((X - mu) / sd)
        maha = np.sqrt(np.einsum("ij,jk,ik->i", Z - ctr, cov_inv, Z - ctr))
        nrm = np.linalg.norm(X, axis=1)
        stats[(mod, k)] = dict(
            norm_mean=float(nrm.mean()), norm_std=float(nrm.std()),
            frac_zero=float((X == 0).mean()),
            centroid_dist_pca=float(np.linalg.norm(Z.mean(0) - ctr)),
            centroid_dist_pca_in_train_sd=float(np.linalg.norm((Z.mean(0) - ctr) / Ztr.std(0))),
            maha_mean=float(maha.mean()), maha_p95=float(np.percentile(maha, 95)),
            var_ratio_vs_train=float(Z.var(0).sum() / Ztr.var(0).sum()),
            knn5_cos_dist_to_train=float(knn_dist_to(Xtr, X).mean()) if k != "train" else None,
            mmd_vs_train=mmd_rbf(Ztr, Z, n=800) if k != "train" else 0.0,
        )
# kNN of train to itself (held-out speakers) as a reference for knn dist
for mod, i in [("face", 0), ("voice", 1)]:
    tr, va = speaker_split(yf, 1)
    stats[(mod, "train")]["knn5_cos_dist_to_train(heldout_speakers_ref)"] = float(knn_dist_to(sets["train"][i][tr], sets["train"][i][va]).mean())
    trZ = pcas[mod][2].transform((sets["train"][i] - pcas[mod][0]) / pcas[mod][1])
    stats[(mod, "train")]["mmd_train_vs_heldout_speakers_ref"] = mmd_rbf(trZ[tr], trZ[va], n=800)
sdf = pd.DataFrame(stats).T
sdf.index.names = ["modality", "set"]
print("\nEDA-6 train vs dev (PCA-64 fit on train)\n", sdf.round(3).to_string())
R["EDA6_shift_table"] = {f"{a}|{b}": v for (a, b), v in sdf.to_dict("index").items()}

# ---------------------------------------------------------------- EDA-5 English vs Bangla (dev, pooled over protocols)
e5 = {}
for mod, i in [("face", 0), ("voice", 1)]:
    mu, sd, p = pcas[mod]
    En = np.vstack([dev[("no_gender", "English")][i], dev[("gender", "English")][i]])
    Bn = np.vstack([dev[("no_gender", "Bangla")][i], dev[("gender", "Bangla")][i]])
    Ze, Zb = p.transform((En - mu) / sd), p.transform((Bn - mu) / sd)
    e5[f"{mod}_mmd_En_vs_Bn"] = mmd_rbf(Ze, Zb, n=800)
    e5[f"{mod}_centroid_dist_En_Bn_in_train_sd"] = float(np.linalg.norm((Ze.mean(0) - Zb.mean(0)) / np.sqrt(p.explained_variance_)))
    # language probe: can a linear classifier tell En from Bn? (5-fold on dev, no speaker labels -> optimistic)
    Z = np.vstack([Ze, Zb]); y = np.array([0] * len(Ze) + [1] * len(Zb))
    rng = np.random.RandomState(0); perm = rng.permutation(len(Z)); Z, y = Z[perm], y[perm]
    accs = []
    for f in range(5):
        te = np.arange(len(Z)) % 5 == f
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(Z[~te], y[~te])
        accs.append(float((clf.predict(Z[te]) == y[te]).mean()))
    e5[f"{mod}_language_probe_acc(5fold, optimistic)"] = float(np.mean(accs))
    # cross-language kNN: how far is each Bangla sample from nearest English dev sample vs nearest Bangla?
    e5[f"{mod}_knn5_Bn_to_En"] = float(knn_dist_to(En, Bn).mean())
    e5[f"{mod}_knn5_En_to_En(loo-ish)"] = float(np.sort(1 - (l2n(En) @ l2n(En).T), axis=1)[:, 1:6].mean())
    e5[f"{mod}_knn5_Bn_to_Bn(loo-ish)"] = float(np.sort(1 - (l2n(Bn) @ l2n(Bn).T), axis=1)[:, 1:6].mean())
# do English and Bangla dev share identities? nearest-neighbour face similarity En<->Bn vs within-train same-speaker level
Fzf = l2n(Xf - Xf.mean(0))
tr_same = []
rng = np.random.RandomState(0)
for _ in range(3000):
    s = rng.choice(np.unique(yf)); ids = np.where(yf == s)[0]
    if len(ids) < 2: continue
    a, b = rng.choice(ids, 2, replace=False); tr_same.append(float(Fzf[a] @ Fzf[b]))
En = np.vstack([dev[("no_gender", "English")][0], dev[("gender", "English")][0]])
Bn = np.vstack([dev[("no_gender", "Bangla")][0], dev[("gender", "Bangla")][0]])
En_z, Bn_z = l2n(En - Xf.mean(0)), l2n(Bn - Xf.mean(0))
nn_BnEn = (Bn_z @ En_z.T).max(1)
e5["face_train_same_speaker_cos_mean"] = float(np.mean(tr_same))
e5["face_train_diff_speaker_cos_mean"] = float(np.sum(Fzf[rng.randint(len(Fzf), size=3000)] * Fzf[rng.randint(len(Fzf), size=3000)], 1).mean())
e5["face_Bn_nearest_En_cos_mean"] = float(nn_BnEn.mean())
e5["face_frac_Bn_with_En_nn_cos>0.6"] = float((nn_BnEn > 0.6).mean())
R["EDA5"] = e5
print("\nEDA-5", json.dumps(e5, indent=1))

# ---------------------------------------------------------------- gender protocol sanity via train-fitted gender probe
e5g = {}
probes = {}
for mod, i in [("face", 0), ("voice", 1)]:
    mu, sd, p = pcas[mod]
    Ztr = p.transform((sets["train"][i] - mu) / sd)
    probes[mod] = LogisticRegression(max_iter=2000, C=0.1).fit(Ztr, gender)
for (prot, lang), (a, b, _) in dev.items():
    gf = probes["face"].predict(pcas["face"][2].transform((a - pcas["face"][0]) / pcas["face"][1]))
    gv = probes["voice"].predict(pcas["voice"][2].transform((b - pcas["voice"][0]) / pcas["voice"][1]))
    e5g[f"{prot}/{lang}"] = dict(face_voice_gender_agree=float((gf == gv).mean()),
                                 face_pred_male_frac=float((gf == "m").mean()), voice_pred_male_frac=float((gv == "m").mean()))
R["GENDER_PROTOCOL_SANITY"] = e5g
print("\nGender protocol sanity (train-fitted probe on dev pairs)", json.dumps(e5g, indent=1))

# ---------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mod, i in zip(axes, ["face", "voice"], [0, 1]):
    mu, sd, p = pcas[mod]
    rng = np.random.RandomState(0)
    for name, X, col, mk in [("train", sets["train"][i], PALETTE["blue"], "o"),
                             ("dev English", np.vstack([dev[("no_gender", "English")][i], dev[("gender", "English")][i]]), PALETTE["orange"], "^"),
                             ("dev Bangla", np.vstack([dev[("no_gender", "Bangla")][i], dev[("gender", "Bangla")][i]]), PALETTE["aqua"], "s")]:
        Z = p.transform((X - mu) / sd)[:, :2]
        sel = rng.choice(len(Z), min(800, len(Z)), replace=False)
        ax.scatter(Z[sel, 0], Z[sel, 1], s=6, alpha=.45, color=col, marker=mk, label=name, linewidths=0)
    ax.set_title(f"EDA-5/6  PCA(train) PC1–PC2 — {mod}", fontsize=10); ax.legend(fontsize=8, frameon=False, markerscale=2)
    ax.grid(color=PALETTE["grid"]); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "eda56_pca_scatter.png", dpi=130); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, mod, i in zip(axes, ["face", "voice"], [0, 1]):
    bins = np.linspace(0, 1, 50)
    Xtr = sets["train"][i]
    tr, va = speaker_split(yf, 1)
    ax.hist(knn_dist_to(Xtr[tr], Xtr[va]), bins, alpha=.7, color=PALETTE["blue"], label="train held-out speakers", density=True)
    ax.hist(knn_dist_to(Xtr, np.vstack([dev[("no_gender", "English")][i], dev[("gender", "English")][i]])), bins, alpha=.6, color=PALETTE["orange"], label="dev English", density=True)
    ax.hist(knn_dist_to(Xtr, np.vstack([dev[("no_gender", "Bangla")][i], dev[("gender", "Bangla")][i]])), bins, alpha=.6, color=PALETTE["aqua"], label="dev Bangla", density=True)
    ax.set_title(f"EDA-6  5-NN cosine distance to train — {mod}", fontsize=10); ax.legend(fontsize=8, frameon=False)
    ax.grid(color=PALETTE["grid"]); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "eda6_knn_dist.png", dpi=130); plt.close()

json.dump(R, open(OUT / "eda_56.json", "w"), indent=1, default=float)
print("saved")
