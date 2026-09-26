"""EDA-001 Q4-Q7 on dev, which has NO identity labels.

Recover pseudo-identities without labels:
  1. cluster dev faces (all 4 files pooled) and dev voices (per language) with an agglomerative
     threshold CALIBRATED ON TRAIN, where labels exist;
  2. link each voice cluster to the face cluster it is paired with far more often than chance
     (positives concentrate on one face cluster, negatives spread out);
  3. voice clusters of both languages linked to the same face cluster = same speaker, En and Bn.
Every number below is an estimate conditioned on that linkage; purity is reported.
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EDA-000_raw"))
from eda_utils import load_train, load_dev, l2n, speaker_split

Xf, Xv, spk, _, txt, gmap = load_train()
mf, sf = Xf.mean(0), Xf.std(0) + 1e-6
_, _, Vt = np.linalg.svd((Xf - mf) / sf, full_matrices=False); P = Vt[:256].T
mv, sv = Xv.mean(0), Xv.std(0) + 1e-6
F = lambda X: l2n(((X - mf) / sf) @ P)
def V(X, mu=None):  # per-set mean centering (EXP-002 showed this is the safe first-moment fix)
    Z = (X - mv) / sv
    return l2n(Z - Z.mean(0))

def calib(E, labels, name):
    best = None
    for t in np.arange(0.3, 1.21, 0.05):
        c = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(E)
        ari = adjusted_rand_score(labels, c)
        if best is None or ari > best[1]: best = (t, ari, len(set(c)))
    print(f"{name}: threshold {best[0]:.2f} ARI {best[1]:.3f} clusters {best[2]} (true 70)")
    return best[0]

# calibrate on 14 held-out-like speakers x3 seeds is overkill; use one 30-speaker subset (dev size is similar)
rng = np.random.RandomState(0); sub = set(rng.choice(np.unique(spk), 30, replace=False))
m = np.array([s in sub for s in spk])
tF = calib(F(Xf[m]), spk[m], "train face")
tV = calib(V(Xv[m]), spk[m], "train voice")

files = {(p, l): load_dev(p, l) for p in ["no_gender", "gender"] for l in ["English", "Bangla"]}
# pool faces across all files, dedup exact duplicates
allF = np.concatenate([files[k][0] for k in files]); keyF = np.concatenate([[k] * len(files[k][0]) for k in files], dtype=object) if False else None
offs, o = {}, 0
for k in files: offs[k] = o; o += len(files[k][0])
uF, invF = np.unique(allF.round(4), axis=0, return_inverse=True)
cF = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=tF).fit_predict(F(uF))[invF.ravel()]
print("dev face clusters (pooled):", len(set(cF)), "sizes top10", np.sort(np.bincount(cF))[::-1][:10])

# face-cluster gender from a probe trained on train faces
gy = np.array([gmap[s] == "m" for s in spk])
probe = LogisticRegression(max_iter=2000, C=0.1).fit(F(Xf), gy)
pm = probe.predict_proba(F(allF))[:, 1]
clusterG = {c: ("m" if pm[cF == c].mean() > .5 else "f") for c in set(cF)}

out = {}
link_rows, voice_rows = [], []
for lang in ["English", "Bangla"]:
    ks = [("no_gender", lang), ("gender", lang)]
    allV = np.concatenate([files[k][1] for k in ks])
    uV, invV = np.unique(allV.round(4), axis=0, return_inverse=True)
    Ev_u = V(uV)
    cV = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=tV).fit_predict(Ev_u)[invV.ravel()]
    EV = Ev_u[invV.ravel()]
    # pair face cluster for every voice row
    fc = np.concatenate([cF[offs[k]:offs[k] + len(files[k][0])] for k in ks])
    tab = pd.crosstab(cV, fc)
    for vc, row in tab.iterrows():
        n = row.sum(); top = row.idxmax(); cnt = row.max()
        exp = n * (tab[top].sum() / tab.values.sum())
        link_rows.append(dict(lang=lang, vc=vc, n=n, face=top, share=cnt / n, enrich=cnt / max(exp, 1e-9)))
    for i in range(len(EV)):
        voice_rows.append(dict(lang=lang, vc=cV[i], emb=EV[i], fc=fc[i]))
    print(f"{lang}: voice clusters {len(set(cV))}")

L = pd.DataFrame(link_rows)
good = L[(L.n >= 4) & (L.share >= 0.4) & (L.enrich >= 3)]
print(f"\nlinked voice clusters: {len(good)}/{len(L)}  (rows covered {good.n.sum()}/{L.n.sum()})")
vmap = {(r.lang, r.vc): r.face for r in good.itertuples()}

# Q4/Q5: estimated positives and negative gender mix per file
q45 = []
for k in files:
    lang = k[1]; ks = [("no_gender", lang), ("gender", lang)]
    Xv_k = files[k][1]; fcs = cF[offs[k]:offs[k] + len(files[k][0])]
    # recover this file's voice clusters by position inside the language pool
    start = 0 if k[0] == "no_gender" else len(files[("no_gender", lang)][1])
    vcs = [r["vc"] for r in voice_rows if r["lang"] == lang][start:start + len(Xv_k)]
    lab = [(vmap.get((lang, v)), f) for v, f in zip(vcs, fcs)]
    known = [(a, b) for a, b in lab if a is not None]
    pos = sum(a == b for a, b in known); neg = [(a, b) for a, b in known if a != b]
    sg = sum(clusterG[a] == clusterG[b] for a, b in neg)
    mm = sum(clusterG[a] == clusterG[b] == "m" for a, b in neg); ff = sum(clusterG[a] == clusterG[b] == "f" for a, b in neg)
    q45.append(dict(file=f"{k[0]}/{lang}", rows=len(Xv_k), linked=len(known), est_pos=pos, est_neg=len(neg),
                    pos_rate=round(pos / max(len(known), 1), 3), neg_same_gender=sg, neg_cross_gender=len(neg) - sg, mm=mm, ff=ff))
Q45 = pd.DataFrame(q45); print("\nQ4/Q5 (estimated via linkage)\n", Q45.to_string(index=False))

# Q7: voice cosine within / across language for the same pseudo-speaker
R = pd.DataFrame([dict(lang=r["lang"], spk=vmap.get((r["lang"], r["vc"])), emb=r["emb"]) for r in voice_rows])
R = R[R.spk.notna()].reset_index(drop=True)
E = np.stack(R.emb.values); S = E @ E.T
same = R.spk.values[:, None] == R.spk.values[None, :]
en = (R.lang == "English").values
iu = np.triu_indices(len(R), 1)
g = np.array([clusterG[s] for s in R.spk.values]); sameg = g[:, None] == g[None, :]
def stat(mask):
    v = S[iu][mask[iu]]; return dict(n=int(len(v)), mean=round(float(v.mean()), 3), sd=round(float(v.std()), 3)) if len(v) else None
EnEn, BnBn, EnBn = np.outer(en, en), np.outer(~en, ~en), np.outer(en, ~en) | np.outer(~en, en)
q7 = {"same spk En-En": stat(same & EnEn), "same spk Bn-Bn": stat(same & BnBn), "same spk En-Bn": stat(same & EnBn),
      "diff spk same-gender En-En": stat(~same & sameg & EnEn), "diff spk same-gender Bn-Bn": stat(~same & sameg & BnBn),
      "diff spk same-gender En-Bn": stat(~same & sameg & EnBn), "diff spk cross-gender (any)": stat(~same & ~sameg)}
nspk_both = len(set(R.spk[en]) & set(R.spk[~en]))
print(f"\nQ7 voice cosine (per-language centred, train-standardised 192-d); speakers seen in both languages: {nspk_both}")
for k, v in q7.items(): print(f"  {k:30s} {v}")

# Q6 on TRAIN (labelled): cross-modal-free unimodal cosine distributions, reference for Q7
m2 = np.ones(len(spk), bool)
for name, E2 in [("face", F(Xf)), ("voice", V(Xv))]:
    r = np.random.RandomState(1); i, j = r.randint(len(spk), size=(2, 60000)); k2 = i != j
    i, j = i[k2], j[k2]; c = np.sum(E2[i] * E2[j], 1)
    s_ = spk[i] == spk[j]; gg = np.array([gmap[a] == gmap[b] for a, b in zip(spk[i], spk[j])])
    print(f"Q6 train {name}: same spk {c[s_].mean():.3f}±{c[s_].std():.3f} | diff spk same-g {c[~s_ & gg].mean():.3f}±{c[~s_ & gg].std():.3f} | diff spk cross-g {c[~s_ & ~gg].mean():.3f}±{c[~s_ & ~gg].std():.3f}")

json.dump(dict(thr=dict(face=float(tF), voice=float(tV)), q45=q45, q7=q7, speakers_both_lang=nspk_both,
               linked=int(len(good)), face_clusters=int(len(set(cF)))), open("pseudo.json", "w"), indent=1, default=str)
