"""SET-06 — one-to-one constraint between face clusters and voice clusters (Sinkhorn), TRUE labels.
In one file each person has one face identity and one voice identity. After block aggregation we have a cluster x
cluster score matrix S (faces x voices). Sinkhorn turns exp(S/tau) into an (approximately) doubly-stochastic matrix, so a
voice cluster that scores high with MANY face clusters (a 'hub', ERR-01) is down-weighted and each face cluster has to
compete for its voice. Uses only the clusters + bridge scores (no trial list, no labels).
v4, 30 held-out speakers, 3 splits, bridge = EXP-007 recipe (3 models), face ArcFace / voice ECAPA-192 clusters."""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(10)
import flag_v2 as V
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
calib = lambda Z, lab: max(np.arange(.4, 1.11, .05), key=lambda t: adjusted_rand_score(lab, clus(Z, t)))
AF, AV = store.face("arcface", "train"), store.voice("ecapa192", "train")


def cluster_matrix(u, w, cf, cv):
    Mf = np.zeros((cf.max() + 1, u.shape[1])); np.add.at(Mf, cf, u)
    Mv = np.zeros((cv.max() + 1, w.shape[1])); np.add.at(Mv, cv, w)
    return V.l2n(Mf) @ V.l2n(Mv).T                          # cos of cluster means


def sinkhorn(S, tau, iters=50):
    """log-domain Sinkhorn; clusters weighted by size would be an option, uniform here."""
    L = S / tau
    for _ in range(iters):
        L = L - np.logaddexp.reduce(L, axis=1, keepdims=True)     # rows sum to 1
        L = L - np.logaddexp.reduce(L, axis=0, keepdims=True)     # cols sum to 1
    return L


rows = []
for seed in (1, 2, 3):
    tr, va = V.speaker_split(store.spk, seed, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]
    sp = store.spk[vai]; sub = tri[::3]
    cf = clus(cent(AF[vai]), calib(cent(AF[sub]), store.spk[sub]))
    cv = clus(cent(AV[vai]), calib(cent(AV[sub]), store.spk[sub]))
    embs = []
    for i in range(3):
        net, prep = V.train_one(store, REC, tri, seed=seed + 100 * i)
        embs.append(V.embed(net, prep, store.Xf[vai], store.Xv[vai], prep.f(store.Xf[tri]).mean(0), prep.v(store.Xv[tri]).mean(0)))
    S = np.mean([V.z(cluster_matrix(u, w, cf, cv)) for u, w in embs], 0)   # z-scored per model, then averaged
    T = {sg: V.build_trials(sp, store.gmap, seed=seed, same_gender=sg) for sg in (False, True)}
    variants = {"block mean (SET-02)": S}
    for tau in (0.05, 0.1, 0.2, 0.5, 1.0):
        variants[f"sinkhorn tau={tau}"] = sinkhorn(S, tau)
    for alpha in (0.25, 0.5):
        P = sinkhorn(S, 0.2)
        variants[f"mix a={alpha} (z(S) + a z(logP))"] = V.z(S) + alpha * V.z(P)
    for name, M in variants.items():
        r = dict(seed=seed, variant=name, n_face_cl=cf.max() + 1, n_voice_cl=cv.max() + 1)
        for sg, (fi, vj, lab) in T.items():
            r["gender" if sg else "no_gender"] = V.eer_from_scores(M[cf[fi], cv[vj]] + 1e-6 * np.random.RandomState(0).randn(len(fi)), lab)
        rows.append(r)
    print("split", seed, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv("set06.csv", index=False)
print(D.groupby("variant", sort=False)[["no_gender", "gender"]].mean().round(2).to_string())
