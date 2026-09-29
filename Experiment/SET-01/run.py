"""SET-01 — test-time SET AGGREGATION with ESTIMATED identities (no labels, no trial list), validated with TRUE labels.

Per evaluation file: cluster its faces (ArcFace-512, agglomerative, cosine threshold calibrated on OTHER speakers) and its
voices (own ECAPA-192, idem), then replace every model embedding by  (1-b) * own + b * mean of its cluster.
Score = cosine of the aggregated embeddings.  Model = EXP-007 recipe trained on the TRAIN speakers of the split.
Validation = v4 held-out speakers of 3 splits (true labels, gender + no_gender trials): the val set plays the role of a
dev file (~14 people, ~1300 rows), so nothing here touches dev labels or pseudo-labels."""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(10)
import flag_v2 as V
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)
AF = np.load(R / "kaggle/output/feats_v2/face_arcface_train.npy"); AV = np.load(R / "kaggle/output/feats_v2/voice_ecapa192_train.npy")
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))


def cluster(Z, thr):
    return AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=thr).fit_predict(Z)


def calib(Z, lab):
    best = max(((t, adjusted_rand_score(lab, cluster(Z, t))) for t in np.arange(0.4, 1.21, 0.05)), key=lambda x: x[1])
    return best


def agg(E, c, b):
    M = np.zeros((c.max() + 1, E.shape[1])); np.add.at(M, c, E); M = V.l2n(M)
    return V.l2n((1 - b) * E + b * M[c])


rows = []
for seed in (1, 2, 3):
    tr, va = V.speaker_split(store.spk, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
    # thresholds calibrated on TRAIN speakers only
    tF, aF = calib(cent(AF[tri][::3]), store.spk[tri][::3]); tV, aV = calib(cent(AV[tri][::3]), store.spk[tri][::3])
    cF, cV = cluster(cent(AF[vai]), tF), cluster(cent(AV[vai]), tV)
    sp = store.spk[vai]
    print(f"split {seed}: face thr {tF:.2f} (train ARI {aF:.3f}) -> val ARI {adjusted_rand_score(sp, cF):.3f}, {cF.max()+1} clusters | "
          f"voice thr {tV:.2f} (train ARI {aV:.3f}) -> val ARI {adjusted_rand_score(sp, cV):.3f}, {cV.max()+1} clusters | true people {len(set(sp))}", flush=True)
    embs = []
    for i in range(3):
        net, prep = V.train_one(store, REC, tri, seed=seed + 100 * i)
        embs.append(V.embed(net, prep, store.Xf[vai], store.Xv[vai], prep.f(store.Xf[tri]).mean(0), prep.v(store.Xv[tri]).mean(0)))
    for sg in (False, True):
        fi, vj, lab = V.build_trials(sp, store.gmap, seed=seed, same_gender=sg)
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            s = np.mean([V.z(np.sum(agg(u, cF, b)[fi] * agg(w, cV, b)[vj], 1)) for u, w in embs], 0)
            # oracle with TRUE identities, for reference
            to = pd.factorize(sp)[0]
            so = np.mean([V.z(np.sum(agg(u, to, b)[fi] * agg(w, to, b)[vj], 1)) for u, w in embs], 0)
            rows.append(dict(seed=seed, protocol="gender" if sg else "no_gender", b=b,
                             eer=round(V.eer_from_scores(s, lab), 2), eer_true_ids=round(V.eer_from_scores(so, lab), 2)))
T = pd.DataFrame(rows); T.to_csv("set01.csv", index=False)
print(T.groupby(["protocol", "b"])[["eer", "eer_true_ids"]].mean().round(2).unstack(0).to_string())
