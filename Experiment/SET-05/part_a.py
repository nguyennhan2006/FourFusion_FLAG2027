"""SET-05 part A — which embedding should CLUSTER THE VOICES?  Full cluster-level pipeline, TRUE labels.
v4, 30 held-out speakers (dev-like density), 3 splits. Bridge fixed: EXP-007 recipe (VGG + organiser 192), 3 models.
Face clusters fixed: ArcFace, threshold calibrated on the 40 train speakers. Only the voice-clustering embedding varies;
each one gets its own threshold calibrated (max ARI) on the train speakers. Aggregation = embedding means (b = 0.99)."""
import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(8)
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/NB4/feats_v3")
import flag_v2 as V
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
calib = lambda Z, lab: max(np.arange(.4, 1.11, .05), key=lambda t: adjusted_rand_score(lab, clus(Z, t)))
VOICE_CL = ["ecapa192", "rdn6vox", "rdn6multi"]
AF = store.face("arcface", "train")


def agg(E, c, b=0.99):
    M = np.zeros((c.max() + 1, E.shape[1])); np.add.at(M, c, E); M = V.l2n(M)
    return V.l2n((1 - b) * E + b * M[c])


rows = []
for seed in (1, 2, 3):
    tr, va = V.speaker_split(store.spk, seed, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]
    sp = store.spk[vai]; sub = tri[::3]
    embs = []
    for i in range(3):
        net, prep = V.train_one(store, REC, tri, seed=seed + 100 * i)
        embs.append(V.embed(net, prep, store.Xf[vai], store.Xv[vai], prep.f(store.Xf[tri]).mean(0), prep.v(store.Xv[tri]).mean(0)))
    cF = clus(cent(AF[vai]), calib(cent(AF[sub]), store.spk[sub]))
    T = {sg: V.build_trials(sp, store.gmap, seed=seed, same_gender=sg) for sg in (False, True)}
    for vc in ["none"] + VOICE_CL:
        if vc == "none":
            cV, thr, ari = np.arange(len(vai)), None, np.nan
        else:
            Xc = store.voice(vc, "train"); thr = calib(cent(Xc[sub]), store.spk[sub])
            cV = clus(cent(Xc[vai]), thr); ari = adjusted_rand_score(sp, cV)
        r = dict(seed=seed, voice_cluster=vc, thr=thr, voice_ari=ari, n_voice_clusters=cV.max() + 1)
        for sg, (fi, vj, lab) in T.items():
            s = np.mean([V.z(np.sum(agg(u, cF)[fi] * agg(w, cV)[vj], 1)) for u, w in embs], 0)
            r["gender" if sg else "no_gender"] = V.eer_from_scores(s, lab)
        rows.append(r); print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
T = pd.DataFrame(rows); T.to_csv("part_a.csv", index=False)
print(T.groupby("voice_cluster")[["voice_ari", "n_voice_clusters", "no_gender", "gender"]].mean().round(3).to_string())
