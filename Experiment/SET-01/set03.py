"""SET-03 — why did g/En not improve on dev?  Dev-like density with TRUE labels: hold out 30 v4 speakers (train on 40),
sweep the voice/face cluster thresholds and the aggregation weight, report gender / no_gender EER + cluster purity."""
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
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)


def agg(E, c, b):
    M = np.zeros((c.max() + 1, E.shape[1])); np.add.at(M, c, E); M = V.l2n(M)
    return V.l2n((1 - b) * E + b * M[c])


def purity_mixgender(c, sp):
    """share of samples in clusters that mix >1 true person / mix both genders"""
    d = pd.DataFrame(dict(c=c, s=sp, g=[store.gmap[x] for x in sp]))
    n_p = d.groupby("c").s.transform("nunique"); n_g = d.groupby("c").g.transform("nunique")
    return round(float((n_p > 1).mean()), 3), round(float((n_g > 1).mean()), 3)


rows = []
for seed in (1, 2, 3):
    tr, va = V.speaker_split(store.spk, seed, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]
    sp = store.spk[vai]
    embs = []
    for i in range(3):
        net, prep = V.train_one(store, REC, tri, seed=seed + 100 * i)
        embs.append(V.embed(net, prep, store.Xf[vai], store.Xv[vai], prep.f(store.Xf[tri]).mean(0), prep.v(store.Xv[tri]).mean(0)))
    trials = {sg: V.build_trials(sp, store.gmap, seed=seed, same_gender=sg) for sg in (False, True)}
    for tf in (0.65, 0.75):
        cF = clus(cent(AF[vai]), tf)
        for tv in (0.5, 0.6, 0.7, 0.8):
            cV = clus(cent(AV[vai]), tv)
            pv, gv = purity_mixgender(cV, sp); pf, gf = purity_mixgender(cF, sp)
            for b in (0.0, 0.5, 0.75, 0.99):
                r = dict(seed=seed, tf=tf, tv=tv, b=b, face_ari=round(adjusted_rand_score(sp, cF), 3), voice_ari=round(adjusted_rand_score(sp, cV), 3),
                         voice_mixed_person=pv, voice_mixed_gender=gv, face_mixed_person=pf)
                for sg, (fi, vj, lab) in trials.items():
                    s = np.mean([V.z(np.sum(agg(u, cF, b)[fi] * agg(w, cV, b)[vj], 1)) for u, w in embs], 0)
                    r["gender" if sg else "no_gender"] = round(V.eer_from_scores(s, lab), 2)
                rows.append(r)
    print("split", seed, "done", flush=True)
T = pd.DataFrame(rows); T.to_csv("set03.csv", index=False)
G = T.groupby(["tf", "tv", "b"])[["gender", "no_gender", "voice_ari", "voice_mixed_person", "face_ari", "face_mixed_person"]].mean().round(3)
print(G.to_string())
