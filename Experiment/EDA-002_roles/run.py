"""EDA-002 — pick an embedding per ROLE of the cluster-level system (true labels only).
  Role 1/2  clustering : ARI of agglomerative clustering on 30 held-out v4 speakers (threshold calibrated on the other 40),
                         for every face / voice embedding we have.
  Role 3    bridge     : cross-modal EER after aggregating by TRUE identity (person level) and per sample (sample level),
                         recipe EXP-007 trained on the 40 train speakers, for each (face, voice) feature pair.
  Dev view  cluster counts per dev file for each clustering embedding (~30 people per file expected; Bangla vs English).
3 splits (seeds 1,2,3). No dev labels, no pseudo-labels."""
import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(6)
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/NB4/feats_v3")
import flag_v2 as V
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
FACES = ["vgg", "arcface"]
VOICES = ["given", "ecapa192", "ecapa6144", "rdn6vox", "rdn6multi", "wavlm_L4-9", "xlsr_L4-6"]
getF = lambda n, s: store.face(n, s); getV = lambda n, s: store.voice(n, s)

# ---- role 1/2: clustering quality
rows = []
for seed in (1, 2, 3):
    tr, va = V.speaker_split(store.spk, seed, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]
    for kind, names, get in [("face", FACES, getF), ("voice", VOICES, getV)]:
        for n in names:
            X = get(n, "train"); sub = tri[::3]
            t = max(np.arange(.3, 1.31, .05), key=lambda t: adjusted_rand_score(store.spk[sub], clus(cent(X[sub]), t)))
            c = clus(cent(X[vai]), t)
            rows.append(dict(seed=seed, kind=kind, emb=n, thr=round(t, 2), ari=adjusted_rand_score(store.spk[vai], c), n_clusters=c.max() + 1))
C = pd.DataFrame(rows)
print("ROLE 1/2 — clustering ARI on 30 unseen speakers (true people = 30)")
print(C.groupby(["kind", "emb"])[["ari", "n_clusters", "thr"]].mean().round(3).sort_values(["kind", "ari"], ascending=[True, False]).to_string(), flush=True)

# ---- dev view: number of clusters per dev file with the train-calibrated threshold
thr = C.groupby("emb").thr.median().to_dict()
dv = []
for k in V.CELLS:
    split = "/".join(k)
    for kind, names, get in [("face", FACES, getF), ("voice", VOICES, getV)]:
        for n in names:
            dv.append(dict(file=split, kind=kind, emb=n, n_clusters=clus(cent(get(n, split)), thr[n]).max() + 1))
print("\nDEV — clusters per file (about 30 people expected; fewer = merging people)")
print(pd.DataFrame(dv).pivot_table(index=["kind", "emb"], columns="file", values="n_clusters").to_string(), flush=True)

# ---- role 3: bridge quality, sample level vs person level (true identities)
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)


def agg(E, c):
    M = np.zeros((c.max() + 1, E.shape[1])); np.add.at(M, c, E); return V.l2n(M)[c]


br = []
PAIRS = [("vgg", v) for v in VOICES] + [("arcface", "given"), ("arcface", "ecapa192"), ("arcface", "rdn6multi")]
for seed in (1, 2, 3):
    tr, va = V.speaker_split(store.spk, seed, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]
    sp = store.spk[vai]; ids = pd.factorize(sp)[0]
    T = {sg: V.build_trials(sp, store.gmap, seed=seed, same_gender=sg) for sg in (False, True)}
    for f, v in PAIRS:
        cfg = dict(REC, face=f, voice=v)
        Xf, Xv = getF(f, "train"), getV(v, "train")
        net, prep = V.train_one(store, cfg, tri, seed=seed)
        u, w = V.embed(net, prep, Xf[vai], Xv[vai], prep.f(Xf[tri]).mean(0), prep.v(Xv[tri]).mean(0))
        U, W = agg(u, ids), agg(w, ids)
        r = dict(seed=seed, face=f, voice=v)
        for sg, (fi, vj, lab) in T.items():
            p = "g" if sg else "ng"
            r[f"sample_{p}"] = V.eer_from_scores(np.sum(u[fi] * w[vj], 1), lab)
            r[f"person_{p}"] = V.eer_from_scores(np.sum(U[fi] * W[vj], 1), lab)
        br.append(r); print(" ", {k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
B = pd.DataFrame(br)
print("\nROLE 3 — bridge EER, sample level vs person level (true identities), mean of 3 splits")
print(B.groupby(["face", "voice"])[["sample_ng", "sample_g", "person_ng", "person_g"]].mean().round(2).sort_values("person_g").to_string())
C.to_csv("clustering.csv", index=False); pd.DataFrame(dv).to_csv("dev_clusters.csv", index=False); B.to_csv("bridge.csv", index=False)
