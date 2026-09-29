"""SET-05 part B — voice clustering on NON-English speech with TRUE labels (MAV-Celeb v1 Urdu, v2 Hindi).
Files of 30 random speakers (dev-like), 5 draws per language.
B1  ARI of voice clusters per embedding, with the threshold (i) calibrated on English (v4 train),
    (ii) calibrated on the OTHER non-English language, (iii) oracle best for this language (ceiling).
B2  file-level EER of the cluster-level pipeline: bridge = EXP-007 recipe trained on all 70 v4 speakers (VGG face +
    organiser 192 voice, both available for the external rows); faces aggregated by TRUE identity so that the voice
    clustering is the only variable; voices aggregated by the candidate clusters. no_gender trials (v1 also gender)."""
import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(10)
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/NB4/feats_v3")
import flag_v2 as V
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
FE, FB = R / "kaggle/output/feats_ext", Path(__file__).parent / "feats_b"
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
GRID = np.round(np.arange(.4, 1.11, .025), 3)
best_t = lambda Z, lab: max(GRID, key=lambda t: adjusted_rand_score(lab, clus(Z, t)))

# English thresholds, calibrated on v4 train exactly like the dev pipeline
sub = np.arange(len(store.spk))[::3]
EN_T = {"given": best_t(cent(store.voice("given", "train")[sub]), store.spk[sub]),
        "ecapa192": best_t(cent(store.voice("ecapa192", "train")[sub]), store.spk[sub]),
        "rdn6vox": best_t(cent(store.voice("rdn6vox", "train")[sub]), store.spk[sub])}
print("English-calibrated thresholds:", EN_T, flush=True)

LANGS = {"urdu": "v1_complete", "hindi": "v2_complete"}
data = {}
for lang, src in LANGS.items():
    m = pd.read_csv(FB / f"{src}_meta.csv")
    data[lang] = dict(m=m, given=np.load(FE / f"ext_{src}_voice.npy")[m.row.values],
                      ecapa192=np.load(FB / f"{src}_ecapa192.npy"), rdn6vox=np.load(FB / f"{src}_rdn6vox.npy"),
                      Fa=np.load(FE / f"ext_{src}_face.npy").astype(np.float32), fm=pd.read_csv(FE / f"ext_{src}_face_meta.csv"))
    print(lang, len(m), "clips,", m.spk.nunique(), "speakers", flush=True)

# ---- B1 thresholds calibrated per non-English language (on its full set) -> used for the OTHER language
NE_T = {lang: {e: best_t(cent(d[e]), d["m"].spk.values) for e in ("given", "ecapa192", "rdn6vox")} for lang, d in data.items()}
print("non-English-calibrated thresholds:", NE_T, flush=True)

# ---- bridge trained on all v4 speakers
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)
nets = [V.train_one(store, REC, np.arange(len(store.spk)), seed=s) for s in (1, 101, 201)]
meta1 = pd.read_csv(FE / "ext_v1_complete_speakers.csv")
g_of = {r.ids: ("m" if str(r.gender).lower().startswith("m") else "f") for r in meta1.itertuples()}


def agg(E, c):
    M = np.zeros((c.max() + 1, E.shape[1])); np.add.at(M, c, E); M = V.l2n(M)
    return V.l2n(0.01 * E + 0.99 * M[c])


rows = []
for lang, d in data.items():
    other = [l for l in LANGS if l != lang][0]
    spk_all = np.array(sorted(d["m"].spk.unique()))
    for draw in range(5):
        rng = np.random.RandomState(draw)
        people = rng.choice(spk_all, min(30, len(spk_all)), replace=False)
        vi = np.where(d["m"].spk.isin(people))[0]
        m = d["m"].iloc[vi].reset_index(drop=True); sp = m.spk.values
        # one face per clip: a frame of the same speaker, preferably from another video (no same-session shortcut)
        fi_rows = []
        for r in m.itertuples():
            cand = d["fm"][(d["fm"].spk == r.spk) & (d["fm"].video != r.video)].index
            cand = cand if len(cand) else d["fm"][d["fm"].spk == r.spk].index
            fi_rows.append(rng.choice(cand))
        Fv = d["Fa"][fi_rows]; Vo = d["given"][vi]
        embs = [V.embed(net, prep, Fv, Vo, prep.f(store.Xf).mean(0), prep.v(store.Xv).mean(0)) for net, prep in nets]
        ids = pd.factorize(sp)[0]
        gmap = {s: g_of.get(s, "m") for s in sp} if lang == "urdu" else {s: "m" for s in sp}
        protos = [False, True] if lang == "urdu" else [False]
        T = {sg: V.build_trials(sp, gmap, seed=draw, same_gender=sg) for sg in protos}
        variants = {"none": np.arange(len(vi)), "oracle": ids}
        for e in ("given", "ecapa192", "rdn6vox"):
            Z = cent(d[e][vi])
            for tname, t in [("EN", EN_T[e]), ("other-lang", NE_T[other][e])]:
                variants[f"{e}|{tname}"] = clus(Z, t)
            rows.append(dict(lang=lang, draw=draw, emb=e, kind="ARI", EN=adjusted_rand_score(sp, variants[f"{e}|EN"]),
                             other_lang=adjusted_rand_score(sp, variants[f"{e}|other-lang"]),
                             oracle_best=max(adjusted_rand_score(sp, clus(Z, t)) for t in GRID[::2])))
        for vname, cV in variants.items():
            r = dict(lang=lang, draw=draw, emb=vname, kind="EER", n_clusters=cV.max() + 1)
            for sg, (fi, vj, lab) in T.items():
                s = np.mean([V.z(np.sum(agg(u, ids)[fi] * agg(w, cV)[vj], 1)) for u, w in embs], 0)
                r["gender" if sg else "no_gender"] = V.eer_from_scores(s, lab)
            rows.append(r)
    print(lang, "done", flush=True)
T = pd.DataFrame(rows); T.to_csv("part_b.csv", index=False)
A = T[T.kind == "ARI"].groupby(["lang", "emb"])[["EN", "other_lang", "oracle_best"]].mean().round(3)
print("\nB1 — ARI of voice clusters on non-English speech (30-speaker files, 5 draws)\n" + A.to_string())
E = T[T.kind == "EER"].groupby(["lang", "emb"])[["n_clusters", "no_gender", "gender"]].mean().round(2)
print("\nB2 — file-level EER (faces aggregated by true identity; only the voice clustering varies)\n" + E.to_string())
