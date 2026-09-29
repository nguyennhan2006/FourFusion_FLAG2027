"""LANG-01 — project a learned LANGUAGE direction out of the voice features (PLAN_V6 I3), TRUE labels.

Direction(s): in the organiser voice space (standardised with v4-train stats), for every external TRAIN person with
both languages: diff = mean(native-language clips) - mean(English clips); the language subspace = top-k principal
directions of those per-person differences (k=1 is simply their normalised mean). Speaker identity cancels inside
each difference, so what remains is language (+ channel of that population).
Projection x' = x - alpha * P P^T x is applied to ALL voice features (v4 train + every eval file) before the bridge.
Unlike CORAL it removes only k fixed directions learned on train data, never statistics of a test file.
Cross-language test: direction from v1 (En<->Urdu) evaluated on held-out v2 persons' HINDI, and the reverse.
English guardrail: v4 held-out file (cluster-level, as the dev pipeline).  Bridge = EXP-007 recipe on v4 train only."""
import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(int(os.environ.get("LANG01_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext")
import flag_v2 as V
sys.path.insert(0, str(R / "Experiment/EXT-03"))

store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
calib = lambda Z, lab: max(np.arange(.4, 1.11, .05), key=lambda t: adjusted_rand_score(lab, clus(Z, t)))
AF, AV = store.face("arcface", "train"), store.voice("ecapa192", "train")
mv, sv = store.Xv.mean(0), store.Xv.std(0) + 1e-6

EXT = {}
for src, lang in [("v1_complete", "urdu"), ("v2_complete", "hindi")]:
    D = store.ext_source(src)
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    persons = np.array(sorted({key(x) for x in D["vm"].spk}))
    EXT[src] = dict(D=D, key=key, lang=lang, folds=np.array_split(np.random.RandomState(0).permutation(persons), 5))
    store.gmap.update(D["gen_of"])


def language_subspace(src, persons, k):
    E = EXT[src]; D, key = E["D"], E["key"]; vm = D["vm"]
    Z = (D["V"] - mv) / sv; diffs = []
    for p, g in vm.groupby(vm.spk.map(key)):
        if p not in persons: continue
        a, b = g[g.lang == E["lang"]].index.values, g[g.lang == "english"].index.values
        if len(a) >= 3 and len(b) >= 3:
            diffs.append(Z[a].mean(0) - Z[b].mean(0))
    Dm = np.stack(diffs)
    if k == 1:
        P = (Dm.mean(0) / np.linalg.norm(Dm.mean(0)))[:, None]
    else:
        P = np.linalg.svd(Dm, full_matrices=False)[2][:k].T
    return P, len(diffs)


def project(X, P, alpha):
    Z = (X - mv) / sv
    Z = Z - alpha * (Z @ P) @ P.T
    return (Z * sv + mv).astype(np.float32)


def eval_file(src, held, rng):
    E = EXT[src]; D, key = E["D"], E["key"]; vm, fm = D["vm"], D["fm"]
    rows = vm[(vm.lang == E["lang"]) & vm.spk.map(lambda x: key(x) in held)]
    pick = []
    for p, g in rows.groupby(rows.spk.map(key)):
        pick += list(rng.choice(g.index, min(len(g), 15), replace=False))
    rows = vm.loc[sorted(pick)]; fi = []
    for r in rows.itertuples():
        cand = fm[(fm.spk.map(key) == key(r.spk)) & (fm.video != r.video)].index
        fi.append(rng.choice(cand if len(cand) else fm[fm.spk.map(key) == key(r.spk)].index))
    return D["Fa"][fi], D["V"][rows.index.values], np.array([key(x) for x in rows.spk])


def agg(E_, c):
    M = np.zeros((c.max() + 1, E_.shape[1])); np.add.at(M, c, E_); M = V.l2n(M)
    return V.l2n(0.01 * E_ + 0.99 * M[c])


OUT = Path(__file__).parent / "results.csv"
recs = []
VARIANTS = [("none", None, 0, 0)] + [(f"{s}->k{k} a{a}", s, k, a) for s in ("v1_complete", "v2_complete") for k in (1, 3) for a in (0.5, 1.0)]
for s in (1, 2, 3, 4, 5):
    tr, va = V.speaker_split(store.spk, s, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]; sub = tri[::3]
    cF = clus(cent(AF[vai]), calib(cent(AF[sub]), store.spk[sub])); cV = clus(cent(AV[vai]), calib(cent(AV[sub]), store.spk[sub]))
    T4 = {sg: V.build_trials(store.spk[vai], store.gmap, seed=s, same_gender=sg) for sg in (False, True)}
    rng = np.random.RandomState(100 + s)
    held = {src: set(E["folds"][s - 1]) for src, E in EXT.items()}
    files = {src: eval_file(src, held[src], rng) for src in EXT}
    Tx = {src: V.build_trials(px, {p: "m" for p in px}, seed=s) for src, (Fx, Vx, px) in files.items()}
    for name, dsrc, k, a in VARIANTS:
        if dsrc is None:
            proj = lambda X: X
            nper = 0
        else:
            keep = set(np.concatenate([f for i, f in enumerate(EXT[dsrc]["folds"]) if i != s - 1]))
            P, nper = language_subspace(dsrc, keep, k)
            proj = lambda X, P=P, a=a: project(X, P, a)
        TF, TV, TS = store.Xf[tri], proj(store.Xv[tri]), store.spk[tri]
        embs = {"v4": [], "v1_complete": [], "v2_complete": []}
        for i in range(2):
            net, prep = V.train_one(store, REC, None, seed=s + 100 * i, rows=(TF, TV, TS))
            mu_f, mu_v = prep.f(TF).mean(0), prep.v(TV).mean(0)
            embs["v4"].append(V.embed(net, prep, store.Xf[vai], proj(store.Xv[vai]), mu_f, mu_v))
            for src, (Fx, Vx, px) in files.items():
                embs[src].append(V.embed(net, prep, Fx, proj(Vx), mu_f, mu_v))
        r = dict(split=s, variant=name, n_persons_dir=nper)
        for sg, (fi, vj, lab) in T4.items():
            r["v4_" + ("g" if sg else "ng")] = V.eer_from_scores(np.mean([V.z(np.sum(agg(u, cF)[fi] * agg(w, cV)[vj], 1)) for u, w in embs["v4"]], 0), lab)
        for src, (Fx, Vx, px) in files.items():
            fi, vj, lab = Tx[src]; ids = pd.factorize(px)[0]; t = EXT[src]["lang"]
            r[t + "_sample"] = V.eer_from_scores(np.mean([V.z(np.sum(u[fi] * w[vj], 1)) for u, w in embs[src]], 0), lab)
            r[t + "_person"] = V.eer_from_scores(np.mean([V.z(np.sum(agg(u, ids)[fi] * agg(w, ids)[vj], 1)) for u, w in embs[src]], 0), lab)
        recs.append(r); pd.DataFrame(recs).to_csv(OUT, index=False)
        print({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
T = pd.DataFrame(recs)
print(T.groupby("variant", sort=False)[["v4_ng", "v4_g", "urdu_sample", "urdu_person", "hindi_sample", "hindi_person"]].mean().round(2).to_string())
print("ALL DONE", flush=True)
