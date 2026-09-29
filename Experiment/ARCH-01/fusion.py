"""ARCH-01b — MLP bridge + higher-rank ridge CCA, fused at matrix level.  TRUE labels, same splits / held-out
files / clusters / scoring as run.py (imported pieces are re-built identically here).
Pre-registered after split 1 of run.py (declared): P0 is production; the others change only the CCA part.
  P0  0.75 MLP + 0.25 C4                    (production member)
  F1  0.75 MLP + 0.25 C16
  F2  0.50 MLP + 0.50 C16
  F3  0.50 MLP + 0.25 C4 + 0.25 C16
  F4  0.50 MLP + 0.50 C8
  C16 / C8 alone (references)
C4 = production CCA (k 4, reg 1, PCA 128); Ck = run.py's CCABase (reg 10, PCA 256), all z-scored matrices."""
import os, sys, time, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(int(os.environ.get("ARCH_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext")
import flag_v2 as V
from flag_lib import RidgeCCA

OUT = Path(__file__).parent / "fusion.csv"
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
calib = lambda Z, lab: max(np.arange(.4, 1.11, .05), key=lambda t: adjusted_rand_score(lab, clus(Z, t)))
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
AF, AV = store.face("arcface", "train"), store.voice("ecapa192", "train")
EXT = {}
for src, lang in [("v1_complete", "urdu"), ("v2_complete", "hindi")]:
    D = store.ext_source(src)
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    persons = np.array(sorted({key(x) for x in D["vm"].spk}))
    EXT[src] = dict(D=D, key=key, lang=lang, folds=np.array_split(np.random.RandomState(0).permutation(persons), 5))
    store.gmap.update(D["gen_of"])


def ext_eval_file(src, held, rng):
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


def cca_matrix(TF, TV, Fx, Vx, k):
    m = RidgeCCA(k=k, reg=10.0, pca_x=256, pca_y=192).fit(TF, TV)
    raw = lambda X, Y: ((((X - m.mx) / m.sx) @ m.Px) @ m.Wx, (((Y - m.my) / m.sy) @ m.Py) @ m.Wy)
    a, b = raw(TF, TV); sa, sb = (a.mean(0), a.std(0) + 1e-6), (b.mean(0), b.std(0) + 1e-6)
    x, y = raw(V.centre(Fx, TF.mean(0)), V.centre(Vx, TV.mean(0)))
    return zm(V.l2n((x - sa[0]) / sa[1]) @ V.l2n((y - sb[0]) / sb[1]).T)


def block_scores(M, fi, vj, cf, cv, b=0.99):
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * M[fi, vj] + b * (Af @ M @ Av.T)[cf[fi], cv[vj]]


SYSTEMS = {"P0": {"mlp": .75, "c4": .25}, "F1": {"mlp": .75, "c16": .25}, "F2": {"mlp": .5, "c16": .5},
           "F3": {"mlp": .5, "c4": .25, "c16": .25}, "F4": {"mlp": .5, "c8": .5}, "C16": {"c16": 1}, "C8": {"c8": 1}}
recs = []
for s in (1, 2, 3, 4, 5):
    t0 = time.time()
    tr, va = V.speaker_split(store.spk, s, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]; sub = tri[::3]
    cF = clus(cent(AF[vai]), calib(cent(AF[sub]), store.spk[sub]))
    cV = clus(cent(AV[vai]), calib(cent(AV[sub]), store.spk[sub]))
    T4 = {sg: V.build_trials(store.spk[vai], store.gmap, seed=s, same_gender=sg) for sg in (False, True)}
    rng_e = np.random.RandomState(100 + s)
    files = {"v4": (store.Xf[vai], store.Xv[vai], None)}
    for src in EXT:
        files[src] = ext_eval_file(src, set(EXT[src]["folds"][s - 1]), rng_e)
    TF, TV, TS = store.Xf[tri], store.Xv[tri], store.spk[tri]
    nets = [V.train_one(store, REC, None, seed=s + 100 * i, rows=(TF, TV, TS)) for i in range(2)]
    for f, (Fx, Vx, px) in files.items():
        emb = [V.embed(net, prep, Fx, Vx, prep.f(TF).mean(0), prep.v(TV).mean(0)) for net, prep in nets]
        c4 = V.cca_proj(TF, TV, Fx, Vx)
        Ms = {"mlp": np.mean([zm(a @ b.T) for a, b in emb], 0), "c4": zm(c4[0] @ c4[1].T),
              "c8": cca_matrix(TF, TV, Fx, Vx, 8), "c16": cca_matrix(TF, TV, Fx, Vx, 16)}
        if f == "v4":
            trials = {("g" if sg else "ng"): t for sg, t in T4.items()}; cf, cv, tgt, lvl = cF, cV, "v4_en", "cluster"
        else:
            gm = {p: store.gmap.get(p, "u") for p in px}
            protos = [False, True] if all(g in "mf" for g in gm.values()) else [False]
            trials = {("g" if sg else "ng"): V.build_trials(px, gm, seed=s, same_gender=sg) for sg in protos}
            cf = cv = pd.factorize(px)[0]; tgt, lvl = EXT[f]["lang"], "person"
        for name, wts in SYSTEMS.items():
            M = sum(w * Ms[m] for m, w in wts.items())
            for p, (fi, vj, lab) in trials.items():
                recs.append(dict(split=s, system=name, target=tgt, protocol=p, level=lvl,
                                 eer=V.eer_from_scores(block_scores(M, fi, vj, cf, cv), lab)))
                recs.append(dict(split=s, system=name, target=tgt, protocol=p, level="sample",
                                 eer=V.eer_from_scores(M[fi, vj], lab)))
    pd.DataFrame(recs).to_csv(OUT, index=False)
    d = pd.DataFrame(recs); d = d[(d.split == s) & (d.level != "sample")]
    print(f"split {s} {time.time() - t0:.0f}s\n" + d.pivot_table(index="system", columns=["target", "protocol"], values="eer").round(2).to_string(), flush=True)
print("ALL DONE", flush=True)
