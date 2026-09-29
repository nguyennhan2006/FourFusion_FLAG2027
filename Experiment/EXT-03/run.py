"""EXT-03 — does the bridge get better at the PERSON level with external South-Asian identities (v1 Urdu, v2 Hindi)
and/or with a set-level loss (SetProto)?  TRUE labels only, 5 paired splits.

Per split s (identical held-out sets for every arm, so arm differences are paired):
  v4 : 30 held-out speakers (speaker_split n_val=30, seed s) -> English file, scored like the dev pipeline:
       sample level AND cluster level (ArcFace face clusters x own ECAPA-192 voice clusters, thresholds calibrated
       on the split's v4 train speakers), gender + no_gender trials.
  v1 : fold s of the v1 PERSONS (ids merged by name) held out -> Urdu file (face from another video than the voice),
       sample level + person level (true identity), gender + no_gender.
  v2 : fold s of the v2 persons held out -> Hindi file, no_gender only (no gender metadata).
Arms (bridge = EXP-007 recipe; external rows = organiser feature space, cap 150 rows/person, all languages):
  A  v4 only                     B1 v4 + v1          B2 v4 + v2          B3 v4 + v1 + v2
  AP A + SetProto (pk sampler, w_proto 0.5)          B3P B3 + SetProto
Unseen-language checks: B1 on Hindi, B2 on Urdu.  Results appended to results.csv (resumable)."""
import os, sys, time, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(int(os.environ.get("EXT03_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext")
import flag_v2 as V

SPLITS = [int(x) for x in os.environ.get("EXT03_SPLITS", "1,2,3,4,5").split(",")]
N_MODELS = int(os.environ.get("EXT03_MODELS", "2"))
EPOCHS = int(os.environ.get("EXT03_EPOCHS", "40"))
# arms with external rows see ~5.6x more rows per epoch; 15 epochs still gives them MORE optimizer steps than A at 40
EPOCHS_EXT = int(os.environ.get("EXT03_EPOCHS_EXT", "15"))
ARMS = os.environ.get("EXT03_ARMS", "A,B1,B2,B3,AP,B3P").split(",")
OUT = Path(__file__).parent / os.environ.get("EXT03_OUT", "results.csv")

store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=EPOCHS)
PROTO = dict(sampler="pk", pk_P=32, pk_K=8, w_proto=0.5)
ARM_CFG = {"A": ((), {}), "B1": (("v1_complete",), {}), "B2": (("v2_complete",), {}),
           "B3": (("v1_complete", "v2_complete"), {}), "AP": ((), PROTO), "B3P": (("v1_complete", "v2_complete"), PROTO)}
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
calib = lambda Z, lab: max(np.arange(.4, 1.11, .05), key=lambda t: adjusted_rand_score(lab, clus(Z, t)))
AF, AV = store.face("arcface", "train"), store.voice("ecapa192", "train")

# ---- external sources: persons (ids merged by name), folds, held-out files
EXT = {}
for src, lang in [("v1_complete", "urdu"), ("v2_complete", "hindi")]:
    D = store.ext_source(src)
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    persons = np.array(sorted({key(x) for x in D["vm"].spk}))
    folds = np.array_split(np.random.RandomState(0).permutation(persons), 5)
    EXT[src] = dict(D=D, key=key, persons=persons, folds=folds, lang=lang)
    store.gmap.update(D["gen_of"])


def ext_rows_for(src, keep_persons, cap, rng):
    D, key = EXT[src]["D"], EXT[src]["key"]
    ids = [x for x in D["vm"].spk.unique() if key(x) in keep_persons]
    return V.pair_ext(D, src, cap, rng, ids=ids)


def ext_eval_file(src, held, rng):
    """Held-out persons' target-language clips (<=15 per person) paired with a face from ANOTHER video."""
    E = EXT[src]; D, key = E["D"], E["key"]
    vm = D["vm"]; fm = D["fm"]
    rows = vm[(vm.lang == E["lang"]) & vm.spk.map(lambda x: key(x) in held)]
    pick = []
    for p, g in rows.groupby(rows.spk.map(key)):
        pick += list(rng.choice(g.index, min(len(g), 15), replace=False))
    rows = vm.loc[sorted(pick)]
    fi = []
    for r in rows.itertuples():
        cand = fm[(fm.spk.map(key) == key(r.spk)) & (fm.video != r.video)].index
        cand = cand if len(cand) else fm[fm.spk.map(key) == key(r.spk)].index
        fi.append(rng.choice(cand))
    return D["Fa"][fi], D["V"][rows.index.values], np.array([key(x) for x in rows.spk])


def agg(E, c):
    M = np.zeros((c.max() + 1, E.shape[1])); np.add.at(M, c, E); M = V.l2n(M)
    return V.l2n(0.01 * E + 0.99 * M[c])


done = set()
if OUT.exists():
    prev = pd.read_csv(OUT); done = set(zip(prev.split, prev.arm))
for s in SPLITS:
    tr, va = V.speaker_split(store.spk, s, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]
    sub = tri[::3]
    cF = clus(cent(AF[vai]), calib(cent(AF[sub]), store.spk[sub]))
    cV = clus(cent(AV[vai]), calib(cent(AV[sub]), store.spk[sub]))
    sp4 = store.spk[vai]
    T4 = {sg: V.build_trials(sp4, store.gmap, seed=s, same_gender=sg) for sg in (False, True)}
    held = {src: set(E["folds"][s - 1]) for src, E in EXT.items()}
    rng_e = np.random.RandomState(100 + s)
    files = {src: ext_eval_file(src, held[src], rng_e) for src in EXT}
    Text = {}
    for src, (Fx, Vx, px) in files.items():
        gm = {p: store.gmap.get(p, "u") for p in px}
        protos = [False, True] if all(g in "mf" for g in gm.values()) else [False]
        Text[src] = {sg: V.build_trials(px, gm, seed=s, same_gender=sg) for sg in protos}
    for arm in ARMS:
        if (s, arm) in done:
            print("skip", s, arm, flush=True); continue
        t0 = time.time(); srcs, extra = ARM_CFG[arm]
        rng = np.random.RandomState(s)
        TF, TV, TS = store.Xf[tri], store.Xv[tri], store.spk[tri]
        for src in srcs:
            keep = set(EXT[src]["persons"]) - held[src]
            EF, EV, ES = ext_rows_for(src, keep, 150, rng)
            TF, TV, TS = np.concatenate([TF, EF]), np.concatenate([TV, EV]), np.concatenate([TS, ES])
        embs = {"v4": [], "v1_complete": [], "v2_complete": []}
        for i in range(N_MODELS):
            ep = EPOCHS_EXT if srcs else EPOCHS
            net, prep = V.train_one(store, dict(REC, epochs=ep, **extra), None, seed=s + 100 * i, rows=(TF, TV, TS))
            mu_f, mu_v = prep.f(TF).mean(0), prep.v(TV).mean(0)
            embs["v4"].append(V.embed(net, prep, store.Xf[vai], store.Xv[vai], mu_f, mu_v))
            for src, (Fx, Vx, px) in files.items():
                embs[src].append(V.embed(net, prep, Fx, Vx, mu_f, mu_v))
        rec = []
        for sg, (fi, vj, lab) in T4.items():
            p = "g" if sg else "ng"
            ss = np.mean([V.z(np.sum(u[fi] * w[vj], 1)) for u, w in embs["v4"]], 0)
            sc = np.mean([V.z(np.sum(agg(u, cF)[fi] * agg(w, cV)[vj], 1)) for u, w in embs["v4"]], 0)
            rec += [("v4_en", p, "sample", V.eer_from_scores(ss, lab)), ("v4_en", p, "cluster", V.eer_from_scores(sc, lab))]
        for src, (Fx, Vx, px) in files.items():
            ids = pd.factorize(px)[0]
            for sg, (fi, vj, lab) in Text[src].items():
                p = "g" if sg else "ng"
                ss = np.mean([V.z(np.sum(u[fi] * w[vj], 1)) for u, w in embs[src]], 0)
                sp_ = np.mean([V.z(np.sum(agg(u, ids)[fi] * agg(w, ids)[vj], 1)) for u, w in embs[src]], 0)
                tgt = "urdu" if src == "v1_complete" else "hindi"
                rec += [(tgt, p, "sample", V.eer_from_scores(ss, lab)), (tgt, p, "person", V.eer_from_scores(sp_, lab))]
        df = pd.DataFrame(rec, columns=["target", "protocol", "level", "eer"])
        df.insert(0, "arm", arm); df.insert(0, "split", s); df["n_train_rows"] = len(TS); df["sec"] = round(time.time() - t0)
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"split {s} arm {arm}: {len(TS)} rows, {time.time() - t0:.0f}s | " +
              " ".join(f"{t}/{p}/{l} {e:.2f}" for t, p, l, e in rec), flush=True)
print("ALL DONE", flush=True)
