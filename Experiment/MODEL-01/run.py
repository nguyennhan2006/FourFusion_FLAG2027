"""MODEL-01 — new encoders as bridge streams (S1), ImageBind as a stream (IB-02), explicit age match (S2). TRUE labels.

Protocol = docs/OPEN_DIRECTIONS.md §1 (same as ARCH-01): v4 speaker_split(s, n_val=30), s = 1..5, bridges trained on
the 40 remaining v4 speakers; held-out Urdu (v1) / Hindi (v2) persons of fold s, positives from ANOTHER video.
Primary metric = SAMPLE-level EER (one face-voice pair, nothing else of the file). Secondary = cluster level (v4,
ArcFace / ECAPA-192 clusters, b = 0.99, as SET-02) and person level (Urdu / Hindi).

Pre-registered arms (fixed before any result; weights are not tuned):
  CTRL             production s007 bridge: MLP (EXP-007 recipe, n = 2 models) + 0.25 CCA-4 on VGG 4096 + BTC 192
  IB               ImageBind zero-shot alone (file-centred cosine of face_ibv / voice_iba)
  CTRL+IB.25/.5    (1 - w) z(CTRL) + w z(IB)
  <S>              the same bridge recipe with face = stream S (voice stays BTC 192)       arm (a) of PLAN_MODELS S1
  CTRL+<S>.25/.5   (1 - w) z(CTRL) + w z(<S>)                                               arm (c)
  CAT<S>           face = [VGG through the control's own Prep (256 PCs), + PCA-32 of S at the mean VGG-PC variance]  (b)
  CTRL+AGE.1/.25   z(CTRL) + w z(-|age_face - age_voice|), ages from vitageprob / w2v2agage (skipped until present)
  streams S: vitage, farl, siglip2, adaface (FLAG_10); btcfacefc6, btcfacepool5 = other layers of the organiser's
             VGG-Face (FLAG_11, OPEN_DIRECTIONS S3). Each only if its arrays exist; a re-run adds the missing ones.
  voice:<S>, CTRL+voice:<S>.25/.5   the same with a voice stream (VGG face): btcvoiceasp = the organiser ECAPA's
                   3072-d pooling layer (S3)
  TTAface / TTAvoice / TTAboth      CTRL's trained bridge, test inputs replaced by test-time-augmented features (S4):
                   face = mean(fc7, fc7 of the mirrored image), voice = btcvoicetta (192 averaged over 3 views)
  R:SET10 … R:SET14  the candidate recipes of build_candidates.py on the same components (registered 28/09, log 63)
z() = z-score over the file's full face x voice matrix, so no score depends on how the trial list pairs samples.
Results are appended to results.csv, one block per (split, arm); a re-run only computes what is missing."""
import os, sys, time
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle"))
torch.set_num_threads(int(os.environ.get("MODEL_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext")          # validation rows (12 s crop), as ARCH-01
import flag_v2 as V
from flag_models import _media_key

SPLITS = [int(x) for x in os.environ.get("MODEL_SPLITS", "1,2,3,4,5").split(",")]
N_MODELS, EPOCHS = int(os.environ.get("MODEL_MODELS", "2")), int(os.environ.get("MODEL_EPOCHS", "40"))
FM, FXF = R / "kaggle/output/feats_models", R / "kaggle/output/feats_ext_full"
FL = Path(os.environ.get("MODEL_FEATS_LAYERS", R / "kaggle/output/feats_layers"))       # FLAG_11 output
OUT = Path(__file__).parent / os.environ.get("MODEL_OUT", "results.csv")
_has = lambda name, kind="face": any((d / f"{kind}_{name}_train.npy").exists() for d in (FM, FL))
STREAMS = [s for s in ["vitage", "farl", "siglip2", "adaface", "btcfacefc6", "btcfacepool5"] if _has(s)]
VSTREAMS = [s for s in ["btcvoiceasp"] if _has(s, "voice")]
TTA = [a for a, ok in [("TTAface", _has("btcfaceflip")), ("TTAvoice", _has("btcvoicetta", "voice")),
                       ("TTAboth", _has("btcfaceflip") and _has("btcvoicetta", "voice"))] if ok]
HAS_AGE = _has("w2v2agage", "voice") and _has("vitageprob")
AGE_MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)       # centres of the 9 FairFace age bins

store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
tr_index = pd.read_csv(FM / "index_train.txt", sep=" ", header=None)
assert (tr_index[4].values == store.spk).all(), "feats_models train rows are not in the organiser order"
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=EPOCHS)
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
calib = lambda Z, lab: max(np.arange(.4, 1.11, .05), key=lambda t: adjusted_rand_score(lab, clus(Z, t)))
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
AF, AV = store.face("arcface", "train"), store.voice("ecapa192", "train")


def _find(fname):
    for d in (FM, FL):
        if (d / fname).exists():
            return d / fname
    raise FileNotFoundError(fname)


def v4_arr(name, kind="face"):
    return np.load(_find(f"{kind}_{name}_train.npy")).astype(np.float32)


# ---------------------------------------------------------------- external held-out files (EXT-03 / ARCH-01 construction)
EXT = {}
for src, lang in [("v1_complete", "urdu"), ("v2_complete", "hindi")]:
    D = store.ext_source(src)
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    persons = np.array(sorted({key(x) for x in D["vm"].spk}))
    # rows of the capped validation meta -> rows of feats_ext_full, which the FLAG_10 streams follow (faces: same order)
    full_v = pd.read_csv(FXF / f"ext_{src}_voice_meta.csv").path.map(_media_key)
    vmap = pd.Series(np.arange(len(full_v)), index=full_v.values)[D["vm"].path.map(_media_key).values].values
    full_f = pd.read_csv(FXF / f"ext_{src}_face_meta.csv").path.map(_media_key)
    assert full_f.equals(D["fm"].path.map(_media_key)), f"{src}: face rows differ between feats_ext and feats_ext_full"
    EXT[src] = dict(D=D, key=key, lang=lang, vmap=vmap,
                    folds=np.array_split(np.random.RandomState(0).permutation(persons), 5))
    store.gmap.update(D["gen_of"])


def ext_arr(src, name, kind):
    return np.load(_find(f"ext_{src}_{kind}_{name}.npy")).astype(np.float32)


def ext_has(src, name, kind):
    """A stream may exist for v4 only (e.g. FLAG_11 run without the v1/v2 datasets): its arms then skip that file."""
    return any((d / f"ext_{src}_{kind}_{name}.npy").exists() for d in (FM, FL))


def ext_eval_rows(src, held, rng):
    """ARCH-01's ext_eval_file, returning row indices: face rows (feats_ext == full order), voice rows (feats_ext)."""
    E = EXT[src]; D, key = E["D"], E["key"]; vm, fm = D["vm"], D["fm"]
    rows = vm[(vm.lang == E["lang"]) & vm.spk.map(lambda x: key(x) in held)]
    pick = []
    for p, g in rows.groupby(rows.spk.map(key)):
        pick += list(rng.choice(g.index, min(len(g), 15), replace=False))
    rows = vm.loc[sorted(pick)]; fi = []
    for r in rows.itertuples():
        cand = fm[(fm.spk.map(key) == key(r.spk)) & (fm.video != r.video)].index
        fi.append(rng.choice(cand if len(cand) else fm[fm.spk.map(key) == key(r.spk)].index))
    return np.array(fi), rows.index.values, np.array([key(x) for x in rows.spk])


# ---------------------------------------------------------------- scoring components (each -> full matrix per file)
class CatPrep:
    """Arm (b): face = [VGG through the control's own Prep (its 256 PCs, unchanged), + PCA-k of the stream].
    Stream PCs are whitened then scaled to the mean variance of the VGG PCs: k more components of comparable size."""
    def __init__(self, base, vgg_tr, s_tr, k=32):
        self.base, self.d = base, vgg_tr.shape[1]
        self.ms, self.ss = s_tr.mean(0), s_tr.std(0) + 1e-6
        _, _, Vt = np.linalg.svd(((s_tr - self.ms) / self.ss).astype(np.float64), full_matrices=False)
        self.P = Vt[:k].T.astype(np.float32)
        self.sd = (((s_tr - self.ms) / self.ss) @ self.P).std(0) + 1e-6
        self.scale = float(np.sqrt(base.f(vgg_tr).var(0).mean()))

    def f(self, X):
        s = ((X[:, self.d:] - self.ms) / self.ss) @ self.P / self.sd * self.scale
        return np.hstack([self.base.f(X[:, :self.d]), s]).astype(np.float32)

    def v(self, X):
        return self.base.v(X)


def bridge(TF, TV, TS, files, seed0, prep_fn=None):
    """Production recipe on the given face features: n MLPs (mean of z-scored cosines) * 0.75 + CCA-4 * 0.25."""
    embs = {f: [] for f in files}
    for i in range(N_MODELS):
        prep = prep_fn() if prep_fn else None
        net, prep = V.train_one(store, REC, None, seed=seed0 + 100 * i, rows=(TF, TV, TS), prep=prep)
        mu_f, mu_v = prep.f(TF).mean(0), prep.v(TV).mean(0)
        for f, (Fx, Vx) in files.items():
            embs[f].append(V.embed(net, prep, Fx, Vx, mu_f, mu_v))
    out = {}
    for f, (Fx, Vx) in files.items():
        M0 = np.mean([zm(a @ b.T) for a, b in embs[f]], 0)
        cf, cv = V.cca_proj(TF, TV, Fx, Vx)
        out[f] = 0.75 * M0 + 0.25 * zm(cf @ cv.T)
    return out


def filecos(Fx, Vx):
    return V.l2n(Fx - Fx.mean(0)) @ V.l2n(Vx - Vx.mean(0)).T


def block_scores(M, fi, vj, cf, cv, b=0.99):
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * M[fi, vj] + b * (Af @ M @ Av.T)[cf[fi], cv[vj]]


RECIPES = {   # arm -> (components needed besides CTRL, recipe); same formulas as build_candidates.py
    "R:SET10": (["AGE"], lambda c: zm(c["CTRL"]) + 0.1 * zm(c["AGE"])),
    "R:SET11": (["farl", "AGE"], lambda c: zm(0.5 * zm(c["CTRL"]) + 0.5 * zm(c["farl"])) + 0.1 * zm(c["AGE"])),
    "R:SET12": (["IB"], lambda c: 0.5 * zm(c["CTRL"]) + 0.5 * zm(c["IB"])),
    "R:SET13": (["IB", "AGE"], lambda c: zm(0.5 * zm(c["CTRL"]) + 0.5 * zm(c["IB"])) + 0.1 * zm(c["AGE"])),
    "R:SET14": (["farl", "IB", "AGE"], lambda c: zm((zm(c["CTRL"]) + zm(c["farl"]) + zm(c["IB"])) / 3) + 0.1 * zm(c["AGE"])),
}


def arms_wanted():
    a = ["CTRL", "IB", "CTRL+IB.25", "CTRL+IB.5"]
    for s in STREAMS:
        a += [s, f"CTRL+{s}.25", f"CTRL+{s}.5", f"CAT{s}"]
    for s in VSTREAMS:
        a += [f"voice:{s}", f"CTRL+voice:{s}.25", f"CTRL+voice:{s}.5"]
    a += TTA
    if HAS_AGE:
        a += ["CTRL+AGE.1", "CTRL+AGE.25"]
    a += [r for r, (need, _) in RECIPES.items() if HAS_AGE and ("farl" not in need or "farl" in STREAMS)]
    return a


done = set()
if OUT.exists():
    prev = pd.read_csv(OUT); done = set(zip(prev.split, prev.arm))
print("streams:", STREAMS, "| voice streams:", VSTREAMS, "| tta:", TTA, "| age:", HAS_AGE, "| arms:", arms_wanted(), flush=True)
for s in SPLITS:
    todo = [a for a in arms_wanted() if (s, a) not in done]
    if not todo:
        print("split", s, "complete"); continue
    t_split = time.time()
    tr, va = V.speaker_split(store.spk, s, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]; sub = tri[::3]
    cF = clus(cent(AF[vai]), calib(cent(AF[sub]), store.spk[sub]))
    cV = clus(cent(AV[vai]), calib(cent(AV[sub]), store.spk[sub]))
    T4 = {sg: V.build_trials(store.spk[vai], store.gmap, seed=s, same_gender=sg) for sg in (False, True)}
    rng_e = np.random.RandomState(100 + s)
    ROWS = {"v4": (vai, vai, None)}                       # file -> (face rows, voice rows, person ids)
    for src in EXT:
        ROWS[src] = ext_eval_rows(src, set(EXT[src]["folds"][s - 1]), rng_e)
    Tx = {}
    for src in EXT:
        px = ROWS[src][2]; gm = {p: store.gmap.get(p, "u") for p in px}
        protos = [False, True] if all(g in "mf" for g in gm.values()) else [False]
        Tx[src] = {sg: V.build_trials(px, gm, seed=s, same_gender=sg) for sg in protos}

    def feats(face_name=None, voice_name=None):
        """{file: (face matrix, voice matrix)} for the eval files; None = the organiser VGG / BTC features."""
        out = {}
        for f, (fr, vr, _) in ROWS.items():
            if f == "v4":
                Fx = store.Xf[fr] if face_name is None else v4_arr(face_name)[fr]
                Vx = store.Xv[vr] if voice_name is None else v4_arr(voice_name, "voice")[vr]
            else:
                if (face_name and not ext_has(f, face_name, "face")) or (voice_name and not ext_has(f, voice_name, "voice")):
                    continue
                D = EXT[f]["D"]
                Fx = D["Fa"][fr] if face_name is None else ext_arr(f, face_name, "face")[fr]
                Vx = D["V"][vr] if voice_name is None else ext_arr(f, voice_name, "voice")[EXT[f]["vmap"][vr]]
            out[f] = (Fx, Vx)
        return out

    TF, TV, TS = store.Xf[tri], store.Xv[tri], store.spk[tri]
    base_files = feats()
    ctrl_files = dict(base_files)
    if any(a in todo for a in TTA):
        flipF, ttaV = (feats("btcfaceflip") if "TTAface" in TTA else None), (feats(None, "btcvoicetta") if "TTAvoice" in TTA else None)
        for f, (Fx, Vx) in base_files.items():
            if flipF is not None and f in flipF:
                ctrl_files[f + "|TTAface"] = ((Fx + flipF[f][0]) / 2, Vx)
            if ttaV is not None and f in ttaV:
                ctrl_files[f + "|TTAvoice"] = (Fx, ttaV[f][1])
            if flipF is not None and ttaV is not None and f in flipF and f in ttaV:
                ctrl_files[f + "|TTAboth"] = ((Fx + flipF[f][0]) / 2, ttaV[f][1])
    COMP = {"CTRL": bridge(TF, TV, TS, ctrl_files, s)}
    for a in TTA:
        COMP[a] = {f: COMP["CTRL"][f + "|" + a] for f in base_files if f + "|" + a in COMP["CTRL"]}
    needs = lambda comp: any(a in RECIPES and comp in RECIPES[a][0] for a in todo)
    if any(a.startswith("CTRL+IB") or a == "IB" for a in todo) or needs("IB"):
        COMP["IB"] = {f: filecos(*xy) for f, xy in feats("ibv", "iba").items()}
    for st in STREAMS:
        mine = [a for a in todo if a in (st, f"CTRL+{st}.25", f"CTRL+{st}.5")]
        if mine or needs(st):
            TFs = v4_arr(st)[tri]
            COMP[st] = bridge(TFs, TV, TS, {f: (Fs, base_files[f][1]) for f, (Fs, _) in feats(st).items()}, s)
        if f"CAT{st}" in todo:
            TFs = v4_arr(st)[tri]
            TFc = np.hstack([TF, TFs])
            cat_files = {f: (np.hstack([base_files[f][0], Fs]), base_files[f][1]) for f, (Fs, _) in feats(st).items()}
            COMP[f"CAT{st}"] = bridge(TFc, TV, TS, cat_files, s,
                                      prep_fn=lambda: CatPrep(V.Prep(TF, TV, REC["pca_f"], REC["pca_v"]), TF, TFs))
    for vs in VSTREAMS:
        if any(a in todo for a in (f"voice:{vs}", f"CTRL+voice:{vs}.25", f"CTRL+voice:{vs}.5")):
            vfiles = {f: (base_files[f][0], Vs) for f, (_, Vs) in feats(None, vs).items()}
            COMP[f"voice:{vs}"] = bridge(TF, v4_arr(vs, "voice")[tri], TS, vfiles, s)
    if HAS_AGE and (any(a.startswith("CTRL+AGE") for a in todo) or needs("AGE")):
        ag = {}
        for f, (fr, vr, _) in ROWS.items():
            if f == "v4":
                af, av = v4_arr("vitageprob")[fr] @ AGE_MIDS, v4_arr("w2v2agage", "voice")[vr][:, 0] * 100
            elif ext_has(f, "vitageprob", "face") and ext_has(f, "w2v2agage", "voice"):
                af = ext_arr(f, "vitageprob", "face")[fr] @ AGE_MIDS
                av = ext_arr(f, "w2v2agage", "voice")[EXT[f]["vmap"][vr]][:, 0] * 100
            else:
                continue
            ag[f] = -np.abs(af[:, None] - av[None, :])
        COMP["AGE"] = ag

    def matrix(arm, f):
        """None when a component has no matrix for this file (stream without v1/v2 arrays)."""
        if arm in COMP:
            return COMP[arm].get(f)
        if arm in RECIPES:
            need, fn = RECIPES[arm]
            if any(f not in COMP[x] for x in ["CTRL"] + need):
                return None
            return fn({x: COMP[x][f] for x in ["CTRL"] + need})
        base, w = arm.rsplit(".", 1)
        other, w = base.split("+", 1)[1], float("0." + w)
        if f not in COMP[other]:
            return None
        if other == "AGE":
            return zm(COMP["CTRL"][f]) + w * zm(COMP["AGE"][f])
        return (1 - w) * zm(COMP["CTRL"][f]) + w * zm(COMP[other][f])

    for arm in todo:
        rec = []
        for f in ROWS:
            M = matrix(arm, f)
            if M is None:
                continue
            if f == "v4":
                for sg, (fi, vj, lab) in T4.items():
                    p = "g" if sg else "ng"
                    rec += [("v4_en", p, "sample", V.eer_from_scores(M[fi, vj], lab)),
                            ("v4_en", p, "cluster", V.eer_from_scores(block_scores(M, fi, vj, cF, cV), lab))]
            else:
                ids = pd.factorize(ROWS[f][2])[0]; tgt = EXT[f]["lang"]
                for sg, (fi, vj, lab) in Tx[f].items():
                    p = "g" if sg else "ng"
                    rec += [(tgt, p, "sample", V.eer_from_scores(M[fi, vj], lab)),
                            (tgt, p, "person", V.eer_from_scores(block_scores(M, fi, vj, ids, ids), lab))]
        df = pd.DataFrame(rec, columns=["target", "protocol", "level", "eer"])
        df.insert(0, "arm", arm); df.insert(0, "split", s)
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"split {s} {arm:16s} | " + " ".join(f"{t}/{p}/{l[0]} {e:.2f}" for t, p, l, e in rec), flush=True)
    print(f"== split {s} done in {time.time() - t_split:.0f}s", flush=True)
print("ALL DONE", flush=True)
