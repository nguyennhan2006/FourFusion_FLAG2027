"""VAL-70 — validation with the members trained on ALL 70 v4 speakers, exactly as submitted. TRUE labels.

Why: three times in a row (SET-07, SET-08, SET-09) the v4 held-out proxy (bridges trained on 40 speakers) predicted a
gain that the dev board did not show; English was wrong in both cells for SET-09. Here nothing is held out of v4: the
member is the production one, and the test persons come from MAV-Celeb v1 (Urdu + English) and v2 (Hindi + English),
never used for training. Dev-like files: 30 persons, up to 15 voices each, every voice paired with a face of the same
person from ANOTHER video (EXT-03 construction), R files per (source, language).
Scored like the submission: z-scored full matrix per file; sample level and person level (true identities as the
blocks, b = 0.99; our estimated clusters are within ~0.9 EER of that, CEIL).

Systems (the s007 member only: SET-08 / SET-09 changed nothing else):
  SET02  s007 = n MLP (EXP-007) + 0.25 CCA-4                             (CB 21.68)
  SET08  0.5 MLP + 0.25 CCA-4 + 0.25 CCA-16                              (CB 22.30: -0.20 / 0.00 / +0.41 / +2.25)
  SET09  0.5 z(s007) + 0.5 z(f007), f007 = s007 recipe on FaRL faces    (CB 22.49: +1.69 / -1.00 / +0.41 / +2.11)
Retro-check: VAL-70 is usable as a gate only if its SET08 - SET02 and SET09 - SET02 signs match the board.
  (It did not: seed noise explains the board, see NOTES.md; VAL-70 is now the gate for Evaluation decisions.)
Round 2 (registered 28/09 before running, after MODEL-01 round 2), all on the s007 member:
  AGE.1 / AGE.25   z(s007) + w z(-|age_face - age_voice|)   ages: vitageprob (FairFace bins) / audeering w2v2 age
  FARL5+AGE.1      SET09's member, then + 0.1 z(age)
Members are cached in out/members.pkl (training them takes ~40 min on CPU).
Round 3 (registered 28/09, log 63, ImageBind allowed): the candidate recipes of MODEL-01/build_candidates.py
  SET10 = AGE.1 · SET11 = FARL5+AGE.1 · SET12 = 0.5 z(s007) + 0.5 z(IB) · SET13 = SET12 + 0.1 z(AGE)
  SET14 = z((z(s007) + z(f007) + z(IB)) / 3) + 0.1 z(AGE)          IB = file-centred cosine of ImageBind embeddings
"""
import os, sys, time, pickle
import numpy as np, pandas as pd, torch
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle"))
torch.set_num_threads(int(os.environ.get("VAL_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext")
import flag_v2 as V
from flag_lib import RidgeCCA
from flag_models import _media_key

N_MODELS, N_FILES, PERSONS = int(os.environ.get("VAL_MODELS", "5")), int(os.environ.get("VAL_FILES", "10")), 30
FM, FXF = R / "kaggle/output/feats_models", R / "kaggle/output/feats_ext_full"
OUT = Path(__file__).parent
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)

# ---------------------------------------------------------------- dev-like files from v1 / v2 persons
EXT = {}
for src, langs in [("v1_complete", ["english", "urdu"]), ("v2_complete", ["english", "hindi"])]:
    D = store.ext_source(src)
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    full_v = pd.read_csv(FXF / f"ext_{src}_voice_meta.csv").path.map(_media_key)
    vmap = pd.Series(np.arange(len(full_v)), index=full_v.values)[D["vm"].path.map(_media_key).values].values
    EXT[src] = dict(D=D, key=key, langs=langs, vmap=vmap)
    store.gmap.update(D["gen_of"])


def file_rows(src, lang, rng):
    """One dev-like file: 30 random persons that have voices in `lang`; faces from another video (EXT-03)."""
    E = EXT[src]; D, key = E["D"], E["key"]; vm, fm = D["vm"], D["fm"]
    rows = vm[vm.lang == lang]
    persons = np.array(sorted({key(x) for x in rows.spk}))
    held = set(rng.choice(persons, min(PERSONS, len(persons)), replace=False))
    rows = rows[rows.spk.map(lambda x: key(x) in held)]
    pick = []
    for p, g in rows.groupby(rows.spk.map(key)):
        pick += list(rng.choice(g.index, min(len(g), 15), replace=False))
    rows = vm.loc[sorted(pick)]; fi = []
    for r in rows.itertuples():
        cand = fm[(fm.spk.map(key) == key(r.spk)) & (fm.video != r.video)].index
        fi.append(rng.choice(cand if len(cand) else fm[fm.spk.map(key) == key(r.spk)].index))
    return np.array(fi), rows.index.values, np.array([key(x) for x in rows.spk])


FILES = []
for src, E in EXT.items():
    for lang in E["langs"]:
        for i in range(N_FILES):
            fi, vi, ids = file_rows(src, lang, np.random.RandomState(1000 * i + len(lang)))
            gm = {p: store.gmap.get(p, "u") for p in ids}
            protos = [False, True] if all(g in "mf" for g in gm.values()) else [False]
            FILES.append(dict(src=src, lang=lang, i=i, fi=fi, vi=vi, ids=ids,
                              trials={sg: V.build_trials(ids, gm, seed=i, same_gender=sg) for sg in protos}))
print(f"{len(FILES)} files:", {(f['src'][:2], f['lang']): len(f['ids']) for f in FILES if f['i'] == 0}, flush=True)

# ---------------------------------------------------------------- members on all 70 v4 speakers (production recipe)
TF, TV, TS = store.Xf, store.Xv, store.spk
FARL_TR = np.load(FM / "face_farl_train.npy").astype(np.float32)


def eval_inputs(face="vgg"):
    out = []
    for f in FILES:
        D = EXT[f["src"]]["D"]
        Fx = D["Fa"][f["fi"]] if face == "vgg" else np.load(FM / f"ext_{f['src']}_face_{face}.npy").astype(np.float32)[f["fi"]]
        out.append((Fx, D["V"][f["vi"]]))
    return out


def mlp_mats(TFx, inputs, off=0):
    """mean over n models of z-scored cosine matrices, seeds as dev_scores (1, 101, ...) + off."""
    acc = [0.0] * len(inputs)
    for i in range(N_MODELS):
        net, prep = V.train_one(store, REC, None, seed=1 + off + 100 * i, rows=(TFx, TV, TS))
        mu_f, mu_v = prep.f(TFx).mean(0), prep.v(TV).mean(0)
        for j, (Fx, Vx) in enumerate(inputs):
            a, b = V.embed(net, prep, Fx, Vx, mu_f, mu_v)
            acc[j] = acc[j] + zm(a @ b.T) / N_MODELS
    return acc


def cca4_mats(TFx, inputs):
    """production CCA-4 (flag_v2.cca_proj, as dev_scores)."""
    out = []
    for Fx, Vx in inputs:
        a, b = V.cca_proj(TFx, TV, Fx, Vx)
        out.append(zm(a @ b.T))
    return out


def cca16_mats(TFx, inputs, k=16):
    """SET-08's CCA-16 (ARCH-01/dev_mats.py): reg 10, PCA 256 / 192, variates standardised with train stats."""
    m = RidgeCCA(k=k, reg=10.0, pca_x=256, pca_y=192).fit(TFx, TV)
    raw = lambda X, Y: ((((X - m.mx) / m.sx) @ m.Px) @ m.Wx, (((Y - m.my) / m.sy) @ m.Py) @ m.Wy)
    a, b = raw(TFx, TV); sa, sb = (a.mean(0), a.std(0) + 1e-6), (b.mean(0), b.std(0) + 1e-6)
    out = []
    for Fx, Vx in inputs:
        x, y = raw(V.centre(Fx, TFx.mean(0)), V.centre(Vx, TV.mean(0)))
        out.append(zm(V.l2n((x - sa[0]) / sa[1]) @ V.l2n((y - sb[0]) / sb[1]).T))
    return out


t0 = time.time()
CACHE = OUT / "out" / "members.pkl"; CACHE.parent.mkdir(exist_ok=True)
key = [(f["src"], f["lang"], f["i"], len(f["fi"])) for f in FILES]
if CACHE.exists() and pickle.load(open(CACHE, "rb"))["key"] == key:
    MLP, C4, C16, F_MLP, F_C4 = pickle.load(open(CACHE, "rb"))["mats"]
    print("members loaded from cache", flush=True)
else:
    vin, fin = eval_inputs("vgg"), eval_inputs("farl")
    MLP, C4, C16 = mlp_mats(TF, vin), cca4_mats(TF, vin), cca16_mats(TF, vin)
    F_MLP, F_C4 = mlp_mats(FARL_TR, fin), cca4_mats(FARL_TR, fin)
    pickle.dump({"key": key, "mats": (MLP, C4, C16, F_MLP, F_C4)}, open(CACHE, "wb"))
    print(f"members trained in {time.time() - t0:.0f}s", flush=True)
MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)
AGE = []
for f in FILES:
    af = np.load(FM / f"ext_{f['src']}_face_vitageprob.npy").astype(np.float32)[f["fi"]] @ MIDS
    av = np.load(FM / f"ext_{f['src']}_voice_w2v2agage.npy").astype(np.float32)[EXT[f["src"]]["vmap"][f["vi"]]][:, 0] * 100
    AGE.append(-np.abs(af[:, None] - av[None, :]))
IB = []
for f in FILES:
    Fi = np.load(FM / f"ext_{f['src']}_face_ibv.npy").astype(np.float32)[f["fi"]]
    Ai = np.load(FM / f"ext_{f['src']}_voice_iba.npy").astype(np.float32)[EXT[f["src"]]["vmap"][f["vi"]]]
    IB.append(V.l2n(Fi - Fi.mean(0)) @ V.l2n(Ai - Ai.mean(0)).T)
SYS = {}
for j in range(len(FILES)):
    s007 = 0.75 * MLP[j] + 0.25 * C4[j]
    f007 = 0.75 * F_MLP[j] + 0.25 * F_C4[j]
    SYS.setdefault("SET02", []).append(s007)
    SYS.setdefault("SET08", []).append(0.5 * MLP[j] + 0.25 * C4[j] + 0.25 * C16[j])
    SYS.setdefault("SET09", []).append(0.5 * zm(s007) + 0.5 * zm(f007))
    SYS.setdefault("AGE.1", []).append(zm(s007) + 0.1 * zm(AGE[j]))
    SYS.setdefault("AGE.25", []).append(zm(s007) + 0.25 * zm(AGE[j]))
    SYS.setdefault("FARL5+AGE.1", []).append(zm(0.5 * zm(s007) + 0.5 * zm(f007)) + 0.1 * zm(AGE[j]))
    SYS.setdefault("SET12", []).append(0.5 * zm(s007) + 0.5 * zm(IB[j]))
    SYS.setdefault("SET13", []).append(zm(0.5 * zm(s007) + 0.5 * zm(IB[j])) + 0.1 * zm(AGE[j]))
    SYS.setdefault("SET14", []).append(zm((zm(s007) + zm(f007) + zm(IB[j])) / 3) + 0.1 * zm(AGE[j]))


def block(M, fi, vj, ids, b=0.99):
    c = pd.factorize(ids)[0]
    A = np.zeros((c.max() + 1, len(c))); A[c, np.arange(len(c))] = 1; A /= A.sum(1, keepdims=True)
    return (1 - b) * M[fi, vj] + b * (A @ M @ A.T)[c[fi], c[vj]]


rows = []
for name, mats in SYS.items():
    for f, M in zip(FILES, mats):
        for sg, (fi, vj, lab) in f["trials"].items():
            for level, sc in [("sample", M[fi, vj]), ("person", block(M, fi, vj, f["ids"]))]:
                rows.append(dict(system=name, src=f["src"][:2], lang=f["lang"], file=f["i"], protocol="g" if sg else "ng",
                                 level=level, eer=V.eer_from_scores(sc, lab)))
df = pd.DataFrame(rows); df.to_csv(OUT / "results.csv", index=False)
base = df[df.system == "SET02"].set_index(["src", "lang", "file", "protocol", "level"]).eer
out = []
for name in [n for n in SYS if n != "SET02"]:
    d = df[df.system == name].set_index(["src", "lang", "file", "protocol", "level"]).eer - base
    g = d.groupby(level=["lang", "protocol", "level"])
    out.append(pd.DataFrame({"system": name, "delta": g.mean().round(2), "better": g.apply(lambda x: f"{(x < 0).sum()}/{len(x)}")}))
res = pd.concat(out).reset_index()
print(df[df.system == "SET02"].groupby(["lang", "protocol", "level"]).eer.mean().round(2).to_string())
print(res.pivot_table(index=["lang", "protocol", "level"], columns="system", values="delta").to_string())
print(res.pivot_table(index=["lang", "protocol", "level"], columns="system", values="better", aggfunc="first").to_string())
res.to_csv(OUT / "summary.csv", index=False)


# ---------------------------------------------------------------- seed variance (VAL_SEEDVAR=1): is n = 5 too few?
# The dev board moved 0.65-1.32 when only s007's seeds changed (seed_noise.py). Here: s007 with 3 seed sets of n = 5
# (offsets 0 / 1000 / 2000, same seeds as seed_noise.py's replicas use +1000 r) and their average (n = 15), for the
# SET02 and SET13 recipes. Reported per cell: mean EER of the n = 5 sets, mean spread (max - min over the 3 sets, per
# file), and the n = 15 EER.
if os.environ.get("VAL_SEEDVAR") == "1":
    vin = eval_inputs("vgg")
    SETS = {0: MLP}
    for off in (1000, 2000):
        cp = OUT / "out" / f"members_seed{off}.pkl"
        if cp.exists() and pickle.load(open(cp, "rb"))["key"] == key:
            SETS[off] = pickle.load(open(cp, "rb"))["mlp"]
        else:
            t1 = time.time(); SETS[off] = mlp_mats(TF, vin, off)
            pickle.dump({"key": key, "mlp": SETS[off]}, open(cp, "wb"))
            print(f"seed set {off} trained in {time.time() - t1:.0f}s", flush=True)
    variants = {f"n5_seed{o}": M_ for o, M_ in SETS.items()}
    variants["n15"] = [sum(SETS[o][j] for o in SETS) / len(SETS) for j in range(len(FILES))]
    recipes = {"SET02": lambda s7, j: s7, "SET13": lambda s7, j: zm(0.5 * zm(s7) + 0.5 * zm(IB[j])) + 0.1 * zm(AGE[j])}
    rows = []
    for vname, mlp in variants.items():
        for rname, fn in recipes.items():
            for j, f in enumerate(FILES):
                M = fn(0.75 * mlp[j] + 0.25 * C4[j], j)
                for sg, (fi, vj, lab) in f["trials"].items():
                    for level, sc in [("sample", M[fi, vj]), ("person", block(M, fi, vj, f["ids"]))]:
                        rows.append(dict(variant=vname, recipe=rname, lang=f["lang"], src=f["src"][:2], file=f["i"],
                                         protocol="g" if sg else "ng", level=level, eer=V.eer_from_scores(sc, lab)))
    sv = pd.DataFrame(rows); sv.to_csv(OUT / "seed_var.csv", index=False)
    k5 = sv[sv.variant.str.startswith("n5")]
    cell = ["recipe", "lang", "protocol", "level"]
    spread = k5.groupby(cell + ["src", "file"]).eer.agg(lambda x: x.max() - x.min()).groupby(cell).mean()
    summ = pd.DataFrame({"n5_mean": k5.groupby(cell).eer.mean(), "n5_spread": spread,
                         "n15": sv[sv.variant == "n15"].groupby(cell).eer.mean()}).round(2)
    summ["n15_minus_n5"] = (summ.n15 - summ.n5_mean).round(2)
    print(summ.to_string())
    summ.to_csv(OUT / "seed_var_summary.csv")
