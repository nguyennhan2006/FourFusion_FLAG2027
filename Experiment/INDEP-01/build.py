"""INDEP-01 — the best we can do WITHOUT using the test set beyond the pair being scored (level 0, PLAN_MODELS §0b).

Same components as SET-13 (CB 21.01): members s007 / s010B / r2mix trained on the 70 train speakers (n = 5, seeds as
dev_scores), ImageBind, age match, and the same weights. Everything that looked at other test samples is replaced:
  * no per-file mean removal: features are standardised with TRAIN statistics only (flag_v2.embed is called with the
    file's own mean as "target", which makes its centring a no-op; CCA / ImageBind use train means);
  * no z-scoring or ranking over a test file: every score is standardised with constants measured on 20 000 random
    TRAIN face-voice pairs, so a pair's score is a fixed function of that face and that voice;
  * no clustering (no block mean).
Reference points: level 1 (file centring + z / rank, no clustering) = BEST/v5_set13 --b 0; level 2 = SET-13.
    python build.py        -> out/submission_INDEP01.zip
"""
import os, sys, time
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); sys.path.insert(0, str(R / "BEST/v4_set02"))
import torch; torch.set_num_threads(int(os.environ.get("INDEP_THREADS", "10")))
import flag_v2 as V
from flag_lib import RidgeCCA
import pipeline as P

OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
FM = R / "kaggle/output/feats_models"
MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
V.polarity_preflight()
rng = np.random.RandomState(0)
n = len(store.spk)
REF = (rng.randint(n, size=20000), rng.randint(n, size=20000))       # random train (face row, voice row) pairs
arr = lambda name, split: np.load(FM / f"{name}_{split.replace('/', '_')}.npy").astype(np.float32)
pair = lambda U, W: np.sum(U * W, 1)


class Std:
    """z-score with constants from the train reference pairs only."""
    def __init__(self, ref):
        self.m, self.s = float(np.mean(ref)), float(np.std(ref) + 1e-9)

    def __call__(self, x):
        return (x - self.m) / self.s


def member(extra):
    """{cell: level-0 score per trial} and the same score on the train reference pairs."""
    cfg = dict(V.BASE, **P.CONFIG["recipe"], **extra)
    Xf, Xv = store.face(cfg["face"], "train"), store.voice(cfg["voice"], "train")
    t = time.time()
    nets = [V.train_one(store, cfg, np.arange(n), seed=1 + 100 * i) for i in range(P.CONFIG["n_models"])]
    print(f"  {extra or 's007'}: {len(nets)} models in {time.time() - t:.0f}s", flush=True)
    emb = lambda net, prep, F_, V_: V.embed(net, prep, F_, V_, prep.f(F_).mean(0), prep.v(V_).mean(0))   # no centring
    cca = RidgeCCA(**V.CCA_CFG).fit(Xf, Xv)
    parts_ref, stds = [], []
    for net, prep in nets:
        u, w = emb(net, prep, Xf, Xv)
        r = pair(u[REF[0]], w[REF[1]]); stds.append(Std(r)); parts_ref.append(stds[-1](r))
    cref = pair(cca.transform_x(Xf)[REF[0]], cca.transform_y(Xv)[REF[1]]); cstd = Std(cref)
    ref = 0.75 * np.mean(parts_ref, 0) + (1 - 0.75) * cstd(cref)
    out = {}
    for k in P.FILES:
        c = "/".join(k)
        Af, Av = store.face(cfg["face"], c), store.voice(cfg["voice"], c)
        deep = np.mean([sd(pair(*emb(net, prep, Af, Av))) for (net, prep), sd in zip(nets, stds)], 0)
        out[c] = 0.75 * deep + 0.25 * cstd(pair(cca.transform_x(Af), cca.transform_y(Av)))
    return out, ref


M, REFS = {}, {}
for name, extra in P.CONFIG["members"].items():
    M[name], REFS[name] = member(extra)

# ImageBind (train-mean centring) and age, on the same reference pairs
Fi_tr, Ai_tr = arr("face_ibv", "train"), arr("voice_iba", "train")
mf, mv = Fi_tr.mean(0), Ai_tr.mean(0)
ib = lambda F_, A_: pair(V.l2n(F_ - mf), V.l2n(A_ - mv))
age_f = lambda split: arr("face_vitageprob", split) @ MIDS
age_v = lambda split: arr("voice_w2v2agage", split)[:, 0] * 100
ib_ref = ib(Fi_tr[REF[0]], Ai_tr[REF[1]]); age_ref = -np.abs(age_f("train")[REF[0]] - age_v("train")[REF[1]])
z7, zib, zag = Std(REFS["s007"]), Std(ib_ref), Std(age_ref)
mix_ref = 0.5 * z7(REFS["s007"]) + 0.5 * zib(ib_ref); zmix = Std(mix_ref)
s7_ref = zmix(mix_ref) + 0.1 * zag(age_ref); z7b = Std(s7_ref)
zo = {m: Std(REFS[m]) for m in P.CONFIG["members"]}

scores = {}
for k in P.FILES:
    c = "/".join(k)
    s7 = zmix(0.5 * z7(M["s007"][c]) + 0.5 * zib(ib(arr("face_ibv", c), arr("voice_iba", c)))) \
        + 0.1 * zag(-np.abs(age_f(c) - age_v(c)))
    comp = {m: zo[m](M[m][c]) for m in P.CONFIG["members"]}
    comp["s007"] = z7b(s7)
    scores[k] = np.mean([comp[m] for m in P.CONFIG["cells"][k[1]]], 0)    # fixed-constant average (not a rank)
zp = OUT / "submission_INDEP01.zip"
V.write_zip(store, zp, scores); V.check_zip(store, zp)
print("wrote", zp, flush=True)
