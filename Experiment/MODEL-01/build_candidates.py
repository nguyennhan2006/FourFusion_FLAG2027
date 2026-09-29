"""Candidate submissions = SET-02 with its s007 member replaced, at the matrix level, by a pre-registered recipe.
Registered 28/09 (log 63) before any result of the combinations was seen; each part passed MODEL-01 on its own:

    SET10  z(s007) + 0.1 z(AGE)                                   S track
    SET11  z(0.5 z(s007) + 0.5 z(f007)) + 0.1 z(AGE)              S track
    SET12  0.5 z(s007) + 0.5 z(IB)                                ImageBind (allowed, log 63)
    SET13  z(0.5 z(s007) + 0.5 z(IB)) + 0.1 z(AGE)
    SET14  z((z(s007) + z(f007) + z(IB)) / 3) + 0.1 z(AGE)
where f007 = s007 recipe on FaRL faces (build_set09.py), IB = file-centred cosine of ImageBind face / voice embeddings,
AGE = -|age(face) - age(voice)| (vitageprob bins / audeering w2v2 age). Everything else of SET-02 is unchanged
(s010B, r2mix, ArcFace / ECAPA-192 clusters, b = 0.99, rank fusion per cell). Full matrices only: the trial list is
never read.
    python build_candidates.py [SET10 SET12 ...]      (default: all)
"""
import os, sys, zipfile
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); sys.path.insert(0, str(R / "BEST/v4_set02"))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_models")
import flag_v2 as V
import pipeline as P

DRY, OUT, FM = R / "BEST/v4_set02/_dryrun", Path(__file__).parent / "out", R / "kaggle/output/feats_models"
REF = R / "Experiment/SET-01/out/submission_SET02.zip"
MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
rank = lambda s: rankdata(s) / (len(s) + 1)
arr = lambda name, split: np.load(FM / f"{name}_{split.replace('/', '_')}.npy").astype(np.float32)

RECIPES = {
    "SET10": lambda s, f, ib, ag: zm(s) + 0.1 * zm(ag),
    "SET11": lambda s, f, ib, ag: zm(0.5 * zm(s) + 0.5 * zm(f)) + 0.1 * zm(ag),
    "SET12": lambda s, f, ib, ag: 0.5 * zm(s) + 0.5 * zm(ib),
    "SET13": lambda s, f, ib, ag: zm(0.5 * zm(s) + 0.5 * zm(ib)) + 0.1 * zm(ag),
    "SET14": lambda s, f, ib, ag: zm((zm(s) + zm(f) + zm(ib)) / 3) + 0.1 * zm(ag),
}

store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
V.polarity_preflight()
M = {m: np.load(DRY / f"{m}_matrix.npz") for m in P.CONFIG["members"]}
F7 = np.load(OUT / "f007_matrix.npz")                                  # from build_set09.py
C = np.load(DRY / "clusters.npz")
EXTRA = {}
for k in P.FILES:
    c = "/".join(k)
    Fi, Ai = arr("face_ibv", c), arr("voice_iba", c)
    ib = V.l2n(Fi - Fi.mean(0)) @ V.l2n(Ai - Ai.mean(0)).T
    ag = -np.abs((arr("face_vitageprob", c) @ MIDS)[:, None] - (arr("voice_w2v2agage", c)[:, 0] * 100)[None, :])
    assert ib.shape == M["s007"][c].shape == ag.shape, (c, ib.shape, M["s007"][c].shape)
    EXTRA[c] = (F7[c].astype(np.float64), ib.astype(np.float64), ag.astype(np.float64))

for name in (sys.argv[1:] or list(RECIPES)):
    scores = {}
    for k in P.FILES:
        c = "/".join(k); tk = f"{k[0]}_{k[1]}"
        mats = {m: M[m][c].astype(np.float64) for m in P.CONFIG["members"]}
        mats["s007"] = RECIPES[name](mats["s007"], *EXTRA[c])
        scores[k] = np.mean([rank(P.block(mats[m], C[tk + "_face"], C[tk + "_voice"], P.CONFIG["b"]))
                             for m in P.CONFIG["cells"][k[1]]], 0)
    zp = OUT / f"submission_{name}.zip"
    V.write_zip(store, zp, scores); V.check_zip(store, zp)
    rc = [pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None)[1].corr(
          pd.read_csv(zipfile.ZipFile(REF).open(fn), sep=" ", header=None)[1], method="spearman") for fn in P.FILES.values()]
    print(f"{name}: rank-corr with SET-02 per cell " + " / ".join(f"{x:.3f}" for x in rc), flush=True)
