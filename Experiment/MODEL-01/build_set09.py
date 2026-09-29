"""SET-09 = SET-02 with its s007 member replaced by 0.5 z(s007) + 0.5 z(f007), f007 = the same recipe on FaRL faces.

This is exactly the MODEL-01 arm "CTRL+farl.5", which passed the pre-registered sample-level gate (results.csv,
summary.py). Nothing else of SET-02 changes: members s010B / r2mix, clusters (ArcFace / ECAPA-192, train-calibrated
thresholds), b = 0.99, rank fusion per cell. s007 appears in both cells, so both cells change.
Level 0-2 only: the fusion is on the full face x voice matrices, the trial list is never read.

    python build_set09.py            both languages                       -> out/submission_SET09.zip
    python build_set09.py English    English cells only (s007 stays in Bangla) -> out/submission_SET09E.zip
(reuses BEST/v4_set02/_dryrun: SET-02 members + clusters, rank-corr 1.0000 with SET-02)
SET-09E was registered after the gate, from the VALIDATION numbers alone: FaRL's gain at the aggregated level holds in
English (v4 cluster g -1.37) but not for the non-English persons (Urdu person g +0.43 [3/5]).
"""
import os, sys, time, zipfile
import numpy as np, pandas as pd
from scipy.stats import rankdata
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); sys.path.insert(0, str(R / "BEST/v4_set02"))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_models")         # face_farl_<split>.npy
import torch; torch.set_num_threads(10)
import flag_v2 as V
import pipeline as P

DRY, OUT = R / "BEST/v4_set02/_dryrun", Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
REF = R / "Experiment/SET-01/out/submission_SET02.zip"
W = 0.5                                                                           # pre-registered arm weight
LANGS = sys.argv[1:] or ["English", "Bangla"]
TAG = "SET09" if len(LANGS) == 2 else "SET09" + "".join(l[0] for l in LANGS)
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)

store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
V.polarity_preflight()
f = OUT / "f007_matrix.npz"
if not f.exists():
    t = time.time()
    V.dev_scores(store, dict(P.CONFIG["recipe"], face="farl"), n_models=P.CONFIG["n_models"], cca_w=P.CONFIG["cca_w"], matrix_path=f)
    print(f"member f007 (FaRL + organiser 192): {time.time() - t:.0f}s", flush=True)
M = {m: np.load(DRY / f"{m}_matrix.npz") for m in P.CONFIG["members"]}
F7 = np.load(f)
C = np.load(DRY / "clusters.npz")
rank = lambda s: rankdata(s) / (len(s) + 1)

scores = {}
for k in P.FILES:
    c = "/".join(k); tk = f"{k[0]}_{k[1]}"
    cf, cv = C[tk + "_face"], C[tk + "_voice"]
    mats = {m: M[m][c] for m in P.CONFIG["members"]}
    if k[1] in LANGS:
        mats["s007"] = (1 - W) * zm(mats["s007"].astype(np.float64)) + W * zm(F7[c].astype(np.float64))
    scores[k] = np.mean([rank(P.block(mats[m], cf, cv, P.CONFIG["b"])) for m in P.CONFIG["cells"][k[1]]], 0)
zp = OUT / f"submission_{TAG}.zip"
V.write_zip(store, zp, scores); V.check_zip(store, zp)
for fn in P.FILES.values():
    x = pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None)[1]
    y = pd.read_csv(zipfile.ZipFile(REF).open(fn), sep=" ", header=None)[1]
    print(f"{fn:45s} rank-corr with SET-02: {x.corr(y, method='spearman'):.4f}", flush=True)
print("wrote", zp)
