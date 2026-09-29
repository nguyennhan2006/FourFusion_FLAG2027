"""How much does the dev score move when NOTHING changes but the random seeds of the s007 member?

SET-08 / SET-09 changed only s007 and moved the board by +0.62 / +0.81, while every independent validation (v4
held-out, VAL-70) said they were better. If re-seeding s007 alone moves dev by as much, those board deltas are noise.
Each replica = SET-02 with s007 retrained on all 70 speakers with other seeds (same recipe, n = 5), everything else
identical (BEST/v4_set02/_dryrun members + clusters). Dev scored with pseudo-arc, used here only to measure the noise
of the dev metric (it tracked the board within ~1 EER per cell on SET-02 / 07 / 08 / 09), not to choose a system."""
import os, sys, subprocess, zipfile
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); sys.path.insert(0, str(R / "BEST/v4_set02"))
import torch; torch.set_num_threads(10)
import flag_v2 as V
import pipeline as P

DRY, OUT = R / "BEST/v4_set02/_dryrun", Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
REPLICAS = [int(x) for x in os.environ.get("REPLICAS", "1,2,3").split(",")]
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
M = {m: np.load(DRY / f"{m}_matrix.npz") for m in P.CONFIG["members"]}
C = np.load(DRY / "clusters.npz")
rank = lambda s: rankdata(s) / (len(s) + 1)
for r in REPLICAS:
    f = OUT / f"s007_seed{r}_matrix.npz"
    if not f.exists():
        V.dev_scores(store, dict(P.CONFIG["recipe"]), n_models=P.CONFIG["n_models"], cca_w=P.CONFIG["cca_w"],
                     seed0=1 + 1000 * r, matrix_path=f)
    S7 = np.load(f)
    scores = {}
    for k in P.FILES:
        c = "/".join(k); tk = f"{k[0]}_{k[1]}"
        mats = {m: M[m][c] for m in P.CONFIG["members"]}
        mats["s007"] = S7[c]
        scores[k] = np.mean([rank(P.block(mats[m], C[tk + "_face"], C[tk + "_voice"], P.CONFIG["b"]))
                             for m in P.CONFIG["cells"][k[1]]], 0)
    zp = OUT / f"submission_SET02_seed{r}.zip"
    V.write_zip(store, zp, scores); V.check_zip(store, zp)
    print(f"replica {r} written", flush=True)
