"""SET-02 end to end (CodaBench 21.68 on dev) — also the Evaluation-phase pipeline.

    python pipeline.py --data <organiser features root> --feats <our extracted features dir> --out <dir>

Stages (each cached in --out, so a re-run only redoes what is missing):
  1. members  : train the bridge recipes on all 70 train speakers and write, per test file, the FULL face x voice score
                matrix (z-scored cosine of the n-model ensemble, fused with CCA w=0.25)      -> <member>_matrix.npz
  2. clusters : cluster the faces (ArcFace) and the voices (own ECAPA-192) of every test file; thresholds are
                calibrated on the TRAIN speakers only (max ARI)                            -> clusters.npz
  3. scores   : s'(f,v) = (1-b) M[f,v] + b * mean of M over (face cluster x voice cluster), b = 0.99;
                rank fusion of the members of each cell                                    -> submission.zip
Nothing here reads a label of a test file or uses which pairs appear together in the trial list.
Transductive (unlabelled test-file clustering): declared in the system description.

For the Evaluation phase: extract ArcFace + speechbrain ECAPA (192 and 6144) for the new files with
kaggle/FLAG_04_extract.ipynb (same code as for dev), point --data / --feats at them, run this script.
"""
import argparse, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kaggle"))

CONFIG = {
    "recipe": dict(head="mlp", drop=0.3, emb=128, epochs=40),          # EXP-007
    "members": {                                                        # member -> extra cfg
        "s007": dict(),                                                 # VGG + organiser 192
        "s010B": dict(voice="ecapa6144"),                               # VGG + speechbrain ECAPA 6144
        "r2mix": dict(voice="ecapa192", voice_view="mix"),              # VGG + ECAPA 192 + Bangla-length crops
    },
    "n_models": 5, "cca_w": 0.25,
    "cells": {"English": ["s007", "s010B"], "Bangla": ["r2mix", "s007"]},
    "cluster_face": "arcface", "cluster_voice": "ecapa192", "b": 0.99,
}
FILES = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cent(Z, l2n):
    return l2n(l2n(Z) - l2n(Z).mean(0))


def clus(Z, t):
    return AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)


def block(M, cf, cv, b):
    M = M.astype(np.float64)
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * np.diag(M) + b * (Af @ M @ Av.T)[cf, cv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="organiser features (train/ + dev/ or the official zip layout)")
    ap.add_argument("--feats", required=True, help="dir with face_arcface_*.npy, voice_ecapa192_*.npy, voice_ecapa6144_*.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", help="a previously submitted zip to compare rank correlations with")
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    import flag_v2 as V
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=1))
    store = V.FeatureStore(a.data, feats_dir=a.feats)
    V.polarity_preflight()

    # 1. members
    M = {}
    for name, extra in CONFIG["members"].items():
        f = out / f"{name}_matrix.npz"
        if not f.exists():
            t = time.time()
            V.dev_scores(store, dict(CONFIG["recipe"], **extra), n_models=CONFIG["n_models"], cca_w=CONFIG["cca_w"], matrix_path=f)
            log(f"member {name}: {time.time() - t:.0f}s")
        M[name] = np.load(f)

    # 2. clusters (thresholds from train speakers only)
    fc = out / "clusters.npz"
    if not fc.exists():
        spk = store.spk; sub = np.arange(len(spk))[::3]
        thr = {}
        for kind, name, get in [("face", CONFIG["cluster_face"], store.face), ("voice", CONFIG["cluster_voice"], store.voice)]:
            Z = cent(get(name, "train"), V.l2n)
            thr[kind] = float(max(np.arange(.4, 1.21, .05), key=lambda t: adjusted_rand_score(spk[sub], clus(Z[sub], t))))
        C = {"thr_face": thr["face"], "thr_voice": thr["voice"]}
        for k in FILES:
            split = "/".join(k)
            C[f"{k[0]}_{k[1]}_face"] = clus(cent(store.face(CONFIG["cluster_face"], split), V.l2n), thr["face"])
            C[f"{k[0]}_{k[1]}_voice"] = clus(cent(store.voice(CONFIG["cluster_voice"], split), V.l2n), thr["voice"])
        np.savez(fc, **C)
        log(f"cluster thresholds (train-calibrated): face {thr['face']:.2f} voice {thr['voice']:.2f}")
    C = np.load(fc)

    # 3. scores
    rank = lambda s: rankdata(s) / (len(s) + 1)
    scores = {}
    for k in FILES:
        c = "/".join(k); tk = f"{k[0]}_{k[1]}"
        cf, cv = C[tk + "_face"], C[tk + "_voice"]
        scores[k] = np.mean([rank(block(M[m][c], cf, cv, CONFIG["b"])) for m in CONFIG["cells"][k[1]]], 0)
        log(f"{c:18s} faces {len(cf)} -> {cf.max() + 1} clusters, voices -> {cv.max() + 1} clusters")
    zp = out / "submission.zip"
    V.write_zip(store, zp, scores); V.check_zip(store, zp)
    if a.reference:
        for fn in FILES.values():
            x = pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None)[1]
            y = pd.read_csv(zipfile.ZipFile(a.reference).open(fn), sep=" ", header=None)[1]
            log(f"{fn:45s} rank-corr with reference: {x.corr(y, method='spearman'):.4f}")


if __name__ == "__main__":
    main()
