"""SET-13 end to end (CodaBench 21.01 on dev) — the proposed Evaluation-phase pipeline (log 66-67).

    python pipeline.py --data <organiser features root> --feats <FLAG_04 output> --models <FLAG_10 output> --out <dir>

= SET-02 (../v4_set02/pipeline.py: members s007 / s010B / r2mix trained on the 70 train speakers, ArcFace / ECAPA-192
clusters per test file, block mean b = 0.99, rank fusion per cell), with ONE change: in every test file the s007
member's full face x voice matrix S is replaced by
        S' = z(0.5 z(S) + 0.5 z(IB)) + 0.1 z(AGE)
  IB  = cosine of ImageBind-huge face / voice embeddings, each centred on the file's own mean   (no training)
  AGE = -|age(face) - age(voice)|: face age = expectation over the 9 FairFace bins of nateraw/vit-age-classifier,
        voice age = audeering wav2vec2-large-robust age head x 100                            (no training)
Nothing reads a label of a test file or which pairs appear together in the trial list; every score comes from the
full matrix. Validation (true labels): v4 held-out cluster level -6.8 / -6.5, VAL-70 person level -7.4 (log 66).

For the Evaluation phase: run kaggle/FLAG_04_extract (ArcFace, ECAPA 192 / 6144) AND kaggle/FLAG_10_models
(ImageBind, ViT-age, w2v2 age-gender) on the new files, then this script with --feats / --models pointing at them.
"""
import argparse, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kaggle")); sys.path.insert(0, str(HERE.parent / "v4_set02"))
import pipeline as BASE                                   # SET-02: CONFIG, FILES, block, clus, cent, log

CONFIG = dict(BASE.CONFIG, s007_recipe=dict(w_ib=0.5, w_age=0.1, ib="face_ibv / voice_iba", age="vitageprob / w2v2agage"))
AGE_MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)       # centres of the 9 FairFace age bins
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)


def extras(models: Path, split, l2n):
    """IB and AGE full matrices of one test file, rows / cols in organiser order (FLAG_10 arrays)."""
    a = lambda name: np.load(models / f"{name}_{split.replace('/', '_')}.npy").astype(np.float32)
    F, A = a("face_ibv"), a("voice_iba")
    ib = l2n(F - F.mean(0)) @ l2n(A - A.mean(0)).T
    age = -np.abs((a("face_vitageprob") @ AGE_MIDS)[:, None] - (a("voice_w2v2agage")[:, 0] * 100)[None, :])
    return ib.astype(np.float64), age.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="organiser features (train/ + test files)")
    ap.add_argument("--feats", required=True, help="FLAG_04 output: face_arcface_*, voice_ecapa192_*, voice_ecapa6144_*")
    ap.add_argument("--models", required=True, help="FLAG_10 output: face_ibv_*, voice_iba_*, face_vitageprob_*, voice_w2v2agage_*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", help="a previously submitted zip to compare rank correlations with")
    ap.add_argument("--members-from", help="reuse cached member matrices + clusters from this dir (e.g. ../v4_set02/_dryrun)")
    ap.add_argument("--b", type=float, default=CONFIG["b"],
                    help="weight of the cluster-block mean; 0 = score every pair on its own (fallback if clustering is not usable)")
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    import flag_v2 as V
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(CONFIG, b=a.b), indent=1, default=str))
    store = V.FeatureStore(a.data, feats_dir=a.feats)
    V.polarity_preflight()
    cache = Path(a.members_from) if a.members_from else out

    # 1. members (identical to SET-02)
    M = {}
    for name, extra in CONFIG["members"].items():
        f = cache / f"{name}_matrix.npz"
        if not f.exists():
            t = time.time()
            V.dev_scores(store, dict(CONFIG["recipe"], **extra), n_models=CONFIG["n_models"], cca_w=CONFIG["cca_w"], matrix_path=f)
            BASE.log(f"member {name}: {time.time() - t:.0f}s")
        M[name] = np.load(f)

    # 2. clusters (identical to SET-02; thresholds from train speakers only)
    fc = cache / "clusters.npz"
    if not fc.exists():
        spk = store.spk; sub = np.arange(len(spk))[::3]
        thr = {}
        for kind, name, get in [("face", CONFIG["cluster_face"], store.face), ("voice", CONFIG["cluster_voice"], store.voice)]:
            Z = BASE.cent(get(name, "train"), V.l2n)
            thr[kind] = float(max(np.arange(.4, 1.21, .05), key=lambda t: adjusted_rand_score(spk[sub], BASE.clus(Z[sub], t))))
        C = {"thr_face": thr["face"], "thr_voice": thr["voice"]}
        for k in BASE.FILES:
            split = "/".join(k)
            C[f"{k[0]}_{k[1]}_face"] = BASE.clus(BASE.cent(store.face(CONFIG["cluster_face"], split), V.l2n), thr["face"])
            C[f"{k[0]}_{k[1]}_voice"] = BASE.clus(BASE.cent(store.voice(CONFIG["cluster_voice"], split), V.l2n), thr["voice"])
        np.savez(fc, **C)
        BASE.log(f"cluster thresholds (train-calibrated): face {thr['face']:.2f} voice {thr['voice']:.2f}")
    C = np.load(fc)

    # 3. scores: s007 -> z(0.5 z(s007) + 0.5 z(IB)) + 0.1 z(AGE), then SET-02's block mean + rank fusion
    rank = lambda s: rankdata(s) / (len(s) + 1)
    R = CONFIG["s007_recipe"]
    scores = {}
    for k in BASE.FILES:
        c = "/".join(k); tk = f"{k[0]}_{k[1]}"
        cf, cv = C[tk + "_face"], C[tk + "_voice"]
        mats = {m: M[m][c].astype(np.float64) for m in CONFIG["members"]}
        ib, age = extras(Path(a.models), c, V.l2n)
        assert ib.shape == mats["s007"].shape, (c, ib.shape, mats["s007"].shape)
        mats["s007"] = zm((1 - R["w_ib"]) * zm(mats["s007"]) + R["w_ib"] * zm(ib)) + R["w_age"] * zm(age)
        scores[k] = np.mean([rank(BASE.block(mats[m], cf, cv, a.b)) for m in CONFIG["cells"][k[1]]], 0)
        BASE.log(f"{c:18s} faces {len(cf)} -> {cf.max() + 1} clusters, voices -> {cv.max() + 1} clusters")
    zp = out / "submission.zip"
    V.write_zip(store, zp, scores); V.check_zip(store, zp)
    if a.reference:
        for fn in BASE.FILES.values():
            x = pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None)[1]
            y = pd.read_csv(zipfile.ZipFile(a.reference).open(fn), sep=" ", header=None)[1]
            BASE.log(f"{fn:45s} rank-corr with reference: {x.corr(y, method='spearman'):.4f}")


if __name__ == "__main__":
    main()
