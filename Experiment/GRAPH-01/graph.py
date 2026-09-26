"""GRAPH-01 — unimodal-graph smoothing of the FULL face x voice score matrix (no use of the trial list).

S' = (1-lam) S + lam/2 (A_f S + S A_v), trial score = diag(S'). A_f / A_v: row-normalised kNN graphs over the
file's faces / voices (self excluded, weights = cosine > 0).
Circularity guard: graph features and pseudo-label features are always CROSSED:
   graph arc (ArcFace + own ECAPA-192)  -> judged by btc labels (VGG + organiser 192)
   graph btc (VGG-PCA + organiser 192)  -> judged by arc labels
Pre-registered grid: k in {3,5,10}, lam in {0.1,0.2,0.3}.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "EDA-000_raw"))
from eda_utils import load_dev, load_train, eer_from_scores, l2n
FE = E.parent / "kaggle/output/feats_v2"
CELLS = [("no_gender", "English"), ("no_gender", "Bangla"), ("gender", "English"), ("gender", "Bangla")]
Xf, Xv, *_ = load_train()
mf, sf = Xf.mean(0), Xf.std(0) + 1e-6
P = np.linalg.svd((Xf - mf) / sf, full_matrices=False)[2][:256].T
cent = lambda Z: l2n(l2n(Z) - l2n(Z).mean(0))


def feats(kind, k):
    a, b, _ = load_dev(*k)
    if kind == "btc":
        return cent(((a - mf) / sf) @ P), cent(b)
    tk = f"{k[0]}_{k[1]}"
    return cent(np.load(FE / f"face_arcface_{tk}.npy")), cent(np.load(FE / f"voice_ecapa192_{tk}.npy"))


def knn(Z, k):
    S = Z @ Z.T; np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    A = np.zeros_like(S); r = np.arange(len(Z))[:, None]
    A[r, idx] = np.maximum(S[r, idx], 0)
    return A / (A.sum(1, keepdims=True) + 1e-9)


def smooth_diag(M, Af, Av, lam):
    # only the diagonal is needed: diag(Af M) = sum_j Af[i,j] M[j,i]; diag(M Av) = sum_j M[i,j] Av[j,i]
    return (1 - lam) * np.diag(M) + lam / 2 * ((Af * M.T).sum(1) + (M * Av.T).sum(1))


LAB = {l: np.load(E / f"SEL-01_selector/pseudo_labels_{l}.npz") for l in ["arc", "btc"]}
CROSS = {"arc": "btc", "btc": "arc"}          # graph kind -> label set used to judge it


def run(matrix_file, name):
    Mz = np.load(matrix_file); rows = []
    for k in CELLS:
        M = Mz["/".join(k)].astype(np.float32)
        for gk, lab in CROSS.items():
            F, Vv = feats(gk, k); L = LAB[lab]; tk = f"{k[0]}_{k[1]}"
            kn, y = L[tk + "_known"], L[tk + "_lab"]
            base = eer_from_scores(np.diag(M)[kn], y[kn])
            for kk in (3, 5, 10):
                Af, Av = knn(F, kk), knn(Vv, kk)
                for lam in (0.1, 0.2, 0.3):
                    s = smooth_diag(M, Af, Av, lam)
                    rows.append(dict(sys=name, cell="/".join(k), graph=gk, judged=lab, k=kk, lam=lam,
                                     base=round(base, 2), eer=round(eer_from_scores(s[kn], y[kn]), 2)))
    R = pd.DataFrame(rows); R["gain"] = (R.base - R.eer).round(2)
    return R


if __name__ == "__main__":
    out = []
    for f in sys.argv[1:]:
        R = run(f, Path(f).stem.replace("_matrix", "")); out.append(R)
        print(R.pivot_table(index=["cell", "graph"], columns=["k", "lam"], values="gain").round(2).to_string())
        print(R.groupby(["cell", "graph"]).base.first().to_string())
    pd.concat(out).to_csv("graph01_" + "_".join(Path(f).stem for f in sys.argv[1:])[:80] + ".csv", index=False)
