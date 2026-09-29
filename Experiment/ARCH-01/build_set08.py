"""SET-08 = SET-02 with the s007 member re-fused as ARCH-01b F3: 0.5 MLP + 0.25 CCA-4 + 0.25 CCA-16
(was 0.75 MLP + 0.25 CCA-4).  Only s007 changes (the configuration ARCH-01b validated: VGG + organiser 192);
s010B, r2mix, clusters, b = 0.99 and the per-cell rank fusion are SET-02's.
The MLP part is recovered exactly from the saved production matrix: M = (S - 0.25 zC4) / 0.75, where zC4 is the
(deterministic) production CCA-4 matrix of the same dev file.  CCA-16 matrices come from dev_mats.py."""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); sys.path.insert(0, str(R / "Experiment/SEL-01_selector"))
import flag_v2 as V
from pseudo_eval import evaluate, compare
HERE = Path(__file__).parent
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
G = R / "Experiment/GRAPH-01/mats"
S007, C16 = np.load(G / "s007_matrix.npz"), np.load(HERE / "out/cca16_matrix.npz")
f3 = {}
for k in V.CELLS:
    c = "/".join(k)
    a, b = V.cca_proj(store.Xf, store.Xv, store.face("vgg", c), store.voice("given", c))
    z4 = zm(a @ b.T)
    M = (S007[c].astype(np.float64) - 0.25 * z4) / 0.75
    f3[c] = (0.5 * M + 0.25 * z4 + 0.25 * C16[c].astype(np.float64)).astype(np.float16)
np.savez_compressed(HERE / "out/s007F3_matrix.npz", **f3)

C = np.load(R / "BEST/v4_set02/_dryrun/clusters.npz")
MM = {"s007F3": f3, "s007": S007, "s010B": np.load(G / "s010B_matrix.npz"), "r2mix": np.load(G / "r2mix_matrix.npz")}
rank = lambda s: rankdata(s) / (len(s) + 1)


def block(Mx, cf, cv, b=0.99):
    Mx = Mx.astype(np.float64)
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * np.diag(Mx) + b * (Af @ Mx @ Av.T)[cf, cv]


CELLS = {"English": ["s007F3", "s010B"], "Bangla": ["r2mix", "s007F3"]}
scores = {}
for k in V.CELLS:
    c = "/".join(k); tk = f"{k[0]}_{k[1]}"
    scores[k] = np.mean([rank(block(MM[m][c], C[tk + "_face"], C[tk + "_voice"])) for m in CELLS[k[1]]], 0)
zp = HERE / "out/submission_SET08.zip"
V.write_zip(store, zp, scores); V.check_zip(store, zp)
ref = R / "Experiment/SET-01/out/submission_SET02.zip"
for k, fn in V.NAMES.items():
    x = pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None)[1]; y = pd.read_csv(zipfile.ZipFile(ref).open(fn), sep=" ", header=None)[1]
    print(f"{fn:45s} rank-corr with SET-02 {x.corr(y, method='spearman'):.3f}")
print("\nmember, sample level (pseudo-arc): s007F3 - s007")
print(compare({k: np.diag(f3['/'.join(k)].astype(float)) for k in V.CELLS},
              {k: np.diag(S007['/'.join(k)].astype(float)) for k in V.CELLS}, n_boot=200).to_string(index=False))
print("\nSET-08 (pseudo-arc)"); print(evaluate(zp, n_boot=100).to_string(index=False))
print("\nSET-08 - SET-02"); print(compare(zp, ref, n_boot=200).to_string(index=False))
