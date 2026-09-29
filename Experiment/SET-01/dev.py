"""SET-02 — set aggregation on dev with the local full score matrices (GRAPH-01/mats, n=5 per recipe).
Clusters: faces ArcFace-512, voices own ECAPA-192, per dev file, thresholds calibrated on ALL train speakers (no dev
labels). Aggregated score s'(f,v) = (1-b) M[f,v] + b * mean of M over (face-cluster(f) x voice-cluster(v)).
Judged with both pseudo-label sets -- CAUTION: arc labels are built from the SAME ArcFace/ECAPA clusters (circular,
optimistic); btc labels use VGG + organiser-voice clusters. The unbiased evidence is SET-01 (true labels)."""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "Experiment/EDA-000_raw")); from eda_utils import load_train, load_dev, eer_from_scores, l2n
FE = R / "kaggle/output/feats_v2"
LAB = {l: np.load(R / f"Experiment/SEL-01_selector/pseudo_labels_{l}.npz") for l in ["arc", "btc"]}
M = {n: np.load(R / f"Experiment/GRAPH-01/mats/{n}_matrix.npz") for n in ["s007", "s010B", "r2mix"]}
rank = lambda s: rankdata(s) / (len(s) + 1)
cent = lambda Z: l2n(l2n(Z) - l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
_, _, spk, _, _, _ = load_train()
sub = np.arange(len(spk))[::3]
calib = lambda Z: max(np.arange(.4, 1.21, .05), key=lambda t: adjusted_rand_score(spk[sub], clus(Z[sub], t)))
tF, tV = calib(cent(np.load(FE / "face_arcface_train.npy"))), calib(cent(np.load(FE / "voice_ecapa192_train.npy")))
print(f"thresholds from train: face {tF:.2f} voice {tV:.2f}")
FN_ = {"no_gender/English": "no_gender/sub_score_v4_English_heard.txt", "gender/English": "gender/sub_score_v4_English_heard.txt",
       "no_gender/Bangla": "no_gender/sub_score_v4_Bangla_unheard.txt", "gender/Bangla": "gender/sub_score_v4_Bangla_unheard.txt"}


def block(Mx, cf, cv, b):
    Mx = Mx.astype(np.float64); nf, nv = cf.max() + 1, cv.max() + 1
    Af = np.zeros((nf, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((nv, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    B = Af @ Mx @ Av.T                                   # cluster x cluster block means
    return (1 - b) * np.diag(Mx) + b * B[cf, cv]


PLANS = {"English": {"cur": ["s007", "s010B"]}, "Bangla": {"cur": ["r2mix", "s007"]}}
rows, keep = [], {}
for c, fn in FN_.items():
    tk = c.replace("/", "_"); lang = c.split("/")[1]
    cf, cv = clus(cent(np.load(FE / f"face_arcface_{tk}.npy")), tF), clus(cent(np.load(FE / f"voice_ecapa192_{tk}.npy")), tV)
    sub03 = -pd.read_csv(zipfile.ZipFile(R / "Experiment/FUSE-03/out/submission_FUSE03.zip").open(fn), sep=" ", header=None)[1].values
    mem = PLANS[lang]["cur"]
    cands = {"FUSE-03 (submitted)": sub03}
    for b in (0.5, 0.75, 0.99, 1.0):
        cands[f"agg b={b}"] = np.mean([rank(block(M[m][c], cf, cv, b)) for m in mem], 0)
    for name, s in cands.items():
        r = dict(cell=c, cand=name, n_face_cl=cf.max() + 1, n_voice_cl=cv.max() + 1)
        for lb, L in LAB.items():
            kn, y = L[tk + "_known"], L[tk + "_lab"]; r[lb] = round(eer_from_scores(s[kn], y[kn]), 2)
        rows.append(r); keep[(c, name)] = s
T = pd.DataFrame(rows); T.to_csv("set02_dev.csv", index=False); print(T.to_string(index=False))
np.savez("set02_scores.npz", **{f"{c.replace('/', '_')}|{n}": v for (c, n), v in keep.items()})
