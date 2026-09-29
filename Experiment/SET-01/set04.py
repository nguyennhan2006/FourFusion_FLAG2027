"""SET-04 — SET-02 with the n=10 members (as in the original submissions) and the voice-cluster threshold 0.8.
Every choice comes from TRUE-label validation (SET-01 for b, SET-03 for the thresholds); dev pseudo-labels are
printed only as a sanity check (they are biased towards clustered systems, see EXPERIMENT_LOG 44).
    English cells : rank(agg s007) + rank(agg s010B)
    Bangla cells  : rank(agg r2mix) + rank(agg s007)
"""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "Experiment/EDA-000_raw")); from eda_utils import load_dev, eer_from_scores, l2n
FE = R / "kaggle/output/feats_v2"
TF, B = 0.75, 0.99                      # face threshold (train-calibrated), weight (SET-01)
# voice threshold per language: 0.8 validated on ENGLISH with true labels (SET-03); for Bangla 0.8 merged ~30 people
# into 24-26 clusters and both pseudo-label sets got 4-6 EER worse on g/Bn -> keep the CB-confirmed 0.7 there.
TV = {"English": 0.80, "Bangla": 0.70}
M = {n: np.load(Path(__file__).parent / f"mats10/{n}_matrix.npz") for n in ["s007", "s010B", "r2mix"]}
LAB = {l: np.load(R / f"Experiment/SEL-01_selector/pseudo_labels_{l}.npz") for l in ["arc", "btc"]}
rank = lambda s: rankdata(s) / (len(s) + 1)
cent = lambda Z: l2n(l2n(Z) - l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
FN_ = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt", ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
       ("gender", "English"): "gender/sub_score_v4_English_heard.txt", ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
MEM = {"English": ["s007", "s010B"], "Bangla": ["r2mix", "s007"]}


def block(Mx, cf, cv, b):
    Mx = Mx.astype(np.float64)
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * np.diag(Mx) + b * (Af @ Mx @ Av.T)[cf, cv]


out = Path(__file__).parent / "out/submission_SET04.zip"
with zipfile.ZipFile(out, "w") as zo:
    for k, fn in FN_.items():
        c, tk = "/".join(k), f"{k[0]}_{k[1]}"
        cf = clus(cent(np.load(FE / f"face_arcface_{tk}.npy")), TF)
        cv = clus(cent(np.load(FE / f"voice_ecapa192_{tk}.npy")), TV[k[1]])
        s = np.mean([rank(block(M[m][c], cf, cv, B)) for m in MEM[k[1]]], 0)
        t = load_dev(*k)[2]; assert len(s) == len(t) and np.isfinite(s).all()
        zo.writestr(fn, "\n".join(f"{p} {-v:.8f}" for p, v in zip(t.pair_id, s)) + "\n")
        chk = " | ".join(f"{lb} {eer_from_scores(s[L[tk + '_known']], L[tk + '_lab'][L[tk + '_known']]):.2f}" for lb, L in LAB.items())
        print(f"{c:18s} face clusters {cf.max() + 1:3d} voice clusters {cv.max() + 1:3d} | sanity pseudo-EER {chk}")
with zipfile.ZipFile(out) as zf:
    assert len(zf.namelist()) == 4
    for k, fn in FN_.items():
        d = pd.read_csv(zf.open(fn), sep=" ", header=None, names=["pid", "s"])
        assert (d.pid.values == load_dev(*k)[2].pair_id.values).all() and np.isfinite(d.s).all()
print("wrote + verified", out)
