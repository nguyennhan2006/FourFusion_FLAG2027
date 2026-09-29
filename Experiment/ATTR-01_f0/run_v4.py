"""ATTR-01 pilot on v4 train — falsification test for a shared pitch cue (docs/PLAN_V4.md §1 B). EDA only.

Question: after fixing gender, does a person's FACE predict their voice pitch well enough to tell them apart
from other people of the same gender?

Pre-registered analysis (fixed before looking at results):
  * F0: flag_extract.f0_stats (YIN + RMS voicing gate) per utterance -> speaker log-F0 = median over utterances
  * face: speaker-mean organiser VGG fc7, standardised, PCA-16 fitted inside each leave-one-speaker-out fold
  * within each gender: ridge(alpha=10) face -> speaker log-F0, leave-one-speaker-out prediction
  * pair test: r = |pred(face s) - F0(voice t)|; AUC of -r for t == s vs t != s (same gender)
  * null: permute speaker F0 within gender, refit everything, 1000 permutations -> p-value
  * sanity: F0 must separate genders strongly, otherwise the extractor is broken and nothing else counts
Sensitivity (reported, not used to pick): alpha in {1, 100}, PCA k in {8, 32}.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import flag_extract as X  # noqa: E402
import flag_lib as L  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
Xf, Xv, spk, gmap, _ = L.load_train(ROOT / "kaggle_upload")
txt = pd.read_csv(ROOT / "kaggle_upload/train/train_English.txt", sep=" ", header=None,
                  names=["pair_id", "label", "voice", "face", "spk", "spk_int"])

# ------------------------------------------------------------------ F0 per utterance (cached)
f0_csv = OUT / "v4_train_f0.csv"
if not f0_csv.exists():
    Z = zipfile.ZipFile(ROOT / "Input/train_set.zip")
    rows = []
    for i, v in enumerate(sorted(set(txt.voice))):
        rows.append(dict(voice=v, **X.f0_stats(X.load_audio(io.BytesIO(Z.read("train_set/train_set/" + v))))))
        if i % 1000 == 0:
            print("f0", i, flush=True)
    pd.DataFrame(rows).to_csv(f0_csv, index=False)
f0 = pd.read_csv(f0_csv)
u = f0.merge(txt[["voice", "spk"]].drop_duplicates(), on="voice")
S = u.groupby("spk").log_f0_median.median().rename("f0").to_frame()
S["gender"] = [gmap[s] for s in S.index]
S["n_utt"] = u.groupby("spk").size()
print(f"utterances with F0: {u.log_f0_median.notna().sum()}/{len(u)}; speakers {len(S)} "
      f"({(S.gender == 'm').sum()} m / {(S.gender == 'f').sum()} f)")

sanity = {g: float(np.exp(S[S.gender == g].f0).median()) for g in ("m", "f")}
auc_gender = roc_auc_score((S.gender == "f").astype(int), S.f0)
print(f"sanity: median F0 m {sanity['m']:.0f} Hz, f {sanity['f']:.0f} Hz, AUC(F0 -> female) {auc_gender:.3f}")
assert auc_gender > 0.9, "F0 extractor does not separate genders: fix it before reading anything else"

# ------------------------------------------------------------------ speaker-mean faces
spk_ids = list(S.index)
Fm = np.stack([Xf[spk == s].mean(0) for s in spk_ids])


def loso_pred(F, y, k, alpha):
    """Leave-one-speaker-out ridge predictions; PCA + standardisation fitted without the held-out speaker."""
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        Zt = (F[tr] - mu) / sd
        P = np.linalg.svd(Zt, full_matrices=False)[2][:k].T          # k is capped by n_train - 1 components
        A = Zt @ P; a = ((F[i] - mu) / sd) @ P
        am, as_ = A.mean(0), A.std(0) + 1e-6
        A, a = (A - am) / as_, (a - am) / as_
        ym = y[tr].mean()
        w = np.linalg.solve(A.T @ A + alpha * np.eye(P.shape[1]), A.T @ (y[tr] - ym))
        pred[i] = ym + a @ w
    return pred


def pair_auc(pred, y):
    r = np.abs(pred[:, None] - y[None, :])
    lab = np.eye(len(y))
    return roc_auc_score(lab.ravel(), -r.ravel())


def analyse(k=16, alpha=10.0, n_perm=1000, seed=0):
    res = {}
    rng = np.random.RandomState(seed)
    for g in ("m", "f"):
        idx = np.where(S.gender.values == g)[0]
        F, y = Fm[idx], S.f0.values[idx]
        pred = loso_pred(F, y, k, alpha)
        rho = pd.Series(pred).corr(pd.Series(y), method="spearman")
        auc = pair_auc(pred, y)
        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            null.append(pair_auc(loso_pred(F, yp, k, alpha), yp))
        null = np.array(null)
        res[g] = dict(n=len(idx), spearman_pred_vs_f0=round(float(rho), 3), pair_auc=round(float(auc), 3),
                      null_auc_mean=round(float(null.mean()), 3), null_auc_p95=round(float(np.percentile(null, 95)), 3),
                      p_value=round(float((1 + (null >= auc).sum()) / (1 + n_perm)), 4))
    return res


main = analyse()
print("\npre-registered (k=16, alpha=10):")
for g, r in main.items():
    print(f"  {g}: {r}")
sens = {f"k{k}_a{a}": analyse(k, a, n_perm=200) for k, a in [(16, 1.0), (16, 100.0), (8, 10.0), (32, 10.0)]}
print("\nsensitivity (200 permutations each):")
for name, r in sens.items():
    print(f"  {name}: " + " | ".join(f"{g} auc {v['pair_auc']} p {v['p_value']}" for g, v in r.items()))
json.dump(dict(sanity=dict(median_hz=sanity, auc_gender=auc_gender), main=main, sensitivity=sens),
          open(OUT / "attr01_v4.json", "w"), indent=1)
