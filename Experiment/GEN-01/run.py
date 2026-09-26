"""GEN-01 — explicit gender-agreement score for the no_gender cells only.

s_gender(f, v) = P(m|face) P(m|voice) + P(f|face) P(f|voice), probes = logistic regression trained on the 70
train speakers using ORGANISER features (VGG face, 192 voice) -- deliberately not ArcFace/ECAPA-own, so the
score is independent of the features that built the SEL-01b pseudo-labels.
Fused by rank into the current best no_gender scores (FUSE-01). Weights pre-registered: 0.1 0.2 0.3 0.5.
The gender cells are untouched (same-gender negatives make this score uninformative there).
"""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "EDA-000_raw"))
from eda_utils import load_train, load_dev, eer_from_scores, l2n

Xf, Xv, spk, _, _, gmap = load_train()
y = np.array([gmap[s] == "m" for s in spk]).astype(int)
mf, sf = Xf.mean(0), Xf.std(0) + 1e-6
P = np.linalg.svd((Xf - mf) / sf, full_matrices=False)[2][:256].T
mv, sv = Xv.mean(0), Xv.std(0) + 1e-6
Fp = lambda X, mu: l2n(((X - X.mean(0) + mu - mf) / sf) @ P)      # per-file centring to the train mean
Vp = lambda X, mu: l2n((X - X.mean(0) + mu - mv) / sv)
tf, tv = Fp(Xf, Xf.mean(0)), Vp(Xv, Xv.mean(0))
cf = LogisticRegression(C=0.1, max_iter=3000); cv = LogisticRegression(C=0.1, max_iter=3000)
for name, clf, X in [("face", cf, tf), ("voice", cv, tv)]:
    p = cross_val_predict(clf, X, y, groups=spk, cv=GroupKFold(5), method="predict_proba")[:, 1]
    print(f"{name} gender probe, speaker-disjoint CV acc: {((p > .5) == y).mean():.3f}")
    clf.fit(X, y)

L = np.load(E / "SEL-01_selector/pseudo_labels_arc.npz")
FUSE = E / "FUSE-01/out/submission_FUSE01.zip"
rank = lambda s: rankdata(s) / (len(s) + 1)
rows, best = [], {}
for p_, l_, fn in [("no_gender", "English", "no_gender/sub_score_v4_English_heard.txt"),
                   ("no_gender", "Bangla", "no_gender/sub_score_v4_Bangla_unheard.txt")]:
    a, b, t = load_dev(p_, l_)
    pf, pv = cf.predict_proba(Fp(a, Xf.mean(0)))[:, 1], cv.predict_proba(Vp(b, Xv.mean(0)))[:, 1]
    sg = pf * pv + (1 - pf) * (1 - pv)
    d = pd.read_csv(zipfile.ZipFile(FUSE).open(fn), sep=" ", header=None, names=["pid", "s"])
    assert (d.pid.values == t.pair_id.values).all()
    base = -d.s.values
    tk = f"{p_}_{l_}"; kn, lab = L[tk + "_known"], L[tk + "_lab"]
    print(f"\n{tk}: face says male {pf.mean():.2f}, voice says male {pv.mean():.2f}, "
          f"gender-mismatch trials {(np.abs(pf - pv) > .5).mean():.2f}")
    for w in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        s = (1 - w) * rank(base) + w * rank(sg)
        rows.append(dict(cell=tk, w=w, pseudo=round(eer_from_scores(s[kn], lab[kn]), 2)))
        best[(tk, w)] = s
R = pd.DataFrame(rows); print(R.pivot(index="w", columns="cell", values="pseudo").to_string())
R.to_csv("gen01.csv", index=False)
np.savez("gen01_scores.npz", **{f"{k[0]}_w{k[1]}": v for k, v in best.items()})
