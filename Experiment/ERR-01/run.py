"""ERR-01 — look at the worst errors of the best submission (FUSE-03) on dev, with SEL-01b pseudo-labels.
FN = pseudo-positive trials with the lowest scores; FP = pseudo-negative trials with the highest scores.
For each: trial face | a typical face of the VOICE's owner (pseudo-identity), plus gender guessed from face and from
voice (linear probes trained on v4 train), clip duration and the score percentile."""
import sys, io, zipfile, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "Experiment/EDA-000_raw")); from eda_utils import load_train, load_dev, eer_from_scores, l2n
FE = R / "kaggle/output/feats_v2"; Z = zipfile.ZipFile(R / "Input/dev_set.zip")
L = np.load(R / "Experiment/SEL-01_selector/pseudo_labels_arc.npz")
dur = pd.read_csv(R / "Experiment/EDA-001_flag_questions/wav_durations.csv"); dmap = dict(zip(dur.name, dur.dur))
Xf, Xv, spk, _, _, gmap = load_train(); y = np.array([gmap[s] == "m" for s in spk])
mf, sf = Xf.mean(0), Xf.std(0) + 1e-6; P = np.linalg.svd((Xf - mf) / sf, full_matrices=False)[2][:256].T
fz = lambda A: l2n(((A - A.mean(0) + mf - mf) / sf) @ P); vz = lambda B: l2n((B - B.mean(0)) / (Xv.std(0) + 1e-6))
gf = LogisticRegression(C=.1, max_iter=3000).fit(l2n(((Xf - mf) / sf) @ P), y)
gv = LogisticRegression(C=.1, max_iter=3000).fit(l2n((Xv - Xv.mean(0)) / (Xv.std(0) + 1e-6)), y)
FN_ = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt", ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
       ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt", ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt"}
K = 8


def img(prot, rel):
    return Image.open(io.BytesIO(Z.read(f"dev_set/{prot}/{rel}"))).convert("RGB")


def run(k):
    prot, lang = k; tk = f"{prot}_{lang}"
    a, b, t = load_dev(*k)
    s = -pd.read_csv(zipfile.ZipFile(R / "Experiment/FUSE-03/out/submission_FUSE03.zip").open(FN_[k]), sep=" ", header=None)[1].values
    pct = pd.Series(s).rank(pct=True).values
    kn, lab, fc = L[tk + "_known"], L[tk + "_lab"], L[tk + "_fc"]
    e = eer_from_scores(s[kn], lab[kn])
    pf, pv = gf.predict_proba(fz(a))[:, 1], gv.predict_proba(vz(b))[:, 1]
    ev = l2n(np.load(FE / f"voice_ecapa192_{tk}.npy")); ev = l2n(ev - ev.mean(0))
    pos = np.where(kn & (lab == 1))[0]
    owner = lambda i: fc[pos[np.argmax(ev[pos] @ ev[i])]]        # face cluster of the voice's owner
    rep = {c: pos[fc[pos] == c][0] for c in np.unique(fc[pos])}  # a representative trial face per person
    d = np.array([dmap.get(f"dev_set/{prot}/{v}", np.nan) for v in t.voice])
    FNi = [i for i in np.argsort(s) if kn[i] and lab[i] == 1][:K]
    FPi = [i for i in np.argsort(-s) if kn[i] and lab[i] == 0][:K]
    rows = []
    fig, ax = plt.subplots(4, K, figsize=(2.1 * K, 9.4))
    for r0, (title, idx) in enumerate([("missed same-person (FN)", FNi), ("accepted different person (FP)", FPi)]):
        for c, i in enumerate(idx):
            o = owner(i); same_face_person = fc[i] == o
            ax[2 * r0, c].imshow(img(prot, t.face[i])); ax[2 * r0 + 1, c].imshow(img(prot, t.face[rep[o]]) if o in rep else np.ones((10, 10, 3)))
            ax[2 * r0, c].set_title(f"pct {pct[i]:.2f} | {d[i]:.1f}s\nface {'M' if pf[i] > .5 else 'F'}{pf[i]:.2f} voice {'M' if pv[i] > .5 else 'F'}{pv[i]:.2f}", fontsize=7)
            ax[2 * r0 + 1, c].set_title("voice owner" + (" (=same)" if same_face_person else ""), fontsize=7)
            rows.append(dict(cell=tk, kind=title.split()[0], trial=t.pair_id[i], pct=round(pct[i], 3), dur=round(d[i], 1),
                             p_male_face=round(pf[i], 2), p_male_voice=round(pv[i], 2), gender_mismatch=(pf[i] > .5) != (pv[i] > .5)))
        ax[2 * r0, 0].set_ylabel("trial face\n" + title, fontsize=8); ax[2 * r0 + 1, 0].set_ylabel("owner of voice", fontsize=8)
    for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"{tk}: pseudo-EER {e:.2f} (FUSE-03)", fontsize=11); plt.tight_layout()
    fig.savefig(f"err_{tk}.png", dpi=80); plt.close(fig)
    # aggregate: what distinguishes errors from correct trials?
    thr = np.quantile(s[kn & (lab == 0)], 1 - e / 100)
    err = kn & (((lab == 1) & (s < thr)) | ((lab == 0) & (s >= thr)))
    gm = (pf > .5) != (pv > .5)
    agg = dict(cell=tk, eer=round(e, 2),
               dur_err=round(np.nanmedian(d[err]), 2), dur_ok=round(np.nanmedian(d[kn & ~err]), 2),
               FN_rate_short=round(np.mean(s[kn & (lab == 1) & (d < 4)] < thr), 3), FN_rate_long=round(np.mean(s[kn & (lab == 1) & (d >= 8)] < thr), 3),
               neg_gender_mismatch_FP_rate=round(np.mean(s[kn & (lab == 0) & gm] >= thr), 3), neg_same_gender_FP_rate=round(np.mean(s[kn & (lab == 0) & ~gm] >= thr), 3))
    return rows, agg


allrows, aggs = [], []
for k in FN_:
    r, g = run(k); allrows += r; aggs.append(g)
pd.DataFrame(allrows).to_csv("err_cases.csv", index=False)
A = pd.DataFrame(aggs); A.to_csv("err_summary.csv", index=False); print(A.to_string(index=False))
