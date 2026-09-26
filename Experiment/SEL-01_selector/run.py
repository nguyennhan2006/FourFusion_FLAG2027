"""SEL-01a — pseudo-identity selector for the Bangla cells, validated retrospectively.

Pseudo-labels come ONLY from unimodal structure + trial co-occurrence, never from any system's
cross-modal score:
  faces  : agglomerative clusters over all 4 dev files (En and Bn dev are the same people)
  voices : clusters per language
  link   : a voice cluster belongs to the face cluster it is paired with far more often than chance
           (each file is exactly 50% targets, so positives concentrate)
  label  : trial (face, voice) is pseudo-positive iff linked(voice cluster) == face cluster
Then every submitted zip gets a pseudo-EER per cell, compared with its real CodaBench EER.
Gate (PB-1): within-cell Spearman |rho| >= 0.6 and >= 75% pairwise order agreement on the Bangla cells.
"""
import sys, json, zipfile, itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EDA-000_raw"))
from eda_utils import load_train, load_dev, l2n, eer_from_scores

E = Path(__file__).resolve().parents[1]
CELLS = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
# distinct systems with real per-cell CodaBench EER (composites 000f/008/011 excluded: they repeat cells)
SYS = {
    "sub_base": ("../kaggle/output/nb2_v2/submissions/sub_base.zip", [24.80, 31.29, 32.18, 39.51]),
    "000c": ("EXP-000c_fop_baseline/out/submission_baseline.zip", [36.31, 38.98, 43.38, 50.00]),
    "000d": ("EXP-000d_cca_linear/out/submission_EXP000d.zip", [33.73, 34.00, 34.42, 41.01]),
    "000e": ("EXP-000d_cca_linear/out/submission_EXP000e_k4.zip", [28.97, 29.02, 37.47, 42.37]),
    "003b": ("EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_asnorm.zip", [25.79, 28.88, 33.81, 39.37]),
    "003c": ("EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip", [25.60, 27.88, 34.01, 37.87]),
    "004":  ("EXP-004_domain/out/submission_EXP004_align-coral.zip", [37.30, 37.84, 43.99, 46.32]),
    "007":  ("EXP-007_fusion/out/submission_EXP007_n10_w0.25.zip", [25.60, 29.02, 30.96, 38.01]),
    "010B": ("EXP-010_feat/out/submission_newfeat.zip", [23.81, 30.87, 32.38, 39.78]),
}
EXTRA = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}   # name=path.zip, no CB yet

# ---------------------------------------------------------------- unimodal spaces used ONLY for pseudo-labels
# SEL_LABELS=btc (default): organiser VGG-4096 faces + organiser 192 voices   -> SEL-01a
# SEL_LABELS=arc          : ArcFace-512 faces (face-face EER 1.7) + own ECAPA-192 -> SEL-01b
import os
LABELS = os.environ.get("SEL_LABELS", "btc")
FEATS = Path(__file__).resolve().parents[2] / "kaggle/output/feats_v2"
Xf, Xv, spk, _, _, gmap = load_train()
tag = lambda k: f"{k[0]}_{k[1]}"
if LABELS == "arc":
    Xf = np.load(FEATS / "face_arcface_train.npy"); Xv = np.load(FEATS / "voice_ecapa192_train.npy")
    devF = lambda k: np.load(FEATS / f"face_arcface_{tag(k)}.npy")
    devV = lambda k: np.load(FEATS / f"voice_ecapa192_{tag(k)}.npy")
    Fz = lambda X: l2n(l2n(X) - l2n(Xf).mean(0))
else:
    devF = lambda k: load_dev(*k)[0]; devV = lambda k: load_dev(*k)[1]
    mf, sf = Xf.mean(0), Xf.std(0) + 1e-6
    P = np.linalg.svd((Xf - mf) / sf, full_matrices=False)[2][:256].T
    Fz = lambda X: l2n(((X - mf) / sf) @ P)
mv, sv = Xv.mean(0), Xv.std(0) + 1e-6
Vz = lambda X: l2n(((X - mv) / sv) - ((X - mv) / sv).mean(0))
print("pseudo-label features:", LABELS)


def best_thr(Z, lab):
    best = (None, -1)
    for t in np.arange(0.4, 1.21, 0.05):
        c = AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
        a = adjusted_rand_score(lab, c)
        if a > best[1]: best = (t, a)
    return best


sub = np.isin(spk, np.random.RandomState(0).choice(np.unique(spk), 30, replace=False))
tF, aF = best_thr(Fz(Xf[sub]), spk[sub]); tV, aV = best_thr(Vz(Xv[sub]), spk[sub])
print(f"calibrated on 30 train speakers: face thr {tF:.2f} ARI {aF:.3f} | voice thr {tV:.2f} ARI {aV:.3f}")

dev = {k: load_dev(*k) for k in CELLS}
allF = np.concatenate([devF(k) for k in CELLS])
uF, invF = np.unique(allF.round(4), axis=0, return_inverse=True)
cF_all = AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=tF).fit_predict(Fz(uF))[invF.ravel()]
off = np.cumsum([0] + [len(dev[k][2]) for k in CELLS])
cF = {k: cF_all[off[i]:off[i + 1]] for i, k in enumerate(CELLS)}

labels, stats = {}, []
for lang in ["English", "Bangla"]:
    ks = [("no_gender", lang), ("gender", lang)]
    V = np.concatenate([devV(k) for k in ks])
    uV, invV = np.unique(V.round(4), axis=0, return_inverse=True)
    cV_all = AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=tV).fit_predict(Vz(uV))[invV.ravel()]
    fc_all = np.concatenate([cF[k] for k in ks])
    tab = pd.crosstab(cV_all, fc_all)
    link = {}
    for vc, row in tab.iterrows():
        n, top = row.sum(), row.idxmax()
        share, enrich = row.max() / n, row.max() / max(n * tab[top].sum() / tab.values.sum(), 1e-9)
        if n >= 4 and share >= 0.4 and enrich >= 3:
            link[vc] = top
    o = 0
    for k in ks:
        n = len(dev[k][2]); cv = cV_all[o:o + n]; o += n
        lv = np.array([link.get(c, -1) for c in cv])
        known = lv >= 0
        lab = (lv == cF[k]).astype(int)
        labels[k] = (known, lab)
        stats.append(dict(cell="/".join(k), rows=n, labelled=int(known.sum()), pos_rate=round(lab[known].mean(), 3)))
print(pd.DataFrame(stats).to_string(index=False))
np.savez(f"pseudo_labels_{LABELS}.npz", **{f"{k[0]}_{k[1]}_{n}": a for k in CELLS
                                            for n, a in zip(["known", "lab", "fc"], [*labels[k], cF[k]])})
if os.environ.get("SEL_LABELS_ONLY"):
    sys.exit(0)


def read(zp, fn, t):
    d = pd.read_csv(zipfile.ZipFile(E / zp if not Path(zp).is_absolute() else zp).open(fn), sep=" ", header=None, names=["pid", "s"])
    assert (d.pid.values == t.pair_id.values).all()
    return -d.s.values                                             # zips are lower = same


rows = []
for name, (zp, cb) in list(SYS.items()) + [(k, (v, [np.nan] * 4)) for k, v in EXTRA.items()]:
    for i, k in enumerate(CELLS):
        known, lab = labels[k]
        s = read(zp, CELLS[k], dev[k][2])
        rows.append(dict(system=name, cell="/".join(k), cb=cb[i], pseudo=round(eer_from_scores(s[known], lab[known]), 2)))
R = pd.DataFrame(rows)
R.to_csv(f"selector_retro_{LABELS}.csv", index=False)

print("\nPseudo-EER vs real CodaBench EER, per cell (8 systems):")
summ = []
for cell, g in R.dropna(subset=["cb"]).groupby("cell", sort=False):
    rho = spearmanr(g.pseudo, g.cb).correlation
    pairs = list(itertools.combinations(range(len(g)), 2))
    ok = sum(np.sign(g.pseudo.iloc[a] - g.pseudo.iloc[b]) == np.sign(g.cb.iloc[a] - g.cb.iloc[b]) for a, b in pairs)
    # the decisions that actually matter: close competitors among the good systems
    top = g[g.system.isin(["003b", "003c", "007", "010B", "sub_base"])]
    tp = list(itertools.combinations(range(len(top)), 2))
    ok_top = sum(np.sign(top.pseudo.iloc[a] - top.pseudo.iloc[b]) == np.sign(top.cb.iloc[a] - top.cb.iloc[b]) for a, b in tp)
    summ.append(dict(cell=cell, spearman=round(rho, 3), pair_acc=f"{ok}/{len(pairs)}", top4_pair_acc=f"{ok_top}/{len(tp)}"))
    print(f"\n{cell}\n" + g.sort_values("cb")[["system", "cb", "pseudo"]].to_string(index=False))
S = pd.DataFrame(summ); print("\n" + S.to_string(index=False))
json.dump(dict(thr=dict(face=tF, voice=tV), ari=dict(face=aF, voice=aV), labels=stats, summary=summ),
          open(f"selector_retro_{LABELS}.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- uncertainty: bootstrap over face clusters
# trials of one person are not independent -> resample pseudo-identities, not trials
def boot_delta(k, sa, sb, n=500, seed=0):
    known, lab = labels[k]
    fc = cF[k][known]; a, b, y = sa[known], sb[known], lab[known]
    ids = np.unique(fc); by = {c: np.where(fc == c)[0] for c in ids}
    rng = np.random.RandomState(seed); d = []
    for _ in range(n):
        idx = np.concatenate([by[c] for c in rng.choice(ids, len(ids))])
        if y[idx].min() == y[idx].max(): continue
        d.append(eer_from_scores(a[idx], y[idx]) - eer_from_scores(b[idx], y[idx]))
    return np.percentile(d, [2.5, 50, 97.5]), float(np.mean(np.array(d) < 0))


print("\nClose competitors: pseudo delta (A - B) with face-cluster bootstrap 95% CI vs real CB delta")
out = []
for k in CELLS:
    for A, B in itertools.combinations(["003b", "003c", "007", "010B", "sub_base"], 2):
        sa, sb = read(SYS[A][0], CELLS[k], dev[k][2]), read(SYS[B][0], CELLS[k], dev[k][2])
        ci, pA = boot_delta(k, sa, sb)
        i = list(CELLS).index(k); cbd = SYS[A][1][i] - SYS[B][1][i]
        sure = ci[0] > 0 or ci[2] < 0
        right = np.sign(ci[1]) == np.sign(cbd)
        out.append(dict(cell="/".join(k), pair=f"{A}-{B}", cb_delta=round(cbd, 2), pseudo_ci=f"[{ci[0]:+.2f}, {ci[2]:+.2f}]",
                        P_A_better=round(pA, 2), confident=sure, correct=bool(right)))
O = pd.DataFrame(out); O.to_csv(f"selector_pairs_{LABELS}.csv", index=False)
print(O.to_string(index=False))
c = O[O.confident]
print(f"\nconfident calls: {int(c.correct.sum())}/{len(c)} correct | uncertain calls: {len(O) - len(c)} "
      f"(of which the median direction was right {int(O[~O.confident].correct.sum())})")
