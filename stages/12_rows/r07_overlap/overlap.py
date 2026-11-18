"""Identity overlap between external persons (v1/v2, ids merged by name) and v4 train + dev, before training on them.
Face centroids in VGG space (standardised with v4 train stats, PCA-256); thresholds = midpoint between the SAME-person
and DIFFERENT-person distributions measured on v4 train (same rule as FLAG_08, log 37). Uses the FULL Kaggle features.

FourFusion reproduction repo (stage r07_overlap): copy of Experiment/SET-07/overlap.py (md5 529d57a96e4cbf83ca6c9faea2a4b534).
Changes (all listed in CHANGES.md; the computation lines are unchanged):
  * paths are arguments instead of the team-machine layout (R / "kaggle_upload", R / "kaggle/output/feats_ext_full",
    R / "kaggle/output/feats_v2", R / "kaggle", cwd / "overlap.csv");
  * --train-only: the dev-based secondary rule (THR_DEV, top-20 dev faces, lines 24-27 and 36 of the original) is skipped and
    no dev file is read; overlap = max_cos_v4train > THR only. The two persons the chain drops are flagged by the v4-train rule
    in the original output too (max_cos_v4train 0.650 / 0.898, next highest 0.621), so the list is expected to be unchanged.
    python overlap.py --data DIR --feats-ext DIR [--lib DIR] [--out overlap.csv] [--train-only]
"""
import os, sys, argparse, numpy as np, pandas as pd                       # FF: argparse added
from pathlib import Path
R = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()                                           # FF: path arguments (originals in the help text)
ap.add_argument("--data", required=True, type=Path,
                help="organiser data root: train_English_{faces,voices}.csv, train_English.txt, meta_file_train_set.csv "
                     "(+ the dev feature csv files unless --train-only); original R / 'kaggle_upload'")
ap.add_argument("--feats-ext", required=True, type=Path,
                help="folder with ext_v{1,2}_complete_{face,voice}.npy / _meta.csv / ext_v1_complete_speakers.csv "
                     "(stage r08_mav_rows, feats_ext/); original R / 'kaggle/output/feats_ext_full'")
ap.add_argument("--feats-dir", type=Path, default=None,
                help="FeatureStore feats_dir; original R / 'kaggle/output/feats_v2' (FLAG_04), which holds no array this script "
                     "reads -> default = --feats-ext")
ap.add_argument("--lib", type=Path, default=Path(__file__).resolve().parent / "lib",
                help="folder with flag_v2.py / flag_lib.py; original R / 'kaggle'")
ap.add_argument("--out", type=Path, default=Path("overlap.csv"), help="original: overlap.csv in the working directory")
ap.add_argument("--train-only", action="store_true", help="skip the dev-based rule; read no dev file")
OPT = ap.parse_args()
sys.path.insert(0, str(OPT.lib))                                           # original: sys.path.insert(0, str(R / "kaggle"))
os.environ["FLAG_FEATS_EXTRA"] = str(OPT.feats_ext)                        # original: str(R / "kaggle/output/feats_ext_full")
import flag_v2 as V
if OPT.train_only:                                                         # FF: FeatureStore.__init__ loads the four dev files
    V.L.load_dev = lambda *a, **k: None                                  # (flag_v2.py:47); never used below in this mode
st = V.FeatureStore(OPT.data, feats_dir=OPT.feats_dir or OPT.feats_ext)        # original: (R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
Xf = st.Xf; mf, sf = Xf.mean(0), Xf.std(0) + 1e-6
P = np.linalg.svd((Xf - mf) / sf, full_matrices=False)[2][:256].T
emb = lambda A: V.l2n(((A - mf) / sf) @ P)
rng = np.random.RandomState(0); same, cent = [], {}
for s_ in np.unique(st.spk):
    i = np.where(st.spk == s_)[0]
    if len(i) < 8: continue
    rng.shuffle(i); h = len(i) // 2
    same.append(V.l2n(emb(Xf[i[:h]]).mean(0, keepdims=True))[0] @ V.l2n(emb(Xf[i[h:]]).mean(0, keepdims=True))[0])
    cent[s_] = V.l2n(emb(Xf[i]).mean(0, keepdims=True))[0]
C4 = np.stack(list(cent.values())); diff = (C4 @ C4.T)[np.triu_indices(len(C4), 1)]
THR = float((np.percentile(diff, 99.5) + np.percentile(same, 1)) / 2)
E4 = emb(Xf); ks = list(cent)
if not OPT.train_only:                                                     # FF: dev-based rule only without --train-only
    absent = [np.sort(E4[st.spk != s_] @ cent[s_])[::-1][:20].mean() for s_ in ks]
    present = [np.sort(E4[st.spk == s_] @ cent[s_])[::-1][:20].mean() for s_ in ks]
    THR_DEV = float((np.percentile(absent, 99) + np.percentile(present, 5)) / 2)
    devE = emb(np.concatenate([st.dev[k][0] for k in V.CELLS]))
    print(f"thresholds: centroid {THR:.3f} (same p1 {np.percentile(same, 1):.3f}, diff p99.5 {np.percentile(diff, 99.5):.3f}) | dev top-20 {THR_DEV:.3f}")
else:
    print(f"thresholds: centroid {THR:.3f} (same p1 {np.percentile(same, 1):.3f}, diff p99.5 {np.percentile(diff, 99.5):.3f}) | dev rule skipped (--train-only)")
rows = []
for src in ("v1_complete", "v2_complete"):
    D = st.ext_source(src); key = lambda x: D["key_of"].get(x, f"{src}:{x}")
    E_ = emb(D["Fa"]); per = D["fm"].spk.map(key).values
    for p in np.unique(per):
        c = V.l2n(E_[per == p].mean(0, keepdims=True))[0]; b = C4 @ c
        rows.append(dict(person=p, max_cos_v4train=float(b.max()), v4train_id=ks[int(b.argmax())],
                         top20_cos_dev=float(np.sort(devE @ c)[::-1][:20].mean()) if not OPT.train_only else float("nan")))
O = pd.DataFrame(rows); O["overlap"] = (O.max_cos_v4train > THR) | ((O.top20_cos_dev > THR_DEV) if not OPT.train_only else False)
O.to_csv(OPT.out, index=False)                                             # original: O.to_csv("overlap.csv", index=False)
if not OPT.train_only:
    print(O.sort_values("top20_cos_dev", ascending=False).head(6).round(3).to_string(index=False))
print(O.sort_values("max_cos_v4train", ascending=False).head(4).round(3).to_string(index=False))
print(f"flagged {int(O.overlap.sum())} of {len(O)} external persons:", list(O[O.overlap].person))
