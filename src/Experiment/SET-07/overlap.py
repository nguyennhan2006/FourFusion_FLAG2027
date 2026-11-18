"""Identity overlap between external persons (v1/v2, ids merged by name) and v4 train + dev, before training on them.
Face centroids in VGG space (standardised with v4 train stats, PCA-256); thresholds = midpoint between the SAME-person
and DIFFERENT-person distributions measured on v4 train (same rule as FLAG_08, log 37). Uses the FULL Kaggle features."""
import os, sys, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle"))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext_full")
import flag_v2 as V
st = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
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
absent = [np.sort(E4[st.spk != s_] @ cent[s_])[::-1][:20].mean() for s_ in ks]
present = [np.sort(E4[st.spk == s_] @ cent[s_])[::-1][:20].mean() for s_ in ks]
THR_DEV = float((np.percentile(absent, 99) + np.percentile(present, 5)) / 2)
devE = emb(np.concatenate([st.dev[k][0] for k in V.CELLS]))
print(f"thresholds: centroid {THR:.3f} (same p1 {np.percentile(same, 1):.3f}, diff p99.5 {np.percentile(diff, 99.5):.3f}) | dev top-20 {THR_DEV:.3f}")
rows = []
for src in ("v1_complete", "v2_complete"):
    D = st.ext_source(src); key = lambda x: D["key_of"].get(x, f"{src}:{x}")
    E_ = emb(D["Fa"]); per = D["fm"].spk.map(key).values
    for p in np.unique(per):
        c = V.l2n(E_[per == p].mean(0, keepdims=True))[0]; b = C4 @ c
        rows.append(dict(person=p, max_cos_v4train=float(b.max()), v4train_id=ks[int(b.argmax())],
                         top20_cos_dev=float(np.sort(devE @ c)[::-1][:20].mean())))
O = pd.DataFrame(rows); O["overlap"] = (O.max_cos_v4train > THR) | (O.top20_cos_dev > THR_DEV)
O.to_csv("overlap.csv", index=False)
print(O.sort_values("top20_cos_dev", ascending=False).head(6).round(3).to_string(index=False))
print(O.sort_values("max_cos_v4train", ascending=False).head(4).round(3).to_string(index=False))
print(f"flagged {int(O.overlap.sum())} of {len(O)} external persons:", list(O[O.overlap].person))
