"""NOISE-01 — label noise in v4 train. For every row: nearest speaker centroid (leave-one-out for its own speaker) with
ArcFace (face) and own ECAPA-192 + organiser 192 (voice). A row whose face AND voice both point to another speaker, or
whose face points elsewhere with a large margin, is suspicious."""
import sys, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle"))
import flag_v2 as V
st = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
spk = st.spk; ids, lab = np.unique(spk, return_inverse=True)
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))


def nearest(Z):
    Z = cent(Z); S = np.zeros((len(ids), Z.shape[1])); np.add.at(S, lab, Z); cnt = np.bincount(lab)
    own = (S[lab] - Z) / np.maximum(cnt[lab] - 1, 1)[:, None]          # leave-one-out own centroid
    C = V.l2n(S / cnt[:, None]); own = V.l2n(own)
    sim = Z @ C.T; sim[np.arange(len(Z)), lab] = np.sum(Z * own, 1)
    pred = sim.argmax(1); other = sim.copy(); other[np.arange(len(Z)), lab] = -9
    return pred, sim[np.arange(len(Z)), lab], other.max(1), ids[other.argmax(1)]


rows = {}
for name, Z in [("face_arcface", st.face("arcface", "train")), ("voice_ecapa192", st.voice("ecapa192", "train")), ("voice_given", st.voice("given", "train"))]:
    pred, s_own, s_oth, who = nearest(Z)
    rows[name] = (pred != lab); rows[name + "_margin"] = s_own - s_oth; rows[name + "_to"] = who
D = pd.DataFrame(rows); D["spk"] = spk
D["face_wrong"] = D.face_arcface; D["voice_wrong"] = D.voice_ecapa192 & D.voice_given
D["both_wrong"] = D.face_wrong & D.voice_wrong
print(f"rows {len(D)} | face points elsewhere {D.face_wrong.mean():.2%} | voice (both encoders) elsewhere {D.voice_wrong.mean():.2%} "
      f"| BOTH {D.both_wrong.mean():.2%} | face elsewhere with margin < -0.1: {(D.face_arcface_margin < -0.1).mean():.2%}")
per = D.groupby("spk")[["face_wrong", "voice_wrong", "both_wrong"]].mean().sort_values("face_wrong", ascending=False)
print("worst speakers:\n", per.head(8).round(3).to_string())
D.to_csv("noise_rows.csv", index=False)
