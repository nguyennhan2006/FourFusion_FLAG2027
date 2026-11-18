r"""LIDSTRICT-01: build lid_strict.json = every speaker whose VOICE must not be learned from (README §2), by category:
  bn_vl    VoxLingua107 strict rule (top-1 bn / as or p(bn) + p(as) >= 0.2) on the stored LID of FLAG_42 / FLAG_53 / FLAG_46
  bn_wl    Whisper-large-v3 layer 3 (FLAG_150 lid_whisper.json, when downloaded to out/)
  mav_audit  LID-AUDIT-01 rule D1 on MAV v1 / v2 (three LID votes per file; replaces bn_wl's MAV lists when out/mav_d1.json exists)
  harm_g3  DQUAL-01 G3 (bottom 5 % within-person consistency of the source: mixed identities), VoxCeleb2 / VoxBlink only
'vox2' / 'vb' = the union the code drops (T3ENC-01.strict_drop); 'mav_v1' / 'mav_v2' = MAV speakers to drop (FLAG_150).
    python Experiment/LIDSTRICT-01/merge.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
EXP = H.parent


def bn_vl():
    lab = [l.split(":")[0].strip() for l in json.load(open(EXP / "T3ENC-01/feats_vox2lid/lid_labels.json"))]
    bn, as_ = lab.index("bn"), lab.index("as")
    out = {"vox2": set(), "vb": set()}
    for d, src in [("T3ENC-01/feats_vox2lid", "vox2"), ("SCALE-ID-01/feats_vox2b", "vox2"), ("SCALE-VB/feats_vb", "vb")]:
        P = np.load(EXP / d / "voice_lid_probs.npy").astype(np.float32)
        s = pd.read_csv(EXP / d / "voice_meta.csv", dtype=str).spk.to_numpy()
        top = P.argmax(1)
        f = (top == bn) | (top == as_) | (P[:, bn] + P[:, as_] >= .2)
        old = set(json.load(open(EXP / d / "lid_summary.json"))["speakers_dropped_ids"])
        assert old <= set(s[f]), d
        out[src] |= set(s[f])
    return out


def main():
    cat = {"bn_vl": bn_vl()}
    w = H / "out/lid_whisper.json"
    if w.exists():
        F = json.loads(w.read_text())["flagged"]
        cat["bn_wl"] = {k: set(F.get(k, [])) for k in ("vox2", "vb", "mav_v1", "mav_v2")}
    a = EXP / "LID-AUDIT-01/out/mav_w.json"                        # LID-AUDIT-01, user 3/11 14:30 "chỉ whisper": MAV at FILE level,
    mav_files = None                                               # Whisper-flagged files only; no MAV speaker is dropped
    if a.exists():
        A = json.loads(a.read_text())
        assert A.get("exp") == "LID-AUDIT-01" and A.get("rule") == "W"
        for k in ("mav_v1", "mav_v2"):
            cat.get("bn_wl", {}).pop(k, None)                      # bn_wl's MAV speaker lists = W or V (V mislabels South Asian voices)
        mav_files = sorted(A["files"])
    d = pd.read_parquet(EXP / "DQUAL-01/persons.parquet")
    g3 = d[(d.G3 == False) & d.source.isin(["VoxCeleb2", "VoxBlink"])]
    cat["harm_g3"] = {"vox2": set(g3.spk[g3.source == "VoxCeleb2"].astype(str)), "vb": set(g3.spk[g3.source == "VoxBlink"].astype(str))}
    U = {k: sorted(set().union(*[c.get(k, set()) for c in cat.values()])) for k in ("vox2", "vb", "mav_v1", "mav_v2")}
    J = dict(exp="LIDSTRICT-01", created=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), model="VoxLingua107 ECAPA + Whisper-large-v3",
             rule="voice of a speaker is never learned from if any clip is flagged Bengali (bn_vl / bn_wl) or the speaker fails DQUAL-01 G3 (harm_g3)",
             layers=sorted(cat), counts={k: len(v) for k, v in U.items()},
             by_category={c: {k: len(v) for k, v in x.items()} for c, x in cat.items()}, **U,
             categories={c: {k: sorted(v) for k, v in x.items()} for c, x in cat.items()})
    if mav_files is not None:   # read by ext_rows() (kaggle/build_l0_notebooks.py) under LID_STRICT=1
        J.update(mav_rule="MAV v1 / v2: drop the voice FILES with a Whisper-large-v3 vote (LID-AUDIT-01 W, user 3/11)", mav_files=mav_files)
    (H / "lid_strict.json").write_text(json.dumps(J, indent=1))
    removed_csv(cat, U)
    print(json.dumps({"counts": J["counts"], "by_category": J["by_category"]}, indent=1))


def removed_csv(cat, U):
    """removed_speakers.csv for the team: one row per speaker whose voice is dropped, ly_do = its categories; keeps the VoxLingua
    detail columns of the first version (412 Bengali speakers); MAV rows only once LID-AUDIT-01 has decided (mav_audit)."""
    name = {"vox2": "VoxCeleb2", "vb": "VoxBlink", "mav_v1": "MAV-Celeb v1", "mav_v2": "MAV-Celeb v2"}
    srcs = ["vox2", "vb"] + (["mav_v1", "mav_v2"] if "mav_audit" in cat else [])
    R = pd.DataFrame([dict(nguon=name[s], speaker_id=p, ly_do="+".join(c for c in cat if p in cat[c].get(s, set())))
                      for s in srcs for p in U[s]])
    f = H / "removed_speakers.csv"
    if f.exists():
        old = pd.read_csv(f, dtype={"speaker_id": str}).drop(columns=["ly_do"], errors="ignore").rename(columns={"luat": "luat_voxlingua"})
        R = R.merge(old, on=["nguon", "speaker_id"], how="left", validate="1:1")
    R.to_csv(f, index=False)
    print("removed_speakers.csv:", R.groupby("nguon").size().to_dict())


if __name__ == "__main__":
    main()
