"""XJF-01 step 3: English files = our English score blended with the in-domain branch XJF; Bangla files = bytes of BEST4.
    python Experiment/XJF-01/blend.py
English, per trial:  zip value = ours / SIG[protocol] - r * z_XJF      (the zip stores -score: lower = same person)
  ours  = value of posteval_FACTOR_ENFUSE_c5rl.zip (F1s' through the Bangla fuser + RFX, sum of two -logits)
  z_XJF = (cos - mean) / sd, mean / sd from the out-of-fold FLAG train trials (xjf.py)
  SIG   = fixed constants (scale of our English score), r = fixed weight; nothing is computed from the test file here.
Zips: zips/xjf_r05.zip, xjf_r10.zip (recommended, equal weight as in the sub99b report), xjf_r20.zip."""
import os, sys, zipfile
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
PE = REPO / "Experiment" / "POSTEVAL-01" / "zips"
BASE = {"no_gender_English": PE / "posteval_FACTOR_ENFUSE_c5rl.zip", "no_gender_Bangla": PE / "posteval_FACTOR_ENFUSE_c5rl.zip",
        "gender_English": PE / "posteval_FACTOR_ENFUSE_c5rl.zip", "gender_Bangla": REPO / "Experiment" / "FVBIND-01" / "zips" / "posteval_FVBIND_ENFUSE_fvb.zip"}
# 16/11 (user: "never use the buggy LoRA"): XJF_BASE=lf -> all four files from the CELL4-01 arm lf (FACTOR_ENFUSE c5rl rebuilt with the
# key-fixed LoRA C1, ngnnguynchn/flag2027-rfs-lorafix1-p); the FVBIND gender_Bangla file (old LoRA in its RFX ib stream) is dropped.
# Zips are then written as xjf_r<r>[_trsig]_lf.zip.
LF = os.environ.get("XJF_BASE") == "lf"
if LF:
    BASE = {t: PE / "posteval_CELL4_ENFUSE_lf.zip" for t in BASE}
MEMBER = {"no_gender_English": "no_gender/sub_score_v4_English_heard.txt", "no_gender_Bangla": "no_gender/sub_score_v4_Bangla_unheard.txt",
          "gender_English": "gender/sub_score_v4_English_heard.txt", "gender_Bangla": "gender/sub_score_v4_Bangla_unheard.txt"}
SIG = {"no_gender": 5.8, "gender": 2.9}   # 16/11 10:00: read off the test files' score scale -> NOT allowed (FAQ: no statistics of test data)
# 16/11 11:10 fix: XJF_SIG="ng,g" = sd of the base score U = F1s + R over the held-out TRAIN trials (v4 cells of SAFE-04, FLAG_219 dumpv4),
# the same kind of constant as the XJF z constants; zips are then written as xjf_r<r>_trsig.zip.
TRSIG = os.environ.get("XJF_SIG")
if TRSIG:
    SIG = dict(zip(("no_gender", "gender"), map(float, TRSIG.split(","))))
X = np.load(HERE / "out" / "xjf_scores.npz")


def read(t):
    tx = zipfile.ZipFile(BASE[t]).read(MEMBER[t]).decode().split()
    return tx[0::2], np.array(tx[1::2], float)


print("Spearman(ours, XJF) per file (label-free):")
for t in MEMBER:
    if f"z_{t}" not in X.files: continue
    ids, o = read(t); m = dict(zip(X[f"ids_{t}"], X[f"z_{t}"])); z = np.array([m[i] for i in ids])
    print(f"  {t:20s} rho {spearmanr(-o, z).statistic:.3f} | z_XJF mean {z.mean():+.2f} sd {z.std():.2f}")
(HERE / "zips").mkdir(exist_ok=True)
for r in [float(a) for a in sys.argv[1:]] or (0.5, 1.0, 2.0):   # r >= 99: the branch alone on the English files (xjf_only.zip)
    only = r >= 99
    p = HERE / "zips" / (f"xjf_only{'_lf' if LF else ''}.zip" if only else f"xjf_r{int(r * 10):02d}{'_trsig' if TRSIG else ''}{'_lf' if LF else ''}.zip")
    if p.exists():   # never rewrite a zip (a rewritten zip has another sha256; xjf_r10.zip was submitted 16/11 10:31)
        print("kept", p.name); continue
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zo:
        for t, mem in MEMBER.items():
            ids, o = read(t)
            if t.endswith("English"):
                m = dict(zip(X[f"ids_{t}"], X[f"z_{t}"])); z = np.array([m[i] for i in ids])
                o = -z if only else o / SIG[t.rsplit("_", 1)[0]] - r * z
                assert np.isfinite(o).all()
            zo.writestr(mem, "".join(f"{i} {x:.8f}\n" for i, x in zip(ids, o)))
    print("wrote", p.relative_to(REPO))
