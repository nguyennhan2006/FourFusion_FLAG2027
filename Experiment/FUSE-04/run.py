"""FUSE-04 — ReDimNet2 round (NB-4) against the current best submission (FUSE-03, CB 28.69).

Pre-registered candidates per cell (fixed before looking at r3 numbers):
  cur          : the cell as submitted in FUSE-03
  X            : each r3 system alone
  cur+X        : rank-mean of the current cell's MEMBERS plus X (English: 007,010B,X ; Bangla: 003c,r2_mix,X)
Judged with both pseudo-label sets (arc = SEL-01b, btc = SEL-01a, independent of own-ECAPA/ArcFace).
"""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "EDA-000_raw"))
from eda_utils import load_dev, eer_from_scores
K = E.parent / "kaggle"
SRC = {"cur": E / "FUSE-03/out/submission_FUSE03.zip",
       "003c": E / "EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip",
       "007": E / "EXP-007_fusion/out/submission_EXP007_n10_w0.25.zip",
       "010B": E / "EXP-010_feat/out/submission_newfeat.zip",
       "r2_mix": K / "NB3/r2_ecapa192_mix.zip"}
R3 = ["r3_rdn6multi", "r3_rdn6multi_mix", "r3_rdn6vox", "r3_rdn6vox_mix"]
for r in R3: SRC[r] = K / f"NB4/submissions/{r}.zip"
CELLS = [("no_gender", "English", "no_gender/sub_score_v4_English_heard.txt"),
         ("no_gender", "Bangla", "no_gender/sub_score_v4_Bangla_unheard.txt"),
         ("gender", "English", "gender/sub_score_v4_English_heard.txt"),
         ("gender", "Bangla", "gender/sub_score_v4_Bangla_unheard.txt")]
MEM = {"English": ["007", "010B"], "Bangla": ["003c", "r2_mix"]}
rank = lambda s: rankdata(s) / (len(s) + 1)
LAB = {l: np.load(E / f"SEL-01_selector/pseudo_labels_{l}.npz") for l in ["arc", "btc"]}
rows, SC = [], {}
for p, l, fn in CELLS:
    t = load_dev(p, l)[2]; S = {}
    for n, zp in SRC.items():
        d = pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None, names=["pid", "s"])
        assert (d.pid.values == t.pair_id.values).all(); S[n] = -d.s.values
    C = {"cur": S["cur"]}
    for x in R3:
        C[x] = S[x]
        C["cur+" + x] = np.mean([rank(S[m]) for m in MEM[l] + [x]], 0)
    for name, s in C.items():
        r = dict(cell=f"{p}/{l}", cand=name)
        for lb, L in LAB.items():
            kn, y = L[f"{p}_{l}_known"], L[f"{p}_{l}_lab"]
            r[lb] = round(eer_from_scores(s[kn], y[kn]), 2)
        rows.append(r); SC[(f"{p}/{l}", name)] = s
    print(f"{p}/{l} Spearman r3 vs cur:", {x: round(pd.Series(S[x]).corr(pd.Series(S["cur"]), "spearman"), 3) for x in R3})
R = pd.DataFrame(rows)
for c, g in R.groupby("cell", sort=False):
    ref = g[g.cand == "cur"].iloc[0]
    g = g.assign(gain_arc=(ref.arc - g.arc).round(2), gain_btc=(ref.btc - g.btc).round(2)).sort_values("arc")
    print(f"\n{c}\n" + g.to_string(index=False))
R.to_csv("fuse04.csv", index=False)
np.savez("fuse04_scores.npz", **{f"{c}|{n}".replace("/", "_"): v for (c, n), v in SC.items()})
