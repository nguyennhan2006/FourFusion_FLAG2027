"""FUSE-02 — add the round-2 systems (NB-3) to per-cell rank fusion. Judged by SEL-01b pseudo-labels.

Pre-registered candidates (fixed before seeing r2 numbers):
  every cell : r2 singles + the current best of that cell (FUSE-01)
  English    : 007+010B (current)  and  007+010B+X  for X in r2
  Bangla     : 003c (current)      and  003c+X, 003c+mix+wavlm
"""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "EDA-000_raw"))
from eda_utils import load_dev, eer_from_scores
NB3 = E.parent / "kaggle/NB3"
SRC = {"003c": E / "EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip",
       "007": E / "EXP-007_fusion/out/submission_EXP007_n10_w0.25.zip",
       "010B": E / "EXP-010_feat/out/submission_newfeat.zip",
       "r2_192": NB3 / "r2_ecapa192.zip", "r2_mix": NB3 / "r2_ecapa192_mix.zip", "r2_wavlm": NB3 / "r2_wavlm.zip"}
CELLS = [("no_gender", "English", "no_gender/sub_score_v4_English_heard.txt"),
         ("no_gender", "Bangla", "no_gender/sub_score_v4_Bangla_unheard.txt"),
         ("gender", "English", "gender/sub_score_v4_English_heard.txt"),
         ("gender", "Bangla", "gender/sub_score_v4_Bangla_unheard.txt")]
R2 = ["r2_192", "r2_mix", "r2_wavlm"]
CAND = {"English": [["007", "010B"]] + [[x] for x in R2] + [["007", "010B", x] for x in R2],
        "Bangla": [["003c"]] + [[x] for x in R2] + [["003c", x] for x in R2] + [["003c", "r2_mix", "r2_wavlm"]]}
L = np.load(E / "SEL-01_selector/pseudo_labels_arc.npz")
rank = lambda s: rankdata(s) / (len(s) + 1)
rows = []
for p, l, fn in CELLS:
    t = load_dev(p, l)[2]; S = {}
    for n, zp in SRC.items():
        d = pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None, names=["pid", "s"])
        assert (d.pid.values == t.pair_id.values).all(); S[n] = -d.s.values
    kn, y = L[f"{p}_{l}_known"], L[f"{p}_{l}_lab"]
    ref = None
    for mem in CAND[l]:
        s = S[mem[0]] if len(mem) == 1 else np.mean([rank(S[m]) for m in mem], 0)
        e = eer_from_scores(s[kn], y[kn]); ref = e if ref is None else ref
        rows.append(dict(cell=f"{p}/{l}", cand="+".join(mem), pseudo=round(e, 2), gain=round(ref - e, 2)))
    # how different are the r2 systems from the incumbents? (low correlation = room for fusion)
    inc = ["007", "010B"] if l == "English" else ["003c"]
    print(f"{p}/{l} Spearman of scores:", {x: {i: round(pd.Series(S[x]).corr(pd.Series(S[i]), 'spearman'), 3) for i in inc} for x in R2})
R = pd.DataFrame(rows); R.to_csv("fuse02.csv", index=False)
for c, g in R.groupby("cell", sort=False):
    print(f"\n{c}\n" + g.sort_values("pseudo").to_string(index=False))
