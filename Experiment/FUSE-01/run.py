"""FUSE-01 — per-cell fusion of already-submitted systems, judged by SEL-01b pseudo-labels.

No training. Candidate set is fixed BEFORE looking at results (pre-registered) to limit dev overfitting:
  singles        : 003c, 007, 010B, sub_base
  rank-mean      : every pair and the triple of {003c, 007, 010B}, + all four
  z-mean         : the triple, to compare rank vs z-score fusion
Decision rule (PLAN_V3 §1b): a fusion replaces the cell's current best only if it beats it by >= 1.0
pseudo-EER AND wins in >= 90% of identity-bootstrap resamples.
"""
import itertools, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
import sys
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "EDA-000_raw"))
from eda_utils import eer_from_scores, load_dev

CELLS = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
SRC = {"003c": "EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip",
       "007": "EXP-007_fusion/out/submission_EXP007_n10_w0.25.zip",
       "010B": "EXP-010_feat/out/submission_newfeat.zip",
       "sub_base": "../kaggle/output/nb2_v2/submissions/sub_base.zip"}
CB = {"003c": [25.60, 27.88, 34.01, 37.87], "007": [25.60, 29.02, 30.96, 38.01],
      "010B": [23.81, 30.87, 32.38, 39.78], "sub_base": [24.80, 31.29, 32.18, 39.51]}
BEST_NOW = {"no_gender/English": "010B", "no_gender/Bangla": "003c", "gender/English": "007", "gender/Bangla": "003c"}
L = np.load(E / "SEL-01_selector/pseudo_labels_arc.npz")

rank = lambda s: rankdata(s) / (len(s) + 1)
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
CANDS = {n: ("single", [n]) for n in SRC}
for r in (2, 3):
    for c in itertools.combinations(["003c", "007", "010B"], r):
        CANDS["rank:" + "+".join(c)] = ("rank", list(c))
CANDS["rank:all4"] = ("rank", list(SRC))
CANDS["z:003c+007+010B"] = ("z", ["003c", "007", "010B"])


def fused(kind, members, S):
    if kind == "single": return S[members[0]]
    f = rank if kind == "rank" else z
    return np.mean([f(S[m]) for m in members], 0)


def boot_win(fc, y, a, b, n=1000, seed=0):
    ids = np.unique(fc); by = {c: np.where(fc == c)[0] for c in ids}
    rng = np.random.RandomState(seed); w = []
    for _ in range(n):
        idx = np.concatenate([by[c] for c in rng.choice(ids, len(ids))])
        w.append(eer_from_scores(a[idx], y[idx]) < eer_from_scores(b[idx], y[idx]))
    return float(np.mean(w))


rows, out_scores = [], {}
for i, (k, fn) in enumerate(CELLS.items()):
    cell = "/".join(k); t = load_dev(*k)[2]
    S = {}
    for n, zp in SRC.items():
        d = pd.read_csv(zipfile.ZipFile(E / zp).open(fn), sep=" ", header=None, names=["pid", "s"])
        assert (d.pid.values == t.pair_id.values).all(); S[n] = -d.s.values        # higher = same
    tk = f"{k[0]}_{k[1]}"; known, y, fc = L[f"{tk}_known"], L[f"{tk}_lab"], L[f"{tk}_fc"]
    base = S[BEST_NOW[cell]]
    for name, (kind, mem) in CANDS.items():
        s = fused(kind, mem, S)
        pe = eer_from_scores(s[known], y[known])
        rows.append(dict(cell=cell, cand=name, pseudo=round(pe, 2),
                         cb=CB[name][i] if name in CB else np.nan,
                         p_beats_best=round(boot_win(fc[known], y[known], s[known], base[known]), 3)
                         if name != BEST_NOW[cell] else np.nan))
        out_scores[(cell, name)] = s
R = pd.DataFrame(rows); R.to_csv("fuse_candidates.csv", index=False)

pick = {}
for cell, g in R.groupby("cell", sort=False):
    ref = g[g.cand == BEST_NOW[cell]].pseudo.iloc[0]
    g = g.assign(gain=(ref - g.pseudo).round(2)).sort_values("pseudo")
    print(f"\n{cell}   (current best on CB: {BEST_NOW[cell]})\n" + g.to_string(index=False))
    win = g[(g.gain >= 1.0) & (g.p_beats_best >= 0.9)]
    pick[cell] = win.cand.iloc[0] if len(win) else BEST_NOW[cell]
    print("-> keep" if not len(win) else "-> REPLACE", pick[cell])
pd.Series(pick).to_json("fuse_pick.json", indent=1)
