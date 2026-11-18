"""SAFE-04 step 2: fusion analysis on the stream tables of lab_streams.py (true-label validation, level 0).

    python lab_fuse.py            -> prints per-stream EER and paired gains of candidate recipes vs XALL_FIXW
"""
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

SP = Path(os.environ.get("SAFE04_SCRATCH", r"C:/Users/ASUS/AppData/Local/Temp/claude/d--Sinh-vi-n-CNhan-FLAG-2027-FourFusion/969f2c71-effc-43ec-a610-c253be9d1ea8/scratchpad"))
FIXW = np.array([0.298, 0.405, 0.948, 0.327])


def eer(s, y):
    s, y = np.asarray(s, np.float64), np.asarray(y)
    ok = np.isfinite(s)
    if ok.sum() < 40 or len(np.unique(y[ok])) < 2:
        return np.nan
    s, y = s[ok], y[ok]
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; d = fnr - fpr
    i = np.nanargmin(np.abs(d))
    if d[i] == 0 or len(fpr) < 2:
        return 100 * (fpr[i] + fnr[i]) / 2
    j = i - 1 if (i > 0 and d[i - 1] * d[i] < 0) else min(i + 1, len(d) - 1)
    if d[j] * d[i] >= 0:
        return 100 * (fpr[i] + fnr[i]) / 2
    t = d[i] / (d[i] - d[j])
    return 100 * float(fpr[i] + t * (fpr[j] - fpr[i]))


def load():
    D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SP / "out/safe04/streams_fold*.parquet")))], ignore_index=True)
    D["cell"] = D.data + " " + D.protocol
    return D


def fixw(D, b="XALL", ib="ib"):
    return np.stack([D[f"{b}_deep0"], D[f"{b}_cca"], D[ib] if ib != "ib" else D[f"{b}_ib"], D[f"{b}_age"]], 1) @ FIXW


def table(D, recipes, cover=None):
    """recipes: {name: fn(D) -> score}; EER per (fold, cell); restrict to rows with finite values of `cover` streams."""
    rows = []
    for (s, cell), g in D.groupby(["fold", "cell"]):
        if cover is not None:
            g = g[np.isfinite(g[cover]).all(1)]
        if len(g) < 40 or g.y.nunique() < 2:
            continue
        r = dict(fold=s, cell=cell, n=len(g))
        for name, fn in recipes.items():
            r[name] = eer(fn(g), g.y.values)
        rows.append(r)
    return pd.DataFrame(rows)


def paired(T, ref, names):
    out = []
    for cell, g in T.groupby("cell"):
        r = dict(cell=cell, ref=round(g[ref].mean(), 2))
        for n in names:
            d = g[ref] - g[n]
            r[n] = f"{d.mean():+.2f} ({(d > 0).sum()}/{len(d)})"
        out.append(r)
    return pd.DataFrame(out).set_index("cell")


if __name__ == "__main__":
    D = load()
    print("folds:", sorted(D.fold.unique()), "| trials", len(D))
    single = {k: (lambda g, k=k: g[k].values) for k in
              ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "XALL_prod0", "CTRL_deep0", "CTRL_prod0", "iba2", "ibal", "gen", "acl", "lb"]}
    T = table(D, single)
    print("\n== single streams, EER mean over folds (all rows; acl / lb NaN rows -> see covered table)")
    print(T.groupby("cell").mean(numeric_only=True).drop(columns=["fold"]).round(2).to_string())
    Tc = table(D, single, cover=["acl", "lb"])
    print("\n== single streams on rows covered by AudioCLIP / LanguageBind")
    print(Tc.groupby("cell").mean(numeric_only=True).drop(columns=["fold"]).round(2).to_string())
