"""EXP-012 — a label-free proxy for EER, so model selection can finally see Bangla.

The problem, stated precisely: our internal validation only contains English, and it has now steered
us wrong on 4 out of 4 decisions that touched Bangla (003b -> 003c -> 007 -> 010B). The last one was
the worst: internal said ECAPA-6144 was 1.31 better, CodaBench said 0.81 worse.

The opportunity: every submitted zip is a set of scores over the dev trials, and CodaBench has told
us the true EER of each cell. That gives ground truth for a regression test - 8 systems x 4 cells -
without needing any trial labels.

What a usable proxy must exploit: each dev file is ~50% target trials (the evaluation plan fixes the
target counts, e.g. 504/1008). So a good system produces a clearly BIMODAL score distribution, and a
bad one produces a single blob. We can measure that without knowing which trial is which.

Proxies tested here (all computed from the submitted scores alone):
    gmm_sep    : 2-component GMM, |mu1 - mu2| / pooled sigma
    bimodality : Sarle's coefficient (skew^2 + 1) / kurtosis
    dip        : Hartigan-style spread of the sorted-score gaps
    kurtosis   : a bimodal mixture is platykurtic
    half_gap   : mean of the top 50% minus mean of the bottom 50%, over the spread
The winner is whichever tracks the true EER best ACROSS cells, especially on the Bangla cells.

Usage: python run.py
"""
import sys, zipfile, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)

CELLS = {"ng_En": "no_gender/sub_score_v4_English_heard.txt",
         "ng_Bn": "no_gender/sub_score_v4_Bangla_unheard.txt",
         "g_En": "gender/sub_score_v4_English_heard.txt",
         "g_Bn": "gender/sub_score_v4_Bangla_unheard.txt"}

# Every submission we have a zip for, with the per-cell EER CodaBench reported.
SYSTEMS = {
    "000d CCA k32":   ("EXP-000d_cca_linear/out/submission_EXP000d.zip",            [33.73, 34.00, 34.42, 41.01]),
    "000e CCA k4":    ("EXP-000d_cca_linear/out/submission_EXP000e_k4.zip",         [28.97, 29.02, 37.47, 42.37]),
    "000f hybrid":    ("EXP-000d_cca_linear/out/submission_EXP000f_hybrid.zip",     [28.97, 29.02, 34.42, 41.01]),
    "003b +asnorm":   ("EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_asnorm.zip", [25.79, 28.88, 33.81, 39.37]),
    "003c raw":       ("EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip", [25.60, 27.88, 34.01, 37.87]),
    "007 n10_w.25":   ("EXP-007_fusion/out/submission_EXP007_n10_w0.25.zip",        [25.60, 29.02, 30.96, 38.01]),
    "010B ecapa6144": ("EXP-010_feat/out/submission_newfeat.zip",                   [23.81, 30.87, 32.38, 39.78]),
    "004 CORAL":      ("EXP-004_domain/out/submission_EXP004_align-coral.zip",      [37.30, 37.84, 43.99, 46.32]),
}


def proxies(s):
    """All computed from the submitted scores only - no labels, no knowledge of which trial is target."""
    s = np.asarray(s, dtype=np.float64)
    s = (s - s.mean()) / (s.std() + 1e-12)
    out = {}

    # 2-component GMM separation
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, n_init=3, random_state=0).fit(s.reshape(-1, 1))
    mu = g.means_.ravel(); sd = np.sqrt(g.covariances_.ravel()); w = g.weights_
    out["gmm_sep"] = float(abs(mu[0] - mu[1]) / np.sqrt((w * sd ** 2).sum() + 1e-12))
    out["gmm_balance"] = float(min(w) / max(w))          # target rate is ~50%, so balance is expected

    # Sarle's bimodality coefficient: > 5/9 suggests bimodal
    n = len(s)
    sk, ku = stats.skew(s), stats.kurtosis(s, fisher=True)
    out["bimodality"] = float((sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
    out["kurtosis"] = float(-ku)                          # platykurtic (negative ku) = more bimodal

    # split-half gap: with a ~50% target rate the top and bottom halves should separate
    q = np.sort(s)
    out["half_gap"] = float(q[n // 2:].mean() - q[:n // 2].mean())

    # largest gap in the middle region of the sorted scores (a valley between two modes)
    lo, hi = int(0.2 * n), int(0.8 * n)
    out["dip"] = float(np.max(np.diff(q[lo:hi])) * n)
    return out


if __name__ == "__main__":
    rows = []
    for name, (zp, eers) in SYSTEMS.items():
        p = ROOT / zp
        if not p.exists():
            print("MISSING", zp); continue
        z = zipfile.ZipFile(p)
        for (cell, fn), eer in zip(CELLS.items(), eers):
            d = pd.read_csv(z.open(fn), sep=" ", header=None, names=["pid", "s"])
            rows.append(dict(system=name, cell=cell, lang="Bn" if cell.endswith("Bn") else "En",
                             eer=eer, **proxies(d.s.values)))
        print(f"{name:16s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "proxies.csv", index=False)

    pcols = [c for c in df.columns if c not in ("system", "cell", "lang", "eer")]
    print("\n=== correlation with true EER (lower EER should mean a more separated distribution) ===")
    res = []
    for c in pcols:
        res.append(dict(proxy=c,
                        all_pearson=df.eer.corr(df[c]),
                        all_spearman=df.eer.corr(df[c], method="spearman"),
                        Bn_spearman=df[df.lang == "Bn"].eer.corr(df[df.lang == "Bn"][c], method="spearman"),
                        En_spearman=df[df.lang == "En"].eer.corr(df[df.lang == "En"][c], method="spearman"),
                        within_cell=np.mean([df[df.cell == k].eer.corr(df[df.cell == k][c], method="spearman")
                                             for k in CELLS])))
    r = pd.DataFrame(res).round(3).sort_values("within_cell")
    print(r.to_string(index=False))
    r.to_csv(OUT / "correlations.csv", index=False)

    best = r.iloc[0].proxy
    print(f"\nbest proxy by within-cell rank correlation: {best}")
    print("\n=== per-cell ranking by that proxy vs the truth ===")
    for cell in CELLS:
        sub = df[df.cell == cell].sort_values(best)
        truth = sub.sort_values("eer").system.tolist()
        pred = sub.system.tolist()
        print(f"\n{cell}:")
        print(f"  by proxy : {pred}")
        print(f"  by truth : {truth}")
