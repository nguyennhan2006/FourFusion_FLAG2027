"""(1) How concentrated are errors per person?  (2) Does per-item bias removal on the FULL score matrix help?
    CSLS-style: s'(f,v) = s(f,v) - a/2 * [ r_f + r_v ],  r_f = mean of face f's top-k scores over all voices of the file,
                r_v = mean of voice v's top-k scores over all faces.  Uses the media set only, never the trial list.
Judged at fusion level exactly as submitted, with BOTH pseudo-label sets (btc labels are independent of ArcFace/ECAPA)."""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "Experiment/EDA-000_raw")); from eda_utils import load_dev, eer_from_scores
LAB = {l: np.load(R / f"Experiment/SEL-01_selector/pseudo_labels_{l}.npz") for l in ["arc", "btc"]}
M = {n: np.load(R / f"Experiment/GRAPH-01/mats/{n}_matrix.npz") for n in ["s007", "s010B", "r2mix"]}
rank = lambda s: rankdata(s) / (len(s) + 1)
FN_ = {"no_gender/English": "no_gender/sub_score_v4_English_heard.txt", "gender/English": "gender/sub_score_v4_English_heard.txt",
       "no_gender/Bangla": "no_gender/sub_score_v4_Bangla_unheard.txt", "gender/Bangla": "gender/sub_score_v4_Bangla_unheard.txt"}
print("(1) error concentration (FUSE-03, arc labels): share of errors from the worst 3 people / share of their trials")
for c, fn in FN_.items():
    tk = c.replace("/", "_"); kn, lab, fc = LAB["arc"][tk + "_known"], LAB["arc"][tk + "_lab"], LAB["arc"][tk + "_fc"]
    s = -pd.read_csv(zipfile.ZipFile(R / "Experiment/FUSE-03/out/submission_FUSE03.zip").open(fn), sep=" ", header=None)[1].values
    e = eer_from_scores(s[kn], lab[kn]); thr = np.quantile(s[kn & (lab == 0)], 1 - e / 100)
    err = kn & (((lab == 1) & (s < thr)) | ((lab == 0) & (s >= thr)))
    per = pd.DataFrame(dict(p=fc[kn], err=err[kn])).groupby("p").err.agg(["sum", "size"])
    per["rate"] = per["sum"] / per["size"]; top = per.sort_values("sum", ascending=False).head(3)
    print(f"  {c:18s} EER {e:.2f} | {len(per)} people | worst-3 people: {top['sum'].sum() / per['sum'].sum():.0%} of errors from "
          f"{top['size'].sum() / per['size'].sum():.0%} of trials | per-person error rate p10/p50/p90 "
          f"{per.rate.quantile(.1):.2f}/{per.rate.quantile(.5):.2f}/{per.rate.quantile(.9):.2f}")

PLAN = {"no_gender/English": ["s007", "s010B"], "gender/English": ["s007", "s010B"], "no_gender/Bangla": ["r2mix"], "gender/Bangla": ["r2mix"]}


def csls(Mx, a, k=10):
    rf = np.sort(Mx, 1)[:, -k:].mean(1); rv = np.sort(Mx, 0)[-k:, :].mean(0)
    return np.diag(Mx) - a / 2 * (rf + rv)


print("\n(2) CSLS hub correction on the full matrix (local recipes), gain in pseudo-EER vs a=0")
rows = []
for c, mem in PLAN.items():
    tk = c.replace("/", "_")
    for a in (0.0, 0.25, 0.5, 1.0):
        for k in (5, 20):
            if a == 0 and k == 20: continue
            s = np.mean([rank(csls(M[m][c].astype(np.float32), a, k)) for m in mem], 0)
            r = dict(cell=c, a=a, k=k)
            for lb, L in LAB.items():
                kn, y = L[tk + "_known"], L[tk + "_lab"]; r[lb] = round(eer_from_scores(s[kn], y[kn]), 2)
            rows.append(r)
T = pd.DataFrame(rows)
for c, g in T.groupby("cell", sort=False):
    b = g[g.a == 0].iloc[0]
    print(f"  {c:18s} base arc {b.arc} btc {b.btc} | " + "  ".join(f"a={r.a},k={r.k}: {b.arc - r.arc:+.2f}/{b.btc - r.btc:+.2f}" for r in g[g.a > 0].itertuples()))
T.to_csv("csls.csv", index=False)
