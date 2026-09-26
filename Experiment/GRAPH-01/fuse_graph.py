"""Fusion-level test of graph smoothing, as it would be submitted:
  English cells : rank(g(s007)) + rank(g(s010B))       vs rank(s007) + rank(s010B)   (local n=5 models)
  ng/Bn         : rank(003c zip) + rank(g(r2mix))       vs rank(003c) + rank(r2mix)
  g/Bn untouched (graph hurt 007/010B there).
Graph = arc (ArcFace + own ECAPA-192). Judged by btc labels (independent) and arc labels (shares features -> optimistic)."""
import numpy as np, pandas as pd, zipfile
from scipy.stats import rankdata
from graph import feats, knn, smooth_diag, LAB, eer_from_scores, load_dev, E
rank = lambda s: rankdata(s) / (len(s) + 1)
M = {n: np.load(f"mats/{n}_matrix.npz") for n in ["s007", "s010B", "r2mix"]}
PLAN = {("no_gender", "English"): ["s007", "s010B"], ("gender", "English"): ["s007", "s010B"], ("no_gender", "Bangla"): ["003c", "r2mix"]}
rows = []
for k, mem in PLAN.items():
    c = "/".join(k); F, V = feats("arc", k)
    zfn = {"no_gender/Bangla": "no_gender/sub_score_v4_Bangla_unheard.txt"}.get(c)
    base_sc = {}
    for m in mem:
        if m == "003c":
            d = pd.read_csv(zipfile.ZipFile(E / "EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip").open(zfn), sep=" ", header=None)
            base_sc[m] = -d[1].values
        else:
            base_sc[m] = np.diag(M[m][c].astype(np.float32))
    for kk in (3, 5, 10):
        Af, Av = knn(F, kk), knn(V, kk)
        for lam in (0.0, 0.1, 0.2, 0.3):
            parts = [rank(base_sc[m]) if (m == "003c" or lam == 0) else rank(smooth_diag(M[m][c].astype(np.float32), Af, Av, lam)) for m in mem]
            s = np.mean(parts, 0); r = dict(cell=c, k=kk, lam=lam)
            for lb, L in LAB.items():
                tk = f"{k[0]}_{k[1]}"; kn, y = L[tk + "_known"], L[tk + "_lab"]
                r[lb] = round(eer_from_scores(s[kn], y[kn]), 2)
            rows.append(r)
R = pd.DataFrame(rows)
for c, g in R.groupby("cell", sort=False):
    b = g[g.lam == 0].iloc[0]
    g = g[g.lam > 0].assign(gain_btc=lambda d: (b.btc - d.btc).round(2), gain_arc=lambda d: (b.arc - d.arc).round(2))
    print(f"\n{c}: no graph btc {b.btc} arc {b.arc}\n" + g.pivot(index="k", columns="lam", values="gain_btc").to_string()
          + "\n  (arc labels)\n" + g.pivot(index="k", columns="lam", values="gain_arc").to_string())
R.to_csv("fuse_graph.csv", index=False)
