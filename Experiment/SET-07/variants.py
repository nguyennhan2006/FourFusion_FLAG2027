"""SET-07 variants, pre-registered before looking at their pseudo-EERs (3 systems, SEL-01b allowed use: choosing
among a few pre-registered systems).  Same clusters, b and rank fusion as SET-02; only the member lists change.
  07a  replace s007 by s007ext everywhere          (= build.py)
  07b  add s007ext as a third member everywhere
  07c  English as SET-02, Bangla = r2mix + s007 + s007ext"""
import sys, numpy as np
from pathlib import Path
from scipy.stats import rankdata
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "Experiment/SEL-01_selector"))
from pseudo_eval import evaluate, CELLS as K
HERE = Path(__file__).parent
C = np.load(R / "BEST/v4_set02/_dryrun/clusters.npz")
G = R / "Experiment/GRAPH-01/mats"
M = {"s007ext": np.load(HERE / "out/s007ext_matrix.npz"), "s007": np.load(G / "s007_matrix.npz"),
     "s010B": np.load(G / "s010B_matrix.npz"), "r2mix": np.load(G / "r2mix_matrix.npz")}
rank = lambda s: rankdata(s) / (len(s) + 1)


def block(Mx, cf, cv, b=0.99):
    Mx = Mx.astype(np.float64)
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * np.diag(Mx) + b * (Af @ Mx @ Av.T)[cf, cv]


SYSTEMS = {"SET-02": {"English": ["s007", "s010B"], "Bangla": ["r2mix", "s007"]},
           "07a": {"English": ["s007ext", "s010B"], "Bangla": ["r2mix", "s007ext"]},
           "07b": {"English": ["s007", "s010B", "s007ext"], "Bangla": ["r2mix", "s007", "s007ext"]},
           "07c": {"English": ["s007", "s010B"], "Bangla": ["r2mix", "s007", "s007ext"]}}
rows = {}
for name, cells in SYSTEMS.items():
    sc = {}
    for k in K:
        c = "/".join(k); tk = f"{k[0]}_{k[1]}"
        sc[k] = np.mean([rank(block(M[m][c], C[tk + "_face"], C[tk + "_voice"])) for m in cells[k[1]]], 0)
    rows[name] = evaluate(sc, n_boot=50).set_index("cell").pseudo_eer
import pandas as pd
print(pd.DataFrame(rows).to_string())
