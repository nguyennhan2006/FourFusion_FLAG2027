"""LANG-01 dev check (one pre-registered variant, chosen from run.py BEFORE this script ran):
v2_complete direction, k=1, alpha=1.0 (cross-language Urdu person +1.73 [4/5], in-language Hindi +0.61 [4/5],
English ng +0.03 / g -0.50).  Direction learned from ALL v2 persons with >=3 Hindi and >=3 English clips.
Paired on dev: s007 recipe (EXP-007, n=5, CCA 0.25) trained twice by the SAME code, with and without the
projection; compared at SAMPLE level with SEL-01b pseudo-labels (reliable for non-clustered systems).
Matrices are saved so the clustered SET-02 assembly can reuse them."""
import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(int(os.environ.get("LANG01_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext")
import flag_v2 as V
sys.path.insert(0, str(R / "Experiment/SEL-01_selector"))
from pseudo_eval import evaluate
HERE = Path(__file__).parent; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
REC = dict(head="mlp", drop=0.3, emb=128, epochs=40)

store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
mv, sv = store.Xv.mean(0), store.Xv.std(0) + 1e-6
D = store.ext_source("v2_complete"); vm = D["vm"]; key = lambda x: D["key_of"].get(x, f"v2_complete:{x}")
Z = (D["V"] - mv) / sv; diffs = []
for p, g in vm.groupby(vm.spk.map(key)):
    a, b = g[g.lang == "hindi"].index.values, g[g.lang == "english"].index.values
    if len(a) >= 3 and len(b) >= 3:
        diffs.append(Z[a].mean(0) - Z[b].mean(0))
P = (np.mean(diffs, 0) / np.linalg.norm(np.mean(diffs, 0)))[:, None]
print("direction from", len(diffs), "v2 persons", flush=True)
project = lambda X: ((((X - mv) / sv) - ((X - mv) / sv) @ P @ P.T) * sv + mv).astype(np.float32)

cells = list(V.CELLS)
scores = {}
for name in ("none", "lang"):
    f = OUT / f"s007{'lang' if name == 'lang' else 'ctrl'}_matrix.npz"
    store._c.clear()
    if name == "lang":
        store._c[("v", "given", "train")] = project(store.Xv)
        for k in cells:
            store._c[("v", "given", "/".join(k))] = project(store.dev[k][1])
    V.dev_scores(store, REC, n_models=5, cca_w=0.25, matrix_path=f)
    M = np.load(f)
    scores[name] = {k: np.diag(M["/".join(k)].astype(np.float64)) for k in cells}
    print(name); print(evaluate(scores[name], n_boot=100).to_string(index=False), flush=True)
