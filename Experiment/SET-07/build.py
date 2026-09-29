"""SET-07 = SET-02 with the s007 member retrained as EXT-03 arm B3P:
v4 (all 70) + MAV-Celeb v1 + v2 (full Kaggle features, cap 150 rows/person, ids merged by name, 2 overlapping persons
excluded) + SetProto (pk sampler, w_proto 0.5), 15 epochs, n=5 models, CCA w=0.25 -- same code path as EXT-03
(train_one(rows=...), centring to the mean of the training rows). s010B and r2mix matrices, clusters, b=0.99 and the
per-cell rank fusion are exactly SET-02's.  Evidence (true labels, 5 paired splits, EXT-03): v4 English cluster-level
ng +3.45 [4/5], g +2.83 [3/5]; unseen-language transfer Urdu->Hindi +3.13, Hindi->Urdu +4.45."""
import os, sys, zipfile, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import rankdata
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(int(os.environ.get("SET07_THREADS", "4")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext_full")
import flag_v2 as V
HERE = Path(__file__).parent; (HERE / "out").mkdir(exist_ok=True)
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
EXCLUDE = tuple(pd.read_csv(HERE / "overlap.csv").query("overlap").person)
print("excluded external persons:", EXCLUDE)
CFG = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=15, sampler="pk", pk_P=32, pk_K=8, w_proto=0.5)
N_MODELS, CCA_W = 5, 0.25

mat_file = HERE / "out/s007ext_matrix.npz"
if not mat_file.exists():
    rng = np.random.RandomState(0)
    TF, TV, TS = store.Xf, store.Xv, store.spk
    for src in ("v1_complete", "v2_complete"):
        D = store.ext_source(src); store.gmap.update(D["gen_of"])
        key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
        ids = [x for x in D["vm"].spk.unique() if key(x) not in EXCLUDE]
        EF, EV, ES = V.pair_ext(D, src, 150, rng, ids=ids)
        TF, TV, TS = np.concatenate([TF, EF]), np.concatenate([TV, EV]), np.concatenate([TS, ES])
    print("training rows", len(TS), "people", len(np.unique(TS)), flush=True)
    nets = [V.train_one(store, CFG, None, seed=1 + 100 * i, rows=(TF, TV, TS)) for i in range(N_MODELS)]
    mats = {}
    zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
    for k in V.CELLS:
        c = "/".join(k); Af, Av = store.face("vgg", c), store.voice("given", c)
        embs = [V.embed(net, prep, Af, Av, prep.f(TF).mean(0), prep.v(TV).mean(0)) for net, prep in nets]
        M = np.mean([zm(Ef @ Ev.T) for Ef, Ev in embs], 0)
        cca = V.cca_proj(store.Xf, store.Xv, Af, Av)
        mats[c] = ((1 - CCA_W) * M + CCA_W * zm(cca[0] @ cca[1].T)).astype(np.float16)
        print(c, "matrix", mats[c].shape, flush=True)
    np.savez_compressed(mat_file, **mats)

# ---- assemble exactly like SET-02, with s007 -> s007ext
C = np.load(R / "BEST/v4_set02/_dryrun/clusters.npz")
M = {"s007ext": np.load(mat_file), "s010B": np.load(R / "Experiment/GRAPH-01/mats/s010B_matrix.npz"),
     "r2mix": np.load(R / "Experiment/GRAPH-01/mats/r2mix_matrix.npz"), "s007": np.load(R / "Experiment/GRAPH-01/mats/s007_matrix.npz")}
rank = lambda s: rankdata(s) / (len(s) + 1)


def block(Mx, cf, cv, b=0.99):
    Mx = Mx.astype(np.float64)
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * np.diag(Mx) + b * (Af @ Mx @ Av.T)[cf, cv]


CELLS = {"English": ["s007ext", "s010B"], "Bangla": ["r2mix", "s007ext"]}
scores = {}
for k in V.CELLS:
    c = "/".join(k); tk = f"{k[0]}_{k[1]}"
    scores[k] = np.mean([rank(block(M[m][c], C[tk + "_face"], C[tk + "_voice"])) for m in CELLS[k[1]]], 0)
zp = HERE / "out/submission_SET07.zip"
V.write_zip(store, zp, scores); V.check_zip(store, zp)
ref = R / "Experiment/SET-01/out/submission_SET02.zip"
for k, fn in V.NAMES.items():
    a = pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None)[1]; b = pd.read_csv(zipfile.ZipFile(ref).open(fn), sep=" ", header=None)[1]
    print(f"{fn:45s} rank-corr with SET-02 {a.corr(b, method='spearman'):.3f}")
