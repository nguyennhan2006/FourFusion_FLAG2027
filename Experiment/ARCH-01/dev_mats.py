"""Full dev face x voice matrices for ridge-CCA members (fit on all 70 train speakers, no training), same
construction as run.py's CCABase: reg 10, PCA 256 face / 192 voice, each dev file centred (own mean removed,
train mean restored), variates standardised with train stats, cosine, z-scored over the matrix.
-> out/cca<k>_matrix.npz, keys like dev_scores' matrix_path ("no_gender/English", ...)."""
import sys, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle"))
import flag_v2 as V
from flag_lib import RidgeCCA
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
TF, TV = store.Xf, store.Xv
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
for k in (8, 16, 32, 64):
    m = RidgeCCA(k=k, reg=10.0, pca_x=256, pca_y=192).fit(TF, TV)
    raw = lambda X, Y: ((((X - m.mx) / m.sx) @ m.Px) @ m.Wx, (((Y - m.my) / m.sy) @ m.Py) @ m.Wy)
    a, b = raw(TF, TV); sa, sb = (a.mean(0), a.std(0) + 1e-6), (b.mean(0), b.std(0) + 1e-6)
    mats = {}
    for c in V.CELLS:
        Af, Av = store.face("vgg", "/".join(c)), store.voice("given", "/".join(c))
        x, y = raw(V.centre(Af, TF.mean(0)), V.centre(Av, TV.mean(0)))
        x, y = V.l2n((x - sa[0]) / sa[1]), V.l2n((y - sb[0]) / sb[1])
        mats["/".join(c)] = zm(x @ y.T).astype(np.float16)
    np.savez_compressed(OUT / f"cca{k}_matrix.npz", **mats)
    print("cca", k, {c: v.shape for c, v in mats.items()}, flush=True)
