# ==== visualisation helpers (embedded in the notebook; one cell) ====
# %% công cụ visualize
import io, json, zipfile
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import spearmanr

plt.rcParams.update({"figure.dpi": 80, "font.size": 9, "axes.titlesize": 10})
FILES = [("no_gender", "English"), ("no_gender", "Bangla"), ("gender", "English"), ("gender", "Bangla")]   # index 0..3 everywhere
LAB = ["không ràng buộc giới · English", "không ràng buộc giới · Bangla", "cùng giới · English", "cùng giới · Bangla"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
EVAL_1618 = [13.27, 11.72, 20.19, 19.55]          # CodaBench Evaluation, display only
EVAL_1699 = [13.78, 11.97, 22.12, 20.11]


def zname(i):
    p, l = FILES[i]
    return f"{p}/sub_score_v4_{l}_{'heard' if l == 'English' else 'unheard'}.txt"


def zip_logits(path):
    """Submission zip stores -logit; returns logit (high = same person) per pair id for the 4 files."""
    with zipfile.ZipFile(path) as z:
        return [-pd.read_csv(io.BytesIO(z.read(zname(i))), sep=" ", header=None, names=["pid", "s"], dtype={"pid": str}).set_index("pid").s
                for i in range(4)]


def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def pca_fit(X):
    X = unit(X)
    mu = X.mean(0)
    _, S_, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return (lambda Y: (unit(Y) - mu) @ Vt[:2].T), S_[:2] ** 2 / np.sum(S_ ** 2)


def eer(pos, neg):
    s = np.r_[pos, neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))][np.argsort(-s)]
    tp, fp = np.cumsum(y) / y.sum(), np.cumsum(1 - y) / (1 - y).sum()
    i = np.argmin(np.abs((1 - tp) - fp))
    return 100 * ((1 - tp[i]) + fp[i]) / 2


def eer_scores(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    return eer(s[y == 1], s[y == 0])


def identity_pairs(ids, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)
    multi = [g for g in (np.flatnonzero(ids == k) for k in np.unique(ids)) if len(g) > 1]
    P = np.array([rng.choice(multi[j], 2, replace=False) for j in rng.integers(0, len(multi), n)])
    a, b = rng.integers(0, len(ids), 3 * n), rng.integers(0, len(ids), 3 * n)
    m = ids[a] != ids[b]
    return P, np.c_[a[m], b[m]][:n]


def identity_panel(ax, X, ids, what):
    """cos of same-identity vs different-identity pairs inside one modality (how well the encoder separates people)."""
    Xu = unit(X)
    P, N = identity_pairs(ids)
    ps, ns = (Xu[P[:, 0]] * Xu[P[:, 1]]).sum(1), (Xu[N[:, 0]] * Xu[N[:, 1]]).sum(1)
    ax.hist(ns, 80, density=True, alpha=.6, label="khác người")
    ax.hist(ps, 80, density=True, alpha=.6, label="cùng người")
    ax.set_title(f"cos cùng {what}: EER nhận dạng {eer(ps, ns):.1f} % ({len(np.unique(ids))} người)")
    ax.legend()


def pca_panel(ax, groups):
    """groups: list of (label, array, colour)."""
    proj, var = pca_fit(np.concatenate([g[1] for g in groups]))
    for lab, X, c in groups:
        q = proj(X)
        ax.scatter(q[:, 0], q[:, 1], s=2 if c == "0.75" else 3, c=c, label=f"{lab} ({len(X)})")
    ax.set_title(f"PCA 2 chiều ({100 * var.sum():.0f} % phương sai)")
    ax.legend(fontsize=7, markerscale=3)


def train_index(d, col):
    """index_train.txt of an extraction run: pair id, label, voice path, face path, ... -> (rows of unique files, identity per row)."""
    t = pd.read_csv(Path(d) / "index_train.txt", sep=" ", header=None)
    paths = t[2 if col == "voice" else 3].astype(str)
    keep = ~paths.duplicated()
    return np.flatnonzero(keep.values), paths[keep].str.split("/").str[2].values


def embedding_figure(d, key, col, title):
    """Arrays <key>_<protocol>_<language>.npy + <key>_train.npy of one extraction run: PCA, identity separation, vector norms."""
    d = Path(d)
    E = [np.load(d / f"{key}_{p}_{l}.npy") for p, l in FILES]
    T = np.load(d / f"{key}_train.npy")
    rows, ids = train_index(d, col)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    pca_panel(ax[0], [(f"người train", T, "0.75")] + [(LAB[i], E[i], COL[i]) for i in range(4)])
    identity_panel(ax[1], T[rows], ids, "mặt" if col == "face" else "giọng")
    for i in range(4):
        ax[2].hist(np.linalg.norm(E[i], axis=1), 50, density=True, histtype="step", color=COL[i], label=LAB[i])
    ax[2].hist(np.linalg.norm(T, axis=1), 50, density=True, histtype="step", color="0.4", label="người train")
    ax[2].set_title("độ dài vector trước chuẩn hoá (lệch miền)")
    ax[2].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return E, T


def logit_hist(series_by_file, title, ax=None):
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    for i, s in enumerate(series_by_file):
        ax.hist(np.asarray(s, float), 50, density=True, histtype="step", lw=1.4, color=COL[i], label=f"{LAB[i]} ({len(s)})")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return ax


def show_text(p, n=25, grep=None):
    p = Path(p)
    if not p.exists():
        print(f"(không có {p.name})")
        return
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if grep:
        lines = [l for l in lines if re.search(grep, l)]
    print(f"--- {p.name} ({len(lines)} dòng, in {min(n, len(lines))})")
    print("\n".join(lines[:n]))


def show_json(p, keys=None, width=1500):
    p = Path(p)
    if not p.exists():
        print(f"(không có {p.name})")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    if keys:
        d = {k: d.get(k) for k in keys}
    print(f"--- {p.name}: " + json.dumps(d, ensure_ascii=False)[:width])


def curve(ax, ys, title, xlabel="bước", labels=None):
    for k, y in enumerate(ys):
        ax.plot(np.asarray(y, float), lw=1, label=None if labels is None else labels[k])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if labels:
        ax.legend(fontsize=7)


def need(*aids):
    """Paths of the artifacts a visualisation cell needs; None (and a message) when one is missing in this session."""
    ps = [out(a) for a in aids]
    if any(p is None for p in ps):
        print("chưa có output để vẽ (stage chưa chạy trong phiên này và chưa được gắn vào input):",
              [a for a, p in zip(aids, ps) if p is None])
        return None
    return ps if len(ps) > 1 else ps[0]
