"""Shared helpers for FLAG EDA + EXP-001 internal validation.

Every EDA layer and every later experiment must build trials with the SAME
functions here so numbers are comparable across notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

DATA = Path(__file__).resolve().parent / "data"
TRAIN_FACE = DATA / "train_set/features/faces/train_English_faces.csv"
TRAIN_VOICE = DATA / "train_set/features/voices/train_English_voices.csv"
TRAIN_TXT = DATA / "train_set/train_set/train_English.txt"
META = Path(__file__).resolve().parents[2] / "Input/meta_file_train_set.csv"
DEV = DATA / "dev_set"

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "text": "#0b0b0b", "muted": "#52514e", "grid": "#e5e4df"}


# ------------------------------------------------------------------ loading
def load_train():
    face = pd.read_csv(TRAIN_FACE, header=None)
    voice = pd.read_csv(TRAIN_VOICE, header=None)
    yf = face.iloc[:, -1].astype(str).values
    yv = voice.iloc[:, -1].astype(str).values
    Xf = face.iloc[:, :-1].values.astype(np.float32)
    Xv = voice.iloc[:, :-1].values.astype(np.float32)
    txt = pd.read_csv(TRAIN_TXT, sep=" ", header=None,
                      names=["pair_id", "label", "voice", "face", "spk", "spk_int"])
    # CSV label column is the integer speaker id (== txt.spk_int), not a video index
    assert (yf == yv).all() and (yf.astype(int) == txt.spk_int.values).all()
    spk = txt.spk.values.astype(str)
    meta = pd.read_csv(META, header=None, names=["spk", "gender"])
    gmap = dict(zip(meta.spk, meta.gender))
    return Xf, Xv, spk, spk, txt, gmap


def load_dev(protocol: str, lang: str):
    """protocol in {no_gender, gender}; lang in {English, Bangla}."""
    d = DEV / protocol
    Xf = pd.read_csv(d / f"features/{lang}_test_faces.csv", header=None).values.astype(np.float32)
    Xv = pd.read_csv(d / f"features/{lang}_test_voices.csv", header=None).values.astype(np.float32)
    txt = pd.read_csv(d / f"{lang}_test.txt", sep=" ", header=None, names=["pair_id", "voice", "face"])
    return Xf, Xv, txt


# ------------------------------------------------------------------ splits
def speaker_split(speakers, seed: int, n_val: int = 14):
    """Speaker-disjoint split, identical convention to EXP-001 (56/14)."""
    uniq = np.array(sorted(set(speakers)))
    rng = np.random.RandomState(seed)
    val = set(rng.choice(uniq, n_val, replace=False))
    is_val = np.array([s in val for s in speakers])
    return ~is_val, is_val


# ------------------------------------------------------------------ metrics
def eer_from_scores(scores, labels):
    """scores: higher = more likely same. Returns EER in %."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


def cohens_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


# ------------------------------------------------------------------ trials
def build_trials(spk, gender_of, seed: int, n_pos: int = 3000, n_neg: int = 3000,
                 same_gender: bool = False, group=None):
    """Cross-modal trials (face index i, voice index j, label).

    positive: same speaker, i != j (and different `group` if given, e.g. voice clip id).
    negative: different speaker; if same_gender, also same gender.
    Mirrors the dev protocol: no_gender -> same_gender=False, gender -> True.
    """
    rng = np.random.RandomState(seed)
    spk = np.asarray(spk)
    idx_by_spk = {s: np.where(spk == s)[0] for s in np.unique(spk)}
    g = np.array([gender_of[s] for s in spk])
    n = len(spk)
    pos = []
    while len(pos) < n_pos:
        i = rng.randint(n)
        cand = idx_by_spk[spk[i]]
        if len(cand) < 2:
            continue
        j = rng.choice(cand)
        if j == i:
            continue
        if group is not None and group[i] == group[j]:
            continue  # identical feature vector (same audio clip) -> trivial positive
        pos.append((i, j))
    neg = []
    while len(neg) < n_neg:
        i, j = rng.randint(n), rng.randint(n)
        if spk[i] == spk[j]:
            continue
        if same_gender and g[i] != g[j]:
            continue
        neg.append((i, j))
    fi = np.array([p[0] for p in pos] + [q[0] for q in neg])
    vj = np.array([p[1] for p in pos] + [q[1] for q in neg])
    lab = np.array([1] * len(pos) + [0] * len(neg))
    return fi, vj, lab


def unimodal_trials(spk, gender_of, seed, n_pos=3000, n_neg=3000, same_gender=False):
    """Same as build_trials but used for face<->face or voice<->voice."""
    return build_trials(spk, gender_of, seed, n_pos, n_neg, same_gender)


def eval_trials(Ef, Ev, fi, vj, lab):
    """Ef, Ev: L2-normalised embeddings in a shared space. Returns dict."""
    cos = np.sum(Ef[fi] * Ev[vj], axis=1)
    d = 2 - 2 * cos
    return dict(eer=eer_from_scores(cos, lab), auc=100 * roc_auc_score(lab, cos),
                pos_d=float(d[lab == 1].mean()), neg_d=float(d[lab == 0].mean()),
                margin=float(d[lab == 0].mean() - d[lab == 1].mean()),
                cohen_d=float(cohens_d(cos[lab == 1], cos[lab == 0])))


# ------------------------------------------------------------------ CCA
class RidgeCCA:
    """Regularised linear CCA on standardised, PCA-reduced inputs.

    fit(X, Y) with paired rows; transform gives k-dim projections of each view.
    """

    def __init__(self, k=64, reg=1e-2, pca_x=256, pca_y=None):
        self.k, self.reg, self.pca_x, self.pca_y = k, reg, pca_x, pca_y

    def _prep(self, X, n_pca):
        mu = X.mean(0)
        sd = X.std(0) + 1e-6
        Z = (X - mu) / sd
        if n_pca is not None and n_pca < Z.shape[1]:
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            P = Vt[:n_pca].T
        else:
            P = np.eye(Z.shape[1])
        return mu, sd, P

    def fit(self, X, Y):
        self.mx, self.sx, self.Px = self._prep(X, self.pca_x)
        self.my, self.sy, self.Py = self._prep(Y, self.pca_y)
        A = ((X - self.mx) / self.sx) @ self.Px
        B = ((Y - self.my) / self.sy) @ self.Py
        n = len(A)
        Cxx = A.T @ A / n + self.reg * np.eye(A.shape[1])
        Cyy = B.T @ B / n + self.reg * np.eye(B.shape[1])
        Cxy = A.T @ B / n
        Lx = np.linalg.cholesky(Cxx)
        Ly = np.linalg.cholesky(Cyy)
        Lxi = np.linalg.inv(Lx)
        Lyi = np.linalg.inv(Ly)
        T = Lxi @ Cxy @ Lyi.T
        U, S, Vt = np.linalg.svd(T, full_matrices=False)
        self.Wx = Lxi.T @ U[:, :self.k]
        self.Wy = Lyi.T @ Vt[:self.k].T
        self.canon_corr = S[:self.k]
        return self

    def transform_x(self, X):
        return l2n((((X - self.mx) / self.sx) @ self.Px) @ self.Wx)

    def transform_y(self, Y):
        return l2n((((Y - self.my) / self.sy) @ self.Py) @ self.Wy)


# ------------------------------------------------------------------ shift
def mmd_rbf(X, Y, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)]
    Z = np.vstack([X, Y])
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    sigma2 = np.median(D2[D2 > 0])
    K = np.exp(-D2 / (2 * sigma2))
    a, b = len(X), len(Y)
    return float(K[:a, :a].mean() + K[a:, a:].mean() - 2 * K[:a, a:].mean())


def knn_dist_to(ref, X, k=5):
    """Mean cosine distance of each X row to its k nearest rows in ref."""
    R, Q = l2n(ref), l2n(X)
    out = np.empty(len(Q))
    for s in range(0, len(Q), 512):
        sims = Q[s:s + 512] @ R.T
        top = np.sort(sims, axis=1)[:, -k:]
        out[s:s + 512] = 1 - top.mean(1)
    return out
