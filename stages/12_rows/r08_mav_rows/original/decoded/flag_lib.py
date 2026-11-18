"""FLAG 2027 — shared library for the Kaggle GPU notebooks.

Everything here was validated locally (see Experiment/*/NOTES.md). Key facts baked in:
  * CSV label column is the speaker int; face/voice rows are aligned.
  * Voice features contain duplicates (4029 unique of 6485). Dedup per voice clip was expected to help but
    a 3-split sweep says the opposite (int_g 32.26 without vs 33.02 with) -> `dedup_clip` defaults to False
    and stays in the grid.
  * Model selection uses int_g (gender-constrained internal EER), 3 speaker-disjoint splits.
  * Dev scoring: per-file mean centering, raw cosine, lower = same. AS-norm does NOT transfer.
  * Full CORAL destroys identity (EXP-004); only first-moment alignment is safe. `align_alpha` interpolates
    between the two so the sweep can test partial whitening.
"""
from __future__ import annotations
import os, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

# ----------------------------------------------------------------- data paths
DATA = Path(os.environ.get("FLAG_DATA", "/kaggle/input"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PCA_FACE = 256
NAMES = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}


_FOUND = {}


def find_file(name, data=None):
    """Locate a data file anywhere under DATA, so the notebook works no matter how Kaggle
    nests the uploaded dataset (with or without train/ and dev/ subfolders)."""
    key = (str(data or DATA), name)
    if key in _FOUND:
        return _FOUND[key]
    root = Path(data or DATA)
    hits = [root / name] if (root / name).exists() else sorted(root.rglob(Path(name).name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}. Contents: "
                                f"{[str(p.relative_to(root)) for p in list(root.rglob('*'))[:20]]}")
    _FOUND[key] = hits[0]
    return hits[0]


def load_train(data=None):
    face = pd.read_csv(find_file("train/train_English_faces.csv", data), header=None)
    voice = pd.read_csv(find_file("train/train_English_voices.csv", data), header=None)
    Xf = face.iloc[:, :-1].values.astype(np.float32)
    Xv = voice.iloc[:, :-1].values.astype(np.float32)
    assert (face.iloc[:, -1].values == voice.iloc[:, -1].values).all()
    txt = pd.read_csv(find_file("train/train_English.txt", data), sep=" ", header=None,
                      names=["pair_id", "label", "voice", "face", "spk", "spk_int"])
    spk = txt.spk.values.astype(str)
    meta = pd.read_csv(find_file("train/meta_file_train_set.csv", data), header=None, names=["spk", "gender"])
    gmap = dict(zip(meta.spk, meta.gender))
    _, vgrp = np.unique(np.round(Xv, 4), axis=0, return_inverse=True)   # voice-clip id (dedup key)
    return Xf, Xv, spk, gmap, vgrp


def _dev_file(stem, protocol, lang, data=None):
    """Dev files come in two layouts and both are supported:
       A  flattened upload      -> dev/no_gender_English_faces.csv
       B  official zip layout   -> dev_set/no_gender/features/English_test_faces.csv
    """
    root = Path(data or DATA)
    cands = list(root.rglob(f"{protocol}_{lang}_{stem}"))
    if not cands:
        cands = [p for p in root.rglob(f"{lang}_test_{stem}") if f"/{protocol}/" in p.as_posix()]
    if not cands and stem == "test.txt":
        cands = [p for p in root.rglob(f"{lang}_test.txt") if f"/{protocol}/" in p.as_posix()]
    if not cands:
        raise FileNotFoundError(f"dev {protocol}/{lang} {stem} not found under {root}")
    return sorted(cands, key=lambda p: len(p.as_posix()))[0]


def load_dev(protocol, lang, data=None):
    Xf = pd.read_csv(_dev_file("faces.csv", protocol, lang, data), header=None).values.astype(np.float32)
    Xv = pd.read_csv(_dev_file("voices.csv", protocol, lang, data), header=None).values.astype(np.float32)
    t = pd.read_csv(_dev_file("test.txt", protocol, lang, data), sep=" ", header=None,
                    names=["pair_id", "voice", "face"])
    assert len(Xf) == len(Xv) == len(t), f"{protocol}/{lang}: {len(Xf)}/{len(Xv)}/{len(t)} rows disagree"
    return Xf, Xv, t


# ----------------------------------------------------------------- protocol
def speaker_split(speakers, seed, n_val=14):
    uniq = np.array(sorted(set(speakers)))
    val = set(np.random.RandomState(seed).choice(uniq, n_val, replace=False))
    is_val = np.array([s in val for s in speakers])
    return ~is_val, is_val


def build_trials(spk, gender_of, seed, n_pos=3000, n_neg=3000, same_gender=False, group=None):
    rng = np.random.RandomState(seed)
    spk = np.asarray(spk)
    idx_by_spk = {s: np.where(spk == s)[0] for s in np.unique(spk)}
    g = np.array([gender_of[s] for s in spk])
    n = len(spk); pos, neg = [], []
    while len(pos) < n_pos:
        i = rng.randint(n); cand = idx_by_spk[spk[i]]
        if len(cand) < 2: continue
        j = rng.choice(cand)
        if j == i or (group is not None and group[i] == group[j]): continue
        pos.append((i, j))
    while len(neg) < n_neg:
        i, j = rng.randint(n), rng.randint(n)
        if spk[i] == spk[j] or (same_gender and g[i] != g[j]): continue
        neg.append((i, j))
    fi = np.array([p[0] for p in pos] + [q[0] for q in neg])
    vj = np.array([p[1] for p in pos] + [q[1] for q in neg])
    return fi, vj, np.array([1] * len(pos) + [0] * len(neg))


def eer_from_scores(scores, labels):
    """EER in %, with `scores` oriented as higher = same speaker.

    The crossing FAR = FRR is interpolated between the two bracketing ROC points; taking the
    nearest point instead quantises the result on small trial sets.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    d = fnr - fpr
    i = np.nanargmin(np.abs(d))
    if d[i] == 0 or len(fpr) < 2:
        return 100 * (fpr[i] + fnr[i]) / 2
    j = i - 1 if (i > 0 and d[i - 1] * d[i] < 0) else min(i + 1, len(d) - 1)
    if d[j] * d[i] >= 0:
        return 100 * (fpr[i] + fnr[i]) / 2
    t = d[i] / (d[i] - d[j])                      # fraction of the way from i to j
    return 100 * float(fpr[i] + t * (fpr[j] - fpr[i]))


def l2n(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
def z(x): return (x - x.mean()) / (x.std() + 1e-8)


# ----------------------------------------------------------------- preprocessing
class Prep:
    """Standardise both modalities; PCA the face view down to PCA_FACE (fit on train rows only).

    The PCA cache key always includes a fingerprint of the actual feature matrix. Keying only on the
    split would silently reuse a 4096-d projection for a 512-d ArcFace matrix once we started
    swapping feature sets.
    """
    _cache = {}

    @staticmethod
    def _fingerprint(Xf_tr, Xv_tr):
        return (Xf_tr.shape, Xv_tr.shape,
                float(Xf_tr[0].sum()), float(Xf_tr[-1].sum()), float(Xf_tr[:, 0].sum()),
                float(Xv_tr[0].sum()), float(Xv_tr[-1].sum()))

    def __init__(self, Xf_tr, Xv_tr, key=None):
        self.mf, self.sf = Xf_tr.mean(0), Xf_tr.std(0) + 1e-6
        self.mv, self.sv = Xv_tr.mean(0), Xv_tr.std(0) + 1e-6
        ck = (key, self._fingerprint(Xf_tr, Xv_tr))
        if ck in Prep._cache:
            self.P = Prep._cache[ck]
            assert self.P.shape[0] == Xf_tr.shape[1], "PCA cache mismatch"
            return
        Z = (Xf_tr - self.mf) / self.sf
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        self.P = Vt[:min(PCA_FACE, Z.shape[1])].T
        Prep._cache[ck] = self.P

    def f(self, X): return (((X - self.mf) / self.sf) @ self.P).astype(np.float32)
    def v(self, X): return ((X - self.mv) / self.sv).astype(np.float32)


def center_to(Z, mu):
    """First-moment alignment: remove this file's mean, put the train mean back (EXP-003c)."""
    return Z - Z.mean(0) + mu


def _msqrt(C, inv=False, eps=1e-3):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps * w.max())
    w = 1 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * w) @ V.T


class Aligner:
    """Fits the train moments once; applies partial CORAL to one dev file.

    alpha=0 -> mean centering only (safe, EXP-003c). alpha=1 -> full CORAL (EXP-004: destroys identity).
    """
    def __init__(self, Ztr, shrink=0.1):
        self.mu = Ztr.mean(0)
        C = np.cov(Ztr.T)
        self.C_half = _msqrt((1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C)))
        self.shrink = shrink

    def __call__(self, Z, alpha=0.0):
        Zc = Z - Z.mean(0)
        if alpha <= 0:
            return Zc + self.mu
        C = np.cov(Zc.T)
        C = (1 - self.shrink) * C + self.shrink * np.trace(C) / len(C) * np.eye(len(C))
        M = _msqrt(C, inv=True) @ self.C_half
        M = (1 - alpha) * np.eye(len(M)) + alpha * M
        return Zc @ M + self.mu


# ----------------------------------------------------------------- linear CCA (used in the fusion)
class RidgeCCA:
    """Regularised linear CCA. Both views are PCA-reduced first: without it a 6144-d voice feature
    would need a 6144x6144 covariance + Cholesky, which is far slower than the rest of the pipeline."""

    def __init__(self, k=4, reg=1.0, pca_x=128, pca_y=192):
        self.k, self.reg, self.pca_x, self.pca_y = k, reg, pca_x, pca_y

    def _prep(self, X, n_pca):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        if n_pca is not None and n_pca < Z.shape[1]:
            _, _, Vt = np.linalg.svd(Z, full_matrices=False); P = Vt[:n_pca].T
        else:
            P = np.eye(Z.shape[1])
        return mu, sd, P

    def fit(self, X, Y):
        self.mx, self.sx, self.Px = self._prep(X, min(self.pca_x, X.shape[1]) if self.pca_x else None)
        self.my, self.sy, self.Py = self._prep(Y, min(self.pca_y, Y.shape[1]) if self.pca_y else None)
        A = ((X - self.mx) / self.sx) @ self.Px
        B = ((Y - self.my) / self.sy) @ self.Py
        n = len(A)
        Cxx = A.T @ A / n + self.reg * np.eye(A.shape[1])
        Cyy = B.T @ B / n + self.reg * np.eye(B.shape[1])
        Lxi = np.linalg.inv(np.linalg.cholesky(Cxx)); Lyi = np.linalg.inv(np.linalg.cholesky(Cyy))
        U, S, Vt = np.linalg.svd(Lxi @ (A.T @ B / n) @ Lyi.T, full_matrices=False)
        self.Wx = Lxi.T @ U[:, :self.k]; self.Wy = Lyi.T @ Vt[:self.k].T
        return self

    def transform_x(self, X): return l2n((((X - self.mx) / self.sx) @ self.Px) @ self.Wx)
    def transform_y(self, Y): return l2n((((Y - self.my) / self.sy) @ self.Py) @ self.Wy)


# ----------------------------------------------------------------- model + losses
class Net(nn.Module):
    def __init__(self, d_f, d_v, emb, hid, drop, n_cls):
        super().__init__()
        def branch(d):
            return nn.Sequential(nn.Dropout(drop), nn.Linear(d, hid), nn.BatchNorm1d(hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(hid, emb))
        self.bf, self.bv = branch(d_f), branch(d_v)
        self.cls = nn.Linear(2 * emb, n_cls)

    def forward(self, f, v):
        u, w = F.normalize(self.bf(f), dim=1), F.normalize(self.bv(v), dim=1)
        return u, w, self.cls(torch.cat([u, w], 1))


def infonce(u, w, y, tau=0.07, g=None, same_gender_only=False):
    S = u @ w.T / tau
    same = (y[:, None] == y[None, :]).float()
    if same_gender_only and g is not None:
        keep = ((g[:, None] == g[None, :]) | (same > 0)).float()
        S = S.masked_fill(keep == 0, -1e4)
    lr_ = F.log_softmax(S, 1); lc = F.log_softmax(S, 0)
    return 0.5 * ((-(lr_ * same).sum(1) / same.sum(1)).mean() + (-(lc * same).sum(0) / same.sum(0)).mean())


BASE_CFG = dict(hid=512, drop=0.5, lr=1e-3, wd=1e-2, bs=256, tau=0.07, emb=128,
                epochs=25, dedup_clip=False, aug="none", aug_p=0.0, same_gender_only=False)


def augment(fb, vb, cfg, sdF, sdV, spk_b, idx_by_spk, Vall):
    a, p = cfg["aug"], cfg["aug_p"]
    if a == "none" or p <= 0:
        return fb, vb
    if a == "noise":
        vb = vb + torch.randn_like(vb) * sdV * p
    elif a == "both":
        vb = vb + torch.randn_like(vb) * sdV * p
        fb = fb + torch.randn_like(fb) * sdF * p
    elif a == "dropdim":
        vb = vb * ((torch.rand_like(vb) > p).float() / (1 - p))
    elif a == "shift":
        vb = vb + torch.randn(vb.shape[0], 1, device=vb.device) * sdV * p
    elif a == "mixup":
        lam = 1 - p * torch.rand(vb.shape[0], 1, device=vb.device)
        partner = torch.tensor([np.random.choice(idx_by_spk[s]) for s in spk_b], device=vb.device)
        vb = lam * vb + (1 - lam) * Vall[partner]
    return fb, vb


def train_model(Xf, Xv, spk, gmap, vgrp, tr_idx, cfg, seed, val_idx=None, eval_every=0):
    """Trains one model. If val_idx given with eval_every>0, returns the internal EER history too."""
    cfg = dict(BASE_CFG, **cfg)
    torch.manual_seed(seed); np.random.seed(seed)
    prep = Prep(Xf[tr_idx], Xv[tr_idx], key=f"{len(tr_idx)}_{int(np.asarray(tr_idx).sum())}")
    Ftr = torch.tensor(prep.f(Xf[tr_idx]), device=DEVICE)
    Vtr = torch.tensor(prep.v(Xv[tr_idx]), device=DEVICE)
    sdF, sdV = Ftr.std(0, keepdim=True), Vtr.std(0, keepdim=True)
    spk_tr = np.asarray(spk)[tr_idx]
    cls = {s: i for i, s in enumerate(np.unique(spk_tr))}
    idx_by_spk = {s: np.where(spk_tr == s)[0] for s in np.unique(spk_tr)}
    Y = torch.tensor([cls[s] for s in spk_tr], device=DEVICE)
    G = torch.tensor((np.array([gmap[s] for s in spk_tr]) == "m").astype(np.int64), device=DEVICE)
    C = np.asarray(vgrp)[tr_idx]
    net = Net(PCA_FACE, Xv.shape[1], cfg["emb"], cfg["hid"], cfg["drop"], len(cls)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    rng = np.random.RandomState(seed)
    hist = []
    for ep in range(1, cfg["epochs"] + 1):
        net.train()
        if cfg["dedup_clip"]:
            perm = rng.permutation(len(tr_idx)); _, first = np.unique(C[perm], return_index=True)
            order = perm[np.sort(first)]
        else:
            order = rng.permutation(len(tr_idx))
        for s in range(0, len(order) - cfg["bs"] // 2, cfg["bs"]):
            b = order[s:s + cfg["bs"]]; bt = torch.tensor(b, device=DEVICE)
            fb, vb = augment(Ftr[bt], Vtr[bt], cfg, sdF, sdV, spk_tr[b], idx_by_spk, Vtr)
            u, w, _ = net(fb, vb)
            loss = infonce(u, w, Y[bt], cfg["tau"], G[bt], cfg["same_gender_only"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if val_idx is not None and eval_every and ep % eval_every == 0:
            u, w = embed_one(net, prep, Xf[val_idx], Xv[val_idx])
            hist.append(dict(ep=ep, loss=float(loss.detach())))
    return net, prep, hist


@torch.no_grad()
def embed_one(net, prep, Xf_, Xv_, align=True, mu_f=None, mu_v=None, alF=None, alV=None, alpha=0.0):
    """align: first-moment centering. Pass Aligner objects + alpha>0 for partial CORAL."""
    A, B = prep.f(Xf_), prep.v(Xv_)
    if alF is not None and alV is not None:
        A, B = alF(A, alpha), alV(B, alpha)
    elif align:
        A = center_to(A, 0.0 if mu_f is None else mu_f)
        B = center_to(B, 0.0 if mu_v is None else mu_v)
    net.eval()
    u, w, _ = net(torch.tensor(A.astype(np.float32), device=DEVICE), torch.tensor(B.astype(np.float32), device=DEVICE))
    return u.cpu().numpy(), w.cpu().numpy()


def fuse_scores(embs, fi, vj, cca=None, cca_weight=0.25):
    """Fuse z-scored cosines. Higher = same speaker.

    `embs` are the deep models; pass the CCA separately as `cca` so its weight does not shrink
    as the ensemble grows. EXP-007: equal weighting over n+1 members gives the CCA 1/4 at n=3 but
    only 1/11 at n=10, which is why bigger ensembles looked worse. The optimum is cca_weight ~0.25.
    """
    deep = np.mean([z(np.sum(Ef[fi] * Ev[vj], 1)) for Ef, Ev in embs], 0)
    if cca is None or cca_weight <= 0:
        return deep
    c = z(np.sum(cca[0][fi] * cca[1][vj], 1))
    return (1 - cca_weight) * deep + cca_weight * c


def write_submission(path, scores_by_cell, devs):
    """scores_by_cell[(protocol, lang)] = higher-is-same score array; written as -score (lower = same)."""
    with zipfile.ZipFile(path, "w") as zf:
        for k, fn in NAMES.items():
            t = devs[k][2]; sc = scores_by_cell[k]
            assert len(sc) == len(t) and np.isfinite(sc).all()
            zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(t.pair_id, -sc)) + "\n")
    return path
