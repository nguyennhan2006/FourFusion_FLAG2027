"""Rebuild the best submission so far — EXP-003c, CodaBench Overall 31.34.

Self-contained: needs only numpy / pandas / torch / scikit-learn and the feature CSVs.
No imports from the Experiment/ tree, so this file is the single source of truth for the result.

    python reproduce.py                 # writes submission_best.zip
    python reproduce.py --internal      # also prints the speaker-disjoint internal EER
    python reproduce.py --data DIR      # feature root (default: ../kaggle_upload)

Recipe (each ingredient earned its place; see README.md for the evidence):
    face 4096 -> standardise -> PCA 256          |  voice 192 -> standardise
    two-branch MLP (512 hidden, BN, dropout 0.5) -> L2-normalised 128-d embeddings
    symmetric InfoNCE, tau 0.07, positives = same speaker (SupCon), AdamW 1e-3 / wd 1e-2, cosine, 15 epochs
    3 training seeds (1, 11, 21) + a 4-dim ridge CCA, fused as the mean of per-model z-scored cosines
    at test time: subtract each dev file's own mean, add the train mean; score = raw cosine
    submitted score = -(fused score), i.e. LOWER = same speaker
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve

PCA_FACE = 256
DEEP_SEEDS = [1, 11, 21]
EPOCHS = 15
CFG = dict(hid=512, emb=128, drop=0.5, lr=1e-3, wd=1e-2, bs=256, tau=0.07)
CCA_CFG = dict(k=4, reg=1.0, pca_x=128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CELLS = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}


# --------------------------------------------------------------------------- data
def find(root: Path, name: str) -> Path:
    p = root / name
    if p.exists():
        return p
    hits = sorted(root.rglob(Path(name).name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def load_train(root: Path):
    face = pd.read_csv(find(root, "train/train_English_faces.csv"), header=None)
    voice = pd.read_csv(find(root, "train/train_English_voices.csv"), header=None)
    # The last CSV column is the speaker int; face and voice rows are aligned (verified in EDA-0).
    assert (face.iloc[:, -1].values == voice.iloc[:, -1].values).all()
    Xf = face.iloc[:, :-1].values.astype(np.float32)
    Xv = voice.iloc[:, :-1].values.astype(np.float32)
    txt = pd.read_csv(find(root, "train/train_English.txt"), sep=" ", header=None,
                      names=["pair_id", "label", "voice", "face", "spk", "spk_int"])
    spk = txt.spk.values.astype(str)
    meta = pd.read_csv(find(root, "train/meta_file_train_set.csv"), header=None, names=["spk", "gender"])
    gmap = dict(zip(meta.spk, meta.gender))
    # 6485 rows hold only 4029 distinct voice vectors: several face crops share one audio clip.
    _, vgrp = np.unique(np.round(Xv, 4), axis=0, return_inverse=True)
    return Xf, Xv, spk, gmap, vgrp


def load_dev(root: Path, protocol: str, lang: str):
    Xf = pd.read_csv(find(root, f"dev/{protocol}_{lang}_faces.csv"), header=None).values.astype(np.float32)
    Xv = pd.read_csv(find(root, f"dev/{protocol}_{lang}_voices.csv"), header=None).values.astype(np.float32)
    t = pd.read_csv(find(root, f"dev/{protocol}_{lang}_test.txt"), sep=" ", header=None,
                    names=["pair_id", "voice", "face"])
    return Xf, Xv, t


# --------------------------------------------------------------------------- pieces
def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def center_to(X, mu):
    """First-moment domain alignment. Full CORAL was tried and cost +10 EER (EXP-004)."""
    return X - X.mean(0) + mu


class Prep:
    """Standardise both modalities; PCA the face down to PCA_FACE. Fit on training rows only."""

    def __init__(self, Xf, Xv, P=None):
        self.mf, self.sf = Xf.mean(0), Xf.std(0) + 1e-6
        self.mv, self.sv = Xv.mean(0), Xv.std(0) + 1e-6
        if P is None:
            _, _, Vt = np.linalg.svd((Xf - self.mf) / self.sf, full_matrices=False)
            P = Vt[:PCA_FACE].T
        self.P = P

    def f(self, X):
        return (((X - self.mf) / self.sf) @ self.P).astype(np.float32)

    def v(self, X):
        return ((X - self.mv) / self.sv).astype(np.float32)


class RidgeCCA:
    """Regularised linear CCA. On its own (k=32) this scored 35.79; k=4 scored 34.46."""

    def __init__(self, k=4, reg=1.0, pca_x=128):
        self.k, self.reg, self.pca_x = k, reg, pca_x

    @staticmethod
    def _prep(X, n_pca):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        if n_pca is not None and n_pca < Z.shape[1]:
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            P = Vt[:n_pca].T
        else:
            P = np.eye(Z.shape[1])
        return mu, sd, P

    def fit(self, X, Y):
        self.mx, self.sx, self.Px = self._prep(X, self.pca_x)
        self.my, self.sy, self.Py = self._prep(Y, None)
        A = ((X - self.mx) / self.sx) @ self.Px
        B = ((Y - self.my) / self.sy) @ self.Py
        n = len(A)
        Lxi = np.linalg.inv(np.linalg.cholesky(A.T @ A / n + self.reg * np.eye(A.shape[1])))
        Lyi = np.linalg.inv(np.linalg.cholesky(B.T @ B / n + self.reg * np.eye(B.shape[1])))
        U, _, Vt = np.linalg.svd(Lxi @ (A.T @ B / n) @ Lyi.T, full_matrices=False)
        self.Wx = Lxi.T @ U[:, :self.k]
        self.Wy = Lyi.T @ Vt[:self.k].T
        return self

    def transform_x(self, X):
        return l2n((((X - self.mx) / self.sx) @ self.Px) @ self.Wx)

    def transform_y(self, Y):
        return l2n((((Y - self.my) / self.sy) @ self.Py) @ self.Wy)


class Net(nn.Module):
    def __init__(self, d_f, d_v, emb, hid, drop, n_cls):
        super().__init__()

        def branch(d):
            return nn.Sequential(nn.Dropout(drop), nn.Linear(d, hid), nn.BatchNorm1d(hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(hid, emb))

        self.bf, self.bv = branch(d_f), branch(d_v)
        self.cls = nn.Linear(2 * emb, n_cls)     # kept so the checkpoint matches the tuned runs

    def forward(self, f, v):
        u, w = F.normalize(self.bf(f), dim=1), F.normalize(self.bv(v), dim=1)
        return u, w, self.cls(torch.cat([u, w], 1))


def infonce(u, w, y, tau):
    """Symmetric InfoNCE; every same-speaker pair in the batch counts as positive (SupCon)."""
    S = u @ w.T / tau
    same = (y[:, None] == y[None, :]).float()
    row = -(F.log_softmax(S, 1) * same).sum(1) / same.sum(1)
    col = -(F.log_softmax(S, 0) * same).sum(0) / same.sum(0)
    return 0.5 * (row.mean() + col.mean())


def train_one(Xf, Xv, spk, vgrp, tr_idx, seed, prep, epochs=EPOCHS, cfg=CFG, dedup_clip=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Ftr = torch.tensor(prep.f(Xf[tr_idx]), device=DEVICE)
    Vtr = torch.tensor(prep.v(Xv[tr_idx]), device=DEVICE)
    spk_tr = np.asarray(spk)[tr_idx]
    cls = {s: i for i, s in enumerate(np.unique(spk_tr))}
    Y = torch.tensor([cls[s] for s in spk_tr], device=DEVICE)
    clip = np.asarray(vgrp)[tr_idx]
    net = Net(PCA_FACE, Xv.shape[1], cfg["emb"], cfg["hid"], cfg["drop"], len(cls)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        net.train()
        if dedup_clip:    # one row per voice clip per epoch
            perm = rng.permutation(len(tr_idx))
            _, first = np.unique(clip[perm], return_index=True)
            order = perm[np.sort(first)]
        else:
            order = rng.permutation(len(tr_idx))
        for s in range(0, len(order) - cfg["bs"] // 2, cfg["bs"]):
            b = torch.tensor(order[s:s + cfg["bs"]], device=DEVICE)
            u, w, _ = net(Ftr[b], Vtr[b])
            loss = infonce(u, w, Y[b], cfg["tau"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return net


@torch.no_grad()
def embed(net, prep, Xf_, Xv_, mu_f, mu_v):
    net.eval()
    A = torch.tensor(center_to(prep.f(Xf_), mu_f), device=DEVICE)
    B = torch.tensor(center_to(prep.v(Xv_), mu_v), device=DEVICE)
    u, w, _ = net(A, B)
    return u.cpu().numpy(), w.cpu().numpy()


def fuse(embs, fi, vj):
    return np.mean([zscore(np.sum(Ef[fi] * Ev[vj], 1)) for Ef, Ev in embs], 0)


# --------------------------------------------------------------------------- internal check
def speaker_split(spk, seed, n_val=14):
    uniq = np.array(sorted(set(spk)))
    val = set(np.random.RandomState(seed).choice(uniq, n_val, replace=False))
    is_val = np.array([s in val for s in spk])
    return ~is_val, is_val


def build_trials(spk, gmap, seed, n_pos=3000, n_neg=3000, same_gender=False):
    """no_gender: any different speaker. gender: different speaker of the SAME gender (mirrors the dev protocol)."""
    rng = np.random.RandomState(seed)
    spk = np.asarray(spk)
    by_spk = {s: np.where(spk == s)[0] for s in np.unique(spk)}
    g = np.array([gmap[s] for s in spk])
    n, pos, neg = len(spk), [], []
    while len(pos) < n_pos:
        i = rng.randint(n)
        cand = by_spk[spk[i]]
        if len(cand) < 2:
            continue
        j = rng.choice(cand)
        if j != i:
            pos.append((i, j))
    while len(neg) < n_neg:
        i, j = rng.randint(n), rng.randint(n)
        if spk[i] == spk[j] or (same_gender and g[i] != g[j]):
            continue
        neg.append((i, j))
    fi = np.array([p[0] for p in pos] + [q[0] for q in neg])
    vj = np.array([p[1] for p in pos] + [q[1] for q in neg])
    return fi, vj, np.array([1] * len(pos) + [0] * len(neg))


def eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


def run_internal(Xf, Xv, spk, gmap, vgrp, seeds=(1, 2, 3)):
    out = {"no_gender": [], "gender": []}
    for s in seeds:
        tr, va = speaker_split(spk, s)
        tri, vai = np.where(tr)[0], np.where(va)[0]
        prep = Prep(Xf[tri], Xv[tri])
        mu_f, mu_v = prep.f(Xf[tri]).mean(0), prep.v(Xv[tri]).mean(0)
        embs = [embed(train_one(Xf, Xv, spk, vgrp, tri, s + 10 * i, prep), prep, Xf[vai], Xv[vai], mu_f, mu_v)
                for i in range(len(DEEP_SEEDS))]
        cca = RidgeCCA(**CCA_CFG).fit(Xf[tri], Xv[tri])
        embs.append((cca.transform_x(center_to(Xf[vai], Xf[tri].mean(0))),
                     cca.transform_y(center_to(Xv[vai], Xv[tri].mean(0)))))
        for sg in (False, True):
            fi, vj, lab = build_trials(spk[vai], gmap, seed=s, same_gender=sg)
            out["gender" if sg else "no_gender"].append(eer(fuse(embs, fi, vj), lab))
        print(f"  split seed {s}: int_ng {out['no_gender'][-1]:.2f}  int_g {out['gender'][-1]:.2f}", flush=True)
    print(f"  mean       : int_ng {np.mean(out['no_gender']):.2f}  int_g {np.mean(out['gender']):.2f}")
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "kaggle_upload"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "submission_best.zip"))
    ap.add_argument("--internal", action="store_true", help="also run the speaker-disjoint internal check")
    args = ap.parse_args()

    root = Path(args.data)
    print("device:", DEVICE, "| data:", root)
    Xf, Xv, spk, gmap, vgrp = load_train(root)
    print(f"train: {Xf.shape} {Xv.shape} | {len(set(spk))} speakers | {len(np.unique(vgrp))} unique voice clips")

    if args.internal:
        print("internal (speaker-disjoint 56/14, 3 splits):")
        run_internal(Xf, Xv, spk, gmap, vgrp)

    all_idx = np.arange(len(spk))
    prep = Prep(Xf, Xv)
    mu_f, mu_v = prep.f(Xf).mean(0), prep.v(Xv).mean(0)
    nets = []
    for s in DEEP_SEEDS:
        nets.append(train_one(Xf, Xv, spk, vgrp, all_idx, s, prep))
        print("trained seed", s, flush=True)
    cca = RidgeCCA(**CCA_CFG).fit(Xf, Xv)

    scores = {}
    for key in CELLS:
        a, b, t = load_dev(root, *key)
        embs = [embed(net, prep, a, b, mu_f, mu_v) for net in nets]
        embs.append((cca.transform_x(center_to(a, Xf.mean(0))), cca.transform_y(center_to(b, Xv.mean(0)))))
        ii = np.arange(len(t))
        scores[key] = fuse(embs, ii, ii)
        print(f"  {key[0]}/{key[1]}: {len(t)} trials")

    with zipfile.ZipFile(args.out, "w") as zf:
        for key, fn in CELLS.items():
            t = load_dev(root, *key)[2]
            sc = scores[key]
            assert len(sc) == len(t) and np.isfinite(sc).all()
            zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(t.pair_id, -sc)) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
