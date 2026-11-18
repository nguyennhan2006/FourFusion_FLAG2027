"""FLAG 2027 v2 — training / ablation / submission on re-extracted features (Kaggle NB-2 and NB-3).

Builds on flag_lib (protocol, EER, CCA, submission writer — unchanged and already validated).
New here:
  * FeatureStore: organiser CSV features + the .npy arrays written by FLAG_04_extract, by name;
  * one configurable model: linear|mlp head, InfoNCE (+same-gender), shared-centre AAM, MSE align,
    orthogonality, gradient-reversal gender head, DANN language head on unlabeled dev voices;
  * duration-matched voice crops as training views, and a short-utterance internal metric;
  * resumable result tables (a finished row is never recomputed).
"""
from __future__ import annotations
import os, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import flag_lib as L
from flag_lib import DEVICE, NAMES, eer_from_scores, build_trials, speaker_split, z, l2n, RidgeCCA

CELLS = list(NAMES)                       # [(protocol, lang)] in submission order
SPLITS = ["train", "no_gender/English", "no_gender/Bangla", "gender/English", "gender/Bangla"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ features
class FeatureStore:
    """face(name, split) / voice(name, split) / voice_crops(name) -> float32 arrays in organiser row order.

    face  : 'vgg' (organiser 4096) | 'arcface'
    voice : 'given' (organiser 192) | 'ecapa192' | 'ecapa6144' | '<ssl>_L<a>-<b>' (mean of layers a..b)
    """

    def __init__(self, data_root=None, feats_dir=None):
        self.root = Path(data_root or os.environ.get("FLAG_DATA", "/kaggle/input"))
        # every folder holding a feats_manifest.json (NB-1 dataset, later extraction rounds, ...) is searched
        dirs = [Path(feats_dir)] if feats_dir else [m.parent for m in sorted(self.root.rglob("feats_manifest.json"))]
        dirs += [Path(p) for p in os.environ.get("FLAG_FEATS_EXTRA", "").split(os.pathsep) if p]
        self.fds = [d for i, d in enumerate(dirs) if d.exists() and d not in dirs[:i]]
        self.fd = self.fds[0] if self.fds else None
        self.Xf, self.Xv, self.spk, self.gmap, self.vgrp = L.load_train(self.root)
        self.dev = {k: L.load_dev(*k, data=self.root) for k in CELLS}
        self._c = {}
        for d in self.fds:
            log("feature dir:", d, "|", len(list(d.glob("*.npy"))), "arrays")
        if not self.fds:
            log("no feats_manifest.json found: only organiser features available")

    def _find(self, fname):
        for d in self.fds:
            if (d / fname).exists():
                return d / fname
        return None

    def _npy(self, stem, split):
        name = f"{stem}_{split.replace('/', '_')}.npy"
        f = self._find(name)
        if f is None:
            raise FileNotFoundError(f"{name} missing in {self.fds}")
        return np.load(f)

    def has(self, kind, name):
        if name in ("vgg", "given"):
            return True
        if self.fd is None:
            return False
        stem = f"{kind}_{name.split('_L')[0]}"
        return self._find(f"{stem}_train.npy") is not None

    def _n(self, split):
        return len(self.spk) if split == "train" else len(self.dev[tuple(split.split("/"))][2])

    def face(self, name, split):
        key = ("f", name, split)
        if key not in self._c:
            if name == "vgg":
                X = self.Xf if split == "train" else self.dev[tuple(split.split("/"))][0]
            else:
                X = self._npy(f"face_{name}", split)
            assert len(X) == self._n(split), f"face {name} {split}: {len(X)} rows vs {self._n(split)}"
            self._c[key] = X.astype(np.float32)
        return self._c[key]

    def voice(self, name, split):
        key = ("v", name, split)
        if key not in self._c:
            if name == "given":
                X = self.Xv if split == "train" else self.dev[tuple(split.split("/"))][1]
            elif "_L" in name:                               # e.g. xlsr_L6-12
                base, rng = name.split("_L")
                a, b = map(int, rng.split("-"))
                X = self._npy(f"voice_{base}", split)[:, a:b + 1].astype(np.float32).mean(1)
            else:
                X = self._npy(f"voice_{name}", split)
            assert len(X) == self._n(split), f"voice {name} {split}: {len(X)} rows vs {self._n(split)}"
            self._c[key] = X.astype(np.float32)
        return self._c[key]

    def voice_crops(self, name):
        """(K, N_train, D) duration-matched crops, or None if not extracted for this encoder."""
        if self.fd is None or name == "given" or "_L" in name:
            return None
        key = ("crops", name)
        if key not in self._c:        # cached: the 6144-d crops are ~0.5 GB and used by every model
            f = self._find(f"voice_{name}crop_train.npy")
            self._c[key] = np.load(f).astype(np.float32) if f is not None else None
        return self._c[key]

    def ext_rows(self, sources, cap=150, seed=0, exclude=()):
        """External training rows (face 4096 VGG, voice 192 organiser-ECAPA) from ext_<src>_*.npy + meta.
        One row per utterance, paired with a face frame of the SAME video (cycling through its sampled frames).
        cap: max rows per speaker (v4 median is 92) so the 70 in-domain speakers are not drowned.
        Speaker ids are namespaced '<src>:<id>'; `exclude` removes ids overlapping v4 train/dev."""
        key = ("ext", tuple(sources), cap, seed, tuple(sorted(exclude)))
        if key in self._c:
            return self._c[key]
        rng = np.random.RandomState(seed)
        parts = [pair_ext(self.ext_source(src), src, cap, rng, exclude) for src in sources]
        for src in sources:
            self.gmap.update(self.ext_source(src)["gen_of"])
        out = tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
        for sp in np.unique(out[2]):
            self.gmap.setdefault(sp, "u")
        log(f"ext rows {sources}: {len(out[2])} rows, {len(np.unique(out[2]))} speakers (cap {cap}, excluded {len(exclude)})")
        self._c[key] = out
        return out

    def ext_source(self, src):
        """Raw arrays + meta of one external source: dict(V, vm, Fa, fm). Cached."""
        key = ("ext_src", src)
        if key not in self._c:
            fv = self._find(f"ext_{src}_voice.npy")
            assert fv is not None, f"ext_{src}_voice.npy not found in {self.fds}"
            D = dict(V=np.load(fv).astype(np.float32), vm=pd.read_csv(fv.parent / f"ext_{src}_voice_meta.csv"),
                     Fa=np.load(fv.parent / f"ext_{src}_face.npy").astype(np.float32),
                     fm=pd.read_csv(fv.parent / f"ext_{src}_face_meta.csv"), key_of={}, gen_of={})
            # optional ext_<src>_speakers.csv (ids,name,split,gender): ids of the SAME person are merged
            # (v1 lists "Imran Khan" as id0001 and id0004) and the real gender is used.
            mf_ = fv.parent / f"ext_{src}_speakers.csv"
            if mf_.exists():
                for r_ in pd.read_csv(mf_).itertuples():
                    k_ = f"{src}:" + str(r_.name).strip().lower().replace(" ", "_")
                    D["key_of"][r_.ids] = k_
                    D["gen_of"][k_] = "m" if str(r_.gender).lower().startswith("m") else "f"
            self._c[key] = D
        return self._c[key]

    def dev_voices_all(self, name):
        """(X, is_bangla) over the four dev files — the only language-labelled audio we have (for DANN)."""
        Xs, ys = [], []
        for k in CELLS:
            X = self.voice(name, "/".join(k))
            Xs.append(X - X.mean(0)); ys.append(np.full(len(X), k[1] == "Bangla"))
        return np.concatenate(Xs), np.concatenate(ys)


def pair_ext(D, src, cap, rng, exclude=(), ids=None, langs=None):
    """One training row per external utterance, paired with a face frame of the SAME video (cycling through
    its sampled frames; any frame of the speaker if that video has none). cap = max rows per speaker.
    ids / langs: keep only these speakers / languages (bilingual validation, flag_bilingual.py)."""
    V, vm, Fa, fm = D["V"], D["vm"], D["Fa"], D["fm"]
    by_vid = {k: np.asarray(g.index) for k, g in fm.groupby(["spk", "lang", "video"])}
    by_spk = {k: np.asarray(g.index) for k, g in fm.groupby("spk")}
    key_of = D.get("key_of", {})
    ok = np.array([f"{src}:{x}" not in exclude and key_of.get(x, "") not in exclude and x in by_spk for x in vm.spk])
    if ids is not None:
        ok &= vm.spk.isin(ids).values
    if langs is not None:
        ok &= vm.lang.isin(langs).values
    vm = vm[ok]
    F_all, V_all, S_all = [], [], []
    for spk_, g in vm.groupby("spk"):
        idx = np.asarray(g.index)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        for j, i in enumerate(idx):
            r = vm.loc[i]
            cand = by_vid.get((r.spk, r.lang, r.video), by_spk[r.spk])
            F_all.append(Fa[cand[j % len(cand)]]); V_all.append(V[i]); S_all.append(key_of.get(spk_, f"{src}:{spk_}"))
    if not S_all:
        return np.zeros((0, Fa.shape[1]), np.float32), np.zeros((0, V.shape[1]), np.float32), np.array([], dtype=str)
    return np.stack(F_all), np.stack(V_all), np.array(S_all)


# ============================================================================ preprocessing
class Prep:
    """Standardise; PCA face to pca_f and (optionally) voice to pca_v, fitted on training rows."""
    _cache = {}

    def __init__(self, Xf, Xv, pca_f=256, pca_v=None):
        self.mf, self.sf = Xf.mean(0), Xf.std(0) + 1e-6
        self.mv, self.sv = Xv.mean(0), Xv.std(0) + 1e-6
        self.Pf = self._pca((Xf - self.mf) / self.sf, pca_f)
        self.Pv = self._pca((Xv - self.mv) / self.sv, pca_v)

    @classmethod
    def _pca(cls, Z, k):
        if not k or k >= Z.shape[1]:
            return None
        ck = (Z.shape, float(Z[0, :8].sum()), float(Z[-1, -8:].sum()), float(Z[:, 0].sum()), k)
        if ck not in cls._cache:
            _, _, Vt = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
            cls._cache[ck] = Vt[:k].T.astype(np.float32)
        return cls._cache[ck]

    def f(self, X):
        Z = (X - self.mf) / self.sf
        return (Z if self.Pf is None else Z @ self.Pf).astype(np.float32)

    def v(self, X):
        Z = (X - self.mv) / self.sv
        return (Z if self.Pv is None else Z @ self.Pv).astype(np.float32)


def centre(Z, mu):
    """Per-set first-moment alignment (EXP-003c): remove this set's mean, restore the train mean."""
    return Z - Z.mean(0) + mu


# ============================================================================ model
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def branch(d_in, emb, head, drop, hid):
    if head == "linear":
        return nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, emb))
    return nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, hid), nn.BatchNorm1d(hid), nn.ReLU(),
                         nn.Dropout(drop), nn.Linear(hid, emb))


class Model(nn.Module):
    def __init__(self, d_f, d_v, n_cls, cfg):
        super().__init__()
        self.bf = branch(d_f, cfg["emb"], cfg["head"], cfg["drop"], cfg["hid"])
        self.bv = branch(d_v, cfg["emb"], cfg["head"], cfg["drop"], cfg["hid"])
        self.W = nn.Parameter(torch.randn(n_cls, cfg["emb"]) * 0.01)       # shared AAM centres
        self.gdisc = nn.Sequential(nn.Linear(cfg["emb"], 64), nn.ReLU(), nn.Linear(64, 2))
        self.ldisc = nn.Sequential(nn.Linear(cfg["emb"], 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, f, v):
        return F.normalize(self.bf(f), dim=1), F.normalize(self.bv(v), dim=1)


def infonce(u, w, y, tau, g=None, same_gender_only=False):
    S = u @ w.T / tau
    same = (y[:, None] == y[None, :]).float()
    if same_gender_only:
        keep = (g[:, None] == g[None, :]) | (same > 0)
        S = S.masked_fill(~keep, -1e4)
    return 0.5 * ((-(F.log_softmax(S, 1) * same).sum(1) / same.sum(1)).mean() +
                  (-(F.log_softmax(S, 0) * same).sum(0) / same.sum(0)).mean())


def aam(zn, W, y, m, s):
    cos = torch.clamp(zn @ F.normalize(W, dim=1).T, -1 + 1e-7, 1 - 1e-7)
    tgt = torch.cos(torch.acos(cos) + m)
    oh = F.one_hot(y, W.shape[0]).float()
    return F.cross_entropy(s * (oh * tgt + (1 - oh) * cos), y)


def orth(u, w, y):
    """Orthogonal projection loss over both modalities: same speaker -> cos 1, different -> cos 0."""
    E, Y = torch.cat([u, w]), torch.cat([y, y])
    C = E @ E.T
    same = (Y[:, None] == Y[None, :]).float()
    eye = torch.eye(len(E), device=E.device)
    pos = ((C * (same - eye)).sum() / (same - eye).sum().clamp(min=1))
    neg = ((C * (1 - same)).abs().sum() / (1 - same).sum().clamp(min=1))
    return (1 - pos) + neg


BASE = dict(
    face="vgg", voice="given", pca_f=256, pca_v=None,
    head="mlp", hid=512, emb=128, drop=0.5,
    lr=1e-3, wd=1e-2, bs=256, epochs=25, tau=0.07,
    w_nce=1.0, same_gender_only=False,
    w_aam=0.0, margin=0.2, scale=30.0,
    w_mse=0.0, w_orth=0.0,
    grl_gender=0.0, dann_lang=0.0,
    voice_view="full",                # full | crops | mix  (crops = duration-matched, plan §0.4)
    ext=(), ext_cap=150,              # external MAV-Celeb sources added to TRAINING only (organiser feature space)
    sampler="random", pk_P=32, pk_K=8,  # "pk": each batch = P speakers x K rows (needed for set-level losses)
    avg_from=None,                    # SWA-style tail averaging: mean of the weights at the end of every epoch >= avg_from
                                      # (0-based), BatchNorm statistics then recomputed on the training rows. None = off.
    w_proto=0.0, proto_min=2,         # SetProto: InfoNCE between per-speaker MEAN face and MEAN voice embeddings,
                                      # each mean over an independent random subset of 2..K rows (test-time
                                      # scoring aggregates clusters, so the bridge is trained at that level too)
)


def pk_batches(spk_idx, P, K, n_rows, rng):
    """One epoch of P-speakers x K-rows batches (rows drawn with replacement for small speakers).
    The number of batches keeps the epoch at about n_rows rows, like the random sampler."""
    speakers = list(spk_idx)
    out = []
    for _ in range(max(1, n_rows // (P * K))):
        chosen = rng.choice(len(speakers), size=min(P, len(speakers)), replace=False)
        b = []
        for c in chosen:
            rows = spk_idx[speakers[c]]
            b.append(rng.choice(rows, size=K, replace=len(rows) < K))
        out.append(np.concatenate(b))
    return out


def proto_infonce(u, w, P, K, kmin, tau, gen):
    """u, w: (P*K, d) rows grouped by speaker. Independent random subsets (size kmin..K) for faces and voices."""
    U, W = u.view(P, K, -1), w.view(P, K, -1)
    mf = torch.zeros(P, K, device=u.device); mv = torch.zeros(P, K, device=u.device)
    for p in range(P):
        a = int(torch.randint(kmin, K + 1, (1,), generator=gen)); c = int(torch.randint(kmin, K + 1, (1,), generator=gen))
        mf[p, torch.randperm(K, generator=gen)[:a]] = 1; mv[p, torch.randperm(K, generator=gen)[:c]] = 1
    pf = F.normalize((U * mf[..., None]).sum(1) / mf.sum(1, keepdim=True), dim=1)
    pv = F.normalize((W * mv[..., None]).sum(1) / mv.sum(1, keepdim=True), dim=1)
    S = pf @ pv.T / tau
    t = torch.arange(P, device=u.device)
    return 0.5 * (F.cross_entropy(S, t) + F.cross_entropy(S.T, t))


def ramp(p):
    """DANN schedule: 0 -> 1 over training."""
    return float(2 / (1 + np.exp(-10 * p)) - 1)


def train_one(store, cfg, tr_idx, seed, rows=None, prep=None, init=None):
    """rows=(F, V, spk): train on exactly these rows instead of store train[tr_idx] (+ cfg ext).
    prep: reuse a fitted Prep (pretrain -> fine-tune must share one input space).
    init: state_dict to start from; the AAM centres W are skipped (their class set differs)."""
    cfg = dict(BASE, **cfg)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    if rows is not None:
        assert cfg["voice_view"] == "full" and not cfg["ext"] and not cfg["dann_lang"], "rows= supports plain training only"
        TF, TV, spk = rows
    else:
        Xf, Xv = store.face(cfg["face"], "train"), store.voice(cfg["voice"], "train")
        TF, TV, spk = Xf[tr_idx], Xv[tr_idx], store.spk[tr_idx]
        if cfg["ext"]:
            assert cfg["face"] == "vgg" and cfg["voice"] == "given" and cfg["voice_view"] == "full",                 "external rows exist only in the organiser feature space (VGG 4096 + organiser ECAPA 192)"
            EF, EV, ES = store.ext_rows(tuple(cfg["ext"]), cfg["ext_cap"], exclude=tuple(cfg.get("ext_exclude", ())))
            TF, TV, spk = np.concatenate([TF, EF]), np.concatenate([TV, EV]), np.concatenate([spk, ES])
    prep = prep or Prep(TF, TV, cfg["pca_f"], cfg["pca_v"])
    Ftr = torch.tensor(prep.f(TF), device=DEVICE)
    views = [torch.tensor(prep.v(TV), device=DEVICE)]
    if cfg["voice_view"] != "full":
        C = store.voice_crops(cfg["voice"])
        assert C is not None, f"voice_view={cfg['voice_view']} but no crops for {cfg['voice']}"
        views += [torch.tensor(prep.v(c[tr_idx]), device=DEVICE) for c in C]
    V = torch.stack(views)                                       # (n_views, n, d)
    cls = {s: i for i, s in enumerate(np.unique(spk))}
    Y = torch.tensor([cls[s] for s in spk], device=DEVICE)
    G = torch.tensor([store.gmap.get(s, "u") == "m" for s in spk], device=DEVICE).long()

    dann = cfg["dann_lang"] > 0
    if dann:
        Xd, isbn = store.dev_voices_all(cfg["voice"])
        Dv = torch.tensor(prep.v(Xd + Xv[tr_idx].mean(0)), device=DEVICE)   # dev files already centred
        Dl = torch.tensor(isbn, device=DEVICE).long()

    net = Model(Ftr.shape[1], V.shape[2], len(cls), cfg).to(DEVICE)
    if init is not None:
        net.load_state_dict({k: v for k, v in init.items() if k != "W"}, strict=False)
    opt =torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    n, bs = len(spk), cfg["bs"]
    pk = cfg["sampler"] == "pk"
    if pk:
        spk_idx = {s: np.where(np.asarray(spk) == s)[0] for s in np.unique(spk)}
        P, K = min(cfg["pk_P"], len(spk_idx)), cfg["pk_K"]
        gen = torch.Generator().manual_seed(seed)
        steps = max(1, n // (P * K)) * cfg["epochs"]
    else:
        assert not cfg["w_proto"], "w_proto needs sampler='pk' (the loss needs K rows per speaker in a batch)"
        steps = len(range(0, max(1, n - bs // 2), bs)) * cfg["epochs"]
    step = 0
    for ep in range(cfg["epochs"]):
        net.train()
        order = rng.permutation(n)
        if cfg["voice_view"] == "crops":
            vsel = rng.randint(1, len(V), size=n)
        elif cfg["voice_view"] == "mix":
            vsel = rng.randint(0, len(V), size=n)
        else:
            vsel = np.zeros(n, dtype=int)
        batches = pk_batches(spk_idx, P, K, n, rng) if pk else \
            [order[s:s + bs] for s in range(0, max(1, n - bs // 2), bs)]
        for b in batches:
            bt = torch.tensor(b, device=DEVICE)
            vt = torch.tensor(vsel[b], device=DEVICE)
            u, w = net(Ftr[bt], V[vt, bt])
            y, g = Y[bt], G[bt]
            loss = 0.0
            if cfg["w_nce"]:
                loss = loss + cfg["w_nce"] * infonce(u, w, y, cfg["tau"], g, cfg["same_gender_only"])
            if cfg["w_aam"]:
                loss = loss + cfg["w_aam"] * 0.5 * (aam(u, net.W, y, cfg["margin"], cfg["scale"]) +
                                                    aam(w, net.W, y, cfg["margin"], cfg["scale"]))
            if cfg["w_mse"]:
                loss = loss + cfg["w_mse"] * ((u - w) ** 2).sum(1).mean()
            if cfg["w_orth"]:
                loss = loss + cfg["w_orth"] * orth(u, w, y)
            if cfg["w_proto"]:
                loss = loss + cfg["w_proto"] * proto_infonce(u, w, P, K, cfg["proto_min"], cfg["tau"], gen)
            lam = ramp(step / steps)
            if cfg["grl_gender"]:
                e = GradReverse.apply(torch.cat([u, w]), cfg["grl_gender"] * lam)
                loss = loss + F.cross_entropy(net.gdisc(e), torch.cat([g, g]))
            if dann:
                j = torch.randint(0, len(Dv), (len(b),), device=DEVICE)
                wd_ = F.normalize(net.bv(Dv[j]), dim=1)
                # train voices are English too; mix them in so "English" is not just "dev"
                e = GradReverse.apply(torch.cat([wd_, w]), cfg["dann_lang"] * lam)
                lab = torch.cat([Dl[j], torch.zeros(len(w), device=DEVICE, dtype=torch.long)])
                loss = loss + F.cross_entropy(net.ldisc(e), lab)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
        sched.step()
        if cfg["avg_from"] is not None and ep >= cfg["avg_from"]:
            with torch.no_grad():
                cur = {k: v.detach().clone().float() for k, v in net.state_dict().items() if v.dtype.is_floating_point}
                n_avg = ep - cfg["avg_from"] + 1
                avg = cur if n_avg == 1 else {k: avg[k] + (cur[k] - avg[k]) / n_avg for k in cur}
    if cfg["avg_from"] is not None:
        _load_average_and_update_bn(net, avg, Ftr, V[0], bs)
    net.eval()
    return net, prep


@torch.no_grad()
def _load_average_and_update_bn(net, avg, Ftr, Vtr, bs):
    """Load averaged weights, then recompute BatchNorm running statistics with one pass over the training rows
    (as torch.optim.swa_utils.update_bn: cumulative average, train mode, no gradient)."""
    sd = net.state_dict()
    sd.update({k: v.to(sd[k].dtype) for k, v in avg.items()})
    net.load_state_dict(sd)
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    net.train()
    for s in range(0, len(Ftr), bs):
        net(Ftr[s:s + bs], Vtr[s:s + bs])
    for m in bns:
        m.momentum = 0.1


@torch.no_grad()
def embed(net, prep, Af, Av, mu_f, mu_v):
    """Per-set centring, then the two branches. Returns L2-normalised numpy embeddings."""
    A = torch.tensor(centre(prep.f(Af), mu_f), device=DEVICE)
    B = torch.tensor(centre(prep.v(Av), mu_v), device=DEVICE)
    u, w = net(A, B)
    return u.cpu().numpy(), w.cpu().numpy()


def fuse(embs, fi, vj, cca=None, w=0.25):
    deep = np.mean([z(np.sum(Ef[fi] * Ev[vj], 1)) for Ef, Ev in embs], 0)
    if cca is None or w <= 0:
        return deep
    return (1 - w) * deep + w * z(np.sum(cca[0][fi] * cca[1][vj], 1))


CCA_CFG = dict(k=4, reg=1.0, pca_x=128, pca_y=192)


def cca_proj(Xf_tr, Xv_tr, Af, Av):
    m = RidgeCCA(**CCA_CFG).fit(Xf_tr, Xv_tr)
    return m.transform_x(centre(Af, Xf_tr.mean(0))), m.transform_y(centre(Av, Xv_tr.mean(0)))


# ============================================================================ internal evaluation
_TRIALS, _CCA = {}, {}


def evaluate(store, cfg, seeds=(1, 2, 3), n_models=3, cca_w=0.25):
    """Speaker-disjoint 56/14 splits x seeds. Returns int_ng, int_g, proxy, and int_g_short when the voice
    encoder has duration-matched crops (val voices replaced by their first crop: a Bangla-length proxy)."""
    cfg = dict(BASE, **cfg)
    Xf, Xv = store.face(cfg["face"], "train"), store.voice(cfg["voice"], "train")
    crops = store.voice_crops(cfg["voice"])
    res = {"no_gender": [], "gender": [], "gender_short": []}
    for s in seeds:
        tr, va = speaker_split(store.spk, s)
        tri, vai = np.where(tr)[0], np.where(va)[0]
        embs, embs_short = [], []
        for i in range(n_models):
            net, prep = train_one(store, cfg, tri, seed=s + 100 * i)
            mu_f, mu_v = prep.f(Xf[tri]).mean(0), prep.v(Xv[tri]).mean(0)
            embs.append(embed(net, prep, Xf[vai], Xv[vai], mu_f, mu_v))
            if crops is not None:
                embs_short.append(embed(net, prep, Xf[vai], crops[0][vai], mu_f, mu_v))
        ck = (cfg["face"], cfg["voice"], s)
        if ck not in _CCA:
            _CCA[ck] = cca_proj(Xf[tri], Xv[tri], Xf[vai], Xv[vai])
        cca = _CCA[ck]
        for sg in (False, True):
            tk = (s, sg)
            if tk not in _TRIALS:
                _TRIALS[tk] = build_trials(store.spk[vai], store.gmap, seed=s, n_pos=3000, n_neg=3000, same_gender=sg)
            fi, vj, lab = _TRIALS[tk]
            res["gender" if sg else "no_gender"].append(eer_from_scores(fuse(embs, fi, vj, cca, cca_w), lab))
            if sg and embs_short:
                res["gender_short"].append(eer_from_scores(fuse(embs_short, fi, vj, None, 0), lab))
    ng, g = float(np.mean(res["no_gender"])), float(np.mean(res["gender"]))
    out = dict(int_ng=round(ng, 2), int_g=round(g, 2), proxy=round((ng + g) / 2, 2),
               sd_g=round(float(np.std(res["gender"])), 2))
    if res["gender_short"]:
        out["int_g_short"] = round(float(np.mean(res["gender_short"])), 2)
    return out


class Table:
    """Resumable results: rows keyed by run name, flushed to csv after every run."""

    def __init__(self, path):
        self.path = Path(path)
        self.df = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def done(self, name):
        return len(self.df) and name in set(self.df.run)

    def get(self, name):
        return self.df[self.df.run == name].iloc[0].to_dict()

    def run(self, store, phase, name, cfg, **kw):
        if self.done(name):
            r = self.get(name); log(f"[{phase}] {name:45s} cached  proxy {r['proxy']}  int_g {r['int_g']}")
            return r
        t0 = time.time()
        r = dict(phase=phase, run=name, **evaluate(store, cfg, **kw), sec=0,
                 cfg=json.dumps(dict(BASE, **cfg), sort_keys=True))
        r["sec"] = round(time.time() - t0)
        self.df = pd.concat([self.df, pd.DataFrame([r])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        extra = f"  short {r['int_g_short']}" if "int_g_short" in r and pd.notna(r.get("int_g_short")) else ""
        log(f"[{phase}] {name:45s} ng {r['int_ng']:6.2f}  g {r['int_g']:6.2f}  proxy {r['proxy']:6.2f}{extra}  ({r['sec']}s)")
        return r

    def best(self, phase, key="proxy"):
        d = self.df[self.df.phase == phase].sort_values(key)
        return d.iloc[0].to_dict()


def pick(table, phase, incumbent_name, key="proxy", min_gain=0.3):
    """Keep the incumbent unless a phase run beats it by at least min_gain (noise floor ~ sd across splits)."""
    inc = table.get(incumbent_name)
    best = table.best(phase, key)
    if best["run"] != incumbent_name and best[key] <= inc[key] - min_gain:
        log(f"[{phase}] winner: {best['run']} ({key} {best[key]} vs incumbent {inc[key]})")
        return json.loads(best["cfg"]), best["run"]
    log(f"[{phase}] keep incumbent {incumbent_name} ({key} {inc[key]}; best challenger {best['run']} {best[key]})")
    return json.loads(inc["cfg"]), incumbent_name


# ============================================================================ unimodal feature checks (P0)
def unimodal_eer(X, spk, gmap, seed=0, n=3000):
    """Speaker verification EER of a single modality (centred cosine). Feature-quality proxy, no training."""
    Z = l2n((X - X.mean(0)) / (X.std(0) + 1e-6))
    out = {}
    for sg in (False, True):
        fi, vj, lab = build_trials(spk, gmap, seed=seed, n_pos=n, n_neg=n, same_gender=sg)
        out["g" if sg else "ng"] = round(eer_from_scores(np.sum(Z[fi] * Z[vj], 1), lab), 2)
    return out


# ============================================================================ submissions
def dev_scores(store, cfg, n_models=10, cca_w=0.25, seed0=1, matrix_path=None):
    """Train on all 70 speakers, score the four dev cells (higher = same).

    matrix_path: also save, per cell, the FULL face x voice score matrix (rows = trial faces, cols = trial
    voices, fp16) fused the same way, except that z-scoring is over the whole matrix instead of over the
    trials (so the diagonal matches the trial scores up to a monotone rescaling). Needed for graph refinement."""
    mats = {}
    cfg = dict(BASE, **cfg)
    Xf, Xv = store.face(cfg["face"], "train"), store.voice(cfg["voice"], "train")
    allidx = np.arange(len(store.spk))
    nets = [train_one(store, cfg, allidx, seed=seed0 + 100 * i) for i in range(n_models)]
    out = {}
    for k in CELLS:
        split = "/".join(k)
        Af, Av = store.face(cfg["face"], split), store.voice(cfg["voice"], split)
        embs = []
        for net, prep in nets:
            embs.append(embed(net, prep, Af, Av, prep.f(Xf).mean(0), prep.v(Xv).mean(0)))
        ii = np.arange(len(Af))
        cca = cca_proj(Xf, Xv, Af, Av) if cca_w > 0 else None
        out[k] = fuse(embs, ii, ii, cca, cca_w)
        if matrix_path is not None:
            zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
            M = np.mean([zm(Ef @ Ev.T) for Ef, Ev in embs], 0)
            if cca is not None:
                M = (1 - cca_w) * M + cca_w * zm(cca[0] @ cca[1].T)
            mats["/".join(k)] = M.astype(np.float16)
    if matrix_path is not None:
        np.savez_compressed(matrix_path, **mats)
    return out


def write_zip(store, path, scores):
    """{(protocol, lang): higher-is-same scores} -> official zip. flag_lib writes -score (LOWER = same)."""
    return L.write_submission(path, scores, store.dev)


def check_zip(store, path):
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
        assert names == sorted(NAMES.values()), names
        for k, fn in NAMES.items():
            d = pd.read_csv(zf.open(fn), sep=" ", header=None, names=["pid", "s"])
            t = store.dev[k][2]
            assert len(d) == len(t) and (d.pid.values == t.pair_id.values).all(), fn
            assert d.s.notna().all() and np.isfinite(d.s).all(), fn
    log("zip OK:", path)
    return True


def merge_cells(standard, gender):
    """no_gender/* from the Standard model, gender/* from the Gender model (plan §5, per-cell)."""
    return {k: (gender if k[0] == "gender" else standard)[k] for k in CELLS}


def polarity_preflight():
    lab = np.r_[np.ones(500), np.zeros(500)]
    good = np.r_[np.random.RandomState(0).normal(3, 1, 500), np.random.RandomState(1).normal(-3, 1, 500)]
    assert eer_from_scores(good, lab) < 5 and eer_from_scores(-good, lab) > 95
    log("polarity: internal scores are higher = same; write_submission writes -score (LOWER = same, "
        "measured on CodaBench: -d2 57.83 vs +d2 42.17 with one checkpoint, docs/DATA.md)")
