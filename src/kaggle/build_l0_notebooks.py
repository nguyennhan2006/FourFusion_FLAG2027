"""Generate the SAFE-L0 notebooks FLAG_18 ... FLAG_22 (plan 30/09: SAFE = every pair scored on its own; baseline to beat
= SET-17 level 0, CB 26.35). Each notebook is self-contained: the full code is in its cells, it imports no team .py, and
its ONLY input is the dataset `flag2027-eda-bundle`. The five notebooks are independent, so they can run in parallel.

    FLAG_18_l0_mcs.ipynb       maximum class separation (Moscati et al., ICASSP 2026; official hcai-mms/MSM-face-voice)
    FLAG_19_l0_paeff.ipynb     PAEFF (Hannan et al., Interspeech 2025; official hannabdul/paeff) + a true hyperbolic InfoNCE
    FLAG_20_l0_verifier.ipynb  learned pair scorers: end-to-end MLP verifier, stream fusion learned out-of-fold
    FLAG_21_l0_hardneg.ipynb   same-gender PK batches (hard negatives for the gender protocol)
    FLAG_22_l0_xvideo.ipynb    cross-clip / cross-video positives, external persons scored at level 0, pretrain -> finetune
    FLAG_23_l0_combo.ipynb     external persons (XALL) x learned / registered stream weights
    FLAG_24_safe_dev.ipynb     dev zips of SET-17 level 0 with the s007 member retrained (v4 / XALL / XALL + weights); also
                               needs the dataset flag2027-safe-dev (build_safe_dev_bundle.py)

This builder imports no team module either (the shared cells below are written out here, ported from
build_cur_notebook.py, whose bridge is bit-identical to kaggle/flag_v2.py).
    python build_l0_notebooks.py
    FLAG_LOCAL=1 FLAG_IN=<dir holding eda_bundle/> FLAG_OUT=<dir> L0_FOLDS=1 L0_NETS=1 L0_STEPS=30 L0_EXACT_EPOCHS=2 \
        python run_notebook_local.py FLAG_18_l0_mcs.ipynb --jupyter
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def nb(cells, path):
    d = {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                    **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
         "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11"}},
         "nbformat": 4, "nbformat_minor": 5}
    (HERE / path).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", path, len(cells), "cells")


PROTOCOL = """
### SAFE-L0 protocol (the same in FLAG_18 ... FLAG_22)

**Level 0:** the score of a pair (face, voice) depends only on that face, that voice and constants measured on the train
rows. There is no statistic of the test file, no clusters and no trial list. The recipe is `BEST/v6_eval/pipeline.py indep()`:
- member = 0.75 * mean over nets of z(net score) + 0.25 * z(CCA-4);
- prod0 = z(0.5 * z(member) + 0.5 * z(ImageBind cos, train means removed)) + 0.1 * z(-|face age - voice age|);
- every z uses (mean, std) of 20 000 random pairs of the fold's v4 train rows.

`CTRL` is the s007 member of SET-17 level 0 (EXP-007 MLP, InfoNCE, PK 32 x 8, 1200 steps, + CCA-4) with its
ImageBind + age recipe. The bundle has no ECAPA-6144, r2mix crops or AudioCLIP, so s010B / r2mix / AudioCLIP are not in
CTRL: the notebooks measure the bridge f(face, voice), which is what every arm changes.

**Folds** (5, paired: every arm sees the same folds, eval trials and reference pairs):
- v4: 30 persons held out (stratified by gender, seed 2000 + fold); the bridge trains on the other 40.
- MAV-Celeb v1/v2: the persons of fold `PFOLD == fold - 1` are held out (= EXTSCALE-01 / CUR-01 folds). The rest (the
  "pool") is used only by the arms that add external persons.

**Eval files per fold** (true labels, 24 voices per person, every voice in one positive and one negative trial, so
50 % positives):
- `v4 ng` / `v4 g`: the positive face comes from another clip of the same person (a row with a different voice vector); the negative face comes from another held-out person, of the same gender for `g`.
- `urdu ng/g`, `hindi ng`, `en_ext ng/g`: the positive face comes from ANOTHER VIDEO (no session shortcut); negatives as above. These are the unheard-language proxies for Bangla.

**Variants scored per arm:**
- `deep0`: the nets only;
- `member0`: + CCA-4;
- `prod0`: + ImageBind + age. **Decision variant.**
- `fuse0`: 0.5 * prod0(CTRL) + 0.5 * prod0(arm), fixed before looking. Tells whether the arm is complementary.

**Gate (registered):** gain >= 1 EER in >= 4/5 folds on `prod0` or `fuse0`, cell by cell. The g cells and the unheard
cells (urdu, hindi) matter most for the 23-EER target.
"""


ENV = r'''
import os, sys, re, io, json, time, shutil, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
OUT = Path(os.environ.get("FLAG_OUT", "/kaggle/working")) / "__OUTDIR__"
(OUT / "scores").mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

# ------------------------------------------------------------------ registered settings (do not tune after looking)
STEPS = int(os.environ.get("L0_STEPS", "1200"))       # optimizer steps of every PK-trained arm
N_NETS = int(os.environ.get("L0_NETS", "3"))
FOLDS = [int(x) for x in os.environ.get("L0_FOLDS", "1,2,3,4,5").split(",")]
ARMS = os.environ.get("L0_ARMS", "__ARMS__").split(",")
EXACT_EPOCHS = int(os.environ["L0_EXACT_EPOCHS"]) if os.environ.get("L0_EXACT_EPOCHS") else None   # smoke test only
M_EVAL, N_V4_HELD, N_REF = 24, 30, 20000
P_PK, K_PK, CAP = 32, 8, 150
W_IB, W_AGE, W_CCA = 0.5, 0.1, 0.25                    # SET-13 / SET-17 level-0 recipe
GATE_GAIN, GATE_WINS = 1.0, 4
OVERLAP = {"v1_complete:shaukat_tarin", "v2_complete:id0073"}   # persons also in v4 (SET-07)
AGE_MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)
print("LOCAL" if LOCAL else "KAGGLE", "| device", DEVICE, "| steps", STEPS, "| nets", N_NETS, "| folds", FOLDS, "| arms", ARMS)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

LOAD = r'''
# ============================================================================ data (one bundle)
man = next(IN.rglob("eda_bundle_manifest.json"), None)
assert man is not None, "attach the dataset flag2027-eda-bundle (kaggle/build_eda_bundle.py)"
B = man.parent
A = lambda n: np.load(B / f"{n}.npy")
T = lambda n: pd.read_csv(B / n)
log("bundle:", B, "|", json.loads(man.read_text())["created"])

# v4 train: one row = the face and the voice of one clip (organiser row order)
V4 = dict(vgg=A("v4_face_vgg"), btc=A("v4_voice_btc"), ibf=A("v4_face_ibv").astype(np.float32),
          iba=A("v4_voice_iba").astype(np.float32), agef=A("v4_face_vitageprob") @ AGE_MIDS,
          agev=A("v4_voice_w2v2agage")[:, 0] * 100)
V4S = T("v4_rows.csv").spk.astype(str).values
GMAP = dict(zip(T("v4_gender.csv").spk, T("v4_gender.csv").gender))          # 'f' / 'm'
VOICE_UID = pd.factorize(pd.Series([r.tobytes() for r in np.round(V4["btc"], 3)]))[0]   # 38 % of v4 rows repeat a voice

# v1 / v2: faces and voices in separate tables; persons merged by name (v1 lists some people twice)
FM_, VM_, FA_, VA_, GEN = [], [], {}, {}, {}
for src in ("v1_complete", "v2_complete"):
    fm, vm = T(f"ext_{src}_face_meta.csv"), T(f"ext_{src}_voice_meta.csv")
    key_of = {}
    sp = B / f"ext_{src}_speakers.csv"
    if sp.exists():
        for r in pd.read_csv(sp).itertuples():
            k = f"{src}:" + str(r.name).strip().lower().replace(" ", "_")
            key_of[r.ids] = k
            GEN[k] = "m" if str(r.gender).lower().startswith("m") else "f"
    key = lambda x, key_of=key_of, src=src: key_of.get(x, f"{src}:{x}")
    FM_.append(fm.assign(p=fm.spk.map(key), src=src)); VM_.append(vm.assign(p=vm.spk.map(key), src=src))
    for k in ("vgg", "ibv", "vitageprob"):
        FA_.setdefault(k, []).append(A(f"ext_{src}_face_{k}").astype(np.float32))
    for k in ("btc", "iba", "w2v2agage", "w2v2aggen"):
        VA_.setdefault(k, []).append(A(f"ext_{src}_voice_{k}").astype(np.float32))
FMETA, VMETA = pd.concat(FM_, ignore_index=True), pd.concat(VM_, ignore_index=True)
FA = {k: np.concatenate(v) for k, v in FA_.items()}; VA = {k: np.concatenate(v) for k, v in VA_.items()}
FA["agef"] = FA["vitageprob"] @ AGE_MIDS; VA["agev"] = VA["w2v2agage"][:, 0] * 100
# LIDSTRICT-01 / LID-AUDIT-01 (user 3/11 "chỉ whisper"): with LID_STRICT=1, the MAV voice FILES of lid_strict.json "mav_files"
# (Whisper-large-v3 vote) never become training rows in ext_rows(); faces stay. No LID_STRICT / no key = unchanged bit for bit.
MAV_DROP = set()
if os.environ.get("LID_STRICT") == "1":
    _ls = [*IN.glob("*/lid_strict.json"), *IN.glob("datasets/*/*/lid_strict.json"), *IN.glob("datasets/*/flag2027-lid-strict-p/**/lid_strict.json"), *IN.glob("flag2027-lid-strict-p/**/lid_strict.json"), Path("Experiment/LIDSTRICT-01/lid_strict.json")]
    _ls = [p for p in _ls if p.exists()] or list(Path("/kaggle/input").rglob("lid_strict.json"))   # T3ENC-01 style fallback
    assert _ls, "LID_STRICT=1 needs lid_strict.json (dataset flag2027-lid-strict-p or Experiment/LIDSTRICT-01/)"
    MAV_DROP = set(json.loads(_ls[0].read_text()).get("mav_files", []))
    if not VMETA.path.str.startswith(("v1/", "v2/")).all():   # CLEANCOMP-01 (3/11): the eda-bundle meta has 'voices/id../..' without the set prefix of lid_strict.json,
        VMETA["path"] = VMETA.src.str[:2] + "/" + VMETA.path   # so MAV_DROP matched nothing (0 of 58); only under LID_STRICT=1, path is used nowhere else
    log("LID_STRICT MAV voice files dropped from training rows:", int(VMETA.path.isin(MAV_DROP).sum()), "of", len(MAV_DROP), "listed |", _ls[0])
male = VA["w2v2aggen"][:, 1] > VA["w2v2aggen"][:, 0]                 # columns (female, male, child)
for p, g in VMETA.assign(m=male).groupby("p").m.mean().items():
    GEN.setdefault(p, "m" if g >= .5 else "f")                       # v2: majority of the predicted calls
GMAP.update(GEN)
PERSONS = sorted(set(VMETA.p) - OVERLAP)
PFOLD = dict(zip(PERSONS, np.random.RandomState(0).permutation(len(PERSONS)) % 5))   # = EXTSCALE-01 / CUR-01 folds
log(f"v4 {len(V4S)} rows / {len(set(V4S))} persons / {VOICE_UID.max() + 1} distinct voices | "
    f"v1+v2 faces {len(FMETA)}, voices {len(VMETA)}, persons {len(PERSONS)}")
'''

LIB = r'''
# ============================================================================ bridge (port of kaggle/flag_v2.py, EXP-007 recipe) + level-0 scoring
REC = dict(pca_f=256, pca_v=None, hid=512, emb=128, drop=0.3, lr=1e-3, wd=1e-2, tau=0.07, epochs=40)
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


class Prep:
    """Standardise; PCA the face to 256 (fitted on training rows)."""
    def __init__(self, Xf, Xv, pca_f=256, pca_v=None):
        self.mf, self.sf = Xf.mean(0), Xf.std(0) + 1e-6
        self.mv, self.sv = Xv.mean(0), Xv.std(0) + 1e-6
        self.Pf = self._pca((Xf - self.mf) / self.sf, pca_f)
        self.Pv = self._pca((Xv - self.mv) / self.sv, pca_v)

    @staticmethod
    def _pca(Z, k):
        if not k or k >= Z.shape[1]:
            return None
        _, _, Vt = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
        return Vt[:k].T.astype(np.float32)

    def f(self, X):
        Z = (X - self.mf) / self.sf
        return (Z if self.Pf is None else Z @ self.Pf).astype(np.float32)

    def v(self, X):
        Z = (X - self.mv) / self.sv
        return (Z if self.Pv is None else Z @ self.Pv).astype(np.float32)


class RawPrep:
    """No preprocessing: the official MSM / PAEFF code feeds the raw encoder features."""
    f = v = staticmethod(lambda X: np.asarray(X, dtype=np.float32))


def branch(d_in, emb, drop, hid):
    return nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, emb))


class Model(nn.Module):
    def __init__(self, d_f, d_v, n_cls, cfg):
        super().__init__()
        self.bf = branch(d_f, cfg["emb"], cfg["drop"], cfg["hid"])
        self.bv = branch(d_v, cfg["emb"], cfg["drop"], cfg["hid"])
        self.W = nn.Parameter(torch.randn(n_cls, cfg["emb"]) * 0.01)    # unused here; kept so the RNG stream matches flag_v2
        self.gdisc = nn.Sequential(nn.Linear(cfg["emb"], 64), nn.ReLU(), nn.Linear(64, 2))
        self.ldisc = nn.Sequential(nn.Linear(cfg["emb"], 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, f, v):
        return F.normalize(self.bf(f), dim=1), F.normalize(self.bv(v), dim=1)

    # level-0 scoring interface shared by every arm: embed each side, then score row-aligned pairs (higher = same)
    def ef(self, x): return F.normalize(self.bf(x), dim=1)
    def ev(self, x): return F.normalize(self.bv(x), dim=1)
    def score(self, a, b): return (a * b).sum(1)


def infonce_S(S, y):
    same = (y[:, None] == y[None, :]).float()
    return 0.5 * ((-(F.log_softmax(S, 1) * same).sum(1) / same.sum(1)).mean() +
                  (-(F.log_softmax(S, 0) * same).sum(0) / same.sum(0)).mean())


def infonce(u, w, y, tau):
    return infonce_S(u @ w.T / tau, y)


def pk_batches(spk_idx, P, K, n_rows, rng):
    speakers = list(spk_idx)
    out = []
    for _ in range(max(1, n_rows // (P * K))):
        chosen = rng.choice(len(speakers), size=min(P, len(speakers)), replace=False)
        out.append(np.concatenate([rng.choice(spk_idx[speakers[c]], size=K, replace=len(spk_idx[speakers[c]]) < K)
                                   for c in chosen]))
    return out


def pk_batches_g(spk_idx, spk_gen, P, K, n_rows, rng, frac):
    """PK batches; with probability `frac` all speakers of a batch share one gender (hard negatives for protocol g).
    A gender with fewer than P speakers (v4 train: about 20 per gender) gives a smaller batch of all its speakers."""
    speakers = list(spk_idx)
    by_g = {g: [i for i, s in enumerate(speakers) if spk_gen.get(s) == g] for g in ("f", "m")}
    out = []
    for _ in range(max(1, n_rows // (P * K))):
        pool = np.arange(len(speakers))
        if rng.rand() < frac:
            g = ("f", "m")[rng.randint(2)]
            if len(by_g[g]) >= 2:
                pool = np.asarray(by_g[g])
        chosen = pool[rng.choice(len(pool), size=min(P, len(pool)), replace=False)]
        out.append(np.concatenate([rng.choice(spk_idx[speakers[c]], size=K, replace=len(spk_idx[speakers[c]]) < K)
                                   for c in chosen]))
    return out


def train_one(TF, TV, spk, seed, epochs, P=P_PK, K=K_PK):
    """= flag_v2.train_one(cfg = EXP-007 recipe + sampler='pk', rows=(TF, TV, spk)); InfoNCE only. The CTRL member."""
    cfg = dict(REC, epochs=epochs)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    prep = Prep(TF, TV, cfg["pca_f"], cfg["pca_v"])
    Ftr = torch.tensor(prep.f(TF), device=DEVICE)
    Vv = torch.stack([torch.tensor(prep.v(TV), device=DEVICE)])
    cls = {s: i for i, s in enumerate(np.unique(spk))}
    Y = torch.tensor([cls[s] for s in spk], device=DEVICE)
    net = Model(Ftr.shape[1], Vv.shape[2], len(cls), cfg).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    n = len(spk)
    spk_idx = {s: np.where(np.asarray(spk) == s)[0] for s in np.unique(spk)}
    P = min(P, len(spk_idx))
    for ep in range(cfg["epochs"]):
        net.train()
        rng.permutation(n); vsel = np.zeros(n, dtype=int)                 # same RNG calls as flag_v2 (full view)
        for b in pk_batches(spk_idx, P, K, n, rng):
            bt = torch.tensor(b, device=DEVICE); vt = torch.tensor(vsel[b], device=DEVICE)
            u, w = net(Ftr[bt], Vv[vt, bt])
            loss = infonce(u, w, Y[bt], cfg["tau"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    return net, prep


def loss_infonce(net, f, v, y, g):
    u, w = net(f, v)
    return infonce(u, w, y, REC["tau"])


def train_pk(TF, TV, spk, seed, epochs, make_net=None, loss_fn=loss_infonce, prep=None, net=None, lr=None,
             spk_gen=None, ghard=0.0, P=P_PK, K=K_PK, cfg_over=None):
    """train_one generalised: same RNG calls, PK sampler, AdamW + cosine. `make_net(d_f, d_v, n_cls)` / `loss_fn(net, f,
    v, y, g)` replace the model / loss; `prep` / `net` continue from a previous phase (pretrain -> finetune); `ghard` =
    share of same-gender batches. With the defaults it reproduces train_one bit for bit (checked in the smoke test)."""
    cfg = dict(REC, epochs=epochs, **(cfg_over or {}))         # cfg_over: FLAG_29 (dropout, weight decay, emb)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    prep = prep if prep is not None else Prep(TF, TV, cfg["pca_f"], cfg["pca_v"])
    Ftr = torch.tensor(prep.f(TF), device=DEVICE); Vtr = torch.tensor(prep.v(TV), device=DEVICE)
    cls = {s: i for i, s in enumerate(np.unique(spk))}
    Y = torch.tensor([cls[s] for s in spk], device=DEVICE)
    G = torch.tensor([1 if (spk_gen or GMAP).get(s) == "m" else 0 for s in spk], device=DEVICE)
    if net is None:
        net = (make_net or (lambda df, dv, nc: Model(df, dv, nc, cfg)))(Ftr.shape[1], Vtr.shape[1], len(cls))
    net = net.to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr or cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    n = len(spk)
    spk_idx = {s: np.where(np.asarray(spk) == s)[0] for s in np.unique(spk)}
    P = min(P, len(spk_idx))
    for ep in range(cfg["epochs"]):
        net.train()
        rng.permutation(n)
        batches = (pk_batches_g(spk_idx, spk_gen or GMAP, P, K, n, rng, ghard) if ghard else pk_batches(spk_idx, P, K, n, rng))
        for b in batches:
            bt = torch.tensor(b, device=DEVICE)
            loss = loss_fn(net, Ftr[bt], Vtr[bt], Y[bt], G[bt])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    return net, prep


def epochs_for(n_rows, steps=None):
    bpe = max(1, n_rows // (P_PK * K_PK))
    return max(1, int(round((steps or STEPS) / bpe))), bpe


def xavier_init(m):
    """init_weights of the official MSM / PAEFF code."""
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight); m.bias.data.fill_(0.01)


def train_official(TF, TV, spk, seed, make_net, loss_fn, epochs, bs, lr, wd, opt="adam", cosine=False):
    """Training loop of the official MSM / PAEFF code: raw features, rows shuffled ONCE, sequential batches of `bs`
    (last partial batch dropped), optimizer stepped per batch, (cosine) scheduler per epoch. No model selection on test
    data (the official scripts keep the best epoch on their test set; we keep the last one)."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(spk))
    cls = {s: i for i, s in enumerate(np.unique(spk))}
    Xf = torch.tensor(np.asarray(TF, np.float32)[perm], device=DEVICE); Xv = torch.tensor(np.asarray(TV, np.float32)[perm], device=DEVICE)
    Y = torch.tensor([cls[s] for s in np.asarray(spk)[perm]], device=DEVICE)
    net = make_net(Xf.shape[1], Xv.shape[1], len(cls)).to(DEVICE)
    net.apply(xavier_init)
    O = torch.optim.Adam if opt == "adam" else torch.optim.AdamW
    optim = O(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-8) if cosine else None
    nb = max(1, len(Y) // bs)
    for ep in range(epochs):
        net.train()
        for i in range(nb):
            sl = slice(i * bs, (i + 1) * bs)
            loss = loss_fn(net, Xf[sl], Xv[sl], Y[sl])
            optim.zero_grad(); loss.backward(); optim.step()
        if sched is not None:
            sched.step()
    net.eval()
    return net, RawPrep(), nb * epochs


class RidgeCCA:
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
        A_ = ((X - self.mx) / self.sx) @ self.Px; B_ = ((Y - self.my) / self.sy) @ self.Py
        n = len(A_)
        Cxx = A_.T @ A_ / n + self.reg * np.eye(A_.shape[1]); Cyy = B_.T @ B_ / n + self.reg * np.eye(B_.shape[1])
        Lxi = np.linalg.inv(np.linalg.cholesky(Cxx)); Lyi = np.linalg.inv(np.linalg.cholesky(Cyy))
        U, S, Vt = np.linalg.svd(Lxi @ (A_.T @ B_ / n) @ Lyi.T, full_matrices=False)
        self.Wx = Lxi.T @ U[:, :self.k]; self.Wy = Lyi.T @ Vt[:self.k].T
        return self

    def transform_x(self, X): return l2n((((X - self.mx) / self.sx) @ self.Px) @ self.Wx)
    def transform_y(self, Y): return l2n((((Y - self.my) / self.sy) @ self.Py) @ self.Wy)


def eer(scores, labels):
    """EER %, higher = same; the FAR = FRR crossing is interpolated (as flag_lib.eer_from_scores)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores); fnr = 1 - tpr; d = fnr - fpr
    i = np.nanargmin(np.abs(d))
    if d[i] == 0 or len(fpr) < 2:
        return 100 * (fpr[i] + fnr[i]) / 2
    j = i - 1 if (i > 0 and d[i - 1] * d[i] < 0) else min(i + 1, len(d) - 1)
    if d[j] * d[i] >= 0:
        return 100 * (fpr[i] + fnr[i]) / 2
    t = d[i] / (d[i] - d[j])
    return 100 * float(fpr[i] + t * (fpr[j] - fpr[i]))


class Std:
    """z-score with constants measured on the reference train pairs only (pipeline.py Std)."""
    def __init__(self, ref):
        self.m, self.s = float(np.mean(ref)), float(np.std(ref) + 1e-9)

    def __call__(self, x):
        return (x - self.m) / self.s


class Member:
    """The trained nets of one arm (+ CCA-4 on the same training rows). Every net has ef / ev / score."""
    def __init__(self, units, TF, TV):
        self.units, self.cca = units, RidgeCCA().fit(TF, TV)

    @torch.no_grad()
    def embed(self, unit, Xf, Xv, chunk=4096):
        net, prep = unit
        E = []
        for X, fn, pf in ((Xf, net.ef, prep.f), (Xv, net.ev, prep.v)):
            E.append(torch.cat([fn(torch.tensor(pf(X[i:i + chunk]), device=DEVICE)) for i in range(0, len(X), chunk)]))
        return E

    @torch.no_grad()
    def score(self, unit, Ef, Ev, chunk=8192):
        net = unit[0]
        return torch.cat([net.score(Ef[i:i + chunk], Ev[i:i + chunk]) for i in range(0, len(Ef), chunk)]).float().cpu().numpy()

    def pair(self, unit, Xf, Xv):
        return self.score(unit, *self.embed(unit, Xf, Xv))


class L0:
    """Level-0 score of row-aligned trials (BEST/v6_eval/pipeline.py indep(), s007 recipe): every constant comes from
    N_REF random pairs of the fold's v4 TRAIN rows; nothing is computed from the test trials. `final(streams)` lets an
    arm replace the fixed recipe (learned fusion, FLAG_20)."""
    def __init__(self, mem, v4rows, final=None, X4=None):
        X4 = V4 if X4 is None else X4                     # FLAG_28: v4 arrays with another face / voice / ImageBind embedding
        rr = np.random.RandomState(0)
        rf, rv = rr.randint(len(v4rows), size=N_REF), rr.randint(len(v4rows), size=N_REF)
        Xf, Xv = X4["vgg"][v4rows], X4["btc"][v4rows]
        self.mem, self.final, self.sd = mem, final, []
        parts = []
        for u in mem.units:
            Ef, Ev = mem.embed(u, Xf, Xv)
            r = mem.score(u, Ef[rf], Ev[rv]); self.sd.append(Std(r)); parts.append(self.sd[-1](r))
        deep = np.mean(parts, 0)
        c = np.sum(mem.cca.transform_x(Xf)[rf] * mem.cca.transform_y(Xv)[rv], 1); self.zc = Std(c)
        member = (1 - W_CCA) * deep + W_CCA * self.zc(c); self.zm = Std(member)
        IBF, IBA = X4["ibf"][v4rows], X4["iba"][v4rows]
        self.mf, self.mv = IBF.mean(0), IBA.mean(0)
        ib = np.sum(l2n(IBF[rf] - self.mf) * l2n(IBA[rv] - self.mv), 1); self.zib = Std(ib)
        age = -np.abs(X4["agef"][v4rows][rf] - X4["agev"][v4rows][rv]); self.zag = Std(age)
        mix = (1 - W_IB) * self.zm(member) + W_IB * self.zib(ib); self.zmix = Std(mix)
        self.zfin = Std(self.zmix(mix) + W_AGE * self.zag(age))

    def streams(self, T):
        deep = np.mean([sd(self.mem.pair(u, T["vgg"], T["btc"])) for u, sd in zip(self.mem.units, self.sd)], 0)
        cca = self.zc(np.sum(self.mem.cca.transform_x(T["vgg"]) * self.mem.cca.transform_y(T["btc"]), 1))
        member = self.zm((1 - W_CCA) * deep + W_CCA * cca)
        ib = self.zib(np.sum(l2n(T["ibf"] - self.mf) * l2n(T["iba"] - self.mv), 1))
        age = self.zag(-np.abs(T["agef"] - T["agev"]))
        prod = self.zfin(self.zmix((1 - W_IB) * member + W_IB * ib) + W_AGE * age)
        S = dict(deep0=deep, member0=member, prod0=prod, cca=cca, ib=ib, age=age)
        if self.final is not None:
            S["prod0"] = self.final(S)
        return S
'''

DATA = r'''
# ============================================================================ folds, training rows, eval files (identical in every L0 notebook)
def trial_dict(Fsrc, Vsrc, Fi, Vi, y, pf, pv):
    return dict(vgg=Fsrc["vgg"][Fi], ibf=Fsrc["ibf"][Fi], agef=Fsrc["agef"][Fi], btc=Vsrc["btc"][Vi], iba=Vsrc["iba"][Vi],
                agev=Vsrc["agev"][Vi], y=np.asarray(y), pf=np.asarray(pf), pv=np.asarray(pv),
                fi=np.asarray(Fi), vi=np.asarray(Vi))                  # source row indices (FLAG_26 error analysis)


def v4_file(held, sg, rng, m=M_EVAL):
    """v4 held-out persons: each voice (distinct clip) gets one positive (a face of the same person from a row with a
    DIFFERENT voice) and one negative (a face of another held-out person; same gender if sg)."""
    rows_of = {p: np.where(V4S == p)[0] for p in held}
    Fi, Vi, Y = [], [], []
    for p in held:
        r = rows_of[p]
        _, first = np.unique(VOICE_UID[r], return_index=True)
        others = [q for q in held if q != p and (not sg or GMAP[q] == GMAP[p])]
        for v in rng.permutation(r[first])[:m]:
            cand = r[VOICE_UID[r] != VOICE_UID[v]]
            if not len(cand) or not others:
                continue
            q = others[rng.randint(len(others))]
            Fi += [rng.choice(cand), rng.choice(rows_of[q])]; Vi += [v, v]; Y += [1, 0]
    Fi, Vi = np.array(Fi, int), np.array(Vi, int)
    return trial_dict(V4, V4, Fi, Vi, Y, V4S[Fi], V4S[Vi])


EXTF = dict(vgg=FA["vgg"], ibf=FA["ibv"], agef=FA["agef"])
EXTV = dict(btc=VA["btc"], iba=VA["iba"], agev=VA["agev"])


def ext_file(held, lang, sg, rng, m=M_EVAL):
    """v1/v2 held-out persons, voices in `lang`: the positive face comes from ANOTHER VIDEO of the same person."""
    vm = VMETA[(VMETA.lang == lang) & VMETA.p.isin(held)]
    face_of = {p: FMETA.index[FMETA.p == p].values for p in held}
    Fi, Vi, Y = [], [], []
    ps = sorted(vm.p.unique())
    for p, g in vm.groupby("p"):
        others = [q for q in ps if q != p and len(face_of[q]) and (not sg or GMAP.get(q) == GMAP.get(p))]
        if not others:
            continue
        for v in rng.permutation(g.index.values)[:m]:
            cand = face_of[p][FMETA.video.values[face_of[p]] != VMETA.video[v]]
            if not len(cand):
                continue
            q = others[rng.randint(len(others))]
            Fi += [rng.choice(cand), rng.choice(face_of[q])]; Vi += [v, v]; Y += [1, 0]
    if len(Y) < 40:
        return None
    Fi, Vi = np.array(Fi, int), np.array(Vi, int)
    return trial_dict(EXTF, EXTV, Fi, Vi, Y, FMETA.p.values[Fi], VMETA.p.values[Vi])


def make_fold(s):
    v4p = np.unique(V4S)
    rng = np.random.RandomState(2000 + s)
    held = []
    for g in ("f", "m"):
        ps = [p for p in v4p if GMAP[p] == g]
        held += list(rng.permutation(ps)[:int(round(N_V4_HELD * len(ps) / len(v4p)))])
    held = sorted(held); train_p = sorted(set(v4p) - set(held))
    fold = dict(s=s, v4_held=held, v4_train=train_p, v4tr=np.where(np.isin(V4S, train_p))[0],
                ext_held=[p for p in PERSONS if PFOLD[p] == s - 1], ext_pool=[p for p in PERSONS if PFOLD[p] != s - 1])
    files = {}
    r4 = np.random.RandomState(3000 + s)
    files[("v4", "ng")] = v4_file(held, False, r4); files[("v4", "g")] = v4_file(held, True, r4)
    r5 = np.random.RandomState(4000 + s)
    for lang in ("urdu", "hindi", "english"):
        for sg in (False, True):
            if lang == "hindi" and sg:                                   # v2 genders are predicted: no g file (as CUR-01)
                continue
            d = ext_file(fold["ext_held"], lang, sg, r5)
            if d is not None:
                files[("en_ext" if lang == "english" else lang, "g" if sg else "ng")] = d
    fold["files"] = files
    return fold


def ext_rows(persons, rng, cap=CAP, xvid=False, fa=None, va=None):
    """Organiser-space training rows of external persons (as flag_v2.pair_ext): one per voice clip (<= cap per person),
    paired with a face of the SAME video (cycling; any face of the person if that video has none). xvid: a face of
    ANOTHER video of the person instead (any face if the person has a single video)."""
    vm = VMETA[VMETA.p.isin(persons) & ~VMETA.path.isin(MAV_DROP)]   # MAV_DROP: LOAD (empty unless LID_STRICT=1)
    fm = FMETA[FMETA.p.isin(persons)]
    by_vid = {k: np.asarray(g.index) for k, g in fm.groupby(["p", "lang", "video"])}
    by_p = {k: np.asarray(g.index) for k, g in fm.groupby("p")}
    Fi, Vi, S, fallback = [], [], [], 0
    for p, g in vm.groupby("p"):
        if p not in by_p:
            continue
        idx = np.asarray(g.index)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        for j, i in enumerate(idx):
            if xvid:
                cand = by_p[p][FMETA.video.values[by_p[p]] != VMETA.video[i]]
                if not len(cand):
                    cand = by_p[p]; fallback += 1
                Fi.append(cand[rng.randint(len(cand))])
            else:
                cand = by_vid.get((p, VMETA.lang[i], VMETA.video[i]), by_p[p])
                Fi.append(cand[j % len(cand)])
            Vi.append(i); S.append(p)
    return (FA["vgg"] if fa is None else fa)[Fi], (VA["btc"] if va is None else va)[Vi], np.array(S), fallback


def decouple_faces(rows, rng):
    """For each v4 row, the face of ANOTHER row of the same person whose voice is a different clip (the row itself if none):
    positives stop being 'face and voice cut from the same clip'."""
    out = rows.copy()
    for i, r in enumerate(rows):
        same = rows[(V4S[rows] == V4S[r]) & (VOICE_UID[rows] != VOICE_UID[r])]
        if len(same):
            out[i] = same[rng.randint(len(same))]
    return out


def train_rows(fold, ext=False, xvid=False, decouple=False, rng=None, arrays=None):
    """arrays = (v4 face, v4 voice, ext face, ext voice) to train on other embeddings (FLAG_28); None = VGG / BTC."""
    rng = rng if rng is not None else np.random.RandomState(4000 + fold["s"])       # external rows: same draw in every arm
    f4, v4_, fe, ve = arrays if arrays is not None else (V4["vgg"], V4["btc"], None, None)
    idx = fold["v4tr"]
    TF, TV, TS = f4[decouple_faces(idx, np.random.RandomState(5000 + fold["s"])) if decouple else idx], v4_[idx], V4S[idx]
    info = dict(fallback=0)
    if ext:
        EF, EV, ES, fb = ext_rows(fold["ext_pool"], rng, xvid=xvid, fa=fe, va=ve)
        TF, TV, TS = np.concatenate([TF, EF]), np.concatenate([TV, EV]), np.concatenate([TS, ES])
        info["fallback"] = fb
    return TF, TV, TS, info


def arm_ctrl(fold, ext=False):
    """CTRL (= s007 member of SET-17 level 0): EXP-007 MLP, InfoNCE, PK 32 x 8, STEPS steps, N_NETS seeds."""
    TF, TV, TS, info = train_rows(fold, ext=ext)
    ep, bpe = epochs_for(len(TS))
    units = [train_one(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe)
'''

RUN = r'''
# ============================================================================ run every (fold, arm); resumable
RES = OUT / "results.csv"
prev = [p for p in IN.rglob("results.csv") if OUT not in p.parents and p.parent.name == OUT.name]
if prev and not RES.exists():
    shutil.copy(prev[0], RES)
    for f in prev[0].parent.glob("scores/*.npz"):
        shutil.copy(f, OUT / "scores" / f.name)
    log("resume from", prev[0])
recs = pd.read_csv(RES).to_dict("records") if RES.exists() else []
done = {(r["fold"], r["arm"]) for r in recs}
assert ARMS[0] == "CTRL", "CTRL must run first in every fold (fuse0 needs its scores)"
for s in FOLDS:
    if all((s, a) in done for a in ARMS):
        continue
    t0 = time.time(); fold = make_fold(s)
    log(f"fold {s}: v4 train {len(fold['v4_train'])} / held {len(fold['v4_held'])}, ext held {len(fold['ext_held'])} / "
        f"pool {len(fold['ext_pool'])}; files " + ", ".join(f"{a} {b} ({len(d['y'])})" for (a, b), d in fold["files"].items())
        + f"  [{time.time() - t0:.0f}s]")
    for arm in ARMS:
        if (s, arm) in done:
            continue
        t0 = time.time()
        out = TRAIN[arm](fold)
        if isinstance(out, dict):                                            # FLAG_28: other embeddings
            mem, info, final, X4, rt = out["mem"], out["info"], out.get("final"), out.get("X4"), out.get("retarget")
        else:
            (mem, info, final), X4, rt = (out if len(out) == 3 else (*out, None)), None, None
        sc = L0(mem, fold["v4tr"], final, X4=X4)
        ctrl_file = OUT / "scores" / f"fold{s}_CTRL.npz"
        ctrl = dict(np.load(ctrl_file)) if ctrl_file.exists() else {}
        keep = {}
        for (data, prot), T_ in fold["files"].items():
            T_ = rt(T_, data) if rt is not None else T_
            S_ = sc.streams(T_)
            k = f"{data}_{prot}"
            keep[f"{k}_y"] = T_["y"]
            keep[f"{k}_ib"] = S_["ib"].astype(np.float32)
            for var in ("deep0", "member0", "prod0"):
                keep[f"{k}_{var}"] = S_[var].astype(np.float32)
            if arm == "CTRL":
                S_["fuse0"] = S_["prod0"]                                    # CTRL fused with itself
            elif f"{k}_prod0" in ctrl:
                S_["fuse0"] = 0.5 * (ctrl[f"{k}_prod0"] + S_["prod0"])
            for var in ("deep0", "member0", "prod0", "fuse0"):
                if var in S_:
                    recs.append(dict(fold=s, arm=arm, data=data, protocol=prot, variant=var, n=len(T_["y"]),
                                     eer=eer(S_[var], T_["y"]), **{k_: v for k_, v in info.items() if np.isscalar(v)}))
        np.savez_compressed(OUT / "scores" / f"fold{s}_{arm}.npz", **keep)
        pd.DataFrame(recs).to_csv(RES, index=False)
        msg = " ".join(f"{r['data']}/{r['protocol']} {r['eer']:.2f}" for r in recs if r["fold"] == s and r["arm"] == arm and r["variant"] == "prod0")
        log(f"fold {s} {arm:14s} ids {info.get('train_ids', '?')} rows {info.get('train_rows', '?')} steps {info.get('steps', '?')} "
            f"[{time.time() - t0:.0f}s] prod0: {msg}")
'''

SUMMARY = r'''
# ============================================================================ paired summary (+ = arm better than its reference)
R_ = pd.read_csv(RES)
R_["cell"] = R_.data + " " + R_.protocol
W = R_.pivot_table(index=["variant", "cell", "fold"], columns="arm", values="eer")
print("CTRL level-0 EER (mean over folds):")
print(W["CTRL"].groupby(level=[0, 1]).mean().unstack(0).round(2)[["deep0", "member0", "prod0"]].to_string())
rows = []
for a, b in COMPARISONS:
    if a not in W or b not in W:
        continue
    for (var, cell), g in (W[b] - W[a]).groupby(level=[0, 1]):
        g = g.dropna()
        if len(g):
            rows.append(dict(comparison=f"{a} vs {b}", variant=var, cell=cell, gain=round(g.mean(), 2),
                             wins=int((g > 0).sum()), folds=len(g), gate=bool(g.mean() >= GATE_GAIN and (g >= GATE_GAIN).sum() >= GATE_WINS)))
S_ = pd.DataFrame(rows); S_.to_csv(OUT / "summary.csv", index=False)
for var in ("prod0", "fuse0", "member0"):
    sv = S_[S_.variant == var]
    if not len(sv):
        continue
    print(f"\n==== {var}: gain (EER, + = better)  |  wins / folds")
    print(sv.pivot_table(index="cell", columns="comparison", values="gain").round(2).to_string())
    print(sv.assign(wf=sv.wins.astype(str) + "/" + sv.folds.astype(str)).pivot_table(
        index="cell", columns="comparison", values="wf", aggfunc="first").to_string())
passed = S_[S_.gate & S_.variant.isin(["prod0", "fuse0"])]
print("\nGATE (>= %.1f EER in >= %d folds):" % (GATE_GAIN, GATE_WINS), "none" if not len(passed) else "")
if len(passed):
    print(passed[["comparison", "variant", "cell", "gain", "wins", "folds"]].to_string(index=False))
print(f"\nNEXT: download /kaggle/working/{OUT.name} (Output tab) and hand it back: results.csv, summary.csv, scores/*.npz")
'''

# ----------------------------------------------------------------------------------------------- FLAG_18: MCS
MCS_INTRO = """
# FLAG 2027 — NB-18: maximum class separation at level 0 (MCS-01)

**Question:** does the inductive bias of **Moscati et al., ICASSP 2026** ("Face-voice association with inductive bias for maximum class separation", official code `hcai-mms/MSM-face-voice`) make a better face ↔ voice bridge than our InfoNCE member, when every pair is scored on its own?
The method: the fused embedding m = w1 * face + w2 * voice has dimension Ns - 1, and the logits are m @ P. P is the fixed (Ns - 1) x Ns matrix of maximally separated class vectors (Kasarla et al., NeurIPS 2022; pairwise cosine -1/(Ns - 1)). Loss = CE + alpha * OPL. The paper reports 22.9 EER unseen-unheard on VoxCeleb (FOP 24.9).

| Arm | What | Differs from |
|---|---|---|
| `CTRL` | s007 member (EXP-007 MLP, InfoNCE, PK, 1200 steps, emb 128) | — |
| `MSM_EXACT` | official code: branch Linear(d, Ns-1) + BN + ReLU + Dropout 0.91 + L2; linear fusion (w1, w2 learned); logits = m @ P; CE + 1.0 * OPL (neg weight 0.7); Adam 1e-4, wd 0.01, batch 512, 49 epochs, raw features, rows shuffled once; score = cos(face, voice) | the paper |
| `FOP_EXACT` | `MSM_EXACT` with a learned linear classifier instead of P (the paper's FOP* ablation) | `MSM_EXACT`: the matrix only |
| `CTRL_D` | `CTRL` with emb = Ns - 1 (the dimension MSM needs) | `CTRL`: the dimension only |
| `MCS_INFO` | `CTRL_D` + 1.0 * (CE(m @ P) + OPL(m)), m = w1 u + w2 w | `CTRL_D`: the MCS head only |

Registered comparisons: MSM_EXACT / FOP_EXACT / MCS_INFO vs CTRL; MSM_EXACT vs FOP_EXACT; MCS_INFO vs CTRL_D.
Decision: the pasted plan's rule. Within about ±0.5 → close; > 1 EER, stable (gate) → keep.
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** about 2-3 min per fold x arm on a T4, about 1 h in total; CPU about 3 h. **Output:** `/kaggle/working/l0_mcs/`. **Resume:** attach this notebook's earlier output; finished (fold, arm) pairs are skipped.
"""

MCS_ARMS = r'''
# ============================================================================ MCS arms
def maxsep(n):
    """Kasarla et al. (2022) eq. 1: (n-1) x n matrix, unit columns, pairwise cosine -1/(n-1)."""
    P = np.array([[1.0, -1.0]])
    for k in range(2, n):
        P = np.block([[np.ones((1, 1)), -np.ones((1, k)) / k], [np.zeros((k - 1, 1)), np.sqrt(1 - 1 / k ** 2) * P]])
    return P.astype(np.float32)


_P = maxsep(40)
assert np.allclose(np.linalg.norm(_P, axis=0), 1, atol=1e-5) and np.allclose((_P.T @ _P)[~np.eye(40, dtype=bool)], -1 / 39, atol=1e-5)


def opl(features, labels):
    """OrthogonalProjectionLoss of the official code (features L2-normalised inside; negative weight 0.7)."""
    f = F.normalize(features, p=2, dim=1)
    mask = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=f.device)
    pos, neg = (mask & ~eye).float(), (~mask).float()
    dot = f @ f.T
    return (1.0 - (pos * dot).sum() / (pos.sum() + 1e-6)) + 0.7 * (neg * dot).abs().sum() / (neg.sum() + 1e-6)


class MSMNet(nn.Module):
    """retrieval_model.FOP of hcai-mms/MSM-face-voice with linear fusion. matrix=False: learned classifier (FOP*)."""
    def __init__(self, d_f, d_v, n_cls, matrix=True):
        super().__init__()
        D = n_cls - 1
        mk = lambda d: nn.Sequential(nn.Linear(d, D), nn.BatchNorm1d(D), nn.ReLU(inplace=True), nn.Dropout(p=0.91))
        self.bv, self.bf = mk(d_v), mk(d_f)
        self.w1, self.w2 = nn.Parameter(torch.rand(1)), nn.Parameter(torch.rand(1))
        if matrix:
            self.register_buffer("P", torch.tensor(maxsep(n_cls)))
        else:
            self.P, self.cls = None, nn.Linear(D, n_cls)

    def ef(self, x): return F.normalize(self.bf(x), dim=1)
    def ev(self, x): return F.normalize(self.bv(x), dim=1)
    def score(self, a, b): return (a * b).sum(1)

    def forward(self, f, v):
        u, w = self.ef(f), self.ev(v)
        m = self.w1 * u + self.w2 * w
        return u, w, m, (m @ self.P if self.P is not None else self.cls(m))


def loss_msm(net, f, v, y, alpha=1.0):
    u, w, m, logits = net(f, v)
    return F.cross_entropy(logits, y) + alpha * opl(m, y)


class MCSInfo(Model):
    """EXP-007 MLP (emb = Ns - 1) + the MCS head on m = w1 u + w2 w."""
    def __init__(self, d_f, d_v, n_cls, cfg):
        super().__init__(d_f, d_v, n_cls, cfg)
        self.w1, self.w2 = nn.Parameter(torch.rand(1)), nn.Parameter(torch.rand(1))
        self.register_buffer("P", torch.tensor(maxsep(n_cls)))


def loss_mcs_info(net, f, v, y, g, lam=1.0):
    u, w = net(f, v)
    m = net.w1 * u + net.w2 * w
    return infonce(u, w, y, REC["tau"]) + lam * (F.cross_entropy(m @ net.P, y) + opl(m, y))


def arm_msm_exact(fold, matrix=True):
    TF, TV, TS, info = train_rows(fold)
    ep = EXACT_EPOCHS or 49                                             # official loop: while epoch < 50, from 1
    units, steps = [], 0
    for i in range(N_NETS):
        net, prep, steps = train_official(TF, TV, TS, fold["s"] + 100 * i, lambda df, dv, nc: MSMNet(df, dv, nc, matrix),
                                          lambda n_, f, v, y: loss_msm(n_, f, v, y), ep, bs=512, lr=1e-4, wd=0.01, opt="adam")
        units.append((net, prep))
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=steps)


def arm_pk(fold, make_net, loss_fn=loss_infonce):
    TF, TV, TS, info = train_rows(fold)
    ep, bpe = epochs_for(len(TS))
    units = [train_pk(TF, TV, TS, fold["s"] + 100 * i, ep, make_net=make_net, loss_fn=loss_fn) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe)


cfgD = lambda nc: dict(REC, emb=nc - 1)
TRAIN = {
    "CTRL": arm_ctrl,
    "MSM_EXACT": lambda fold: arm_msm_exact(fold, True),
    "FOP_EXACT": lambda fold: arm_msm_exact(fold, False),
    "CTRL_D": lambda fold: arm_pk(fold, lambda df, dv, nc: Model(df, dv, nc, cfgD(nc))),
    "MCS_INFO": lambda fold: arm_pk(fold, lambda df, dv, nc: MCSInfo(df, dv, nc, cfgD(nc)), loss_mcs_info),
}
COMPARISONS = [("MSM_EXACT", "CTRL"), ("FOP_EXACT", "CTRL"), ("CTRL_D", "CTRL"), ("MCS_INFO", "CTRL"),
               ("MSM_EXACT", "FOP_EXACT"), ("MCS_INFO", "CTRL_D")]
'''

# ----------------------------------------------------------------------------------------------- FLAG_19: PAEFF
PAEFF_INTRO = """
# FLAG 2027 — NB-19: PAEFF and hyperbolic alignment at level 0 (PAEFF-01)

**Question:** does **PAEFF (Hannan et al., Interspeech 2025**, official code `hannabdul/paeff`) beat our InfoNCE member when every pair is scored on its own?
The method: face and voice branches, then two parts:
- **Alignment:** a symmetric CLIP loss on the embeddings mapped to the Poincaré ball.
- **EGFF fusion:** x = u * a + (1 - a) * w, with a = sigmoid(conv1d(gelu(u) * gelu(w))), then tanh(Linear). CE + OPL are applied on the fused feature.

The paper reports 22.9 EER unseen-unheard on VoxCeleb (FOP 24.9).

**Note from reading the official code:** the branches are L2-normalised before `ToPoincare`, so every point lands on the same radius. The Poincaré dot product in the CLIP loss is then a rescaled cosine: the "hyperbolic" part acts as a temperature change. `HYP_INFO` is the version where geometry can matter: unnormalised embeddings (norm clipped at 2.3) mapped to the ball, logits = -scale * Poincaré distance.

| Arm | What | Differs from |
|---|---|---|
| `CTRL` | s007 member | — |
| `PAEFF_EXACT` | official code: branch Linear(d,256)+BN+ReLU+Dropout .4+Linear(256,256)+L2; Poincaré c = 0.05 (+ Riemannian gradient); EGFF; loss 0.35 CE + 0.35 OPL(fused) + 0.3 CLIP(τ .07, Poincaré points); AdamW 2e-5, wd .01, cosine, batch 1024, 59 epochs, raw features; score = -‖gelu(u) - gelu(w)‖² | the paper |
| `PAEFF_B` | `PAEFF_EXACT` with lr 1e-3 and about 1200 steps (same number of steps as CTRL). With about 3 700 rows, the official 2e-5 x 177 steps barely trains | `PAEFF_EXACT`: optimisation budget only |
| `HYP_INFO` | `CTRL` with the InfoNCE logits replaced by -s * d_Poincaré(u, w), c = 0.05, norm clip 2.3, s learned (init 5); score = -d_Poincaré | `CTRL`: geometry only |

Registered comparisons: each arm vs CTRL; PAEFF_B vs PAEFF_EXACT.
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** about 1 h on a T4, 3 h CPU. **Output:** `/kaggle/working/l0_paeff/`. **Resume:** attach this notebook's earlier output.
"""

PAEFF_ARMS = r'''
# ============================================================================ PAEFF arms
def opl(features, labels):
    """OrthogonalProjectionLoss of the official code (negative weight 0.7)."""
    f = F.normalize(features, p=2, dim=1)
    mask = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=f.device)
    pos, neg = (mask & ~eye).float(), (~mask).float()
    dot = f @ f.T
    return (1.0 - (pos * dot).sum() / (pos.sum() + 1e-6)) + 0.7 * (neg * dot).abs().sum() / (neg.sum() + 1e-6)


class RiemannianGrad(torch.autograd.Function):
    """pmath.RiemannianGradient of the official code: backward scales the gradient by (1 - c ||x||^2)^2 / 4."""
    c = 0.05

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (1 - RiemannianGrad.c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4


def to_poincare(u, c=0.05, grad_fix=False):
    """expmap0 + project (pmath._expmap0 / _project); tanh clamped at 15 as in pmath.tanh."""
    sc = c ** 0.5
    n = u.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    x = (sc * n).clamp(-15, 15).tanh() * u / (sc * n)
    xn = x.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    maxn = (1 - 1e-3) / sc
    x = torch.where(xn > maxn, x / xn * maxn, x)
    return RiemannianGrad.apply(x) if grad_fix else x


def mobius_add(x, y, c):
    xy = (x * y).sum(-1, keepdim=True); x2 = x.pow(2).sum(-1, keepdim=True); y2 = y.pow(2).sum(-1, keepdim=True)
    return ((1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y) / (1 + 2 * c * xy + c ** 2 * x2 * y2).clamp_min(1e-5)


def poincare_dist(x, y, c=0.05):
    """Row-aligned (or broadcast) Poincaré distance 2/sqrt(c) artanh(sqrt(c) ||-x (+) y||)."""
    sc = c ** 0.5
    z = (sc * mobius_add(-x, y, c).norm(dim=-1)).clamp(max=1 - 1e-5)
    return 2 / sc * torch.atanh(z)


class PAEFFNet(nn.Module):
    """retrieval_model.PAEFF of hannabdul/paeff (dim_embed 256, ToPoincare c = 0.05)."""
    def __init__(self, d_f, d_v, n_cls, D=256, c=0.05):
        super().__init__()
        mk = lambda d: nn.Sequential(nn.Linear(d, D), nn.BatchNorm1d(D), nn.ReLU(inplace=True), nn.Dropout(p=0.4), nn.Linear(D, D))
        self.bv, self.bf = mk(d_v), mk(d_f)
        self.conv = nn.Conv1d(1, 1, 5, padding=2)
        self.res_mix = nn.Linear(D, D)
        self.logits_layer = nn.Linear(D, n_cls)
        self.c = c

    def branches(self, f, v):
        return F.normalize(self.bf(f), dim=1), F.normalize(self.bv(v), dim=1)

    def forward(self, f, v):
        u, w = self.branches(f, v)
        ff, vf = F.gelu(u), F.gelu(w)
        attn = torch.sigmoid(self.conv((ff * vf).unsqueeze(1)).squeeze(1))
        fused = torch.tanh(self.res_mix(u * attn + (1 - attn) * w))
        return fused, self.logits_layer(fused), to_poincare(u, self.c, True), to_poincare(w, self.c, True)

    # test time (online_evaluation.test): the gelu'd branch outputs, squared L2 distance
    def ef(self, x): return F.gelu(F.normalize(self.bf(x), dim=1))
    def ev(self, x): return F.gelu(F.normalize(self.bv(x), dim=1))
    def score(self, a, b): return -((a - b) ** 2).sum(1)


def clip_loss(a, b):
    logits = (a @ b.T) * (1.0 / 0.07)
    lab = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2


def loss_paeff(net, f, v, y, lw=(0.35, 0.35, 0.3)):
    fused, logits, uh, wh = net(f, v)
    return lw[0] * F.cross_entropy(logits, y) + lw[1] * opl(fused, y) + lw[2] * clip_loss(uh, wh)


class HypModel(Model):
    """EXP-007 MLP whose embeddings live in the Poincaré ball: no L2 norm, norm clipped at r, expmap0 (c), logits =
    -s * distance with s learned (init 5, capped at 100)."""
    def __init__(self, d_f, d_v, n_cls, cfg, c=0.05, r=2.3):
        super().__init__(d_f, d_v, n_cls, cfg)
        self.c, self.r = c, r
        self.log_s = nn.Parameter(torch.tensor(math.log(5.0)))

    def _h(self, x):
        return to_poincare(x * torch.clamp(self.r / x.norm(dim=1, keepdim=True).clamp_min(1e-5), max=1.0), self.c)

    def ef(self, x): return self._h(self.bf(x))
    def ev(self, x): return self._h(self.bv(x))
    def score(self, a, b): return -poincare_dist(a, b, self.c)
    def forward(self, f, v): return self.ef(f), self.ev(v)


def loss_hyp(net, f, v, y, g):
    u, w = net(f, v)
    S = -net.log_s.exp().clamp(max=100) * poincare_dist(u[:, None, :], w[None, :, :], net.c)
    return infonce_S(S, y)


def arm_paeff(fold, lr, target_steps=None):
    TF, TV, TS, info = train_rows(fold)
    nb = max(1, len(TS) // 1024)
    ep = EXACT_EPOCHS or (int(round(target_steps / nb)) if target_steps else 59)   # official loop: while epoch < 60, from 1
    units, steps = [], 0
    for i in range(N_NETS):
        net, prep, steps = train_official(TF, TV, TS, fold["s"] + 100 * i, lambda df, dv, nc: PAEFFNet(df, dv, nc),
                                          lambda n_, f, v, y: loss_paeff(n_, f, v, y), ep, bs=1024, lr=lr, wd=0.01,
                                          opt="adamw", cosine=True)
        units.append((net, prep))
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=steps)


def arm_hyp(fold):
    TF, TV, TS, info = train_rows(fold)
    ep, bpe = epochs_for(len(TS))
    units = [train_pk(TF, TV, TS, fold["s"] + 100 * i, ep, make_net=lambda df, dv, nc: HypModel(df, dv, nc, dict(REC)),
                      loss_fn=loss_hyp) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe)


TRAIN = {"CTRL": arm_ctrl, "PAEFF_EXACT": lambda fold: arm_paeff(fold, 2e-5),
         "PAEFF_B": lambda fold: arm_paeff(fold, 1e-3, STEPS), "HYP_INFO": arm_hyp}
COMPARISONS = [("PAEFF_EXACT", "CTRL"), ("PAEFF_B", "CTRL"), ("HYP_INFO", "CTRL"), ("PAEFF_B", "PAEFF_EXACT")]
'''

# ----------------------------------------------------------------------------------------------- FLAG_20: verifier
VER_INTRO = """
# FLAG 2027 — NB-20: learned pair scorers at level 0 (VER-01)

**Question:** is cosine the best scorer once the bridge is trained? Can the fixed stream weights of the recipe (0.25 CCA, 0.5 ImageBind, 0.1 age) be learned from train persons only?
Every scorer below reads one pair only (SAFE).

| Arm | What | Differs from |
|---|---|---|
| `CTRL` | s007 member + fixed recipe | — |
| `VER_MLP` | end-to-end: CTRL MLP + a small verifier head g([u, w, u*w, abs(u-w)]) → 128 → 1. Loss = InfoNCE + BCE over in-batch pairs: every positive, and as many negatives, half of them same-gender. Score = g (then the same recipe). "Fuse after align", small; FAME26 hạng 1 found large cross-attention worse (28.92 vs 23.99) | `CTRL`: the scorer only |
| `VER_FUSE` | CTRL nets; prod0 = logistic regression on the four level-0 streams (deep, CCA, ImageBind, age) instead of 0.5 / 0.1 / 0.25. The regression is fitted **out of fold** on train persons: 4 person groups; bridges retrained on the other 3; trials (ng + g) built inside the held group | `CTRL`: the fusion weights only |
| `VER_FUSE_X` | `VER_FUSE` + second-order terms (products, squares) of the four streams | `VER_FUSE`: interactions |

Registered comparisons: each arm vs CTRL; VER_FUSE_X vs VER_FUSE. The learned coefficients are written to `fuse_coef.csv`: they say how much each stream is really worth at level 0.
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** VER_FUSE trains 4 x N_NETS extra bridges per fold, about 1.5 h on a T4 in total; CPU about 4 h. **Output:** `/kaggle/working/l0_verifier/`. **Resume:** attach this notebook's earlier output.
"""

VER_ARMS = r'''
# ============================================================================ verifier arms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline


class VerModel(Model):
    """EXP-007 MLP + pair head g([u, w, u*w, |u-w|]) -> 128 -> 1; the level-0 score is g."""
    def __init__(self, d_f, d_v, n_cls, cfg):
        super().__init__(d_f, d_v, n_cls, cfg)
        e = cfg["emb"]
        self.head = nn.Sequential(nn.Linear(4 * e, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def score(self, a, b):
        return self.head(torch.cat([a, b, a * b, (a - b).abs()], 1)).squeeze(1)


def loss_ver(net, f, v, y, g):
    u, w = net(f, v)
    same = y[:, None] == y[None, :]
    pos = same.nonzero()
    neg_all = (~same).nonzero()
    neg_sg = ((~same) & (g[:, None] == g[None, :])).nonzero()
    k = len(pos)
    pick = lambda idx, n: idx[torch.randperm(len(idx), device=idx.device)[:n]]
    neg = torch.cat([pick(neg_sg, k // 2), pick(neg_all, k - min(k // 2, len(neg_sg)))])
    I = torch.cat([pos, neg]); lab = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(u.device)
    bce = F.binary_cross_entropy_with_logits(net.score(u[I[:, 0]], w[I[:, 1]]), lab)
    return infonce(u, w, y, REC["tau"]) + bce


def arm_ver(fold):
    TF, TV, TS, info = train_rows(fold)
    ep, bpe = epochs_for(len(TS))
    units = [train_pk(TF, TV, TS, fold["s"] + 100 * i, ep, make_net=lambda df, dv, nc: VerModel(df, dv, nc, dict(REC)),
                      loss_fn=loss_ver) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe)


STREAMS = ["deep0", "cca", "ib", "age"]
_CTRL, _OOF = {}, {}


def ctrl_member(fold):
    if fold["s"] not in _CTRL:
        _CTRL[fold["s"]] = arm_ctrl(fold)
    return _CTRL[fold["s"]]


def oof_streams(fold, k_groups=4):
    """Level-0 streams of train persons, each scored by bridges that never saw them (4 person groups)."""
    s = fold["s"]
    if s in _OOF:
        return _OOF[s]
    persons = np.random.RandomState(700 + s).permutation(fold["v4_train"])
    X, Y = [], []
    for gi in range(k_groups):
        grp = sorted(persons[gi::k_groups]); rest = sorted(set(persons) - set(grp))
        idx = np.where(np.isin(V4S, rest))[0]
        TF, TV, TS = V4["vgg"][idx], V4["btc"][idx], V4S[idx]
        ep, _ = epochs_for(len(TS))
        mem = Member([train_one(TF, TV, TS, seed=s + 100 * i + 7, epochs=ep) for i in range(N_NETS)], TF, TV)
        sc = L0(mem, idx)
        rng = np.random.RandomState(900 + 10 * s + gi)
        for sg in (False, True):
            T_ = v4_file(grp, sg, rng)
            S_ = sc.streams(T_)
            X.append(np.stack([S_[k] for k in STREAMS], 1)); Y.append(T_["y"])
        log(f"  OOF group {gi + 1}/{k_groups}: {len(grp)} persons, {sum(len(y) for y in Y)} rows so far")
    _OOF[s] = (np.concatenate(X), np.concatenate(Y))
    return _OOF[s]


def arm_fuse(fold, poly=False):
    mem, info = ctrl_member(fold)
    X, Y = oof_streams(fold)
    clf = (make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
           if poly else LogisticRegression(C=1.0, max_iter=5000)).fit(X, Y)
    if not poly:
        pd.DataFrame([dict(fold=fold["s"], **dict(zip(STREAMS, clf.coef_[0].round(4))), intercept=round(float(clf.intercept_[0]), 4),
                           oof_rows=len(Y))]).to_csv(OUT / "fuse_coef.csv", mode="a", header=not (OUT / "fuse_coef.csv").exists(), index=False)
        log("  learned stream weights:", dict(zip(STREAMS, clf.coef_[0].round(3))), "(fixed recipe: deep .75*.5, cca .25*.5, ib .5, age .1)")
    final = lambda S: clf.decision_function(np.stack([S[k] for k in STREAMS], 1))
    return mem, dict(info, oof_rows=len(Y)), final


TRAIN = {"CTRL": lambda fold: ctrl_member(fold), "VER_MLP": arm_ver,
         "VER_FUSE": lambda fold: arm_fuse(fold, False), "VER_FUSE_X": lambda fold: arm_fuse(fold, True)}
COMPARISONS = [("VER_MLP", "CTRL"), ("VER_FUSE", "CTRL"), ("VER_FUSE_X", "CTRL"), ("VER_FUSE_X", "VER_FUSE")]
'''

# ----------------------------------------------------------------------------------------------- FLAG_21: hard negatives
HARD_INTRO = """
# FLAG 2027 — NB-21: same-gender hard negatives at level 0 (HARD-01)

**Question:** the g cells are the gap to 23 EER (level 0: g/En 27.29, g/Bn 34.33). In a PK batch of 32 random speakers, about half of the InfoNCE negatives differ in gender, and the bridge can reject them from gender alone. Do batches whose speakers all share one gender teach the within-gender cues that protocol g needs?

| Arm | Training rows | Batches |
|---|---|---|
| `CTRL` | v4 train (40 persons) | PK 32 x 8 (random speakers) |
| `R_GH50` | v4 train | 50 % of batches: every speaker of the batch has the same gender (gender drawn at random; up to 32 speakers, about 20 in v4 train), else random |
| `R_GH100` | v4 train | every batch one gender |
| `XALL` | v4 train + external pool (all v1/v2 rows, <= 150 / person) | random |
| `XALL_GH50` | as `XALL` | 50 % same-gender |

Same optimizer budget (STEPS) for every arm. Registered comparisons: R_GH50 / R_GH100 / XALL vs CTRL; XALL_GH50 vs XALL. Expected: gains in g cells, possible losses in ng cells (the gender cue is still learned from the other half of the batches in GH50).
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** about 1-1.5 h on a T4, 3-4 h CPU. **Output:** `/kaggle/working/l0_hardneg/`. **Resume:** attach this notebook's earlier output.
"""

HARD_ARMS = r'''
# ============================================================================ hard-negative arms
def arm_gh(fold, frac, ext=False):
    TF, TV, TS, info = train_rows(fold, ext=ext)
    ep, bpe = epochs_for(len(TS))
    units = [train_pk(TF, TV, TS, fold["s"] + 100 * i, ep, ghard=frac) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe)


TRAIN = {"CTRL": arm_ctrl, "R_GH50": lambda fold: arm_gh(fold, 0.5), "R_GH100": lambda fold: arm_gh(fold, 1.0),
         "XALL": lambda fold: arm_ctrl(fold, ext=True), "XALL_GH50": lambda fold: arm_gh(fold, 0.5, ext=True)}
COMPARISONS = [("R_GH50", "CTRL"), ("R_GH100", "CTRL"), ("XALL", "CTRL"), ("XALL_GH50", "XALL"), ("XALL_GH50", "CTRL")]
'''

# ----------------------------------------------------------------------------------------------- FLAG_22: cross-video / external
XVID_INTRO = """
# FLAG 2027 — NB-22: cross-clip positives and external persons at level 0 (XVID-01)

**Questions:**
1. **External persons at level 0.** Earlier we concluded that adding persons has diminishing returns (SCALE-ID, EXT-VAL70, CUR-01). Every one of those studies measured the production system, with clusters (and ImageBind). Does the conclusion hold when every pair is scored on its own?
2. **Session shortcuts.** In a v4 training row, the face and the voice are cut from the SAME clip: 38 % of v4 rows repeat a voice. The bridge can learn "same recording" instead of "same person". Do cross-clip / cross-video training pairs help the level-0 score on truly cross-video trials?
3. **Pretrain → finetune** (the FAME26 first-place schedule), rehearsed at small scale before any VoxCeleb extraction.

| Arm | Training rows | Differs from |
|---|---|---|
| `CTRL` | v4 train (40 persons), same-clip rows | — |
| `R_DEC` | v4 train; each voice paired with a face from ANOTHER clip of the same person | `CTRL`: pairing only |
| `XALL` | v4 train + external pool (v1/v2 persons not held out; face from the same video, as CUR-01) | `CTRL`: + persons |
| `XALL_XVID` | as `XALL`, external face from ANOTHER video of the person | `XALL`: pairing only |
| `XALL_XVID_DEC` | `XALL_XVID` + v4 rows decoupled as in `R_DEC` | `XALL_XVID`: v4 pairing |
| `XALL_PTFT` | pretrain on the external pool (75 % of the steps), then finetune on v4 train (25 %, lr 3e-4); normalisation fitted on both | `XALL`: schedule only |

Same optimizer budget (STEPS) for every arm. Registered comparisons: R_DEC / XALL vs CTRL; XALL_XVID vs XALL; XALL_XVID_DEC vs XALL_XVID; XALL_PTFT vs XALL. **If XALL passes the gate at level 0, the VoxCeleb extraction (X1, thousands of persons) is worth reopening for SAFE.**
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** about 1.5-2 h on a T4, 4-5 h CPU. **Output:** `/kaggle/working/l0_xvideo/`. **Resume:** attach this notebook's earlier output.
"""

XVID_ARMS = r'''
# ============================================================================ cross-clip / external arms
def arm_rows(fold, **kw):
    TF, TV, TS, info = train_rows(fold, **kw)
    ep, bpe = epochs_for(len(TS))
    units = [train_one(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe)


def arm_ptft(fold, frac_pt=0.75, lr_ft=3e-4):
    rng = np.random.RandomState(4000 + fold["s"])
    idx = fold["v4tr"]
    F4, V4_, S4 = V4["vgg"][idx], V4["btc"][idx], V4S[idx]
    FE, VE, SE, fb = ext_rows(fold["ext_pool"], rng)
    TF, TV = np.concatenate([F4, FE]), np.concatenate([V4_, VE])
    prep = Prep(TF, TV, REC["pca_f"], REC["pca_v"])                      # one normalisation for both phases
    ep1, b1 = epochs_for(len(SE), int(round(STEPS * frac_pt)))
    ep2, b2 = epochs_for(len(S4), STEPS - int(round(STEPS * frac_pt)))
    n_cls = len(set(SE) | set(S4))
    units = []
    for i in range(N_NETS):
        seed = fold["s"] + 100 * i
        net, _ = train_pk(FE, VE, SE, seed, ep1, prep=prep, make_net=lambda df, dv, nc: Model(df, dv, n_cls, dict(REC)))
        net, _ = train_pk(F4, V4_, S4, seed + 50, ep2, prep=prep, net=net, lr=lr_ft)
        units.append((net, prep))
    return Member(units, TF, TV), dict(fallback=fb, train_ids=n_cls, train_rows=len(TF), steps=ep1 * b1 + ep2 * b2)


TRAIN = {"CTRL": arm_ctrl, "R_DEC": lambda fold: arm_rows(fold, decouple=True),
         "XALL": lambda fold: arm_rows(fold, ext=True), "XALL_XVID": lambda fold: arm_rows(fold, ext=True, xvid=True),
         "XALL_XVID_DEC": lambda fold: arm_rows(fold, ext=True, xvid=True, decouple=True), "XALL_PTFT": arm_ptft}
COMPARISONS = [("R_DEC", "CTRL"), ("XALL", "CTRL"), ("XALL_XVID", "XALL"), ("XALL_XVID_DEC", "XALL_XVID"),
               ("XALL_PTFT", "XALL"), ("XALL_XVID_DEC", "CTRL")]
'''

# ----------------------------------------------------------------------------------------------- FLAG_23: XALL x learned weights
COMBO_INTRO = """
# FLAG 2027 — NB-23: external persons x learned stream weights at level 0 (COMBO-01)

**Question:** the two SAFE levers that passed the gate (log 91 / 92) are:
- **XALL:** 161 persons in the bridge instead of 40, +0.9…+3.4;
- **VER_FUSE:** stream weights learned out of fold, +1…+1.35 in the cross-video cells.

Do the two add up, and must the weights be re-learned once the bridge has external persons?

| Arm | Bridge | Final score | Differs from |
|---|---|---|---|
| `CTRL` | 40 v4 persons | fixed recipe (0.25 CCA, 0.5 ImageBind, 0.1 age) | — |
| `CTRL_FIXW` | as CTRL | **fixed** weights, registered from VER-01 (`fuse_coef.csv` mean over 5 folds): deep .298, CCA .405, ImageBind .948, age .327. Nothing is fitted in this notebook | `CTRL`: weights only |
| `VER_FUSE` | as CTRL | logistic regression on the 4 streams, fitted out of fold on v4 train persons (= VER-01) | `CTRL` |
| `XALL` | 40 v4 + external pool (= XVID-01) | fixed recipe | `CTRL` |
| `XALL_FIXW` | as XALL | the registered fixed weights | `XALL`: weights only |
| `XALL_FUSE` | as XALL | logistic regression fitted out of fold. The OOF bridges are trained on 3/4 of the v4 train persons + the whole external pool; the trials come from the held quarter | `XALL_FIXW`: fitted vs fixed |

Registered comparisons: CTRL_FIXW / XALL / XALL_FIXW / XALL_FUSE vs CTRL; XALL_FIXW / XALL_FUSE vs XALL; XALL_FUSE vs XALL_FIXW.
**What we will use:** the best XALL arm that passes the gate against XALL. If none does, keep XALL with the fixed recipe. `CTRL`, `VER_FUSE` and `XALL` must reproduce the numbers of VER-01 / XVID-01 exactly.
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** the OOF fits train 4 x 3 extra bridges per fold for each of VER_FUSE and XALL_FUSE. About 2-2.5 h on a T4. **Output:** `/kaggle/working/l0_combo/` (+ `fuse_coef.csv`). **Resume:** attach this notebook's earlier output.
"""

COMBO_ARMS = r'''
# ============================================================================ XALL x stream-weight arms
from sklearn.linear_model import LogisticRegression

STREAMS = ["deep0", "cca", "ib", "age"]
FIXW = np.array([0.298, 0.405, 0.948, 0.327])          # registered: mean of VER-01 fuse_coef.csv over 5 folds (log 91)
_MEM, _OOF = {}, {}


def member(fold, ext):
    k = (fold["s"], ext)
    if k not in _MEM:
        _MEM[k] = arm_ctrl(fold, ext=ext)                  # = CTRL / XALL of VER-01 / XVID-01 (same rows, seeds, steps)
    return _MEM[k]


def oof_streams(fold, ext, k_groups=4):
    """Level-0 streams of v4 train persons, each scored by bridges that never saw them. With ext the bridges also train
    on the whole external pool (as XALL does); constants of the L0 scorer come from the bridges' own v4 rows."""
    s = fold["s"]
    if (s, ext) in _OOF:
        return _OOF[(s, ext)]
    persons = np.random.RandomState(700 + s).permutation(fold["v4_train"])     # = VER-01 groups
    X, Y = [], []
    for gi in range(k_groups):
        grp = sorted(persons[gi::k_groups]); rest = sorted(set(persons) - set(grp))
        idx = np.where(np.isin(V4S, rest))[0]
        TF, TV, TS = V4["vgg"][idx], V4["btc"][idx], V4S[idx]
        if ext:
            EF, EV, ES, _ = ext_rows(fold["ext_pool"], np.random.RandomState(4000 + s))
            TF, TV, TS = np.concatenate([TF, EF]), np.concatenate([TV, EV]), np.concatenate([TS, ES])
        ep, _ = epochs_for(len(TS))
        mem = Member([train_one(TF, TV, TS, seed=s + 100 * i + 7, epochs=ep) for i in range(N_NETS)], TF, TV)
        sc = L0(mem, idx)
        rng = np.random.RandomState(900 + 10 * s + gi)
        for sg in (False, True):
            T_ = v4_file(grp, sg, rng)
            S_ = sc.streams(T_)
            X.append(np.stack([S_[k] for k in STREAMS], 1)); Y.append(T_["y"])
        log(f"  OOF ({'XALL' if ext else 'v4'}) group {gi + 1}/{k_groups}: {len(grp)} persons, {sum(len(y) for y in Y)} rows")
    _OOF[(s, ext)] = (np.concatenate(X), np.concatenate(Y))
    return _OOF[(s, ext)]


def arm_weights(fold, ext, how):
    mem, info = member(fold, ext)
    if how == "fixed":
        return mem, info, (lambda S: np.stack([S[k] for k in STREAMS], 1) @ FIXW)
    X, Y = oof_streams(fold, ext)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(X, Y)
    pd.DataFrame([dict(fold=fold["s"], bridge="XALL" if ext else "v4", **dict(zip(STREAMS, clf.coef_[0].round(4))),
                       intercept=round(float(clf.intercept_[0]), 4), oof_rows=len(Y))]).to_csv(
        OUT / "fuse_coef.csv", mode="a", header=not (OUT / "fuse_coef.csv").exists(), index=False)
    log("  learned stream weights:", dict(zip(STREAMS, clf.coef_[0].round(3))))
    return mem, dict(info, oof_rows=len(Y)), (lambda S: clf.decision_function(np.stack([S[k] for k in STREAMS], 1)))


TRAIN = {"CTRL": lambda fold: member(fold, False), "CTRL_FIXW": lambda fold: arm_weights(fold, False, "fixed"),
         "VER_FUSE": lambda fold: arm_weights(fold, False, "oof"), "XALL": lambda fold: member(fold, True),
         "XALL_FIXW": lambda fold: arm_weights(fold, True, "fixed"), "XALL_FUSE": lambda fold: arm_weights(fold, True, "oof")}
COMPARISONS = [("CTRL_FIXW", "CTRL"), ("VER_FUSE", "CTRL"), ("XALL", "CTRL"), ("XALL_FIXW", "CTRL"), ("XALL_FUSE", "CTRL"),
               ("XALL_FIXW", "XALL"), ("XALL_FUSE", "XALL"), ("XALL_FUSE", "XALL_FIXW")]
'''

# ----------------------------------------------------------------------------------------------- FLAG_24: dev submission, SET-17 level 0 with a new s007
DEV_INTRO = """
# FLAG 2027 — NB-24: SET-17 level 0 on dev with the s007 member swapped (SAFE-02)

Builds CodaBench dev zips for the SAFE track. Every pair is scored on its own (level 0), as SET-17 level 0 (CB **26.35**,
log 87). The only change is the **s007 member**:
- `s010B` / `r2mix` / AudioCLIP are kept exactly as submitted (their level-0 component scores are shipped in `flag2027-safe-dev`);
- s007 is retrained here with the recipe that was validated at level 0 (PK 32 x 8, 1200 steps, InfoNCE + CCA-4, XVID-01 / COMBO-01), then scored with the same ImageBind + age (+ 0.25 AudioCLIP in the English cells) recipe and constants from 20 000 train pairs (`BEST/v6_eval/pipeline.py indep()`).

| Zip | s007 member | Purpose |
|---|---|---|
| `submission_REF.zip` | the submitted one (shipped scores) | **self-test**: must be identical to SET-17 level 0 (rank-corr 1.0000); do not submit |
| `submission_S007_V4.zip` | retrained, 70 v4 persons, PK recipe | control of the recipe change (no external persons) |
| `submission_S007_XALL.zip` | retrained, 70 v4 + all 151 MAV-Celeb v1/v2 persons (<= 150 rows each; the 2 persons also in v4 are already removed): 221 persons | **the SAFE candidate (XVID-01)**, submit first |
| `submission_S007_XALL_FIXW.zip` | as XALL; s007 streams combined with the registered VER-01 weights (deep .298, CCA .405, ImageBind .948, age .327) instead of 0.25 / 0.5 / 0.1 | submit only if COMBO-01 says XALL_FIXW > XALL |

The dev trial list is used only to write the rows in the organiser's order. No statistic of a dev file enters any score; the notebook re-scores half of each file and checks that nothing moves.
Persons of v1/v2 that are also in v4 (`v1:shaukat_tarin`, `v2:id0073`) are excluded, and no v1/v2 person is in dev (SET-07 overlap check).

**Inputs (Add Input):** `flag2027-eda-bundle` and `flag2027-safe-dev` (upload `kaggle/flag2027-safe-dev.zip`, made by `python build_safe_dev_bundle.py`).
**Settings:** GPU T4 (or CPU); Internet OFF. **Time:** 3 arms x 5 nets, about 20-30 min on a T4 (each XALL net recomputes the face PCA on about 26 000 rows). **Output:** `/kaggle/working/safe_dev/` with the zips and `rank_corr.csv`.
"""

DEV_ENV = r'''
import os, sys, re, io, json, time, shutil, math, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
OUT = Path(os.environ.get("FLAG_OUT", "/kaggle/working")) / "safe_dev"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

# ------------------------------------------------------------------ registered settings
STEPS = int(os.environ.get("L0_STEPS", "1200"))       # as validated (XVID-01)
N_NETS = int(os.environ.get("L0_NETS", "5"))           # = the n of SET-17 level 0 (CB 26.35)
ARMS = os.environ.get("L0_ARMS", "REF,S007_V4,S007_XALL,S007_XALL_FIXW").split(",")
N_REF, P_PK, K_PK, CAP = 20000, 32, 8, 150
W_IB, W_AGE, W_CCA, W_ACL = 0.5, 0.1, 0.25, 0.25       # SET-17 level 0 (BEST/v6_eval: --ib-all English --aclip 0.25)
FIXW = np.array([0.298, 0.405, 0.948, 0.327])          # registered VER-01 weights: deep, CCA, ImageBind, age (log 91)
M_EVAL, N_V4_HELD, EXACT_EPOCHS = 24, 30, None         # used by the shared cells only
OVERLAP = {"v1_complete:shaukat_tarin", "v2_complete:id0073"}
AGE_MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)
print("LOCAL" if LOCAL else "KAGGLE", "| device", DEVICE, "| steps", STEPS, "| nets", N_NETS, "| arms", ARMS)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

DEV_RUN = r'''
# ============================================================================ dev bundle, s007 swap, level-0 scoring, zips
f = next(IN.rglob("safe_dev_manifest.json"), None)
assert f is not None, "attach the dataset flag2027-safe-dev (kaggle/build_safe_dev_bundle.py)"
DB = f.parent
DMAN = json.loads(f.read_text())
D = {k: np.load(DB / f"{k}.npy") for k in DMAN["arrays"]}
log("safe-dev bundle:", DB, "|", DMAN["created"], "|", len(D), "arrays")
CELLS = [("no_gender", "English"), ("no_gender", "Bangla"), ("gender", "English"), ("gender", "Bangla")]
NAMES = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
MEMBERS = {"English": ["s007", "s010B"], "Bangla": ["r2mix", "s007"]}
key = lambda k: f"{k[0]}_{k[1]}"

# the train rows of the dev bundle are the organiser's, in the order of the eda bundle
TR = dict(vgg=D["train__vgg"], btc=D["train__btc"], ibf=D["train__ibf"], iba=D["train__iba"], agef=D["train__agef"],
          agev=D["train__agev"], aclf=D["train__aclf"], aclv=D["train__aclv"])
assert np.array_equal(TR["vgg"], V4["vgg"]) and np.array_equal(TR["btc"], V4["btc"]), "train rows differ from the eda bundle"
for k in ("ibf", "iba", "agef", "agev"):
    log(f"train {k}: max |dev bundle - eda bundle| = {np.abs(TR[k] - V4[k]).max():.2e}")


def dev_trials(k, rows=None):
    t = {n: D[f"{key(k)}__{n}"] for n in ("vgg", "btc", "ibf", "iba", "agef", "agev", "aclf", "aclv")}
    return t if rows is None else {n: v[rows] for n, v in t.items()}


class DevL0:
    """Level-0 recipe of s007 in SET-17 level 0 (pipeline.py indep + recipe0), every constant from N_REF random pairs of
    ALL train rows (np.random.RandomState(0), as pipeline.py REF): the same code path as L0 of the validation notebooks,
    plus AudioCLIP in the English cells and the optional registered stream weights."""
    def __init__(self, mem, fixw=None):
        self.mem, self.fixw = mem, fixw
        rr = np.random.RandomState(0)
        rf, rv = rr.randint(len(TR["vgg"]), size=N_REF), rr.randint(len(TR["vgg"]), size=N_REF)
        self.sd, parts = [], []
        for u in mem.units:
            Ef, Ev = mem.embed(u, TR["vgg"], TR["btc"])
            r = mem.score(u, Ef[rf], Ev[rv]); self.sd.append(Std(r)); parts.append(self.sd[-1](r))
        deep = np.mean(parts, 0)
        c = np.sum(mem.cca.transform_x(TR["vgg"])[rf] * mem.cca.transform_y(TR["btc"])[rv], 1); self.zc = Std(c)
        member = (1 - W_CCA) * deep + W_CCA * self.zc(c); self.zm = Std(member)
        self.mf, self.mv = TR["ibf"].mean(0), TR["iba"].mean(0)
        ib = np.sum(l2n(TR["ibf"][rf] - self.mf) * l2n(TR["iba"][rv] - self.mv), 1); self.zib = Std(ib)
        age = -np.abs(TR["agef"][rf] - TR["agev"][rv]); self.zag = Std(age)
        self.maf, self.mav = TR["aclf"].mean(0), TR["aclv"].mean(0)
        acl = np.sum(l2n(TR["aclf"][rf] - self.maf) * l2n(TR["aclv"][rv] - self.mav), 1); self.zacl = Std(acl)
        ref = dict(deep=deep, cca=self.zc(c), member=self.zm(member), ib=self.zib(ib), age=self.zag(age), acl=self.zacl(acl))
        base = self._base(ref); self.zbase = Std(base)
        self.zfin = {False: Std(self.zbase(base) + W_AGE * ref["age"] * (self.fixw is None)),
                     True: Std(self.zbase(base) + W_AGE * ref["age"] * (self.fixw is None) + W_ACL * ref["acl"])}

    def _base(self, s):
        if self.fixw is None:                                     # SET-17: zmix(0.5 z(member) + 0.5 z(IB)); age added after
            return (1 - W_IB) * s["member"] + W_IB * s["ib"]
        return np.stack([s["deep"], s["cca"], s["ib"], s["age"]], 1) @ self.fixw   # registered weights (age inside)

    def streams(self, T):
        deep = np.mean([sd(self.mem.pair(u, T["vgg"], T["btc"])) for u, sd in zip(self.mem.units, self.sd)], 0)
        cca = self.zc(np.sum(self.mem.cca.transform_x(T["vgg"]) * self.mem.cca.transform_y(T["btc"]), 1))
        member = self.zm((1 - W_CCA) * deep + W_CCA * cca)
        ib = self.zib(np.sum(l2n(T["ibf"] - self.mf) * l2n(T["iba"] - self.mv), 1))
        age = self.zag(-np.abs(T["agef"] - T["agev"]))
        acl = self.zacl(np.sum(l2n(T["aclf"] - self.maf) * l2n(T["aclv"] - self.mav), 1))
        return dict(deep=deep, cca=cca, member=member, ib=ib, age=age, acl=acl)

    def comp(self, T, english):
        s = self.streams(T)
        x = self.zbase(self._base(s)) + W_AGE * s["age"] * (self.fixw is None)
        if english:
            x = x + W_ACL * s["acl"]
        return self.zfin[english](x)


def write_zip(path, scores):
    """scores {(protocol, lang): higher = same} -> the organiser zip; -score, LOWER = same (docs/DATA.md)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k in CELLS:
            pid = D[f"{key(k)}__pair_id"]
            assert len(pid) == len(scores[k]) and np.isfinite(scores[k]).all()
            z.writestr(NAMES[k], "".join(f"{p} {-s:.6f}\n" for p, s in zip(pid, scores[k])))
    log("wrote", path)


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        return {k: pd.read_csv(z.open(NAMES[k]), sep=" ", header=None)[1].values for k in CELLS}


def train_s007(ext):
    """s007 on ALL 70 v4 persons (+ every v1/v2 person with ext), PK 32 x 8, STEPS steps, N_NETS seeds 1 + 100 i."""
    TF, TV, TS = V4["vgg"], V4["btc"], V4S
    if ext:
        EF, EV, ES, _ = ext_rows(PERSONS, np.random.RandomState(4000))
        TF, TV, TS = np.concatenate([TF, EF]), np.concatenate([TV, EV]), np.concatenate([TS, ES])
    ep, bpe = epochs_for(len(TS))
    t0 = time.time()
    units = [train_one(TF, TV, TS, seed=1 + 100 * i, epochs=ep) for i in range(N_NETS)]
    log(f"s007 {'XALL' if ext else 'v4'}: {len(set(TS))} persons, {len(TS)} rows, {ep * bpe} steps x {N_NETS} nets [{time.time() - t0:.0f}s]")
    return Member(units, TF, TV)


ZIPS, MEMS = {}, {}
for arm in ARMS:
    scores = {}
    if arm == "REF":
        for k in CELLS:
            comp = [D[f"{key(k)}__comp_{m}"] for m in MEMBERS[k[1]]]
            scores[k] = np.mean(comp, 0)
            assert np.abs(scores[k] - D[f"{key(k)}__final"]).max() < 1e-9, k
    else:
        ext = "XALL" in arm
        if ext not in MEMS:
            MEMS[ext] = train_s007(ext)
        sc = DevL0(MEMS[ext], FIXW if arm.endswith("FIXW") else None)
        for k in CELLS:
            T_ = dev_trials(k)
            s007 = sc.comp(T_, k[1] == "English")
            comp = [s007 if m == "s007" else D[f"{key(k)}__comp_{m}"] for m in MEMBERS[k[1]]]
            scores[k] = np.mean(comp, 0)
            half = np.sort(np.random.RandomState(1).choice(len(s007), len(s007) // 2, replace=False))   # level-0 self-test
            d = np.abs(sc.comp(dev_trials(k, half), k[1] == "English") - s007[half]).max()
            assert d < 1e-4, f"{arm} {k}: a score moved by {d} when half of the file was removed"
    ZIPS[arm] = OUT / f"submission_{arm}.zip"
    write_zip(ZIPS[arm], scores)

# rank correlation of every zip with the submitted SET-17 level 0 (REF) and with each other
from scipy.stats import spearmanr
ref = read_zip(ZIPS["REF"]) if "REF" in ZIPS else None
rows = []
for arm, p in ZIPS.items():
    z_ = read_zip(p)
    for k in CELLS:
        r = dict(arm=arm, cell=f"{k[0]}/{k[1]}")
        if ref is not None:
            r["rank_corr_vs_REF"] = round(float(spearmanr(z_[k], ref[k])[0]), 4)
        rows.append(r)
RC = pd.DataFrame(rows); RC.to_csv(OUT / "rank_corr.csv", index=False)
print(RC.pivot_table(index="cell", columns="arm", values="rank_corr_vs_REF").to_string())
if ref is not None:
    assert (RC[RC.arm == "REF"].rank_corr_vs_REF == 1).all()
    D_ref = DMAN.get("reference_zip")
    log("REF = shipped SET-17 level-0 scores (" + str(D_ref) + "): self-test passed")
print("\nSUBMIT (CodaBench dev, one at a time): submission_S007_XALL.zip first. The others only if their validation says so.")
print("Download /kaggle/working/safe_dev (Output tab) and hand it back with the CodaBench scores.")
'''

# ----------------------------------------------------------------------------------------------- FLAG_26: error analysis
ERR_INTRO = """
# FLAG 2027 — NB-26: error analysis of the level-0 system on true-label validation (ERR-03)

**Question:** where does the level-0 score go wrong, and which feature is to blame? Everything is measured on **validation with true labels**:
- 30 held-out v4 persons, with the positive face from another clip;
- held-out MAV-Celeb v1/v2 persons, with the positive face from another video (English, Urdu, Hindi).

The notebook never reads dev or test data. The 30/09 ruling forbids taking any statistic, pseudo-label or decision rule from test data. Folds, trials and level-0 scorer are the same as in FLAG_18 … 23. Two bridges are compared: `CTRL` (40 v4 persons) and `XALL` (+ the external pool).

| Part | What it measures | Tells us |
|---|---|---|
| A. `stream_eer.csv`, `corr.csv` | EER of every stream alone (deep bridge, CCA-4, ImageBind, age), of the member and of the production score; Spearman correlation between streams, within positives and within negatives | which feature is weak in which cell; whether streams are redundant |
| B. `ablation.csv` | production recipe with one stream removed (no deep / no CCA / no ImageBind / no age / ImageBind + age only), and with the registered VER-01 weights | which stream earns its weight, per cell |
| C. `aggregation.csv` | the same scores after averaging over the other samples of the TRUE face person, of the TRUE voice person, or both (an oracle, possible on validation only) | whether errors are per-sample noise (averaging fixes them) or systematic (averaging does not); face side vs voice side |
| D. `drift.csv` | mean and sd of every stream on positives and on negatives, and d', per cell, in units of the train-pair constants | streams whose scale or offset moves with language or domain; such a shift silently changes their weight in the fixed fusion |
| E. `subgroup.csv` | production EER by voice attribute: gender, predicted voice age, clip duration (v1/v2 only), typicality (distance to the train mean) | which kinds of samples fail: bias signals, and which data to add |

The trial table `trials.csv.gz` is also written: every trial, every stream, every attribute. It allows later analysis without retraining.
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** about 40-60 min on a T4. **Output:** `/kaggle/working/l0_err/`. **Resume:** attach this notebook's earlier output; finished (fold, bridge) parts are skipped.
"""

ERR_RUN = r"""
# ============================================================================ error analysis (validation, true labels only)
from scipy.stats import spearmanr
BRIDGES = {"CTRL": False, "XALL": True}
ST = ["deep0", "cca", "ib", "age", "member0", "prod0"]
PARTS = OUT / "parts"; PARTS.mkdir(exist_ok=True)
for p in IN.rglob("parts/*_f*"):                                        # resume from an attached earlier output
    if OUT not in p.parents and p.parent.parent.name == OUT.name and not (PARTS / p.name).exists():
        shutil.copy(p, PARTS / p.name); log("resume: copied", p.name)
zst = lambda a, r: (a - r.mean()) / (r.std() + 1e-9)


def ref_streams(sc, v4rows):
    # level-0 streams of the scorer's own 20 000 reference train pairs (the same pairs as L0.__init__)
    rr = np.random.RandomState(0)
    rf, rv = rr.randint(len(v4rows), size=N_REF), rr.randint(len(v4rows), size=N_REF)
    F_, V_ = v4rows[rf], v4rows[rv]
    return sc.streams(trial_dict(V4, V4, F_, V_, np.zeros(N_REF), V4S[F_], V4S[V_]))


def recipe(X, R, deep=0.75, cca=0.25, ib=0.5, age=0.1):
    # the SET-17 level-0 recipe on z-scored streams; every intermediate z uses the reference pairs R
    if deep or cca:
        mr = deep * R["deep0"] + cca * R["cca"]
        mx, mr = zst(deep * X["deep0"] + cca * X["cca"], mr), zst(mr, mr)
    else:
        mx = mr = 0.0
    mix_x, mix_r = (1 - ib) * mx + ib * X["ib"], (1 - ib) * mr + ib * R["ib"]
    return zst(mix_x, mix_r) + age * X["age"]


FIXW = np.array([0.298, 0.405, 0.948, 0.327])          # registered VER-01 weights (log 91)
ABL = {"full": {}, "no_deep": dict(deep=0, cca=1), "no_cca": dict(deep=1, cca=0), "no_ib": dict(ib=0),
       "no_age": dict(age=0), "ib_age_only": dict(deep=0, cca=0, ib=1)}


@torch.no_grad()
def matrix(sc, T):
    # every stream on the full (unique faces x unique voices) matrix of one eval file, level-0 constants
    uf, fpos = np.unique(T["fi"], return_index=True)
    uv, vpos = np.unique(T["vi"], return_index=True)
    Xf, Xv = T["vgg"][fpos], T["btc"][vpos]
    deep = []
    for u, sd in zip(sc.mem.units, sc.sd):
        Ef, Ev = sc.mem.embed(u, Xf, Xv)
        deep.append(sd((Ef @ Ev.T).float().cpu().numpy()))              # CTRL / XALL nets score by cosine
    M = dict(deep0=np.mean(deep, 0))
    M["cca"] = sc.zc(sc.mem.cca.transform_x(Xf) @ sc.mem.cca.transform_y(Xv).T)
    M["member0"] = sc.zm((1 - W_CCA) * M["deep0"] + W_CCA * M["cca"])
    M["ib"] = sc.zib(l2n(T["ibf"][fpos] - sc.mf) @ l2n(T["iba"][vpos] - sc.mv).T)
    M["age"] = sc.zag(-np.abs(T["agef"][fpos][:, None] - T["agev"][vpos][None, :]))
    M["prod0"] = sc.zfin(sc.zmix((1 - W_IB) * M["member0"] + W_IB * M["ib"]) + W_AGE * M["age"])
    fr, vr = np.searchsorted(uf, T["fi"]), np.searchsorted(uv, T["vi"])
    return M, fr, vr, T["pf"][fpos], T["pv"][vpos]


def group_mean(M, g, axis):
    # average the rows (axis 0) or the columns (axis 1) of M within the groups g
    codes, inv = np.unique(g, return_inverse=True)
    G = np.zeros((len(codes), len(g))); G[inv, np.arange(len(g))] = 1; G /= G.sum(1, keepdims=True)
    return (G @ M)[inv] if axis == 0 else (M @ G.T)[:, inv]


def voice_attrs(T, data, fold):
    idx = fold["v4tr"]
    mu, sd = V4["btc"][idx].mean(0), V4["btc"][idx].std(0) + 1e-6
    typ = np.linalg.norm((T["btc"] - mu) / sd, axis=1) / np.sqrt(T["btc"].shape[1])
    dur = VMETA.dur.values[T["vi"]] if data != "v4" and "dur" in VMETA else np.full(len(T["y"]), np.nan)
    return dict(v_gender=[GMAP.get(p, "u") for p in T["pv"]], v_age=T["agev"], v_typ=typ, v_dur=dur)


for s in FOLDS:
    todo = [b for b in BRIDGES if not (PARTS / f"trials_f{s}_{b}.csv.gz").exists()]
    if not todo:
        continue
    fold = make_fold(s)
    for b in todo:
        t0 = time.time()
        mem, info = arm_ctrl(fold, ext=BRIDGES[b])
        sc = L0(mem, fold["v4tr"])
        R = ref_streams(sc, fold["v4tr"])
        rows, agg = [], []
        for (data, prot), T_ in fold["files"].items():
            S_ = sc.streams(T_); y = T_["y"]
            M, fr, vr, fp, vp = matrix(sc, T_)
            d = np.abs(M["prod0"][fr, vr] - S_["prod0"]).max()
            assert d < 1e-3, f"matrix path disagrees with the pair path by {d}"
            X = {k: S_[k] for k in ST}
            for name, kw in ABL.items():
                X["abl_" + name] = recipe(S_, R, **kw)
            X["abl_fixw"] = np.stack([S_[k] for k in ("deep0", "cca", "ib", "age")], 1) @ FIXW
            assert abs(eer(X["abl_full"], y) - eer(S_["prod0"], y)) < 1e-6, "recipe() does not reproduce prod0"
            for k in ST:                                               # C. aggregation oracle over TRUE persons
                Mk = M[k]
                for how, A_ in (("sample", Mk), ("face_person", group_mean(Mk, fp, 0)), ("voice_person", group_mean(Mk, vp, 1)),
                                ("both", group_mean(group_mean(Mk, fp, 0), vp, 1))):
                    agg.append(dict(fold=s, bridge=b, data=data, protocol=prot, stream=k, avg=how, eer=eer(A_[fr, vr], y)))
            rows.append(pd.DataFrame(dict(fold=s, bridge=b, data=data, protocol=prot, y=y, pf=T_["pf"], pv=T_["pv"],
                                          **{k: np.asarray(v, dtype=np.float32) for k, v in X.items()},
                                          **voice_attrs(T_, data, fold))))
        pd.concat(rows).to_csv(PARTS / f"trials_f{s}_{b}.csv.gz", index=False)
        pd.DataFrame(agg).to_csv(PARTS / f"agg_f{s}_{b}.csv", index=False)
        log(f"fold {s} {b}: {sum(len(r) for r in rows)} trials, {len(agg)} aggregation rows [{time.time() - t0:.0f}s]")
"""

ERR_SUMMARY = r"""
# ============================================================================ tables A-E (+ trials.csv.gz for later analysis)
TR = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob("trials_f*.csv.gz"))], ignore_index=True)
AG = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob("agg_f*.csv"))], ignore_index=True)
TR.to_csv(OUT / "trials.csv.gz", index=False)
TR["cell"] = TR.data + " " + TR.protocol; AG["cell"] = AG.data + " " + AG.protocol
pd.set_option("display.width", 250)


def per_fold(cols):
    out = []
    for (b, c, f_), g in TR.groupby(["bridge", "cell", "fold"]):
        out.append(dict(bridge=b, cell=c, fold=f_, **{k: eer(g[k].values, g.y.values) for k in cols}))
    return pd.DataFrame(out).groupby(["bridge", "cell"])[cols].mean()


# A. streams alone, and how much they agree
A_ = per_fold(ST).round(2); A_.to_csv(OUT / "stream_eer.csv")
print("A. EER of each stream alone (mean of the folds)"); print(A_.to_string())
corr = []
for (b, c), g in TR.groupby(["bridge", "cell"]):
    for lab, h in (("pos", g[g.y == 1]), ("neg", g[g.y == 0])):
        for i, a in enumerate(ST[:4]):
            for bb in ST[i + 1:4]:
                corr.append(dict(bridge=b, cell=c, trials=lab, pair=f"{a}~{bb}", rho=spearmanr(h[a], h[bb])[0]))
C_ = pd.DataFrame(corr); C_.to_csv(OUT / "corr.csv", index=False)
print("\nA'. Spearman between streams on NEGATIVES (CTRL)")
print(C_[(C_.trials == "neg") & (C_.bridge == "CTRL")].pivot_table(index="cell", columns="pair", values="rho").round(2).to_string())

# B. ablation (+ = the variant is worse than the full recipe, i.e. the removed stream earns its weight)
abl = [c for c in TR if c.startswith("abl_")]
B_ = per_fold(abl)
D_ = B_.drop(columns="abl_full").sub(B_["abl_full"], axis=0).round(2); D_.to_csv(OUT / "ablation.csv")
print("\nB. EER change vs the full recipe (+ = worse)"); print(D_.to_string())

# C. aggregation oracle over the TRUE persons
C2 = AG.groupby(["bridge", "cell", "stream", "avg"]).eer.mean().unstack("avg")[["sample", "face_person", "voice_person", "both"]].round(2)
C2.to_csv(OUT / "aggregation.csv")
print("\nC. EER after averaging over the TRUE person (validation oracle, not usable at test time) - prod0")
print(C2.xs("prod0", level="stream").to_string())
print("\nC. same, deep bridge alone")
print(C2.xs("deep0", level="stream").to_string())

# D. drift: units of the train reference pairs (negatives should sit near mean 0, sd 1)
drift = []
for (b, c), g in TR.groupby(["bridge", "cell"]):
    for k in ST:
        p, n = g[g.y == 1][k], g[g.y == 0][k]
        drift.append(dict(bridge=b, cell=c, stream=k, pos_mean=p.mean(), neg_mean=n.mean(), pos_sd=p.std(), neg_sd=n.std(),
                          dprime=(p.mean() - n.mean()) / np.sqrt(0.5 * (p.var() + n.var()))))
DR = pd.DataFrame(drift).round(3); DR.to_csv(OUT / "drift.csv", index=False)
print("\nD. d' per stream (CTRL)"); print(DR[DR.bridge == "CTRL"].pivot_table(index="cell", columns="stream", values="dprime").round(2).to_string())
print("\nD'. mean / sd of NEGATIVES per stream (CTRL; train pairs = 0 / 1)")
print(DR[DR.bridge == "CTRL"].pivot_table(index="cell", columns="stream", values=["neg_mean", "neg_sd"]).round(2).to_string())

# E. subgroups by voice attribute (each voice is in one positive and one negative trial: every bin is balanced)
sub = []
for attr in ("v_gender", "v_age", "v_typ", "v_dur"):
    for (b, c), g in TR.groupby(["bridge", "cell"]):
        x = g[attr]
        if x.isna().all():
            continue
        bins = x.astype(str) if attr == "v_gender" else pd.qcut(x.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
        for (f_, bn), h in g.groupby([g.fold, bins]):
            if h.y.nunique() == 2 and len(h) >= 20:
                sub.append(dict(bridge=b, cell=c, attr=attr, bin=bn, fold=f_, n=len(h), eer=eer(h.prod0.values, h.y.values)))
SB = (pd.DataFrame(sub).groupby(["bridge", "cell", "attr", "bin"])
      .agg(eer=("eer", "mean"), n=("n", "sum"), folds=("fold", "nunique")).round(2).reset_index())
SB.to_csv(OUT / "subgroup.csv", index=False)
print("\nE. production EER by voice attribute (CTRL; terciles within each cell)")
print(SB[SB.bridge == "CTRL"].pivot_table(index=["attr", "bin"], columns="cell", values="eer").to_string())
print(f"\nNEXT: download /kaggle/working/{OUT.name} (Output tab) and hand it back.")
"""

# ----------------------------------------------------------------------------------------------- FLAG_28: EMBED-SCOUT
SCOUT_ARMS = "CTRL,V_rdn6vox,V_wsr221,V_wssim34,V_wavlmsv,V_unisatsv,V_sbresnet,F_facenet,IB_ibvflip"

SCOUT_INTRO = """
# FLAG 2027 — NB-28: EMBED-SCOUT, new pretrained embeddings in the level-0 system

**Question:** ERR-03 found the MLP bridge (VGG face <-> organiser ECAPA voice) to be the weak part. Does another pretrained voice or face embedding make a better bridge? Is it complementary to the current one?

Every arm is **XALL_FIXW**, the best arm of COMBO-01: bridge on 40 v4 + 121 v1/v2 persons, EXP-007 MLP + CCA-4, ImageBind + age, the registered VER-01 weights. Exactly **one input** is swapped:

| Arm | Swapped input | Embedding (FLAG_27, dataset `flag2027-feats-scout`) |
|---|---|---|
| `CTRL` | none | = XALL_FIXW of COMBO-01 (must reproduce it) |
| `V_rdn6vox` | voice | ReDimNet2-B6 LM, VoxCeleb2 |
| `V_wsr221` | voice | WeSpeaker ResNet221-LM, VoxCeleb2 |
| `V_wssim34` | voice | WeSpeaker SimAM-ResNet34, VoxBlink2 -> VoxCeleb2 |
| `V_wavlmsv` | voice | WavLM Base+ SV |
| `V_unisatsv` | voice | UniSpeech-SAT Base+ SV |
| `V_sbresnet` | voice | SpeechBrain ResNet, VoxCeleb |
| `F_facenet` | face (bridge) | FaceNet InceptionResnetV1, VGGFace2 |
| `IB_ibvflip` | ImageBind face | mean of the image and its mirror (same bridge as CTRL) |

The voice and face swaps change the bridge **and** CCA-4. ImageBind and age are untouched. Same folds, rows, seeds and 1200-step budget. An arm whose arrays are missing is skipped with a message.

**Read the result as follows:**
- `prod0` gain vs CTRL: the embedding replaces the current one;
- `fuse0` gain (0.5 CTRL + 0.5 arm): the embedding adds a second, complementary bridge;
- section 7: Spearman correlation, on negatives, of the arm's bridge with ImageBind and with the current bridge. A low correlation with a good standalone `deep0` means complementary information.

**Gate:** as before, >= 1 EER in >= 4/5 folds, cell by cell.
""" + PROTOCOL + """
**Inputs:** `flag2027-eda-bundle` + `flag2027-feats-scout` (the output of FLAG_27; a partial run is fine: missing models are skipped). Optional: the COMBO-01 output (`l0_combo/results.csv`), to check that CTRL reproduces XALL_FIXW.
**Settings:** GPU T4 (or CPU); Internet OFF. **Time:** about 10-15 min per arm on a T4, about 1.5-2 h for 9 arms. **Output:** `/kaggle/working/l0_scout/`. **Resume:** attach this notebook's earlier output.
"""

SCOUT_CELL = r"""
# ============================================================================ EMBED-SCOUT: new embeddings in the bridge
FIXW = np.array([0.298, 0.405, 0.948, 0.327])          # registered VER-01 weights (log 91); CTRL = XALL_FIXW of COMBO-01
VOICE_ENC = ["rdn6vox", "wsr221", "wssim34", "wavlmsv", "unisatsv", "sbresnet"]
FACE_ENC, IB_ENC = ["facenet"], ["ibvflip"]
SRCS = ("v1_complete", "v2_complete")                    # = the order of FMETA / VMETA (LOAD cell)


def find_arr(name):
    hits = sorted(p for p in IN.rglob(name) if "eda_bundle" not in p.parts)
    return np.load(hits[0]).astype(np.float32) if hits else None


SCOUT = {}
for kind, enc in [("voice", e) for e in VOICE_ENC] + [("face", e) for e in FACE_ENC + IB_ENC]:
    v4a = find_arr(f"{kind}_{enc}_train.npy")
    ex = [find_arr(f"ext_{src}_{kind}_{enc}.npy") for src in SRCS]
    if v4a is None or any(x is None for x in ex):
        log(f"{kind} {enc}: arrays not found (attach flag2027-feats-scout) -> its arm is skipped")
        continue
    exa = np.concatenate(ex)
    n4, ne = len(V4S), (len(FMETA) if kind == "face" else len(VMETA))
    assert len(v4a) == n4 and len(exa) == ne, f"{enc}: rows {len(v4a)} / {len(exa)} vs {n4} / {ne}"
    assert np.isfinite(v4a).all() and np.isfinite(exa).all(), f"{enc}: non-finite values"
    SCOUT[enc] = dict(kind=kind, v4=v4a, ext=exa)
    log(f"{kind:5s} {enc:9s}: v4 {v4a.shape}, v1/v2 {exa.shape}")

_MEM = {}


def scout(fold, face="vgg", voice="btc", ibf="ibv"):
    # XALL_FIXW with one input swapped; members are cached per (fold, face, voice), so IB arms reuse CTRL's bridge
    F4, FE = (V4["vgg"], FA["vgg"]) if face == "vgg" else (SCOUT[face]["v4"], SCOUT[face]["ext"])
    Vv4, VE = (V4["btc"], VA["btc"]) if voice == "btc" else (SCOUT[voice]["v4"], SCOUT[voice]["ext"])
    I4, IE = (V4["ibf"], FA["ibv"]) if ibf == "ibv" else (SCOUT[ibf]["v4"], SCOUT[ibf]["ext"])
    key = (fold["s"], face, voice)
    if key not in _MEM:
        TF, TV, TS, info = train_rows(fold, ext=True, arrays=(F4, Vv4, FE, VE))
        ep, bpe = epochs_for(len(TS))
        units = [train_one(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep) for i in range(N_NETS)]
        _MEM[key] = (Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe,
                                                  d_face=TF.shape[1], d_voice=TV.shape[1]))
    mem, info = _MEM[key]

    def retarget(T, data):
        v4 = data == "v4"
        return dict(T, vgg=(F4 if v4 else FE)[T["fi"]], btc=(Vv4 if v4 else VE)[T["vi"]], ibf=(I4 if v4 else IE)[T["fi"]])

    return dict(mem=mem, info=info, X4=dict(V4, vgg=F4, btc=Vv4, ibf=I4), retarget=retarget,
                final=lambda S: np.stack([S[k] for k in ("deep0", "cca", "ib", "age")], 1) @ FIXW)


TRAIN = {"CTRL": lambda fold: scout(fold)}
TRAIN.update({f"V_{e}": (lambda fold, e=e: scout(fold, voice=e)) for e in VOICE_ENC})
TRAIN.update({f"F_{e}": (lambda fold, e=e: scout(fold, face=e)) for e in FACE_ENC})
TRAIN.update({f"IB_{e}": (lambda fold, e=e: scout(fold, ibf=e)) for e in IB_ENC})
skipped = [a for a in ARMS if a != "CTRL" and a.split("_", 1)[1] not in SCOUT]
ARMS = [a for a in ARMS if a not in skipped]
COMPARISONS = [(a, "CTRL") for a in ARMS if a != "CTRL"]
log("arms:", ARMS, "| skipped (no arrays):", skipped or "none")
"""

SCOUT_SUMMARY = r"""
# ============================================================================ complementarity (Spearman on negatives / positives)
from scipy.stats import spearmanr
rows = []
for f in sorted((OUT / "scores").glob("fold*_CTRL.npz")):
    s = int(re.search(r"fold(\d+)_", f.name).group(1))
    C = dict(np.load(f))
    for g in sorted((OUT / "scores").glob(f"fold{s}_*.npz")):
        arm = g.stem.split("_", 1)[1]
        if arm == "CTRL":
            continue
        A_ = dict(np.load(g))
        for k in [x[:-2] for x in C if x.endswith("_y")]:
            y = C[k + "_y"]
            for lab, m in (("neg", y == 0), ("pos", y == 1)):
                rows.append(dict(fold=s, arm=arm, cell=k, trials=lab,
                                 rho_arm_ib=spearmanr(A_[k + "_deep0"][m], A_[k + "_ib"][m])[0],
                                 rho_ctrl_ib=spearmanr(C[k + "_deep0"][m], C[k + "_ib"][m])[0],
                                 rho_arm_ctrl=spearmanr(A_[k + "_deep0"][m], C[k + "_deep0"][m])[0]))
if rows:
    CO = pd.DataFrame(rows).groupby(["arm", "cell", "trials"]).mean(numeric_only=True).drop(columns="fold").round(2)
    CO.to_csv(OUT / "complementarity.csv")
    print("Spearman on NEGATIVES (mean of folds). rho_arm_ctrl = the new bridge vs the current one;")
    print("rho_arm_ib / rho_ctrl_ib = each bridge vs ImageBind (lower = more complementary)")
    print(CO.xs("neg", level="trials").to_string())
    R_ = pd.read_csv(RES); R_["cell"] = R_.data + " " + R_.protocol
    print("\ndeep0 (bridge alone) EER, mean of folds:")
    print(R_[R_.variant == "deep0"].pivot_table(index="cell", columns="arm", values="eer").round(2).to_string())
# CTRL must reproduce XALL_FIXW of COMBO-01 (same folds, rows, seeds, weights) when that output is attached
ref = [p for p in IN.rglob("results.csv") if p.parent.name == "l0_combo"]
if ref:
    a = pd.read_csv(RES); b = pd.read_csv(ref[0])
    m = a[a.arm == "CTRL"].merge(b[b.arm == "XALL_FIXW"], on=["fold", "data", "protocol", "variant"])
    m = m[m.variant.isin(["deep0", "member0", "prod0"])]
    print(f"\nCTRL vs COMBO-01 XALL_FIXW: {len(m)} rows, max |EER diff| = {(m.eer_x - m.eer_y).abs().max():.4f} (expected 0)")
"""

# ----------------------------------------------------------------------------------------------- FLAG_29: bridge regularisation
REG_INTRO = """
# FLAG 2027 — NB-29: a bridge that overfits the training identities less (REG-01)

**Question:** ERR-03 (log 99) found that the MLP bridge **overfits the training identities**. On new persons its negatives sit +1.2…+1.7 sd above the train-pair constants (+0.3 with XALL), and its errors are systematic. Adding persons (XALL) was the cure so far. Does a less overfitted bridge help further, at the same data?

Every arm is **XALL_FIXW** (COMBO-01's best: XALL rows, MLP + CCA-4, ImageBind + age, registered VER-01 weights). Only the bridge training changes:

| Arm | Change | Expected if the overfitting story is right |
|---|---|---|
| `CTRL` | none (EXP-007: dropout 0.3, wd 0.01, emb 128, 1200 steps) | — |
| `S600` | 600 steps | better if training longer memorises identities |
| `S2400` | 2400 steps | worse (control for the direction) |
| `DROP5` | dropout 0.5 | better |
| `WD10` | weight decay 0.1 | better |
| `EMB32` | embedding 32 instead of 128 | better (less capacity) |

The cosine-annealed learning rate always spans the arm's own budget. Section 7 reports, per arm, the mean of the bridge stream on negatives in units of the train-pair constants: the ERR-03 measure of overfitting, which should move toward 0 if an arm overfits less.
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** about 1.5 h on a T4. **Output:** `/kaggle/working/l0_reg/`. **Resume:** attach this notebook's earlier output.
"""

REG_ARMS = r"""
# ============================================================================ regularisation arms (all on XALL rows, FIXW weights)
FIXW = np.array([0.298, 0.405, 0.948, 0.327])          # registered VER-01 weights (log 91)
fixw = lambda S: np.stack([S[k] for k in ("deep0", "cca", "ib", "age")], 1) @ FIXW


def arm_reg(fold, steps=None, **over):
    TF, TV, TS, info = train_rows(fold, ext=True)
    ep, bpe = epochs_for(len(TS), steps)
    if over:
        units = [train_pk(TF, TV, TS, fold["s"] + 100 * i, ep, cfg_over=over) for i in range(N_NETS)]
    else:
        units = [train_one(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe), fixw


TRAIN = {"CTRL": lambda fold: arm_reg(fold), "S600": lambda fold: arm_reg(fold, steps=600),
         "S2400": lambda fold: arm_reg(fold, steps=2400), "DROP5": lambda fold: arm_reg(fold, drop=0.5),
         "WD10": lambda fold: arm_reg(fold, wd=0.1), "EMB32": lambda fold: arm_reg(fold, emb=32)}
COMPARISONS = [(a, "CTRL") for a in TRAIN if a != "CTRL"] + [("S600", "S2400")]
"""

REG_SUMMARY = r"""
# ============================================================================ overfitting measure per arm (ERR-03, part D)
rows = []
for f in sorted((OUT / "scores").glob("fold*_*.npz")):
    arm = f.stem.split("_", 1)[1]
    Z = dict(np.load(f))
    for k in [x[:-2] for x in Z if x.endswith("_y")]:
        y = Z[k + "_y"]
        rows.append(dict(arm=arm, cell=k, neg_mean_deep=float(Z[k + "_deep0"][y == 0].mean()),
                         pos_mean_deep=float(Z[k + "_deep0"][y == 1].mean())))
if rows:
    OV = pd.DataFrame(rows).groupby(["cell", "arm"]).mean().round(2)
    OV.to_csv(OUT / "overfit.csv")
    print("mean of the bridge stream on NEGATIVES (units of the train-pair constants; 0 = no identity overfitting)")
    print(OV.neg_mean_deep.unstack("arm").to_string())
# CTRL must reproduce COMBO-01 XALL_FIXW when that output is attached
ref = [p for p in IN.rglob("results.csv") if p.parent.name == "l0_combo"]
if ref:
    a = pd.read_csv(RES); b = pd.read_csv(ref[0])
    m = a[a.arm == "CTRL"].merge(b[b.arm == "XALL_FIXW"], on=["fold", "data", "protocol", "variant"])
    m = m[m.variant.isin(["deep0", "member0", "prod0"])]
    print(f"\nCTRL vs COMBO-01 XALL_FIXW: {len(m)} rows, max |EER diff| = {(m.eer_x - m.eer_y).abs().max():.4f} (expected 0)")
"""

# ----------------------------------------------------------------------------------------------- FLAG_36: same-gender SupCon / simplex prototypes
LOSSG_INTRO = """
# FLAG 2027 — NB-36: same-gender SupCon and fixed simplex prototypes for the bridge (LOSSG-01)

**Question:** two changes to the bridge loss, proposed 05/10. Do they help the bridge where the system is weakest, the **g cells** (negatives of the same gender)?
1. **SupCon restricted to the same gender:** a supervised contrastive term (Khosla et al., 2020) on the 2B embeddings of a PK batch (faces and voices as two views; positives = the same person). The denominator keeps only rows of the **same gender** as the anchor, so the model is never rewarded for telling men from women. It is ADDED to the InfoNCE loss, which keeps its cross-gender negatives. This differs from HARD-01 (log 92), where whole batches were made single-gender and the ng cells lost the gender cue.
2. **Fixed simplex prototypes** (maximum class separation; Kasarla et al., NeurIPS 2022; the organisers' Moscati et al., ICASSP 2026): one fixed unit vector per training person, pairwise cosine -1/(Ns - 1) (embedding dimension Ns - 1). Faces and voices are each pulled to their person's prototype (cross-entropy on cosine / tau). This differs from MCS_INFO (FLAG_18), which put CE + OPL on the fused face + voice vector.

Every arm is **XALL_FIXW** (as FLAG_29's CTRL: XALL rows = 40 v4 + the external pool, MLP + CCA-4, ImageBind + age, the registered VER-01 weights). Only the bridge loss (and, for the prototype arms, the embedding dimension) changes:

| Arm | Bridge loss | Embedding | Differs from |
|---|---|---|---|
| `CTRL` | InfoNCE (EXP-007) | 128 | — |
| `SCG50` | InfoNCE + 0.5 x SupCon (same gender) | 128 | `CTRL`: the SupCon term |
| `SCG100` | InfoNCE + 1.0 x SupCon (same gender) | 128 | `CTRL`: the SupCon term |
| `CTRL_D` | InfoNCE | Ns - 1 | `CTRL`: the dimension only |
| `SPX` | InfoNCE + 1.0 x prototype CE (face and voice) | Ns - 1 | `CTRL_D`: the prototypes |
| `SCG50_SPX` | InfoNCE + 0.5 x SupCon + 1.0 x prototype CE | Ns - 1 | `SPX`: the SupCon term |

**Decision (registered 05/10, before any result).** The decision is NOT taken in this notebook. This notebook only produces the bridge scores. Locally, `Experiment/LOSSG-01/gate.py` puts each arm's bridge stream in place of `XALL_deep0` inside CA-FUSE (BEST v9's Bangla fuser, refit per fold). An arm is kept only if, against the same replacement with this notebook's `CTRL`, it gains >= 1.0 EER on the mean of the g cells (with >= 4/5 folds better in >= 2 g cells) and is not worse than -0.2 in either held-out-language check (Urdu or Hindi removed from the fuser's training rows). The paired summary below (FIXW recipe) and the overfitting measure (mean bridge score of NEGATIVE pairs on new persons, ERR-03) are informative only.
**Prior:** HARD-01 (same-gender batches) and MCS-01 (simplex head) both closed. The bridge also gets a negative weight in CA-FUSE for same-gender pairs (log 124). Expected chance to pass: about 10 %.
""" + PROTOCOL + """
**Settings:** GPU T4 (or CPU); Internet OFF. **Input:** `flag2027-eda-bundle` only. **Time:** about 1.5-2 h on a T4 (6 arms x 5 folds x 3 nets, about 3 min per fold x arm); CPU about 5 h. **Output:** `/kaggle/working/l0_lossg/` (results.csv, summary.csv, overfit.csv, scores/*.npz, trials_fold*.npz). **Resume:** attach this notebook's earlier output; finished (fold, arm) pairs are skipped.
"""

LOSSG_ARMS = r'''
# ============================================================================ loss arms (all on XALL rows, FIXW weights = FLAG_29 CTRL)
FIXW = np.array([0.298, 0.405, 0.948, 0.327])          # registered VER-01 weights (log 91)
fixw = lambda S: np.stack([S[k] for k in ("deep0", "cca", "ib", "age")], 1) @ FIXW


def maxsep(n):
    """Kasarla et al. (2022) eq. 1: (n-1) x n matrix, unit columns, pairwise cosine -1/(n-1) (= FLAG_18)."""
    P = np.array([[1.0, -1.0]])
    for k in range(2, n):
        P = np.block([[np.ones((1, 1)), -np.ones((1, k)) / k], [np.zeros((k - 1, 1)), np.sqrt(1 - 1 / k ** 2) * P]])
    return P.astype(np.float32)


_P = maxsep(160)
assert np.allclose(np.linalg.norm(_P, axis=0), 1, atol=1e-5) and np.allclose((_P.T @ _P)[~np.eye(160, dtype=bool)], -1 / 159, atol=1e-5)


def supcon_g(u, w, y, g, tau):
    """SupCon (Khosla et al. 2020) on the 2B unit embeddings [faces; voices] of a PK batch, positives = same person
    (both modalities, self excluded). The softmax denominator keeps only rows of the anchor's gender: a different-
    gender row is never a negative. Positives share the anchor's gender, so they always stay in the denominator."""
    z = torch.cat([u, w]); yy = torch.cat([y, y]); gg = torch.cat([g, g])
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (yy[:, None] == yy[None, :]) & ~eye
    keep = (gg[:, None] == gg[None, :]) & ~eye
    S = (z @ z.T / tau).masked_fill(~keep, -1e4)
    logp = S - torch.logsumexp(S, 1, keepdim=True)
    return -((logp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


class ProtoNet(Model):
    """EXP-007 MLP with emb = Ns - 1 and the fixed simplex prototypes P (one unit column per training person)."""
    def __init__(self, d_f, d_v, n_cls, cfg):
        super().__init__(d_f, d_v, n_cls, cfg)
        self.register_buffer("P", torch.tensor(maxsep(n_cls)))


def make_loss(w_sc=0.0, w_proto=0.0):
    def loss_fn(net, f, v, y, g):
        u, w = net(f, v)
        loss = infonce(u, w, y, REC["tau"])
        if w_sc:
            loss = loss + w_sc * supcon_g(u, w, y, g, REC["tau"])
        if w_proto:
            loss = loss + w_proto * 0.5 * (F.cross_entropy(u @ net.P / REC["tau"], y) + F.cross_entropy(w @ net.P / REC["tau"], y))
        return loss
    return loss_fn


def arm_ctrl_fixw(fold):
    """= FLAG_29 CTRL (XALL_FIXW): train_one on XALL rows."""
    TF, TV, TS, info = train_rows(fold, ext=True)
    ep, bpe = epochs_for(len(TS))
    units = [train_one(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe), fixw


def arm_loss(fold, w_sc=0.0, w_proto=0.0, dim_ns=False):
    """train_pk on XALL rows (same RNG calls as train_one: with no extra term and emb 128 it reproduces CTRL bit for bit)."""
    TF, TV, TS, info = train_rows(fold, ext=True)
    ep, bpe = epochs_for(len(TS))
    mk = (lambda df, dv, nc: ProtoNet(df, dv, nc, dict(REC, emb=nc - 1))) if (dim_ns or w_proto) else None
    units = [train_pk(TF, TV, TS, fold["s"] + 100 * i, ep, make_net=mk, loss_fn=make_loss(w_sc, w_proto)) for i in range(N_NETS)]
    return Member(units, TF, TV), dict(info, train_ids=len(set(TS)), train_rows=len(TS), steps=ep * bpe), fixw


TRAIN = {"CTRL": arm_ctrl_fixw,
         "PK0": lambda fold: arm_loss(fold),                              # wiring check only (smoke test): must equal CTRL
         "SCG50": lambda fold: arm_loss(fold, w_sc=0.5), "SCG100": lambda fold: arm_loss(fold, w_sc=1.0),
         "CTRL_D": lambda fold: arm_loss(fold, dim_ns=True), "SPX": lambda fold: arm_loss(fold, w_proto=1.0),
         "SCG50_SPX": lambda fold: arm_loss(fold, w_sc=0.5, w_proto=1.0)}
COMPARISONS = [("SCG50", "CTRL"), ("SCG100", "CTRL"), ("CTRL_D", "CTRL"), ("SPX", "CTRL_D"), ("SPX", "CTRL"),
               ("SCG50_SPX", "SPX"), ("SCG50_SPX", "CTRL")]
'''

LOSSG_EXPORT = r'''
# ============================================================================ trial indices for the local CA-FUSE gate (Experiment/LOSSG-01/gate.py)
for s in FOLDS:
    fold = make_fold(s)
    np.savez_compressed(OUT / f"trials_fold{s}.npz", **{f"{d}_{p}_{k}": T_[k] for (d, p), T_ in fold["files"].items() for k in ("fi", "vi", "y")})
log("trial indices written:", sorted(f.name for f in OUT.glob("trials_fold*.npz")))
print(f"NEXT: download /kaggle/working/{OUT.name} (results.csv, summary.csv, overfit.csv, scores/*.npz, trials_fold*.npz) "
      "and hand it back: the decision is taken locally by Experiment/LOSSG-01/gate.py (CA-FUSE, registered rule)")
'''

NOTEBOOKS = [
    ("FLAG_18_l0_mcs.ipynb", "l0_mcs", "CTRL,MSM_EXACT,FOP_EXACT,CTRL_D,MCS_INFO", MCS_INTRO, MCS_ARMS, "MCS arms (official MSM code, ported)"),
    ("FLAG_19_l0_paeff.ipynb", "l0_paeff", "CTRL,PAEFF_EXACT,PAEFF_B,HYP_INFO", PAEFF_INTRO, PAEFF_ARMS, "PAEFF arms (official PAEFF code, ported) + hyperbolic InfoNCE"),
    ("FLAG_20_l0_verifier.ipynb", "l0_verifier", "CTRL,VER_MLP,VER_FUSE,VER_FUSE_X", VER_INTRO, VER_ARMS, "Verifier arms"),
    ("FLAG_21_l0_hardneg.ipynb", "l0_hardneg", "CTRL,R_GH50,R_GH100,XALL,XALL_GH50", HARD_INTRO, HARD_ARMS, "Hard-negative arms"),
    ("FLAG_22_l0_xvideo.ipynb", "l0_xvideo", "CTRL,R_DEC,XALL,XALL_XVID,XALL_XVID_DEC,XALL_PTFT", XVID_INTRO, XVID_ARMS, "Cross-clip / external arms"),
    ("FLAG_23_l0_combo.ipynb", "l0_combo", "CTRL,CTRL_FIXW,VER_FUSE,XALL,XALL_FIXW,XALL_FUSE", COMBO_INTRO, COMBO_ARMS, "XALL x stream-weight arms"),
]

def build_lossg():
    nb([("markdown", LOSSG_INTRO), ("code", ENV.replace("__OUTDIR__", "l0_lossg").replace("__ARMS__", "CTRL,SCG50,SCG100,CTRL_D,SPX,SCG50_SPX")),
        ("markdown", "## 1. Data (one bundle)"), ("code", LOAD),
        ("markdown", "## 2. Bridge and level-0 scoring (same cells as FLAG_18 ... 29)"), ("code", LIB),
        ("markdown", "## 3. Folds, training rows, eval files (same cell)"), ("code", DATA),
        ("markdown", "## 4. Loss arms: same-gender SupCon, fixed simplex prototypes"), ("code", LOSSG_ARMS),
        ("markdown", "## 5. Run every (fold, arm)"), ("code", RUN),
        ("markdown", "## 6. Paired summary (FIXW recipe; informative, the decision is the local CA-FUSE gate)"), ("code", SUMMARY),
        ("markdown", "## 7. Overfitting measure per arm (ERR-03)"), ("code", REG_SUMMARY),
        ("markdown", "## 8. Trial indices for the local CA-FUSE gate"), ("code", LOSSG_EXPORT)], "FLAG_36_l0_lossg.ipynb")


if __name__ == "__main__":
    import sys
    if "--only" in sys.argv:                          # python build_l0_notebooks.py --only lossg  (leaves FLAG_18 ... 29 untouched)
        {"lossg": build_lossg}[sys.argv[sys.argv.index("--only") + 1]]()
        sys.exit(0)
    build_lossg()
    for path, outdir, arms, intro, arm_code, title in NOTEBOOKS:
        nb([("markdown", intro), ("code", ENV.replace("__OUTDIR__", outdir).replace("__ARMS__", arms)),
            ("markdown", "## 1. Data (one bundle)"), ("code", LOAD),
            ("markdown", "## 2. Bridge and level-0 scoring (bridge = port of kaggle/flag_v2.py; scoring = BEST/v6_eval/pipeline.py indep())"), ("code", LIB),
            ("markdown", "## 3. Folds, training rows, eval files"), ("code", DATA),
            ("markdown", f"## 4. {title}"), ("code", arm_code),
            ("markdown", "## 5. Run every (fold, arm)"), ("code", RUN),
            ("markdown", "## 6. Paired summary"), ("code", SUMMARY)], path)
    nb([("markdown", DEV_INTRO), ("code", DEV_ENV), ("markdown", "## 1. Data (eda bundle: v4 train + MAV-Celeb v1/v2)"), ("code", LOAD),
        ("markdown", "## 2. Bridge and level-0 scoring (same cells as FLAG_18 ... 23)"), ("code", LIB),
        ("markdown", "## 3. Training rows (same cell as FLAG_18 ... 23; only ext_rows is used here)"), ("code", DATA),
        ("markdown", "## 4. Dev: swap s007, score at level 0, write the zips, self-tests"), ("code", DEV_RUN)], "FLAG_24_safe_dev.ipynb")
    nb([("markdown", ERR_INTRO), ("code", ENV.replace("__OUTDIR__", "l0_err").replace("__ARMS__", "CTRL,XALL")),
        ("markdown", "## 1. Data (one bundle)"), ("code", LOAD),
        ("markdown", "## 2. Bridge and level-0 scoring (same cells as FLAG_18 ... 24)"), ("code", LIB),
        ("markdown", "## 3. Folds, training rows, eval files (same cell as FLAG_18 ... 23)"), ("code", DATA),
        ("markdown", "## 4. Train CTRL / XALL per fold, score, keep every trial"), ("code", ERR_RUN),
        ("markdown", "## 5. Tables A-E"), ("code", ERR_SUMMARY)], "FLAG_26_l0_erranalysis.ipynb")
    nb([("markdown", SCOUT_INTRO), ("code", ENV.replace("__OUTDIR__", "l0_scout").replace("__ARMS__", SCOUT_ARMS)),
        ("markdown", "## 1. Data (eda bundle)"), ("code", LOAD),
        ("markdown", "## 2. Bridge and level-0 scoring (same cells as FLAG_18 ... 26)"), ("code", LIB),
        ("markdown", "## 3. Folds, training rows, eval files (same cell as FLAG_18 ... 26)"), ("code", DATA),
        ("markdown", "## 4. EMBED-SCOUT arms: load the new embeddings, swap them into the bridge"), ("code", SCOUT_CELL),
        ("markdown", "## 5. Run every (fold, arm)"), ("code", RUN),
        ("markdown", "## 6. Paired summary vs CTRL (= XALL_FIXW)"), ("code", SUMMARY),
        ("markdown", "## 7. Complementarity: correlation with ImageBind and with the current bridge"), ("code", SCOUT_SUMMARY)],
       "FLAG_28_l0_scout.ipynb")
    nb([("markdown", REG_INTRO), ("code", ENV.replace("__OUTDIR__", "l0_reg").replace("__ARMS__", "CTRL,S600,S2400,DROP5,WD10,EMB32")),
        ("markdown", "## 1. Data (one bundle)"), ("code", LOAD),
        ("markdown", "## 2. Bridge and level-0 scoring (same cells as FLAG_18 ... 28)"), ("code", LIB),
        ("markdown", "## 3. Folds, training rows, eval files (same cell)"), ("code", DATA),
        ("markdown", "## 4. Regularisation arms"), ("code", REG_ARMS),
        ("markdown", "## 5. Run every (fold, arm)"), ("code", RUN),
        ("markdown", "## 6. Paired summary"), ("code", SUMMARY),
        ("markdown", "## 7. Overfitting measure per arm"), ("code", REG_SUMMARY)], "FLAG_29_l0_reg.ipynb")
