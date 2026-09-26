"""EXP-009 — the FAME 2026 winner's recipe (23.99 EER), rebuilt on our data.

Their system, as described in arXiv:2512.04814:
  * face  : VGGFace, layer before the output, 4096-d          <- exactly what the organisers ship
  * voice : ECAPA-TDNN, layer before the output, 6144-d       <- we ship 192-d; needs FLAG_03 to extract
  * extra : age-gender streams (ECAPA 1536-d audio, ViT 768-d image), concatenated per modality
  * head  : ONE linear mapping layer per modality down to 192-d, everything else frozen
  * reg   : dropout 0.9 (!)
  * loss  : Additive Angular Margin (AAM), cosine scoring

That head is far shallower than ours (2-layer MLP + BN + ReLU, dropout 0.3-0.5) and far more
regularised. With 70 speakers that combination is plausible, so it is worth isolating.

This script tests the four components that do NOT need new features, on the features we already have:
    head      : linear   vs  mlp        (ours)
    dropout   : 0.9      vs  0.3
    loss      : aam      vs  infonce    (ours)   vs  aam+align
    embedding : 192      vs  128        (ours)
The 6144-d voice and the age-gender streams are added in the Kaggle notebook once extracted.

Usage: python run.py grid | python run.py final
"""
import sys, zipfile, itertools, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EXP-003_deep"))
import train as T
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
NAMES = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
CCA_CFG = dict(k=4, reg=1.0, pca_x=128)
PCA_FACE = 256


def z(x): return (x - x.mean()) / (x.std() + 1e-8)
def center(X, mu): return X - X.mean(0) + mu


class Prep:
    _cache = {}

    def __init__(self, Xf, Xv, key):
        self.mf, self.sf = Xf.mean(0), Xf.std(0) + 1e-6
        self.mv, self.sv = Xv.mean(0), Xv.std(0) + 1e-6
        ck = (key, Xf.shape, float(Xf[0].sum()))
        if ck not in Prep._cache:
            _, _, Vt = np.linalg.svd((Xf - self.mf) / self.sf, full_matrices=False)
            Prep._cache[ck] = Vt[:min(PCA_FACE, Xf.shape[1])].T
        self.P = Prep._cache[ck]

    def f(self, X, pca=True):
        Z = (X - self.mf) / self.sf
        return (Z @ self.P if pca else Z).astype(np.float32)

    def v(self, X):
        return ((X - self.mv) / self.sv).astype(np.float32)


class Branch(nn.Module):
    """head='linear' reproduces the winner's single mapping layer; 'mlp' is our current branch."""

    def __init__(self, d_in, emb, head, drop, hid=512):
        super().__init__()
        if head == "linear":
            self.net = nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, emb))
        else:
            self.net = nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, hid), nn.BatchNorm1d(hid),
                                     nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, emb))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SharedAAM(nn.Module):
    """One class-centre matrix W used by BOTH modalities.

    Sharing W is what ties the two spaces together: face and voice of the same speaker are pulled
    to the same centre. It is still only an indirect alignment with 70 centres, which is why
    `align_w` can add an explicit cross-modal term.
    """

    def __init__(self, emb, n_cls, margin=0.2, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_cls, emb))
        nn.init.xavier_normal_(self.W)
        self.m, self.s = margin, scale

    def logits(self, zn, y):
        cos = torch.clamp(zn @ F.normalize(self.W, dim=1).T, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        oh = F.one_hot(y, self.W.shape[0]).float()
        return self.s * (oh * target + (1 - oh) * cos)

    def forward(self, zf, zv, y):
        return 0.5 * (F.cross_entropy(self.logits(zf, y), y) + F.cross_entropy(self.logits(zv, y), y))


def infonce(u, w, y, tau=0.07):
    S = u @ w.T / tau
    same = (y[:, None] == y[None, :]).float()
    return 0.5 * ((-(F.log_softmax(S, 1) * same).sum(1) / same.sum(1)).mean() +
                  (-(F.log_softmax(S, 0) * same).sum(0) / same.sum(0)).mean())


def train_one(Xf, Xv, spk, tr_idx, cfg, seed, prep):
    torch.manual_seed(seed); np.random.seed(seed)
    use_pca = cfg.get("pca_face", True)
    Ftr = torch.tensor(prep.f(Xf[tr_idx], pca=use_pca))
    Vtr = torch.tensor(prep.v(Xv[tr_idx]))
    spk_tr = np.asarray(spk)[tr_idx]
    cls = {s: i for i, s in enumerate(np.unique(spk_tr))}
    Y = torch.tensor([cls[s] for s in spk_tr])
    bf = Branch(Ftr.shape[1], cfg["emb"], cfg["head"], cfg["drop"])
    bv = Branch(Vtr.shape[1], cfg["emb"], cfg["head"], cfg["drop"])
    aam = SharedAAM(cfg["emb"], len(cls), cfg.get("margin", 0.2), cfg.get("scale", 30.0))
    params = list(bf.parameters()) + list(bv.parameters()) + list(aam.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    rng = np.random.RandomState(seed)
    for _ in range(cfg["epochs"]):
        bf.train(); bv.train()
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(order) - cfg["bs"] // 2, cfg["bs"]):
            b = torch.tensor(order[s:s + cfg["bs"]])
            zf, zv, y = bf(Ftr[b]), bv(Vtr[b]), Y[b]
            if cfg["loss"] == "infonce":
                loss = infonce(zf, zv, y)
            elif cfg["loss"] == "aam":
                loss = aam(zf, zv, y)
            else:                                   # aam + explicit cross-modal alignment
                loss = aam(zf, zv, y) + cfg.get("align_w", 1.0) * infonce(zf, zv, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return bf, bv


@torch.no_grad()
def embed(bf, bv, prep, Xf_, Xv_, mu_f, mu_v, use_pca=True):
    bf.eval(); bv.eval()
    A = torch.tensor(center(prep.f(Xf_, pca=use_pca), mu_f))
    B = torch.tensor(center(prep.v(Xv_), mu_v))
    return bf(A).numpy(), bv(B).numpy()


def evaluate(cfg, Xf, Xv, spk, gmap, seeds=(1, 2, 3), n_models=3, cca_weight=0.25, tag="x"):
    """tag is only for logging; the PCA cache is keyed on the split + a feature fingerprint."""
    out = {"no_gender": [], "gender": []}
    for s in seeds:
        tr, va = speaker_split(spk, s); tri, vai = np.where(tr)[0], np.where(va)[0]
        prep = Prep(Xf[tri], Xv[tri], key=f"split{s}")
        use_pca = cfg.get("pca_face", True)
        mu_f = prep.f(Xf[tri], pca=use_pca).mean(0); mu_v = prep.v(Xv[tri]).mean(0)
        embs = []
        for i in range(n_models):
            bf, bv = train_one(Xf, Xv, spk, tri, cfg, s + 100 * i, prep)
            embs.append(embed(bf, bv, prep, Xf[vai], Xv[vai], mu_f, mu_v, use_pca))
        cca = cca_for_split(Xf, Xv, tri, vai, s)
        for sg in (False, True):
            fi, vj, lab = build_trials(spk[vai], gmap, seed=s, n_pos=3000, n_neg=3000, same_gender=sg)
            deep = np.mean([z(np.sum(Ef[fi] * Ev[vj], 1)) for Ef, Ev in embs], 0)
            sc = (1 - cca_weight) * deep + cca_weight * z(np.sum(cca[0][fi] * cca[1][vj], 1))
            out["gender" if sg else "no_gender"].append(eer_from_scores(sc, lab))
    return float(np.mean(out["no_gender"])), float(np.mean(out["gender"]))


def cca_for_split(Xf, Xv, tri, vai, s):
    """CCA projections of the val rows; identical for every config, so compute once per split."""
    ck = ("cca", s, Xf.shape, Xv.shape)
    if ck not in Prep._cache:
        f = OUT / f"cca_s{s}_{Xf.shape[1]}_{Xv.shape[1]}.npz"
        if f.exists():
            d = np.load(f); Prep._cache[ck] = (d["a"], d["b"])
        else:
            m = RidgeCCA(**CCA_CFG).fit(Xf[tri], Xv[tri])
            a = m.transform_x(center(Xf[vai], Xf[tri].mean(0)))
            b = m.transform_y(center(Xv[vai], Xv[tri].mean(0)))
            np.savez(f, a=a, b=b); Prep._cache[ck] = (a, b)
    return Prep._cache[ck]


BASE = dict(lr=1e-3, wd=1e-2, bs=256, epochs=40, emb=192, head="linear", drop=0.9, loss="aam", pca_face=True)

if __name__ == "__main__":
    Xf, Xv, spk, _, txt, gmap = load_train()
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"

    if mode == "grid":
        GRID = {
            "ours (mlp+infonce, drop .3, emb128)": dict(BASE, head="mlp", drop=0.3, loss="infonce", emb=128),
            "FAME1 (linear+aam, drop .9, emb192)": dict(BASE),
            "FAME1 no-pca-face (4096 raw)":        dict(BASE, pca_face=False),
            "FAME1 + align":                       dict(BASE, loss="aam_align", align_w=1.0),
            "FAME1 drop .5":                       dict(BASE, drop=0.5),
            "FAME1 drop .7":                       dict(BASE, drop=0.7),
            "FAME1 emb128":                        dict(BASE, emb=128),
            "FAME1 margin .3":                     dict(BASE, margin=0.3),
            "FAME1 margin .1":                     dict(BASE, margin=0.1),
            "FAME1 scale 16":                      dict(BASE, scale=16.0),
            "FAME1 mlp head":                      dict(BASE, head="mlp"),
            "FAME1 infonce (isolate the loss)":    dict(BASE, loss="infonce"),
            "FAME1 100ep":                         dict(BASE, epochs=100),
            "FAME1 200ep":                         dict(BASE, epochs=200),
            "FAME1 100ep lr3e-3":                  dict(BASE, epochs=100, lr=3e-3),
            "FAME1 100ep + align":                 dict(BASE, epochs=100, loss="aam_align"),
            "ours + aam (swap loss only)":         dict(BASE, head="mlp", drop=0.3, emb=128, loss="aam"),
            "ours + drop .9 (swap reg only)":      dict(BASE, head="mlp", drop=0.9, emb=128, loss="infonce", epochs=100),
        }
        rows = []
        for name, cfg in GRID.items():
            t0 = time.time()
            ng, g = evaluate(cfg, Xf, Xv, spk, gmap, tag=name[:12])
            rows.append(dict(cfg=name, int_ng=round(ng, 2), int_g=round(g, 2),
                             proxy=round((ng + g) / 2, 2), sec=round(time.time() - t0)))
            pd.DataFrame(rows).to_csv(OUT / "grid.csv", index=False)
            print(f"{name:40s} int_ng {ng:6.2f}  int_g {g:6.2f}  ({round(time.time()-t0)}s)", flush=True)
        print()
        print(pd.DataFrame(rows).sort_values("int_g").to_string(index=False))

    else:
        df = pd.read_csv(OUT / "grid.csv").sort_values("int_g")
        name = df.iloc[0].cfg
        cfg = {k: v for k, v in BASE.items()}
        print("final cfg from grid:", name)
        allidx = np.arange(len(spk))
        prep = Prep(Xf, Xv, key="final")
        use_pca = cfg.get("pca_face", True)
        mu_f, mu_v = prep.f(Xf, pca=use_pca).mean(0), prep.v(Xv).mean(0)
        models = [train_one(Xf, Xv, spk, allidx, cfg, 1 + 100 * i, prep) for i in range(10)]
        m = RidgeCCA(**CCA_CFG).fit(Xf, Xv)
        devs = {k: load_dev(*k) for k in NAMES}
        with zipfile.ZipFile(OUT / "submission_EXP009_fame1.zip", "w") as zf:
            for k, fn in NAMES.items():
                a, b, t = devs[k]; ii = np.arange(len(t))
                embs = [embed(bf, bv, prep, a, b, mu_f, mu_v, use_pca) for bf, bv in models]
                deep = np.mean([z(np.sum(Ef * Ev, 1)) for Ef, Ev in embs], 0)
                cc = z(np.sum(m.transform_x(center(a, Xf.mean(0))) * m.transform_y(center(b, Xv.mean(0))), 1))
                sc = 0.75 * deep + 0.25 * cc
                zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(t.pair_id, -sc)) + "\n")
        print("wrote", OUT / "submission_EXP009_fame1.zip")
