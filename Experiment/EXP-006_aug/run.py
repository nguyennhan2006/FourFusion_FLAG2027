"""EXP-006 — input-space augmentation so the model stops relying on language-sensitive voice dimensions.

EDA-5 put the English/Bangla shift almost entirely in voice; EXP-004 showed we cannot remove it at test
time without destroying identity. The remaining lever is to make the encoder invariant to it during training.

Augmentations (voice branch unless stated):
  noise<s>   : additive Gaussian noise, std = s * per-dim train std
  dropdim<p> : randomly zero a fraction p of voice dims (scaled), like feature dropout
  mixup<a>   : convex mix of two voices of the SAME speaker (identity preserved, nuisance averaged)
  shift<s>   : add a random constant offset vector drawn from the train mean-shift scale (mimics a file offset)
  both<s>    : noise on voice AND face

Internal (English-only) can only show the cost of an augmentation, not its Bangla benefit — so the rule is
"pick anything that does not hurt internal, then measure Bangla on CodaBench".
Usage: python run.py internal | python run.py final <aug1,aug2,...>
"""
import sys, zipfile
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EXP-003_deep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EXP-004_domain"))
import train as T
from run import Aligner, NAMES
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
CFG_NAME, EPOCHS = "C_infonce_e128", 15
CCA_K4 = dict(k=4, reg=1.0, pca_x=128)
AUGS = ["none", "noise0.1", "noise0.3", "noise0.5", "dropdim0.1", "dropdim0.3",
        "mixup0.3", "shift0.5", "shift1.0", "both0.3"]


def z(x): return (x - x.mean()) / (x.std() + 1e-8)


def apply_aug(fb, vb, aug, rng_t, sdF, sdV, spk_b, idx_by_spk, Vall):
    if aug == "none":
        return fb, vb
    kind = "".join(c for c in aug if c.isalpha())
    p = float(aug[len(kind):])
    if kind == "noise":
        vb = vb + torch.randn_like(vb) * sdV * p
    elif kind == "both":
        vb = vb + torch.randn_like(vb) * sdV * p
        fb = fb + torch.randn_like(fb) * sdF * p
    elif kind == "dropdim":
        m = (torch.rand_like(vb) > p).float() / (1 - p)
        vb = vb * m
    elif kind == "shift":
        # one random offset per batch element, drawn along the train std -> mimics a per-file offset
        vb = vb + torch.randn(vb.shape[0], 1) * sdV * p
    elif kind == "mixup":
        lam = 1 - p * torch.rand(vb.shape[0], 1)
        partner = torch.tensor([np.random.choice(idx_by_spk[s]) for s in spk_b])
        vb = lam * vb + (1 - lam) * Vall[partner]
    return fb, vb


def run_aug(aug, tr_idx, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = T.CFGS[CFG_NAME]
    prep = T.Prep(T.Xf[tr_idx], T.Xv[tr_idx], cache=T.OUT / f"prep_n{len(tr_idx)}_s{int(np.asarray(tr_idx).sum())}.npz")
    Ftr, Vtr = torch.tensor(prep.f(T.Xf[tr_idx])), torch.tensor(prep.v(T.Xv[tr_idx]))
    sdF, sdV = Ftr.std(0, keepdim=True), Vtr.std(0, keepdim=True)
    spk_tr = T.yf[tr_idx]; cls = {s: i for i, s in enumerate(np.unique(spk_tr))}
    idx_by_spk = {s: np.where(spk_tr == s)[0] for s in np.unique(spk_tr)}
    Ytr = torch.tensor([cls[s] for s in spk_tr]); Gtr = torch.tensor(T.g_int[tr_idx]); Ctr = T.vgrp[tr_idx]
    net = T.Net(T.PCA_FACE, 192, cfg["emb"], cfg["hid"], cfg["drop"], len(cls))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(tr_idx)); _, first = np.unique(Ctr[perm], return_index=True); order = perm[np.sort(first)]
        for s in range(0, len(order) - cfg["bs"] // 2, cfg["bs"]):
            b = order[s:s + cfg["bs"]]; bt = torch.tensor(b)
            fb, vb = apply_aug(Ftr[bt], Vtr[bt], aug, rng, sdF, sdV, spk_tr[b], idx_by_spk, Vtr)
            u, w, logits = net(fb, vb)
            loss = T.infonce(u, w, Ytr[bt], Gtr[bt], cfg["tau"], False)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net, prep


def embed(models, cca, Xf_, Xv_):
    es = []
    for net, prep, aF, aV in models:
        A = (prep.f(Xf_) - prep.f(Xf_).mean(0) + aF.mu).astype(np.float32)
        B = (prep.v(Xv_) - prep.v(Xv_).mean(0) + aV.mu).astype(np.float32)
        net.eval()
        with torch.no_grad():
            u, w, _ = net(torch.tensor(A), torch.tensor(B))
        es.append((u.numpy(), w.numpy()))
    if cca is not None:
        fa = Xf_ - Xf_.mean(0) + cca["alF"].mu; va = Xv_ - Xv_.mean(0) + cca["alV"].mu
        es.append((cca["m"].transform_x(fa), cca["m"].transform_y(va)))
    return es


def build(aug, tr_idx, seeds):
    ms = []
    for s in seeds:
        net, prep = run_aug(aug, tr_idx, s, EPOCHS)
        ms.append((net, prep, Aligner(prep.f(T.Xf[tr_idx])), Aligner(prep.v(T.Xv[tr_idx]))))
    return ms


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "internal":
        rows = []
        for seed in T.SEEDS:
            tr, va = speaker_split(T.yf, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
            trials = {sg: build_trials(T.yf[vai], T.gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg) for sg in [False, True]}
            cca = dict(m=RidgeCCA(**CCA_K4).fit(T.Xf[tri], T.Xv[tri]), alF=Aligner(T.Xf[tri]), alV=Aligner(T.Xv[tri]))
            for aug in AUGS:
                ms = build(aug, tri, [seed, seed + 10, seed + 20])
                es = embed(ms, cca, T.Xf[vai], T.Xv[vai])
                for sg, (fi, vj, lab) in trials.items():
                    sc = np.mean([z(np.sum(Ef[fi] * Ev[vj], 1)) for Ef, Ev in es], 0)
                    rows.append(dict(seed=seed, aug=aug, protocol="gender" if sg else "no_gender", eer=eer_from_scores(sc, lab)))
                pd.DataFrame(rows).to_csv(OUT / "internal.csv", index=False)
                print("seed", seed, aug, "done", flush=True)
        df = pd.DataFrame(rows)
        print(df.groupby(["aug", "protocol"]).eer.mean().unstack().round(2).sort_values("gender").to_string())
    else:
        devs = {k: load_dev(*k) for k in NAMES}
        cca = dict(m=RidgeCCA(**CCA_K4).fit(T.Xf, T.Xv), alF=Aligner(T.Xf), alV=Aligner(T.Xv))
        for aug in sys.argv[2].split(","):
            ms = build(aug, np.arange(len(T.yf)), [1, 11, 21])
            with zipfile.ZipFile(OUT / f"submission_EXP006_{aug}.zip", "w") as zf:
                for k, fn in NAMES.items():
                    x, y, t = devs[k]
                    es = embed(ms, cca, x, y)
                    ii = np.arange(len(t))
                    sc = np.mean([z(np.sum(Ef[ii] * Ev[ii], 1)) for Ef, Ev in es], 0)
                    zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(t.pair_id, -sc)) + "\n")
            print("wrote", aug, flush=True)
