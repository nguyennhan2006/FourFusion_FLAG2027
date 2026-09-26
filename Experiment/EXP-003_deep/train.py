"""EXP-003/004/005 — deep cross-modal projection on top of standardised + PCA features.
Pipelines:  A = ce_opl (FOP-like: CE on fused + alpha*OPL)     C = infonce (symmetric, SupCon positives)
            B = infonce_sg (InfoNCE with only same-gender negatives)  AB = ce_opl_sg (OPL negatives same gender only)
Model selection = int_g (gender-constrained internal EER), 3 speaker-disjoint seeds. CPU-friendly.
Usage: python train.py sweep | python train.py final <cfg_name>
"""
import sys, json, time, zipfile, itertools
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EDA-000_raw"))
from eda_utils import *

import os
torch.set_num_threads(int(os.environ.get('NT', '4')))
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
Xf, Xv, yf, _, txt, gmap = load_train()
gender = np.array([gmap[s] for s in yf]); g_int = (gender == "m").astype(np.int64)
vgrp = np.load(Path(__file__).resolve().parents[1] / "EDA-000_raw/out/voice_group.npy")
SEEDS = [1, 2, 3]
PCA_FACE = 256


class Prep:
    def __init__(self, Xf_tr, Xv_tr, cache=None):
        self.mf, self.sf = Xf_tr.mean(0), Xf_tr.std(0) + 1e-6
        self.mv, self.sv = Xv_tr.mean(0), Xv_tr.std(0) + 1e-6
        if cache is not None and cache.exists():
            self.P = np.load(cache)["P"]; return
        Z = (Xf_tr - self.mf) / self.sf
        _, _, Vt = np.linalg.svd(Z, full_matrices=False); self.P = Vt[:PCA_FACE].T
        if cache is not None: np.savez(cache, P=self.P)
    def f(self, X): return (((X - self.mf) / self.sf) @ self.P).astype(np.float32)
    def v(self, X): return ((X - self.mv) / self.sv).astype(np.float32)


class Net(nn.Module):
    def __init__(self, d_in_f, d_in_v, emb, hid, drop, n_cls):
        super().__init__()
        def branch(d):
            return nn.Sequential(nn.Dropout(drop), nn.Linear(d, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, emb))
        self.bf, self.bv = branch(d_in_f), branch(d_in_v)
        self.cls = nn.Linear(2 * emb, n_cls)
    def forward(self, f, v):
        u, w = F.normalize(self.bf(f), dim=1), F.normalize(self.bv(v), dim=1)
        return u, w, self.cls(torch.cat([u, w], 1))


def opl(u, w, y, g, same_gender_only, neg_w=0.7):
    """Orthogonal projection loss (FOP): pull same-id pairs, push different-id pairs to orthogonal."""
    S = u @ w.T
    same = (y[:, None] == y[None, :]).float()
    neg = 1 - same
    if same_gender_only:
        neg = neg * (g[:, None] == g[None, :]).float()
    pos = (1 - S) * same
    return pos.sum() / same.sum().clamp(min=1) + neg_w * (S.abs() * neg).sum() / neg.sum().clamp(min=1)


def infonce(u, w, y, g, tau, same_gender_only):
    S = u @ w.T / tau
    same = (y[:, None] == y[None, :]).float()
    if same_gender_only:                       # drop cross-gender negatives from the denominator
        keep = ((g[:, None] == g[None, :]) | (same > 0)).float()
        S = S.masked_fill(keep == 0, -1e4)
    logp_r = F.log_softmax(S, 1); logp_c = F.log_softmax(S, 0)
    l_r = -(logp_r * same).sum(1) / same.sum(1); l_c = -(logp_c * same).sum(0) / same.sum(0)
    return 0.5 * (l_r.mean() + l_c.mean())


def run(cfg, tr_idx, va_idx, seed, epochs, eval_every=1, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    tag = f"n{len(tr_idx)}_s{int(np.asarray(tr_idx).sum())}"   # cache key = exact training row set
    prep = Prep(Xf[tr_idx], Xv[tr_idx], cache=OUT / f"prep_{tag}.npz")
    Ftr, Vtr = torch.tensor(prep.f(Xf[tr_idx])), torch.tensor(prep.v(Xv[tr_idx]))
    spk_tr = yf[tr_idx]; cls = {s: i for i, s in enumerate(np.unique(spk_tr))}
    Ytr = torch.tensor([cls[s] for s in spk_tr]); Gtr = torch.tensor(g_int[tr_idx]); Ctr = vgrp[tr_idx]
    net = Net(PCA_FACE, 192, cfg["emb"], cfg["hid"], cfg["drop"], len(cls))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    if va_idx is not None:
        Fva, Vva = torch.tensor(prep.f(Xf[va_idx])), torch.tensor(prep.v(Xv[va_idx]))
        trials = {sg: build_trials(yf[va_idx], gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg) for sg in [False, True]}
    hist, best = [], dict(int_g=1e9)
    n = len(tr_idx); bs = cfg["bs"]
    rng = np.random.RandomState(seed)
    for ep in range(1, epochs + 1):
        net.train()
        if cfg["dedup_clip"]:   # one row per voice clip per epoch -> no identical voices in a batch
            perm = rng.permutation(n); _, first = np.unique(Ctr[perm], return_index=True); order = perm[np.sort(first)]
        else:
            order = rng.permutation(n)
        for s in range(0, len(order) - bs // 2, bs):
            b = torch.tensor(order[s:s + bs])
            u, w, logits = net(Ftr[b], Vtr[b]); y, g = Ytr[b], Gtr[b]
            if cfg["loss"].startswith("ce_opl"):
                loss = F.cross_entropy(logits, y) + cfg["alpha"] * opl(u, w, y, g, cfg["loss"].endswith("_sg"))
            else:
                loss = infonce(u, w, y, g, cfg["tau"], cfg["loss"].endswith("_sg"))
                if cfg.get("ce_w", 0) > 0: loss = loss + cfg["ce_w"] * F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if va_idx is not None and ep % eval_every == 0:
            net.eval()
            with torch.no_grad():
                u, w, _ = net(Fva, Vva); u, w = u.numpy(), w.numpy()
            r = {("gender" if sg else "no_gender"): eval_trials(u, w, *t)["eer"] for sg, t in trials.items()}
            hist.append(dict(ep=ep, loss=float(loss.detach()), int_ng=r["no_gender"], int_g=r["gender"]))
            if r["gender"] < best["int_g"]: best = dict(ep=ep, int_ng=r["no_gender"], int_g=r["gender"])
            if verbose: print(hist[-1])
    return net, prep, hist, best


CFGS = {}
base = dict(hid=512, drop=0.5, lr=1e-3, wd=1e-2, bs=256, dedup_clip=True, alpha=1.0, tau=0.07)
for emb in [32, 64, 128]:
    CFGS[f"A_ce_opl_e{emb}"] = dict(base, loss="ce_opl", emb=emb)
    CFGS[f"AB_ce_opl_sg_e{emb}"] = dict(base, loss="ce_opl_sg", emb=emb)
    CFGS[f"C_infonce_e{emb}"] = dict(base, loss="infonce", emb=emb)
    CFGS[f"B_infonce_sg_e{emb}"] = dict(base, loss="infonce_sg", emb=emb)
CFGS["C_infonce_e64_ce"] = dict(base, loss="infonce", emb=64, ce_w=0.5)
CFGS["A_ce_opl_e64_nodedup"] = dict(base, loss="ce_opl", emb=64, dedup_clip=False)
CFGS["C_infonce_e64_tau0.2"] = dict(base, loss="infonce", emb=64, tau=0.2)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    EPOCHS = int(os.environ.get('EPOCHS', '40'))
    if mode == "sweep":
        only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CFGS)
        rows = []
        for cname in only:
            cfg = CFGS[cname]
            for seed in SEEDS:
                tr, va = speaker_split(yf, seed)
                t0 = time.time()
                _, _, hist, best = run(cfg, np.where(tr)[0], np.where(va)[0], seed, EPOCHS)
                rows.append(dict(cfg=cname, seed=seed, **best, last_int_g=hist[-1]["int_g"], last_int_ng=hist[-1]["int_ng"], sec=round(time.time() - t0)))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(OUT / f"sweep_{os.environ.get('JOB', '0')}.csv", index=False)
        df = pd.DataFrame(rows)
        print(df.groupby("cfg")[["int_ng", "int_g", "ep"]].mean().round(2).sort_values("int_g").to_string())
    elif mode == "final":
        cname = sys.argv[2]; cfg = CFGS[cname]
        sw = pd.concat([pd.read_csv(f) for f in OUT.glob("sweep_*.csv")]); ep = int(round(sw[sw.cfg == cname].ep.mean()))
        names = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt", ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
                 ("gender", "English"): "gender/sub_score_v4_English_heard.txt", ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
        devs = {k: load_dev(*k) for k in names}
        scores = {k: [] for k in names}
        for seed in SEEDS:
            net, prep, _, _ = run(cfg, np.arange(len(yf)), None, seed, ep)
            net.eval()
            for k, (a, b, t) in devs.items():
                with torch.no_grad():
                    u, w, _ = net(torch.tensor(prep.f(a)), torch.tensor(prep.v(b)))
                scores[k].append(2 - 2 * (u * w).sum(1).numpy())
        zname = OUT / f"submission_{cname}_ep{ep}_3seed.zip"
        with zipfile.ZipFile(zname, "w") as zf:
            for k, fn in names.items():
                d = np.mean(scores[k], 0)
                zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(devs[k][2].pair_id, d)) + "\n")
        print("wrote", zname)
