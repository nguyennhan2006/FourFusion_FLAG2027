"""EXP-005 — three independent levers on top of EXP-003c (31.34):
  (a) tune the InfoNCE model (hid / dropout / tau / emb / pca_face / epochs)
  (b) larger seed ensemble (3 -> 10)
  (c) partial CORAL: interpolate between mean-only centering (safe) and full CORAL (destroys identity)

Usage: python run.py tune | python run.py alpha | python run.py final
"""
import sys, zipfile
import numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EXP-003_deep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EXP-004_domain"))
import train as T
from run import Aligner, msqrt, NAMES
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
CCA_K4 = dict(k=4, reg=1.0, pca_x=128)


def z(x): return (x - x.mean()) / (x.std() + 1e-8)


def partial_align(Z, al: Aligner, alpha: float, shrink=0.1):
    """alpha=0 -> mean centering only; alpha=1 -> full CORAL. Interpolates the whitening matrix with I."""
    Zc = Z - Z.mean(0)
    if alpha <= 0:
        return Zc + al.mu
    C = np.cov(Zc.T); C = (1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C))
    M = msqrt(C, inv=True) @ al.C_half
    M = (1 - alpha) * np.eye(len(M)) + alpha * M
    return Zc @ M + al.mu


# ---------------------------------------------------------------- (a) tuning
BASE = dict(hid=512, drop=0.5, lr=1e-3, wd=1e-2, bs=256, dedup_clip=True, alpha=1.0, tau=0.07, loss="infonce", emb=128)
TUNE = {"base": {}, "hid1024": dict(hid=1024), "hid256": dict(hid=256),
        "drop0.3": dict(drop=0.3), "drop0.7": dict(drop=0.7),
        "tau0.05": dict(tau=0.05), "tau0.1": dict(tau=0.1), "tau0.15": dict(tau=0.15),
        "emb256": dict(emb=256), "emb64": dict(emb=64),
        "wd1e-1": dict(wd=1e-1), "lr3e-3": dict(lr=3e-3), "lr3e-4": dict(lr=3e-4),
        "bs128": dict(bs=128), "bs512": dict(bs=512), "nodedup": dict(dedup_clip=False)}
for n, d in TUNE.items():
    T.CFGS[f"T_{n}"] = dict(BASE, **d)


def build_models(tr_idx, cfg_name, seeds, epochs):
    out = []
    for s in seeds:
        net, prep, _, _ = T.run(T.CFGS[cfg_name], tr_idx, None, s, epochs)
        out.append((net, prep))
    return out


def embed(models, cca, alF, alV, Xf_, Xv_, alpha):
    es = []
    for net, prep in models:
        A = partial_align(prep.f(Xf_), alF, alpha).astype(np.float32)
        B = partial_align(prep.v(Xv_), alV, alpha).astype(np.float32)
        net.eval()
        with torch.no_grad():
            u, w, _ = net(torch.tensor(A), torch.tensor(B))
        es.append((u.numpy(), w.numpy()))
    if cca is not None:
        fa = partial_align(Xf_, cca["alF"], alpha); va = partial_align(Xv_, cca["alV"], alpha)
        es.append((cca["m"].transform_x(fa), cca["m"].transform_y(va)))
    return es


def score(es, fi, vj):
    return np.mean([z(np.sum(Ef[fi] * Ev[vj], 1)) for Ef, Ev in es], 0)


if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "tune":
        rows = []
        for seed in T.SEEDS:
            tr, va = speaker_split(T.yf, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
            trials = {sg: build_trials(T.yf[vai], T.gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg) for sg in [False, True]}
            alF = Aligner(T.Xf[tri]); alV = Aligner(T.Xv[tri])
            for n in TUNE:
                for ep in [10, 15, 25]:
                    models = build_models(tri, f"T_{n}", [seed], ep)
                    pf = models[0][1]
                    aF = Aligner(pf.f(T.Xf[tri])); aV = Aligner(pf.v(T.Xv[tri]))
                    es = embed(models, None, aF, aV, T.Xf[vai], T.Xv[vai], 0.0)
                    for sg, (fi, vj, lab) in trials.items():
                        rows.append(dict(seed=seed, cfg=n, ep=ep, protocol="gender" if sg else "no_gender",
                                         eer=eer_from_scores(score(es, fi, vj), lab)))
                    pd.DataFrame(rows).to_csv(OUT / "tune.csv", index=False)
                print("seed", seed, n, "done", flush=True)
        df = pd.DataFrame(rows)
        piv = df.groupby(["cfg", "ep", "protocol"]).eer.mean().unstack().round(2)
        print(piv.sort_values("gender").to_string())

    elif mode == "alpha":
        best = pd.read_csv(OUT / "tune.csv")
        b = best.groupby(["cfg", "ep"]).eer.mean().idxmin(); cfg_name, ep = f"T_{b[0]}", int(b[1])
        print("best cfg", cfg_name, "ep", ep)
        rows = []
        for seed in T.SEEDS:
            tr, va = speaker_split(T.yf, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
            trials = {sg: build_trials(T.yf[vai], T.gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg) for sg in [False, True]}
            for nseed in [3, 10]:
                models = build_models(tri, cfg_name, [seed + 100 * i for i in range(nseed)], ep)
                pf = models[0][1]
                aF, aV = Aligner(pf.f(T.Xf[tri])), Aligner(pf.v(T.Xv[tri]))
                cca = dict(m=RidgeCCA(**CCA_K4).fit(T.Xf[tri], T.Xv[tri]), alF=Aligner(T.Xf[tri]), alV=Aligner(T.Xv[tri]))
                for use_cca in [True, False]:
                    for alpha in [0.0, 0.1, 0.25, 0.5]:
                        es = embed(models, cca if use_cca else None, aF, aV, T.Xf[vai], T.Xv[vai], alpha)
                        for sg, (fi, vj, lab) in trials.items():
                            rows.append(dict(seed=seed, nseed=nseed, cca=use_cca, alpha=alpha,
                                             protocol="gender" if sg else "no_gender",
                                             eer=eer_from_scores(score(es, fi, vj), lab)))
                pd.DataFrame(rows).to_csv(OUT / "alpha.csv", index=False)
                print("seed", seed, "nseed", nseed, "done", flush=True)
        df = pd.DataFrame(rows)
        print(df.groupby(["nseed", "cca", "alpha", "protocol"]).eer.mean().unstack().round(2).sort_values("gender").to_string())

    elif mode == "final":
        best = pd.read_csv(OUT / "tune.csv")
        b = best.groupby(["cfg", "ep"]).eer.mean().idxmin(); cfg_name, ep = f"T_{b[0]}", int(b[1])
        a = pd.read_csv(OUT / "alpha.csv")
        ab = a.groupby(["nseed", "cca", "alpha"]).eer.mean().idxmin()
        nseed, use_cca, alpha = int(ab[0]), bool(ab[1]), float(ab[2])
        print("final:", cfg_name, "ep", ep, "nseed", nseed, "cca", use_cca, "alpha", alpha)
        models = build_models(np.arange(len(T.yf)), cfg_name, [1 + 100 * i for i in range(nseed)], ep)
        pf = models[0][1]
        aF, aV = Aligner(pf.f(T.Xf)), Aligner(pf.v(T.Xv))
        cca = dict(m=RidgeCCA(**CCA_K4).fit(T.Xf, T.Xv), alF=Aligner(T.Xf), alV=Aligner(T.Xv))
        devs = {k: load_dev(*k) for k in NAMES}
        for tag, uc, al in [(f"best_a{alpha}", use_cca, alpha), ("best_a0", use_cca, 0.0), ("best_a0_nocca", False, 0.0), ("best_a0.25", use_cca, 0.25)]:
            with zipfile.ZipFile(OUT / f"submission_EXP005_{tag}_n{nseed}_{cfg_name}_ep{ep}.zip", "w") as zf:
                for k, fn in NAMES.items():
                    x, y, t = devs[k]
                    es = embed(models, cca if uc else None, aF, aV, x, y, al)
                    ii = np.arange(len(t))
                    zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(t.pair_id, -score(es, ii, ii))) + "\n")
            print("wrote", tag, flush=True)
