"""EXP-004 — domain alignment of dev files onto the train distribution, applied in the model's input space.

Variants (all unsupervised, per dev file):
  none      : nothing
  mean      : subtract the file mean, add the train mean            (= EXP-003c "file centering")
  coral     : mean + covariance whitening onto the train covariance (CORAL, shrinkage-regularised)
  std       : mean + per-dimension std matching (diagonal CORAL)
Each variant can be applied to both modalities or to voice only (`_v` suffix), since EDA-5 put the
language shift almost entirely in voice.

Scores are raw cosine (EXP-003c showed AS-norm does not transfer to the dev cohort).
Usage: python run.py internal | python run.py final
"""
import sys, zipfile
import numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EXP-003_deep"))
import train as T
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
DEEP_CFG, DEEP_EP, DEEP_SEEDS = "C_infonce_e128", 15, [1, 11, 21]
CCA_K4 = dict(k=4, reg=1.0, pca_x=128)
NAMES = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}


def msqrt(C, inv=False, eps=1e-3):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps * w.max())
    w = 1 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * w) @ V.T


class Aligner:
    """Fit on the train feature matrix (already in model input space); apply to one dev file."""
    def __init__(self, Ztr, shrink=0.1):
        self.mu = Ztr.mean(0)
        C = np.cov(Ztr.T)
        self.C_half = msqrt((1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C)))
        self.sd = Ztr.std(0) + 1e-6

    def __call__(self, Z, mode, shrink=0.1):
        if mode == "none":
            return Z
        Zc = Z - Z.mean(0)
        if mode == "mean":
            return Zc + self.mu
        if mode == "std":
            return Zc / (Z.std(0) + 1e-6) * self.sd + self.mu
        if mode == "coral":
            C = np.cov(Zc.T)
            C = (1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C))
            return Zc @ msqrt(C, inv=True) @ self.C_half + self.mu
        raise ValueError(mode)


def z(x): return (x - x.mean()) / (x.std() + 1e-8)


MODES = ["none", "mean", "std", "coral"]
VARIANTS = [(m, both) for m in MODES for both in ([True, False] if m != "none" else [True])]


def embed_all(nets, preps, cca, Xf_, Xv_, alignF, alignV, mode, voice_only):
    """Returns list of (Ef, Ev) for each deep seed + the CCA model, all aligned the same way."""
    out = []
    for net, prep in zip(nets, preps):
        A = alignF(prep.f(Xf_), "none" if voice_only else mode)
        B = alignV(prep.v(Xv_), mode)
        net.eval()
        with torch.no_grad():
            u, w, _ = net(torch.tensor(A.astype(np.float32)), torch.tensor(B.astype(np.float32)))
        out.append((u.numpy(), w.numpy()))
    # CCA consumes raw features, so align in raw space with the same recipe
    fa = cca["alignF"](Xf_, "none" if voice_only else mode)
    va = cca["alignV"](Xv_, mode)
    out.append((cca["m"].transform_x(fa), cca["m"].transform_y(va)))
    return out


def build(tr_idx, seeds):
    nets, preps = [], []
    for s in seeds:
        net, prep, _, _ = T.run(T.CFGS[DEEP_CFG], tr_idx, None, s, DEEP_EP)
        nets.append(net); preps.append(prep)
    cca = dict(m=RidgeCCA(**CCA_K4).fit(T.Xf[tr_idx], T.Xv[tr_idx]),
               alignF=Aligner(T.Xf[tr_idx]), alignV=Aligner(T.Xv[tr_idx]))
    alignF = Aligner(np.vstack([p.f(T.Xf[tr_idx]) for p in preps[:1]]))
    alignV = Aligner(np.vstack([p.v(T.Xv[tr_idx]) for p in preps[:1]]))
    return nets, preps, cca, alignF, alignV


if __name__ == "__main__":
    mode_arg = sys.argv[1]
    if mode_arg == "internal":
        rows = []
        for seed in T.SEEDS:
            tr, va = speaker_split(T.yf, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
            nets, preps, cca, alignF, alignV = build(tri, [seed, seed + 10, seed + 20])
            trials = {sg: build_trials(T.yf[vai], T.gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg) for sg in [False, True]}
            for m, both in VARIANTS:
                embs = embed_all(nets, preps, cca, T.Xf[vai], T.Xv[vai], alignF, alignV, m, not both)
                for sg, (fi, vj, lab) in trials.items():
                    sc = np.mean([z(np.sum(Ef[fi] * Ev[vj], 1)) for Ef, Ev in embs], 0)
                    rows.append(dict(seed=seed, align=m + ("" if both else "_v"),
                                     protocol="gender" if sg else "no_gender", eer=eer_from_scores(sc, lab)))
            print("seed", seed, "done", flush=True)
        df = pd.DataFrame(rows); df.to_csv(OUT / "exp004_internal.csv", index=False)
        print(df.groupby(["align", "protocol"]).eer.mean().unstack().round(2).sort_values("gender").to_string())
    else:
        nets, preps, cca, alignF, alignV = build(np.arange(len(T.yf)), DEEP_SEEDS)
        devs = {k: load_dev(*k) for k in NAMES}
        for m, both in VARIANTS:
            tag = m + ("" if both else "_v")
            with zipfile.ZipFile(OUT / f"submission_EXP004_align-{tag}.zip", "w") as zf:
                for k, fn in NAMES.items():
                    a, b, t = devs[k]
                    embs = embed_all(nets, preps, cca, a, b, alignF, alignV, m, not both)
                    sc = np.mean([z(np.sum(Ef * Ev, 1)) for Ef, Ev in embs], 0)
                    zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(t.pair_id, -sc)) + "\n")
            print("wrote", tag, flush=True)
