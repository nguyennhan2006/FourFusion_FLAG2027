"""EXP-007 — why does a LARGER deep ensemble score worse, and what is the right fusion weight?

Kaggle EXP-006 sweep (3 splits, equal-weight z-score fusion of n deep models + 1 CCA):
    n=3 -> int_g 30.48 | n=5 -> 30.70 | n=10 -> 31.12 | n=1 -> 31.14
Monotone in n for n>=3, which is backwards for an ensemble. Hypothesis: the fusion averages
n+1 equally weighted members, so the single CCA member's weight falls from 1/4 to 1/11 —
we are not seeing ensembling hurt, we are seeing the CCA being diluted.

Test: fix n=10 and give the deep ensemble and the CCA an explicit weight w:
    score = (1 - w) * mean_i z(cos_deep_i) + w * z(cos_cca)
If the hypothesis holds, w ~ 0.25 at n=10 should recover (or beat) the n=3 number.

Usage: python run.py internal | python run.py final
"""
import sys, zipfile
import numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EXP-003_deep"))
import train as T
from eda_utils import *

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)

# Winner of the Kaggle architecture sweep (int_g 31.14 at 1 seed/split).
BEST = dict(hid=512, emb=128, drop=0.3, lr=1e-3, wd=1e-2, bs=256, tau=0.07,
            loss="infonce", alpha=1.0, dedup_clip=False)
EPOCHS = 40
CCA_CFG = dict(k=4, reg=1.0, pca_x=128)
NAMES = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}


def z(x): return (x - x.mean()) / (x.std() + 1e-8)


def center(X, mu): return X - X.mean(0) + mu


def deep_scores(nets, preps, mu_f, mu_v, Xf_, Xv_, fi, vj):
    out = []
    for net, prep in zip(nets, preps):
        net.eval()
        with torch.no_grad():
            u, w, _ = net(torch.tensor(center(prep.f(Xf_), mu_f)), torch.tensor(center(prep.v(Xv_), mu_v)))
        u, w = u.numpy(), w.numpy()
        out.append(z(np.sum(u[fi] * w[vj], 1)))
    return out


def combine(deep_s, cca_s, n, w):
    """w = weight given to the CCA member; w=None reproduces equal weighting over n+1 members."""
    d = np.mean(deep_s[:n], 0)
    if w is None:
        return (np.sum(deep_s[:n], 0) + cca_s) / (n + 1)
    return (1 - w) * d + w * cca_s


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "internal":
        rows = []
        for seed in T.SEEDS:
            tr, va = speaker_split(T.yf, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
            prep0 = T.Prep(T.Xf[tri], T.Xv[tri], cache=T.OUT / f"prep_n{len(tri)}_s{int(tri.sum())}.npz")
            mu_f, mu_v = prep0.f(T.Xf[tri]).mean(0), prep0.v(T.Xv[tri]).mean(0)
            nets, preps = [], []
            for i in range(10):
                net, prep, _, _ = T.run(dict(BEST), tri, None, seed + 100 * i, EPOCHS)
                nets.append(net); preps.append(prep)
            cca = RidgeCCA(**CCA_CFG).fit(T.Xf[tri], T.Xv[tri])
            Ef, Ev = cca.transform_x(center(T.Xf[vai], T.Xf[tri].mean(0))), cca.transform_y(center(T.Xv[vai], T.Xv[tri].mean(0)))
            for sg in [False, True]:
                fi, vj, lab = build_trials(T.yf[vai], T.gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg)
                ds = deep_scores(nets, preps, mu_f, mu_v, T.Xf[vai], T.Xv[vai], fi, vj)
                cs = z(np.sum(Ef[fi] * Ev[vj], 1))
                prot = "gender" if sg else "no_gender"
                for n in [1, 3, 5, 10]:
                    rows.append(dict(seed=seed, n=n, w="equal", protocol=prot,
                                     eer=eer_from_scores(combine(ds, cs, n, None), lab)))
                for w in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
                    rows.append(dict(seed=seed, n=10, w=w, protocol=prot,
                                     eer=eer_from_scores(combine(ds, cs, 10, w), lab)))
                    rows.append(dict(seed=seed, n=3, w=w, protocol=prot,
                                     eer=eer_from_scores(combine(ds, cs, 3, w), lab)))
            pd.DataFrame(rows).to_csv(OUT / "internal.csv", index=False)
            print("seed", seed, "done", flush=True)
        df = pd.DataFrame(rows)
        print("\n-- equal weighting (reproduces the Kaggle ensemble.csv trend)")
        print(df[df.w == "equal"].groupby(["n", "protocol"]).eer.mean().unstack().round(2).to_string())
        print("\n-- explicit CCA weight w")
        print(df[df.w != "equal"].assign(w=lambda d: d.w.astype(float))
                .groupby(["n", "w", "protocol"]).eer.mean().unstack().round(2).to_string())
    else:
        df = pd.read_csv(OUT / "internal.csv")
        g = df[(df.w != "equal") & (df.protocol == "gender")].assign(w=lambda d: d.w.astype(float))
        best = g.groupby(["n", "w"]).eer.mean().idxmin()
        n_best, w_best = int(best[0]), float(best[1])
        print("best fusion:", "n", n_best, "w", w_best, "int_g", round(g.groupby(["n", "w"]).eer.mean().min(), 2))
        allidx = np.arange(len(T.yf))
        prep0 = T.Prep(T.Xf, T.Xv, cache=T.OUT / f"prep_n{len(allidx)}_s{int(allidx.sum())}.npz")
        mu_f, mu_v = prep0.f(T.Xf).mean(0), prep0.v(T.Xv).mean(0)
        nets, preps = [], []
        for i in range(max(n_best, 3)):
            net, prep, _, _ = T.run(dict(BEST), allidx, None, 1 + 100 * i, EPOCHS)
            nets.append(net); preps.append(prep); print("trained", i, flush=True)
        cca = RidgeCCA(**CCA_CFG).fit(T.Xf, T.Xv)
        devs = {k: load_dev(*k) for k in NAMES}
        for tag, n, w in [(f"n{n_best}_w{w_best}", n_best, w_best), ("n3_equal", 3, None)]:
            with zipfile.ZipFile(OUT / f"submission_EXP007_{tag}.zip", "w") as zf:
                for k, fn in NAMES.items():
                    a, b, t = devs[k]; ii = np.arange(len(t))
                    ds = deep_scores(nets[:n], preps[:n], mu_f, mu_v, a, b, ii, ii)
                    Ef, Ev = cca.transform_x(center(a, T.Xf.mean(0))), cca.transform_y(center(b, T.Xv.mean(0)))
                    sc = combine(ds, z(np.sum(Ef * Ev, 1)), n, w)
                    zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(t.pair_id, -sc)) + "\n")
            print("wrote", tag, flush=True)
