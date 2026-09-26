"""EXP-003b — deep (InfoNCE) + E (file centering, AS-norm) + ensemble with CCA. Internal eval + dev submissions.
Usage: python combine.py internal | python combine.py final"""
import sys, os, zipfile
import numpy as np, pandas as pd, torch
from pathlib import Path
import train as T
from eda_utils import *
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EXP-002_cca_centering"))
from run import asnorm, center

OUT = T.OUT
DEEP = {"C": "C_infonce_e128", "B": "B_infonce_sg_e128"}
CCA = {"k4": dict(k=4, reg=1.0, pca_x=128), "k32": dict(k=32, reg=10.0, pca_x=256)}
sw = pd.concat([pd.read_csv(f) for f in OUT.glob("sweep_*.csv")])
EP = {n: int(round(sw[sw.cfg == c].ep.mean())) for n, c in DEEP.items()}


def deep_embed(net, prep, Xf_, Xv_):
    net.eval()
    with torch.no_grad():
        u, w, _ = net(torch.tensor(prep.f(Xf_)), torch.tensor(prep.v(Xv_)))
    return u.numpy(), w.numpy()


def scores_for(Ef, Ev, fi, vj, snorm):
    cos = np.sum(Ef[fi] * Ev[vj], 1)
    return cos if snorm == "raw" else asnorm(Ef, Ev, fi, vj)


def z(x): return (x - x.mean()) / (x.std() + 1e-8)


MODE = sys.argv[1] if len(sys.argv) > 1 else None
if MODE == "internal":
    rows = []
    for seed in T.SEEDS:
        tr, va = speaker_split(T.yf, seed); tri, vai = np.where(tr)[0], np.where(va)[0]
        trials = {sg: build_trials(T.yf[vai], T.gmap, seed=seed, n_pos=3000, n_neg=3000, same_gender=sg) for sg in [False, True]}
        mf, mv = T.Xf[tri].mean(0), T.Xv[tri].mean(0)
        feats = {"none": (T.Xf[vai], T.Xv[vai]), "file": (center(T.Xf[vai], "file", mf), center(T.Xv[vai], "file", mv))}
        embs = {}
        for n, cfg in CCA.items():
            m = RidgeCCA(**cfg).fit(T.Xf[tri], T.Xv[tri])
            for cm, (a, b) in feats.items(): embs[(n, cm)] = (m.transform_x(a), m.transform_y(b))
        for n, cname in DEEP.items():
            for s2 in [seed, seed + 10, seed + 20]:   # 3 training seeds per split -> ensemble
                net, prep, _, _ = T.run(T.CFGS[cname], tri, None, s2, EP[n])
                for cm, (a, b) in feats.items(): embs[(f"{n}{s2 - seed}", cm)] = deep_embed(net, prep, a, b)
        for sg, (fi, vj, lab) in trials.items():
            prot = "gender" if sg else "no_gender"
            for cm in ["none", "file"]:
                for sn in ["raw", "asnorm"]:
                    S = {k[0]: scores_for(*embs[k], fi, vj, sn) for k in embs if k[1] == cm}
                    combos = {"k4": ["k4"], "k32": ["k32"], "C": ["C0"], "B": ["B0"],
                              "C_ens3": ["C0", "C10", "C20"], "B_ens3": ["B0", "B10", "B20"],
                              "C_ens3+k4": ["C0", "C10", "C20", "k4"], "C_ens3+k32": ["C0", "C10", "C20", "k32"],
                              "B_ens3+k32": ["B0", "B10", "B20", "k32"], "C_ens3+B_ens3": ["C0", "C10", "C20", "B0", "B10", "B20"],
                              "C_ens3+B_ens3+k32": ["C0", "C10", "C20", "B0", "B10", "B20", "k32"]}
                    for name, parts in combos.items():
                        sc = np.mean([z(S[p]) for p in parts], 0)
                        rows.append(dict(seed=seed, protocol=prot, center=cm, snorm=sn, model=name, eer=eer_from_scores(sc, lab)))
        print("seed", seed, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "combine_internal.csv", index=False)
    piv = df.groupby(["model", "center", "snorm", "protocol"]).eer.mean().unstack("protocol").round(2)
    print(piv.sort_values("gender").to_string())

elif MODE == "final":
    names = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt", ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
             ("gender", "English"): "gender/sub_score_v4_English_heard.txt", ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
    devs = {k: load_dev(*k) for k in names}
    mf, mv = T.Xf.mean(0), T.Xv.mean(0)
    feats = {k: (center(a, "file", mf), center(b, "file", mv)) for k, (a, b, _) in devs.items()}
    embs = {}
    for n, cfg in CCA.items():
        m = RidgeCCA(**cfg).fit(T.Xf, T.Xv)
        for k, (a, b) in feats.items(): embs[(n, k)] = (m.transform_x(a), m.transform_y(b))
    for n, cname in DEEP.items():
        for s2 in [1, 11, 21]:
            net, prep, _, _ = T.run(T.CFGS[cname], np.arange(len(T.yf)), None, s2, EP[n])
            for k, (a, b) in feats.items(): embs[(f"{n}{s2 - 1}", k)] = deep_embed(net, prep, a, b)
            print("trained", cname, s2, flush=True)
    recipes = {"C_ens3": ["C0", "C10", "C20"], "B_ens3": ["B0", "B10", "B20"],
               "C_ens3+k4": ["C0", "C10", "C20", "k4"], "B_ens3+k32": ["B0", "B10", "B20", "k32"],
               "C_ens3+B_ens3": ["C0", "C10", "C20", "B0", "B10", "B20"], "C_ens3+B_ens3+k32": ["C0", "C10", "C20", "B0", "B10", "B20", "k32"]}
    for rname, parts in recipes.items():
        for sn in ["raw", "asnorm"]:
            with zipfile.ZipFile(OUT / f"submission_EXP003_{rname}_file_{sn}.zip", "w") as zf:
                for k, fn in names.items():
                    ii = np.arange(len(devs[k][2]))
                    sc = np.mean([z(scores_for(*embs[(p, k)], ii, ii, sn)) for p in parts], 0)
                    zf.writestr(fn, "\n".join(f"{p} {s:.6f}" for p, s in zip(devs[k][2].pair_id, -sc)) + "\n")   # lower = same
            print("wrote", rname, sn)
