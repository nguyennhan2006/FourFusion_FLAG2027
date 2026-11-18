"""XJF-01 pre-registered test (README section 6): does adding the XJF branch help on held-out TRAIN persons?
    python Experiment/XJF-01/xjf_holdout.py      -> out/holdout_report.txt, out/holdout_scores.parquet
Trials = the v4 g / v4 ng cells of SAFE-04 (out/dumpv4/: fold, y, fi / vi = rows of train_English.txt, U_c0r = F1s' + RFX t3/nf/nv oof).
Per outer fold k: XJF (same recipe as xjf.py, 10 seeds) learns on FLAG train + MAV-Celeb persons NOT in the held-out list of fold k;
z constants from inner out-of-fold trials (4 inner folds by person of the training part, 3 seeds, pairs drawn as in xjf.py);
U is divided by the sd of U_c0r on the v4 rows outside fold k (per cell). Fused = U / sd + z (r = 1). Labels: train only."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], "__no_eval__"]                     # xjf.py reads an Evaluation zip path from argv; none is needed here
SRC = (HERE / "xjf.py").read_text(encoding="utf-8")
CUT = SRC.index("ez = zipfile.ZipFile(EVZ)")               # run only the data loading + model definitions of xjf.py
CUT2 = SRC.index("# ------------------------------------------------------------------------------------------------ out-of-fold on FLAG train")
G = {"__file__": str(HERE / "xjf.py"), "__name__": "xjf_lib"}
exec(compile(SRC[:CUT] + "EVAL = {}\n" + SRC[SRC.index("T = lambda x, mu"):CUT2], "xjf.py", "exec"), G)
log, train, cos, eer = G["log"], G["train"], G["cos"], G["eer"]
t_pid, ef_by, ev_ok, ev_pid, gmap, TFt, TVt = G["t_pid"], G["ef_by"], G["ev_ok"], G["ev_pid"], G["gmap"], G["TFt"], G["TVt"]
SEEDS, OOF_SEEDS = G["SEEDS"], G["OOF_SEEDS"]

D = HERE / "out" / "dumpv4" / "sp" / "out" / "safe04"
V = pd.read_parquet(D / "xjfdump_v4.parquet"); HOLD = json.loads((D / "xjfdump_persons.json").read_text())
# row check: fi / vi index train_English.txt rows -> the person of that row must be the trial's face / voice person
assert (t_pid[V.fi.to_numpy()] == V.fp.to_numpy(str)).all() and (t_pid[V.vi.to_numpy()] == V.vp.to_numpy(str)).all(), "fi / vi do not index train_English.txt"
log(f"trials {len(V)}; fi / vi line up with train_English.txt (person match 100 %)")


def mav_key(p):                                              # "v1_complete:ali_zafar" -> "v1_ali_zafar" (the ef_pid / ev_pid form of xjf.py)
    return p.split("_complete:")[0] + "_" + p.split(":", 1)[1] if "_complete:" in p else None


def fit(flag_rows, drop_mav, seeds):
    keep = G["ev_ok"].copy(); G["ev_ok"][:] = keep & ~np.isin(ev_pid, list(drop_mav))   # train() samples MAV voices where ev_ok
    try:
        return [train(s, flag_rows) for s in seeds]
    finally:
        G["ev_ok"][:] = keep


def draw(te_p, rr, prot, n=6000):                            # the OOF trial generator of xjf.py
    by = {p: np.nonzero(t_pid == p)[0] for p in te_p}; fi, vi, y = [], [], []
    for _ in range(n // 2):
        p = rr.choice(te_p); a, b = rr.choice(by[p], 2, replace=False); fi.append(a); vi.append(b); y.append(1)
    while len(y) < n:
        p, q = rr.choice(te_p, 2, replace=False)
        if prot == "gender" and gmap[p] != gmap[q]: continue
        fi.append(rr.choice(by[p])); vi.append(rr.choice(by[q])); y.append(0)
    return np.array(fi), np.array(vi), np.array(y)


CELL = {"v4 g": "gender", "v4 ng": "no_gender"}
SUB = json.loads((HERE / "out" / "consts.json").read_text()); SIG_SUB = {"v4 ng": 5.0983, "v4 g": 3.8376}
V["x_cos"] = np.nan; V["x_z"] = np.nan; V["x_zsub"] = np.nan; V["U_sd"] = np.nan
t0 = time.time()
for k in sorted(V.fold.unique()):
    hold = set(HOLD[str(k)]); flag_tr = sorted(set(t_pid) - hold); te = (V.fold == k).to_numpy()
    drop_mav = {m for m in map(mav_key, hold) if m}; n_mav = len(drop_mav & set(ef_by))
    assert set(V.loc[te, "fp"]) | set(V.loc[te, "vp"]) <= hold, "trial persons must be held out"
    rows = np.nonzero(np.isin(t_pid, flag_tr))[0]
    # inner OOF for the z constants (training part only)
    pr = np.array(flag_tr); pg = np.array([gmap[p] for p in pr]); fo = np.zeros(len(pr), int)
    for g in sorted(set(pg)):
        ix = np.nonzero(pg == g)[0]; fo[np.random.default_rng(11 + k).permutation(ix)] = np.arange(len(ix)) % 4
    acc = {"no_gender": [], "gender": []}
    for j in range(4):
        ip = pr[fo == j]; ms = fit(np.nonzero(np.isin(t_pid, pr[fo != j]))[0], drop_mav, OOF_SEEDS); rr = np.random.default_rng(1000 * k + j)
        for prot in acc:
            fi, vi, _ = draw(ip, rr, prot); acc[prot].append(cos(ms, TFt[fi], TVt[vi]))
    const = {p: (float(np.concatenate(a).mean()), float(np.concatenate(a).std())) for p, a in acc.items()}
    ms = fit(rows, drop_mav, SEEDS)
    c = cos(ms, TFt[V.fi.to_numpy()[te]], TVt[V.vi.to_numpy()[te]]); V.loc[te, "x_cos"] = c
    for cell, prot in CELL.items():
        m = te & (V.cell == cell).to_numpy(); mo = ~te & (V.cell == cell).to_numpy()
        cc = V.loc[m, "x_cos"].to_numpy()
        V.loc[m, "x_z"] = (cc - const[prot][0]) / const[prot][1]
        V.loc[m, "x_zsub"] = (cc - SUB[prot][0]) / SUB[prot][1]
        V.loc[m, "U_sd"] = V.loc[mo, "U_c0r"].std()
    log(f"fold {k}: {len(flag_tr)} FLAG persons learn, {n_mav} MAV persons dropped, z const {const} [{time.time() - t0:.0f}s]")

rep = ["U_c0r", "U_c0s", "U_c0t"]
for u in rep:
    V[f"FUSE_{u}"] = V[u] / V["U_sd"] + V["x_z"]
V["FUSE_sub"] = V["U_c0r"] / V.cell.map(SIG_SUB) + V["x_zsub"]
cols = ["U_c0r", "x_cos", "FUSE_U_c0r", "FUSE_sub"] + [c for u in rep[1:] for c in (u, f"FUSE_{u}")]
R = pd.DataFrame([{"cell": c, "fold": f, **{s: eer(g[s].to_numpy(float), g.y.to_numpy()) for s in cols}}
                  for (c, f), g in V.groupby(["cell", "fold"])])
log("EER per fold:\n" + R.round(2).to_string(index=False))
log("EER mean of folds:\n" + R.groupby("cell")[cols].mean().round(2).T.to_string())
for cell in CELL:
    r = R[R.cell == cell]
    for base, fu in [("U_c0r", "FUSE_U_c0r"), ("U_c0r", "FUSE_sub"), ("U_c0s", "FUSE_U_c0s"), ("U_c0t", "FUSE_U_c0t")]:
        d = (r[base] - r[fu]).to_numpy()
        log(f"{cell}: gain {fu} vs {base} = EER(base) - EER(fused): mean {d.mean():+.2f}, folds {np.round(d, 2).tolist()}, wins {int((d > 0).sum())}/5")
V.to_parquet(HERE / "out" / "holdout_scores.parquet")
(HERE / "out" / "holdout_report.txt").write_text("\n".join(G["LOG"]) + "\n", encoding="utf-8"); log("saved out/holdout_report.txt")
