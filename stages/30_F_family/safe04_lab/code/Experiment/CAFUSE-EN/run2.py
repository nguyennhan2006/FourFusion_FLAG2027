r"""CAFUSE-EN2: rank the English recipes on CROSS-VIDEO English positives (en_ext cells: MAV-Celeb v1/v2 persons speaking
English, positive face from another video), the closest validation to dev English (dev positives are not same-session;
v4 cells may be). AudioCLIP is left out (it covers 6 % of en_ext). Registered before running:
  SET17_like   mean over s007 / s010B analogs of prod0 (SET-17 level 0 without AudioCLIP)
  CA_EN_noacl  logistic regression on [s007 deep/cca, s010B deep/cca, ib, age] + 6 conditions + products
  CA_B3        CA-FUSE on the B3 streams (= the English side of submission_CA_ALL.zip)
Fitted out-of-fold on English cells (v4 + en_ext). Report every recipe on en_ext g / ng and v4 g / ng.
    FOLDS=1,2,3,4,5 python run2.py ; python run2.py report
"""
import os, sys, time, glob, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["SAFE04_NOMAIN"] = "1"
os.environ.setdefault("L0_NETS", "3")
R0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0 / "Experiment/SAFE-04"))


def run(s):
    import numpy as np, pandas as pd
    from rel_sweep5 import make_fold, SP, log, V4, V4S, VA, train_rows, epochs_for, train_one, Member, L0, N_NETS
    from rel_sweep import ext_cat
    E6 = np.load(R0 / "kaggle/output/feats_eval/voice_ecapa6144_train.npy").astype(np.float32)
    E6x = ext_cat("voice", "ecapa6144", "feats_eval")
    t0 = time.time()
    fold = make_fold(s)
    TF, TV, TS, _ = train_rows(fold, ext=False, arrays=(V4["vgg"], E6, None, None))
    ep, bpe = epochs_for(len(TS))
    units = [train_one(TF, TV, TS, seed=s + 100 * i, epochs=ep) for i in range(N_NETS)]
    sc = L0(Member(units, TF, TV), fold["v4tr"], X4=dict(V4, btc=E6))
    rows = []
    for (data, prot), d in fold["files"].items():
        if data not in ("v4", "en_ext"):
            rows.append(pd.DataFrame({k: np.full(len(d["y"]), np.nan) for k in ("s010B_deep0", "s010B_cca", "s010B_prod0")}))
            continue
        S_ = sc.streams(dict(d, btc=(E6 if data == "v4" else E6x)[d["vi"]]))
        rows.append(pd.DataFrame(dict(s010B_deep0=S_["deep0"], s010B_cca=S_["cca"], s010B_prod0=S_["prod0"])))
    pd.concat(rows, ignore_index=True).to_parquet(SP / f"out/safe04/cafuse_en2_fold{s}.parquet")
    log(f"fold {s} done [{time.time() - t0:.0f}s]")


def report():
    import numpy as np, pandas as pd
    from rel_sweep5 import SP, V4, FA, VA
    from lab_fuse import table
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    parts = []
    for f in sorted(glob.glob(str(SP / "out/safe04/cafuse_en2_fold*.parquet"))):
        s = int(f.split("fold")[-1].split(".")[0])
        a = pd.read_parquet(SP / f"out/safe04/streams_fold{s}.parquet")
        a["rel"] = pd.read_parquet(SP / f"out/safe04/relonly_multi_fold{s}.parquet").relc_fia_eb.values
        a["relg"] = pd.read_parquet(SP / f"out/safe04/relonly_g_fold{s}.parquet").relg_fia_eb.values
        parts.append(pd.concat([a, pd.read_parquet(f)], axis=1))
    D = pd.concat(parts, ignore_index=True)
    D = D[D.data.isin(["v4", "en_ext"])].reset_index(drop=True)
    D["cell"] = D.data + " " + D.protocol
    v4m = D.data.values == "v4"
    pick = lambda A4, AE, idx: np.where(v4m, A4[np.clip(idx, 0, len(A4) - 1)], AE[np.clip(idx, 0, len(AE) - 1)])
    D["agef"], D["agev"] = pick(V4["agef"], FA["agef"], D.fi.values), pick(V4["agev"], VA["agev"], D.vi.values)
    D["agegap"], D["gdiff"] = np.abs(D.agef - D.agev), np.abs(D.pf_m - D.pv_m)
    D["is_en"] = 1.0
    CO = ["pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]
    ST_EN = ["CTRL_deep0", "CTRL_cca", "s010B_deep0", "s010B_cca", "CTRL_ib", "CTRL_age"]
    ST_B3 = ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "rel", "relg"]
    D["SET17_like"] = 0.5 * (D.CTRL_prod0 + D.s010B_prod0)
    X = lambda df, st: np.concatenate([df[st].values, df[CO].values, (df[st].values[:, :, None] * df[CO].values[:, None, :]).reshape(len(df), -1)], 1)
    for name, st in (("CA_EN_noacl", ST_EN), ("CA_B3", ST_B3)):
        D[name] = np.nan
        for s in sorted(D.fold.unique()):
            tr, te = D.fold != s, D.fold == s
            sc = StandardScaler().fit(X(D[tr], st))
            D.loc[te, name] = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X(D[tr], st)), D.loc[tr, "y"].values).decision_function(sc.transform(X(D[te], st)))
    T = table(D, {k: (lambda g, k=k: g[k].values) for k in ("SET17_like", "CA_EN_noacl", "CA_B3")})
    print("English cells, EER mean of 5 folds (en_ext = cross-video positives, the dev-like proxy)")
    print(T.groupby("cell").mean(numeric_only=True).drop(columns=["fold", "n"]).round(2).to_string())
    for arm in ("CA_EN_noacl", "CA_B3"):
        d = T["SET17_like"] - T[arm]; g = d.groupby(T.cell)
        m, w, w1 = g.mean(), g.apply(lambda x: (x > 0).sum()), g.apply(lambda x: (x >= 1).sum())
        print(f"  {arm:>12} vs SET17_like: " + " | ".join(f"{c}: {m[c]:+.2f} ({w[c]}/5, >=1 in {w1[c]})" for c in m.index))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1").split(",")]:
            run(s)
