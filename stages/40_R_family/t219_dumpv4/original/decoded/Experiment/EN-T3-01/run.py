r"""EN-T3-01 (registered 10/10 in README.md): the T3 head streams (anP512_deep / anP512_cca of BIGB-01, t3_relg) added to the
English fuser CA-EN (v9's English side), on the English validation cells (v4 + en_ext; en_ext = cross-video positives =
the deciding proxy), out-of-fold with purge. English table built exactly as CAFUSE-EN/run3.py (AudioCLIP of FLAG_32).
    python run.py > report.txt          (local, a few minutes)
"""
import os, sys, glob, json, importlib.util, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["SAFE04_NOMAIN"] = "1"
os.environ.setdefault("L0_NETS", "3")
HERE = Path(__file__).resolve().parent
R0 = HERE.parents[1]
sys.path[:0] = [str(R0 / "Experiment/SAFE-04"), str(R0 / "Experiment/STACK-02")]
ACL = R0 / "Experiment/CAFUSE-EN/feats_av_full"
CO = ["pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]                       # = run3 (English cells only: no is_en)
CAEN = ["CTRL_deep0", "CTRL_cca", "s010B_deep0", "s010B_cca", "CTRL_ib", "CTRL_age", "acl_full"]
T3 = ["anP512_deep", "anP512_cca", "t3_relg"]
ARMS = {"CA_EN": CAEN, "CA_EN_T3": CAEN + T3, "CA_EN_T3d": CAEN + ["anP512_deep"],
        "CA_T3": T3 + ["CTRL_ib", "CTRL_age", "acl_full"], "CA_B3": ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "rel", "relg"]}
DECISION, REF = "CA_EN_T3", "CA_EN"


def table_en():
    """run3.py's English table (v4 + en_ext rows with finite AudioCLIP), plus the T3 columns attached per fold."""
    import numpy as np, pandas as pd
    from rel_sweep5 import SP, V4, FA, VA, make_fold, N_REF, Std, l2n_, FMETA, VMETA, V4S
    R4 = R0 / "kaggle/output/feats_av"
    F4, A4 = np.load(R4 / "face_aclip_train.npy").astype(np.float32), np.load(R4 / "voice_aclip_train.npy").astype(np.float32)
    ext = {}
    for kind, meta in (("face", FMETA), ("voice", VMETA)):
        parts = []
        for src in ("v1_complete", "v2_complete"):
            a = np.load(ACL / f"ext_{src}_{kind}_aclip.npy").astype(np.float32)
            rows = np.load(ACL / f"ext_{src}_{kind}_rows.npy")
            n = int((meta.src == src).sum()); full = np.full((n, a.shape[1]), np.nan, np.float32); full[rows] = a
            parts.append(full)
        ext[kind] = np.concatenate(parts)
    parts = []
    for f in sorted(glob.glob(str(SP / "out/safe04/cafuse_en2_fold*.parquet"))):
        s = int(f.split("fold")[-1].split(".")[0])
        a = pd.read_parquet(SP / f"out/safe04/streams_fold{s}.parquet")
        a["rel"] = pd.read_parquet(SP / f"out/safe04/relonly_multi_fold{s}.parquet").relc_fia_eb.values
        a["relg"] = pd.read_parquet(SP / f"out/safe04/relonly_g_fold{s}.parquet").relg_fia_eb.values
        a = pd.concat([a, pd.read_parquet(f)], axis=1)
        b = pd.read_parquet(SP / f"out/safe04/bigb01f_fold{s}.parquet"); t = pd.read_parquet(SP / f"out/safe04/t3enc01f_fold{s}.parquet")
        for X_ in (b, t):
            assert len(X_) == len(a) and (X_.y.values == a.y.values).all() and (X_.vi.values == a.vi.values).all(), f
        a["anP512_deep"], a["anP512_cca"], a["t3_relg"] = b.anP512_deep.values, b.anP512_cca.values, t.t3_relg.values
        fold = make_fold(s); tr = fold["v4tr"]; rr = np.random.RandomState(0)
        rf, rv = tr[rr.randint(len(tr), size=N_REF)], tr[rr.randint(len(tr), size=N_REF)]
        mf, mv = F4[tr].mean(0), A4[tr].mean(0)
        cs = lambda F_, A_: np.sum(l2n_(F_ - mf) * l2n_(A_ - mv), 1)
        z = Std(cs(F4[rf], A4[rv]))
        v4m = a.data.values == "v4"
        Fx = np.where(v4m[:, None], F4[np.clip(a.fi.values, 0, len(F4) - 1)], ext["face"][np.clip(a.fi.values, 0, len(ext["face"]) - 1)])
        Ax = np.where(v4m[:, None], A4[np.clip(a.vi.values, 0, len(A4) - 1)], ext["voice"][np.clip(a.vi.values, 0, len(ext["voice"]) - 1)])
        a["acl_full"] = z(cs(Fx, Ax))
        parts.append(a)
    D = pd.concat(parts, ignore_index=True)
    D = D[D.data.isin(["v4", "en_ext"])].reset_index(drop=True)
    D["cell"] = D.data + " " + D.protocol
    D = D[np.isfinite(D.acl_full)].reset_index(drop=True)
    v4m = D.data.values == "v4"
    pick = lambda A4_, AE, idx: np.where(v4m, A4_[np.clip(idx, 0, len(A4_) - 1)], AE[np.clip(idx, 0, len(AE) - 1)])
    D["agef"], D["agev"] = pick(V4["agef"], FA["agef"], D.fi.values), pick(V4["agev"], VA["agev"], D.vi.values)
    D["agegap"], D["gdiff"] = np.abs(D.agef - D.agev), np.abs(D.pf_m - D.pv_m)
    D["fp"] = np.where(v4m, V4S[np.clip(D.fi.values, 0, len(V4S) - 1)], FMETA.p.values[np.clip(D.fi.values, 0, len(FMETA) - 1)])
    D["vp"] = np.where(v4m, V4S[np.clip(D.vi.values, 0, len(V4S) - 1)], VMETA.p.values[np.clip(D.vi.values, 0, len(VMETA) - 1)])
    return D


def main():
    import numpy as np, pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from lab_fuse import table, eer
    spec = importlib.util.spec_from_file_location("stack02run", R0 / "Experiment/STACK-02/run.py")
    S2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S2)
    persons = {int(k): set(v) for k, v in json.loads((R0 / "Experiment/STACK-02/fold_persons.json").read_text()).items()}
    D = table_en()
    print(f"English rows: {len(D)} ({D.cell.value_counts().to_dict()})")
    X = lambda df, st: np.concatenate([df[st].values, df[CO].values, (df[st].values[:, :, None] * df[CO].values[:, None, :]).reshape(len(df), -1)], 1)
    for name, st in ARMS.items():
        D[name] = np.nan
        for s in sorted(D.fold.unique()):
            te = D.fold == s
            trd = S2.purge(D[D.fold != s], persons[int(s)])
            sc = StandardScaler().fit(X(trd, st))
            D.loc[te, name] = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X(trd, st)), trd.y.values).decision_function(sc.transform(X(D[te], st)))
    T = table(D, {k: (lambda g, k=k: g[k].values) for k in ARMS})
    print("\n== English cells, EER mean of 5 folds (purged; en_ext = cross-video positives, the deciding proxy)")
    print(T.groupby("cell").mean(numeric_only=True)[list(ARMS)].round(2).to_string())
    # condition 3: en_ext g with all positives but only negatives whose predicted age gap is < 8 years, per fold
    rows = []
    for (s, cell), g in D[D.cell == "en_ext g"].groupby(["fold", "cell"]):
        sub = g[(g.y == 1) | (g.agegap < 8)]
        rows.append(dict(fold=s, **{k: eer(sub[k].values, sub.y.values) for k in ARMS}))
    A8 = pd.DataFrame(rows)
    print(f"\n== en_ext g on age-matched negatives only (predicted gap < 8 y; {int(((D.cell == 'en_ext g') & (D.y == 0) & (D.agegap < 8)).sum())} negatives)")
    print(A8.mean(numeric_only=True)[list(ARMS)].round(2).to_string())
    print(f"\n== vs {REF} (+ = better): mean (folds better, folds >= 1) | registered rule: en_ext g >= +1.0 with >= 1 in >= 4/5 folds, "
          "en_ext ng >= -0.2, age-matched gain >= overall gain")
    for arm in [a for a in ARMS if a != REF]:
        d = T[REF] - T[arm]; g = d.groupby(T.cell)
        m, w, w1 = g.mean(), g.apply(lambda x: int((x > 0).sum())), g.apply(lambda x: int((x >= 1).sum()))
        g8 = float((A8[REF] - A8[arm]).mean())
        ok = m["en_ext g"] >= 1.0 and w1["en_ext g"] >= 4 and m["en_ext ng"] >= -0.2 and g8 >= m["en_ext g"]
        print(f"  {arm:>10}: " + " | ".join(f"{c}: {m[c]:+.2f} ({w[c]}/5, >=1 in {w1[c]})" for c in m.index)
              + f" | age-matched en_ext g {g8:+.2f} -> {'PASSES' if ok else 'fails'}" + ("   <- DECISION" if arm == DECISION else ""))


if __name__ == "__main__":
    main()
