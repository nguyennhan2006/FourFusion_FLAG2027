"""RFXFUSE-01 + LORAFIX-01 (READMEs registered 15/11 before any number), v4 cells of SAFE-04 only. Kaggle CPU (FLAG_208):
    python lab_entry.py RFXFUSE-01 report    (lab data + T3S job tables + rfdc0 of FLAG_195 [+ rflL1 / rflL2 of FLAG_207])
Phase A (RFXFUSE-01): U_CA = F1s' + R_CA (the validated RFX form: CA-FUSE of trRFS over every SAFE-04 row) vs U_RFSF = F1s' + RFSF,
RFSF = the RFS final-fuser form fitted on the v4 rows of the other folds (purged): g cells LR on [t3, ib, nf, nv]; ng cells LR on
[t3, ib, nf, nv, age] x [pf_m, pv_m, gender_agree] with interactions. Phase B (LORAFIX-01, when the rfl tables exist): RFSF with the
ib of the re-learned LoRA (L1 / L2, nested pick) vs the old ib, other streams of rfdc0. Nothing reads dev / CB / Evaluation."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


FA = _load("factor01", EXPD / "FACTOR-01" / "run.py")
M = FA.M
E, S6, table, OUT, KEY, say, FAST, NB = M.E, M.S6, M.table, M.OUT, M.KEY, M.say, M.FAST, M.NB
TAG = "rfxfuse"
ALLC = ("t3", "nf", "nv", "ib", "age", "gender_agree")
V4C = ["v4 g", "v4 ng"]


def js(p):
    return dict(mean=round(float(np.mean(p)), 3), wins=int((np.asarray(p) > 0).sum()), folds=[round(float(v), 3) for v in p])


def fmt(p):
    return ", ".join(f"f{i + 1} {v:+.2f}" for i, v in enumerate(p)) + f" | mean {np.mean(p):+.2f}, {(np.asarray(p) > 0).sum()}/{len(p)}"


def rfsf(D, V, persons, ib):
    """RFS-final-fuser form on the v4 cells: per outer fold, fitted on the v4 rows of the other folds (purged, same cell)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = pd.Series(np.nan, index=V.index)
    def X(df, cell):
        st = np.c_[df["c0_t3"], df[ib], df["c0_nf"], df["c0_nv"]]
        if cell.endswith(" g"):
            return st
        st = np.c_[st, df["c0_age"]]; co = np.c_[df["pf_m"], df["pv_m"], df["c0_gender_agree"]]
        return np.c_[st, co, (st[:, :, None] * co[:, None, :]).reshape(len(df), -1)]
    for s in sorted(V.fold.unique()):
        for cell in V4C:
            tr = V[(V.fold != s) & (V.cell == cell)]; tr = tr[~tr.fp.isin(persons[int(s)]) & ~tr.vp.isin(persons[int(s)])]
            te = V[(V.fold == s) & (V.cell == cell)]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X(tr, cell), tr.y.values)
            out.loc[te.index] = m.decision_function(X(te, cell))
    return out


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    CT.merge_t3s()
    assert (OUT / "rfdc0lab_fold1.parquet").exists(), "attach the FLAG_195 runs (rfdc0 tables)"
    loras = [a for a in ("L1", "L2") if (OUT / f"rfl{a}lab_fold1.parquet").exists()]
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for p, a, cs in [("rfd", "c0", ALLC)] + [("rfl", a, ("ib",)) for a in loras]:
            P = pd.read_parquet(OUT / f"{p}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{p}{a} fold {s}: rows"
            for c in cs:
                T[f"{a}_{c}"] = P[f"{p}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials; phase A (RFXFUSE-01){' + phase B (LORAFIX-01): ' + str(loras) if loras else ' only (no rfl tables)'}")
    for c in ("t3", "nf", "nv"):
        D[f"c0_{c}"] = D[f"c0_{c}"].fillna(float(np.nanmean(D[f"c0_{c}"].values)))
    groups = {"trRFS_c0": [f"c0_{c}" for c in ("t3", "nf", "nv")]}
    ok = np.asarray(~E.mav_flagged(D), dtype=bool)
    D["trRFS_c0"] = np.nan
    for s in sorted(D.fold.unique()):
        te = (D.fold == s).values
        D.loc[te, "trRFS_c0"] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), groups["trRFS_c0"])(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    EDA.run_arms(D, S2, persons, {"F1s": F1S, "R_CA": ["trRFS_c0"]})
    V = D[D.cell.isin(V4C)].copy()
    ibs = ["c0_ib"] + [f"{a}_ib" for a in loras]
    for c in ibs + ["c0_age", "c0_gender_agree"]:
        miss = float(np.isnan(V[c].values).mean())
        say(f"  v4 rows: {c} missing {100 * miss:.2f} %")
        V[c] = V[c].fillna(float(np.nanmean(V[c].values)))
    V["U_CA"] = V["F1s|oof"] + V["R_CA|oof"]
    for ib in ibs:
        k = ib.split("_")[0]
        V[f"RFSF_{k}"] = rfsf(D, V, persons, ib); V[f"U_RFSF_{k}"] = V["F1s|oof"] + V[f"RFSF_{k}"]
    cols = ["F1s|oof", "R_CA|oof", "U_CA"] + [f"RFSF_{ib.split('_')[0]}" for ib in ibs] + [f"U_RFSF_{ib.split('_')[0]}" for ib in ibs] + ibs
    T = table(V, {c: (lambda g, c=c: g[c].values) for c in cols})
    T = T[T.cell.isin(V4C)]
    say("EER on the v4 cells (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[cols].round(2).T.to_string())
    per = {c: T[T.cell == c].set_index("fold").sort_index() for c in V4C}
    summ = {"loras": loras, "eer_v4": T.groupby("cell").mean(numeric_only=True)[cols].round(3).to_dict()}

    say("\n== phase A (RFXFUSE-01): U_RFSF_c0 - U_CA as EER(U_RFSF) - EER(U_CA) (> 0 = the validated CA form better)")
    dg = (per["v4 g"]["U_RFSF_c0"] - per["v4 g"]["U_CA"]).values; dn = (per["v4 ng"]["U_RFSF_c0"] - per["v4 ng"]["U_CA"]).values
    rg = (per["v4 g"]["RFSF_c0"] - per["v4 g"]["R_CA|oof"]).values
    say(f"  v4 g  U: {fmt(dg)}\n  v4 ng U: {fmt(dn)}\n  (R level, v4 g: {fmt(rg)})")
    if np.mean(dg) >= 0 and (dg > 0).sum() >= 3 and np.mean(dn) >= -0.2:
        va = "CA form favoured -> phase C (dump the Evaluation t3 / nf / nv, export R_CA, reference zip)"
    elif -np.mean(dg) >= 0.3 and (dg < 0).sum() >= 4:
        va = "RFS fuser favoured -> keep it (LORAFIX-01 more relevant)"
    else:
        va = "in between -> keep the locked system, paper only"
    say(f"  VERDICT A: {va}")
    summ["A"] = dict(v4g=js(dg), v4ng=js(dn), R_v4g=js(rg), verdict=va)

    if loras:
        say("\n== phase B (LORAFIX-01): ib standalone and U_RFSF with the re-learned LoRA vs the old one (> 0 = new better)")
        for a in loras:
            pg = (per["v4 g"]["c0_ib"] - per["v4 g"][f"{a}_ib"]).values; pn = (per["v4 ng"]["c0_ib"] - per["v4 ng"][f"{a}_ib"]).values
            ug = (per["v4 g"]["U_RFSF_c0"] - per["v4 g"][f"U_RFSF_{a}"]).values; un = (per["v4 ng"]["U_RFSF_c0"] - per["v4 ng"][f"U_RFSF_{a}"]).values
            say(f"  {a}: ib v4 g {fmt(pg)}\n      ib v4 ng {fmt(pn)}\n      U  v4 g {fmt(ug)}\n      U  v4 ng {fmt(un)}")
            summ[f"B_{a}"] = dict(ib_v4g=js(pg), ib_v4ng=js(pn), U_v4g=js(ug), U_v4ng=js(un))
        g = per["v4 g"]; n = per["v4 ng"]; folds = list(g.index); picks, sg, sn = {}, [], []
        for s in folds:
            inner = {a: g.loc[[f for f in folds if f != s], f"U_RFSF_{a}"].mean() for a in loras}
            a = min(inner, key=inner.get); picks[int(s)] = a
            sg.append(g.loc[s, "U_RFSF_c0"] - g.loc[s, f"U_RFSF_{a}"]); sn.append(n.loc[s, "U_RFSF_c0"] - n.loc[s, f"U_RFSF_{a}"])
        sg, sn = np.array(sg), np.array(sn)
        passed = bool(np.mean(sg) >= 0.3 and (sg > 0).sum() >= 4 and np.mean(sn) >= -0.2)
        best = min(loras, key=lambda a: g[f"U_RFSF_{a}"].mean())
        say(f"  nested pick {picks}: v4 g {fmt(sg)} | v4 ng mean {np.mean(sn):+.2f}")
        say(f"  GATE B: {'PASS -> reference zip FACTOR_ENFUSE with LoRA ' + best if passed else 'FAIL -> keep the old LoRA (fix the system description)'}"
            f" (>= +0.30, >= 4/5 folds, v4 ng >= -0.20)")
        summ["B"] = dict(picks=picks, v4g=js(sg), v4ng_mean=round(float(np.mean(sn)), 3), passed=passed, best=best)
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
