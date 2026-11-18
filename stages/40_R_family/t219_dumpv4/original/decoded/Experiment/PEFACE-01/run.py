"""PEFACE-01 stage 1a (README section 3, "Bac 1a", registered 15/11 before any validation number of PE-Core in the CA head).
Kaggle CPU (FLAG_197):  python lab_entry.py PEFACE-01 report
Inputs: lab data + T3S job tables (merged -> ct01r2_fold<s>: F1s' streams + trCA), ctpe01r2_fold<s> (trCA of the fast-lane CA head with
PE-Core, ngnnguynchn/flag2027-pefast-r2ca-tables-p, embedded in the notebook as trCAP) and rfbc0lab_fold<s> (RFX c5rl, FLAG_193).
F1sP' = S6 + T3 v11 + trCAP vs F1s' = S6 + T3 v11 + trCA; family: UP_c0 = 2 [0.5 F1sP' + 0.5 R_c0] vs U_c0. Gates: >= +0.3 on the g cells,
>= 4/5 folds, held-out Urdu / Hindi >= -0.2 (system and family). Nothing reads dev / CB / Evaluation."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent
TAG = "pe1a"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def report():
    t00 = time.time()
    FA = _load("factor01", EXPD / "FACTOR-01" / "run.py"); M = FA.M
    E, S6, table, OUT, KEY, say, NB, FAST = M.E, M.S6, M.table, M.OUT, M.KEY, M.say, M.NB, M.FAST
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    STR, REG = ("t3", "nf", "nv"), ("oof", "hold_urdu", "hold_hindi")
    CT.merge_t3s()
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        q = OUT / f"ctpe01r2_fold{s}.npz"
        if q.exists():   # compact embed (15/11): trCA float32 + y + sha1 of (fi, vi) of the source table's rows
            import hashlib
            Z = np.load(q); sha = hashlib.sha1(np.asarray(T.fi, np.int64).tobytes() + np.asarray(T.vi, np.int64).tobytes()).hexdigest()
            assert len(Z["trCA"]) == len(T) and (Z["y"] == T.y.to_numpy().astype(np.int8)).all() and str(Z["sha"]) == sha, f"ctpe fold {s}: rows"
            T["trCAP"] = Z["trCA"].astype(np.float64)
        else:
            assert FAST, f"{q.name} missing (attach / embed the PE-Core CA tables)"
            T["trCAP"] = T["trCA"].values + np.random.RandomState(s).normal(0, .3, len(T))    # smoke stand-in only
        P = pd.read_parquet(OUT / f"rfbc0lab_fold{s}.parquet")
        assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"rfbc0 fold {s}: rows"
        for c in STR:
            T[f"c0_{c}"] = P[f"rfbc0_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    F1S = S6 + E.V11_T3 + ["trCA"]; F1SP = S6 + E.V11_T3 + ["trCAP"]
    say(f"== PEFACE-01 stage 1a: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}")
    for c in STR:
        D[f"c0_{c}"] = D[f"c0_{c}"].fillna(float(np.nanmean(D[f"c0_{c}"].values)))
    ok = ~E.mav_flagged(D)
    D["trRFS_c0"] = np.nan
    for s in sorted(D.fold.unique()):
        te = (D.fold == s).values
        D.loc[te, "trRFS_c0"] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), [f"c0_{c}" for c in STR])(D[te])
    S2.Design = CT._nested_design(S2, persons, {"trRFS_c0": [f"c0_{c}" for c in STR]})
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    EDA.run_arms(D, S2, persons, {"F1s": F1S, "F1sP": F1SP, "R_c0": ["trRFS_c0"]})
    for r in REG:
        D[f"U_c0|{r}"] = 2 * (0.5 * D[f"F1s|{r}"] + 0.5 * D[f"R_c0|{r}"])
        D[f"UP_c0|{r}"] = 2 * (0.5 * D[f"F1sP|{r}"] + 0.5 * D[f"R_c0|{r}"])
    fam = ["F1s", "F1sP", "R_c0", "U_c0", "UP_c0"]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[fam].round(2).T.to_string())
    Ts = table(D, {c: (lambda g, c=c: g[c].values) for c in ("trCA", "trCAP")})
    say("standalone CA head (no fuser):\n" + Ts.groupby("cell").mean(numeric_only=True)[["trCA", "trCAP"]].round(2).T.to_string())
    js = lambda q: dict(mean=round(float(q.mean()), 3), wins=int((q > 0).sum()), folds=[round(float(v), 3) for v in q])
    summ = {}
    for cand, ctl, name in (("F1sP", "F1s", "system"), ("UP_c0", "U_c0", "family")):
        p, h = EDA.cmp(D, cand, ctl)
        lo, hi = S2.boot(D.rename(columns={f"{ctl}|oof": "ref_", f"{cand}|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
        ga = EDA.rule(p, h, thr=.3)
        say(f"\n  {name}: {cand} - {ctl} (g cells): {EDA.fmt(p)} | held-out {h} | CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}]"
            f" | GATE: {'PASS' if ga else 'FAIL'} (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
        summ[name] = dict(js(p), hold=h, ci95=[round(lo, 3), round(hi, 3)], passed=ga)
    ps = summ["system"]["mean"]
    say(f"  dropout grid (stage 1, 9 GPU lab runs): {'RUN (system delta >= 0)' if ps >= 0 else 'NOT RUN (system delta < 0, README stage 1a)'}")
    summ["grid"] = ps >= 0
    summ["eer_cells"] = T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict()
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
