"""RFXARCH-01 (README registered 15/11 before any number): RFX projector architecture / loss, SAFE-04. Kaggle CPU (FLAG_206):
    python lab_entry.py RFXARCH-01 report    (lab data + T3S job tables + rfa<arm>lab_fold<s> of the two FLAG_205 runs)
U_x = F1s' + R_x (1:1), x in {c0, res, id05, id10, resid05}; control U_c0. SEL = nested choice on the g cells; gate SEL - U_c0 >= +0.3,
>= 4/5 folds, held-out Urdu / Hindi >= -0.2. R level reported only. Nothing reads dev / CB / Evaluation."""
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
TAG, PFX = "rfxarch", "rfa"
ARMS = ["c0", "res", "id05", "id10", "resid05"]                    # FLAG_205 = kaggle/build_flag173.py ARMS11
CTL = "U_c0"
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")


def js(p):
    return dict(mean=round(float(p.mean()), 3), wins=int((p > 0).sum()), folds=[round(float(v), 3) for v in p])


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    CT.merge_t3s()
    have = [a for a in ARMS if (OUT / f"{PFX}{a}lab_fold1.parquet").exists()]
    if not FAST:
        assert have == ARMS, f"tables {have} != registered {ARMS}"
    assert "c0" in have, "no control table"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for a in have:
            P = pd.read_parquet(OUT / f"{PFX}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{a} fold {s}: rows"
            for c in STR:
                T[f"{a}_{c}"] = P[f"{PFX}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}, arms {have}")
    for a in have:
        for c in STR:
            D[f"{a}_{c}"] = D[f"{a}_{c}"].fillna(float(np.nanmean(D[f"{a}_{c}"].values)))
    groups = {f"trRFS_{a}": [f"{a}_{c}" for c in STR] for a in have}
    ok = np.asarray(~E.mav_flagged(D), dtype=bool)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    EDA.run_arms(D, S2, persons, {"F1s": F1S, **{f"R_{a}": [f"trRFS_{a}"] for a in have}})
    for a in have:
        EDA.add_sum(D, f"U_{a}", ["F1s", f"R_{a}"])
    fam = ["F1s"] + [f"R_{a}" for a in have] + [f"U_{a}" for a in have]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    Tg = T[T.cell.str.endswith(" g")].groupby("fold")[fam].mean()
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[fam].round(2).T.to_string())
    summ = {"arms": have, "eer_cells": T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict()}

    say("\n== each arm vs c0 on the g cells (> 0 = arm better): U level | R level")
    per = {}
    for a in [a for a in have if a != "c0"]:
        pu, hu = EDA.cmp(D, f"U_{a}", CTL); pr, hr = EDA.cmp(D, f"R_{a}", "R_c0")
        say(f"  {a:<8} U {EDA.fmt(pu)} | held-out {hu}\n  {'':8} R {EDA.fmt(pr)} | held-out {hr}")
        per[a] = dict(U=js(pu), hold_U=hu, R=js(pr), hold_R=hr)
    summ["vs_c0"] = per

    ua = [f"U_{a}" for a in have]; picks = {}
    for r in REG:
        D[f"SEL|{r}"] = np.nan
    for s in sorted(D.fold.unique()):
        e = Tg.loc[Tg.index != s, ua].mean(); pick = e.idxmin(); picks[int(s)] = pick
        m = (D.fold == s).values
        for r in REG:
            D.loc[m, f"SEL|{r}"] = D.loc[m, f"{pick}|{r}"]
        say(f"  fold {int(s)}: pick {pick} (inner g EER {e.min():.2f}; {CTL} {e[CTL]:.2f})")
    pa, ha = EDA.cmp(D, "SEL", CTL)
    lo, hi = S2.boot(D.rename(columns={f"{CTL}|oof": "ref_", "SEL|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
    ga = EDA.rule(pa, ha, thr=.3)
    final = Tg[ua].mean().idxmin()
    say(f"  SEL - {CTL}: {EDA.fmt(pa)} | held-out {ha} | CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}]")
    say(f"  best arm over the 5 folds: {final} (mean g EER {Tg[final].mean():.2f}, {CTL} {Tg[CTL].mean():.2f})")
    say(f"  GATE: {'PASS -> Evaluation zip (final = ' + final + ')' if ga else 'FAIL -> keep FACTOR_ENFUSE, paper only'}"
        f" (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
    summ["SEL"] = dict(picks=picks, best_5fold=final, a=dict(js(pa), hold=ha, ci95=[round(lo, 3), round(hi, 3)]), passed=ga)
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
