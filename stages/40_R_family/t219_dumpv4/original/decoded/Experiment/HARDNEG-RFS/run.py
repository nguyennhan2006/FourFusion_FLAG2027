"""HARDNEG-RFS + EN-HEARD report (READMEs registered 12/11 ~21:00 before any number), SAFE-04 validation, Kaggle CPU (FLAG_181):
    python lab_entry.py HARDNEG-RFS report     (lab data + T3S job tables + rfh<arm>lab_fold<s> of FLAG_180 in <SP>/out/safe04)
U_x = CA-FUSE(S6 + V11_T3 + trCA) + CA-FUSE(trRFS_x) as RFSTUNE-01 / ENPROG-01. HARDNEG: SEL_h over {a0, h*} on g cells, system rule.
EN-HEARD: SEL_e over {a0, e*} on the two English cells, English rule. Nothing reads dev / CB / Evaluation. Writes engx_report.txt / _summary.json."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P = _load("enprog01", EXPD / "ENPROG-01" / "run.py")               # dl, fmt, en_rule, js; P.M = RFSTUNE-01 (say / TXT, OUT, KEY)
M, E, S6, OUT, KEY, say, FAST, NB, table = P.M, P.E, P.S6, P.OUT, P.KEY, P.say, P.FAST, P.NB, P.table
TAG, PFX = "engx", "rfh"
ARMS = ["a0", "e3s2", "h025", "h05", "h10", "h05k", "h05d1", "e1", "e2", "e3", "e3d1"]   # = kaggle/build_flag173.py ARMS4 (order)
FAM = {"HARDNEG": ["a0", "h025", "h05", "h10", "h05k", "h05d1"], "ENHEARD": ["a0", "e1", "e2", "e3", "e3d1", "e3s2"]}
ENG, ENNG, OTH, EN = P.ENG, P.ENNG, P.OTH, P.EN


def merge(CT):
    CT.merge_t3s()
    got = [a for a in ARMS if (OUT / f"{PFX}{a}lab_fold1.parquet").exists()]
    if not FAST:
        assert got == ARMS, f"arms {got} != registered {ARMS}"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for a in got:
            Q = pd.read_parquet(OUT / f"{PFX}{a}lab_fold{s}.parquet")
            assert len(Q) == len(T) and all((Q[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{a} fold {s}"
            for c in ("t3", "nf", "nv"):
                T[f"{PFX}{a}_{c}"] = Q[f"{PFX}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    say(f"merged {TAG}: {got}")
    return got


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    arms = merge(CT)
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    cols = {a: [f"{PFX}{a}_t3", f"{PFX}{a}_nf", f"{PFX}{a}_nv"] for a in arms}
    for a, cs in cols.items():
        for c in cs:
            D[c] = D[c].fillna(float(np.nanmean(D[c].values)))
    groups = {f"trRFS_{a}": cs for a, cs in cols.items()}
    ok = ~E.mav_flagged(D)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    say("\n== fused arms (CA-FUSE C = 1, nested + purged, 3 regimes)")
    EDA.run_arms(D, S2, persons, {"F1s": S6 + E.V11_T3 + ["trCA"], **{f"R_{a}": [f"trRFS_{a}"] for a in arms}})
    for a in arms:
        EDA.add_sum(D, f"U_{a}", ["F1s", f"R_{a}"])
    names = ["F1s"] + [f"U_{a}" for a in arms] + [f"R_{a}" for a in arms]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in names})
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[names].round(2).T.to_string())
    summ = {"arms": arms, "eer_cells": T.groupby("cell").mean(numeric_only=True)[names].round(3).to_dict()}
    if "a0" not in arms:
        say("no a0 (smoke): stop"); return
    say("\n== each arm vs a0 (U level; > 0 = arm better)")
    per = {}
    for a in [a for a in arms if a != "a0"]:
        p, h = EDA.cmp(D, f"U_{a}", "U_a0")
        pg, png = P.dl(T, f"U_{a}", "U_a0", ENG), P.dl(T, f"U_{a}", "U_a0", ENNG)
        say(f"  {a:<6} g cells {P.fmt(p)} | held-out {h}\n         en g {P.fmt(pg)} | en ng {P.fmt(png)}")
        per[a] = dict(g=P.js(p), hold=h, en_g=P.js(pg), en_ng=P.js(png))
    summ["vs_a0"] = per
    folds = sorted(int(s) for s in T.fold.unique()); gcells = sorted(c for c in T.cell.unique() if c.endswith(" g"))
    for fam, fa in FAM.items():
        fa = [a for a in fa if a in arms]; ua = [f"U_{a}" for a in fa]; cells = gcells if fam == "HARDNEG" else EN
        say(f"\n== {fam}: nested choice over {fa} on {'g cells' if fam == 'HARDNEG' else 'the two English cells'}")
        picks = {}
        for r in ("oof", "hold_urdu", "hold_hindi"):
            D[f"SEL{fam}|{r}"] = np.nan
        for s in folds:
            pick = T[(T.fold != s) & T.cell.isin(cells)].groupby("fold")[ua].mean().mean().idxmin(); picks[s] = pick[2:]
            m = (D.fold == s).values
            for r in ("oof", "hold_urdu", "hold_hindi"):
                D.loc[m, f"SEL{fam}|{r}"] = D.loc[m, f"{pick}|{r}"]
        p, h = EDA.cmp(D, f"SEL{fam}", "U_a0")
        T2 = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in (f"SEL{fam}", "U_a0")})
        pg, png = P.dl(T2, f"SEL{fam}", "U_a0", ENG), P.dl(T2, f"SEL{fam}", "U_a0", ENNG)
        if fam == "HARDNEG":
            lo, hi = S2.boot(D.rename(columns={"U_a0|oof": "ref_", f"SEL{fam}|oof": "cand_"}), "ref_", "cand_", gcells, "both", B=NB)
            passed = EDA.rule(p, h, thr=.3)
            say(f"  picks {picks}\n  SEL_h - a0 g cells {P.fmt(p)} | held-out {h} | CI95 [{lo:+.2f}, {hi:+.2f}] -> system rule {'PASS' if passed else 'FAIL'}")
            say(f"     en g {P.fmt(pg)} | en ng {P.fmt(png)}")
            summ[fam] = dict(picks=picks, g=P.js(p), hold=h, ci95=[round(lo, 3), round(hi, 3)], passed=passed, en_g=P.js(pg), en_ng=P.js(png))
        else:
            passed = P.en_rule(pg, png)
            say(f"  picks {picks}\n  SEL_e - a0 en g {P.fmt(pg)} | en ng {P.fmt(png)} -> English rule {'PASS' if passed else 'FAIL'}")
            say(f"     (cost elsewhere) g cells {P.fmt(p)} | held-out {h}")
            summ[fam] = dict(picks=picks, en_g=P.js(pg), en_ng=P.js(png), passed=passed, g=P.js(p), hold=h)
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
