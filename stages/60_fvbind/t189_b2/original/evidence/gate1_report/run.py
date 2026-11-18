"""FVBIND-01 gate 1 (README section 3, registered 14/11): local CPU, seconds.
    python Experiment/FVBIND-01/run.py gate [dir ...]     (default: every fvb*_fold*.parquet under Experiment/FVBIND-01/out)
EER per (fold, cell) of every standalone stream fvb<arm>; g-mean = mean of the g cells (v4 g, en_ext g, urdu g); arm B_x passes when
B_x - B0 >= +1.0 on the 5-fold mean AND better on >= 4/5 folds. IB0 / C1ref are references only. Writes gate1_report.txt / gate1.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent


def eer(s, y):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


def gate(dirs):
    files = sorted({p for d in dirs for p in Path(d).rglob("fvb*_fold*.parquet")})
    arms = sorted({p.name.split("_fold")[0][3:] for p in files})
    rows, txt = [], []
    say = lambda s: (print(s, flush=True), txt.append(s))
    for p in files:
        arm, s = p.name.split("_fold")[0][3:], int(p.stem[-1])
        D = pd.read_parquet(p); D["cell"] = D.data.astype(str) + " " + D.protocol.astype(str).map({"gender": "g", "no_gender": "ng"}).fillna(D.protocol.astype(str))
        for c, g in D.groupby("cell"):
            x, y = g[f"fvb{arm}"].to_numpy(), g.y.to_numpy(); k = np.isfinite(x)
            if k.sum() > 10 and len(set(y[k])) == 2:
                rows.append(dict(arm=arm, fold=s, cell=c, eer=eer(x[k], y[k]), finite=float(k.mean())))
    T = pd.DataFrame(rows)
    say(f"FVBIND-01 gate 1: arms {arms}, files {len(files)}")
    say("EER (mean of folds), finite share in brackets:\n" + T.pivot_table(index="arm", columns="cell", values="eer").round(2).to_string())
    say(T.pivot_table(index="arm", columns="cell", values="finite").round(3).to_string())
    gc = sorted(c for c in T.cell.unique() if c.endswith(" g"))
    G = T[T.cell.isin(gc)].pivot_table(index="fold", columns="arm", values="eer", aggfunc="mean")
    say(f"g-mean per fold (cells {gc}):\n" + G.round(2).to_string())
    out = {"g_cells": gc, "g_mean": G.round(3).to_dict()}
    if "B0" in G:
        for arm in [a for a in G.columns if a != "B0"]:
            ca = sorted(set(T.cell[(T.arm == arm) & T.cell.isin(gc)]))      # common g cells (C1ref: v4 g only)
            Gc = T[T.cell.isin(ca)].pivot_table(index="fold", columns="arm", values="eer", aggfunc="mean")
            d = (Gc["B0"] - Gc[arm]).dropna(); ok = len(d) == 5 and d.mean() >= 1.0 and (d > 0).sum() >= 4
            ref = arm in ("IB0", "C1ref")
            say(f"  {arm} - B0 on g-mean: {d.mean():+.2f} (folds {', '.join(f'{v:+.2f}' for v in d)}; wins {(d > 0).sum()}/{len(d)})"
                + ("  [reference]" if ref else f"  GATE 1: {'PASS' if ok else 'FAIL'}"))
            out[arm] = dict(mean=round(float(d.mean()), 3), wins=int((d > 0).sum()), folds=[round(float(v), 3) for v in d], passed=None if ref else bool(ok))
    (HERE / "gate1_report.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    (HERE / "gate1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ gate 2 (README section 3 / 3b): Kaggle CPU, FLAG_194
#   python lab_entry.py FVBIND-01 report   (lab data + T3S job tables + rfb<arm>lab_fold<s> of FLAG_193, arms c0 / fvb)
# U_x = 2 [0.5 F1s' + 0.5 R_x], R_x = CA-FUSE of the arm's t3 / nf / nv (nested + purged, 3 regimes), as TRANSL-01 / FACTOR-01.
# Gate (registered): U_fvb - U_c0 on the g cells >= +0.3, better on >= 4/5 folds, held-out Urdu / Hindi >= -0.2.
G2_ARMS = ["c0", "fvb"]
G2_TAG, G2_CTL, G2_CAND = "fvbg2", "U_c0", "U_fvb"


def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def report():
    import time
    t00 = time.time(); EXPD = HERE.parent
    FA = _load("factor01", EXPD / "FACTOR-01" / "run.py"); M = FA.M
    E, S6, table, OUT, KEY, say, NB = M.E, M.S6, M.table, M.OUT, M.KEY, M.say, M.NB
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    STR, REG = ("t3", "nf", "nv"), ("oof", "hold_urdu", "hold_hindi")
    CT.merge_t3s()
    got = [a for a in G2_ARMS if (OUT / f"rfb{a}lab_fold1.parquet").exists()]
    assert got == G2_ARMS, f"tables {got} != registered {G2_ARMS}"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for a in got:
            P = pd.read_parquet(OUT / f"rfb{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"rfb{a} fold {s}: rows"
            for c in STR:
                T[f"{a}_{c}"] = P[f"rfb{a}_{c}"].values
        T.to_parquet(OUT / f"{G2_TAG}_fold{s}.parquet")
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(G2_TAG)
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== FVBIND-01 gate 2: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}, arms {got}")
    for a in got:
        for c in STR:
            D[f"{a}_{c}"] = D[f"{a}_{c}"].fillna(float(np.nanmean(D[f"{a}_{c}"].values)))
    groups = {f"trRFS_{a}": [f"{a}_{c}" for c in STR] for a in got}
    ok = ~E.mav_flagged(D)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    EDA.run_arms(D, S2, persons, {"F1s": F1S, **{f"R_{a}": [f"trRFS_{a}"] for a in got}})
    for a in got:
        for r in REG:
            D[f"U_{a}|{r}"] = 2 * (0.5 * D[f"F1s|{r}"] + 0.5 * D[f"R_{a}|{r}"])
    fam = ["F1s"] + [f"R_{a}" for a in got] + [f"U_{a}" for a in got]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[fam].round(2).T.to_string())
    p, h = EDA.cmp(D, G2_CAND, G2_CTL); pr, hr = EDA.cmp(D, "R_fvb", "R_c0")
    lo, hi = S2.boot(D.rename(columns={f"{G2_CTL}|oof": "ref_", f"{G2_CAND}|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
    ga = EDA.rule(p, h, thr=.3)
    say(f"\n  {G2_CAND} - {G2_CTL} (g cells): {EDA.fmt(p)} | held-out {h} | CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}]")
    say(f"  R_fvb - R_c0 (g cells): {EDA.fmt(pr)} | held-out {hr}")
    say(f"  GATE 2: {'PASS -> Evaluation path (RFX c5rl + v_fvb), zip frozen, not submitted' if ga else 'FAIL -> keep FACTOR_ENFUSE, paper only'}"
        f" (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
    js = lambda q: dict(mean=round(float(q.mean()), 3), wins=int((q > 0).sum()), folds=[round(float(v), 3) for v in q])
    summ = dict(arms=got, U=dict(js(p), hold=h, ci95=[round(lo, 3), round(hi, 3)]), R=dict(js(pr), hold=hr), passed=ga,
                eer_cells=T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict())
    (OUT / f"{G2_TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{G2_TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{G2_TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    cmd, rest = (sys.argv[1] if len(sys.argv) > 1 else "gate"), sys.argv[2:]
    {"gate": lambda: gate(rest or [HERE / "out"]), "report": report}[cmd]()
