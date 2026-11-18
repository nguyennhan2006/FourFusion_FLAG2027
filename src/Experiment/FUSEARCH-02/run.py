"""FUSEARCH-02 (README registered 15/11 before any number): confirmation of the one-fuser-on-raw-streams candidates of FUSEARCH-01 on
two fresh RFX replicates (c5rl seeds 4-6 / 7-9). Kaggle CPU (FLAG_214):
    python lab_entry.py FUSEARCH-02 report   (lab data + T3S job tables + rfdc0 of FLAG_195 + rfr{c0r,c0s} of FLAG_213)
Per replicate x: control U_x = F1s' + R_x, J01_x = one CA-FUSE(F1s' streams + raw t3 / nf / nv of x, C 0.1), J01ps_x = same per
protocol; SEL nested over the three; main gate SEL - U_x on the g cells >= +0.3, >= 4/5 folds, held-out >= -0.2; side conditions
LB4-Bangla >= 0 and worst g cell >= 0. PASS = all of it on both fresh replicates. Nothing reads dev / CB / Evaluation."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


FS = _load("fusearch01", EXPD / "FUSEARCH-01" / "run.py")
M, E, S6, table, OUT, KEY, say, FAST, NB = FS.M, FS.E, FS.S6, FS.table, FS.OUT, FS.KEY, FS.say, FS.FAST, FS.NB
TAG = "fusearch2"
SRCS = [("rfd", "c0"), ("rfr", "c0r"), ("rfr", "c0s")]
FRESH = ["c0r", "c0s"]
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")


def js(p):
    return dict(mean=round(float(np.mean(p)), 3), wins=int((np.asarray(p) > 0).sum()), folds=[round(float(v), 3) for v in p])


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    CT.merge_t3s()
    got = [(p, a) for p, a in SRCS if (OUT / f"{p}{a}lab_fold1.parquet").exists()]
    if not FAST:   # 15/11 18:50: replicate 0 (rfdc0, reported only) is optional when the fresh labs run on another account
        assert [x for x in got if x[0] == "rfr"] == SRCS[1:], f"fresh tables {got} != registered {SRCS[1:]}"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for p, a in got:
            P = pd.read_parquet(OUT / f"{p}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{p}{a} fold {s}: rows"
            for c in STR:
                T[f"{a}_{c}"] = P[f"{p}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    reps = [a for _, a in got]
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG); D["is_gp"] = D.cell.str.endswith(" g")
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials, replicates {reps} (decision on {FRESH}; c0 = FUSEARCH-01, reported only)")
    for a in reps:
        for c in STR:
            D[f"{a}_{c}"] = D[f"{a}_{c}"].fillna(float(np.nanmean(D[f"{a}_{c}"].values)))
    groups = {f"trRFS_{a}": [f"{a}_{c}" for c in STR] for a in reps}
    ok = np.asarray(~E.mav_flagged(D), dtype=bool)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    FS.fuse(D, S2, persons, "F1s", F1S)
    for a in reps:
        FS.fuse(D, S2, persons, f"R_{a}", [f"trRFS_{a}"])
        for r in REG:
            D[f"U_{a}|{r}"] = D[f"F1s|{r}"] + D[f"R_{a}|{r}"]
        raw = F1S + [f"{a}_{c}" for c in STR]
        FS.fuse(D, S2, persons, f"J01_{a}", raw, C=0.1)
        FS.fuse(D, S2, persons, f"J01ps_{a}", raw, C=0.1, split=True)
    summ, verdicts = {"replicates": reps}, {}
    for a in reps:
        cands = [f"U_{a}", f"J01_{a}", f"J01ps_{a}"]
        T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in cands})
        Tg = T[T.cell.str.endswith(" g")].groupby("fold")[cands].mean()
        say(f"\n== replicate {a}{' (FUSEARCH-01 tables, reported only)' if a not in FRESH else ''}: EER (oof, mean of folds)\n"
            + T.groupby("cell").mean(numeric_only=True)[cands].round(2).T.to_string())
        for r in REG:
            D[f"SEL_{a}|{r}"] = np.nan
        picks = {}
        for s in sorted(D.fold.unique()):
            e = Tg.loc[Tg.index != s].mean(); pick = e.idxmin(); picks[int(s)] = pick
            m = (D.fold == s).values
            for r in REG:
                D.loc[m, f"SEL_{a}|{r}"] = D.loc[m, f"{pick}|{r}"]
        out = {}
        for n in cands[1:] + [f"SEL_{a}"]:
            p, h = EDA.cmp(D, n, cands[0])
            T2 = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in (cands[0], n)}).set_index(["fold", "cell"])
            x0, x1 = T2[cands[0]].unstack("cell"), T2[n].unstack("cell")
            lbb = lambda x: (((x["urdu ng"] + x["hindi ng"]) / 2 + x["urdu g"]) / 2).values
            dl, dw = lbb(x0) - lbb(x1), (x0[gc].max(1) - x1[gc].max(1)).values
            say(f"  {n:<12} g {EDA.fmt(p)} | held-out {h} | LB4-Bangla {dl.mean():+.2f} ({(dl > 0).sum()}/5) | worst g {dw.mean():+.2f} ({(dw > 0).sum()}/5)")
            out[n] = dict(g=js(p), hold=h, lb4bn=js(dl), worst=js(dw))
        sel = out[f"SEL_{a}"]
        main = EDA.rule(EDA.cmp(D, f"SEL_{a}", cands[0])[0], sel["hold"], thr=.3)
        side = sel["lb4bn"]["mean"] >= 0 and sel["worst"]["mean"] >= 0
        say(f"  SEL picks {picks}: main gate {'PASS' if main else 'FAIL'}, side conditions {'PASS' if side else 'FAIL'}")
        summ[a] = dict(picks=picks, cmp=out, main=main, side=side); verdicts[a] = main and side
    passed = all(verdicts.get(a, False) for a in FRESH)
    say(f"\n== DECISION (README section 2): {'PASS -> Evaluation path for the one-fuser system' if passed else 'FAIL -> keep FACTOR_ENFUSE'}"
        f" (per fresh replicate main + side: {verdicts})")
    summ["passed"] = passed
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
