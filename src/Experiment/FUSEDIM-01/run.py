"""FUSEDIM-01 (README registered 14/11 before any number): width of the RFX shared space and the way F1s' and RFX are fused, SAFE-04.
Kaggle CPU (FLAG_196):
    python lab_entry.py FUSEDIM-01 report      (lab data + T3S job tables + rfd<arm>lab_fold<s> of the two FLAG_195 runs)
A (README section 1): RFX c5rl projector width in {64, 128, 256, 512 = c0, 1024, 2048}; B: ENSD = mean of the six fused R.
C: fusion of l_F (F1s' fuser) and l_R (R_x): lin{w} = 2 [w l_F + (1 - w) l_R], w in {.5, .6, .65, .7, .8}; prob{w} = w sig(l_F) +
(1 - w) sig(l_R), w in {.5, .65}; stk2 / stkc / hgb = second-level fusers learned on the other outer folds' oof scores (purged,
MAV-flagged rows dropped, held language dropped for the held-out regimes); joint = one-level CA-FUSE(F1s' streams + trRFS_x).
Control U_c0_lin50. SEL = nested choice on the g cells over the 76 candidates; gate SEL - control >= +0.3, >= 4/5 folds, held-out
Urdu / Hindi >= -0.2. Nothing reads dev / CB / Evaluation. Writes fusedim_report.txt, fusedim_summary.json."""
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
TAG, PFX = "fusedim", "rfd"
D_ARMS = ["c0", "d2048", "d64", "d1024", "d128", "d256"]           # FLAG_195 = kaggle/build_flag173.py ARMS9 (order = part balance)
DIMS = {"c0": 512, "d64": 64, "d128": 128, "d256": 256, "d1024": 1024, "d2048": 2048}
LIN = [0.5, 0.6, 0.65, 0.7, 0.8]
PROB = [0.5, 0.65]
STK = ["stk2", "stkc", "hgb"]
CTL = "U_c0_lin50"
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")
CO = ["is_en", "pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]   # = STACK-02 CO (the F1s' fuser conditions)


def merge(CT):
    CT.merge_t3s()
    got = [a for a in D_ARMS if (OUT / f"{PFX}{a}lab_fold1.parquet").exists()]
    if not FAST:
        assert got == D_ARMS, f"tables {got} != registered {D_ARMS}"
    assert "c0" in got, "no control table"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for a in got:
            P = pd.read_parquet(OUT / f"{PFX}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{PFX}{a} fold {s}: rows"
            for c in STR:
                T[f"{a}_{c}"] = P[f"{PFX}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    say(f"merged {TAG}: {got}")
    return got


def js(p):
    return dict(mean=round(float(p.mean()), 3), wins=int((p > 0).sum()), folds=[round(float(v), 3) for v in p])


def _feats(df, x, r, kind):
    f, g = df[f"F1s|{r}"].values, df[f"R_{x}|{r}"].values
    if kind == "stk2":
        return np.c_[f, g]
    c = df[CO].values.astype(float)
    if kind == "stkc":
        return np.c_[f, g, c, f[:, None] * c, g[:, None] * c]
    return np.c_[f, g, c]                                             # hgb


def _model(kind):
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                              l2_regularization=1.0, monotonic_cst=[1, 1] + [0] * len(CO), early_stopping=False,
                                              random_state=0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def stack(D, S2, persons, ok, x, kind):
    """Second-level fuser (README section 1C): outer fold s learns on the oof scores of the other folds (purged of fold s persons,
    MAV-flagged rows dropped; held-out regimes also drop the held language), then scores fold s rows of each regime."""
    n = f"U_{x}_{kind}"
    for r in REG:
        D[f"{n}|{r}"] = np.nan
    w = {}
    for s in sorted(D.fold.unique()):
        base = S2.purge(D[(D.fold != s).values & ok], persons[int(s)])
        for r, held in S2.HELD.items():
            tr = base[~base.data.isin(held)] if held else base
            m = _model(kind).fit(_feats(tr, x, "oof", kind), tr.y.values)
            te = D[(D.fold == s) & (D.data.isin(held) if held else True)]
            D.loc[te.index, f"{n}|{r}"] = m.decision_function(_feats(te, x, r, kind))
            if kind == "stk2" and r == "oof":                         # learned weights on the standardised [l_F, l_R] (report only)
                c = m[-1].coef_[0] / m[0].scale_
                w[int(s)] = round(float(c[0] / (c[0] + c[1])), 3)
    return n, w


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    have = merge(CT)
    S2, persons = E.stack02(), E.fold_persons()
    assert list(S2.CO) == CO, f"fuser conditions changed: {S2.CO}"
    D = E.safe04(TAG)
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}, tables {have}; F1s streams {F1S}")
    for a in have:
        for c in STR:
            col = f"{a}_{c}"
            if a == "c0":
                say(f"  coverage {col}: {100 * np.isfinite(D[col].values).mean():.1f} % finite")
            D[col] = D[col].fillna(float(np.nanmean(D[col].values)))   # MAV rows without an RF feature (as FACTOR-01)
    groups = {f"trRFS_{a}": [f"{a}_{c}" for c in STR] for a in have}
    ok = np.asarray(~E.mav_flagged(D), dtype=bool)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    summ = {"arms": have, "dims": {a: DIMS[a] for a in have}, "lin": LIN, "prob": PROB, "stack": STK, "control": CTL}

    say("\n== level 1: F1s and R_x (CA-FUSE C = 1, nested + purged, 3 regimes); joint = one-level CA-FUSE(F1s streams + trRFS_x)")
    EDA.run_arms(D, S2, persons, {"F1s": F1S, **{f"R_{a}": [f"trRFS_{a}"] for a in have}})
    EDA.run_arms(D, S2, persons, {f"U_{a}_joint": F1S + [f"trRFS_{a}"] for a in have})
    rs = list(have)
    if len(have) >= 2:                                                # B: ensemble over widths
        for r in REG:
            D[f"R_ENSD|{r}"] = sum(D[f"R_{a}|{r}"] for a in have) / len(have)
        rs.append("ENSD")

    say("\n== level 2: fusion of l_F and l_R per R_x")
    cands, stkw = [], {}
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    for x in rs:
        for w in LIN:
            n = f"U_{x}_lin{int(round(100 * w))}"
            for r in REG:
                D[f"{n}|{r}"] = 2 * (w * D[f"F1s|{r}"] + (1 - w) * D[f"R_{x}|{r}"])
            cands.append(n)
        for w in PROB:
            n = f"U_{x}_prob{int(round(100 * w))}"
            for r in REG:
                D[f"{n}|{r}"] = w * sig(D[f"F1s|{r}"].values) + (1 - w) * sig(D[f"R_{x}|{r}"].values)
            cands.append(n)
        for k in STK:
            t0 = time.time(); n, wk = stack(D, S2, persons, ok, x, k); cands.append(n)
            if wk: stkw[x] = wk
            say(f"  {n:<18} [{time.time() - t0:.0f}s]")
    cands += [f"U_{a}_joint" for a in have]
    assert CTL in cands and len(set(cands)) == len(cands)
    say(f"  {len(cands)} candidates; stk2 learned F1s' share w (oof, per fold): {stkw}")
    summ["n_candidates"], summ["stk2_w"] = len(cands), stkw

    fam = ["F1s"] + [f"R_{x}" for x in rs] + cands
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    Tg = T[T.cell.str.endswith(" g")].groupby("fold")[fam].mean()
    show = ["F1s"] + [f"R_{x}" for x in rs] + [f"U_{x}_lin50" for x in rs]
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[show].round(2).T.to_string())
    summ["eer_cells"] = T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict()

    say("\n== A: width vs c0 (> 0 = width better): R level | U level (lin50)")
    perA = {}
    for a in [a for a in have if a != "c0"]:
        pr, hr = EDA.cmp(D, f"R_{a}", "R_c0"); pu, hu = EDA.cmp(D, f"U_{a}_lin50", CTL)
        say(f"  {a:<6} ({DIMS[a]:>4}) R {EDA.fmt(pr)} | held-out {hr}\n  {'':13} U {EDA.fmt(pu)} | held-out {hu}")
        perA[a] = dict(R=js(pr), hold_R=hr, U=js(pu), hold_U=hu)
    if "ENSD" in rs:
        pr, hr = EDA.cmp(D, "R_ENSD", "R_c0"); pu, hu = EDA.cmp(D, "U_ENSD_lin50", CTL)
        say(f"  ENSD        R {EDA.fmt(pr)} | held-out {hr}\n  {'':13} U {EDA.fmt(pu)} | held-out {hu}")
        perA["ENSD"] = dict(R=js(pr), hold_R=hr, U=js(pu), hold_U=hu)
    summ["A_width"] = perA

    say(f"\n== every candidate vs {CTL} on the g cells (> 0 = candidate better)")
    per = {}
    for n in [c for c in cands if c != CTL]:
        p, h = EDA.cmp(D, n, CTL)
        say(f"  {n:<18} {EDA.fmt(p)} | held-out {h}")
        per[n] = dict(js(p), hold=h)
    summ[f"vs_{CTL}"] = per

    def sel(name, pool):
        for r in REG:
            D[f"{name}|{r}"] = np.nan
        picks = {}
        for s in sorted(D.fold.unique()):
            e = Tg.loc[Tg.index != s, pool].mean(); pick = e.idxmin(); picks[int(s)] = pick
            m = (D.fold == s).values
            for r in REG:
                D.loc[m, f"{name}|{r}"] = D.loc[m, f"{pick}|{r}"]
        p, h = EDA.cmp(D, name, CTL)
        return picks, p, h

    say("\n== SEL: nested candidate choice over all candidates (outer fold s: lowest mean g-cell EER over the other folds)")
    picks, pa, ha = sel("SEL", cands)
    for s, pk in picks.items():
        say(f"  fold {s}: pick {pk}")
    lo, hi = S2.boot(D.rename(columns={f"{CTL}|oof": "ref_", "SEL|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
    ga = EDA.rule(pa, ha, thr=.3)
    final = Tg[cands].mean().idxmin()
    say(f"  SEL - {CTL}: {EDA.fmt(pa)} | held-out {ha} | CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}]")
    say(f"  best candidate over the 5 folds: {final} (mean g EER {Tg[final].mean():.2f}, {CTL} {Tg[CTL].mean():.2f})")
    say(f"  GATE: {'PASS -> step 3 (final = ' + final + ', Evaluation path)' if ga else 'FAIL -> keep FACTOR_ENFUSE, paper only'}"
        f" (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
    summ["SEL"] = dict(picks=picks, best_5fold=final, a=dict(js(pa), hold=ha, ci95=[round(lo, 3), round(hi, 3)]), passed=ga,
                       final=final if ga else None)

    say("\n== reported only: SEL_A (widths, lin50) and SEL_C (fusions on c0)")
    poolA = [f"U_{x}_lin50" for x in rs]
    poolC = [c for c in cands if c.startswith("U_c0_")]
    for nm, pool in (("SEL_A", poolA), ("SEL_C", poolC)):
        pk, p, h = sel(nm, pool)
        say(f"  {nm}: {EDA.fmt(p)} | held-out {h} | picks {pk}")
        summ[nm] = dict(js(p), hold=h, picks=pk)

    top = list(Tg[cands].mean().sort_values().index[:10])
    say("\n== reported only: the 10 best candidates over the 5 folds, EER per cell (oof, mean of folds)\n"
        + T.groupby("cell").mean(numeric_only=True)[[CTL] + [c for c in top if c != CTL]].round(2).T.to_string())
    summ["top10"] = top

    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
