"""FUSEDIM-02 (README registered 15/11 before any number): fusions that keep the small-width gain of FUSEDIM-01, SAFE-04.
Kaggle CPU (FLAG_199):
    python lab_entry.py FUSEDIM-02 report   (lab data + T3S job tables + rfd{c0,d128,d256} of FLAG_195 + rfm{m512,m256,m128} of FLAG_198)
Candidates (README section 1B): mw (c0 main member + d128 / d256 side member), RM (one RFX fuser over three widths), stk4 (learned
weights over l_F, R_c0, R_d128, R_d256 on the other outer folds' oof scores), JM (one-level CA-FUSE), Matryoshka projectors m512 /
m256 / m128 (lin50 / lin60), mrl2 (m512 + m128 views), mwm128 (c0 + m128). Control U_c0_lin50. SEL = nested choice on the g cells;
gate SEL - control >= +0.3, >= 4/5 folds, held-out Urdu / Hindi >= -0.2. Hypothesis H (within-class correlation with F1s) and the
R level of the Matryoshka arms are reported only. Nothing reads dev / CB / Evaluation. Writes fusedim2_report.txt / _summary.json."""
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
TAG = "fusedim2"
SRCS = [("rfd", "c0"), ("rfd", "d128"), ("rfd", "d256"), ("rfm", "m512"), ("rfm", "m256"), ("rfm", "m128")]  # FLAG_195 / FLAG_198
CTL = "U_c0_lin50"
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")


def merge(CT):
    CT.merge_t3s()
    got = [(p, a) for p, a in SRCS if (OUT / f"{p}{a}lab_fold1.parquet").exists()]
    if not FAST:
        assert got == SRCS, f"tables {got} != registered {SRCS}"
    assert ("rfd", "c0") in got, "no control table"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for p, a in got:
            P = pd.read_parquet(OUT / f"{p}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{p}{a} fold {s}: rows"
            for c in STR:
                T[f"{a}_{c}"] = P[f"{p}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    say(f"merged {TAG}: {got}")
    return [a for _, a in got]


def js(p):
    return dict(mean=round(float(p.mean()), 3), wins=int((p > 0).sum()), folds=[round(float(v), 3) for v in p])


def stack(D, S2, persons, ok, name, members):
    """Second-level logistic fuser over the fused members (README section 1B stk4): outer fold s learns on the oof scores of the
    other folds (purged of fold s persons, MAV-flagged rows dropped; held-out regimes also drop the held language)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for r in REG:
        D[f"{name}|{r}"] = np.nan
    W = {}
    for s in sorted(D.fold.unique()):
        base = S2.purge(D[(D.fold != s).values & ok], persons[int(s)])
        for r, held in S2.HELD.items():
            tr = base[~base.data.isin(held)] if held else base
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
            m.fit(tr[[f"{c}|oof" for c in members]].values, tr.y.values)
            te = D[(D.fold == s) & (D.data.isin(held) if held else True)]
            D.loc[te.index, f"{name}|{r}"] = m.decision_function(te[[f"{c}|{r}" for c in members]].values)
            if r == "oof":
                c = m[-1].coef_[0] / m[0].scale_
                W[int(s)] = [round(float(x / c.sum()), 3) for x in c]
    return W


def lin(D, name, parts):
    """D[name|r] = sum_i w_i D[member_i|r] (weights already include the factor 2 of the U sum)."""
    for r in REG:
        D[f"{name}|{r}"] = sum(w * D[f"{c}|{r}"] for c, w in parts)


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    have = merge(CT)
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}, tables {have}; F1s streams {F1S}")
    for a in have:
        for c in STR:
            col = f"{a}_{c}"
            if a in ("c0", "m512"):
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
    summ = {"tables": have, "control": CTL}
    W3 = [w for w in ("c0", "d128", "d256") if w in have]

    say("\n== level 1: F1s, R_x, R_M (one RFX fuser over the widths), JM (one-level CA-FUSE)")
    EDA.run_arms(D, S2, persons, {"F1s": F1S, **{f"R_{a}": [f"trRFS_{a}"] for a in have}})
    if len(W3) == 3:
        EDA.run_arms(D, S2, persons, {"R_M": [f"trRFS_{a}" for a in W3], "U_JM": F1S + [f"trRFS_{a}" for a in W3]})

    say("\n== level 2: candidates")
    cands = []
    lin(D, CTL, [("F1s", 1.0), ("R_c0", 1.0)]); cands.append(CTL)
    for x in [w for w in ("d128", "d256") if w in have]:
        for a in (1 / 3, 1 / 2):
            for w in (0.5, 0.6):
                n = f"U_mw{x}_a{int(round(100 * a))}_w{int(round(100 * w))}"
                lin(D, n, [("F1s", 2 * w), ("R_c0", 2 * (1 - w) * (1 - a)), (f"R_{x}", 2 * (1 - w) * a)]); cands.append(n)
    if len(W3) == 3:
        for w in (0.5, 0.6):
            n = f"U_RM_w{int(round(100 * w))}"; lin(D, n, [("F1s", 2 * w), ("R_M", 2 * (1 - w))]); cands.append(n)
        W4 = stack(D, S2, persons, ok, "U_stk4", ["F1s"] + [f"R_{a}" for a in W3]); cands.append("U_stk4")
        cands.append("U_JM")
        say(f"  stk4 learned weights [F1s, R_c0, R_d128, R_d256] (share of the sum, oof, per fold): {W4}")
        summ["stk4_w"] = W4
    for k in [k for k in ("m512", "m256", "m128") if k in have]:
        for w in (0.5, 0.6):
            n = f"U_{k}_lin{int(round(100 * w))}"; lin(D, n, [("F1s", 2 * w), (f"R_{k}", 2 * (1 - w))]); cands.append(n)
    if "m512" in have and "m128" in have:
        lin(D, "U_mrl2", [("F1s", 1.0), ("R_m512", 0.5), ("R_m128", 0.5)]); cands.append("U_mrl2")
    if "m128" in have:
        lin(D, "U_mwm128", [("F1s", 1.0), ("R_c0", 0.5), ("R_m128", 0.5)]); cands.append("U_mwm128")
    assert len(set(cands)) == len(cands)
    say(f"  {len(cands)} candidates: {cands}")
    summ["candidates"] = cands

    Rs = [f"R_{a}" for a in have] + (["R_M"] if "R_M|oof" in D.columns else [])
    fam = ["F1s"] + Rs + cands
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    Tg = T[T.cell.str.endswith(" g")].groupby("fold")[fam].mean()
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[fam].round(2).T.to_string())
    summ["eer_cells"] = T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict()

    say("\n== reported only: hypothesis H, within-class Spearman with F1s on the g cells (mean over folds x cells x classes)")
    wc = EDA.wcorr(D, ["F1s|oof"] + [f"{r}|oof" for r in Rs], gc)
    say(wc.round(3).to_string()); summ["wcorr_g"] = wc.round(4).to_dict()

    say("\n== reported only: R level of the widths / Matryoshka arms (> 0 = row better)")
    perR = {}
    for a in [a for a in have if a != "c0"] + (["M"] if "R_M|oof" in D.columns else []):
        p, h = EDA.cmp(D, f"R_{a}", "R_c0"); say(f"  R_{a:<5} vs R_c0   {EDA.fmt(p)} | held-out {h}"); perR[f"{a}_vs_c0"] = dict(js(p), hold=h)
    for k, ref in (("m128", "d128"), ("m256", "d256")):
        if k in have and ref in have:
            p, h = EDA.cmp(D, f"R_{k}", f"R_{ref}"); say(f"  R_{k:<5} vs R_{ref} {EDA.fmt(p)} | held-out {h}"); perR[f"{k}_vs_{ref}"] = dict(js(p), hold=h)
    summ["R_level"] = perR

    say(f"\n== every candidate vs {CTL} on the g cells (> 0 = candidate better)")
    per = {}
    for n in [c for c in cands if c != CTL]:
        p, h = EDA.cmp(D, n, CTL)
        say(f"  {n:<18} {EDA.fmt(p)} | held-out {h}")
        per[n] = dict(js(p), hold=h)
    summ[f"vs_{CTL}"] = per

    say("\n== SEL: nested candidate choice (outer fold s: lowest mean g-cell EER over the other folds)")
    for r in REG:
        D[f"SEL|{r}"] = np.nan
    picks = {}
    for s in sorted(D.fold.unique()):
        e = Tg.loc[Tg.index != s, cands].mean(); pick = e.idxmin(); picks[int(s)] = pick
        m = (D.fold == s).values
        for r in REG:
            D.loc[m, f"SEL|{r}"] = D.loc[m, f"{pick}|{r}"]
        say(f"  fold {int(s)}: pick {pick} (inner g EER {e.min():.2f}; {CTL} {e[CTL]:.2f})")
    pa, ha = EDA.cmp(D, "SEL", CTL)
    lo, hi = S2.boot(D.rename(columns={f"{CTL}|oof": "ref_", "SEL|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
    ga = EDA.rule(pa, ha, thr=.3)
    final = Tg[cands].mean().idxmin()
    say(f"  SEL - {CTL}: {EDA.fmt(pa)} | held-out {ha} | CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}]")
    say(f"  best candidate over the 5 folds: {final} (mean g EER {Tg[final].mean():.2f}, {CTL} {Tg[CTL].mean():.2f})")
    say(f"  GATE: {'PASS -> step 3 (final = ' + final + ', Evaluation path)' if ga else 'FAIL -> keep FACTOR_ENFUSE, paper only'}"
        f" (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
    summ["SEL"] = dict(picks=picks, best_5fold=final, a=dict(js(pa), hold=ha, ci95=[round(lo, 3), round(hi, 3)]), passed=ga,
                       final=final if ga else None)

    top = list(Tg[cands].mean().sort_values().index[:10])
    say("\n== reported only: the 10 best candidates over the 5 folds, EER per cell (oof, mean of folds)\n"
        + T.groupby("cell").mean(numeric_only=True)[[CTL] + [c for c in top if c != CTL]].round(2).T.to_string())
    summ["top10"] = top

    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
