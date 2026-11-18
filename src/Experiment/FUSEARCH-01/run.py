"""FUSEARCH-01 (README registered 15/11 before any number): fusion structure of U' on SAFE-04: protocol-specific fusers, one fuser on
the raw streams of every system (C 1 / 0.1 / 0.01), several RFX systems. Kaggle CPU (FLAG_212):
    python lab_entry.py FUSEARCH-01 report   (lab data + T3S job tables + rfd* of FLAG_195, rfm* of FLAG_198, rfl* of FLAG_207)
Control U_c0 = F1s' + R_c0. SEL = nested choice on the g cells over the 12 candidates; gate SEL - U_c0 >= +0.3, >= 4/5 folds,
held-out Urdu / Hindi >= -0.2. The protocol of a file is known from its folder (gender/ | no_gender/), like the file language.
Nothing reads dev / CB / Evaluation. Writes fusearch_report.txt / fusearch_summary.json."""
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
TAG = "fusearch"
SRCS = ([("rfd", a) for a in ("c0", "d64", "d128", "d256", "d1024", "d2048")] + [("rfm", a) for a in ("m512", "m256", "m128")]
        + [("rfl", a) for a in ("L1", "L2")])
JM4 = ["c0", "d256", "m256", "L1"]
WP = [0.4, 0.5, 0.6, 0.7]
CTL = "U_c0"
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")


def js(p):
    return dict(mean=round(float(p.mean()), 3), wins=int((p > 0).sum()), folds=[round(float(v), 3) for v in p])


def fuse(D, S2, persons, name, st, C=1.0, split=False):
    """CA-FUSE(st) nested per fold, purged, 3 regimes (= EDA run_arms) with regulariser C; split=True: pool and test rows of the same
    protocol only (one fuser per protocol)."""
    t0 = time.time()
    for regime, held in S2.HELD.items():
        col = f"{name}|{regime}"; D[col] = np.nan
        for s in sorted(D.fold.unique()):
            for pg in ((True, False) if split else (None,)):
                mt = (D.fold == s) & (D.data.isin(held) if held else True)
                mp = (D.fold != s) & ~D.data.isin(held)
                if pg is not None:
                    mt, mp = mt & (D.is_gp == pg), mp & (D.is_gp == pg)
                te = D[mt]
                if len(te):
                    D.loc[te.index, col] = S2.Design(S2.purge(D[mp], persons[int(s)]), te, st).score(st, C)
    say(f"  {name:<10} {len(st)} streams, C {C}, {'per protocol' if split else 'pooled'} [{time.time() - t0:.0f}s]")


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
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
    have = [a for _, a in got]
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    D["is_gp"] = D.cell.str.endswith(" g")
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials ({int(D.is_gp.sum())} gender-protocol), RFX tables {have}")
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

    say("\n== level 1 fusers")
    fuse(D, S2, persons, "F1s", F1S)
    for a in have:
        fuse(D, S2, persons, f"R_{a}", [f"trRFS_{a}"])
    fuse(D, S2, persons, "F1s_ps", F1S, split=True)
    fuse(D, S2, persons, "R_c0_ps", ["trRFS_c0"], split=True)
    raw = F1S + ["c0_t3", "c0_nf", "c0_nv"]
    for C, nm in ((1.0, "U_J1"), (0.1, "U_J01"), (0.01, "U_J001")):
        fuse(D, S2, persons, nm, raw, C=C)
    fuse(D, S2, persons, "U_J01ps", raw, C=0.1, split=True)
    jm = [a for a in JM4 if a in have]
    fuse(D, S2, persons, "U_JM", F1S + [f"trRFS_{a}" for a in jm], C=0.1)
    fuse(D, S2, persons, "U_JMps", F1S + [f"trRFS_{a}" for a in jm], C=0.1, split=True)

    def lin(name, parts):
        for r in REG:
            D[f"{name}|{r}"] = sum(w * D[f"{c}|{r}"] for c, w in parts)
    lin(CTL, [("F1s", 1.0), ("R_c0", 1.0)])
    lin("U_ps", [("F1s_ps", 1.0), ("R_c0_ps", 1.0)])
    lin("U_psF", [("F1s_ps", 1.0), ("R_c0", 1.0)])
    lin("U_psR", [("F1s", 1.0), ("R_c0_ps", 1.0)])
    lin("U_M", [("F1s", 1.0)] + [(f"R_{a}", 1.0 / len(have)) for a in have])
    for w in WP:
        lin(f"W{int(100 * w)}", [("F1s", 2 * w), ("R_c0", 2 * (1 - w))])
    wc = [f"W{int(100 * w)}" for w in WP]
    Tw = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in wc})
    Twg, Twn = (Tw[Tw.cell.str.endswith(suf)].groupby("fold")[wc].mean() for suf in (" g", " ng"))
    for r in REG:
        D[f"U_pw|{r}"] = np.nan
    wpick = {}
    for s in sorted(D.fold.unique()):
        pg, pn = Twg.loc[Twg.index != s].mean().idxmin(), Twn.loc[Twn.index != s].mean().idxmin(); wpick[int(s)] = (pg, pn)
        for r in REG:
            for flag, pk in ((True, pg), (False, pn)):
                m = ((D.fold == s) & (D.is_gp == flag)).values
                D.loc[m, f"U_pw|{r}"] = D.loc[m, f"{pk}|{r}"]
    say(f"  U_pw: per-protocol F1s' share (g, ng) per outer fold {wpick}")

    cands = [CTL, "U_ps", "U_psF", "U_psR", "U_pw", "U_J1", "U_J01", "U_J001", "U_J01ps", "U_M", "U_JM", "U_JMps"]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in ["F1s", "F1s_ps", "R_c0", "R_c0_ps"] + cands})
    Tg = T[T.cell.str.endswith(" g")].groupby("fold")[cands].mean()
    Tn = T[T.cell.str.endswith(" ng")].groupby("fold")[cands].mean()
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True).round(2).T.to_string())
    summ = {"tables": have, "jm": jm, "w_picks": wpick, "eer_cells": T.groupby("cell").mean(numeric_only=True).round(3).to_dict()}

    say(f"\n== every candidate vs {CTL}: g cells (gate statistic) | ng cells mean (> 0 = candidate better)")
    per = {}
    for n in cands[1:]:
        p, h = EDA.cmp(D, n, CTL); dn = (Tn[CTL] - Tn[n]).values
        say(f"  {n:<9} g {EDA.fmt(p)} | held-out {h} | ng {dn.mean():+.2f} ({(dn > 0).sum()}/5)")
        per[n] = dict(js(p), hold=h, ng_mean=round(float(dn.mean()), 3))
    summ[f"vs_{CTL}"] = per

    say("\n== reported only (README 2b): LB4 proxy of the 4 test files, minDCF (p 0.05) on the g cells, worst g cell (lower = better)")
    from sklearn.metrics import roc_curve
    def mindcf(y, s, p=0.05):
        fpr, tpr, _ = roc_curve(y, s); return float(np.min(p * (1 - tpr) + (1 - p) * fpr) / min(p, 1 - p))
    piv = T.set_index(["fold", "cell"])
    def lb4(c):
        x = piv[c].unstack("cell")
        return (((x["v4 ng"] + x["en_ext ng"]) / 2 + (x["v4 g"] + x["en_ext g"]) / 2 + (x["urdu ng"] + x["hindi ng"]) / 2 + x["urdu g"]) / 4).values
    extra = {}
    for c in cands:
        dcf = [np.mean([mindcf(g.y.values, g[f"{c}|oof"].values) for _, g in D[(D.fold == s) & D.cell.isin(gc)].groupby("cell")]) for s in sorted(D.fold.unique())]
        extra[c] = dict(LB4=lb4(c), DCF=np.array(dcf), WORST=piv[c].unstack("cell")[gc].max(1).values)
    rows = []
    for c in cands:
        r = {k: round(float(v.mean()), 3) for k, v in extra[c].items()}
        if c != CTL:
            for k in extra[c]:
                d = extra[CTL][k] - extra[c][k]; r[f"d{k}"] = f"{d.mean():+.3f} ({(d > 0).sum()}/5)"
        rows.append(pd.Series(r, name=c))
    say(pd.DataFrame(rows).to_string())
    summ["reported_metrics"] = {c: {k: [round(float(x), 4) for x in v] for k, v in extra[c].items()} for c in cands}

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
    say(f"  GATE: {'PASS -> Evaluation zip (final = ' + final + ')' if ga else 'FAIL -> keep FACTOR_ENFUSE, paper only'}"
        f" (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
    summ["SEL"] = dict(picks=picks, best_5fold=final, a=dict(js(pa), hold=ha, ci95=[round(lo, 3), round(hi, 3)]), passed=ga)
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
