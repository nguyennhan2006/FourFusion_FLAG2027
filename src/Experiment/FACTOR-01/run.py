"""FACTOR-01 (README registered 13/11 before any number): factors of the two families of U' (F1s', RFS), then the RFS projector
fed with F1s' encoder blocks (RFX). SAFE-04 validation only. Kaggle CPU (FLAG_184):
    python lab_entry.py FACTOR-01 report       (lab data + T3S job tables + rfx<arm>lab_fold<s> of FLAG_183 in <SP>/out/safe04)
Step 0 (atlas, description only): standalone EER of every stream, drop-one in the F1s fuser (F1s and U level), within-class
correlations. Step 2 (gate, README section 4): U_x = CA-FUSE(S6 + V11_T3 + trCA) + CA-FUSE(trRFS_x) as RFSTUNE-01; SEL = nested
arm choice on the g cells of U_x; (a) SEL - U_a0 >= +0.3, >= 4/5 folds, held-out Urdu / Hindi >= -0.2 and (b) R_SEL - R_a0 >= +0.5,
>= 4/5 folds (same picks, RFS family alone). Nothing reads dev / CB / Evaluation. Writes factor_report.txt, factor_summary.json."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


M = _load("rfstune01", EXPD / "RFSTUNE-01" / "run.py")              # sys.path (E2E, SAFE-04), say / TXT, OUT, KEY, FAST, NB
E, S6, table, OUT, KEY, say, FAST, NB = M.E, M.S6, M.table, M.OUT, M.KEY, M.say, M.FAST, M.NB
TAG, PFX = "factor", "rfx"
# = kaggle/build_flag173.py ARMS5 (registered, README section 3)
ARMS = ["a0", "ALL", "c5", "F1", "farl", "ALLlo", "pel", "F1lo", "rl", "ALLhi", "sf", "F1hi", "ibv", "c5rl", "c5x"]
CTL = "a0"
CELLS_G = ["en_ext g", "v4 g", "urdu g"]


def merge(CT):
    CT.merge_t3s()                                                    # 20 T3S job tables -> ct01r2_fold<s> (F1s streams + trCA)
    got = [a for a in ARMS if (OUT / f"{PFX}{a}lab_fold1.parquet").exists()]
    if not FAST:
        assert got == ARMS, f"arms in the tables {got} != registered {ARMS}"
    assert CTL in got, "no control table"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for a in got:
            P = pd.read_parquet(OUT / f"{PFX}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{PFX}{a} fold {s}: rows"
            for c in ("t3", "nf", "nv"):
                T[f"{PFX}{a}_{c}"] = P[f"{PFX}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    say(f"merged {TAG}: arms {got}")
    return got


def js(p):
    return dict(mean=round(float(p.mean()), 3), wins=int((p > 0).sum()), folds=[round(float(v), 3) for v in p])


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    arms = merge(CT)
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}, arms {arms}; F1s streams {F1S}")
    cols = {a: [f"{PFX}{a}_t3", f"{PFX}{a}_nf", f"{PFX}{a}_nv"] for a in arms}
    for a, cs in cols.items():
        for c in cs:
            if a == CTL:
                say(f"  coverage {c}: {100 * np.isfinite(D[c].values).mean():.1f} % finite")
            D[c] = D[c].fillna(float(np.nanmean(D[c].values)))       # MAV rows without an RF feature: column mean (as EDA-3FAM-01)
    groups = {f"trRFS_{a}": cs for a, cs in cols.items()}
    ok = ~E.mav_flagged(D)
    for g, cs in groups.items():                                      # standalone values (replaced inside every fuser split)
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    summ = {"arms": arms, "control": CTL, "f1s_streams": F1S}

    # ------------------------------------------------------------ step 0: atlas (description only)
    say("\n== STEP 0 atlas (a): standalone EER per stream (oof, mean of folds; '-' = the stream ranks reversed, shown as 100 - EER)")
    st = F1S + [f"trRFS_{CTL}", f"{PFX}{CTL}_t3", f"{PFX}{CTL}_nf", f"{PFX}{CTL}_nv"]
    fill = {c: float(np.nanmean(D[c].values)) for c in st}
    T0 = table(D, {c: (lambda g, c=c: g[c].fillna(fill[c]).values) for c in st})
    A0 = T0.groupby("cell").mean(numeric_only=True)[st]
    sign = {c: "-" if A0[c].mean() > 50 else "+" for c in st}
    A0 = A0.apply(lambda col: 100 - col if sign[col.name] == "-" else col)
    say(A0.round(2).T.assign(sign=pd.Series(sign)).to_string())
    summ["atlas_standalone"] = A0.round(3).to_dict()

    say("\n== STEP 0 atlas (b): drop one stream from the F1s fuser (CA-FUSE nested + purged), F1s level and U level (with R_a0)")
    drops = {f"F1s-{x}": [c for c in F1S if c != x] for x in F1S}
    EDA.run_arms(D, S2, persons, {"F1s": F1S, f"R_{CTL}": [f"trRFS_{CTL}"], **drops})
    EDA.add_sum(D, f"U_{CTL}", ["F1s", f"R_{CTL}"])
    dsum = {}
    for x in F1S:
        EDA.add_sum(D, f"U-{x}", [f"F1s-{x}", f"R_{CTL}"])
        pf, hf = EDA.cmp(D, f"F1s-{x}", "F1s"); pu, hu = EDA.cmp(D, f"U-{x}", f"U_{CTL}")
        say(f"  without {x:<12} F1s g {EDA.fmt(pf)}\n  {'':20s} U   g {EDA.fmt(pu)} | held-out {hu}   (< 0 = the stream helps)")
        dsum[x] = dict(F1s=js(pf), U=js(pu), hold_U=hu)
    summ["atlas_drop_one"] = dsum

    say("\n== STEP 0 atlas (c): within-class Spearman on the g cells (mean over folds x cells x classes)")
    Rm = EDA.wcorr(D, F1S + [f"trRFS_{CTL}"], gc)
    say(Rm.round(2).to_string()); summ["atlas_wcorr_g"] = Rm.round(3).to_dict()

    # ------------------------------------------------------------ step 2: arms, SEL, gate
    say("\n== STEP 2 fused arms (CA-FUSE C = 1, nested + purged, 3 regimes)")
    EDA.run_arms(D, S2, persons, {f"R_{a}": [f"trRFS_{a}"] for a in arms if a != CTL})
    for a in arms:
        EDA.add_sum(D, f"U_{a}", ["F1s", f"R_{a}"])
    fam = ["F1s"] + [f"U_{a}" for a in arms] + [f"R_{a}" for a in arms]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[fam].round(2).T.to_string())
    Tg = T[T.cell.str.endswith(" g")].groupby("fold")[fam].mean()
    summ["eer_cells"] = T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict()

    say(f"\n== each arm vs {CTL} on the g cells (> 0 = arm better): U level | R level (RFS family alone)")
    per = {}
    for a in [a for a in arms if a != CTL]:
        pu, hu = EDA.cmp(D, f"U_{a}", f"U_{CTL}"); pr, hr = EDA.cmp(D, f"R_{a}", f"R_{CTL}")
        say(f"  {a:<6} U {EDA.fmt(pu)} | held-out {hu}\n  {'':6} R {EDA.fmt(pr)} | held-out {hr}")
        per[a] = dict(U=js(pu), hold_U=hu, R=js(pr), hold_R=hr)
    summ[f"vs_{CTL}"] = per

    say("\n== SEL: nested arm choice (outer fold s: lowest mean g-cell EER of U_<arm>|oof over the other folds)")
    ua = [f"U_{a}" for a in arms]; picks = {}
    for lvl in ("SEL", "SELR"):
        for r in ("oof", "hold_urdu", "hold_hindi"):
            D[f"{lvl}|{r}"] = np.nan
    for s in sorted(D.fold.unique()):
        e = Tg.loc[Tg.index != s, ua].mean(); pick = e.idxmin()[2:]; picks[int(s)] = pick
        m = (D.fold == s).values
        for r in ("oof", "hold_urdu", "hold_hindi"):
            D.loc[m, f"SEL|{r}"] = D.loc[m, f"U_{pick}|{r}"]; D.loc[m, f"SELR|{r}"] = D.loc[m, f"R_{pick}|{r}"]
        say(f"  fold {int(s)}: pick {pick} (inner g EER {e.min():.2f}; {CTL} {e[f'U_{CTL}']:.2f})")
    pa, ha = EDA.cmp(D, "SEL", f"U_{CTL}"); pb, hb = EDA.cmp(D, "SELR", f"R_{CTL}")
    lo, hi = S2.boot(D.rename(columns={f"U_{CTL}|oof": "ref_", "SEL|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
    ga = EDA.rule(pa, ha, thr=.3)
    gb = bool(pb.mean() >= .5 and (pb > 0).sum() >= 4)
    say(f"  (a) SEL - U_{CTL}: {EDA.fmt(pa)} | held-out {ha} | CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}] -> {'PASS' if ga else 'FAIL'}"
        f" (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
    say(f"  (b) R_SEL - R_{CTL}: {EDA.fmt(pb)} | held-out {hb} -> {'PASS' if gb else 'FAIL'} (>= +0.50, >= 4/5 folds)")
    final = Tg[ua].mean().idxmin()[2:]
    passed = ga and gb
    say(f"  best arm over the 5 folds: {final} (mean g EER {Tg[f'U_{final}'].mean():.2f}, {CTL} {Tg[f'U_{CTL}'].mean():.2f})")
    say(f"  GATE (a and b): {'PASS -> step 3 (final RFX = ' + final + ', Evaluation path)' if passed else 'FAIL -> close, paper only'}")
    summ["SEL"] = dict(picks=picks, best_5fold=final, a=dict(js(pa), hold=ha, ci95=[round(lo, 3), round(hi, 3)], passed=ga),
                       b=dict(js(pb), hold=hb, passed=gb), passed=passed, final_arm=final if passed else None)
    p2, h2 = EDA.cmp(D, f"U_{CTL}", "F1s")
    say(f"  (context) U_{CTL} - F1s: {EDA.fmt(p2)} | held-out {h2}")

    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
