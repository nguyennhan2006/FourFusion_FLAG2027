"""RFXSTD-01 (README registered 16/11 00:22 before any number): RFX projector training loop made standard, SAFE-04. Kaggle CPU (FLAG_221):
    python lab_entry.py RFXSTD-01 report    (lab data + T3S job tables + rfrc0r / rfrc0s of FLAG_213 + rfk<arm>lab_fold<s> of the two FLAG_220 runs)
Per replicate x in {r (seeds 4-6), s (seeds 7-9)}: U_a = F1s' + R_a (1:1), control U_c0x, arms cos / ema / ps / wd / dm / all. SEL_x = nested
choice on the g cells; gate SEL_x - U_c0x >= +0.3, >= 4/5 folds, held-out Urdu / Hindi >= -0.2. k0r = rerun of c0r (noise floor).
PASS = gate on both replicates, same best knob on both, gain above |k0r - c0r|. Nothing reads dev / CB / Evaluation."""
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
TAG = "rfxstd"
KNOBS = ["cos", "ema", "ps", "wd", "dm", "all"]                    # FLAG_220 = kaggle/build_flag173.py ARMS20
REPS = ["r", "s"]
SRCS = [("rfr", "c0r"), ("rfr", "c0s"), ("rfk", "k0r")] + [("rfk", k + x) for k in KNOBS for x in REPS]
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")


def js(p):
    return dict(mean=round(float(p.mean()), 3), wins=int((p > 0).sum()), folds=[round(float(v), 3) for v in p])


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    CT.merge_t3s()
    src = [(p, a) for p, a in SRCS if (OUT / f"{p}{a}lab_fold1.parquet").exists()]
    have = [a for _, a in src]
    if not FAST:
        assert src == SRCS, f"tables {have} != registered {[a for _, a in SRCS]}"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for p, a in src:
            P = pd.read_parquet(OUT / f"{p}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{a} fold {s}: rows"
            for c in STR:
                T[f"{a}_{c}"] = P[f"{p}{a}_{c}"].values
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

    noise = None
    if "k0r" in have and "c0r" in have:
        pu, hu = EDA.cmp(D, "U_k0r", "U_c0r"); pr, hr = EDA.cmp(D, "R_k0r", "R_c0r")
        noise = abs(float(pu.mean()))
        say(f"\n== noise floor: k0r (rerun of c0r, same seeds) - c0r on the g cells\n  U {EDA.fmt(pu)} | held-out {hu}\n  R {EDA.fmt(pr)} | held-out {hr}")
        summ["rerun"] = dict(U=js(pu), hold_U=hu, R=js(pr), hold_R=hr)

    res = {}
    for x in REPS:
        ctl = f"c0{x}"; arms = [k + x for k in KNOBS if k + x in have]
        if ctl not in have or not arms:
            continue
        say(f"\n== replicate {x} (control U_{ctl}): each arm vs control on the g cells (> 0 = arm better): U level | R level")
        per = {}
        for a in arms:
            pu, hu = EDA.cmp(D, f"U_{a}", f"U_{ctl}"); pr, hr = EDA.cmp(D, f"R_{a}", f"R_{ctl}")
            say(f"  {a:<6} U {EDA.fmt(pu)} | held-out {hu}\n  {'':6} R {EDA.fmt(pr)} | held-out {hr}")
            per[a] = dict(U=js(pu), hold_U=hu, R=js(pr), hold_R=hr)
        ua = [f"U_{ctl}"] + [f"U_{a}" for a in arms]; picks = {}; sel = f"SEL{x}"
        for r in REG:
            D[f"{sel}|{r}"] = np.nan
        for s in sorted(D.fold.unique()):
            e = Tg.loc[Tg.index != s, ua].mean(); pick = e.idxmin(); picks[int(s)] = pick
            m = (D.fold == s).values
            for r in REG:
                D.loc[m, f"{sel}|{r}"] = D.loc[m, f"{pick}|{r}"]
        pa, ha = EDA.cmp(D, sel, f"U_{ctl}")
        lo, hi = S2.boot(D.rename(columns={f"U_{ctl}|oof": "ref_", f"{sel}|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
        ga = bool(EDA.rule(pa, ha, thr=.3))
        best = Tg[ua].mean().idxmin(); knob = best[2:-1] if best != f"U_{ctl}" else "c0"
        gain = float(Tg[f"U_{ctl}"].mean() - Tg[best].mean())
        say(f"  SEL_{x} picks {picks}\n  SEL_{x} - U_{ctl}: {EDA.fmt(pa)} | held-out {ha} | CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}]")
        say(f"  best arm over the 5 folds: {best} (knob {knob}, gain {gain:+.2f}); gate {'PASS' if ga else 'FAIL'} (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
        res[x] = dict(vs_ctl=per, picks=picks, sel=dict(js(pa), hold=ha, ci95=[round(lo, 3), round(hi, 3)]), gate=ga, best=best, knob=knob, gain=round(gain, 3))
    summ["rep"] = res
    full = all(x in res for x in REPS)
    same = full and len({res[x]["knob"] for x in REPS}) == 1 and res[REPS[0]]["knob"] != "c0"
    above = full and noise is not None and min(res[x]["gain"] for x in REPS) > noise
    passed = bool(full and all(res[x]["gate"] for x in REPS) and same and above)
    say(f"\n== DECISION (README section 2): {'PASS -> report to the user (knob ' + res[REPS[0]]['knob'] + ')' if passed else 'FAIL -> keep the current RFX recipe and FACTOR_ENFUSE, paper only'}"
        f" (gate per replicate {({x: res[x]['gate'] for x in res})}, best knob per replicate {({x: res[x]['knob'] for x in res})}, same knob {bool(same)},"
        f" rerun noise |k0r - c0r| {noise if noise is None else round(noise, 3)}, above noise {bool(above)})")
    summ["decision"] = dict(passed=passed, same_knob=bool(same), noise=noise, above_noise=bool(above))
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
