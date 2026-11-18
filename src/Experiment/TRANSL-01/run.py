"""TRANSL-01 (README registered 14/11 before any number): "translator" knobs on the RFX c5rl projector, SAFE-04. Kaggle CPU (FLAG_188):
    python lab_entry.py TRANSL-01 report      (lab data + T3S job tables + rfxc5rl lab tables of FLAG_183 + rfz<arm>lab_fold<s> of FLAG_187)
Candidates (README section 2): U_x = 2 [0.5 F1s + 0.5 R_x], x in {c0, orth1, orth2, cyc1, cyc2, mpos} (R_x = CA-FUSE of the arm's
t3 / nf / nv, nested + purged). Control U_c0 (c5rl re-run in FLAG_187). SEL = nested choice on the g cells; gate SEL - U_c0 >= +0.3,
>= 4/5 folds, held-out Urdu / Hindi >= -0.2. U_c5rl (FLAG_183 table) only as a GPU-run sensitivity line. Nothing reads dev / CB / Evaluation."""
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
TAG = "transl"
Z_ARMS = ["c0", "orth1", "orth2", "cyc1", "cyc2", "mpos"]          # FLAG_187 (prefix rfz) = kaggle/build_flag173.py ARMS7
SRCS = [("rfx", "c5rl")] + [("rfz", a) for a in Z_ARMS]
CTL = "U_c0"
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")


def merge(CT):
    CT.merge_t3s()
    got = [(p, a) for p, a in SRCS if (OUT / f"{p}{a}lab_fold1.parquet").exists()]
    if not FAST:   # rfxc5rl (FLAG_183) is optional: sensitivity line only, readable only from the account of the 183 runs
        assert [x for x in got if x[0] == "rfz"] == SRCS[1:], f"tables {got} != registered {SRCS[1:]}"
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


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    have = merge(CT)
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}, tables {have}")
    for a in have:
        for c in STR:
            D[f"{a}_{c}"] = D[f"{a}_{c}"].fillna(float(np.nanmean(D[f"{a}_{c}"].values)))   # MAV rows without an RF feature (as FACTOR-01)
    groups = {f"trRFS_{a}": [f"{a}_{c}" for c in STR] for a in have}
    ok = ~E.mav_flagged(D)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    summ = {"arms": have, "control": CTL}

    say("\n== fused families: F1s and R_x (CA-FUSE C = 1, nested + purged, 3 regimes)")
    EDA.run_arms(D, S2, persons, {"F1s": F1S, **{f"R_{a}": [f"trRFS_{a}"] for a in have}})
    for a in have:
        for r in REG:
            D[f"U_{a}|{r}"] = 2 * (0.5 * D[f"F1s|{r}"] + 0.5 * D[f"R_{a}|{r}"])
    cands = [f"U_{a}" for a in have if a in Z_ARMS]
    assert CTL in cands
    fam = ["F1s"] + [f"R_{a}" for a in have] + [f"U_{a}" for a in have]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[fam].round(2).T.to_string())
    Tg = T[T.cell.str.endswith(" g")].groupby("fold")[cands].mean()
    summ["eer_cells"] = T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict()

    say(f"\n== every arm vs the control on the g cells (> 0 = arm better); U level, then R level (R_c0)")
    per = {}
    for a in [x for x in have if x != "c0"]:
        p, h = EDA.cmp(D, f"U_{a}", CTL); pr, hr = EDA.cmp(D, f"R_{a}", "R_c0")
        say(f"  U_{a:<7} {EDA.fmt(p)} | held-out {h}    R_{a:<7} {EDA.fmt(pr)} | held-out {hr}")
        per[a] = dict(U=dict(js(p), hold=h), R=dict(js(pr), hold=hr))
    summ["vs_control"] = per
    if "c5rl" in have:
        p, h = EDA.cmp(D, "U_c5rl", CTL)
        say(f"  sensitivity, U_c5rl (FLAG_183) - U_c0 (FLAG_187): {EDA.fmt(p)} | held-out {h}")
        summ["c5rl183_minus_c0"] = dict(js(p), hold=h)

    say("\n== SEL: nested arm choice (outer fold s: lowest mean g-cell EER over the other folds)")
    picks = {}
    for r in REG:
        D[f"SEL|{r}"] = np.nan
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
    say(f"  best arm over the 5 folds: {final} (mean g EER {Tg[final].mean():.2f}, {CTL} {Tg[CTL].mean():.2f})")
    say(f"  GATE: {'PASS -> step 3 (final = ' + final + ', Evaluation path, zip frozen, not submitted)' if ga else 'FAIL -> keep FACTOR_ENFUSE, paper only'}"
        f" (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
    summ["SEL"] = dict(picks=picks, best_5fold=final, a=dict(js(pa), hold=ha, ci95=[round(lo, 3), round(hi, 3)]), passed=ga,
                       final=final if ga else None)
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
