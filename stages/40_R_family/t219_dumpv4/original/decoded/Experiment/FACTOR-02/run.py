"""FACTOR-02 (README registered 13/11 before any number): further tuning of the RFX family on SAFE-04. Kaggle CPU (FLAG_186):
    python lab_entry.py FACTOR-02 report      (lab data + T3S job tables + rfx<arm>lab_fold<s> of FLAG_183 + rfy<arm>lab_fold<s> of FLAG_185)
Candidates (README section 1): 10 single projectors (4 from FLAG_183, 5 from FLAG_185, ALLlo6 = mean of the ALLlo / ALLlor streams),
2 projector ensembles (ENS3, ENS10: mean of the fused R_x per regime), each with F1s' weight w in {0.35, 0.5, 0.65}:
U_{x,w} = 2 [w F1s + (1 - w) R_x]. Control U_{c5rl, 0.5}. SEL = nested choice on the g cells; gate SEL - control >= +0.3, >= 4/5 folds,
held-out Urdu / Hindi >= -0.2. FACTOR2_PHASE=1: FLAG_183 tables only, preliminary (no gate). Nothing reads dev / CB / Evaluation."""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


F = _load("factor01", EXPD / "FACTOR-01" / "run.py")
M = F.M
E, S6, table, OUT, KEY, say, FAST, NB = M.E, M.S6, M.table, M.OUT, M.KEY, M.say, M.FAST, M.NB
TAG = "factor2"
PHASE1 = os.environ.get("FACTOR2_PHASE", "") == "1"
X_ARMS = ["c5rl", "ALLlo", "F1lo", "ALL"]                            # FLAG_183 (prefix rfx)
Y_ARMS = ["c5rllo", "ALL52", "ALL41", "ALLslo", "ALLlor", "ALLlob2k"]  # FLAG_185 (prefix rfy) = kaggle/build_flag173.py ARMS6
SINGLES = X_ARMS + ([] if PHASE1 else ["c5rllo", "ALL52", "ALL41", "ALLslo", "ALLlo6", "ALLlob2k"])
ENS = {"ENS3": ["c5rl", "ALLlo", "F1lo"], "ENS10": list(SINGLES)}
WS = [0.35, 0.5, 0.65]
CTL = "U_c5rl_w50"
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")


def merge(CT):
    CT.merge_t3s()
    srcs = [("rfx", a) for a in X_ARMS] + ([] if PHASE1 else [("rfy", a) for a in Y_ARMS])
    got = [(p, a) for p, a in srcs if (OUT / f"{p}{a}lab_fold1.parquet").exists()]
    if not FAST:
        assert got == srcs, f"tables {got} != registered {srcs}"
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
    say(f"== {TAG} {'PHASE 1 (FLAG_183 tables only, preliminary, NO gate)' if PHASE1 else 'PHASE 2 (full, gate)'}: {len(D)} trials, "
        f"folds {sorted(int(s) for s in D.fold.unique())}, tables {have}")
    for a in have:
        for c in STR:
            D[f"{a}_{c}"] = D[f"{a}_{c}"].fillna(float(np.nanmean(D[f"{a}_{c}"].values)))   # MAV rows without an RF feature (as FACTOR-01)
    if "ALLlor" in have:                                              # E4: 6-seed projector = mean of the two 3-seed runs' streams
        for c in STR:
            D[f"ALLlo6_{c}"] = 0.5 * (D[f"ALLlo_{c}"] + D[f"ALLlor_{c}"])
    singles = [a for a in SINGLES if f"{a}_t3" in D.columns]
    groups = {f"trRFS_{a}": [f"{a}_{c}" for c in STR] for a in singles}
    ok = ~E.mav_flagged(D)
    for g, cs in groups.items():
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    summ = {"phase": 1 if PHASE1 else 2, "singles": singles, "ens": ENS, "ws": WS, "control": CTL}

    say("\n== fused families: F1s and R_x (CA-FUSE C = 1, nested + purged, 3 regimes)")
    EDA.run_arms(D, S2, persons, {"F1s": F1S, **{f"R_{a}": [f"trRFS_{a}"] for a in singles}})
    rs = list(singles)
    for e, mem in ENS.items():                                        # E1: projector ensembles (mean of fused R per regime)
        mem = [m for m in mem if m in singles]
        if len(mem) < 2: continue
        for r in REG:
            D[f"R_{e}|{r}"] = sum(D[f"R_{m}|{r}"] for m in mem) / len(mem)
        rs.append(e); say(f"  {e} = mean of {mem}")
    cands = []
    for x in rs:                                                      # E2: weight of F1s'
        for w in WS:
            n = f"U_{x}_w{int(round(100 * w))}"
            for r in REG:
                D[f"{n}|{r}"] = 2 * (w * D[f"F1s|{r}"] + (1 - w) * D[f"R_{x}|{r}"])
            cands.append(n)
    assert CTL in cands
    fam = ["F1s"] + [f"R_{x}" for x in rs] + cands
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    show = ["F1s"] + [f"R_{x}" for x in rs] + [f"U_{x}_w50" for x in rs]
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[show].round(2).T.to_string())
    Tg = T[T.cell.str.endswith(" g")].groupby("fold")[cands].mean()
    summ["eer_cells"] = T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict()

    say(f"\n== every candidate vs {CTL} on the g cells (> 0 = candidate better)")
    per = {}
    for n in [c for c in cands if c != CTL]:
        p, h = EDA.cmp(D, n, CTL)
        say(f"  {n:<18} {EDA.fmt(p)} | held-out {h}")
        per[n] = dict(js(p), hold=h)
    summ[f"vs_{CTL}"] = per

    say("\n== SEL: nested candidate choice (outer fold s: lowest mean g-cell EER over the other folds)")
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
    say(f"  best candidate over the 5 folds: {final} (mean g EER {Tg[final].mean():.2f}, {CTL} {Tg[CTL].mean():.2f})")
    if PHASE1:
        say("  PHASE 1: preliminary, the gate is read only in phase 2 (README section 2)")
    else:
        say(f"  GATE: {'PASS -> step 3 (final = ' + final + ', Evaluation path)' if ga else 'FAIL -> keep FACTOR_ENFUSE, paper only'}"
            f" (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
    summ["SEL"] = dict(picks=picks, best_5fold=final, a=dict(js(pa), hold=ha, ci95=[round(lo, 3), round(hi, 3)]),
                       passed=None if PHASE1 else ga, final=None if PHASE1 or not ga else final)
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
