"""RFSTUNE-01 (README registered 11/11 21:40 before any number): RFS projector recipe retuned on our clean store, SAFE-04 validation.
    python lab_entry.py RFSTUNE-01 report        (Kaggle CPU, FLAG_174; lab data + T3S job tables + rft<arm>lab_fold<s> in <SP>/out/safe04)
Same path as EDA-3FAM-01: ct01r2_fold<s> (CT.merge_t3s: F1s streams + trCA) + the RFS-LAB tables of every arm (FLAG_173).
U_<arm> = CA-FUSE(S6 + V11_T3 + trCA) + CA-FUSE(trRFS_<arm>), trRFS_<arm> = LR on [<prefix><arm>_t3, _nf, _nv] nested in the fuser split.
SEL = nested choice of the arm (outer fold s: lowest mean g-cell EER of U_<arm>|oof over the other 4 folds). Rule (README section 6):
SEL - U_<control> on g cells >= +0.3, >= 4/5 folds, held-out Urdu / Hindi >= -0.2. Nothing reads dev / CB / Evaluation.
Writes <TAG>_report.txt, <TAG>_summary.json next to the tables.
12/11: parameters as module globals (PREFIX, TAG, ARMS, CONTROL, NOISE, PAIRS, SEL_REFS) so RFSTUNE-02 reuses report(); the defaults
below are RFSTUNE-01 exactly (same arms, same comparisons, same file names)."""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent
sys.path.insert(0, str(EXPD / "E2E")); sys.path.insert(0, str(EXPD / "SAFE-04"))
import e2e_lib as E
from stream_gate import S6
from lab_fuse import SP, table

KEY = ["fold", "data", "protocol", "y", "fi", "vi"]
OUT = SP / "out/safe04"
TXT = []
FAST = os.environ.get("FL3Z_SMOKE") == "1"
NB = 20 if FAST else 300
# ---- parameters (RFSTUNE-01 defaults; RFSTUNE-02/run.py overrides them)
PREFIX, TAG = "rft", "rfstune"
ARMS = ["a0", "a0r", "s1", "s2", "t1", "t2", "d1", "d2", "l1", "l2", "p1", "p2", "w1", "w2", "wd"]   # = kaggle/build_flag173.py
CONTROL, NOISE, PAIRS, SEL_REFS = "a0", ("a0r", "a0"), [], ["a0"]   # PAIRS: extra (x, y) comparisons; SEL is compared with every SEL_REFS


def say(*a):
    s = " ".join(str(x) for x in a); TXT.append(s); print(s, flush=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def merge(CT):
    CT.merge_t3s()                                                    # 20 T3S job tables -> ct01r2_fold<s> (F1s streams + trCA)
    n = len(PREFIX)
    arms = sorted({p.name[n:].split("lab_fold")[0] for p in OUT.glob(f"{PREFIX}*lab_fold1.parquet")})
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for a in arms:
            P = pd.read_parquet(OUT / f"{PREFIX}{a}lab_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{PREFIX}{a} fold {s}: rows"
            for c in P.columns:
                if c not in KEY:
                    T[c] = P[c].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    say(f"merged {TAG}: arms {arms}")
    return arms


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py")              # run_arms / add_sum / cmp / fmt / rule (same definitions)
    EDA.TXT = TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    arms = merge(CT)
    if not FAST:
        assert arms == sorted(ARMS), f"arms in the tables {arms} != registered {sorted(ARMS)}"
    ctl = CONTROL if CONTROL in arms else arms[0]
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    say(f"== {TAG}: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}, arms {arms}, control {ctl}")
    cols = {a: [f"{PREFIX}{a}_t3", f"{PREFIX}{a}_nf", f"{PREFIX}{a}_nv"] for a in arms}
    for a, cs in cols.items():
        for c in cs:
            say(f"  coverage {c}: {100 * np.isfinite(D[c].values).mean():.1f} % finite") if a == arms[0] else None
            D[c] = D[c].fillna(float(np.nanmean(D[c].values)))       # MAV rows without an RF feature: column mean (as EDA-3FAM-01)
    groups = {f"trRFS_{a}": cs for a, cs in cols.items()}
    ok = ~E.mav_flagged(D)
    for g, cs in groups.items():                                      # standalone values (replaced inside every fuser split)
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    base = S6 + E.V11_T3
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    summ = {"arms": arms, "control": ctl}

    say("\n== fused arms (CA-FUSE C = 1, nested + purged, 3 regimes)")
    EDA.run_arms(D, S2, persons, {"F1s": base + ["trCA"], **{f"R_{a}": [f"trRFS_{a}"] for a in arms}})
    for a in arms:
        EDA.add_sum(D, f"U_{a}", ["F1s", f"R_{a}"])
    fam = ["F1s"] + [f"U_{a}" for a in arms] + [f"R_{a}" for a in arms]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam})
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[fam].round(2).T.to_string())
    Tg = T[T.cell.str.endswith(" g")].groupby("fold")[fam].mean()
    say("g-cell EER per fold:\n" + Tg.round(2).T.to_string())
    summ["eer_cells"] = T.groupby("cell").mean(numeric_only=True)[fam].round(3).to_dict()

    say(f"\n== each arm vs {ctl} (U level; > 0 = arm better), g cells")
    per = {}
    for a in [a for a in arms if a != ctl]:
        p, h = EDA.cmp(D, f"U_{a}", f"U_{ctl}"); per[a] = dict(g=round(float(p.mean()), 3), wins=int((p > 0).sum()), hold=h)
        say(f"  {a:<5} {EDA.fmt(p)} | held-out {h}")
    summ[f"vs_{ctl}"] = per
    if NOISE[0] in arms and NOISE[1] in arms:
        p, h = EDA.cmp(D, f"U_{NOISE[0]}", f"U_{NOISE[1]}")
        summ["noise_floor"] = dict(pair=list(NOISE), g=round(float(p.mean()), 3), abs_fold=round(float(p.abs().mean()), 3))
        say(f"  seed noise floor {NOISE[0]} - {NOISE[1]}: mean {p.mean():+.2f}, mean |fold| {p.abs().mean():.2f}")
    if PAIRS:
        say("\n== registered pairs (x - y, g cells; > 0 = x better)")
        summ["pairs"] = {}
        for x, y in PAIRS:
            if x in arms and y in arms:
                p, h = EDA.cmp(D, f"U_{x}", f"U_{y}")
                summ["pairs"][f"{x}-{y}"] = dict(g=round(float(p.mean()), 3), wins=int((p > 0).sum()), hold=h, folds=[round(float(v), 3) for v in p])
                say(f"  {x} - {y}: {EDA.fmt(p)} | held-out {h}")

    say("\n== SEL: nested arm choice (outer fold s: lowest mean g-cell EER of U_<arm>|oof over the other folds)")
    ua = [f"U_{a}" for a in arms]; picks = {}
    for r in ("oof", "hold_urdu", "hold_hindi"):
        D[f"SEL|{r}"] = np.nan
    for s in sorted(D.fold.unique()):
        e = Tg.loc[Tg.index != s, ua].mean(); pick = e.idxmin(); picks[int(s)] = pick[2:]
        m = (D.fold == s).values
        for r in ("oof", "hold_urdu", "hold_hindi"):
            D.loc[m, f"SEL|{r}"] = D.loc[m, f"{pick}|{r}"]
        say(f"  fold {int(s)}: pick {pick[2:]} (inner g EER {e.min():.2f}; {ctl} {e[f'U_{ctl}']:.2f})")
    final = Tg[ua].mean().idxmin()[2:]
    summ["SEL"] = dict(picks=picks, best_5fold=final)
    for ref in [r for r in SEL_REFS if r in arms]:
        p, h = EDA.cmp(D, "SEL", f"U_{ref}")
        lo, hi = S2.boot(D.rename(columns={f"U_{ref}|oof": "ref_", "SEL|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
        passed = EDA.rule(p, h, thr=.3)
        say(f"  SEL - U_{ref}: {EDA.fmt(p)} | held-out {h} | CI95 person bootstrap [{lo:+.2f}, {hi:+.2f}] -> "
            f"{'PASS' if passed else 'FAIL'} (>= +0.30, >= 4/5 folds, held-out >= -0.20)")
        summ["SEL"][f"vs_{ref}"] = dict(g=round(float(p.mean()), 3), wins=int((p > 0).sum()), hold=h, ci95=[round(lo, 3), round(hi, 3)], passed=passed)
    say(f"  best arm over the 5 folds: {final} (mean g EER {Tg[f'U_{final}'].mean():.2f}, {ctl} {Tg[f'U_{ctl}'].mean():.2f})")
    if ctl == "a0":                                                   # RFSTUNE-01 README section 6 wording
        v = summ["SEL"].get("vs_a0", {})
        summ["SEL"].update(g=v.get("g"), wins=v.get("wins"), hold=v.get("hold"), ci95=v.get("ci95"), passed=v.get("passed"),
                           final_arm=final if v.get("passed") else None)
    p2, h2 = EDA.cmp(D, f"U_{ctl}", "F1s")
    say(f"  (context) U_{ctl} - F1s: {EDA.fmt(p2)} | held-out {h2}")

    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
