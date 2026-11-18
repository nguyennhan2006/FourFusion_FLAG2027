"""ENPROG-01 (README registered 12/11 before any number of this program): five English-oriented experiments, SAFE-04 validation only.
    python lab_entry.py ENPROG-01 report      (Kaggle CPU, FLAG_178; lab data + T3S job tables + rf{e,t,u}<arm>lab_fold<s> in <SP>/out/safe04)
E1 ENCABL-01 encoder-block ablation, E2 MVIEW-01 per-encoder experts fused by CA-FUSE, E3 ENWT-01 English weight of F1s : RFS,
E4 ENLANG-01 English-specific arm choice, E5 ENCORR-01 correlations. Same path as RFSTUNE-01 (F1s = CA-FUSE(S6 + V11_T3 + trCA),
R_x = CA-FUSE(trRFS_x), U_x = F1s + R_x). Nothing reads dev / CB / Evaluation. Writes enprog_report.txt, enprog_summary.json."""
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
TAG = "enprog"
# registered arms of the three lab rounds (= kaggle/build_flag173.py ARMS3, ARMS, ARMS2)
ROUNDS = {"rfe": ["a0", "a0r", "zwh", "zsam", "zr221", "zxv", "zclip", "zibf", "osam", "or221", "oxv", "owh"],
          "rft": ["a0", "a0r", "s1", "s2", "t1", "t2", "d1", "d2", "l1", "l2", "p1", "p2", "w1", "w2", "wd"],
          "rfu": ["a0", "d1", "d1r", "f6v5", "f8v3", "f7v4", "f5v2", "f4v1", "d1p1", "d1s2", "d1l1"]}
CTL = "rfea0"
LOBO = ["zwh", "zsam", "zr221", "zxv", "zclip", "zibf"]
EXPERTS = ["osam", "or221", "oxv", "owh", "zclip", "zibf"]
WGRID = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
ENG, ENNG, OTH = ["en_ext g"], ["en_ext ng"], ["v4 g", "urdu g"]
EN = ENG + ENNG


def merge(CT):
    CT.merge_t3s()                                                    # 20 T3S job tables -> ct01r2_fold<s> (F1s streams + trCA)
    have = {}
    for pfx, arms in ROUNDS.items():
        got = [a for a in arms if (OUT / f"{pfx}{a}lab_fold1.parquet").exists()]
        if not FAST:
            assert got == arms, f"{pfx}: arms in the tables {got} != registered {arms}"
        if got:
            have[pfx] = got
    assert "rfe" in have, "no rfe tables"
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for pfx, arms in have.items():
            for a in arms:
                P = pd.read_parquet(OUT / f"{pfx}{a}lab_fold{s}.parquet")
                assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{pfx}{a} fold {s}: rows"
                for c in ("t3", "nf", "nv"):
                    T[f"{pfx}{a}_{c}"] = P[f"{pfx}{a}_{c}"].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    say(f"merged {TAG}: " + "; ".join(f"{p} {a}" for p, a in have.items()))
    return [f"{p}{a}" for p, arms in have.items() for a in arms]


def dl(T, a, b, cells):
    """per fold, mean over `cells` of EER(b) - EER(a) (oof table T; > 0 = a better)."""
    t = T[T.cell.isin(cells)]
    return (t[b] - t[a]).groupby(t.fold).mean()


def fmt(p):
    return ", ".join(f"f{int(s)} {g:+.2f}" for s, g in p.items()) + f" | mean {p.mean():+.2f}, {int((p > 0).sum())}/5"


def en_rule(pg, png):
    return bool(pg.mean() >= .5 and (pg > 0).sum() >= 4 and png.mean() >= -.2)


def js(p):
    return dict(mean=round(float(p.mean()), 3), wins=int((p > 0).sum()), folds=[round(float(v), 3) for v in p])


def report():
    t00 = time.time()
    EDA = _load("eda3", EXPD / "EDA-3FAM-01" / "run.py"); EDA.TXT = M.TXT
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    arms = merge(CT)
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    ctl = CTL if CTL in arms else arms[0]
    say(f"== {TAG}: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}, {len(arms)} arms, control {ctl}")
    cols = {x: [f"{x}_t3", f"{x}_nf", f"{x}_nv"] for x in arms}
    for x, cs in cols.items():
        for c in cs:
            D[c] = D[c].fillna(float(np.nanmean(D[c].values)))       # MAV rows without an RF feature: column mean (as EDA-3FAM-01)
    groups = {f"trRFS_{x}": cs for x, cs in cols.items()}
    ok = ~E.mav_flagged(D)
    for g, cs in groups.items():                                      # standalone values (replaced inside every fuser split)
        D[g] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, g] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cs)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)
    ex = [f"rfe{a}" for a in EXPERTS if f"rfe{a}" in arms]
    fused = {"F1s": S6 + E.V11_T3 + ["trCA"], **{f"R_{x}": [f"trRFS_{x}"] for x in arms}}
    if len(ex) >= 2:
        fused["R_mv"] = [f"trRFS_{x}" for x in ex]; fused["R_mvA"] = [f"trRFS_{ctl}"] + fused["R_mv"]
    say("\n== fused arms (CA-FUSE C = 1, nested + purged, 3 regimes)")
    EDA.run_arms(D, S2, persons, fused)
    us = [x for x in arms] + [n[2:] for n in ("R_mv", "R_mvA") if n in fused]
    for x in us:
        EDA.add_sum(D, f"U_{x}", ["F1s", f"R_{x}"])
    wn = {}
    for w in WGRID:                                                   # E3: U_w = w F1s + (1 - w) R_a0 (U' = w 0.5, up to scale)
        n = f"W{int(round(100 * w)):03d}"; wn[n] = w
        for r in ("oof", "hold_urdu", "hold_hindi"):
            D[f"{n}|{r}"] = w * D[f"F1s|{r}"] + (1 - w) * D[f"R_{ctl}|{r}"]
    names = ["F1s"] + [f"U_{x}" for x in us] + [f"R_{x}" for x in us] + list(wn)
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in names})
    show = ["F1s", f"U_{ctl}", f"R_{ctl}"] + [c for c in ("U_mv", "U_mvA", "R_mv", "R_mvA") if c in T] + \
           [f"U_rfe{a}" for a in LOBO + EXPERTS if f"U_rfe{a}" in T] + [f"R_rfe{a}" for a in LOBO + EXPERTS if f"R_rfe{a}" in T]
    show = list(dict.fromkeys(show))
    say("EER (oof, mean of folds):\n" + T.groupby("cell").mean(numeric_only=True)[show].round(2).T.to_string())
    summ = {"arms": arms, "control": ctl, "eer_cells": T.groupby("cell").mean(numeric_only=True)[show].round(3).to_dict()}

    say("\n== checks")
    for pfx in ("rft", "rfu"):
        if f"{pfx}a0" in arms:
            d = float((T[f"U_{pfx}a0"] - T[f"U_{ctl}"]).abs().max())
            say(f"  U_{pfx}a0 vs U_{ctl}: max |EER difference| over fold x cell {d:.4f} (deterministic trainer -> 0)"); summ[f"a0_same_{pfx}"] = d
    nf = {}
    if "rfea0r" in arms:
        for nm, cells in (("en_g", ENG), ("en_ng", ENNG), ("oth_g", OTH)):
            p = dl(T, "U_rfea0r", f"U_{ctl}", cells); nf[nm] = dict(mean=round(float(p.mean()), 3), abs_fold=round(float(p.abs().mean()), 3))
            say(f"  noise floor a0r - a0 ({nm}): mean {p.mean():+.2f}, mean |fold| {p.abs().mean():.2f}")
    summ["noise_floor"] = nf

    say(f"\n== E1 ENCABL-01: one block set to 0 vs {ctl} (U level; > 0 = without the block better)")
    e1 = {}
    for a in LOBO:
        x = f"rfe{a}"
        if x not in arms:
            continue
        pg, png, po = dl(T, f"U_{x}", f"U_{ctl}", ENG), dl(T, f"U_{x}", f"U_{ctl}", ENNG), dl(T, f"U_{x}", f"U_{ctl}", OTH)
        rg = dl(T, f"R_{x}", f"R_{ctl}", ENG)
        carries = pg.mean() <= -.5 and (pg < 0).sum() >= 4
        hurts = pg.mean() >= .5 and (pg > 0).sum() >= 4 and png.mean() >= -.2
        lab = "carries English" if carries else "hurts English" if hurts else "within noise"
        if (carries or hurts) and (abs(po.mean()) < .2 or np.sign(po.mean()) != np.sign(pg.mean())):
            lab += " + English-specific"
        say(f"  {a:<6} en g {fmt(pg)}\n         en ng {fmt(png)}\n         other g {fmt(po)}\n         R level en g mean {rg.mean():+.2f} -> {lab}")
        e1[a] = dict(en_g=js(pg), en_ng=js(png), oth_g=js(po), R_en_g=round(float(rg.mean()), 3), label=lab)
    summ["E1"] = e1

    say(f"\n== E2 MVIEW-01: per-encoder experts {ex} fused by CA-FUSE vs U_{ctl}")
    e2 = {}
    for nm in ("mv", "mvA"):
        if f"U_{nm}" not in T:
            continue
        p, h = EDA.cmp(D, f"U_{nm}", f"U_{ctl}")
        lo, hi = S2.boot(D.rename(columns={f"U_{ctl}|oof": "ref_", f"U_{nm}|oof": "cand_"}), "ref_", "cand_",
                         sorted(c for c in D.cell.unique() if c.endswith(" g")), "both", B=NB)
        pg, png = dl(T, f"U_{nm}", f"U_{ctl}", ENG), dl(T, f"U_{nm}", f"U_{ctl}", ENNG)
        rs, re_ = EDA.rule(p, h, thr=.3), en_rule(pg, png)
        say(f"  U_{nm} - U_{ctl} g cells: {fmt(p)} | held-out {h} | CI95 [{lo:+.2f}, {hi:+.2f}] -> system rule {'PASS' if rs else 'FAIL'}")
        say(f"     en g {fmt(pg)} | en ng {fmt(png)} -> English rule {'PASS' if re_ else 'FAIL'}")
        e2[nm] = dict(g=js(p), hold=h, ci95=[round(lo, 3), round(hi, 3)], system_pass=rs, en_g=js(pg), en_ng=js(png), english_pass=re_,
                      R_vs_ctl_g=round(float(dl(T, f"R_{nm}", f"R_{ctl}", ENG + OTH).mean()), 3))
    summ["E2"] = e2

    say("\n== E3 ENWT-01: U_w = w F1s + (1 - w) RFS; nested w on the two English cells vs w = 0.5 (U')")
    ws = list(wn); folds = sorted(int(s) for s in T.fold.unique())
    say("  mean EER per w:\n" + pd.DataFrame({c: T[T.cell.isin(cl)].groupby("fold")[ws].mean().mean() for c, cl in
                                             (("English", EN), ("en g", ENG), ("other g", OTH))}).round(2).T.to_string())
    pick_en, pick_ot, dg, dng = {}, {}, {}, {}
    for s in folds:
        ie = T[(T.fold != s) & T.cell.isin(EN)].groupby("fold")[ws].mean().mean(); pick_en[s] = ie.idxmin()
        io = T[(T.fold != s) & T.cell.isin(OTH)].groupby("fold")[ws].mean().mean(); pick_ot[s] = io.idxmin()
        o = T[T.fold == s].set_index("cell")
        dg[s] = float(o.loc[ENG[0], "W050"] - o.loc[ENG[0], pick_en[s]]); dng[s] = float(o.loc[ENNG[0], "W050"] - o.loc[ENNG[0], pick_en[s]])
    pg, png = pd.Series(dg), pd.Series(dng); r3 = en_rule(pg, png)
    say(f"  picks on English: { {s: wn[v] for s, v in pick_en.items()} }; picks on other g: { {s: wn[v] for s, v in pick_ot.items()} }")
    say(f"  SELw - U' en g {fmt(pg)} | en ng {fmt(png)} -> English rule {'PASS' if r3 else 'FAIL'}")
    summ["E3"] = dict(picks_en={s: wn[v] for s, v in pick_en.items()}, picks_oth={s: wn[v] for s, v in pick_ot.items()},
                      en_g=js(pg), en_ng=js(png), english_pass=r3)

    say(f"\n== E4 ENLANG-01: nested arm choice on the English cells over {len(arms)} arms vs U_{ctl}")
    ua = [f"U_{x}" for x in arms]; pe, pgc, dg, dng = {}, {}, {}, {}
    gcells = sorted(c for c in T.cell.unique() if c.endswith(" g"))
    for s in folds:
        pe[s] = T[(T.fold != s) & T.cell.isin(EN)].groupby("fold")[ua].mean().mean().idxmin()
        pgc[s] = T[(T.fold != s) & T.cell.isin(gcells)].groupby("fold")[ua].mean().mean().idxmin()
        o = T[T.fold == s].set_index("cell")
        dg[s] = float(o.loc[ENG[0], f"U_{ctl}"] - o.loc[ENG[0], pe[s]]); dng[s] = float(o.loc[ENNG[0], f"U_{ctl}"] - o.loc[ENNG[0], pe[s]])
    pg, png = pd.Series(dg), pd.Series(dng); r4 = en_rule(pg, png)
    diff = sum(pe[s] != pgc[s] for s in folds)
    say(f"  picks on English: { {s: v[2:] for s, v in pe.items()} }\n  picks on g cells: { {s: v[2:] for s, v in pgc.items()} } ({diff}/5 differ)")
    say(f"  SEL_en - U_{ctl} en g {fmt(pg)} | en ng {fmt(png)} -> English rule {'PASS' if r4 else 'FAIL'}")
    summ["E4"] = dict(picks_en={s: v[2:] for s, v in pe.items()}, picks_g={s: v[2:] for s, v in pgc.items()}, differ=diff,
                      en_g=js(pg), en_ng=js(png), english_pass=r4)

    say("\n== E5 ENCORR-01 (a): within-class Spearman (fold x cell, mean of the two classes)")
    cc = [c for c in ["trCA", "anP512_deep", f"trRFS_{ctl}"] + [f"trRFS_{x}" for x in ex] + ["F1s|oof", f"R_{ctl}|oof"] if c in D]
    e5 = {}
    for nm, cells in (("en_g", ENG), ("en_ng", ENNG), ("oth_g", OTH)):
        Rm = EDA.wcorr(D, cc, cells)
        say(f"  {nm}:\n" + Rm.round(2).to_string())
        e5[f"a_{nm}"] = Rm.round(3).to_dict()
    say("  rho(F1s, R_a0): " + ", ".join(f"{nm} {e5[f'a_{nm}']['F1s|oof'][f'R_{ctl}|oof']:.3f}" for nm in ("en_g", "en_ng", "oth_g")))
    say("\n== E5 ENCORR-01 (b): arm effects, English g vs other g (each arm vs the a0 of its round)")
    rows = []
    for x in arms:
        pfx, a = x[:3], x[3:]
        if a == "a0":
            continue
        c0 = f"U_{pfx}a0" if f"U_{pfx}a0" in T else f"U_{ctl}"
        rows.append(dict(arm=x, en_g=float(dl(T, f"U_{x}", c0, ENG).mean()), en_ng=float(dl(T, f"U_{x}", c0, ENNG).mean()),
                         oth_g=float(dl(T, f"U_{x}", c0, OTH).mean())))
    A = pd.DataFrame(rows).set_index("arm")
    say(A.sort_values("en_g").round(2).to_string())
    if len(A) >= 4:
        from scipy.stats import pearsonr, spearmanr
        r, rho = float(pearsonr(A.en_g, A.oth_g)[0]), float(spearmanr(A.en_g, A.oth_g)[0])
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000 if not FAST else 50):
            i = rng.integers(0, len(A), len(A))
            if np.std(A.en_g.values[i]) > 0 and np.std(A.oth_g.values[i]) > 0:
                bs.append(np.corrcoef(A.en_g.values[i], A.oth_g.values[i])[0, 1])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        read = "English follows the other cells" if r >= .5 else "English needs its own tuning" if r <= 0 else "partly"
        say(f"  {len(A)} arms: Pearson r {r:+.2f} (bootstrap over arms CI95 [{lo:+.2f}, {hi:+.2f}]), Spearman rho {rho:+.2f} -> {read}")
        e5["b"] = dict(n=len(A), r=round(r, 3), ci95=[round(float(lo), 3), round(float(hi), 3)], rho=round(rho, 3), reading=read,
                       arms=A.round(3).to_dict(orient="index"))
    summ["E5"] = e5

    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
