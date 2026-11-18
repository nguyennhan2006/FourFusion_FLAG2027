"""LANGLANE-01 (README registered 15/11 before any number): two separate validation suites, one for English (heard) and one for the
unheard language (Bangla proxy = Urdu / Hindi, each scored only by fusers that never saw it), and language-lane systems (English and
Bangla files scored by different fusers, chosen each on its own suite).
    python lab_entry.py LANGLANE-01 report    (Kaggle CPU FLAG_217: lab data + T3S job tables + rfr{c0r,c0s} of FLAG_213 + rfrc0t of FLAG_216)
    python lab_entry.py LANGLANE-01 backtest  (Kaggle CPU FLAG_218: inputs of FLAG_135; new English measures vs past CB English directions)
Nothing reads dev / CB / Evaluation labels. Writes langlane_report.txt / langlane_summary.json and langlanebt_report.txt / langlanebt_calls.csv."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


SRCS = [("rfr", "c0r"), ("rfr", "c0s"), ("rfr", "c0t")]
FRESH = "c0t"
REG = ("oof", "hold_urdu", "hold_hindi")
STR = ("t3", "nf", "nv")
EN, NONEN = ("v4", "en_ext"), ("urdu", "hindi")
K_UP = 5                      # soft split: rows of the lane's own languages repeated K_UP times in the fuser pool
HARD_Q = 1 / 3                # English hard cell: positives + the third of negatives with the closest face-age of the two persons
BN_C = ["BU", "BJ01", "BJ001", "BJps", "BONLY", "BW5", "BJW5"]
EN_C = ["EU", "EJ01", "EONLY"]


def js(p):
    p = np.asarray(p, float)
    return dict(mean=round(float(np.nanmean(p)), 3), wins=int((p > 0).sum()), folds=[round(float(v), 3) for v in p])


def hard_mask(D):
    """True for every positive and for the negatives of the g cells whose two persons (face person fp, voice person vp) have the closest
    face age: |A(fp) - A(vp)| in the lowest HARD_Q of the negatives of that (fold, cell); A(p) = mean agef over p's rows as face person.
    Scale-free (no age unit assumed). Negatives of ng cells and negatives with an unknown age are False (only the g cells are read)."""
    age = D.groupby("fp").agef.mean()
    gap = np.abs(D.fp.map(age).values - D.vp.map(age).values).astype(float)
    y = D.y.values
    keep = y == 1
    for (s, cell), idx in D.groupby(["fold", "cell"]).indices.items():      # positional indices
        if not cell.endswith(" g"):
            continue
        neg = idx[(y[idx] == 0) & np.isfinite(gap[idx])]
        if len(neg) >= 10:
            keep[neg[gap[neg] <= np.quantile(gap[neg], HARD_Q)]] = True
    return keep


# ---------------------------------------------------------------- part B: suites + lane candidates on SAFE-04 (FLAG_217)
def report():
    t00 = time.time()
    FS = _load("fusearch01", EXPD / "FUSEARCH-01" / "run.py")
    M, E, S6, table, OUT, KEY, say, FAST = FS.M, FS.E, FS.S6, FS.table, FS.OUT, FS.KEY, FS.say, FS.FAST
    TAG = "langlane"
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    CT.merge_t3s()
    got = [(p, a) for p, a in SRCS if (OUT / f"{p}{a}lab_fold1.parquet").exists()]
    if not FAST:
        assert ("rfr", "c0r") in got and ("rfr", "c0s") in got, f"seen replicates missing: {got}"
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
    D = E.safe04(TAG).reset_index(drop=True); D["is_gp"] = D.cell.str.endswith(" g")
    F1S = S6 + E.V11_T3 + ["trCA"]
    say(f"== {TAG}: {len(D)} trials, RFX replicates {reps} (fresh = {FRESH}{'' if FRESH in reps else ' MISSING'})")
    say("   data rows: " + str(D.data.value_counts().to_dict()))
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
    HARD = hard_mask(D)
    say(f"   English hard cell rows: " + str(D[HARD & D.data.isin(EN).values].cell.value_counts().to_dict()))

    def fuse2(name, st, C=1.0, split=False, only=None, up=None):
        """FUSEARCH-01 fuse (CA-FUSE nested per fold, purged, 3 regimes) + pool restricted to the languages `only` (hard split) or
        with the rows of `up` repeated K_UP times (soft split)."""
        t0 = time.time()
        for regime, held in S2.HELD.items():
            col = f"{name}|{regime}"; D[col] = np.nan
            for s in sorted(D.fold.unique()):
                for pg in ((True, False) if split else (None,)):
                    mt = (D.fold == s) & (D.data.isin(held) if held else True)
                    mp = (D.fold != s) & ~D.data.isin(held)
                    if only is not None:
                        mp = mp & D.data.isin(only)
                    if pg is not None:
                        mt, mp = mt & (D.is_gp == pg), mp & (D.is_gp == pg)
                    te = D[mt]
                    if not len(te):
                        continue
                    pool = S2.purge(D[mp], persons[int(s)])
                    if up is not None:
                        pool = pd.concat([pool] + [pool[pool.data.isin(up)]] * (K_UP - 1))
                    if pool.y.nunique() < 2:
                        continue
                    D.loc[te.index, col] = S2.Design(pool, te, st).score(st, C)
        say(f"  {name:<12} {len(st):>2} streams, C {C}, {'per protocol' if split else 'pooled'}"
            f"{', only ' + '+'.join(only) if only else ''}{', x' + str(K_UP) + ' ' + '+'.join(up) if up else ''} [{time.time() - t0:.0f}s]")

    def lin(name, parts):
        for r in REG:
            D[f"{name}|{r}"] = sum(D[f"{c}|{r}"] for c in parts)

    say("\n== fusers")
    fuse2("F1s", F1S); fuse2("F1s_nE", F1S, only=NONEN); fuse2("F1s_E", F1S, only=EN); fuse2("F1s_up", F1S, up=NONEN)
    for a in reps:
        fuse2(f"R_{a}", [f"trRFS_{a}"]); fuse2(f"R_{a}_nE", [f"trRFS_{a}"], only=NONEN)
        fuse2(f"R_{a}_E", [f"trRFS_{a}"], only=EN); fuse2(f"R_{a}_up", [f"trRFS_{a}"], up=NONEN)
        raw = F1S + [f"{a}_{c}" for c in STR]
        fuse2(f"J01_{a}", raw, C=0.1); fuse2(f"J001_{a}", raw, C=0.01)
        fuse2(f"J01ps_{a}", raw, C=0.1, split=True); fuse2(f"J01up_{a}", raw, C=0.1, up=NONEN)
        lin(f"U_{a}", ["F1s", f"R_{a}"]); lin(f"ONLY_{a}", ["F1s_nE", f"R_{a}_nE"])
        lin(f"W5_{a}", ["F1s_up", f"R_{a}_up"]); lin(f"EONLY_{a}", ["F1s_E", f"R_{a}_E"])

    def col_of(cand, a):
        return {"BU": f"U_{a}", "BJ01": f"J01_{a}", "BJ001": f"J001_{a}", "BJps": f"J01ps_{a}", "BONLY": f"ONLY_{a}",
                "BW5": f"W5_{a}", "BJW5": f"J01up_{a}", "EU": f"U_{a}", "EJ01": f"J01_{a}", "EONLY": f"EONLY_{a}"}[cand]

    def piv(col, regime, rows=None):
        """EER per (fold, cell) of col|regime, optionally on a row subset -> DataFrame fold x cell."""
        X = D if rows is None else D[rows]
        T = table(X, {"x": lambda g: g[f"{col}|{regime}"].values})
        return T.pivot(index="fold", columns="cell", values="x")

    def bn_suite(col):
        """unheard-language suite, per fold: g/Bn ~ urdu g scored with Urdu held out of the fuser; ng/Bn ~ mean(urdu ng held-out Urdu,
        hindi ng held-out Hindi); LBN = mean of the two. Also the components and urdu g of the oof regime (heard)."""
        hu, hh, oo = piv(col, "hold_urdu"), piv(col, "hold_hindi"), piv(col, "oof")
        g, ng = hu["urdu g"], (hu["urdu ng"] + hh["hindi ng"]) / 2
        return pd.DataFrame({"LBN": (g + ng) / 2, "g_bn": g, "ng_bn": ng, "urdu_g_hold": hu["urdu g"], "urdu_ng_hold": hu["urdu ng"],
                             "hindi_ng_hold": hh["hindi ng"], "urdu_g_oof": oo["urdu g"]})

    def en_suite(col):
        """English suite (oof regime), per fold: LEN = mean(mean(v4 g, en_ext g), mean(v4 ng, en_ext ng)); EH = mean of the two hard g
        cells (positives + age-matched third of the negatives); per source v4 / en_ext = mean of its g and ng cells."""
        oo, hd = piv(col, "oof"), piv(col, "oof", HARD)
        return pd.DataFrame({"LEN": ((oo["v4 g"] + oo["en_ext g"]) / 2 + (oo["v4 ng"] + oo["en_ext ng"]) / 2) / 2,
                             "EH": (hd["v4 g"] + hd["en_ext g"]) / 2, "src_v4": (oo["v4 g"] + oo["v4 ng"]) / 2,
                             "src_en_ext": (oo["en_ext g"] + oo["en_ext ng"]) / 2, "v4 g": oo["v4 g"], "v4 ng": oo["v4 ng"],
                             "en_ext g": oo["en_ext g"], "en_ext ng": oo["en_ext ng"]})

    summ = {"replicates": reps, "fresh": FRESH, "bangla": {}, "english": {}}
    bn_pass = {}
    for a in reps:
        say(f"\n==================== replicate {a}{' (FRESH, decision)' if a == FRESH else ' (seen in FUSEARCH-02)'}")
        # ---- Bangla lane
        S = {c: bn_suite(col_of(c, a)) for c in BN_C}
        say("-- unheard-language suite, mean of folds (lower = better)\n"
            + pd.DataFrame({c: S[c].mean() for c in BN_C}).T.round(2).to_string())
        L = pd.DataFrame({c: S[c].LBN for c in BN_C})
        picks, sel = {}, {}
        for s in L.index:
            e = L.loc[L.index != s].mean(); picks[int(s)] = e.idxmin()
            sel[s] = S[picks[int(s)]].loc[s]
        SEL = pd.DataFrame(sel).T
        ref = S["BU"]
        d = {k: (ref[k] - SEL[k]).values for k in ("LBN", "urdu_g_hold", "hindi_ng_hold", "urdu_g_oof")}
        main = bool(np.mean(d["LBN"]) >= .3 and (d["LBN"] > 0).sum() >= 4)
        guard = bool(min(np.mean(d["urdu_g_hold"]), np.mean(d["hindi_ng_hold"]), np.mean(d["urdu_g_oof"])) >= -.2)
        bn_pass[a] = main and guard
        say("-- every Bangla candidate vs BU (> 0 = candidate better): LBN per fold | guards (urdu g held-out, hindi ng held-out, urdu g oof)")
        cmpb = {}
        for c in BN_C[1:]:
            dc = (ref.LBN - S[c].LBN).values
            gd = [float((ref[k] - S[c][k]).mean()) for k in ("urdu_g_hold", "hindi_ng_hold", "urdu_g_oof")]
            say(f"  {c:<6} LBN {dc.mean():+.2f} ({(dc > 0).sum()}/5) [{', '.join(f'{v:+.2f}' for v in dc)}] | guards "
                + " ".join(f"{v:+.2f}" for v in gd))
            cmpb[c] = dict(LBN=js(dc), guards=[round(v, 3) for v in gd])
        say(f"  SEL_BN picks {picks}: LBN {d['LBN'].mean():+.2f} ({(d['LBN'] > 0).sum()}/5); guards "
            f"{np.mean(d['urdu_g_hold']):+.2f} / {np.mean(d['hindi_ng_hold']):+.2f} / {np.mean(d['urdu_g_oof']):+.2f}"
            f" -> main {'PASS' if main else 'FAIL'}, guards {'PASS' if guard else 'FAIL'}")
        summ["bangla"][a] = dict(picks=picks, sel={k: js(v) for k, v in d.items()}, main=main, guard=guard, passed=main and guard,
                                 cands=cmpb, lbn_mean={c: round(float(S[c].LBN.mean()), 3) for c in BN_C})
        # ---- English lane (oof regime)
        Q = {c: en_suite(col_of(c, a)) for c in EN_C}
        say("-- English suite, mean of folds (lower = better)\n" + pd.DataFrame({c: Q[c].mean() for c in EN_C}).T.round(2).to_string())
        eng = {}
        for metric in ("LEN", "EH"):
            Lm = pd.DataFrame({c: Q[c][metric] for c in EN_C})
            pk = {int(s): Lm.loc[Lm.index != s].mean().idxmin() for s in Lm.index}
            SE = pd.DataFrame({s: Q[pk[int(s)]].loc[s] for s in Lm.index}).T
            dm = (Q["EU"][metric] - SE[metric]).values
            dsrc = [float((Q["EU"][k] - SE[k]).mean()) for k in ("src_v4", "src_en_ext")]
            ge = bool(dm.mean() >= .3 and (dm > 0).sum() >= 4 and min(dsrc) >= -.2)
            say(f"  SEL_EN by {metric}: picks {pk}: {metric} {dm.mean():+.2f} ({(dm > 0).sum()}/5), sources v4 {dsrc[0]:+.2f} / en_ext "
                f"{dsrc[1]:+.2f} -> {'PASS' if ge else 'FAIL'} (counts only if FLAG_218 finds {metric} usable)")
            eng[metric] = dict(picks=pk, d=js(dm), src=[round(v, 3) for v in dsrc], passed=ge)
        for c in EN_C[1:]:
            say(f"  {c:<6} vs EU: LEN {(Q['EU'].LEN - Q[c].LEN).mean():+.2f}, EH {(Q['EU'].EH - Q[c].EH).mean():+.2f}, "
                f"v4 {(Q['EU'].src_v4 - Q[c].src_v4).mean():+.2f}, en_ext {(Q['EU'].src_en_ext - Q[c].src_en_ext).mean():+.2f}")
        summ["english"][a] = eng

    say("\n==================== DECISION (README section 4)")
    n_pass = sum(bn_pass.values())
    if FRESH not in reps:
        verdict = "UNDECIDED (fresh replicate c0t missing)"
    else:
        verdict = "PASS" if bn_pass[FRESH] and n_pass >= 2 else "FAIL"
    lbn_all = pd.DataFrame({a: summ["bangla"][a]["lbn_mean"] for a in reps}).mean(1)
    final = lbn_all.idxmin()
    say(f"  Bangla lane per replicate {bn_pass} -> {verdict}; final Bangla candidate (lowest LBN over replicates and folds) = {final}"
        f" (LBN {lbn_all[final]:.2f}, BU {lbn_all['BU']:.2f})")
    say("  English lane: stays EU (= FACTOR_ENFUSE English) unless FLAG_218 finds a usable English measure and SEL_EN passes on it")
    summ.update(verdict=verdict, final_bangla=final, lbn_over_reps={k: round(float(v), 3) for k, v in lbn_all.items()})

    say("\n== reported only: LB4 of the 4 test files (English cells oof, Bangla cells held-out) for LANE = EU + final vs control")
    for a in reps:
        def lb4(cb):
            e, b = en_suite(col_of("EU", a)), bn_suite(col_of(cb, a))
            return ((e["v4 g"] + e["en_ext g"]) / 2 + (e["v4 ng"] + e["en_ext ng"]) / 2 + b.g_bn + b.ng_bn) / 4
        dl = (lb4("BU") - lb4(final)).values
        say(f"  {a}: LB4 control {lb4('BU').mean():.2f}, LANE {lb4(final).mean():.2f}, gain {dl.mean():+.2f} ({(dl > 0).sum()}/5)")
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


# ---------------------------------------------------------------- part A: English suite backtest (FLAG_218)
def backtest():
    """BACKTEST-01's cases (+ ENFUSE-01 with its CB, now known) scored with the new English measures; rules as BACKTEST-01."""
    BT = _load("backtest01", EXPD / "BACKTEST-01" / "run.py")
    S2, persons = BT.E.stack02(), BT.E.fold_persons()
    D = BT.table().reset_index(drop=True)
    HARD = hard_mask(D)
    cases = dict(BT.CASES)
    (pa, pb), = BT.PENDING.values()
    cases["ENFUSE-01 (324)"] = (pa, pb, -0.84, "fusion form")
    lines = [f"== LANGLANE-01 part A (registered 15/11): English table {len(D)} trials; hard g rows "
             + str(D[HARD].cell.value_counts().to_dict()),
             "delta = EER(A) - EER(B), + = B better on validation; CI95 from BACKTEST-01's person resamples"]
    groups = list(pd.Series(np.arange(len(D))).groupby(D.vp.values).indices.values())
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(BT.N_BOOT)]
    every = np.arange(len(D))
    cells = ["v4 g", "v4 ng", "en_ext g", "en_ext ng"]

    def per_cell(sa, sb, idx):
        out = {}
        for cell in cells:
            for h in (False, True):
                if h and not cell.endswith(" g"):
                    continue
                m = idx[(D.cell.values[idx] == cell) & (HARD[idx] if h else True)]
                y = D.y.values[m]
                out[cell + (" H" if h else "")] = BT.eer(sa[m], y) - BT.eer(sb[m], y) if 0 < y.sum() < len(y) else np.nan
        return out

    def measures(c):
        return {"N1 LEN (= M6)": np.nanmean([c["v4 g"], c["en_ext g"], c["v4 ng"], c["en_ext ng"]]),
                "N2 EH": np.nanmean([c["v4 g H"], c["en_ext g H"]]),
                "src v4 g": c["v4 g"], "src en_ext g": c["en_ext g"], "src v4 g H": c["v4 g H"], "src en_ext g H": c["en_ext g H"]}

    rows = []
    for name, (a, b, cb, fam) in cases.items():
        sa, sb = BT.scores(D, a, S2, persons), BT.scores(D, b, S2, persons)
        assert np.isfinite(sa).all() and np.isfinite(sb).all(), name
        pt = measures(per_cell(sa, sb, every))
        dist = pd.DataFrame([measures(per_cell(sa, sb, i)) for i in boots])
        ci = {k: np.nanpercentile(dist[k], [2.5, 97.5]) for k in pt}
        def call1(k):
            lo, hi = ci[k]; return "B better" if lo > 0 else "B worse" if hi < 0 else "inconclusive"
        def call2(k1, k2):
            c1, c2 = call1(k1), call1(k2); return c1 if c1 == c2 else "inconclusive"
        calls = {"N1 LEN (= M6)": call1("N1 LEN (= M6)"), "N2 EH": call1("N2 EH"),
                 "N3 SA-g": call2("src v4 g", "src en_ext g"), "N4 SA-EH": call2("src v4 g H", "src en_ext g H")}
        val = {"N1 LEN (= M6)": pt["N1 LEN (= M6)"], "N2 EH": pt["N2 EH"],
               "N3 SA-g": np.nanmean([pt["src v4 g"], pt["src en_ext g"]]), "N4 SA-EH": np.nanmean([pt["src v4 g H"], pt["src en_ext g H"]])}
        for k, call in calls.items():
            rows.append(dict(case=name, measure=k, delta=val[k], call=call, cb_delta=cb, family=fam,
                             correct=None if call == "inconclusive" else (call == "B better") == (cb < 0),
                             detail="; ".join(f"{q} {pt[q]:+.2f} [{ci[q][0]:+.2f}, {ci[q][1]:+.2f}]" for q in pt)))
        lines.append(f"  {name} (CB English B - A {cb:+.2f}): " + "; ".join(f"{q} {pt[q]:+.2f} [{ci[q][0]:+.2f}, {ci[q][1]:+.2f}]" for q in pt))
        print(lines[-1], flush=True)
    R = pd.DataFrame(rows)
    out = ["\n== calls: delta -> call / correct vs CB"]
    usable = []
    fam0 = ["B3 / T3 streams", "s007 / s010B / AudioCLIP"]
    for k in R.measure.unique():
        m = R[R.measure == k]
        out.append(f"{k}: " + " | ".join(f"{r.case.split(' (')[0]} {r.delta:+.2f} {r.call}" + ("" if r.correct is None else (" OK" if r.correct else " WRONG"))
                                         for r in m.itertuples()))
        made = m[m.correct.notna()]
        fam_ok = all(((made.family == f) & (made.correct == True)).any() for f in fam0)   # noqa: E712
        okk = bool((made.correct == True).all() and len(made) >= 3 and fam_ok)            # noqa: E712
        rho = pd.Series(m.delta.values).corr(pd.Series(-m.cb_delta.values), method="spearman")
        out.append(f"   -> {int((made.correct == True).sum())} correct / {int((made.correct == False).sum())} wrong / "   # noqa: E712
                   f"{len(m) - len(made)} inconclusive; both original families right {fam_ok}; Spearman(delta, -CB) {rho:+.2f} -> "
                   f"{'USABLE (hint only)' if okk else 'not usable'}")
        if okk:
            usable.append(k)
    out.append("VERDICT part A: " + (f"usable English measures {usable} (hint only; an English change still needs the user's decision)"
                                     if usable else "no usable English measure -> the English lane stays FACTOR_ENFUSE"))
    lines += out
    print("\n".join(out), flush=True)
    from lab_fuse import SP
    o = SP / "out/safe04"
    R.to_csv(o / "langlanebt_calls.csv", index=False)
    (o / "langlanebt_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"report": report, "backtest": backtest}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
