"""EDA-3FAM-01 (README registered 5/11 before any number): F1s / RFS / T3S on the same SAFE-04 pairs (5 folds, purged).
    python lab_entry.py EDA-3FAM-01 report        (Kaggle CPU, FLAG_171; needs the lab data + the tables below in <SP>/out/safe04)
Tables: ct01r2<face>_fold<s> (T3S-01, 20 job tables), wh02_fold<s> (WHISPER-02 merged: trCA / trCAW / trCAWO + 'b' copies),
rflab_fold<s> / rfnwlab_fold<s> (RFS-LAB, FLAG_170). Nothing reads dev / CB. Writes eda3fam_report.txt, eda3fam_summary.json,
eda3fam_oof.parquet next to the tables."""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent
sys.path.insert(0, str(EXPD / "E2E")); sys.path.insert(0, str(EXPD / "SAFE-04"))
import e2e_lib as E
from stream_gate import S6
from lab_fuse import SP, eer, table

TAG = "eda3fam"
KEY = ["fold", "data", "protocol", "y", "fi", "vi"]
OUT = SP / "out/safe04"
TXT = []
FACES, VOICES = ("farl", "clip", "b3s", "c5s"), ("sim", "r293", "ecapa", "cat4", "wsv", "rdn")
FAST = os.environ.get("FL3Z_SMOKE") == "1"                         # smoke (stand-in tables): no ALLraw, greedy on one fold, small bootstraps
NB = 20 if FAST else 300


def say(*a):
    s = " ".join(str(x) for x in a); TXT.append(s); print(s, flush=True)


def _ct():
    spec = importlib.util.spec_from_file_location("ctrun", EXPD / "CLEANTUNE-01/run.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ 1. one table per fold, rows checked
def merge(CT):
    CT.merge_t3s()                                                    # 20 job tables -> ct01r2_fold<s> (T3S branches + trCA)
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet")
        for pre, ren in (("wh02", {"trCA": "whCA", "trCAb": "whCAb"}), ("rflab", {}), ("rfnwlab", {})):
            P = pd.read_parquet(OUT / f"{pre}_fold{s}.parquet")
            assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"{pre} fold {s}: rows"
            for c in P.columns:
                if c not in KEY:
                    T[ren.get(c, c)] = P[c].values
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
    say(f"merged {TAG}: columns {[c for c in T.columns if c not in KEY]}")


# ------------------------------------------------------------------ helpers
def run_arms(D, S2, persons, arms):
    """CA-FUSE(streams) nested per fold, purged, regimes oof / hold Urdu / hold Hindi (= E.gate) -> D['<arm>|<regime>']."""
    for name, st in arms.items():
        t0 = time.time()
        for regime, held in S2.HELD.items():
            col = f"{name}|{regime}"; D[col] = np.nan
            for s in sorted(D.fold.unique()):
                te = D[(D.fold == s) & (D.data.isin(held) if held else True)]
                pool = S2.purge(D[(D.fold != s) & ~D.data.isin(held)], persons[int(s)])
                D.loc[te.index, col] = S2.Design(pool, te, st).score(st, 1.0)
        say(f"  arm {name:<12} {len(st)} streams [{time.time() - t0:.0f}s]")


def add_sum(D, name, parts):
    for r in ("oof", "hold_urdu", "hold_hindi"):
        D[f"{name}|{r}"] = sum(D[f"{p}|{r}"] for p in parts)


def cmp(D, a, b):
    """a vs b: per fold, mean over g cells of EER(b) - EER(a) (> 0 = a better); held-out = mean over the held language's cells."""
    T = table(D, {x: (lambda g, x=x: g[x].values) for x in (f"{a}|oof", f"{b}|oof")})
    T = T[T.cell.str.endswith(" g")]
    p = (T[f"{b}|oof"] - T[f"{a}|oof"]).groupby(T.fold).mean()
    h = {}
    for r in ("hold_urdu", "hold_hindi"):
        T2 = table(D, {x: (lambda g, x=x: g[x].values) for x in (f"{a}|{r}", f"{b}|{r}")}).dropna()
        h[r] = round(float((T2[f"{b}|{r}"] - T2[f"{a}|{r}"]).groupby(T2.cell).mean().mean()), 3)
    return p, h


def fmt(p):
    return ", ".join(f"f{int(s)} {g:+.2f}" for s, g in p.items()) + f" | mean {p.mean():+.2f}, {int((p > 0).sum())}/5"


def rule(p, h, thr=.3):
    return bool(p.mean() >= thr and (p > 0).sum() >= 4 and min(h.values()) >= -.2)


def wcorr(D, cols, cells):
    """within-class Spearman per (fold, cell), mean of the two classes, then mean over folds and cells -> DataFrame."""
    acc, n = np.zeros((len(cols), len(cols))), 0
    for (s, cell), g in D[D.cell.isin(cells)].groupby(["fold", "cell"]):
        R = [g.loc[g.y == y, cols].corr(method="spearman").values for y in (0, 1)]
        acc += np.nan_to_num((R[0] + R[1]) / 2); n += 1
    return pd.DataFrame(acc / n, index=cols, columns=cols)


def thr_err(g, col):
    """row-level error flags at the EER operating point of one (fold, cell) group."""
    s, y = g[col].values, g.y.values
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(y, s); i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return np.where(y == 1, s < th[i], s >= th[i])


# ------------------------------------------------------------------ 2. report
def report():
    CT = _ct(); merge(CT)
    S2, persons = E.stack02(), E.fold_persons()
    D = E.safe04(TAG)
    import rel_sweep5 as R5
    br = sorted(c for c in D.columns if c.startswith("trT3S_") and c.count("_") == 2)
    assert len(br) == 24, br
    rf3, rfnw3 = ["rf_t3", "rf_nf", "rf_nv"], ["rfnw_t3", "rfnw_nf", "rfnw_nv"]
    say(f"== EDA-3FAM-01: {len(D)} trials, folds {sorted(int(s) for s in D.fold.unique())}")
    for c in ["trCA", "whCA", "trCAW", "trCAWO"] + rf3 + rfnw3 + ["rf_ib", "rf_age"] + br[:1]:
        say(f"  coverage {c}: {100 * np.isfinite(D[c].values).mean():.1f} % finite"
            + (" (v4 rows only by design)" if c in ("rf_ib", "rf_age") else ""))
    for c in rf3 + rfnw3:                                              # MAV rows without an RF feature: column mean (reported above)
        D[c] = D[c].fillna(float(np.nanmean(D[c].values)))
    groups = dict(CT._t3s_groups(D.columns), trRFS=rf3, trRFSnw=rfnw3,
                  **{f"trT3Sf_{f}": [c for c in br if c.split("_")[1] == f] for f in FACES},
                  **{f"trT3Sv_{v}": [c for c in br if c.split("_")[2] == v] for v in VOICES})
    ok = ~E.mav_flagged(D)
    for c, cols in groups.items():                                    # standalone values (LR of the purged other folds), as T3S-01
        D[c] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, c] = CT._lr(S2.purge(D[~te & ok], persons[int(s)]), cols)(D[te])
    S2.Design = CT._nested_design(S2, persons, groups)                # inside every fuser split the LR columns are refitted (no leak)
    base = S6 + E.V11_T3
    NR = {"rel", "relg", "t3_relg"} if E.NOREL else set()           # NOREL-01 (5/11, BTC FAQ): train-cohort streams leave even the descriptive tables
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g")); ngc = sorted(c for c in D.cell.unique() if c.endswith(" ng"))
    summ = {}

    # ---- A1 standalone
    st = ["trCA", "whCA", "trCAW", "trCAWO", "rf_t3", "rfnw_t3", "rf_nf", "rf_nv", "trRFS", "trRFSnw", "trT3S", "trT3Sw", "trT3Sr",
          "anP512_deep", "XALL_deep0", "rel", "relg"] + [f"trT3Sf_{f}" for f in FACES] + [f"trT3Sv_{v}" for v in VOICES]
    st = [c for c in st if c not in NR]
    A1 = table(D, {c: (lambda g, c=c: g[c].values) for c in st}).groupby("cell").mean(numeric_only=True)[st]
    say("\n== A1 standalone EER (mean of 5 folds)\n" + A1.T.round(2).to_string())
    A1v4 = table(D[D.data == "v4"], {c: (lambda g, c=c: g[c].values) for c in ("rf_t3", "rf_ib", "rf_age", "trCA")}).groupby("cell").mean(numeric_only=True)
    say("v4 cells only (rf_ib / rf_age exist there):\n" + A1v4.round(2).to_string())
    summ["A1"] = A1.round(3).to_dict()

    # ---- A2 error correlation + clustering
    cc = ["trCA", "whCA", "trCAW", "trCAWO", "rf_t3", "rfnw_t3", "trRFS", "trT3S", "trT3Sw", "trT3Sr", "anP512_deep", "t3_relg",
          "XALL_deep0", "XALL_cca", "rel", "relg"] + br
    cc = [c for c in cc if c not in NR]
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import squareform
    for nm, cells in (("g", gc), ("ng", ngc)):
        Rm = wcorr(D, cc, cells)
        main = cc[:16]
        say(f"\n== A2 within-class Spearman, {nm} cells (mean of folds, cells, classes)\n" + Rm.loc[main, main].round(2).to_string())
        Dm = squareform(np.clip(1 - Rm.values, 0, None), checks=False); Z = linkage(Dm, "average")
        order = [cc[i] for i in leaves_list(Z)]
        for t in (.1, .2, .3):
            lab = fcluster(Z, t, "distance"); cl = {}
            for c, l in zip(cc, lab): cl.setdefault(int(l), []).append(c)
            say(f"  clusters at 1 - rho <= {t}: " + " | ".join(",".join(v) for v in sorted(cl.values(), key=len, reverse=True)))
        say(f"  dendrogram order: {order}")
        summ[f"A2_{nm}"] = Rm.round(3).to_dict()
        if nm == "g":
            Rg = Rm

    # ---- D1 (registered): source of diversity, rho against CA (whCA = the WHISPER-02 run's own CA control)
    r = lambda a, b: float(Rg.loc[a, b])
    d_obj, d_in = r("whCA", "trCAW") - r("whCA", "rf_t3"), r("whCA", "rfnw_t3") - r("whCA", "rf_t3")
    verdict = ("objective" if d_obj >= .05 and d_obj >= d_in else "input" if d_in >= .05 and d_in > d_obj else "elsewhere")
    say(f"\n== D1 (README §3): rho(CA, CAW) {r('whCA', 'trCAW'):.3f}, rho(CA, rf_t3) {r('whCA', 'rf_t3'):.3f}, rho(CA, rfnw_t3) "
        f"{r('whCA', 'rfnw_t3'):.3f}, rho(CA, CAWO) {r('whCA', 'trCAWO'):.3f}; ceiling rho(CA run 1, CA run 2) {r('whCA', 'trCA'):.3f}")
    say(f"   delta_objective {d_obj:+.3f}, delta_input {d_in:+.3f} -> source of diversity: {verdict.upper()}")
    summ["D1"] = dict(d_obj=round(d_obj, 3), d_in=round(d_in, 3), verdict=verdict,
                      rho={k: round(r("whCA", k), 3) for k in ("trCAW", "trCAWO", "rf_t3", "rfnw_t3", "trCA", "trT3S", "trRFS")})

    # ---- fused arms
    say("\n== fused arms (CA-FUSE C = 1, nested + purged, 3 regimes)")
    arms = {"F1s": base + ["trCA"], "RFSf": ["trRFS"], "RFSnwf": ["trRFSnw"], "T3Sf": ["trT3S"],
            "F1sR": base + ["trCA", "trRFS"], "F1sRT": base + ["trCA", "trRFS", "trT3S"], "F1sT": base + ["trCA", "trT3S"],
            "F1sRraw": base + ["trCA"] + rf3, "ALLraw": base + ["trCA"] + rf3 + (br[:2] if FAST else br),
            "F1s+CAW": base + ["trCA", "trCAW"], "F1s+CAWO": base + ["trCA", "trCAWO"], "F1s+rft3": base + ["trCA", "rf_t3"],
            "F1s+rfnwt3": base + ["trCA", "rfnw_t3"], "F1s+T3Sw": base + ["trCA", "trT3Sw"], "F1s+T3Sr": base + ["trCA", "trT3Sr"]}
    arms.update({f"F1s+f_{f}": base + ["trCA", f"trT3Sf_{f}"] for f in FACES})
    arms.update({f"F1s+v_{v}": base + ["trCA", f"trT3Sv_{v}"] for v in VOICES})
    run_arms(D, S2, persons, arms)
    add_sum(D, "U", ["F1s", "RFSf"]); add_sum(D, "U3", ["F1s", "RFSf", "T3Sf"]); add_sum(D, "Unw", ["F1s", "RFSnwf"])
    fam = ["F1s", "RFSf", "RFSnwf", "T3Sf", "U", "U3", "Unw", "F1sR", "F1sRT", "F1sT", "F1sRraw", "ALLraw"]
    T = table(D, {c: (lambda g, c=c: g[f"{c}|oof"].values) for c in fam}).groupby("cell").mean(numeric_only=True)[fam]
    say("EER (oof, mean of folds):\n" + T.round(2).T.to_string())
    summ["fused_eer"] = T.round(3).to_dict()

    # ---- A3 complement to F1s (registered as reported-only), D2, D3
    say("\n== A3 complement: (F1s + X) - F1s, g cells (> 0 = X helps)")
    a3 = {}
    for a in [k for k in arms if k.startswith("F1s+")] + ["F1sR", "F1sT", "F1sRT"]:
        p, h = cmp(D, a, "F1s"); a3[a] = dict(g=round(float(p.mean()), 3), wins=int((p > 0).sum()), hold=h)
        say(f"  {a:<12} {fmt(p)} | held-out {h}")
    summ["A3"] = a3
    say("\n== D2 (README §3): one fuser for every stream vs the family sum, g cells")
    d2 = {}
    for a, b in (("F1sR", "U"), ("F1sRraw", "U"), ("F1sRT", "U3"), ("ALLraw", "U3")):
        p, h = cmp(D, a, b); d2[f"{a}-{b}"] = dict(g=round(float(p.mean()), 3), wins=int((p > 0).sum()), hold=h, passed=rule(p, h))
        say(f"  {a} - {b}: {fmt(p)} | held-out {h} -> {'PASS' if rule(p, h) else 'fail'}")
    summ["D2"] = d2
    say("\n== D3 (README §3): third family, g cells")
    d3 = {}
    for a, b in (("U3", "U"), ("F1sRT", "F1sR")):
        p, h = cmp(D, a, b); d3[f"{a}-{b}"] = dict(g=round(float(p.mean()), 3), wins=int((p > 0).sum()), hold=h, passed=rule(p, h))
        say(f"  {a} - {b}: {fmt(p)} | held-out {h} -> {'PASS' if rule(p, h) else 'fail'}")
    summ["D3"] = d3
    for a, b in (("U", "F1s"), ("U", "RFSf"), ("Unw", "U")):
        p, h = cmp(D, a, b); say(f"  (context) {a} - {b}: {fmt(p)} | held-out {h}")
    for nm, (a, b) in {"U-F1s": ("U", "F1s")}.items():
        lo, hi = S2.boot(D.rename(columns={f"{b}|oof": "ref_", f"{a}|oof": "cand_"}), "ref_", "cand_", gc, "both", B=NB)
        say(f"  CI95 person bootstrap {nm} (g cells): [{lo:+.2f}, {hi:+.2f}]")

    # ---- A4 (iii) greedy forward selection, nested in the outer fold, on top of S6 + v11 T3
    say("\n== A4 greedy forward selection (inner leave-one-fold-out on the 4 training folds, g-cell EER, step >= 0.05)")
    cand = ["trCA", "trRFS", "trT3S", "rf_t3", "rfnw_t3", "trCAW", "trCAWO", "trT3Sw", "trT3Sr"]
    picks, gre = {}, []

    def g_err(trn_rows, te_rows, streams):
        sc = S2.Design(trn_rows, te_rows, streams).score(streams, 1.0)
        tt = te_rows.assign(_s=sc)
        return float(np.nanmean([eer(g._s.values, g.y.values) for c_, g in tt.groupby("cell") if c_.endswith(" g")]))
    for s in sorted(D.fold.unique())[:1 if FAST else 5]:
        trf = [t for t in sorted(D.fold.unique()) if t != s]; chosen = []
        cur = np.mean([g_err(S2.purge(D[D.fold.isin([u for u in trf if u != t])], persons[int(t)]), D[D.fold == t], base) for t in trf])
        for _ in range(3):
            best = None
            for c in [c for c in cand if c not in chosen]:
                e = np.mean([g_err(S2.purge(D[D.fold.isin([u for u in trf if u != t])], persons[int(t)]), D[D.fold == t], base + chosen + [c]) for t in trf])
                if best is None or e < best[1]: best = (c, e)
            if best and cur - best[1] >= .05:
                chosen.append(best[0]); cur = best[1]
            else:
                break
        picks[int(s)] = chosen
        e_out = g_err(S2.purge(D[D.fold != s], persons[int(s)]), D[D.fold == s], base + chosen)
        e_f1s = g_err(S2.purge(D[D.fold != s], persons[int(s)]), D[D.fold == s], base + ["trCA"])
        gre.append(e_f1s - e_out); say(f"  fold {int(s)}: picked {chosen} | outer g EER {e_out:.2f} (F1s {e_f1s:.2f}, gain {e_f1s - e_out:+.2f})")
    say(f"  greedy - F1s mean {np.mean(gre):+.2f}, {sum(x > 0 for x in gre)}/5")
    summ["A4_greedy"] = dict(picks=picks, gain=[round(x, 3) for x in gre])
    say("  leave-one-family-out from F1sRT (each a CA-FUSE without that family):")
    lofo = {"-F1s": base + ["trRFS", "trT3S"], "-RFS": base + ["trCA", "trT3S"], "-T3S": base + ["trCA", "trRFS"], "-base": ["trCA", "trRFS", "trT3S"]}
    run_arms(D, S2, persons, {f"lofo{k}": v for k, v in lofo.items()})
    for k in lofo:
        p, h = cmp(D, "F1sRT", f"lofo{k}"); say(f"    F1sRT - ({k}): {fmt(p)} | held-out {h}")

    # ---- A5 strata (fused family scores, oof); MAV rows carry the voice duration / video
    say("\n== A5 strata: EER (mean of folds) of the family scores; paired gain U - F1s and RFSf - F1s with person-bootstrap CI")
    VM, FM = R5.VMETA, R5.FMETA; mav = D.data.values != "v4"
    D["dur"] = np.where(mav, pd.to_numeric(VM["dur"], errors="coerce").values[np.clip(D.vi.values, 0, len(VM) - 1)] if "dur" in VM else np.nan, np.nan)
    D["samevid"] = np.where(mav, FM.video.astype(str).values[np.clip(D.fi.values, 0, len(FM) - 1)] == VM.video.astype(str).values[np.clip(D.vi.values, 0, len(VM) - 1)], False)
    q = np.nanquantile(D.dur.values[mav], [1 / 3, 2 / 3]) if np.isfinite(D.dur.values[mav]).any() else [np.nan, np.nan]
    D["s_dur"] = np.where(~mav, "v4", np.where(D.dur < q[0], "dur short", np.where(D.dur < q[1], "dur mid", "dur long")))
    D["s_lang"] = np.where(D.data.isin(["v4", "en_ext"]), "english", "urdu/hindi")
    qa = np.nanquantile(D.agegap.values, [1 / 3, 2 / 3]); D["s_age"] = np.where(D.agegap < qa[0], "agegap low", np.where(D.agegap < qa[1], "agegap mid", "agegap high"))
    D["s_gen"] = np.where(D.pf_m > .5, "face male", "face female")
    D["s_pos"] = np.where(~mav, "v4", np.where(D.samevid, "MAV pos same video", "MAV pos other video"))
    say(f"  duration tertiles (MAV voices): {np.round(q, 2).tolist()} s; agegap tertiles {np.round(qa, 1).tolist()}")
    strata = {}
    for sv in ("s_dur", "s_lang", "s_age", "s_gen", "s_pos"):
        for prot in ("g", "ng"):
            sub = D[D.protocol == prot]
            if sv == "s_pos":   # positives of one type + every negative of the same fold
                parts = [pd.concat([sub[(sub.y == 1) & (sub[sv] == lv)], sub[sub.y == 0]]).assign(_st=lv) for lv in sub[sv].unique() if lv != "v4"]
                sub = pd.concat(parts) if parts else sub.iloc[:0].assign(_st="")
            else:
                sub = sub.assign(_st=sub[sv].values)
            for lv, g in sub.groupby("_st"):
                gg = g.assign(cell=f"{prot} {lv}")
                T5 = table(gg, {c: (lambda x, c=c: x[f"{c}|oof"].values) for c in ("F1s", "RFSf", "T3Sf", "U", "U3")})
                if T5.empty: continue
                m = T5.mean(numeric_only=True)
                lo, hi = S2.boot(gg.rename(columns={"F1s|oof": "ref_", "U|oof": "cand_"}), "ref_", "cand_", [f"{prot} {lv}"], "both", B=NB)
                lo2, hi2 = S2.boot(gg.rename(columns={"F1s|oof": "ref_", "RFSf|oof": "cand_"}), "ref_", "cand_", [f"{prot} {lv}"], "both", B=NB)
                strata[f"{prot} {lv}"] = dict(n=int(len(g)), **{c: round(float(m[c]), 2) for c in ("F1s", "RFSf", "T3Sf", "U", "U3")},
                                              U_minus_F1s_ci=[round(lo, 2), round(hi, 2)], RFS_minus_F1s_ci=[round(lo2, 2), round(hi2, 2)])
                say(f"  {prot:2s} {lv:<20} n {len(g):6d} | F1s {m.F1s:6.2f} RFS {m.RFSf:6.2f} T3S {m.T3Sf:6.2f} U {m.U:6.2f} U3 {m.U3:6.2f} | "
                    f"gain U-F1s CI [{lo:+.2f}, {hi:+.2f}] RFS-F1s CI [{lo2:+.2f}, {hi2:+.2f}]")
    summ["A5"] = strata

    # ---- A6 hard persons
    say("\n== A6 hard persons (error rate at each (fold, cell) EER point; a row counts for its face and voice person; >= 10 rows)")
    hard = {}
    for c in ("F1s", "RFSf", "T3Sf"):
        e = np.zeros(len(D)); ix = D.index
        for _, g in D.groupby(["fold", "cell"]):
            e[ix.get_indexer(g.index)] = thr_err(g, f"{c}|oof")
        per = pd.concat([pd.DataFrame(dict(p=D.fp.values, e=e)), pd.DataFrame(dict(p=D.vp.values, e=e))]).groupby("p").e.agg(["mean", "size"])
        per = per[per["size"] >= 10]; hard[c] = set(per["mean"].nlargest(max(1, int(.2 * len(per)))).index)
        say(f"  {c}: {len(per)} persons, mean error {per['mean'].mean():.3f}, top-20 % threshold {per['mean'].quantile(.8):.3f}")
    jac = lambda a, b: len(hard[a] & hard[b]) / max(1, len(hard[a] | hard[b]))
    say(f"  Jaccard of the hard-person sets: F1s/RFS {jac('F1s', 'RFSf'):.2f}, F1s/T3S {jac('F1s', 'T3Sf'):.2f}, RFS/T3S {jac('RFSf', 'T3Sf'):.2f} "
        f"(independent sets of 20 % -> about 0.11)")
    summ["A6_jaccard"] = {k: round(v, 3) for k, v in dict(F1s_RFS=jac("F1s", "RFSf"), F1s_T3S=jac("F1s", "T3Sf"), RFS_T3S=jac("RFSf", "T3Sf")).items()}

    # ---- A7 score shape + weights
    say("\n== A7 binormal check (g cells, per fold z-score by negatives): sigma1 / sigma0, d'; optimal weight vs 1 : 1")
    a7 = {}
    for c in ("F1s", "RFSf", "T3Sf"):
        rows = []
        for (s, cell), g in D[D.cell.isin(gc)].groupby(["fold", "cell"]):
            x0, x1 = g.loc[g.y == 0, f"{c}|oof"].values, g.loc[g.y == 1, f"{c}|oof"].values
            rows.append((x1.std() / x0.std(), (x1.mean() - x0.mean()) / x0.std()))
        a7[c] = dict(sig_ratio=round(float(np.mean([r_[0] for r_ in rows])), 3), dprime=round(float(np.mean([r_[1] for r_ in rows])), 3))
        say(f"  {c}: sigma1/sigma0 {a7[c]['sig_ratio']:.2f}, d' {a7[c]['dprime']:.2f}")
    from sklearn.linear_model import LogisticRegression
    D["Uw|oof"] = np.nan; ws = []
    for s in sorted(D.fold.unique()):
        tr, te = D[D.fold != s], D[D.fold == s]
        m = LogisticRegression(max_iter=2000).fit(tr[["F1s|oof", "RFSf|oof"]].values, tr.y.values); ws.append(m.coef_[0][1] / m.coef_[0][0])
        D.loc[te.index, "Uw|oof"] = m.decision_function(te[["F1s|oof", "RFSf|oof"]].values)
    for r_ in ("hold_urdu", "hold_hindi"):
        D[f"Uw|{r_}"] = D[f"U|{r_}"]
    p, h = cmp(D, "Uw", "U")
    say(f"  weight RFS / F1s per fold {np.round(ws, 2).tolist()}; learned weights - 1 : 1 (g cells): {fmt(p)}")
    a7["w_rfs_over_f1s"] = [round(float(x), 3) for x in ws]; a7["Uw_minus_U_g"] = round(float(p.mean()), 3)
    summ["A7"] = a7

    (OUT / "eda3fam_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (OUT / "eda3fam_report.txt").write_text("\n".join(TXT) + "\n", encoding="utf-8")
    keep = KEY + ["cell", "fp", "vp", "dur", "samevid", "agegap", "pf_m", "pv_m"] + [c for c in D.columns if c.endswith("|oof")] + \
        ["trCA", "whCA", "trCAW", "trCAWO", "trRFS", "trRFSnw", "trT3S", "rf_t3", "rfnw_t3"]
    D[[c for c in dict.fromkeys(keep) if c in D]].to_parquet(OUT / "eda3fam_oof.parquet")
    say("EDA-3FAM-01 report written")


if __name__ == "__main__":
    {"report": report, "merge": lambda: merge(_ct())}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
