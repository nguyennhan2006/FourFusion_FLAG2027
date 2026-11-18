r"""STACK-02 (registered 07/10 in README.md before any STACK-02 score): does a stack of many weak streams beat CA-FUSE once
every learned or selected step is inside the outer fold and the fuser never trains on the persons of its test fold?

Per regime (oof | hold_urdu | hold_hindi) and outer fold s:
  pool   = rows of folds != s, minus the held-out language (hold regimes), minus every row whose FACE or VOICE person
           appears in fold s (purge; only v4 persons can repeat across folds, audit_folds.txt)
  reps   = per family, K_REP columns by standalone g-cell EER on the pool (2nd needs |r| < R_MAX with the 1st)
  inner  = leave-one-fold-out over the pool's folds, each inner train purged of the inner test fold's persons;
           criterion = mean over the g cells of the mean over inner folds of EER (the gate's statistic)
  SEL    = backward elimination over families from the full eligible set (S6 always kept); the set on the path with
           the lowest criterion is chosen, the full set is a candidate, exact ties go to the larger set
  arms   = BASE6 (CA-FUSE, C = 1: the reference), BASE6_C005, ALL_<pool>, SEL_<pool> (C = 0.05), HRCLEAN (HEADROOM-01's
           fixed set), and unpurged BASE6_np / HRCLEAN_np (diagnostic: size of the person leak)
    python run.py <regime> <poolset>      poolset = novox2 (primary) | vox2 (secondary) | diag
    python run.py report > report.txt
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "SAFE-04"))
from lab_fuse import SP, eer, table
from pools import POOLS, K_REP, R_MAX, C_STACK

S6 = ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "rel", "relg"]
CO = ["is_en", "pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]
HELD = {"oof": (), "hold_urdu": ("urdu",), "hold_hindi": ("hindi",)}
HRCLEAN = S6 + ["pe", "pe4", "relgS_pe", "li_learn", "li_mm3", "A1_cfe", "A2_cfe", "relS_wsr221", "relgS_wssim34",
                "relgS_wlmlg_L10", "d_SPX", "ra_r64", "d_XV5K", "ad_s2000_a0.5", "ck_cmpc", "ck_wen", "ovl_age", "relc_all_eb"]
OUT = SP / "out/safe04/stack02"
ELIG = json.loads((HERE / "eligible.json").read_text())        # written from the provenance table, before any score


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gcells(df):
    return sorted(c for c in df.cell.unique() if c.endswith(" g"))


def purge(df, persons):
    return df[~df.fp.isin(persons) & ~df.vp.isin(persons)]


class Design:
    """Standardised CA-FUSE design [streams | CO | streams x CO] for one (train, test) split; any stream subset is a
    column slice (the scaler is per column, so slicing = refitting it on the subset)."""
    def __init__(self, trn, tst, streams):
        self.pos = {c: i for i, c in enumerate(streams)}; ns = len(streams)
        def X(df):
            s, c = df[streams].values, df[CO].values
            return np.concatenate([s, c, (s[:, :, None] * c[:, None, :]).reshape(len(df), -1)], 1)
        sc = StandardScaler().fit(X(trn))
        self.Xtr, self.Xte = sc.transform(X(trn)), sc.transform(X(tst))
        self.ytr, self.ns = trn.y.values, ns

    def idx(self, st):
        I = [self.pos[c] for c in st]; k = len(CO)
        return I + list(range(self.ns, self.ns + k)) + [self.ns + k + i * k + j for i in I for j in range(k)]

    def score(self, st, C):
        j = self.idx(st)
        m = LogisticRegression(C=C, max_iter=8000).fit(self.Xtr[:, j], self.ytr)
        return m.decision_function(self.Xte[:, j])


def g_eer(df, col):
    """mean over the g cells of the mean over folds of EER (the gate's statistic)."""
    T = table(df, {col: lambda g: g[col].values})
    T = T[T.cell.str.endswith(" g")]
    return T.groupby("cell")[col].mean().mean()


def choose_reps(pool, fams):
    """K_REP representatives per family by standalone (orientation-free) g-cell EER on the pool rows."""
    reps = {}
    for f in fams:
        cols = POOLS[f]["cols"]
        T = table(pool, {c: (lambda g, c=c: g[c].values) for c in cols})
        T = T[T.cell.str.endswith(" g")]
        e = T.groupby("cell")[cols].mean().mean()
        e = np.minimum(e, 100 - e).sort_values()
        pick = [e.index[0]]
        for c in e.index[1:]:
            if len(pick) >= min(K_REP, len(cols)):
                break
            if all(abs(np.corrcoef(pool[c].values, pool[p].values)[0, 1]) < R_MAX for p in pick):
                pick.append(c)
        reps[f] = pick
    return reps


def inner_designs(pool, streams):
    out = []
    for u in sorted(pool.fold.unique()):
        te = pool[pool.fold == u]
        tr = purge(pool[pool.fold != u], set(te.fp) | set(te.vp))
        out.append((Design(tr, te, streams), te[["fold", "cell", "y"]].reset_index(drop=True)))
    return out


def crit(designs, st, C):
    vals = []
    for d, te in designs:
        t = te.copy(); t["s"] = d.score(st, C); vals.append(t)
    return g_eer(pd.concat(vals, ignore_index=True), "s")


def backward(designs, fams, reps):
    cur = list(fams); cols = lambda fs: S6 + [c for f in fs for c in reps[f]]
    path = [(list(cur), crit(designs, cols(cur), C_STACK))]
    while cur:
        trial = [(f, crit(designs, cols([x for x in cur if x != f]), C_STACK)) for f in cur]
        f, v = min(trial, key=lambda t: t[1])
        cur = [x for x in cur if x != f]; path.append((list(cur), v))
    best = min(range(len(path)), key=lambda i: (round(path[i][1], 10), i))     # ties -> earlier = larger set
    return path[best][0], path


def run(regime, poolset):
    D = pd.read_parquet(SP / "out/safe04/stack02_rows.parquet")
    persons = {int(k): set(v) for k, v in json.loads((HERE / "fold_persons.json").read_text()).items()}
    held = HELD[regime]; OUT.mkdir(parents=True, exist_ok=True)
    fams = [f for f in ELIG["families"] if poolset != "novox2" or not POOLS[f]["vox2"]]
    res, sel = pd.DataFrame(index=D.index), {}
    for s in sorted(D.fold.unique()):
        t0 = time.time()
        te = D[(D.fold == s) & (D.data.isin(held) if held else True)]
        base = D[(D.fold != s) & ~D.data.isin(held)]
        pool = purge(base, persons[int(s)])
        if poolset == "diag":
            for name, st, C, trn in (("BASE6", S6, 1.0, pool), ("BASE6_C005", S6, C_STACK, pool), ("HRCLEAN", HRCLEAN, C_STACK, pool),
                                     ("BASE6_np", S6, 1.0, base), ("HRCLEAN_np", HRCLEAN, C_STACK, base)):
                res.loc[te.index, name] = Design(trn, te, st).score(st, C)
            log(f"{regime} diag fold {s}: pool {len(pool)} of {len(base)} rows [{time.time() - t0:.0f}s]")
            continue
        reps = choose_reps(pool, fams)
        streams = S6 + [c for f in fams for c in reps[f]]
        designs = inner_designs(pool, streams)
        chosen, path = backward(designs, fams, reps)
        outer = Design(pool, te, streams)
        res.loc[te.index, f"ALL_{poolset}"] = outer.score(streams, C_STACK)
        res.loc[te.index, f"SEL_{poolset}"] = outer.score(S6 + [c for f in chosen for c in reps[f]], C_STACK)
        sel[int(s)] = dict(reps=reps, chosen=chosen, path=[(p, round(v, 3)) for p, v in path], pool_rows=len(pool))
        log(f"{regime} {poolset} fold {s}: chose {len(chosen)}/{len(fams)} {chosen} | inner full {path[0][1]:.2f} "
            f"best {min(v for _, v in path):.2f} S6-only {path[-1][1]:.2f} [{time.time() - t0:.0f}s]")
    res.to_parquet(OUT / f"{regime}_{poolset}.parquet")
    if sel:
        (HERE / f"selections_{regime}_{poolset}.json").write_text(json.dumps(sel, indent=1))


def weer(s, y, w):
    """EER with integer row weights (= repeating rows); same linear interpolation as lab_fuse.eer."""
    o = np.argsort(-s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    tp, fp = np.cumsum(w * (y == 1)), np.cumsum(w * (y == 0))
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, fp = np.r_[0, tp[last]], np.r_[0, fp[last]]
    if tp[-1] == 0 or fp[-1] == 0:
        return np.nan
    fpr, d = fp / fp[-1], (1 - tp / tp[-1]) - fp / fp[-1]
    i = int(np.argmax(d <= 0))
    if d[i] == 0 or i == 0:
        return 100 * fpr[i]
    u = d[i - 1] / (d[i - 1] - d[i])
    return 100 * (fpr[i - 1] + u * (fpr[i] - fpr[i - 1]))


def boot(D, ref, col, cells, sides, B=1000, seed=0):
    """Paired person bootstrap of the gain EER(ref) - EER(col): inside every (fold, cell) persons are resampled with
    replacement (the same draw for both arms). sides = "voice": a row weighs m(voice person) (= stream_gate._boot_gain);
    sides = "both": m(face person) x m(voice person) (pigeonhole bootstrap, one person factor on both pair sides).
    A replicate = mean over cells of the mean over folds. -> (2.5 %, 97.5 %)."""
    rng = np.random.RandomState(seed); groups = []
    for (s, cell), g in D[D.cell.isin(cells)].groupby(["fold", "cell"]):
        if g[[ref, col]].notna().all().all() and g.y.nunique() == 2:
            up = np.unique(np.concatenate([g.fp.values, g.vp.values]) if sides == "both" else g.vp.values)
            groups.append((cell, len(up), np.searchsorted(up, g.fp.values) if sides == "both" else None,
                           np.searchsorted(up, g.vp.values), g[ref].values, g[col].values, g.y.values))
    reps = []
    for _ in range(B):
        per = {}
        for cell, n, fi, vi, a, c, y in groups:
            m = np.bincount(rng.randint(n, size=n), minlength=n)
            w = m[vi] * (m[fi] if fi is not None else 1)
            d = weer(a, y, w) - weer(c, y, w)
            if np.isfinite(d):
                per.setdefault(cell, []).append(d)
        reps.append(np.mean([np.mean(v) for v in per.values()]))
    return np.percentile(reps, [2.5, 97.5])


def report():
    D = pd.read_parquet(SP / "out/safe04/stack02_rows.parquet")[["fold", "data", "cell", "y", "fp", "vp"]]
    R = {}
    for regime in HELD:
        parts = [pd.read_parquet(f) for f in sorted(OUT.glob(f"{regime}_*.parquet"))]
        R[regime] = pd.concat([D] + parts, axis=1)
    arms = [c for c in R["oof"].columns if c not in D.columns]
    print("== arms:", arms)
    print("== standalone check: BASE6_np must reproduce HEADROOM-01's CA (S6) oof; HRCLEAN_np its CLEAN gain (+1.02 g)\n")
    rows = []
    for regime, Dr in R.items():
        T = table(Dr, {a: (lambda g, a=a: g[a].values) for a in arms if Dr[a].notna().any()})
        for ref, cand in [("BASE6", a) for a in arms if a not in ("BASE6", "BASE6_np")] + [("BASE6_np", "HRCLEAN_np")]:
            if cand not in T or ref not in T:
                continue
            sub = T[["fold", "cell", ref, cand]].dropna()
            d = (sub[ref] - sub[cand]).groupby(sub.cell)
            m, w = d.mean(), d.apply(lambda x: int((x > 0).sum()))
            gc = [k for k in m.index if k.endswith(" g")]
            rows.append(dict(regime=regime, ref=ref, arm=cand, all=m.mean(), g=m[gc].mean() if gc else np.nan,
                             g_w4=int(sum(w[k] >= 4 for k in gc)), g_no_v4=m[[k for k in gc if k != "v4 g"]].mean() if gc else np.nan,
                             all_no_v4=m[[k for k in m.index if not k.startswith("v4")]].mean(),
                             cells=" ".join(f"{k.replace(' ', '/')} {m[k]:+.2f}({w[k]})" for k in m.index)))
    G = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    for regime in HELD:
        print(f"== {regime}: gain vs reference (+ = better), per cell mean (folds better)")
        print(G[G.regime == regime].drop(columns="regime").round(2).to_string(index=False), "\n")
    gc = gcells(R["oof"])
    for arm in [a for a in arms if a.startswith(("SEL_", "ALL_", "HRCLEAN", "BASE6_C"))]:
        g = G[(G.arm == arm) & (G.ref == "BASE6")].set_index("regime")
        if "oof" not in g.index:
            continue
        ok = (g.loc["oof", "g"] >= 1.0 and g.loc["oof", "g_w4"] >= 2
              and all(r in g.index and g.loc[r, "all"] >= -0.2 for r in ("hold_urdu", "hold_hindi")))
        lo1, hi1 = boot(R["oof"], "BASE6", arm, gc, "voice")
        lo2, hi2 = boot(R["oof"], "BASE6", arm, gc, "both")
        print(f"{arm:<12} oof g {g.loc['oof', 'g']:+.2f} (g cells >= 4/5: {g.loc['oof', 'g_w4']}) | hold urdu "
              f"{g['all'].get('hold_urdu', np.nan):+.2f} hindi {g['all'].get('hold_hindi', np.nan):+.2f} | 95 % CI voice-person "
              f"[{lo1:+.2f}, {hi1:+.2f}], both sides [{lo2:+.2f}, {hi2:+.2f}] -> {'PASSES' if ok else 'fails'} the registered gate"
              + ("   <- PRIMARY" if arm == "SEL_novox2" else ""), flush=True)
    print("\n== selections (families chosen per outer fold)")
    for f in sorted(HERE.glob("selections_*.json")):
        S = json.loads(f.read_text())
        print(f"  {f.stem[11:]}: " + " | ".join(f"f{s} {len(v['chosen'])}" for s, v in S.items()))
        cnt = pd.Series([x for v in S.values() for x in v["chosen"]]).value_counts()
        print("    family kept in n of 5 folds:", cnt.to_dict())


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        run(sys.argv[1], sys.argv[2])
