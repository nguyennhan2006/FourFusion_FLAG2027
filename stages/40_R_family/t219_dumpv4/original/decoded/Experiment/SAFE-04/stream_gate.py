r"""Shared gate for a NEW level-0 stream (PEAV-01, LAYERID-01, LOCALID-01; registered 03/10, the rule of CFE-01):

  KEEP a stream only if CA-FUSE + stream beats CA-FUSE (BEST v8's Bangla fuser, refit per fold) by >= 1.0 EER on the
  g cells (mean over the g cells, with >= 4/5 folds better in >= 2 of them) AND is not worse than -0.2 on average in
  either held-out-language check (Urdu or Hindi removed from the fuser's training rows, scored on the removed language).

Also printed: the standalone EER of each new stream next to XALL_ib (ImageBind zero-shot) and XALL_deep0 (bridge); a
95 % person-cluster bootstrap interval of the out-of-fold g-cell gain (voice persons resampled inside every fold and cell;
added 04/10, informative only, the registered rule above is unchanged); and the identity-hard subsets (y=1 with a large
face-voice age gap, y=0 with the same gender and a small gap) with their class counts, both with the raw gap and with the
gap corrected for the face-voice age-estimator offset b (~ -10.6 y, from the fold's v4 train persons; LANGNORM-01): with
the raw gap a typical TRUE pair already has |gap| ~ 10 y, so the raw-gap hard set is distorted (review of 04/10).
Input: one parquet per fold with the new columns, row-aligned with SAFE-04's streams_fold<s>.parquet. Nothing reads dev.
    from stream_gate import load, gate; D = load("peav01"); gate(D, ["pe", "pe4"])
"""
import glob, os
import numpy as np, pandas as pd

S6 = ["XALL_deep0", "XALL_cca", "XALL_ib", "XALL_age", "rel", "relg"]
if os.environ.get("NOREL") == "1":   # NOREL-01 (5/11, BTC Rules & FAQ): no cohort-based normalisation even with a train-only cohort ->
    S6 = S6[:4]                       # rel / relg (train-identity anchors, Z-normed per anchor) leave every fuser; unset = unchanged
CO = ["is_en", "pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]


def load(tag):
    """SAFE-04 streams + rel / relg + the new per-fold parquet <scratch>/out/safe04/<tag>_fold<s>.parquet."""
    from rel_sweep5 import SP, V4, FA, VA, V4S, VMETA, make_fold
    parts = []
    for f in sorted(glob.glob(str(SP / f"out/safe04/{tag}_fold*.parquet"))):
        s = int(f.split("fold")[-1].split(".")[0])
        a = pd.read_parquet(SP / f"out/safe04/streams_fold{s}.parquet")
        a["rel"] = pd.read_parquet(SP / f"out/safe04/relonly_multi_fold{s}.parquet").relc_fia_eb.values
        a["relg"] = pd.read_parquet(SP / f"out/safe04/relonly_g_fold{s}.parquet").relg_fia_eb.values
        c = pd.read_parquet(f)
        assert len(a) == len(c), (f, len(a), len(c))
        if {"y", "fi", "vi"} <= set(c.columns):
            assert (c.y.values == a.y.values).all() and (c.fi.values == a.fi.values).all() and (c.vi.values == a.vi.values).all(), f
        parts.append(pd.concat([a, c.drop(columns=[k for k in c.columns if k in a.columns])], axis=1))
    assert parts, f"no {tag}_fold*.parquet in {SP / 'out/safe04'}"
    D = pd.concat(parts, ignore_index=True); D["cell"] = D.data + " " + D.protocol
    v4m = D.data.values == "v4"
    pick = lambda A4, AE, idx: np.where(v4m, A4[np.clip(idx, 0, len(A4) - 1)], AE[np.clip(idx, 0, len(AE) - 1)])
    D["agef"], D["agev"] = pick(V4["agef"], FA["agef"], D.fi.values), pick(V4["agev"], VA["agev"], D.vi.values)
    D["agegap"], D["gdiff"] = np.abs(D.agef - D.agev), np.abs(D.pf_m - D.pv_m)
    D["is_en"] = D.data.isin(["v4", "en_ext"]).astype(float)
    D["vp"] = np.where(v4m, V4S[np.clip(D.vi.values, 0, len(V4S) - 1)], VMETA.p.values[np.clip(D.vi.values, 0, len(VMETA) - 1)])
    af4, av4 = np.asarray(V4["agef"], np.float64).reshape(-1), np.asarray(V4["agev"], np.float64).reshape(-1)
    bmap = {}
    for s in sorted(D.fold.unique()):                # face-voice age offset from the fold's v4 train persons only
        tr = make_fold(int(s))["v4tr"]
        bmap[s] = float(np.mean([af4[tr][V4S[tr] == p].mean() - av4[tr][V4S[tr] == p].mean() for p in np.unique(V4S[tr])]))
    D["agegap_c"] = np.abs(D.agef - D.agev - D.fold.map(bmap))
    return D


def _X(df, st):
    return np.concatenate([df[st].values, df[CO].values, (df[st].values[:, :, None] * df[CO].values[:, None, :]).reshape(len(df), -1)], 1)


def _ca(trn, tst, st):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(_X(trn, st))
    return LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(_X(trn, st)), trn.y.values).decision_function(sc.transform(_X(tst, st)))


def _boot_gain(D, ref, col, cells, B=200, seed=0):
    """Person-cluster bootstrap of the paired gain EER(ref) - EER(col): voice persons resampled with replacement inside
    every (fold, cell); a replicate = mean over cells of the mean over folds. -> (2.5 %, 97.5 %) percentiles."""
    from lab_fuse import eer
    rng = np.random.RandomState(seed)
    groups = []
    for (s, cell), g in D[D.cell.isin(cells)].groupby(["fold", "cell"]):
        if g[col].notna().all() and g.y.nunique() == 2:
            up = np.unique(g.vp.values)
            groups.append((cell, up, {p: np.where(g.vp.values == p)[0] for p in up}, g[ref].values, g[col].values, g.y.values))
    reps = []
    for _ in range(B):
        per = {}
        for cell, up, idx, a, c, y in groups:
            ii = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
            d = eer(a[ii], y[ii]) - eer(c[ii], y[ii])
            if np.isfinite(d):
                per.setdefault(cell, []).append(d)
        if per:
            reps.append(np.mean([np.mean(v) for v in per.values()]))
    return np.percentile(reps, [2.5, 97.5]) if reps else (np.nan, np.nan)


def gate(D, new, sets=None, hard=True):
    """new: list of new stream columns; sets: {name: list of streams added to S6} (default: each new stream alone).
    Returns {set name: verdict dict}."""
    from lab_fuse import table
    sets = sets or {n: [n] for n in new}
    D = D.copy()
    for c in new:                                   # a stream with missing rows cannot enter the fuser: report coverage
        cov = np.isfinite(D[c].values).mean()
        if cov < 1:
            print(f"!! {c}: {100 * cov:.1f} % of rows finite; missing rows set to 0 (= the train mean of a z-scored stream)")
            D[c] = D[c].fillna(0.0)
    all_sets = {"CA": S6, **{f"CA+{k}": S6 + v for k, v in sets.items()}}
    for regime, held in (("oof", ()), ("hold_urdu", ("urdu",)), ("hold_hindi", ("hindi",))):
        for name, st in all_sets.items():
            col = f"{name}|{regime}"; D[col] = np.nan
            for s in sorted(D.fold.unique()):
                trn = D[(D.fold != s) & ~D.data.isin(held)]
                te = (D.fold == s) & (D.data.isin(held) if held else True)
                D.loc[te, col] = _ca(trn, D[te], st)
    rec = {c: (lambda g, c=c: g[c].values) for c in D.columns if "|" in c}
    rec.update({n: (lambda g, n=n: g[n].values) for n in new + ["XALL_ib", "XALL_deep0", "rel"]})
    T = table(D, rec)
    M = T.groupby("cell").mean(numeric_only=True)
    print("== standalone streams, EER mean of 5 folds (XALL_ib = ImageBind zero-shot, the stream to beat)")
    print(M[["XALL_ib", "XALL_deep0", "rel"] + new].round(2).to_string())
    print("\n== CA-FUSE + new stream(s): gain vs CA-FUSE in the same regime (+ = better), per cell mean (folds better, folds >= 1)")
    verdict = {}
    for name in sets:
        res = {}
        for regime in ("oof", "hold_urdu", "hold_hindi"):
            ref, c = f"CA|{regime}", f"CA+{name}|{regime}"
            sub = T[["cell", ref, c]].dropna()
            d = sub[ref] - sub[c]; g = d.groupby(sub.cell)
            m, w, w1 = g.mean(), g.apply(lambda x: (x > 0).sum()), g.apply(lambda x: (x >= 1).sum())
            gcells = [k for k in m.index if k.endswith(" g")]
            res[regime] = dict(mean=m.mean(), g=m[gcells].mean(), g_w4=int(sum(w[k] >= 4 for k in gcells)))
            print(f"  {regime:>10} CA+{name:<16}: all {m.mean():+.2f} | g {m[gcells].mean():+.2f} | "
                  + " ".join(f"{k.replace(' ', '/')} {m[k]:+.2f}({w[k]},{w1[k]})" for k in m.index))
        ok = res["oof"]["g"] >= 1.0 and res["oof"]["g_w4"] >= 2 and res["hold_urdu"]["mean"] >= -0.2 and res["hold_hindi"]["mean"] >= -0.2
        gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
        lo, hi = _boot_gain(D, "CA|oof", f"CA+{name}|oof", gc)
        verdict[name] = dict(res, keep=bool(ok), g_ci95=(float(lo), float(hi)))
        print(f"  oof g-cell gain {res['oof']['g']:+.2f}, 95 % person-bootstrap interval [{lo:+.2f}, {hi:+.2f}]")
        print(f"  -> CA+{name}: {'KEEP (passes the registered gate)' if ok else 'close (fails the registered gate)'}\n")
    if hard:
        for col, gap, what in (("hard", "agegap", "raw age gap"), ("hard_c", "agegap_c", "age gap corrected for the offset b")):
            D[col] = ((D.y == 1) & (D[gap] > 15)) | ((D.y == 0) & (D.gdiff < 0.5) & (D[gap] < 5))
            Hd = D[D[col]]
            H = table(Hd, {k: rec[k] for k in ["CA|oof"] + [f"CA+{n}|oof" for n in sets]})
            cnt = Hd.groupby("cell").y.agg(pos="sum", n="count"); cnt["neg"] = cnt.n - cnt.pos
            print(f"== identity-hard subset, {what} (y=1 & gap > 15 y; y=0 & same gender & gap < 5 y): EER mean of folds")
            print(H.groupby("cell").mean(numeric_only=True).drop(columns=["fold", "n"]).round(2)
                  .join(cnt[["pos", "neg"]].astype(int)).to_string())
    return verdict
