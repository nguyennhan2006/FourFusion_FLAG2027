r"""E2E lab library (docs/PLAN_E2E_2026-10-10.md): the parts every end-to-end experiment notebook (FLAG_51 / 52 / 53,
kaggle/build_e2e_lab.py) shares, so that one Kaggle run goes from the lab features to a gate verdict AND a dev
submission.zip, with no step on the local machine.

  components()        the per-file inputs of the current system on the test split (components_<split>.npz of
                      `flag2027-e2e-dev` for dev, or of a FLAG_50 run for the Evaluation data): pair ids, the 6 B3 streams +
                      7 CA-FUSE conditions, the 3 T3 streams of v11's head, the 7 CA-EN streams + 6 conditions, v11 / v9 scores,
                      raw T3 features and w2v2 gender probabilities of every trial
  t3_split()          the T3 face / voice arrays of one test file, normalised exactly as T3ENC-01/run.py t3()
  gate()              the registered default gate against v11 (CLAUDE.md section 4.3), purged, in the notebook
  fit_fuser()         CA-FUSE on every SAFE-04 row (= BIGB-01/build_v11.py fit), any stream list
  bangla_scores() / write_zip() / zip_report()   the dev / eval zip (English files unchanged unless asked)
Paths: E2E_DIR (folder with e2e_manifest.json + components_<split>.npz + fusers/ + ref/), SAFE04_SCRATCH (the lab scratch
whose out/safe04 holds the SAFE-04 reference tables; the notebook copies them from E2E_DIR/safe04).
"""
import json, os, sys, time, zipfile, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
E2E = Path(os.environ.get("E2E_DIR", ""))
SPLIT = os.environ.get("E2E_SPLIT", "dev")
NOREL = os.environ.get("NOREL") == "1"   # NOREL-01 (5/11, BTC Rules & FAQ): no train-identity cohort at test time -> rel / relg /
V11_T3 = ["anP512_deep", "anP512_cca"] + ([] if NOREL else ["t3_relg"])   # t3_relg leave every fuser (stream_gate.S6 = 4); unset = unchanged
B3_KEEP = 4 if NOREL else 6                                 # columns of the components' b3 block a fuser sees (deep, cca, ib, age[, rel, relg])
CO = ["is_en", "pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"]
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
design = lambda Z, C: np.concatenate([Z, C, (Z[:, :, None] * C[:, None, :]).reshape(len(Z), -1)], 1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ components of the test split
def manifest():
    return json.loads((E2E / "e2e_manifest.json").read_text())


_C = {}


def components():
    """{i: dict(key, zip_name, is_en, pid, b3, co, en7, enco, t3, v11, v9, vgen, face_farl16, ...)} in zip order."""
    if "c" not in _C:
        man = manifest()
        z = np.load(E2E / f"components_{man['split']}.npz", allow_pickle=False)
        out = {}
        for f in man["files"]:
            i = f["i"]
            out[i] = dict(f, **{k[: -len(f"_{i}")]: z[k] for k in z.files if k.endswith(f"_{i}")})
        _C["c"] = out
    return _C["c"]


def pmale(g3):
    g = np.asarray(g3, np.float64)
    return g[:, 1] / (g[:, 0] + g[:, 1] + 1e-9)


def t3_split(c):
    """(faces, voices, voice-male mask) of one test file: per-encoder L2 norm, then concatenation (= T3ENC-01 t3())."""
    F = np.concatenate([l2n(c["face_farl16"].astype(np.float32)), l2n(c["face_clipL"].astype(np.float32))], 1)
    V = np.concatenate([l2n(c["voice_sam100p"].astype(np.float32)), l2n(c["voice_sam100f"].astype(np.float32))], 1)
    return F, V, pmale(c["vgen"]) > 0.5


# ============================================================================ SAFE-04 tables + gate
def stack02():
    """STACK-02/run.py (purge, Design, boot, HELD), loaded by path as in ALLNEG-01 report()."""
    sys.path.insert(0, str(EXP / "STACK-02"))
    spec = importlib.util.spec_from_file_location("stack02run", EXP / "STACK-02/run.py")
    S2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S2)
    return S2


def safe04(tag):
    """SAFE-04 table of experiment `tag` (stream_gate.load: S6 + rel / relg + the tag's columns) with v11's T3 streams
    (anP512 of bigb01f, t3_relg of t3enc01f), the face person column (fp) and the fold persons."""
    from stream_gate import load
    from lab_fuse import SP
    D = load(tag)
    B = pd.concat([pd.read_parquet(f) for f in sorted((SP / "out/safe04").glob("bigb01f_fold*.parquet"))], ignore_index=True)
    T3 = pd.concat([pd.read_parquet(f) for f in sorted((SP / "out/safe04").glob("t3enc01f_fold*.parquet"))], ignore_index=True)
    folds = sorted(D.fold.unique())
    B, T3 = B[B.fold.isin(folds)].reset_index(drop=True), T3[T3.fold.isin(folds)].reset_index(drop=True)
    for X in (B, T3):
        assert len(X) == len(D) and (X.y.values == D.y.values).all() and (X.vi.values == D.vi.values).all(), "row alignment"
    D["anP512_deep"], D["anP512_cca"], D["t3_relg"] = B.anP512_deep.values, B.anP512_cca.values, T3.t3_relg.values
    R2 = pd.read_parquet(SP / "out/safe04/stack02_rows.parquet")
    R2 = R2[R2.fold.isin(folds)].reset_index(drop=True)
    assert len(R2) == len(D) and (R2.y.values == D.y.values).all()
    D["fp"] = R2.fp.values
    return D


def fold_persons():
    from lab_fuse import SP
    for p in (SP / "out/safe04/fold_persons.json", EXP / "STACK-02/fold_persons.json"):
        if p.exists():
            return {int(k): set(v) for k, v in json.loads(p.read_text()).items()}
    raise FileNotFoundError("fold_persons.json")


def gate(tag, arms, decision, standalone=(), out=None, ci_arms=None, prep=None):
    """Registered default gate (CLAUDE.md 4.3) of CA-FUSE(arms[a]) against v11 = CA-FUSE(S6 + anP512_deep, anP512_cca,
    t3_relg), C = 1, purged, regimes oof / hold Urdu / hold Hindi. arms = {name: full stream list}. Prints and returns
    {arm: verdict}; `out` = text file that receives the same report. ci_arms = arms that get the (slow) person bootstrap
    (default: all); prep(D) -> D adds columns computed by the experiment (e.g. a nested selection)."""
    import io, contextlib
    from stream_gate import S6
    from lab_fuse import table
    S2 = stack02()
    D = safe04(tag)
    if prep is not None:
        D = prep(D)
    persons = fold_persons()
    sets = {"v11": S6 + V11_T3, **arms}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"== {tag}: folds {sorted(int(s) for s in D.fold.unique())}, {len(D)} trials; reference v11 = S6 + {V11_T3}")
        for c in sorted({c for st in arms.values() for c in st} - set(sets["v11"])):
            f = np.isfinite(D[c].values).mean()
            if f < 1:
                print(f"!! {c}: {100 * f:.1f} % finite; missing set to 0"); D[c] = D[c].fillna(0.0)
        cols = ["anP512_deep"] + [c for c in standalone if c in D]
        if cols:
            print("\n== standalone EER (mean of folds); anP512_deep = v11's T3 head")
            print(table(D, {c: (lambda g, c=c: g[c].values) for c in cols}).groupby("cell").mean(numeric_only=True)[cols].round(2).to_string())
        for regime, held in S2.HELD.items():
            for s in sorted(D.fold.unique()):
                te = D[(D.fold == s) & (D.data.isin(held) if held else True)]
                pool = S2.purge(D[(D.fold != s) & ~D.data.isin(held)], persons[int(s)])
                for name, st in sets.items():
                    col = f"{name}|{regime}"
                    if col not in D:
                        D[col] = np.nan
                    D.loc[te.index, col] = S2.Design(pool, te, st).score(st, 1.0)
        T2 = table(D, {k: (lambda g, k=k: g[k].values) for k in D.columns if "|" in k})
        gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
        print("\n== gate: CA-FUSE(arm) vs v11, purged; + = better; per cell: mean gain (folds better)")
        verdicts = {}
        for name in arms:
            res = {}
            for regime in S2.HELD:
                sub = T2[["cell", "fold", f"v11|{regime}", f"{name}|{regime}"]].dropna()
                d = (sub[f"v11|{regime}"] - sub[f"{name}|{regime}"]); g_ = d.groupby(sub.cell)
                m, w = g_.mean(), g_.apply(lambda x: int((x > 0).sum()))
                worse = [k for k in m.index if g_.get_group(k).le(-0.5).sum() >= 4]
                g = [k for k in m.index if k.endswith(" g")]
                res[regime] = dict(all=float(m.mean()), g=float(m[g].mean()) if g else np.nan, w4=int(sum(w[k] >= 4 for k in g)),
                                   g_no_v4=float(m[[k for k in g if k != "v4 g"]].mean()) if g else np.nan,
                                   cells={k: (round(float(m[k]), 2), int(w[k])) for k in m.index}, worse_ge05_4of5=worse)
            ok = (res["oof"]["g"] >= 1.0 and res["oof"]["w4"] >= 2 and res["hold_urdu"]["all"] >= -0.2
                  and res["hold_hindi"]["all"] >= -0.2)
            if ci_arms is None or name in ci_arms:
                D2 = D.rename(columns={"v11|oof": "ref_", f"{name}|oof": "cand_"})
                lo, hi = S2.boot(D2, "ref_", "cand_", gc, "both")
            else:
                lo, hi = np.nan, np.nan
            verdicts[name] = dict(res, passes=bool(ok), ci95_both=(float(lo), float(hi)), decision=name == decision)
            o = res["oof"]
            print(f"  {name:<10} oof all {o['all']:+.2f} | g {o['g']:+.2f} (>= 4/5 in {o['w4']} g cells; without v4 {o['g_no_v4']:+.2f}) | "
                  f"hold urdu {res['hold_urdu']['all']:+.2f} hindi {res['hold_hindi']['all']:+.2f} | CI both [{lo:+.2f}, {hi:+.2f}] -> "
                  f"{'PASSES' if ok else 'fails'}" + ("   <- DECISION" if name == decision else ""))
            print("             " + " ".join(f"{k.replace(' ', '/')} {v[0]:+.2f}({v[1]})" for k, v in o["cells"].items()))
            if o["worse_ge05_4of5"]:
                print(f"             cells worse by >= 0.5 in >= 4/5 folds: {o['worse_ge05_4of5']}")
    txt = buf.getvalue(); print(txt, end="")
    if out:
        Path(out).write_text(txt, encoding="utf-8")
    return verdicts


# ============================================================================ final fuser + zip
def mav_flagged(D):
    """LID_STRICT=1 (5/11): True for the SAFE-04 pairs of D whose voice is one of lid_strict.json's Whisper-flagged MAV files
    (MAV_DROP of the lab LOAD; 'vi' of a non-v4 row indexes VMETA = v1 then v2). Final fusers are not fitted on them (40 of ~37k
    pairs of the ct01 tables). All False without LID_STRICT, so nothing changes then."""
    if os.environ.get("LID_STRICT") != "1":
        return np.zeros(len(D), bool)
    import lab_streams as L
    bad = np.isin(L.VMETA.path.values, list(L.MAV_DROP))
    assert int(bad.sum()) == len(L.MAV_DROP) > 0, f"MAV_DROP matched {int(bad.sum())} of {len(L.MAV_DROP)} VMETA rows"
    return (D.data.values != "v4") & bad[D.vi.values.astype(int)]


def fit_fuser(tag, streams, C=1.0, names=None):
    """CA-FUSE on every SAFE-04 row (no purge: the final fuser sees no test person), as BIGB-01/build_v11.py fit()."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    D = safe04(tag)
    m = mav_flagged(D)
    if m.any():
        print(f"fit_fuser({tag}): {int(m.sum())} pairs with a Whisper-flagged MAV voice left out (LID_STRICT)", flush=True)
        D = D[~m].reset_index(drop=True)
    X = design(D[streams].values, D[CO].values)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), D.y.values)
    return dict(streams=names or streams, validation_streams=streams, conditions=CO, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), n_trials=int(len(D)), safe04_tag=tag, C=C)


def apply_fuser(model, Z, C):
    m = {k: np.array(model[k], np.float64) for k in ("mean", "scale", "coef")}
    return ((design(Z, C) - m["mean"]) / m["scale"]) @ m["coef"] + model["intercept"]


def check_v11_fuser():
    """The v11 fuser refitted from these SAFE-04 tables must SCORE like fusers/v11_bangla.json (wiring of the whole chain).
    Scores, not coefficients: the 55-input logistic fit is ill-conditioned and lbfgs at its default tolerance stops short of
    the optimum, so coefficients depend on the library build (Kaggle 11/10: max |coef diff| 0.063; locally, newton-cg vs lbfgs
    0.33 with Spearman 0.99996), while a wiring error (wrong rows / columns) breaks the ranking."""
    from stream_gate import S6
    if NOREL:                                   # NOREL-01: the stored v11 fuser holds rel / relg / t3_relg weights -> no reference to check against
        log(f"wiring (NOREL=1): fuser on {S6 + V11_T3} refitted, no comparison with v11_bangla.json (it uses rel / relg / t3_relg)")
        return fit_fuser("bigb01f", S6 + V11_T3)
    ref = json.loads((E2E / "fusers/v11_bangla.json").read_text())
    m = fit_fuser("bigb01f", S6 + V11_T3)
    D = safe04("bigb01f")
    Z, C = D[S6 + V11_T3].values, D[CO].values
    a, b = apply_fuser(m, Z, C), apply_fuser(ref, Z, C)
    if os.environ.get("CLEANCOMP") == "1":      # CLEANCOMP-01: tables relearned under LID_STRICT; the stored v11 fuser no longer applies (info only)
        rc = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
        log(f"wiring (CLEANCOMP=1): strict-table fuser vs the old v11_bangla.json on the strict tables: Spearman {rc:.4f}")
        assert rc >= 0.8, rc
        return m
    d = float(np.abs(np.array(m["coef"]) - np.array(ref["coef"])).max())
    rc = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    log(f"wiring: v11 fuser refitted on the SAFE-04 tables vs v11_bangla.json: Spearman of the scores {rc:.6f} "
        f"(max |score diff| {np.abs(a - b).max():.3f}, max |coef diff| {d:.2e})")
    assert rc >= 0.9995, (rc, d)
    return m


def bangla_scores(model, new, keep_t3=None):
    """{i: score} for the Bangla files: Z = [B3 (6) | v11 T3 columns keep_t3 | new streams], new = {i: (n, k)}, in the
    order of the fuser's stream list (S6 + [V11_T3[j] for j in keep_t3] + new names). keep_t3=(2,) = a new head REPLACES
    v11's anP512 deep / cca and keeps t3_relg. NOREL=1: b3 columns 0-3 and T3 columns (0, 1) only (rel / relg / t3_relg never enter)."""
    keep_t3 = ((0, 1) if NOREL else (0, 1, 2)) if keep_t3 is None else keep_t3
    assert not (NOREL and 2 in keep_t3), "NOREL=1: t3_relg (column 2) must not enter a fuser"
    out = {}
    for i, c in components().items():
        if c["is_en"]:
            continue
        Z = np.concatenate([c["b3"][:, :B3_KEEP], c["t3"][:, list(keep_t3)]]
                           + ([np.asarray(new[i], np.float64).reshape(len(c["pid"]), -1)] if new else []), 1)
        out[i] = apply_fuser(model, Z, c["co"])
        assert np.isfinite(out[i]).all(), c["key"]
    return out


def write_zip(path, scores):
    """scores = {i: higher-is-same}; files missing from `scores` keep the v11 score. Written as -score, 6 decimals."""
    comps = components()
    with zipfile.ZipFile(path, "w") as zf:
        for i, c in comps.items():
            s = np.asarray(scores.get(i, c["v11"]), np.float64)
            assert len(s) == len(c["pid"]) and np.isfinite(s).all(), c["key"]
            zf.writestr(c["zip_name"], "\n".join(f"{p} {v:.6f}" for p, v in zip(c["pid"], -s)) + "\n")
    return path


def zip_report(path, ref="v11"):
    """Per file: rows, identical to the reference zip, Spearman vs the reference."""
    comps = components(); out = {}
    refz = E2E / "ref" / f"submission_{ref}.zip"
    with zipfile.ZipFile(path) as a:
        rz = zipfile.ZipFile(refz) if refz.exists() else None
        for i, c in comps.items():
            x = pd.read_csv(a.open(c["zip_name"]), sep=" ", header=None, names=["pid", "s"], dtype={"pid": str})
            assert (x.pid.values == c["pid"]).all() and np.isfinite(x.s).all(), c["key"]
            r = dict(rows=len(x))
            if rz is not None:
                y = pd.read_csv(rz.open(c["zip_name"]), sep=" ", header=None, names=["pid", "s"], dtype={"pid": str})
                r.update(identical=a.read(c["zip_name"]) == rz.read(c["zip_name"]),
                         spearman=round(float(x.s.corr(y.s, method="spearman")), 4))
            out[c["key"]] = r
    return out


def level0_selftest(fn, n, seed=1):
    """fn(rows) -> scores of those trials only; the scores of a random half must equal the full-file scores."""
    full = fn(None)
    half = np.sort(np.random.RandomState(seed).choice(n, n // 2, replace=False))
    d = float(np.abs(fn(half) - full[half]).max())
    return full, d
