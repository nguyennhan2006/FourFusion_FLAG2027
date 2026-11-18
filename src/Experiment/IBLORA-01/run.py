"""IBLORA-01 bac 2 (README "Cong", registered 2/11; implementation note 12/11 before any number), SAFE-04, Kaggle CPU (FLAG_182):
    python lab_entry.py IBLORA-01 report
English fuser of F1s' (= F0s: CLEAN-EN-01 form, logistic C = 1 on [trB0c | ImageBind | age] x conditions, CLEANTUNE-01
fit_en_clean on the strict ct01g1 tables, Whisper-flagged MAV voices left out) with CTRL_ib replaced by ibLoRA (the CTRL_ib recipe
on the IBLORA-01s arrays: cos after removing the fold's v4-train means, z by N_REF random v4-train pairs of the fold).
OOF over the 5 folds with purge, cells en_ext g / ng and v4 g / ng. Rule: EN_L - EN_0 worse by >= 1.0 on en_ext g or v4 g -> broken.
Nothing reads dev / CB / Evaluation. Writes iblora2_report.txt / iblora2_summary.json to <SP>/out/safe04."""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent
sys.path[:0] = [str(EXPD / "E2E"), str(EXPD / "SAFE-04"), str(EXPD / "STACK-02")]
TAG, HEAD = "iblora2", "trB0c"
TXT = []


def say(s):
    print(s, flush=True); TXT.append(str(s))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def find(name):
    roots = [Path(os.environ.get("IBL_DIR", "/kaggle/input"))]
    for r in roots:
        hits = sorted(r.rglob(name)) if r.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name}: attach phantrnanna/flag2027-feats-iblora1s-p (IBLORA-01s arrays)")


def iblora_arrays(FMETA, VMETA):
    """v4-train rows + the MAV v1 / v2 ext rows (v1 then v2, FLAG_12 order = LOAD's FA / VA)."""
    F4, A4 = np.load(find("face_ibLv_train.npy")).astype(np.float32), np.load(find("voice_ibLa_train.npy")).astype(np.float32)
    ext = {}
    for kind, meta, nm in (("face", FMETA, "ibLv"), ("voice", VMETA, "ibLa")):
        parts = []
        for src in ("v1_complete", "v2_complete"):
            a = np.load(find(f"ext_{src}_{kind}_{nm}.npy")).astype(np.float32)
            assert len(a) == int((meta.src == src).sum()), f"{kind} {src}: {len(a)} rows != meta {(meta.src == src).sum()}"
            parts.append(a)
        ext[kind] = np.concatenate(parts)
    m = json.loads(find("bn42_clean.json").read_text())
    m = {k: (v if isinstance(v, (str, int, float, bool)) else f"<{type(v).__name__} {len(v)}>") for k, v in m.items()} if isinstance(m, dict) else str(type(m))
    return F4, A4, ext, m


def table():
    """EN-T3-01's English table + trB0c of the strict ct01g1 tables (as CLEANTUNE-01 fit_en_clean) + ib0 / ibLoRA per fold."""
    import e2e_lib as E
    from lab_fuse import SP
    from rel_sweep5 import V4, FA, VA, make_fold, N_REF, Std, l2n_, FMETA, VMETA
    EN = _load("ent301", EXPD / "EN-T3-01/run.py")
    D = EN.table_en()
    key = ["fold", "data", "y", "vi", "fi"]
    B = pd.concat([pd.read_parquet(SP / f"out/safe04/ct01g1_fold{s}.parquet", columns=["data", "y", "vi", "fi", HEAD]).assign(fold=s)
                   for s in range(1, 6)], ignore_index=True).drop_duplicates(key)
    n = len(D)
    D = D.merge(B, on=key, how="left", validate="many_to_one")
    assert len(D) == n and np.isfinite(D[HEAD]).all(), f"{HEAD} attach"
    m = E.mav_flagged(D)
    say(f"English table: {len(D)} trials; {int(m.sum())} with a Whisper-flagged MAV voice left out (LID_STRICT, as fit_en_clean)")
    D = D[~m].reset_index(drop=True)
    F4, A4, ext, man = iblora_arrays(FMETA, VMETA)
    say(f"IBLORA-01s arrays: v4 face {F4.shape} voice {A4.shape}, ext face {ext['face'].shape} voice {ext['voice'].shape}; bn42_clean {man}")
    cs = lambda F_, A_, mf, mv: np.sum(l2n_(F_ - mf) * l2n_(A_ - mv), 1)
    v4m = D.data.values == "v4"
    pick = lambda X4, XE, idx: np.where(v4m[:, None], X4[np.clip(idx, 0, len(X4) - 1)], XE[np.clip(idx, 0, len(XE) - 1)])
    srcs = {"ib0": (V4["ibf"], V4["iba"], FA["ibv"], VA["iba"]), "ibLoRA": (F4, A4, ext["face"], ext["voice"])}
    for col, (f4, a4, fe, ae) in srcs.items():
        Fx, Ax = pick(f4, fe, D.fi.values), pick(a4, ae, D.vi.values)
        D[col] = np.nan
        for s in sorted(D.fold.unique()):
            tr = make_fold(int(s))["v4tr"]; rr = np.random.RandomState(0)
            rf, rv = tr[rr.randint(len(tr), size=N_REF)], tr[rr.randint(len(tr), size=N_REF)]
            mf, mv = f4[tr].mean(0), a4[tr].mean(0)
            z = Std(cs(f4[rf], a4[rv], mf, mv))
            te = (D.fold == s).values
            D.loc[te, col] = z(cs(Fx[te], Ax[te], mf, mv))
        assert np.isfinite(D[col]).all(), col
    r = {c: float(np.corrcoef(D[c], D.CTRL_ib)[0, 1]) for c in ("ib0", "ibLoRA")}
    say(f"wiring: corr(ib0, CTRL_ib) = {r['ib0']:.4f} (must be >= 0.99: same recipe on the same arrays) | corr(ibLoRA, CTRL_ib) = {r['ibLoRA']:.3f}")
    assert r["ib0"] >= 0.99, "ib0 does not reproduce CTRL_ib: row mapping broken"
    return D, EN.CO, r


def report():
    t0 = time.time()
    import e2e_lib as E
    from lab_fuse import table as ltable, SP
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    D, CO, r = table()
    say(f"cells {D.cell.value_counts().to_dict()}")
    S2, persons = E.stack02(), E.fold_persons()
    ARMS = {"EN_0": [HEAD, "CTRL_ib", "CTRL_age"], "EN_L": [HEAD, "ibLoRA", "CTRL_age"], "EN_0L": [HEAD, "CTRL_ib", "ibLoRA", "CTRL_age"]}

    def fit(Dt, st):
        X = E.design(Dt[st].values, Dt[CO].values); sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), Dt.y.values)
        return dict(streams=st, conditions=CO, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                    intercept=float(lr.intercept_[0]), n_trials=int(len(Dt)), C=1.0)

    cols = {}
    for name, st in ARMS.items():
        sc = pd.Series(np.nan, index=D.index)
        for s in sorted(D.fold.unique()):
            te = D[D.fold == s]
            m = fit(S2.purge(D[D.fold != s], persons[int(s)]), st)
            sc[te.index] = E.apply_fuser(m, te[st].values, te[CO].values)
        cols[name] = sc.values
    for c in ("CTRL_ib", "ibLoRA", HEAD):
        cols[f"{c} alone"] = D[c].values
    Dx = D.assign(**{f"v{i}": v for i, v in enumerate(cols.values())})
    T = ltable(Dx, {n: (lambda g, i=i: g[f"v{i}"].values) for i, n in enumerate(cols)})
    names = list(cols)
    M = T.groupby("cell")[names].mean()
    say("\n== OOF EER per cell (mean of 5 folds, purge)\n" + M.round(2).to_string())
    summ = dict(corr=r, eer=M.round(3).to_dict(), cells={})
    for arm in ("EN_L", "EN_0L"):
        d = (M["EN_0"] - M[arm]).round(2)                              # > 0 = arm better
        wins = T.assign(w=T[arm] < T["EN_0"]).groupby("cell").w.sum()
        say(f"\n{arm} vs EN_0 per cell (+ = {arm} better): {d.to_dict()} | folds better: {wins.to_dict()}")
        summ["cells"][arm] = dict(delta=d.to_dict(), wins={k: int(v) for k, v in wins.to_dict().items()})
    dl = summ["cells"]["EN_L"]["delta"]
    broken = any(dl.get(c, 0.0) <= -1.0 for c in ("en_ext g", "v4 g"))
    summ["broken"] = bool(broken)
    say(f"\nbac 2 rule (EN_L worse by >= 1.0 on en_ext g or v4 g -> broken): {'BROKEN -> close' if broken else 'not broken -> bac 3 (frozen Evaluation zip, README)'}")
    (SP / "out/safe04" / f"{TAG}_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (SP / "out/safe04" / f"{TAG}_report.txt").write_text("\n".join(TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
