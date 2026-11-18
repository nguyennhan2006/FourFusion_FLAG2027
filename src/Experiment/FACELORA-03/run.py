r"""FACELORA-03 step 2 (registered 12/10 in README.md; amended 14/10 before any result): the current CC recipe (AUXP-01 V4 on D4,
aux VoxCeleb2 + VoxBlink: no frames for AVSpeech / LRS3) with the CLIP-L column of every face replaced by the FLAG_65 adapters.
  B0 [farl16 | clipL] (frozen, control)      A3 [farl16 | clipA3]      B3 [farl16 | clipB3]  <- DECISION
Faces carry every variant: [farl16 0:512 | clipL 512:1280 | clipA3 1280:2048 | clipB3 2048:2816]; each arm slices its 1280 columns,
so every arm sees the same rows and seeds. Auxiliary faces without an adapted embedding (frame missing) are left out in every arm.
    FOLDS=1,2 python run.py ; python run.py report
Env: FL3_DIRS (local: dirs holding face_clipA3_train.npy / face_clipB3_train.npy; Kaggle: under /kaggle/input), TR_SEEDS (3),
FL3_ARMS (subset), TR_MAX_STEPS (smoke tests only).
"""
import os, sys, re, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
_s = importlib.util.spec_from_file_location("auxp01run", EXP / "AUXP-01/run.py")
A1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(A1)
C2, C1, M = A1.C2, A1.C1, A1.M
M.TAG = C1.TAG = TAG = os.environ.get("FL3_TAG", "facelora03")         # 15/10: per-arm runs write facelora03_<arm>, merged by `merge`
V4CFG = dict(face_dropout=.55, voice_dropout=.30, aux_ids=64, ids_per_batch=64, epochs=80)      # AUXP-01 V4
COLS = {"farl16": np.arange(0, 512), "clipL": np.arange(512, 1280), "clipA3": np.arange(1280, 2048), "clipB3": np.arange(2048, 2816)}
ARMS_ = {"B0": "clipL", "A3": "clipA3", "B3": "clipB3"}
if os.environ.get("FL3_ARMS"):
    ARMS_ = {k: v for k, v in ARMS_.items() if k in os.environ["FL3_ARMS"].split(",")}
M.FACE = dict(M.FACE, **{a: np.r_[COLS["farl16"], COLS[c]] for a, c in ARMS_.items()})
M.PAIRS = dict(M.PAIRS, **{a: (a, "cat") for a in ARMS_})
M.N_SEEDS = int(os.environ.get("TR_SEEDS", "3"))
M.ARMS = {a: dict(aux=("vox", "vb"), pairs=[a], seeds=M.N_SEEDS, cfg=dict(M.t3r.SUB21A, **V4CFG)) for a in ARMS_}
M.DECISION = DECISION = "B3" if "B3" in ARMS_ else next(iter(ARMS_))
M.BASE_CFG = M.t3r.SUB21A
ADAPTED = ("clipA3", "clipB3")
LOAD = {c for c in ARMS_.values() if c in ADAPTED}                    # adapted arrays the selected arms need; the others are zeros
_D = {}
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
VOICE_ROWS = {}     # UTTSTAB-01: per row builder, (voice_meta.csv row, video) of every row it returned; read by sources_fl3


def sliced(src, fs, vs):
    """t3r.Source restricted to the columns of one arm (index arrays as well as slices)."""
    import copy, torch
    ix = lambda X, s: X[:, s] if isinstance(s, slice) else X[:, torch.as_tensor(s, device=X.device)]
    out = copy.copy(src); out.F, out.V = ix(src.F, fs), ix(src.V, vs)
    if getattr(src, "fn_bank", None) is not None:                     # FACENEG-01: the bank of negative faces, same face columns
        out.fn_bank = ix(src.fn_bank, fs)
    return out


M.sliced = sliced


def adir(arr):
    if arr not in _D:
        c = [Path(p) for p in os.environ.get("FL3_DIRS", "").split(os.pathsep) if p]
        if Path("/kaggle/input").exists():
            c += [f.parent for f in Path("/kaggle/input").rglob(f"face_{arr}_train.npy")]
        hit = [d for d in c if (d / f"face_{arr}_train.npy").exists()]
        assert hit, f"adapted arrays {arr} not found (FL3_DIRS / /kaggle/input): attach the FLAG_65 outputs"
        if os.environ.get("BN42") == "1":                             # CLEANTUNE-01 (31/10): only adapters of FLAG_109 (Bengali-clean)
            m = [d for d in hit if (d / "bn42_clean.json").exists()]
            if not m and os.environ.get("BN42_STANDIN") == "1":         # smoke only: wiring before the FLAG_109 arrays exist
                print(f"!! BN42 STAND-IN: unmarked {arr} arrays {hit[0]} (smoke only)", flush=True)
                m = hit[:1]
            assert len(m) == 1, f"BN42=1: need exactly one {arr} directory with bn42_clean.json (FLAG_109 output), found {len(m)}"
            if os.environ.get("LID_STRICT") == "1" and os.environ.get("BN42_STANDIN") != "1":   # LIDSTRICT-01: adapter of the bn + as rule
                import run as T                                         # T3ENC-01 (as in the row builders below; T is not module-level)
                mk = json.loads((m[0] / "bn42_clean.json").read_text())
                assert mk.get("lid_strict") == T.strict_counts(), f"LID_STRICT=1: {arr} adapter was not trained under this lid_strict.json ({mk})"
            hit = m
        _D[arr] = hit[0]
    return _D[arr]


def adapted(part):
    """[clipA3 | clipB3] of one part, per-encoder L2 (NaN rows stay NaN): v4 / ext (v1 then v2) / vox (FLAG_37) / vox2b / vb."""
    out = []
    n = None
    for arr in ADAPTED:
        if arr not in LOAD:
            out.append(None); continue
        d = adir(arr)
        if part == "v4":
            a = np.load(d / f"face_{arr}_train.npy")
        elif part == "ext":
            a = np.concatenate([np.load(d / f"ext_{s}_face_{arr}.npy") for s in ("v1_complete", "v2_complete")])
        else:
            a = np.load(d / f"{part}_face_{arr}.npy")
        out.append(l2n(a.astype(np.float32))); n = len(a)
    if n is None:                                                         # B0-only run: no adapted array is read
        return None
    return np.concatenate([x if x is not None else np.zeros((n, 768), np.float32) for x in out], 1)


def alignment(frozen, adapt, n=400, seed=0):
    ok = np.flatnonzero(np.isfinite(adapt).all(1) & np.isfinite(frozen).all(1))
    r = np.random.RandomState(seed).choice(ok, min(n, len(ok)), replace=False)
    return float(((adapt[r] @ frozen[r].T).argmax(1) == np.arange(len(r))).mean())


def with_adapted(Fz, part):
    """[frozen farl16 | clipL] (1280) -> + [clipA3 | clipB3] (2816), with a row-alignment check against the frozen CLIP-L."""
    Fa = adapted(part)
    if Fa is None:
        print(f"FACELORA-03 {part}: {len(Fz)} rows, B0-only run (adapted columns = 0)", flush=True)
        return np.concatenate([Fz, np.zeros((len(Fz), 1536), np.float32)], 1)
    assert len(Fz) == len(Fa), (part, len(Fz), len(Fa))
    al = {a: alignment(Fz[:, COLS["clipL"]], Fa[:, k * 768:(k + 1) * 768]) for k, a in enumerate(ADAPTED) if a in LOAD}
    print(f"FACELORA-03 {part}: {len(Fa)} rows, finite adapted {np.isfinite(Fa).all(1).mean():.4f}, alignment {al}", flush=True)
    assert min(al.values()) >= 0.2, f"adapted rows of {part} do not line up with the frozen rows: {al}"
    return np.concatenate([Fz, Fa], 1)


def patch():
    """T3ENC-01 loaders: t3('face', v4 / ext / vox) -> 2816 columns; vox_rows skips faces without every adapted embedding."""
    import run as T
    assert Path(T.__file__).parent.name == "T3ENC-01", T.__file__
    if getattr(T, "_facelora03", False):
        return T
    t3_orig = T.t3

    def t3(kind, part):
        if kind != "face":
            return t3_orig(kind, part)
        key = ("face", part)
        if key not in _D:
            F = with_adapted(t3_orig("face", part), part)
            if part != "vox":
                assert np.isfinite(F).all(), f"non-finite adapted rows in {part}"
            _D[key] = F
        return _D[key]

    def vox_rows(F, V):
        fm = pd.read_csv(T.VK / "face_meta.csv", dtype={"video": str, "utt": str})
        vm = pd.read_csv(T.VK / "voice_meta.csv", dtype={"video": str, "utt": str})
        assert len(fm) == len(F) and len(vm) == len(V)
        ok = np.isfinite(F).all(1)
        keep = (set(fm.spk[ok]) & set(vm.spk)) - T.dropped_speakers()
        face_of = {s: g.index.values for s, g in fm[ok].groupby("spk")}
        fvid, vvid = fm.video.values, vm.video.values
        Fi, Vi, S = [], [], []
        for s, idx in sorted(vm.groupby("spk").indices.items()):
            if s not in keep:
                continue
            for j, v in enumerate(sorted(idx)):
                cand = [f for f in face_of[s] if fvid[f] != vvid[v]] or list(face_of[s])
                Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vox2:" + s)
        print(f"FACELORA-03 FLAG_37: {int((~ok).sum())} faces without every embedding left out; {len(S)} rows", flush=True)
        VOICE_ROWS["vox"] = (np.asarray(Vi, np.int64), vvid[np.asarray(Vi, np.int64)])
        return F[Fi], V[Vi], np.array(S)

    T.t3, T.vox_rows, T._facelora03 = t3, vox_rows, True
    return T


def new_rows4(kinds, log):
    """= SCALE-CC-01 new_rows (FLAG_53) with 2816-column faces; faces without every adapted embedding left out."""
    VB2 = C1.vox2b_dir()
    fm = pd.read_csv(VB2 / "face_meta.csv", dtype=str)
    vm = pd.read_csv(VB2 / "voice_meta.csv", dtype={"spk": str, "video": str, "utt": str, "kind": str})
    Fz = np.concatenate([l2n(np.load(VB2 / f"face_{n}.npy").astype(np.float32)) for n in ("farl16", "clipL")], 1)
    F = with_adapted(Fz, "vox2b")
    V = np.concatenate([l2n(np.load(VB2 / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
    assert len(fm) == len(F) and len(vm) == len(V)
    fk = fm.kind.isin(kinds).values & np.isfinite(F).all(1)
    vk = vm.kind.isin(kinds).values
    fm, F, vm, V = fm[fk].reset_index(drop=True), F[fk], vm[vk].reset_index(drop=True), V[vk]
    keep = (set(fm.spk) & set(vm.spk)) - C1.drop53()
    face_of, fvid, vvid = fm.groupby("spk").indices, fm.video.values, vm.video.values
    Fi, Vi, S = [], [], []
    for s, idx in sorted(vm.groupby("spk").indices.items()):
        if s not in keep:
            continue
        for j, v in enumerate(sorted(idx)):
            cand = [f for f in face_of[s] if fvid[f] != vvid[v]] or list(face_of[s])
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vox2:" + s)
    log(f"FACELORA-03 FLAG_53 rows {kinds}: {len(S)} rows, {len(set(S))} speakers")
    VOICE_ROWS["vox2b"] = (np.flatnonzero(vk)[np.asarray(Vi, np.int64)], vvid[np.asarray(Vi, np.int64)])  # rows of the unfiltered csv
    return F[Fi], V[Vi], np.array(S)


def vb_rows4(log):
    """= SCALE-VB vb_rows (Bengali LID speaker filter, label-noise filter, cross-video faces) with 2816-column faces; faces
    without every adapted embedding left out."""
    import run as T
    VBM = M.vb_module()
    VB = VBM.VB
    fm = pd.read_csv(VB / "face_meta.csv", dtype={"video": str, "utt": str}); vm = pd.read_csv(VB / "voice_meta.csv", dtype={"video": str, "utt": str})
    Fz = np.concatenate([l2n(np.load(VB / f"face_{n}.npy").astype(np.float32)) for n in ("farl16", "clipL")], 1)
    F = with_adapted(Fz, "vb")
    V = np.concatenate([l2n(np.load(VB / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
    drop = set(json.loads((VB / "lid_summary.json").read_text())["speakers_dropped_ids"]) | T.strict_drop("vb")
    xm = pd.read_csv(EXP / "VOX2-01/feats_vox2/voice_meta.csv", dtype={"video": str, "utt": str})
    XV = l2n(np.load(EXP / "T3ENC-01/feats_vox2t3/voice_sam100f.npy").astype(np.float32))
    keepx = ~xm.spk.isin(T.dropped_speakers()).values
    thr = float(np.nanquantile(VBM.loo_cos(XV[keepx], xm.spk.values[keepx]), 0.01))
    c = VBM.loo_cos(l2n(np.load(VB / "voice_sam100f.npy").astype(np.float32)), vm.spk.values)
    lid_ok = ~vm.spk.isin(drop).values
    ok = lid_ok & ~(lid_ok & (c < thr))
    fok = np.isfinite(F).all(1)
    keep_spk = set(vm.spk[ok]) & set(fm.spk[~fm.spk.isin(drop) & fok])
    face_of = {s: g.index.values for s, g in fm[fok].groupby("spk")}
    fvid, vvid = fm.video.values, vm.video.values
    Fi, Vi, S = [], [], []
    vok = vm[ok]
    for s, idx in sorted(vok.groupby("spk").indices.items()):
        if s not in keep_spk:
            continue
        for j, v in enumerate(sorted(vok.index.values[idx])):
            cand = [f for f in face_of[s] if fvid[f] != vvid[v]] or list(face_of[s])
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vb:" + s)
    log(f"FACELORA-03 VoxBlink rows: {len(S)} rows, {len(set(S))} speakers ({int((~fok).sum())} faces without every embedding)")
    VOICE_ROWS["vb"] = (np.asarray(Vi, np.int64), vvid[np.asarray(Vi, np.int64)])
    return F[Fi], V[Vi], np.array(S)


def _voice_rows(parts, keep):
    """UTTSTAB-01: {source, row, video} of a Source's rows (input order) built from the builders' outputs [(builder, mask)] then
    `keep`; None when a builder has not recorded its rows (another vox_rows than this module's)."""
    if not all(b in VOICE_ROWS for b, _ in parts):
        return None
    cols = {"source": [], "row": [], "video": []}
    for b, m in parts:
        r, v = VOICE_ROWS[b]
        cols["source"].append(np.full(len(r[m]), b)); cols["row"].append(r[m]); cols["video"].append(v[m])
    return {k: np.concatenate(x)[keep] for k, x in cols.items()}


def sources_fl3(R, T, main_rows, aux, device):
    """D4 (main v4 + MAV + VoxCeleb2 SA; aux the other VoxCeleb2 speakers + VoxBlink), 2816-column faces."""
    t3r = M.t3r
    TF, TV, TS = main_rows
    width = 1280 + sum(len(COLS[a]) for a in ADAPTED)                 # 2816 here; ENC-SCALE-02 re-points ADAPTED / COLS to DINOv2
    assert TF.shape[1] == width and np.isfinite(TF).all(), "main rows must carry every adapted column"
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    ok = ~np.isin(S2, ["vox2:" + x for x in C1.drop53()])
    F3, V3, S3 = new_rows4(("new", "more"), R.log)
    FX, VX, SX = np.concatenate([F2[ok], F3]), np.concatenate([V2[ok], V3]), np.concatenate([S2[ok], S3]).astype(str)
    grp = {k: v[0] for k, v in json.loads((EXP / "SCALE-CC-01/vox2b_groups.json").read_text()).items()}
    grp.update({k: v[0] for k, v in json.loads(C1.D3.D2.GROUPS.read_text()).items()})
    sa = np.array([grp.get(s.split(":", 1)[1], "OTH") == "SA" for s in SX])
    gm = dict(R.GMAP)
    gm.update({"vox2:" + s: g for s, g in json.loads((EXP / "SCALE-CC-01/vox2_gender.json").read_text()).items()})
    MS = np.concatenate([np.asarray(TS).astype(str), SX[sa]])
    missing = sorted({s for s in MS if s not in gm})
    assert not missing, f"main identities without gender: {missing[:5]}"
    main = t3r.Source(np.concatenate([TF, FX[sa]]), np.concatenate([TV, VX[sa]]), MS, device, male=gm)
    out = []
    if "vox" in aux:
        out.append(("vox", t3r.Source(FX[~sa], VX[~sa], SX[~sa], device)))
        out[-1][1].voice_rows = _voice_rows([("vox", ok), ("vox2b", slice(None))], ~sa)
    if "vb" in aux:
        out.append(("vb", t3r.Source(*vb_rows4(R.log), device)))
        out[-1][1].voice_rows = _voice_rows([("vb", slice(None))], slice(None))
    R.log(f"FACELORA-03 D4: main {len(main.ids)} ids / {main.n_rows} rows | aux " + ", ".join(f"{k} {len(a.ids)} ids / {a.n_rows} rows" for k, a in out))
    return main, out


M.sources = sources_fl3


def paired_g(D, S2, persons, a_cols, b_cols):
    from lab_fuse import table
    D = D.copy(); D["a_"], D["b_"] = np.nan, np.nan
    for s in sorted(D.fold.unique()):
        te = D[D.fold == s]; pool = S2.purge(D[D.fold != s], persons[int(s)])
        dz = S2.Design(pool, te, list(dict.fromkeys(a_cols + b_cols)))
        D.loc[te.index, "a_"], D.loc[te.index, "b_"] = dz.score(a_cols, 1.0), dz.score(b_cols, 1.0)
    T_ = table(D, {"a": lambda g: g["a_"].values, "b": lambda g: g["b_"].values})
    T_ = T_[T_.cell.str.endswith(" g")]
    return (T_["b"] - T_["a"]).groupby(T_.fold).mean()


def report():
    """Rule 1 (standalone, paired, g cells en_ext / urdu / v4): B3 - B0; rule 2: default gate of CA-FUSE(v11 + trB3) vs v11;
    rule 3 (amended 14/10): gate passes and CA-FUSE(v11 + trB3) - CA-FUSE(v11 + trB0) >= +0.3 at >= 4/5 folds."""
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP, table
    base = S6 + E.V11_T3
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    v = E.gate(TAG, {a: base + [f"tr{a}"] for a in ARMS_}, DECISION, standalone=[f"tr{a}" for a in ARMS_], out=out, ci_arms=[DECISION])
    D = E.safe04(TAG)
    S2, persons = E.stack02(), E.fold_persons()
    T_ = table(D, {a: (lambda g, a=a: g[f"tr{a}"].values) for a in ARMS_})
    G = T_[T_.cell.isin(["en_ext g", "urdu g", "v4 g"])].groupby("fold")[list(ARMS_)].mean()
    txt = ["\n== RULE 1: standalone, mean of en_ext g + urdu g + v4 g per fold (lower = better)\n" + G.round(2).to_string()]
    d1 = G["B3"] - G["B0"]
    r1 = "YES" if (d1.mean() <= -0.5 and (d1 < 0).sum() >= 4) else ("HURTS" if d1.mean() >= 0.3 else "NO DIFFERENCE")
    txt.append(f"RULE 1 (B3 - B0 standalone): {d1.mean():+.2f} ({int((d1 < 0).sum())}/5 better) -> {r1}"
               + (f" | A3 - B0 {(G['A3'] - G['B0']).mean():+.2f}, B3 - A3 {(G['B3'] - G['A3']).mean():+.2f}" if "A3" in G else ""))
    per = paired_g(D, S2, persons, base + ["trB3"], base + ["trB0"])
    rep = bool(v["B3"]["passes"] and per.mean() >= 0.3 and (per > 0).sum() >= 4)
    txt.append(f"RULE 2 (gate B3 vs v11): {'PASSES' if v['B3']['passes'] else 'fails'}")
    txt.append(f"RULE 3 (CA-FUSE B3 - B0, g cells): " + ", ".join(f"fold {s} {g:+.2f}" for s, g in per.items())
               + f" | mean {per.mean():+.2f}, {int((per > 0).sum())}/5 better -> {'YES: final head B3 + zip + one CB run' if rep else 'no'}")
    # R1b (registered 15/10 before any result, docs/REOPEN_2026-10-15.md): ENS-CC-01 reopened, diversity from the face ENCODER:
    # both CC streams (frozen-CLIP-L head trB0 + adapted-CLIP-L head trB3) in one fuser vs the best single one
    if "trB0" in D and "trB3" in D:
        e0 = paired_g(D, S2, persons, base + ["trB0", "trB3"], base + ["trB0"])
        e3 = paired_g(D, S2, persons, base + ["trB0", "trB3"], base + ["trB3"])
        ens = bool(e0.mean() >= 0.3 and (e0 > 0).sum() >= 4 and e3.mean() >= 0.3 and (e3 > 0).sum() >= 4)
        txt.append(f"R1b (B0+B3 in one fuser): vs B0 {e0.mean():+.2f} ({int((e0 > 0).sum())}/5), vs B3 {e3.mean():+.2f} ({int((e3 > 0).sum())}/5)"
                   + f" -> {'ENSEMBLE HELPS (candidate: both streams)' if ens else 'no ensemble gain'}")
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    (HERE / "report.txt").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def merge():
    """facelora03_<arm>_fold<s>.parquet (per-arm runs, same trial tables) -> facelora03_fold<s>.parquet with every tr<arm> column."""
    from lab_fuse import SP
    for s in range(1, 6):
        parts = sorted((SP / "out/safe04").glob(f"facelora03_*_fold{s}.parquet"))
        assert parts, f"no per-arm table for fold {s}"
        base = None
        for f in parts:
            d = pd.read_parquet(f)
            if base is None:
                base = d
            else:
                assert len(d) == len(base) and (d.y.values == base.y.values).all() and (d.vi.values == base.vi.values).all(), f
                for c in [c for c in d.columns if c.startswith("tr") and c not in base]:
                    base[c] = d[c].values
        base.to_parquet(SP / "out/safe04" / f"facelora03_fold{s}.parquet")
        print(f"fold {s}: {[f.name for f in parts]} -> {[c for c in base.columns if c.startswith('tr')]}")


# ============================================================================ final head + dev zip (README, amended 15/10 before any step-2 result)
def test_alignment(fz, a, n=400, seed=0):
    """Row order of an adapted test array against the frozen CLIP-L of the same file (test files repeat faces):
    retrieval = the nearest frozen row of a sampled adapted row has the SAME frozen face; dup_same = trials with the same frozen
    face have the same adapted face (cos >= .999)."""
    r = np.random.RandomState(seed).choice(len(a), min(n, len(a)), replace=False)
    j = (a[r] @ fz.T).argmax(1)
    ret = float(((fz[j] * fz[r]).sum(1) >= .9999).mean())
    _, inv = np.unique(np.round(fz, 4), axis=0, return_inverse=True)
    first, same = {}, []
    for i, g in enumerate(inv.ravel()):
        if g in first:
            same.append(float(a[i] @ a[first[g]]) >= .999)
        else:
            first[g] = i
    return dict(retrieval=round(ret, 3), dup_pairs=len(same), dup_same=round(float(np.mean(same)), 4) if same else 1.0)


def patch_test():
    """e2e_lib.t3_split (test files: [farl16 | clipL], 1280) -> + [clipA3 | clipB3] from the FLAG_65 arrays of the test lists
    (face_clip<X>_<gender|no_gender>_<English|Bangla>.npy, one row per trial in list order); zeros for an arm not loaded."""
    import e2e_lib as E
    if getattr(E, "_facelora03", False):
        return
    split0 = E.t3_split

    def t3_split(c):
        F, V, m = split0(c)
        name = c["key"].replace("/", "_")
        fz = F[:, COLS["clipL"]]
        parts = []
        for arr in ADAPTED:
            if arr not in LOAD:
                parts.append(np.zeros((len(F), 768), np.float32)); continue
            a = l2n(np.load(adir(arr) / f"face_{arr}_{name}.npy").astype(np.float32))
            assert a.shape == (len(F), 768) and np.isfinite(a).all(), (c["key"], arr, a.shape, len(F))
            al = test_alignment(fz, a)
            print(f"FACELORA-03 test {c['key']}: {arr} {len(a)} rows, {al}", flush=True)
            assert al["retrieval"] >= 0.2 and al["dup_same"] >= 0.99, (c["key"], arr, al)
            parts.append(a)
        return np.concatenate([F] + parts, 1), V, m

    E.t3_split, E._facelora03 = t3_split, True


def fit_en():
    """EN-CC-01's English fuser CA_EN_CC with tr<DECISION> in place of trV4: EN-T3-01's English table (v4 + en_ext rows, finite
    AudioCLIP) + tr<DECISION> of the merged tables (attached as EN-CC-01 run.py does), logistic fit as EN-T3-01/build_v12.py."""
    import e2e_lib as E
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from rel_sweep5 import SP
    s_ = importlib.util.spec_from_file_location("ent301", EXP / "EN-T3-01/run.py")
    EN = importlib.util.module_from_spec(s_); s_.loader.exec_module(EN)
    D = EN.table_en()
    col = f"tr{DECISION}"
    parts = []
    for s in range(1, 6):
        a = pd.read_parquet(SP / f"out/safe04/streams_fold{s}.parquet", columns=["y", "vi", "fi"])
        b = pd.read_parquet(SP / f"out/safe04/{TAG}_fold{s}.parquet", columns=["data", "y", "vi", "fi", col])
        assert len(a) == len(b) and (a.y.values == b.y.values).all() and (a.vi.values == b.vi.values).all(), s
        parts.append(b.assign(fold=s))
    key = ["fold", "data", "y", "vi", "fi"]
    B = pd.concat(parts, ignore_index=True).drop_duplicates(key)
    n = len(D)
    D = D.merge(B, on=key, how="left", validate="many_to_one")
    assert len(D) == n and np.isfinite(D[col]).all(), f"{col} attach"
    st = EN.CAEN + [col]
    X = E.design(D[st].values, D[EN.CO].values)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), D.y.values)
    print(f"English fuser CA_EN_CC[{col}]: {X.shape[1]} inputs on {len(D)} trials {D.cell.value_counts().to_dict()}", flush=True)
    return dict(streams=EN.CAEN + [col], validation_streams=st, conditions=EN.CO, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), n_trials=int(len(D)))


def cmp_zip(a, b):
    """Per-file correlation of two submission zips (same trial order)."""
    import zipfile
    if not Path(b).exists():
        return None
    out = {}
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        for n in za.namelist():
            x = pd.read_csv(za.open(n), sep=" ", header=None, dtype={0: str})
            y = pd.read_csv(zb.open(n), sep=" ", header=None, dtype={0: str})
            assert (x[0].values == y[0].values).all(), n
            out[n] = round(float(np.corrcoef(x[1], y[1])[0, 1]), 4)
    return out


def zipit():
    """Dev zip of the decision arm: BANGLA = AUXP-01's fuser (S6 + v11 T3 + trV4) with tr<DECISION> in place of trV4, ENGLISH =
    EN-CC-01's CA_EN_CC with tr<DECISION> in place of trV4 (fit_en); the arm's test streams come from `final`."""
    import e2e_lib as E
    from stream_gate import S6
    od = M.out_dir()
    E.check_v11_fuser()
    col = f"tr{DECISION}"
    Z = np.load(od / f"{TAG}_final_{E.SPLIT}.npz")
    comps = E.components()
    new = {i: np.asarray(Z[f"{col}_{i}"], np.float64) for i in comps}
    assert all(len(new[i]) == len(c["pid"]) and np.isfinite(new[i]).all() for i, c in comps.items())
    bn = E.fit_fuser(TAG, S6 + E.V11_T3 + [col])
    en = fit_en()
    scores = E.bangla_scores(bn, {i: v for i, v in new.items() if not comps[i]["is_en"]})
    for i, c in comps.items():
        if c["is_en"]:
            scores[i] = E.apply_fuser(en, np.concatenate([c["en7"], new[i][:, None]], 1), c["enco"])
            assert np.isfinite(scores[i]).all(), c["key"]
    zp = od / f"submission_{TAG}_{E.SPLIT}.zip"
    E.write_zip(zp, scores)
    rep = E.zip_report(zp, "v11")
    ref = cmp_zip(zp, EXP / "EN-CC-01/submission_en_cc01_dev.zip") if E.SPLIT == "dev" else None
    for name, mdl in (("bangla", bn), ("english", en)):
        (od / f"{TAG}_fuser_{name}.json").write_text(json.dumps(mdl, indent=1))
    (od / f"{TAG}_zip_checks.json").write_text(json.dumps(dict(arm=DECISION, files=rep, corr_vs_en_cc01=ref), indent=1, default=str))
    print(f"-> {zp.name}: {rep}\ncorrelation per file with EN-CC-01 (CB 20.01): {ref}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "merge":
        merge()
    elif cmd == "report":
        report()
    elif cmd == "final":
        patch(); patch_test()
        print(f"FACELORA-03 final head: {DECISION}, {M.N_SEEDS} seeds, aux {M.ARMS[DECISION]['aux']}", flush=True)
        M.final()
    elif cmd == "zip":
        zipit()
    else:
        patch()
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            M.run(s)
