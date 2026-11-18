r"""BIGB-01 -> v11 dev zip (registered in README.md: "if P512 passes: v10 with the T3 head trained by the P512 recipe -> v11,
one CB run"). Identical to T3ENC-01/build_v10.py except the T3 head:
  head    InfoNCE, PK 512 x 2 (ALLNEG-01 train_pk + make_loss(0), = BIGB-01 arm P512), epochs so that ~1200 steps, 3 nets,
          trained on all 70 v4 + 151 MAV-Celeb v1/v2 persons + the Bengali-filtered VoxCeleb2 identities
  fuser   CA-FUSE on S6 + anP512_deep, anP512_cca, t3_relg (+ the 7 conditions and products), C = 1, fit on the SAFE-04 rows
          of bigb01f (t3_relg from t3enc01f, unchanged)
English files byte for byte from v9 (as v10). Needs VOX_LID (the registered Bengali filter).
    VOX_LID=<lid_summary.json> python build_v11.py fit | head | zip
"""
import json, os, sys, time, zipfile, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
R0 = HERE.parents[1]
T3D = R0 / "Experiment/T3ENC-01"
assert os.environ.get("VOX_LID"), "v11 is registered on the Bengali-filtered VoxCeleb2 rows: set VOX_LID"
FUSER, DEVS, ZIP = HERE / "cafuse_t3_fuser_v11.json", HERE / "dev_t3_streams_v11.npz", HERE / "submission_v11.zip"
V9 = R0 / "Experiment/CAFUSE-EN/submission_CA_EN_BN.zip"
V10 = T3D / "submission_v10f.zip"
SCR = Path(r"C:/Users/ASUS/AppData/Local/Temp/claude/d--Sinh-vi-n-CNhan-FLAG-2027-FourFusion/969f2c71-effc-43ec-a610-c253be9d1ea8/scratchpad")
FEATS = [R0 / "kaggle/output/feats_eval", R0 / "kaggle/output/feats_av", R0 / "kaggle/output/feats_models"]
EXTD = [SCR / "in/eda_bundle", R0 / "kaggle/output/feats_models"]
T3S = ["anP512_deep", "anP512_cca", "t3_relg"]
P, K = 512, 2
CELLS_BN = ["gender/Bangla", "no_gender/Bangla"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def allneg():
    spec = importlib.util.spec_from_file_location("allneg01run", R0 / "Experiment/ALLNEG-01/run.py")
    A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)          # puts SAFE-04, VOX2-01, T3ENC-01 on sys.path
    return A


def fit():
    allneg()
    from stream_gate import load, S6, CO, _X
    from lab_fuse import SP
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    D = load("bigb01f")
    T3 = pd.concat([pd.read_parquet(f) for f in sorted((SP / "out/safe04").glob("t3enc01f_fold*.parquet"))], ignore_index=True)
    assert (T3.y.values == D.y.values).all() and (T3.vi.values == D.vi.values).all() and (T3.fi.values == D.fi.values).all()
    for c in ("t3B_deep", "t3B_cca", "t3_relg"):
        D[c] = T3[c].values
    out = {}
    for name, st in (("check_v10", S6 + ["t3B_deep", "t3B_cca", "t3_relg"]), ("v11", S6 + T3S)):
        X = _X(D, st)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), D.y.values)
        out[name] = dict(streams=["deep", "cca", "ib", "age", "rel", "relg", "t3_deep", "t3_cca", "t3_relg"], validation_streams=st,
                         conditions=CO, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                         intercept=float(lr.intercept_[0]), n_trials=int(len(D)), safe04_tag="bigb01f (+ t3_relg of t3enc01f)")
    v10 = json.loads((T3D / "cafuse_t3_fuserf.json").read_text())
    d = float(np.abs(np.array(out["check_v10"]["coef"]) - np.array(v10["coef"])).max())
    log(f"wiring: the v10 fuser refitted on these rows vs T3ENC-01/cafuse_t3_fuserf.json: max |coef diff| {d:.2e} "
        f"({len(D)} vs {v10['n_trials']} trials)")
    FUSER.write_text(json.dumps(out["v11"], indent=1))
    log(f"v11 fuser: {len(out['v11']['coef'])} inputs on {len(D)} SAFE-04 trials -> {FUSER.name}")


def head():
    import torch
    A = allneg()
    import rel_sweep5 as R
    import run as T                                       # T3ENC-01/run.py: t3(), vox_rows(), dropped_speakers()
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))
    t0 = time.time()
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    allp = dict(s=0, v4tr=np.arange(len(R.V4S)), ext_pool=list(R.PERSONS), v4_train=sorted(set(R.V4S)))
    TF, TV, TS, info = R.train_rows(allp, ext=True, arrays=(F4, V4v, FE, VE))
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    TF, TV, TS = np.concatenate([TF, F2]), np.concatenate([TV, V2]), np.concatenate([TS, S2])
    bpe = max(1, len(TS) // (P * K)); ep = max(1, int(round(R.STEPS / bpe)))          # = ALLNEG-01 run(), arm P512
    log(f"final T3 head (P512): {len(set(TS))} identities ({len(set(S2))} VoxCeleb2, {len(T.dropped_speakers())} dropped by the "
        f"LID filter), {len(TS)} rows, {ep * bpe} steps (P {P} K {K}), {R.N_NETS} nets")
    units = [R.train_pk(TF, TV, TS, seed=0 + 100 * i, epochs=ep, loss_fn=A.make_loss(0.0), P=P, K=K) for i in range(R.N_NETS)]
    sc = R.L0(R.Member(units, TF, TV), allp["v4tr"], X4=dict(R.V4, vgg=F4, btc=V4v))
    # t3_relg: unchanged from v10 (anchors = every v4 person + every v1/v2 person with faces and voices)
    R.FACE["t3f"], R.VOICE["t3v"] = (F4, FE), (V4v, VE)
    gf, gv = R.id_rows(allp, "t3f", "face"), R.id_rows(allp, "t3v", "voice")
    rel = R.RelMulti([R.Side([a for a, _ in gf], [b for _, b in gf], colnorm=True)],
                     [R.Side([a for a, _ in gv], [b for _, b in gv], colnorm=True)])
    amale = np.array([R.GMAP.get(p) == "m" for p in rel.ids])
    keep = lambda vm: (amale[None, :] == vm[:, None])
    rr = np.random.RandomState(0); n = len(R.V4S)
    rf, rv = rr.randint(n, size=R.N_REF), rr.randint(n, size=R.N_REF)
    zg = R.Std(rel.score([F4[rf]], [V4v[rv]], R.V4S[rf], R.V4S[rv], keep=keep(R.VGEN4[rv])))
    M_, FEV = T3D / "feats_t3enc", R0 / "kaggle/output/feats_eval"

    def dev(cell, rows=None):
        tag = cell.replace("/", "_")
        f = np.concatenate([l2n(np.load(M_ / f"face_{n_}_{tag}.npy").astype(np.float32)) for n_ in ("farl16", "clipL")], 1)
        v = np.concatenate([l2n(np.load(M_ / f"voice_{n_}_{tag}.npy").astype(np.float32)) for n_ in ("sam100p", "sam100f")], 1)
        vg = R.pmale_voice(np.load(FEV / f"voice_w2v2aggen_{tag}.npy")) > 0.5
        if rows is not None:
            f, v, vg = f[rows], v[rows], vg[rows]
        z = np.zeros((len(f), 1), np.float32)
        S_ = sc.streams(dict(vgg=f, btc=v, ibf=np.ones((len(f), R.V4["ibf"].shape[1]), np.float32), iba=np.ones((len(f), R.V4["iba"].shape[1]), np.float32),
                             agef=z[:, 0], agev=z[:, 0]))
        return np.stack([S_["deep0"], S_["cca"], zg(rel.score([f], [v], keep=keep(vg)))], 1)
    out, checks = {}, {}
    D10 = np.load(T3D / "dev_t3_streamsf.npz")
    for cell in CELLS_BN:
        Z = dev(cell)
        half = np.sort(np.random.RandomState(1).choice(len(Z), len(Z) // 2, replace=False))
        diff = float(np.abs(dev(cell, half) - Z[half]).max())
        assert diff < 1e-5 and np.isfinite(Z).all(), (cell, diff)
        z10 = D10[cell.replace("/", "_")]
        drelg = float(np.abs(Z[:, 2] - z10[:, 2]).max())                                     # t3_relg must equal v10's
        rdeep = float(pd.Series(Z[:, 0]).corr(pd.Series(z10[:, 0]), method="spearman"))
        assert drelg < 1e-5, (cell, drelg)
        out[cell] = Z; checks[cell] = dict(rows=len(Z), level0_half_file_max_abs_diff=diff, t3_relg_vs_v10_max_abs_diff=drelg,
                                           t3_deep_rank_corr_vs_v10=rdeep)
        log(f"{cell}: T3 streams on {len(Z)} trials | level-0 self-test max |diff| {diff:.1e} | t3_relg = v10 ({drelg:.1e}) | "
            f"rank-corr of t3_deep vs v10 {rdeep:.3f}")
    np.savez(DEVS, **{c.replace("/", "_"): v for c, v in out.items()}, streams=np.array(T3S),
             info=json.dumps(dict(recipe=f"InfoNCE PK {P}x{K}", identities=len(set(TS)), vox_ids=len(set(S2)),
                                  dropped=sorted(T.dropped_speakers()), rows=len(TS), steps=ep * bpe, nets=R.N_NETS, checks=checks)))
    log(f"-> {DEVS.name} [{time.time() - t0:.0f}s]")


def zipit():
    sys.path[:0] = [str(R0 / "BEST/v7_safe"), str(R0 / "BEST/v6_eval"), str(R0 / "kaggle")]
    import safe_lib as S
    import flag_v2 as V
    import torch
    torch.set_num_threads(10)
    os.environ["FLAG_FEATS_EXTRA"] = os.pathsep.join(str(p) for p in FEATS[1:])
    store = V.FeatureStore(str(R0 / "kaggle_upload"), feats_dir=str(FEATS[0]))
    npy = lambda stem, split: store._npy(stem, split).astype(np.float32)

    def arrays(split):
        return dict(vgg=store.face("vgg", split), btc=store.voice("given", split), ibf=npy("face_ibv", split),
                    iba=npy("voice_iba", split), agef=npy("face_vitageprob", split) @ S.AGE_MIDS,
                    agev=npy("voice_w2v2agage", split)[:, 0] * 100, farl=npy("face_farl", split),
                    ada=npy("face_adaface", split), ecapa=npy("voice_ecapa192", split), vgen=npy("voice_w2v2aggen", split))

    def find(name, required=True):
        for d in EXTD:
            for f in (Path(d) / name, *Path(d).glob(f"*/{name}")):
                if f.exists():
                    return f
        assert not required, name
        return None
    ext = S.load_ext(find)
    b3 = S.B3Scorer(S.XallMember.load(R0 / "BEST/v7_safe/members/s007_xall.pt"), arrays("train"), store.spk, store.gmap, ext, log=log)
    ca9 = S.CAFuse(b3, json.loads((R0 / "BEST/v7_safe/cafuse_fuser.json").read_text()), arrays("train"), store.spk, store.gmap, ext, log=log)
    m11 = json.loads(FUSER.read_text())
    mean, scale, coef = (np.array(m11[k], np.float64) for k in ("mean", "scale", "coef"))
    DZ = np.load(DEVS)
    read = lambda zp: {k: pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None, names=["pid", "s"]) for k, fn in V.NAMES.items()}
    v9, v10 = read(V9), read(V10)
    scores, checks = {}, {"fuser": FUSER.name, "dev_t3_streams": DEVS.name, "t3_head": json.loads(str(DZ["info"]))}
    for k in V.CELLS:
        c = "/".join(k)
        assert (v9[k].pid.values == store.dev[k][2].pair_id.values).all(), c
        if k[1] == "English":
            scores[k] = -v9[k].s.values.astype(np.float64)                    # zips store -score: copied unchanged
            continue
        T = arrays(c)
        s9 = ca9(T, is_en=False)
        d9 = float(np.abs(s9 + v9[k].s.values).max())                         # wiring: must reproduce the v9 file
        r, rg = b3.relational(T)
        Z = np.concatenate([b3.streams(T), r[:, None], rg[:, None], DZ[c.replace("/", "_")]], 1)
        C = ca9.conditions(T, is_en=False)
        X = np.concatenate([Z, C, (Z[:, :, None] * C[:, None, :]).reshape(len(Z), -1)], 1)
        s = ((X - mean) / scale) @ coef + m11["intercept"]
        assert np.isfinite(s).all(), c
        rc9 = float(pd.Series(s).corr(pd.Series(s9), method="spearman"))
        rc10 = float(pd.Series(s).corr(pd.Series(-v10[k].s.values), method="spearman"))
        checks[c] = dict(rows=len(s), v9_reproduced_max_abs_diff=d9, rank_corr_vs_v9=rc9, rank_corr_vs_v10=rc10)
        log(f"{c}: v9 reproduced (max |diff| {d9:.1e}) | v11 on {len(s)} trials | rank-corr vs v9 {rc9:.3f}, vs v10 {rc10:.3f}")
        scores[k] = s
    V.write_zip(store, ZIP, scores); V.check_zip(store, ZIP)
    with zipfile.ZipFile(ZIP) as a, zipfile.ZipFile(V9) as b9:
        checks["files_identical_to_v9"] = {n: a.read(n) == b9.read(n) for n in a.namelist()}
    (HERE / "checks_v11.json").write_text(json.dumps(checks, indent=1, default=str))
    log("files identical to v9:", checks["files_identical_to_v9"])
    log("->", ZIP)


if __name__ == "__main__":
    {"fit": fit, "head": head, "zip": zipit}[sys.argv[1]]()
