r"""EN-T3-01 -> v12 dev zip (registered in README.md): ENGLISH files scored by CA_EN_T3 = v9's English fuser design (CA-EN:
s007 deep / cca, s010B deep / cca, ImageBind, age, AudioCLIP + 6 conditions + products) with the three T3 streams of v11's
final head (anP512 deep / cca, t3_relg) added; fuser fitted on every English validation row (v4 + en_ext). BANGLA files byte
for byte from v11 (BIGB-01/submission_v11.zip).
    VOX_LID=<lid_summary.json> python build_v12.py fit     # fuser_en_t3.json
    VOX_LID=... python build_v12.py head                   # T3 streams of the two dev English files (v11's final head)
    VOX_LID=... python build_v12.py zip                    # CA-EN streams on dev English (v9 code path) + T3 -> submission_v12.zip
"""
import importlib.util, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
R0 = HERE.parents[1]
FUSER, DEVS, ZIP = HERE / "fuser_en_t3.json", HERE / "dev_t3_streams_en.npz", HERE / "submission_v12.zip"
V9, V11 = R0 / "Experiment/CAFUSE-EN/submission_CA_EN_BN.zip", R0 / "Experiment/BIGB-01/submission_v11.zip"
SCR = Path(r"C:/Users/ASUS/AppData/Local/Temp/claude/d--Sinh-vi-n-CNhan-FLAG-2027-FourFusion/969f2c71-effc-43ec-a610-c253be9d1ea8/scratchpad")
FEATS = [R0 / "kaggle/output/feats_eval", R0 / "kaggle/output/feats_av", R0 / "kaggle/output/feats_models"]
EXTD = [SCR / "in/eda_bundle", R0 / "kaggle/output/feats_models"]
CELLS_EN = ["gender/English", "no_gender/English"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
design = lambda Z, C: np.concatenate([Z, C, (Z[:, :, None] * C[:, None, :]).reshape(len(Z), -1)], 1)
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
assert os.environ.get("VOX_LID"), "the T3 head is the Bengali-filtered one of v11: set VOX_LID"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    E = load("ent3run", HERE / "run.py")
    D = E.table_en()
    st = E.ARMS[E.DECISION]
    X = design(D[st].values, D[E.CO].values)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), D.y.values)
    model = dict(streams=["s007_deep", "s007_cca", "s010B_deep", "s010B_cca", "ib", "age", "acl", "t3_deep", "t3_cca", "t3_relg"],
                 validation_streams=st, conditions=E.CO, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                 intercept=float(lr.intercept_[0]), n_trials=int(len(D)), rows="v4 + en_ext English validation rows (finite AudioCLIP)")
    FUSER.write_text(json.dumps(model, indent=1))
    log(f"CA_EN_T3 fuser: {len(model['coef'])} inputs on {len(D)} English trials ({D.cell.value_counts().to_dict()}) -> {FUSER.name}")


def head():
    """= BIGB-01/build_v11.py head() (same rows, seeds, P512 recipe -> v11's final head), scored on the two English dev files."""
    import torch
    B = load("buildv11", R0 / "Experiment/BIGB-01/build_v11.py")
    A = B.allneg()
    import rel_sweep5 as R
    import run as T
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))
    t0 = time.time()
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    allp = dict(s=0, v4tr=np.arange(len(R.V4S)), ext_pool=list(R.PERSONS), v4_train=sorted(set(R.V4S)))
    TF, TV, TS, _ = R.train_rows(allp, ext=True, arrays=(F4, V4v, FE, VE))
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    TF, TV, TS = np.concatenate([TF, F2]), np.concatenate([TV, V2]), np.concatenate([TS, S2])
    bpe = max(1, len(TS) // (B.P * B.K)); ep = max(1, int(round(R.STEPS / bpe)))
    log(f"final T3 head (P512, = v11): {len(set(TS))} identities, {len(TS)} rows, {ep * bpe} steps, {R.N_NETS} nets")
    units = [R.train_pk(TF, TV, TS, seed=0 + 100 * i, epochs=ep, loss_fn=A.make_loss(0.0), P=B.P, K=B.K) for i in range(R.N_NETS)]
    sc = R.L0(R.Member(units, TF, TV), allp["v4tr"], X4=dict(R.V4, vgg=F4, btc=V4v))
    R.FACE["t3f"], R.VOICE["t3v"] = (F4, FE), (V4v, VE)
    gf, gv = R.id_rows(allp, "t3f", "face"), R.id_rows(allp, "t3v", "voice")
    rel = R.RelMulti([R.Side([a for a, _ in gf], [b for _, b in gf], colnorm=True)], [R.Side([a for a, _ in gv], [b for _, b in gv], colnorm=True)])
    amale = np.array([R.GMAP.get(p) == "m" for p in rel.ids])
    keep = lambda vm: (amale[None, :] == vm[:, None])
    rr = np.random.RandomState(0); n = len(R.V4S)
    rf, rv = rr.randint(n, size=R.N_REF), rr.randint(n, size=R.N_REF)
    zg = R.Std(rel.score([F4[rf]], [V4v[rv]], R.V4S[rf], R.V4S[rv], keep=keep(R.VGEN4[rv])))
    M_, FEV = R0 / "Experiment/T3ENC-01/feats_t3enc", R0 / "kaggle/output/feats_eval"

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
    D11 = np.load(R0 / "Experiment/BIGB-01/dev_t3_streams_v11.npz")
    zb = dev("gender/Bangla")                                                     # wiring: the head must equal v11's (Bangla)
    dwire = float(np.abs(zb - D11["gender_Bangla"]).max())
    log(f"wiring: T3 streams on gender/Bangla vs v11's dev_t3_streams_v11.npz: max |diff| {dwire:.1e}")
    assert dwire < 1e-4, dwire
    for cell in CELLS_EN:
        Z = dev(cell)
        half = np.sort(np.random.RandomState(1).choice(len(Z), len(Z) // 2, replace=False))
        diff = float(np.abs(dev(cell, half) - Z[half]).max())
        assert diff < 1e-5 and np.isfinite(Z).all(), (cell, diff)
        out[cell] = Z; checks[cell] = dict(rows=len(Z), level0_half_file_max_abs_diff=diff)
        log(f"{cell}: T3 streams on {len(Z)} trials | level-0 self-test {diff:.1e}")
    np.savez(DEVS, **{c.replace("/", "_"): v for c, v in out.items()}, info=json.dumps(dict(checks=checks, wiring_vs_v11=dwire)))
    log(f"-> {DEVS.name} [{time.time() - t0:.0f}s]")


def zipit():
    """CA-EN streams on dev English exactly as CAFUSE-EN/build_dev.py (v9), + the T3 streams; Bangla copied from v11."""
    from sklearn.linear_model import LogisticRegression
    model = json.loads(FUSER.read_text())
    sys.path[:0] = [str(R0 / "BEST/v7_safe"), str(R0 / "BEST/v6_eval"), str(R0 / "kaggle")]
    P6 = load("v6p", R0 / "BEST/v6_eval/pipeline.py")
    P6.CONFIG["members"] = {k: v for k, v in P6.CONFIG["members"].items() if k in ("s007", "s010B")}
    import safe_lib as S
    import flag_v2 as V
    import torch
    torch.set_num_threads(10)
    os.environ["FLAG_FEATS_EXTRA"] = os.pathsep.join(str(p) for p in FEATS[1:])
    store = V.FeatureStore(str(R0 / "kaggle_upload"), feats_dir=str(FEATS[0]))
    arr = lambda name, split: np.load(store._find(f"{name}_{split.replace('/', '_')}.npy")).astype(np.float32)
    M = P6.train_members(V, store, 5, R0 / "BEST/v6_eval/_members")
    rng = np.random.RandomState(0); n = len(store.spk)
    REF = (rng.randint(n, size=P6.CONFIG["ref_pairs"]), rng.randint(n, size=P6.CONFIG["ref_pairs"]))
    SC = {m: P6.Scorer(V, store, cfg, nets, REF) for m, (cfg, nets) in M.items()}
    pair, Std = P6.pair, P6.Std
    Fi_tr, Ai_tr = arr("face_ibv", "train"), arr("voice_iba", "train"); mf, mv = Fi_tr.mean(0), Ai_tr.mean(0)
    age_f = lambda sp: arr("face_vitageprob", sp) @ P6.AGE_MIDS
    age_v = lambda sp: arr("voice_w2v2agage", sp)[:, 0] * 100
    ib0 = lambda F_, A_: pair(V.l2n(F_ - mf), V.l2n(A_ - mv))
    zib = Std(ib0(Fi_tr[REF[0]], Ai_tr[REF[1]])); zag = Std(-np.abs(age_f("train")[REF[0]] - age_v("train")[REF[1]]))
    Fa_tr, Aa_tr = arr("face_aclip", "train"), arr("voice_aclip", "train"); maf, mav = Fa_tr.mean(0), Aa_tr.mean(0)
    acl0 = lambda F_, A_: pair(V.l2n(F_ - maf), V.l2n(A_ - mav))
    zacl = Std(acl0(Fa_tr[REF[0]], Aa_tr[REF[1]]))

    def find(name, required=True):
        for d in EXTD:
            for f in (Path(d) / name, *Path(d).glob(f"*/{name}")):
                if f.exists():
                    return f
        assert not required, name
    FMETA, VMETA, GEN, EA = S.load_ext(find)
    GM = dict(store.gmap); [GM.setdefault(k, v) for k, v in GEN.items()]
    Xg = np.concatenate([Fi_tr, np.asarray(EA[("face", "ibf")], np.float32)])
    yg = np.concatenate([[GM.get(p) == "m" for p in store.spk], [GM.get(p) == "m" for p in FMETA.p.values]])
    gclf = LogisticRegression(C=1.0, max_iter=3000).fit(S.l2n(Xg), yg)
    DZ = np.load(DEVS)
    mean, scale, coef = (np.array(model[k]) for k in ("mean", "scale", "coef"))
    fuser9 = json.loads((R0 / "Experiment/CAFUSE-EN/fuser_en.json").read_text())

    def scores(split, rows=None, with_t3=True):
        tk = (lambda X: X) if rows is None else (lambda X: X[rows])
        out = []
        for m in ("s007", "s010B"):
            Af, Av = map(tk, SC[m].rows(split))
            deep = np.mean([sd(pair(*SC[m].embed0(net, prep, Af, Av))) for (net, prep), sd in zip(SC[m].nets, SC[m].stds)], 0)
            cca = SC[m].cstd(pair(SC[m].cca.transform_x(Af), SC[m].cca.transform_y(Av)))
            out += [deep, cca]
        out += [zib(ib0(tk(arr("face_ibv", split)), tk(arr("voice_iba", split)))), zag(-np.abs(tk(age_f(split)) - tk(age_v(split)))),
                zacl(acl0(tk(arr("face_aclip", split)), tk(arr("voice_aclip", split))))]
        Z = np.stack(out, 1)
        pf = gclf.predict_proba(S.l2n(tk(arr("face_ibv", split))))[:, 1]; pv = S.pmale(tk(arr("voice_w2v2aggen", split)))
        af, av = tk(age_f(split)), tk(age_v(split))
        C = np.stack([pf, pv, np.abs(pf - pv), af, av, np.abs(af - av)], 1)
        if not with_t3:                                                           # v9's English score (wiring check)
            return ((design(Z, C) - np.array(fuser9["mean"])) / np.array(fuser9["scale"])) @ np.array(fuser9["coef"]) + fuser9["intercept"]
        Z = np.concatenate([Z, tk(DZ[split.replace("/", "_")])], 1)
        return ((design(Z, C) - mean) / scale) @ coef + model["intercept"]

    read = lambda zp: {k: pd.read_csv(zipfile.ZipFile(zp).open(fn), sep=" ", header=None, names=["pid", "s"]) for k, fn in V.NAMES.items()}
    v9, v11 = read(V9), read(V11)
    out, checks = {}, {"fuser": FUSER.name, "dev_t3_streams": DEVS.name}
    for k in V.CELLS:
        c = "/".join(k)
        assert (v11[k].pid.values == store.dev[k][2].pair_id.values).all(), c
        if k[1] != "English":
            out[k] = -v11[k].s.values.astype(np.float64)                          # zips store -score: Bangla copied from v11
            continue
        d9 = float(np.abs(scores(c, with_t3=False) + v9[k].s.values).max())       # v9 English reproduced
        s = scores(c)
        half = np.sort(np.random.RandomState(1).choice(len(s), len(s) // 2, replace=False))
        diff = float(np.abs(scores(c, half) - s[half]).max())
        assert diff < 1e-5 and np.isfinite(s).all(), (c, diff)
        rc = float(pd.Series(s).corr(pd.Series(-v9[k].s.values), method="spearman"))
        checks[c] = dict(rows=len(s), v9_reproduced_max_abs_diff=d9, level0_half_file_max_abs_diff=diff, rank_corr_vs_v9=rc)
        log(f"{c}: v9 English reproduced (max |diff| {d9:.1e}) | CA_EN_T3 on {len(s)} trials | rank-corr vs v9 {rc:.3f} | self-test {diff:.1e}")
        out[k] = s
    V.write_zip(store, ZIP, out); V.check_zip(store, ZIP)
    with zipfile.ZipFile(ZIP) as a, zipfile.ZipFile(V11) as b:
        checks["files_identical_to_v11"] = {n_: a.read(n_) == b.read(n_) for n_ in a.namelist()}
    (HERE / "checks_v12.json").write_text(json.dumps(checks, indent=1, default=str))
    log("files identical to v11:", checks["files_identical_to_v11"])
    log("->", ZIP)


if __name__ == "__main__":
    {"fit": fit, "head": head, "zip": zipit}[sys.argv[1]]()
