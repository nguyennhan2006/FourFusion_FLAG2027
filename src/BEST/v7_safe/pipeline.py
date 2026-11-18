"""BEST v7 (CodaBench dev 23.28, log 105): the level-0 system of the Evaluation phase.

    English files  <- SET-17 level 0 (BEST/v6_eval/pipeline.py --mode indep --ib-all English --aclip 0.25, n = 5)
    Bangla files   <- B3 (Experiment/SAFE-04): s007 XALL (5 nets, 70 v4 + 151 MAV-Celeb v1/v2 persons) with the
                      registered weights FIXW + 0.599 rel + 0.478 relg (relational bridge over 221 train identities)

--bangla-fuser cafuse_fuser.json  (BEST v8, CodaBench dev 22.72, log 113): the Bangla files are scored by CAFuse, the
                      condition-aware fusion of the same B3 streams (Experiment/CAFUSE-01) instead of B3. English unchanged.
--english-fuser fuser_en.json      (with --bangla-fuser: BEST v9, CodaBench dev 22.63, log 116): the English files are
                      scored by CA-EN, the condition-aware fusion of SET-17's own level-0 streams (Experiment/CAFUSE-EN)
                      instead of SET-17 level 0. The v6 run is then skipped: CA-EN computes the s007 / s010B streams itself.

Every pair is scored on its own; every constant comes from train data (level 0, organiser ruling of 30/09).
The language rule (English / Bangla recipe chosen per file from the organiser's file language) was chosen after
seeing CodaBench: declared in the system description.

    python pipeline.py --data ../../kaggle_upload \
        --feats ../../kaggle/output/feats_eval ../../kaggle/output/feats_av ../../kaggle/output/feats_models \
        --ext <eda_bundle dir> ../../kaggle/output/feats_models \
        --members-from ../v6_eval/_members members --out _dev \
        --reference best=../../Experiment/SAFE-04/dev_zips/submission_REL2_E0_B3.zip

Needs, per split (train + every test file), arrays by name in the --feats folders:
  FLAG_12: face_ibv, voice_iba, face_vitageprob, voice_w2v2agage, voice_w2v2aggen, voice_ecapa192, voice_ecapa6144
  FLAG_14: face_aclip, voice_aclip (train + English files) FLAG_10: face_farl, face_adaface
and in the --ext folders the MAV-Celeb v1/v2 meta + arrays (eda bundle: ext_<src>_{face,voice}_meta.csv,
ext_v1_complete_speakers.csv, face_vgg / face_ibv / face_farl, voice_btc / voice_ecapa192 / voice_w2v2aggen;
face_adaface from feats_models).
Outputs in --out: submission.zip, v6/ (the SET-17 level-0 run; not with --english-fuser), config.json, checks.json
(level-0 self-tests, non-finite scores replaced, comparisons with --reference zips, sha256 of the model files and of the
zip, library versions).

BEST v9 (the Evaluation configuration): the same command plus
        --bangla-fuser cafuse_fuser.json --english-fuser fuser_en.json \
        --reference v9=../../Experiment/CAFUSE-EN/submission_CA_EN_BN.zip
"""
import argparse, hashlib, json, os, platform, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "v6_eval")); sys.path.insert(0, str(HERE.parents[1] / "kaggle"))
import safe_lib as S


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def finite(c, s):
    """Non-finite scores (a broken sample, e.g. an unreadable file) -> 0.0, a neutral score. Decided pair by pair, so
    the file stays level 0; the count goes to checks.json."""
    bad = ~np.isfinite(s)
    if bad.any():
        log(f"!! {c}: {int(bad.sum())} non-finite scores -> 0.0")
    return np.where(bad, 0.0, s), int(bad.sum())


def read_zip(path, names):
    with zipfile.ZipFile(path) as z:
        return {k: pd.read_csv(z.open(fn), sep=" ", header=None, names=["pid", "s"]) for k, fn in names.items()}


def english_ca(model, store, V, members_dir, n_models, gclf):
    """BEST v9 English side (Experiment/CAFUSE-EN/build_dev.py, log 115-116). CA-EN = logistic regression (fuser_en.json,
    fitted once on the 12 896 v4 validation trials of the 5 SAFE-04 folds) on
      7 streams     [s007 deep, s007 cca, s010B deep, s010B cca, ImageBind, age, AudioCLIP]   (z-scored, level 0)
      6 conditions  [P(male) face, P(male) voice, |difference|, age face, age voice, |age gap|]
      their 42 products.
    Streams and constants exactly as BEST/v6_eval/pipeline.py --mode indep computes them (members s007 + s010B, n = 5;
    train reference pairs of RandomState(0)); P(male) face = safe_lib.face_gender_model, as CAFuse.
    -> score(split, rows=None): higher = same, every trial on its own."""
    import importlib.util                                        # v6 is also called pipeline.py: load it by path
    spec = importlib.util.spec_from_file_location("v6_pipeline", HERE.parent / "v6_eval/pipeline.py")
    P6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P6)
    P6.CONFIG["members"] = {k: v for k, v in P6.CONFIG["members"].items() if k in ("s007", "s010B")}
    assert model["streams"] == ["s007_deep", "s007_cca", "s010B_deep", "s010B_cca", "ib", "age", "acl"], model["streams"]
    assert model["conditions"] == ["pf_m", "pv_m", "gdiff", "agef", "agev", "agegap"], model["conditions"]
    mean, scale, coef = (np.array(model[k], np.float64) for k in ("mean", "scale", "coef"))
    M = P6.train_members(V, store, n_models, Path(members_dir))
    rng = np.random.RandomState(0)
    n = len(store.spk)
    REF = (rng.randint(n, size=P6.CONFIG["ref_pairs"]), rng.randint(n, size=P6.CONFIG["ref_pairs"]))
    SC = {m: P6.Scorer(V, store, cfg, nets, REF) for m, (cfg, nets) in M.items()}
    pair, Std = P6.pair, P6.Std
    arr = lambda name, split: store._npy(name, split).astype(np.float32)
    Fi_tr, Ai_tr = arr("face_ibv", "train"), arr("voice_iba", "train")
    mf, mv = Fi_tr.mean(0), Ai_tr.mean(0)
    age_f = lambda sp: arr("face_vitageprob", sp) @ P6.AGE_MIDS
    age_v = lambda sp: arr("voice_w2v2agage", sp)[:, 0] * 100
    ib0 = lambda F_, A_: pair(V.l2n(F_ - mf), V.l2n(A_ - mv))
    zib = Std(ib0(Fi_tr[REF[0]], Ai_tr[REF[1]]))
    zag = Std(-np.abs(age_f("train")[REF[0]] - age_v("train")[REF[1]]))
    Fa_tr, Aa_tr = arr("face_aclip", "train"), arr("voice_aclip", "train")
    maf, mav = Fa_tr.mean(0), Aa_tr.mean(0)
    acl0 = lambda F_, A_: pair(V.l2n(F_ - maf), V.l2n(A_ - mav))
    zacl = Std(acl0(Fa_tr[REF[0]], Aa_tr[REF[1]]))
    log(f"CA-EN: {len(coef)} inputs, fitted on {model.get('n_trials')} validation trials; members {({m: len(v[1]) for m, v in M.items()})}")

    def score(split, rows=None):
        tk = (lambda X: X) if rows is None else (lambda X: X[rows])
        Z = []
        for m in ("s007", "s010B"):
            Af, Av = map(tk, SC[m].rows(split))
            Z.append(np.mean([sd(pair(*SC[m].embed0(net, prep, Af, Av))) for (net, prep), sd in zip(SC[m].nets, SC[m].stds)], 0))
            Z.append(SC[m].cstd(pair(SC[m].cca.transform_x(Af), SC[m].cca.transform_y(Av))))
        af, av = tk(age_f(split)), tk(age_v(split))
        Z += [zib(ib0(tk(arr("face_ibv", split)), tk(arr("voice_iba", split)))), zag(-np.abs(af - av)),
              zacl(acl0(tk(arr("face_aclip", split)), tk(arr("voice_aclip", split))))]
        Z = np.stack(Z, 1)
        pf = gclf.predict_proba(S.l2n(tk(arr("face_ibv", split))))[:, 1]
        pv = S.pmale(tk(arr("voice_w2v2aggen", split)))
        C = np.stack([pf, pv, np.abs(pf - pv), af, av, np.abs(af - av)], 1)
        X = np.concatenate([Z, C, (Z[:, :, None] * C[:, None, :]).reshape(len(Z), -1)], 1)
        return ((X - mean) / scale) @ coef + model["intercept"]
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--feats", required=True, nargs="+")
    ap.add_argument("--ext", required=True, nargs="+", help="folders with the MAV-Celeb v1/v2 meta + arrays")
    ap.add_argument("--members-from", nargs="*", default=[],
                    help="cached members: v6 folder (s007_seed*.pt, s010B_seed*.pt) and/or the folder of s007_xall.pt. "
                         "What is missing is TRAINED here (v6 members into <out>/members_v6, XALL into <out>/members_xall)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-models", type=int, default=5, help="SET-17 level-0 members (5 = the submitted 23.28)")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--xall-threads", type=int, help="CPU threads while TRAINING the XALL nets (default --threads). "
                    "CPU training depends on the thread count: the cached nets of 23.28 came from 12 threads (v6 members: 10)")
    ap.add_argument("--v6-out", help="reuse an earlier v6 run (folder with submission_indep.zip)")
    ap.add_argument("--reference", nargs="*", default=[], help="name=zip: compare the result with it, file by file")
    ap.add_argument("--bangla-fuser", help="cafuse_fuser.json: score the Bangla files with the condition-aware fusion of the "
                    "B3 streams (CAFUSE-01, CB 22.72) instead of B3 (23.28). Omit it to reproduce BEST v7 exactly")
    ap.add_argument("--english-fuser", help="fuser_en.json: score the English files with CA-EN (CAFUSE-EN; with --bangla-fuser "
                    "= BEST v9, CB 22.63) instead of SET-17 level 0. Omit it to reproduce BEST v7 / v8 exactly")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(a.threads)
    xall = next((Path(d) / "s007_xall.pt" for d in a.members_from if (Path(d) / "s007_xall.pt").exists()), None)
    v6m = next((d for d in a.members_from if (Path(d) / "s007_seed1.pt").exists() and (Path(d) / "s010B_seed1.pt").exists()),
               None) or str(out / "members_v6")                  # v6 trains (and caches) the missing seeds there
    (out / "config.json").write_text(json.dumps(dict(vars(a), FIXW=S.FIXW.tolist(), W_REL=S.W_REL, W_RELG=S.W_RELG,
                                                     rel_faces=S.REL_FACES, rel_voices=S.REL_VOICES), indent=1))

    # ---- 1. English side: SET-17 level 0 through the v6 pipeline (unchanged code); with --english-fuser CA-EN (below)
    v6dir = Path(a.v6_out) if a.v6_out else out / "v6"
    if not a.english_fuser and not (v6dir / "submission_indep.zip").exists():
        import importlib.util                                    # v6 is also called pipeline.py: load it by path
        spec = importlib.util.spec_from_file_location("v6_pipeline", HERE.parent / "v6_eval/pipeline.py")
        P6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P6)
        # v7 needs only s007 + s010B from v6 (English files). r2mix (Bangla cells of SET-17) is dropped: v7 scores Bangla
        # with B3, and r2mix's training crops follow the dev Bangla durations. English scores do not change (indep()
        # scores every member on its own; the reference pairs do not depend on the member list).
        P6.CONFIG["members"] = {k: v for k, v in P6.CONFIG["members"].items() if k != "r2mix"}
        P6.CONFIG["cells"] = dict(P6.CONFIG["cells"], Bangla=["s007"])
        sys.argv = ["pipeline.py", "--data", a.data, "--feats", *a.feats, "--out", str(v6dir), "--mode", "indep",
                    "--ib-all", "English", "--aclip", "0.25", "--n-models", str(a.n_models), "--members-from", v6m,
                    "--threads", str(a.threads)]
        log("v6 (SET-17 level 0):", " ".join(sys.argv[1:]))
        P6.main()
    import flag_v2 as V
    os.environ["FLAG_FEATS_EXTRA"] = os.pathsep.join(a.feats[1:])
    store = V.FeatureStore(a.data, feats_dir=a.feats[0])
    v6 = None if a.english_fuser else read_zip(v6dir / "submission_indep.zip", V.NAMES)

    # ---- 2. Bangla side: B3
    def npy(stem, split):
        return store._npy(stem, split).astype(np.float32)

    def arrays(split):
        return dict(vgg=store.face("vgg", split), btc=store.voice("given", split), ibf=npy("face_ibv", split),
                    iba=npy("voice_iba", split), agef=npy("face_vitageprob", split) @ S.AGE_MIDS,
                    agev=npy("voice_w2v2agage", split)[:, 0] * 100, farl=npy("face_farl", split),
                    ada=npy("face_adaface", split), ecapa=npy("voice_ecapa192", split), vgen=npy("voice_w2v2aggen", split))

    def find(name, required=True):
        for d in a.ext:
            for f in (Path(d) / name, *Path(d).glob(f"*/{name}")):
                if f.exists():
                    return f
        assert not required, f"{name} not found in {a.ext}"
        return None

    ext = S.load_ext(find)
    log(f"external persons: {ext[1].p.nunique()} (faces {len(ext[0])}, voices {len(ext[1])})")
    if xall is None:                                             # train the s007 XALL nets (FLAG_24 recipe)
        xall = out / "members_xall" / "s007_xall.pt"
        if not xall.exists():
            xall.parent.mkdir(parents=True, exist_ok=True)
            log("training s007 XALL (5 nets, 70 v4 + MAV-Celeb v1/v2 persons) ->", xall)
            torch.set_num_threads(a.xall_threads or a.threads)
            S.save_xall(S.train_xall(store.face("vgg", "train"), store.voice("given", "train"), store.spk, ext, log=log), xall)
            torch.set_num_threads(a.threads)
    sc = S.B3Scorer(S.XallMember.load(xall), arrays("train"), store.spk, store.gmap, ext, log=log)
    if a.bangla_fuser:                                           # v8: condition-aware fusion of the same B3 streams
        b3 = sc
        ca = S.CAFuse(b3, json.loads(Path(a.bangla_fuser).read_text()), arrays("train"), store.spk, store.gmap, ext, log=log)
        sc = lambda T: ca(T, is_en=False)
    en = None
    if a.english_fuser:                                          # v9: CA-EN on SET-17's own streams
        gclf = ca.gclf if a.bangla_fuser else S.face_gender_model(arrays("train")["ibf"], store.spk, store.gmap, ext)[0]
        en = english_ca(json.loads(Path(a.english_fuser).read_text()), store, V, v6m, a.n_models, gclf)
    scores, checks = {}, {}
    for k in V.CELLS:
        c = "/".join(k)
        if k[1] == "English" and en is None:
            d = v6[k]
            assert (d.pid.values == store.dev[k][2].pair_id.values).all(), c
            scores[k], nbad = finite(c, -d.s.values.astype(np.float64))
            checks[c] = dict(rows=len(d), nonfinite=nbad)
            continue
        if k[1] == "English":
            name, f = "CA-EN", (lambda rows=None, c=c: en(c, rows))
        else:
            T = arrays(c)
            name = "CA-FUSE" if a.bangla_fuser else "B3"
            f = lambda rows=None, T=T: sc(T if rows is None else {n: v[rows] for n, v in T.items()})
        s = f()
        half = np.sort(np.random.RandomState(1).choice(len(s), len(s) // 2, replace=False))
        sh, s0 = f(half), s[half]
        same_nan = bool((np.isfinite(sh) == np.isfinite(s0)).all())
        ok = np.isfinite(s0) & np.isfinite(sh)
        diff = float(np.abs(sh[ok] - s0[ok]).max()) if ok.any() else 0.0
        scores[k], nbad = finite(c, s)
        checks[c] = dict(rows=len(s), invariance_max_abs_diff=diff, nonfinite=nbad, passed=diff < 1e-5 and same_nan)
        log(f"{c}: {name} on {len(s)} trials; level-0 self-test (half the file) max |diff| {diff:.1e}")
        assert diff < 1e-5 and same_nan, c

    # ---- 3. zip + checks
    zp = out / "submission.zip"
    V.write_zip(store, zp, scores); V.check_zip(store, zp)
    for spec in a.reference:
        name, ref = spec.split("=", 1)
        mine, other = read_zip(zp, V.NAMES), read_zip(ref, V.NAMES)
        for k, fn in V.NAMES.items():
            x, y = mine[k], other[k]
            same = bool((x.pid.values == y.pid.values).all())
            rank_changes = int((x.s.rank(method="min").values != y.s.rank(method="min").values).sum())   # 0 -> same EER
            r = dict(same_pair_ids=same, max_abs_diff=float(np.abs(x.s - y.s).max()), rank_changes=rank_changes,
                     rank_corr=float(x.s.corr(y.s, method="spearman")))
            checks[f"vs {name}: {fn}"] = r
            log(f"vs {name} {fn}: same ids {same}, max |diff| {r['max_abs_diff']:.1e}, rank changes {rank_changes}, "
                f"rank-corr {r['rank_corr']:.6f}")
    used = {"bangla_fuser": a.bangla_fuser, "english_fuser": a.english_fuser, "s007_xall": str(xall),
            **{f"{m}_seed{1 + 100 * i}": str(Path(v6m) / f"{m}_seed{1 + 100 * i}.pt")
               for m in ("s007", "s010B") for i in range(a.n_models)}, "submission.zip": str(zp)}
    checks["sha256"] = {n: sha256(f) for n, f in used.items() if f and Path(f).exists()}
    import sklearn, scipy
    checks["versions"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              torch=torch.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    log("sha256:", {n: h[:12] for n, h in checks["sha256"].items()})
    log("->", zp)


if __name__ == "__main__":
    main()
