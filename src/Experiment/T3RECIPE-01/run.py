r"""T3RECIPE-01 (registered 10/10 in README.md before any result): Track 3's projector recipe (SUB19B: linear projector with
high dropout, shared AAM + cross-modal SupCon + prototype alignment, gender-balanced batches, same-gender SupCon, one AAM head
per auxiliary source, 40 epochs, an ensemble of encoder pairs x seeds; t3r.py) on our rows and the 4 T3 encoders, as a NEW
stream added to v11's Bangla CA-FUSE. Gate (default, CLAUDE.md 4.3) vs v11 = S6 + anP512_deep, anP512_cca, t3_relg.
  R        main = v4tr + ext_pool (MAV-Celeb), aux = VoxCeleb2 (LID-filtered) + VoxBlink-clean (LID + noise filter of
           SCALE-VB), 7 encoder pairs x 3 seeds                                                     <- DECISION
  RnoVB    as R without VoxBlink                 RCC    as R, one pair (all faces x all voices)      Rmain  as R, no aux
  R19      as R with SUB19B's hyper-parameters, 1 seed per pair (diagnostic vs trR_s0)
Amendment 11/10, before any result: every arm uses Track 3's SUB21a hyper-parameters (dropout face .70 / voice .40, 32 identities
per auxiliary step; their report of 11/10), SUB19B stays as the diagnostic arm R19.
    VOX_LID=... FOLDS=1,2,3,4,5 python run.py      # SAFE-04 folds -> <scratch>/out/safe04/t3recipe01_fold<s>.parquet
    VOX_LID=... python run.py report               # gate (e2e_lib.gate) -> report.txt next to the parquet files
    VOX_LID=... E2E_DIR=... python run.py final    # decision arm on every train person -> final_<split>.npz (test streams)
    VOX_LID=... E2E_DIR=... python run.py zip      # CA-FUSE refit (S6 + v11 T3 + trR) -> submission.zip (English = v11)
Env: TR_ARMS=R,RCC subset; TR_SEEDS (3); TR_EPOCHS (40) / TR_MAX_STEPS (smoke tests only); E2E_OUT (default <scratch>/out/e2e).
"""
import os, sys, time, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
spec = importlib.util.spec_from_file_location("allneg01run", EXP / "ALLNEG-01/run.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)          # puts SAFE-04, VOX2-01, T3ENC-01 on sys.path
assert A.LID, "T3RECIPE-01 uses the Bengali-filtered VoxCeleb2 rows: set VOX_LID"
sys.path.insert(0, str(EXP / "E2E"))                 # never HERE on sys.path: `import run` must stay T3ENC-01/run.py
_s = importlib.util.spec_from_file_location("t3r", HERE / "t3r.py"); t3r = importlib.util.module_from_spec(_s); _s.loader.exec_module(t3r)
sys.modules["t3r"] = t3r
TAG = "t3recipe01"
FACE = {"farl16": slice(0, 512), "clipL": slice(512, 1280), "cat": slice(0, 1280)}          # columns of T3ENC-01 t3("face")
VOICE = {"sam100p": slice(0, 256), "sam100f": slice(256, 512), "cat": slice(0, 512)}       # columns of t3("voice")
PAIRS = {"fp": ("farl16", "sam100p"), "ff": ("farl16", "sam100f"), "cp": ("clipL", "sam100p"), "cf": ("clipL", "sam100f"),
         "fC": ("farl16", "cat"), "cC": ("clipL", "cat"), "CC": ("cat", "cat")}
ARMS = {"R": dict(aux=("vox", "vb"), pairs=list(PAIRS)), "RnoVB": dict(aux=("vox",), pairs=list(PAIRS), seeds=1),
        "RCC": dict(aux=("vox", "vb"), pairs=["CC"]), "Rmain": dict(aux=(), pairs=list(PAIRS), seeds=1),
        "R19": dict(aux=("vox", "vb"), pairs=list(PAIRS), seeds=1, cfg=t3r.SUB19B)}
# amendment 11/10, before any result (Track 3's report: dropout .70 / .40 gave CB -0.55; 64 auxiliary identities per step no
# gain): the default hyper-parameters of every arm are SUB21a; R19 keeps SUB19B (1 seed per pair, compared with trR_s0)
BASE_CFG = t3r.SUB21A
# amendment 10/10, before any result (GPU time): the two data diagnostics use 1 seed per pair; R's seed-0 models give the
# matching column trR_s0, so R_s0 vs RnoVB vs Rmain compare ensembles of the same size
DECISION = "R"
if os.environ.get("TR_ARMS"):
    ARMS = {k: v for k, v in ARMS.items() if k in os.environ["TR_ARMS"].split(",")}
N_SEEDS = int(os.environ.get("TR_SEEDS", "3"))
CFG = {"epochs": int(os.environ["TR_EPOCHS"])} if os.environ.get("TR_EPOCHS") else {}
if os.environ.get("TR_MAX_STEPS"):                    # local smoke tests only
    CFG["max_steps"] = int(os.environ["TR_MAX_STEPS"])


def vb_module():
    spec2 = importlib.util.spec_from_file_location("scalevbrun", EXP / "SCALE-VB/run.py")
    M = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(M)
    return M


def sources(R, T, main_rows, aux, device):
    """main_rows = (TF, TV, TS) of the main identities -> t3r.Source objects (full T3 concatenations; pairs slice columns)."""
    TF, TV, TS = main_rows
    main = t3r.Source(TF, TV, TS, device, male=R.GMAP)
    assert main.male is not None and main.male.any() and (~main.male).any()
    out = []
    if "vox" in aux:
        F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
        out.append(("vox", t3r.Source(F2, V2, S2, device)))
    if "vb" in aux:
        FB, VB_, SB, info = vb_module().vb_rows(R.log)
        out.append(("vb", t3r.Source(FB, VB_, SB, device)))
    return main, out


def sliced(src, fs, vs):
    """A copy of a t3r.Source restricted to the columns of one encoder pair (views: the device tensors are shared)."""
    import copy
    out = copy.copy(src); out.F, out.V = src.F[:, fs], src.V[:, vs]
    return out


def arm_cfg(arm):
    return dict(ARMS[arm].get("cfg", BASE_CFG), **CFG)


def train_arm(R, main, aux_src, arm, seed0, device, tlog):
    """[(pair, seed, model)] of one arm."""
    o = ARMS[arm]
    aux = [s for k, s in aux_src if k in o["aux"]]
    models = []
    for j, p in enumerate(o["pairs"]):
        fs, vs = FACE[PAIRS[p][0]], VOICE[PAIRS[p][1]]
        m_ = sliced(main, fs, vs); a_ = [sliced(a, fs, vs) for a in aux]
        for i in range(min(N_SEEDS, o.get("seeds", N_SEEDS))):
            t0 = time.time()
            net, steps = t3r.train(m_, a_, m_.F.shape[1], m_.V.shape[1], arm_cfg(arm), seed=seed0 + 100 * j + i, device=device)
            models.append((p, i, net))
            tlog(f"  {arm} pair {p} seed {i}: {steps} steps [{time.time() - t0:.0f}s]")
    return models


def zscorers(R, models, F4, V4v, v4rows, device):
    """Std of every model from N_REF random pairs of the v4 train rows (RandomState(0), as L0)."""
    rr = np.random.RandomState(0)
    rf, rv = v4rows[rr.randint(len(v4rows), size=R.N_REF)], v4rows[rr.randint(len(v4rows), size=R.N_REF)]
    return [R.Std(t3r.pair_cos(net, F4[rf][:, FACE[PAIRS[p][0]]], V4v[rv][:, VOICE[PAIRS[p][1]]], device)) for p, _, net in models]


def stream(models, zs, Xf, Xv, device, per_pair=False):
    """Mean over the arm's models of the z-scored cosine; optionally the seed-mean per pair too."""
    parts = {}
    for (p, i, net), z in zip(models, zs):
        parts.setdefault(p, []).append(z(t3r.pair_cos(net, Xf[:, FACE[PAIRS[p][0]]], Xv[:, VOICE[PAIRS[p][1]]], device)))
    allz = np.mean([x for v in parts.values() for x in v], 0)
    return (allz, {p: np.mean(v, 0) for p, v in parts.items()}) if per_pair else allz


def run(s):
    import pandas as pd, torch
    import rel_sweep5 as R
    import run as T                                       # T3ENC-01/run.py
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    fold = R.make_fold(s); files = list(fold["files"].items())
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"], fi=d["fi"], vi=d["vi"])) for (data, prot), d in files],
                  ignore_index=True)
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    TF, TV, TS, info = R.train_rows(fold, ext=True, arrays=(F4, V4v, FE, VE))
    need = sorted({a for o in ARMS.values() for a in o["aux"]})
    main, aux_src = sources(R, T, (TF, TV, TS), need, device)
    R.log(f"fold {s}: main {len(main.ids)} identities / {main.n_rows} rows ({int(main.male.sum())} male); aux "
          f"{[(k, len(a.ids), a.n_rows) for k, a in aux_src]} | device {device}")
    Xf = np.concatenate([(F4 if data == "v4" else FE)[d["fi"]] for (data, prot), d in files])
    Xv = np.concatenate([(V4v if data == "v4" else VE)[d["vi"]] for (data, prot), d in files])
    for arm in ARMS:
        models = train_arm(R, main, aux_src, arm, 1000 * s, device, R.log)
        zs = zscorers(R, models, F4, V4v, fold["v4tr"], device)
        col, pp = stream(models, zs, Xf, Xv, device, per_pair=True)
        D[f"tr{arm}"] = col
        if os.environ.get("T3R_QF") or os.environ.get("T3R_EMA"):                  # QFEMA-01: extra columns only, tr<arm> above is untouched
            def _cols(ema):
                st = [t3r.pair_stats(net, Xf[:, FACE[PAIRS[p][0]]], Xv[:, VOICE[PAIRS[p][1]]], device, ema=ema) for p, _, net in models]
                if st[0] is None:
                    return None
                return st
            if os.environ.get("T3R_QF"):
                st = _cols(False)
                D[f"tr{arm}_nf"], D[f"tr{arm}_nv"] = np.mean([x[1] for x in st], 0), np.mean([x[2] for x in st], 0)
            if os.environ.get("T3R_EMA"):
                st = _cols(True)
                if st is not None:                                                  # z-score with the EMA models' own Std on the same v4 pairs
                    rr = np.random.RandomState(0); vr = fold["v4tr"]
                    rf, rv = vr[rr.randint(len(vr), size=R.N_REF)], vr[rr.randint(len(vr), size=R.N_REF)]
                    ze = [R.Std(t3r.pair_stats(net, F4[rf][:, FACE[PAIRS[p][0]]], V4v[rv][:, VOICE[PAIRS[p][1]]], device, ema=True)[0])
                          for p, _, net in models]
                    D[f"tr{arm}_ema"] = np.mean([z(x[0]) for z, x in zip(ze, st)], 0)
                    D[f"tr{arm}_ema_nf"], D[f"tr{arm}_ema_nv"] = np.mean([x[1] for x in st], 0), np.mean([x[2] for x in st], 0)
        if arm == DECISION:
            for p, v in pp.items():
                D[f"tr{arm}_{p}"] = v
            k0 = [k for k, (p, i, net) in enumerate(models) if i == 0]             # seed-0 models only (size of the diagnostics)
            D[f"tr{arm}_s0"] = stream([models[k] for k in k0], [zs[k] for k in k0], Xf, Xv, device)
        D.to_parquet(R.SP / f"out/safe04/{TAG}_fold{s}.parquet")
        R.log(f"fold {s} {arm}: {len(models)} models [{time.time() - t0:.0f}s]")
    R.log(f"fold {s} done [{time.time() - t0:.0f}s]")


def report():
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    base = S6 + E.V11_T3
    arms = {a: base + [f"tr{a}"] for a in ARMS}
    arms["R_repl"] = S6 + ["t3_relg", "trR"]                       # diagnostic: the recipe REPLACES v11's head
    arms["R_s0"] = base + ["trR_s0"]                                # diagnostic: R with 1 seed per pair (= RnoVB / Rmain size)
    cols = [f"tr{a}" for a in ARMS] + ["trR_s0"] + [f"trR_{p}" for p in PAIRS]
    return E.gate(TAG, arms, DECISION, standalone=cols, out=SP / "out/safe04" / f"{TAG}_report.txt")


def out_dir():
    import rel_sweep5 as R
    d = Path(os.environ.get("E2E_OUT", R.SP / "out/e2e")); d.mkdir(parents=True, exist_ok=True)
    return d


def final():
    """Decision arm on every train person (all v4 rows + every MAV-Celeb person, VoxCeleb2 + VoxBlink aux), streams of every
    test file of the split (components of E2E_DIR) with the level-0 self-test; models saved for the Evaluation run."""
    import torch
    import rel_sweep5 as R
    import run as T
    import e2e_lib as E
    from run_bridge import patch_large
    patch_large(vars(R))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    od = out_dir(); mf = od / f"{TAG}_final_models.pt"
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    allp = dict(s=0, v4tr=np.arange(len(R.V4S)), ext_pool=list(R.PERSONS), v4_train=sorted(set(R.V4S)))
    if mf.exists():                                               # resume / Evaluation run: the trained final models
        ck = torch.load(mf, map_location="cpu", weights_only=False)
        models = []
        for p, i, sd in ck["models"]:
            fs, vs = FACE[PAIRS[p][0]], VOICE[PAIRS[p][1]]
            width = lambda s: s.stop - s.start if isinstance(s, slice) else len(s)     # 16/10: index-array columns (FACELORA-03 B3)
            net = t3r.DualLinearProjector(width(fs), width(vs), ck["cfg"]["dim"], 0, 0); net.load_state_dict(sd); net.eval()
            models.append((p, i, net))
        R.log(f"final models loaded from {mf.name}: {len(models)}")
    else:
        TF, TV, TS, info = R.train_rows(allp, ext=True, arrays=(F4, V4v, FE, VE))
        main, aux_src = sources(R, T, (TF, TV, TS), ARMS[DECISION]["aux"], device)
        R.log(f"final {DECISION}: main {len(main.ids)} identities / {main.n_rows} rows; aux {[(k, len(a.ids), a.n_rows) for k, a in aux_src]}")
        models = train_arm(R, main, aux_src, DECISION, 0, device, R.log)
        torch.save(dict(models=[(p, i, net.state_dict()) for p, i, net in models], cfg=arm_cfg(DECISION), arm=DECISION,
                        main_ids=len(main.ids), aux={k: len(a.ids) for k, a in aux_src}), mf)
    zs = zscorers(R, models, F4, V4v, allp["v4tr"], device)
    out, checks = {}, {}
    for i, c in E.components().items():
        Xf, Xv, _ = E.t3_split(c)
        fn = lambda rows: stream(models, zs, Xf if rows is None else Xf[rows], Xv if rows is None else Xv[rows], device)
        s_, d = E.level0_selftest(fn, len(Xf))
        assert d < 1e-5 and np.isfinite(s_).all(), (c["key"], d)
        out[f"tr{DECISION}_{i}"] = s_; checks[c["key"]] = dict(rows=len(s_), level0_half_file_max_abs_diff=d)
        R.log(f"{c['key']}: tr{DECISION} on {len(s_)} trials | level-0 self-test {d:.1e}")
    np.savez(od / f"{TAG}_final_{E.SPLIT}.npz", **out, info=json.dumps(dict(arm=DECISION, models=len(models), checks=checks)))
    R.log(f"-> {TAG}_final_{E.SPLIT}.npz [{time.time() - t0:.0f}s]")


def zipit():
    import e2e_lib as E
    from stream_gate import S6
    od = out_dir()
    E.check_v11_fuser()
    st = S6 + E.V11_T3 + [f"tr{DECISION}"]
    model = E.fit_fuser(TAG, st)
    (od / f"{TAG}_fuser.json").write_text(json.dumps(model, indent=1))
    Z = np.load(od / f"{TAG}_final_{E.SPLIT}.npz")
    new = {i: Z[f"tr{DECISION}_{i}"] for i, c in E.components().items() if not c["is_en"]}
    scores = E.bangla_scores(model, new)
    zp = od / f"submission_{TAG}_{E.SPLIT}.zip"
    E.write_zip(zp, scores)
    rep = E.zip_report(zp, "v11")
    (od / f"{TAG}_zip_checks.json").write_text(json.dumps(dict(fuser_streams=st, files=rep), indent=1))
    E.log(f"-> {zp.name}: {rep}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "report":
        report()
    elif cmd == "final":
        final()
    elif cmd == "zip":
        zipit()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            run(s)
