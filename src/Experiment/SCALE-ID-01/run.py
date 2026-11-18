r"""SCALE-ID-01 (registered 10/10 in README.md before any result): v11's T3 head (InfoNCE, PK 512 x 2) with more VoxCeleb2
data, extracted by FLAG_53 into feats_vox2b (FaRL16 / CLIP-L faces, SimAM100 p / ft voices, LID rule of FLAG_42):
  kind new    the VoxCeleb2 speakers FLAG_37 could not reach (vox2_mp4_1.zip is truncated) + the 120 VoxCeleb2 test speakers
  kind more   6 more faces + 8 more voices for FLAG_37's speakers (FLAG_43's design)
The new head REPLACES v11's head (anP512 deep / cca) in the Bangla CA-FUSE; t3_relg unchanged.
  B    = v11's head recipe on this device (v4tr + ext_pool + the 4 852 filtered VoxCeleb2 ids), ~1 200 steps   (GPU control)
  X    = B + kind new          M = B + kind more          XM = B + both                                       <- DECISION: XM
  X / M / XM train for the SAME number of epochs as B (steps scaled by their row count / B's)
A speaker flagged by FLAG_53's LID rule loses every clip (old rows included).
Gate (default, CLAUDE.md 4.3): CA-FUSE(S6 + scX_deep, scX_cca, t3_relg) vs v11 (S6 + anP512_deep, anP512_cca, t3_relg);
sensitivity: the same with this run's B as the reference.
    VOX_LID=... FOLDS=1,2,3,4,5 python run.py ; python run.py report ; E2E_DIR=... python run.py final ; python run.py zip
"""
import os, sys, time, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
spec = importlib.util.spec_from_file_location("allneg01run", EXP / "ALLNEG-01/run.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)          # puts SAFE-04, VOX2-01, T3ENC-01 on sys.path
assert A.LID, "SCALE-ID-01 uses the Bengali-filtered VoxCeleb2 rows: set VOX_LID"
sys.path.insert(0, str(EXP / "E2E"))                 # never HERE on sys.path: `import run` must stay T3ENC-01/run.py
TAG = "scaleid01"
VB2 = HERE / "feats_vox2b"
P, K = 512, 2
ARMS = {"B": dict(kinds=()), "X": dict(kinds=("new",)), "M": dict(kinds=("more",)), "XM": dict(kinds=("new", "more"))}
DECISION = "XM"
if os.environ.get("SC_ARMS"):
    ARMS = {k: v for k, v in ARMS.items() if k in os.environ["SC_ARMS"].split(",")}
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def lid_drop():
    return set(json.loads((VB2 / "lid_summary.json").read_text())["speakers_dropped_ids"])


def new_rows(kinds, log=print):
    """FLAG_53 rows of the given kinds: every voice clip of a kept speaker with a face of ANOTHER video of the same speaker
    (any face if it has one video) = T3ENC-01 vox_rows; for kind more the faces of FLAG_37 count too (old face rows) only
    through the old rows: here the new faces of the speaker are used. Speakers dropped by FLAG_53's LID rule are removed."""
    import pandas as pd
    fm = pd.read_csv(VB2 / "face_meta.csv", dtype=str); vm = pd.read_csv(VB2 / "voice_meta.csv", dtype={"spk": str, "video": str, "utt": str, "kind": str})
    F = np.concatenate([l2n(np.load(VB2 / f"face_{n}.npy").astype(np.float32)) for n in ("farl16", "clipL")], 1)
    V = np.concatenate([l2n(np.load(VB2 / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
    assert len(fm) == len(F) and len(vm) == len(V)
    fk, vk = fm.kind.isin(kinds).values, vm.kind.isin(kinds).values
    fm, F, vm, V = fm[fk].reset_index(drop=True), F[fk], vm[vk].reset_index(drop=True), V[vk]
    drop = lid_drop()
    keep = (set(fm.spk) & set(vm.spk)) - drop
    face_of, fvid, vvid = fm.groupby("spk").indices, fm.video.values, vm.video.values
    Fi, Vi, S = [], [], []
    for s, idx in sorted(vm.groupby("spk").indices.items()):
        if s not in keep:
            continue
        for j, v in enumerate(sorted(idx)):
            cand = [f for f in face_of[s] if fvid[f] != vvid[v]] or list(face_of[s])
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vox2:" + s)       # same label as FLAG_37's rows
    log(f"FLAG_53 rows {kinds}: {len(S)} rows, {len(set(S))} speakers")
    return F[Fi], V[Vi], np.array(S)


def head(R, T, TF, TV, TS, seed0, o, steps0=None):
    bpe = max(1, len(TS) // (P * K))
    steps = steps0 if steps0 else R.STEPS
    ep = max(1, int(round(steps / bpe)))
    units = [R.train_pk(TF, TV, TS, seed=seed0 + 100 * i, epochs=ep, loss_fn=A.make_loss(0.0), P=P, K=K) for i in range(R.N_NETS)]
    return units, ep * bpe


def rows_for(R, T, fold_like, kinds):
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    TF, TV, TS, info = R.train_rows(fold_like, ext=True, arrays=(F4, V4v, FE, VE))
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    parts = [(TF, TV, TS)]
    if kinds:                                                       # FLAG_53's LID rule also removes old rows of a flagged speaker
        ok = ~np.isin(S2, ["vox2:" + x for x in lid_drop()])
        parts.append((F2[ok], V2[ok], S2[ok])); parts.append(new_rows(kinds, R.log))
    else:
        parts.append((F2, V2, S2))
    return (F4, V4v, FE, VE), tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def run(s):
    import pandas as pd, torch
    import rel_sweep5 as R
    import run as T
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "4")))
    t0 = time.time()
    fold = R.make_fold(s); files = list(fold["files"].items())
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"], fi=d["fi"], vi=d["vi"])) for (data, prot), d in files],
                  ignore_index=True)
    nB = len(rows_for(R, T, fold, ())[1][2])
    for arm, o in ARMS.items():
        (F4, V4v, FE, VE), (TF, TV, TS) = rows_for(R, T, fold, o["kinds"])
        steps = int(round(R.STEPS * len(TS) / nB)) if o["kinds"] else None     # same epochs as B
        units, n_steps = head(R, T, TF, TV, TS, fold["s"], o, steps)
        sc = R.L0(R.Member(units, TF, TV), fold["v4tr"], X4=dict(R.V4, vgg=F4, btc=V4v))
        trial = lambda d, data: dict(d, vgg=(F4 if data == "v4" else FE)[d["fi"]], btc=(V4v if data == "v4" else VE)[d["vi"]])
        S_ = [sc.streams(trial(d, data)) for (data, prot), d in files]
        D[f"sc{arm}_deep"] = np.concatenate([x["deep0"] for x in S_]); D[f"sc{arm}_cca"] = np.concatenate([x["cca"] for x in S_])
        D.to_parquet(R.SP / f"out/safe04/{TAG}_fold{s}.parquet")
        R.log(f"fold {s} {arm}: {len(set(TS))} identities, {len(TS)} rows, {n_steps} steps (P {P} K {K}, {R.N_NETS} nets) [{time.time() - t0:.0f}s]")
    R.log(f"fold {s} done [{time.time() - t0:.0f}s]")


def report():
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    arms = {a: S6 + [f"sc{a}_deep", f"sc{a}_cca", "t3_relg"] for a in ARMS if a != "B"}
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    v = E.gate(TAG, arms, DECISION, standalone=[f"sc{a}_deep" for a in ARMS], out=out)
    if "B" in ARMS:                                      # sensitivity: the reference = this run's own B (same device)
        D = E.safe04(TAG)
        import pandas as pd
        print(f"\nwiring: corr(scB_deep, anP512_deep) = {np.corrcoef(D.scB_deep, D.anP512_deep)[0, 1]:.4f} (GPU vs the CPU run of v11)")
        armsB = {f"{a}_vsB": S6 + [f"sc{a}_deep", f"sc{a}_cca", "t3_relg"] for a in ARMS if a != "B"}
        E.V11_T3_BACKUP = E.V11_T3
        E.V11_T3 = ["scB_deep", "scB_cca", "t3_relg"]       # sensitivity only: reference streams = this run's B
        try:
            E.gate(TAG, armsB, None, out=SP / "out/safe04" / f"{TAG}_report_vsB.txt")
        finally:
            E.V11_T3 = E.V11_T3_BACKUP
        with open(out, "a", encoding="utf-8") as fh:
            fh.write("\n== sensitivity (reference = this run's B):\n" + (SP / "out/safe04" / f"{TAG}_report_vsB.txt").read_text(encoding="utf-8"))
    return v


def out_dir():
    import rel_sweep5 as R
    d = Path(os.environ.get("E2E_OUT", R.SP / "out/e2e")); d.mkdir(parents=True, exist_ok=True)
    return d


def final():
    """Decision arm on every train person + all VoxCeleb2 rows (old + new), EXPORTED (Experiment/E2E/v11_head.py export / Head:
    the Evaluation run reloads the same head), then the deep / cca streams of every test file (components) with the
    level-0 self-test."""
    import torch
    import rel_sweep5 as R
    import run as T
    import e2e_lib as E
    import v11_head as VH
    from run_bridge import patch_large
    patch_large(vars(R))
    t0 = time.time()
    od = out_dir(); hf = od / f"{TAG}_final_head.pt"
    if hf.exists():
        ck = torch.load(hf, map_location="cpu", weights_only=False); R.log(f"final head loaded from {hf.name}")
    else:
        allp = dict(s=0, v4tr=np.arange(len(R.V4S)), ext_pool=list(R.PERSONS), v4_train=sorted(set(R.V4S)))
        nB = len(rows_for(R, T, allp, ())[1][2])
        (F4, V4v, FE, VE), (TF, TV, TS) = rows_for(R, T, allp, ARMS[DECISION]["kinds"])
        units, n_steps = head(R, T, TF, TV, TS, 0, ARMS[DECISION], int(round(R.STEPS * len(TS) / nB)))
        sc = R.L0(R.Member(units, TF, TV), allp["v4tr"], X4=dict(R.V4, vgg=F4, btc=V4v))
        ck = VH.export(sc); ck["info"] = dict(arm=DECISION, identities=len(set(TS)), rows=len(TS), steps=n_steps)
        torch.save(ck, hf)
        R.log(f"final {DECISION}: {len(set(TS))} identities, {len(TS)} rows, {n_steps} steps -> {hf.name}")
    H = VH.Head(ck, R)
    out, checks = {}, {}
    for i, c in E.components().items():
        f, v, _ = E.t3_split(c)
        fn = lambda rows: np.stack([H.deep(f if rows is None else f[rows], v if rows is None else v[rows]),
                                    H.cca(f if rows is None else f[rows], v if rows is None else v[rows])], 1)
        Z, d = E.level0_selftest(fn, len(f))
        assert d < 1e-5 and np.isfinite(Z).all(), (c["key"], d)
        out[f"sc{DECISION}_{i}"] = Z; checks[c["key"]] = dict(rows=len(Z), level0_half_file_max_abs_diff=d)
        R.log(f"{c['key']}: sc{DECISION} deep / cca on {len(Z)} trials | level-0 self-test {d:.1e}")
    np.savez(od / f"{TAG}_final_{E.SPLIT}.npz", **out, info=json.dumps(dict(head=ck["info"], checks=checks)))
    R.log(f"-> {TAG}_final_{E.SPLIT}.npz [{time.time() - t0:.0f}s]")


def zipit():
    import e2e_lib as E
    from stream_gate import S6
    od = out_dir()
    E.check_v11_fuser()
    st = S6 + ["t3_relg", f"sc{DECISION}_deep", f"sc{DECISION}_cca"]
    model = E.fit_fuser(TAG, st)
    (od / f"{TAG}_fuser.json").write_text(json.dumps(model, indent=1))
    Z = np.load(od / f"{TAG}_final_{E.SPLIT}.npz")
    new = {i: Z[f"sc{DECISION}_{i}"] for i, c in E.components().items() if not c["is_en"]}
    scores = E.bangla_scores(model, new, keep_t3=(2,))           # the new head REPLACES anP512 deep / cca; t3_relg kept
    zp = od / f"submission_{TAG}_{E.SPLIT}.zip"
    E.write_zip(zp, scores)
    rep = E.zip_report(zp, "v11")
    (od / f"{TAG}_zip_checks.json").write_text(json.dumps(dict(fuser_streams=st, files=rep), indent=1))
    E.log(f"-> {zp.name}: {rep}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    {"report": report, "final": final, "zip": zipit}.get(cmd, lambda: [run(int(x)) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")])()
