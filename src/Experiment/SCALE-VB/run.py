r"""SCALE-VB (registered 08/10 in README.md, amended 10/10 before any result): the T3 head (InfoNCE, PK 512 x 2) with the
VoxBlink-clean identities of FLAG_46 added to v4tr + ext_pool + the Bengali-filtered VoxCeleb2 rows. Control = v11's head
(anP512_* of BIGB-01, identical code path); gate vs v11.
    VOX_LID=<FLAG_42 lid_summary.json> FOLDS=1,2,3,4,5 python run.py      # VB_ARMS=VB5k,VBall,... for a subset
    VOX_LID=... python run.py report > report.txt
    VOX_LID=... python run.py rows                                         # only the preprocessing counts
"""
import os, sys, time, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
spec = importlib.util.spec_from_file_location("allneg01run", EXP / "ALLNEG-01/run.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)          # puts SAFE-04, VOX2-01, T3ENC-01 on sys.path
assert A.LID, "SCALE-VB is registered on the Bengali-filtered VoxCeleb2 rows: set VOX_LID"
TAG, REF_TAG = "scalevb01f", "bigb01f"
VB = HERE / "feats_vb"
P, K = 512, 2
ARMS = {"VB5k": dict(vox=True, vb=5000), "VBall": dict(vox=True, vb="all"), "VBall_L": dict(vox=True, vb="all", steps=3600),
        "VBonly": dict(vox=False, vb="all")}
DECISION = "VBall"
try:                                                   # GPU runs (FLAG_48) are not bit-identical to the CPU reference: add a
    import torch                                       # same-device control B (= v11's recipe: P512 on VoxCeleb2, no VoxBlink)
    _GPU = torch.cuda.is_available()
except ImportError:
    _GPU = False
if _GPU or os.environ.get("VB_B") == "1":
    ARMS = {"B": dict(vox=True, vb=0), **ARMS}
if os.environ.get("VB_ARMS"):
    ARMS = {k: v for k, v in ARMS.items() if k in os.environ["VB_ARMS"].split(",")}
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def loo_cos(V, spk):
    """Per clip: cosine to the mean of the OTHER clips of the same speaker (NaN for a speaker with one clip)."""
    import pandas as pd
    out = np.full(len(V), np.nan)
    for _, idx in pd.Series(np.arange(len(V))).groupby(np.asarray(spk)).indices.items():
        if len(idx) < 2:
            continue
        S = V[idx].sum(0)
        out[idx] = np.einsum("ij,ij->i", V[idx], l2n((S[None] - V[idx]) / (len(idx) - 1)))
    return out


_VB = {}


def vb_rows(log=print):
    """VoxBlink training rows, preprocessing locked in README.md: (1) speaker-level Bengali LID filter, (2) label-noise filter
    (clip LOO cosine of SimAM ft below the 1st percentile of the same statistic on the filtered VoxCeleb2 clips), (3) every
    voice clip with a face of ANOTHER video of the same speaker (any face of the speaker if it has one video)."""
    if "rows" in _VB:                                                             # same rows for every fold and arm
        return _VB["rows"]
    import pandas as pd
    import run as T                                                               # T3ENC-01/run.py
    fm = pd.read_csv(VB / "face_meta.csv", dtype={"video": str, "utt": str}); vm = pd.read_csv(VB / "voice_meta.csv", dtype={"video": str, "utt": str})
    F = np.concatenate([l2n(np.load(VB / f"face_{n}.npy").astype(np.float32)) for n in ("farl16", "clipL")], 1)
    V = np.concatenate([l2n(np.load(VB / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
    assert len(fm) == len(F) and len(vm) == len(V), (len(fm), len(F), len(vm), len(V))
    drop = set(json.loads((VB / "lid_summary.json").read_text())["speakers_dropped_ids"])
    # threshold from VoxCeleb2 (already hand-verified labels), Bengali-filtered as in T3ENC-01
    xm = pd.read_csv(EXP / "VOX2-01/feats_vox2/voice_meta.csv", dtype={"video": str, "utt": str})
    XV = l2n(np.load(EXP / "T3ENC-01/feats_vox2t3/voice_sam100f.npy").astype(np.float32))
    keepx = ~xm.spk.isin(T.dropped_speakers()).values
    thr = float(np.nanquantile(loo_cos(XV[keepx], xm.spk.values[keepx]), 0.01))
    vf = l2n(np.load(VB / "voice_sam100f.npy").astype(np.float32))
    c = loo_cos(vf, vm.spk.values)
    lid_ok = ~vm.spk.isin(drop).values
    noisy = lid_ok & (c < thr)                                                    # NaN (single clip) compares False: kept
    ok = lid_ok & ~noisy
    keep_spk = set(vm.spk[ok]) & set(fm.spk[~fm.spk.isin(drop)])
    face_of, fvid, vvid = fm.groupby("spk").indices, fm.video.values, vm.video.values
    Fi, Vi, S, same = [], [], [], 0
    vok = vm[ok]
    for s, idx in sorted(vok.groupby("spk").indices.items()):
        if s not in keep_spk:
            continue
        rows = vok.index.values[idx]
        for j, v in enumerate(sorted(rows)):
            cand = [f for f in face_of[s] if fvid[f] != vvid[v]]
            if not cand:
                cand = list(face_of[s]); same += 1
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vb:" + s)
    info = dict(clips=len(vm), speakers=int(vm.spk.nunique()), lid_dropped_speakers=len(drop & set(vm.spk)),
                lid_dropped_clips=int((~lid_ok).sum()), noise_threshold=round(thr, 4), noise_dropped_clips=int(noisy.sum()),
                single_clip_kept=int((lid_ok & np.isnan(c)).sum()), rows=len(S), speakers_kept=len(set(S)),
                same_video_pairs=same, same_video_share=round(same / max(1, len(S)), 4))
    log(f"VoxBlink rows: {info}")
    _VB["rows"] = (F[Fi], V[Vi], np.array(S), info)
    return _VB["rows"]


def run(s):
    import pandas as pd, torch
    import rel_sweep5 as R
    import run as T
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))
    t0 = time.time()
    fold = R.make_fold(s); files = list(fold["files"].items())
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"], fi=d["fi"], vi=d["vi"])) for (data, prot), d in files],
                  ignore_index=True)
    F4, V4v, FE, VE = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    arrs = (F4, V4v, FE, VE); X4 = dict(R.V4, vgg=F4, btc=V4v)
    trial = lambda d, data: dict(d, vgg=(F4 if data == "v4" else FE)[d["fi"]], btc=(V4v if data == "v4" else VE)[d["vi"]])
    TF0, TV0, TS0, info = R.train_rows(fold, ext=True, arrays=arrs)
    FX, VX, SX = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    FB, VBv, SB, binfo = vb_rows(R.log)
    vb_ids = np.array(sorted(set(SB)))
    for arm, o in ARMS.items():
        if o["vb"] == "all":
            mb = np.ones(len(SB), bool)
        elif o["vb"] == 0:
            mb = np.zeros(len(SB), bool)
        else:                                                                     # fixed subset of speakers, RandomState(0)
            pick = set(np.random.RandomState(0).choice(vb_ids, o["vb"], replace=False)); mb = np.isin(SB, list(pick))
        parts = [(TF0, TV0, TS0)] + ([(FX, VX, SX)] if o["vox"] else []) + [(FB[mb], VBv[mb], SB[mb])]
        TF, TV, TS = (np.concatenate([p[i] for p in parts]) for i in range(3))
        bpe = max(1, len(TS) // (P * K)); ep = max(1, int(round(o.get("steps", R.STEPS) / bpe)))
        units = [R.train_pk(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep, loss_fn=A.make_loss(0.0), P=P, K=K) for i in range(R.N_NETS)]
        sc = R.L0(R.Member(units, TF, TV), fold["v4tr"], X4=X4)
        S_ = [sc.streams(trial(d, data)) for (data, prot), d in files]
        D[f"vb{arm}_deep"] = np.concatenate([x["deep0"] for x in S_]); D[f"vb{arm}_cca"] = np.concatenate([x["cca"] for x in S_])
        R.log(f"fold {s} {arm}: {len(set(TS))} identities ({len(set(SB[mb]))} VoxBlink), {len(TS)} rows, {ep * bpe} steps "
              f"(P {P} K {K}, {R.N_NETS} nets) [{time.time() - t0:.0f}s]")
        D.to_parquet(R.SP / f"out/safe04/{TAG}_fold{s}.parquet")
    R.log(f"fold {s} done [{time.time() - t0:.0f}s]")


def report():
    import pandas as pd
    sys.path.insert(0, str(EXP / "STACK-02"))
    spec2 = importlib.util.spec_from_file_location("stack02run", EXP / "STACK-02/run.py")
    S2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(S2)
    from stream_gate import load, S6
    from lab_fuse import SP, table
    D = load(TAG); B = load(REF_TAG)
    T3 = pd.concat([pd.read_parquet(f) for f in sorted((SP / "out/safe04").glob("t3enc01f_fold*.parquet"))], ignore_index=True)
    for X in (B, T3):
        assert (X.y.values == D.y.values).all() and (X.vi.values == D.vi.values).all()
    D["ref_deep"], D["ref_cca"], D["t3_relg"] = B.anP512_deep.values, B.anP512_cca.values, T3.t3_relg.values
    R2 = pd.read_parquet(SP / "out/safe04/stack02_rows.parquet"); D["fp"] = R2.fp.values
    persons = {int(k): set(v) for k, v in json.loads((EXP / "STACK-02/fold_persons.json").read_text()).items()}
    arms = [a for a in ARMS if f"vb{a}_deep" in D]
    print("== standalone EER of each head (mean of folds); ref = v11's head (anP512 of BIGB-01)")
    cols = ["ref_deep"] + [f"vb{a}_deep" for a in arms]
    print(table(D, {c: (lambda g, c=c: g[c].values) for c in cols}).groupby("cell").mean(numeric_only=True)[cols].round(2).to_string())
    def gate(ref_cols, label):
        sets = {"ref": S6 + ref_cols + ["t3_relg"], **{a: S6 + [f"vb{a}_deep", f"vb{a}_cca", "t3_relg"] for a in arms if a != "B"}}
        cols = {}
        for regime, held in S2.HELD.items():
            for s in sorted(D.fold.unique()):
                te = D[(D.fold == s) & (D.data.isin(held) if held else True)]
                pool = S2.purge(D[(D.fold != s) & ~D.data.isin(held)], persons[int(s)])
                for name, st in sets.items():
                    col = f"{name}|{regime}"
                    cols.setdefault(col, pd.Series(np.nan, index=D.index))
                    cols[col].loc[te.index] = S2.Design(pool, te, st).score(st, 1.0)
        E = pd.concat([D[["fold", "cell", "y"]], pd.DataFrame(cols)], axis=1)
        T2 = table(E, {k: (lambda g, k=k: g[k].values) for k in cols})
        gc = sorted(c for c in E.cell.unique() if c.endswith(" g"))
        print(f"\n== gate: CA-FUSE(S6 + arm deep, cca, t3_relg) vs {label}; purged; + = better; per cell (folds better)")
        for name in [a for a in arms if a != "B"]:
            out = {}
            for regime in S2.HELD:
                sub = T2[["cell", "fold", f"ref|{regime}", f"{name}|{regime}"]].dropna()
                dd = sub.assign(d=sub[f"ref|{regime}"] - sub[f"{name}|{regime}"])
                m, w = dd.groupby("cell").d.mean(), dd.groupby("cell").d.apply(lambda x: int((x > 0).sum()))
                harm = [k for k in m.index if int((dd[dd.cell == k].d <= -0.5).sum()) >= 4]
                g = [k for k in m.index if k.endswith(" g")]
                out[regime] = dict(all=m.mean(), g=m[g].mean() if g else np.nan, w4=int(sum(w[k] >= 4 for k in g)),
                                   g_no_v4=m[[k for k in g if k != "v4 g"]].mean() if g else np.nan, harm=harm,
                                   cells=" ".join(f"{k.replace(' ', '/')} {m[k]:+.2f}({w[k]})" for k in m.index))
            ok = out["oof"]["g"] >= 1.0 and out["oof"]["w4"] >= 2 and out["hold_urdu"]["all"] >= -0.2 and out["hold_hindi"]["all"] >= -0.2
            E2 = E.rename(columns={"ref|oof": "ref_", f"{name}|oof": "cand_"}).assign(fp=D.fp.values, vp=D.vp.values)
            lo, hi = S2.boot(E2, "ref_", "cand_", gc, "both")
            print(f"  {name:<8} oof all {out['oof']['all']:+.2f} | g {out['oof']['g']:+.2f} (>= 4/5 in {out['oof']['w4']} g cells; without v4 "
                  f"{out['oof']['g_no_v4']:+.2f}) | hold urdu {out['hold_urdu']['all']:+.2f} hindi {out['hold_hindi']['all']:+.2f} | CI both sides "
                  f"[{lo:+.2f}, {hi:+.2f}] -> {'PASSES' if ok else 'fails'}" + ("   <- DECISION" if name == DECISION else ""))
            print(f"           {out['oof']['cells']}")
            print(f"           harm check (cells worse by >= 0.5 in >= 4/5 folds, oof): {out['oof']['harm'] or 'none'}")
    gate(["ref_deep", "ref_cca"], "v11 (S6 + anP512_* of BIGB-01, CPU), the registered gate")
    if "vbB_deep" in D:
        print(f"\nsame-device control: corr(vbB_deep, v11 anP512_deep) = {np.corrcoef(D.vbB_deep, D.ref_deep)[0, 1]:.4f} (CPU: 1.0000; GPU >= 0.99)")
        gate(["vbB_deep", "vbB_cca"], "this run's own B (same device; sensitivity, not the registered gate)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "report":
        report()
    elif cmd == "rows":
        vb_rows()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            run(s)
