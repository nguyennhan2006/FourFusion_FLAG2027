r"""T3ENC-01 (registered 07/10 in README.md before any result): the Track 3 encoder family (faces FaRL ep16 + CLIP ViT-L/14,
voices SimAM-ResNet100 VoxBlink2 pretrained + VoxCeleb2 fine-tuned; each L2-normalised, then concatenated) with SAFE-04's
XALL head recipe (EXP-007 MLP, InfoNCE, PK 32 x 8, 3 nets + RidgeCCA; code L0 with arrays=, as FLAG_28 / VOX2-01).
  A       head on v4tr + ext_pool (MAV-Celeb only), 1200 steps
  B       A + the 4,877 VoxCeleb2 identities of FLAG_37 / FLAG_41 (voice with a face of ANOTHER video), 1200 steps  <- decision
  BL      as B, 3600 steps (diagnostic)
  C       rel / relg profiles in the T3 space (anchors = the fold's train persons; no training)
  CHECK=1 also: the same code path on VGG / BTC must reproduce SAFE-04's XALL_deep0 (wiring)
    FOLDS=1,2,3,4,5 python run.py ; python run.py report > report.txt
"""
import os, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
os.environ["SAFE04_NOMAIN"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "SAFE-04")); sys.path.insert(0, str(HERE.parents[0] / "VOX2-01"))
LID = os.environ.get("VOX_LID")                       # FLAG_42 lid_summary.json: the registered Bengali filter
TAG = "t3enc01f" if LID else "t3enc01"
M_, X_ = HERE / "feats_t3enc", HERE / "feats_vox2t3"
VK = HERE.parents[0] / "VOX2-01/feats_vox2"
ARMS = {"A": (False, None), "B": (True, None), "BL": (True, 3600)}
if os.environ.get("T3_ARMS"):
    ARMS = {k: v for k, v in ARMS.items() if k in os.environ["T3_ARMS"].split(",")}


def strict_drop(src):
    """LIDSTRICT-01 (2/11): with LID_STRICT=1, the bn + as rule's speakers of src ('vox2' / 'vb'); empty without it."""
    import json
    if os.environ.get("LID_STRICT") != "1":
        return set()
    c = [HERE.parents[0] / "LIDSTRICT-01/lid_strict.json"]
    if Path("/kaggle/input").exists():
        c += list(Path("/kaggle/input").rglob("lid_strict.json"))
    hit = [p for p in c if p.exists()]
    assert hit, "LID_STRICT=1 needs LIDSTRICT-01/lid_strict.json (code dir or an attached dataset)"
    J = json.loads(hit[0].read_text())
    assert J.get("exp") == "LIDSTRICT-01" and all(len(J[k]) == J["counts"][k] for k in ("vox2", "vb")), "wrong lid_strict.json"
    return set(J[src])


def strict_counts():
    """[vox2, vb] of the active lid_strict.json (the adapter marker must carry the same pair)"""
    return [len(strict_drop("vox2")), len(strict_drop("vb"))]


def dropped_speakers():
    import json
    return (set(json.loads(Path(LID).read_text())["speakers_dropped_ids"]) if LID else set()) | strict_drop("vox2")
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def t3(kind, part):
    """part: 'v4' (v4 train rows), 'ext' (v1 then v2 rows = FMETA / VMETA order), 'vox' (FLAG_37 key-list rows)."""
    names = ("farl16", "clipL") if kind == "face" else ("sam100p", "sam100f")
    out = []
    for n in names:
        if part == "v4":
            a = np.load(M_ / f"{kind}_{n}_train.npy")
        elif part == "ext":
            a = np.concatenate([np.load(M_ / f"ext_{s}_{kind}_{n}.npy") for s in ("v1_complete", "v2_complete")])
        else:
            a = np.load(X_ / f"{kind}_{n}.npy")
        out.append(l2n(a.astype(np.float32)))
    return np.concatenate(out, 1)


def vox_rows(F, V):
    """VoxCeleb2 training rows (= VOX2-01 vox_rows): every voice clip with a face of ANOTHER video of the same speaker."""
    import pandas as pd
    fm = pd.read_csv(VK / "face_meta.csv", dtype={"video": str, "utt": str})
    vm = pd.read_csv(VK / "voice_meta.csv", dtype={"video": str, "utt": str})
    assert len(fm) == len(F) and len(vm) == len(V)
    keep, face_of, fvid, vvid = set(fm.spk) & set(vm.spk), fm.groupby("spk").indices, fm.video.values, vm.video.values
    keep -= dropped_speakers()                           # Bengali filter (registered 08/10): every clip of a flagged speaker
    Fi, Vi, S = [], [], []
    for s, idx in sorted(vm.groupby("spk").indices.items()):
        if s not in keep:
            continue
        for j, v in enumerate(sorted(idx)):
            cand = [f for f in face_of[s] if fvid[f] != vvid[v]] or list(face_of[s])
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vox2:" + s)
    return F[Fi], V[Vi], np.array(S)


def run(s):
    import pandas as pd, torch
    import rel_sweep5 as R
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))
    t0 = time.time()
    fold = R.make_fold(s); files = list(fold["files"].items())
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"], fi=d["fi"], vi=d["vi"])) for (data, prot), d in files],
                  ignore_index=True)
    F4, V4v, FE, VE = t3("face", "v4"), t3("voice", "v4"), t3("face", "ext"), t3("voice", "ext")
    assert len(F4) == len(R.V4S) and len(FE) == len(R.FMETA) and len(VE) == len(R.VMETA)
    trial = lambda d, data, F_, V_, Fe, Ve: dict(d, vgg=(F_ if data == "v4" else Fe)[d["fi"]], btc=(V_ if data == "v4" else Ve)[d["vi"]])
    todo = dict(ARMS)
    if os.environ.get("CHECK") == "1":
        todo = {"CHK": (False, None), **todo}
    for arm, (vox, steps) in todo.items():
        if arm == "CHK":                                   # VGG / BTC through the SAME arrays= path: must equal XALL_deep0
            arrs = (R.V4["vgg"], R.V4["btc"], R.FA["vgg"], R.VA["btc"]); X4 = R.V4
        else:
            arrs = (F4, V4v, FE, VE); X4 = dict(R.V4, vgg=F4, btc=V4v)
        TF, TV, TS, info = R.train_rows(fold, ext=True, arrays=arrs)
        if vox:
            F2, V2, S2 = vox_rows(t3("face", "vox"), t3("voice", "vox"))
            TF, TV, TS = np.concatenate([TF, F2]), np.concatenate([TV, V2]), np.concatenate([TS, S2])
        ep, bpe = R.epochs_for(len(TS), steps)
        units = [R.train_one(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep) for i in range(R.N_NETS)]
        sc = R.L0(R.Member(units, TF, TV), fold["v4tr"], X4=X4)
        S_ = [sc.streams(trial(d, data, *arrs) if arm != "CHK" else d) for (data, prot), d in files]
        D[f"t3{arm}_deep"] = np.concatenate([x["deep0"] for x in S_]); D[f"t3{arm}_cca"] = np.concatenate([x["cca"] for x in S_])
        R.log(f"fold {s} {arm}: {len(set(TS))} identities, {len(TS)} rows, {ep * bpe} steps [{time.time() - t0:.0f}s]")
    # C: rel / relg in the T3 space (= LAYERID-01's construction: Side per encoder, anchors = the fold's train persons)
    R.FACE["t3f"], R.VOICE["t3v"] = (F4, FE), (V4v, VE)
    tr = fold["v4tr"]; rr = np.random.RandomState(0)
    rf, rv = tr[rr.randint(len(tr), size=R.N_REF)], tr[rr.randint(len(tr), size=R.N_REF)]
    gf, gv = R.id_rows(fold, "t3f", "face"), R.id_rows(fold, "t3v", "voice")
    rel = R.RelMulti([R.Side([a for a, _ in gf], [b for _, b in gf], colnorm=True)],
                     [R.Side([a for a, _ in gv], [b for _, b in gv], colnorm=True)])
    amale = np.array([R.GMAP.get(p) == "m" for p in rel.ids])
    keep = lambda vm: (amale[None, :] == vm[:, None])
    for tag, kp in (("t3_rel", False), ("t3_relg", True)):
        z = R.Std(rel.score([F4[rf]], [V4v[rv]], R.V4S[rf], R.V4S[rv], keep=keep(R.VGEN4[rv]) if kp else None))
        col = []
        for (data, prot), d in files:
            j = 0 if data == "v4" else 1
            vm = (R.VGEN4 if j == 0 else R.VGENE)[d["vi"]]
            col.append(z(rel.score([(F4, FE)[j][d["fi"]]], [(V4v, VE)[j][d["vi"]]], keep=keep(vm) if kp else None)))
        D[tag] = np.concatenate(col)
    del R.FACE["t3f"], R.VOICE["t3v"]
    D.to_parquet(R.SP / f"out/safe04/{TAG}_fold{s}.parquet")
    R.log(f"fold {s} done [{time.time() - t0:.0f}s]")


def report():
    import json, pandas as pd
    from stream_gate import load, S6
    from lab_fuse import SP, table
    sys.path.insert(0, str(HERE.parents[0] / "STACK-02"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("stack02run", HERE.parents[0] / "STACK-02/run.py")
    S2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S2)
    D = load(TAG)
    R2 = pd.read_parquet(SP / "out/safe04/stack02_rows.parquet")
    assert (R2.y.values == D.y.values).all() and (R2.fi.values == D.fi.values).all() and (R2.vi.values == D.vi.values).all()
    D["fp"] = R2.fp.values
    D["BASE6_purged"] = pd.read_parquet(SP / "out/safe04/stack02/oof_diag.parquet")["BASE6"].values
    if "t3CHK_deep" in D:
        k = D.t3CHK_deep.notna()
        print(f"wiring ({k.sum()} rows): corr(t3CHK_deep, XALL_deep0) = {np.corrcoef(D.t3CHK_deep[k], D.XALL_deep0[k])[0, 1]:.4f} (must be 1.0000)")
    new = [c for c in D.columns if c.startswith("t3") and c != "t3CHK_deep" and c != "t3CHK_cca"]
    print("\n== 1. transfer check: standalone EER (mean of folds); CA-FUSE = BASE6 with purge (STACK-02)")
    T = table(D, {c: (lambda g, c=c: g[c].values) for c in ["BASE6_purged", "XALL_deep0", "XALL_cca", "relg"] + new})
    M = T.groupby("cell").mean(numeric_only=True)[["BASE6_purged", "XALL_deep0", "XALL_cca", "relg"] + new]
    print(M.round(2).to_string())
    ext_g = [c for c in M.index if c.endswith(" g") and not c.startswith("v4")]
    d_head = (M.loc[ext_g, "XALL_deep0"] - M.loc[ext_g, "t3B_deep"]).mean()
    d_ca = (M.loc[ext_g, "t3B_deep"] - M.loc[ext_g, "BASE6_purged"]).mean()
    print(f"   ext g cells: t3B_deep better than XALL_deep0 by {d_head:+.2f}; t3B_deep - CA-FUSE = {d_ca:+.2f}"
          f" -> head clearly better (>= 3)? {'yes' if d_head >= 3 else 'no'}")
    persons = {int(k): set(v) for k, v in json.loads((HERE.parents[0] / "STACK-02/fold_persons.json").read_text()).items()}
    sets = {"DECISION B: +t3B_deep,cca,relg": S6 + ["t3B_deep", "t3B_cca", "t3_relg"],
            "A: +t3A_deep,cca,relg": S6 + ["t3A_deep", "t3A_cca", "t3_relg"],
            "BL: +t3BL_deep,cca,relg": S6 + ["t3BL_deep", "t3BL_cca", "t3_relg"],
            "C: +t3_rel,relg": S6 + ["t3_rel", "t3_relg"],
            "replace B: deep0,cca <- t3B": ["t3B_deep", "t3B_cca", "XALL_ib", "XALL_age", "rel", "relg"]}
    sets = {k: v for k, v in sets.items() if all(c in D for c in v)}
    res = {}
    for regime, held in S2.HELD.items():
        for s in sorted(D.fold.unique()):
            te = D[(D.fold == s) & (D.data.isin(held) if held else True)]
            pool = S2.purge(D[(D.fold != s) & ~D.data.isin(held)], persons[int(s)])
            for name, st in {"BASE6": S6, **sets}.items():
                col = f"{name}|{regime}"
                if col not in D:
                    D[col] = np.nan
                D.loc[te.index, col] = S2.Design(pool, te, st).score(st, 1.0)
    T2 = table(D, {k: (lambda g, k=k: g[k].values) for k in D.columns if "|" in k})
    print("\n== 2. gate (purged; status PURGED, second-order leak not removed): gain vs BASE6 (+ = better), per cell (folds better)")
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    for name in sets:
        out = {}
        for regime in S2.HELD:
            sub = T2[["cell", f"BASE6|{regime}", f"{name}|{regime}"]].dropna()
            d = (sub[f"BASE6|{regime}"] - sub[f"{name}|{regime}"]).groupby(sub.cell)
            m, w = d.mean(), d.apply(lambda x: int((x > 0).sum()))
            g = [k for k in m.index if k.endswith(" g")]
            out[regime] = dict(all=m.mean(), g=m[g].mean() if g else np.nan, w4=int(sum(w[k] >= 4 for k in g)),
                               g_no_v4=m[[k for k in g if k != "v4 g"]].mean() if g else np.nan,
                               cells=" ".join(f"{k.replace(' ', '/')} {m[k]:+.2f}({w[k]})" for k in m.index))
        ok = out["oof"]["g"] >= 1.0 and out["oof"]["w4"] >= 2 and out["hold_urdu"]["all"] >= -0.2 and out["hold_hindi"]["all"] >= -0.2
        D2 = D.rename(columns={f"BASE6|oof": "ref_", f"{name}|oof": "cand_"})
        lo, hi = S2.boot(D2, "ref_", "cand_", gc, "both")
        print(f"  {name:<34} oof all {out['oof']['all']:+.2f} | g {out['oof']['g']:+.2f} (>= 4/5 in {out['oof']['w4']} g cells; "
              f"without v4 {out['oof']['g_no_v4']:+.2f}) | hold urdu {out['hold_urdu']['all']:+.2f} hindi {out['hold_hindi']['all']:+.2f} | "
              f"CI both sides [{lo:+.2f}, {hi:+.2f}] -> {'PASSES' if ok else 'fails'}" + ("   <- DECISION" if name.startswith("DECISION") else ""))
        print(f"      {out['oof']['cells']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            run(s)
