r"""ENBASE-01 (registered 10/10 in README.md): does the T3 gain shrink as the real English base (s007 + s010B, v4 rows only)
gets stronger with more v4 training persons? 5 splits x nested N = 20/30/40/50 (+ N50 with 5 nets for the base).
    VOX_LID=<FLAG_42 lid_summary.json> FOLDS=1,2,3,4,5 python run.py      # one split per FOLDS value (FLAG_48: one per process)
    python run.py report > report.txt                                   # local: fusion F1 / F0, EER, dEER, slopes
"""
import os, sys, time, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP, R0 = HERE.parents[0], HERE.parents[1]
TAG = "enbase01"
NS = tuple(int(x) for x in os.environ.get("ENBASE_NS", "20,30,40,50").split(","))          # override only for the smoke test
N_HELD = 20
E6_DIRS = [R0 / "kaggle/output/feats_e6144", R0 / "kaggle/output/feats_eval"]


def e6144(R):
    """Real ECAPA-6144 arrays: v4 train rows, then v1 + v2 voice rows (VMETA order). Checked for completeness."""
    d = next(p for p in E6_DIRS if (p / "voice_ecapa6144_train.npy").exists() and np.load(p / "voice_ecapa6144_train.npy", mmap_mode="r").shape[1] == 6144)
    E4 = np.load(d / "voice_ecapa6144_train.npy").astype(np.float32)
    EE = np.concatenate([np.load(d / f"ext_{s}_voice_ecapa6144.npy").astype(np.float32) for s in ("v1_complete", "v2_complete")])
    assert E4.shape == (len(R.V4S), 6144) and EE.shape == (len(R.VMETA), 6144), (E4.shape, EE.shape)
    assert np.isfinite(E4).all() and np.isfinite(EE).all()
    # sanity: the voices separate v4 persons (same vs different person cosine)
    rr = np.random.RandomState(0); i, j = rr.randint(len(E4), size=(2, 20000)); n4 = E4 / (np.linalg.norm(E4, axis=1, keepdims=True) + 1e-8)
    c = (n4[i] * n4[j]).sum(1); same = R.V4S[i] == R.V4S[j]
    R.log(f"ECAPA-6144 from {d}: v4 {E4.shape}, v1+v2 {EE.shape}; cos same person {c[same].mean():.3f} vs different {c[~same].mean():.3f}")
    return E4, EE


def split(R, s):
    """20 held-out v4 persons by gender (9 f / 11 m), then the other 50 in a fixed gender-balanced order (nested subsets)."""
    v4p = np.unique(R.V4S)
    f = [p for p in v4p if R.GMAP[p] == "f"]; m = [p for p in v4p if R.GMAP[p] == "m"]
    r = np.random.RandomState(7000 + s); nf = int(round(N_HELD * len(f) / len(v4p)))
    held = sorted(list(r.permutation(f)[:nf]) + list(r.permutation(m)[:N_HELD - nf]))
    rf = [p for p in f if p not in held]; rm = [p for p in m if p not in held]
    r2 = np.random.RandomState(7100 + s); rf, rm = list(r2.permutation(rf)), list(r2.permutation(rm))
    subsets = {}
    for N in NS:
        k = int(round(N * len(rf) / (len(rf) + len(rm)))); subsets[N] = sorted(rf[:k] + rm[:N - k])
    for a, b in zip(NS, NS[1:]):
        assert set(subsets[a]) <= set(subsets[b])
    return held, subsets


def run(s):
    import pandas as pd, torch
    spec = importlib.util.spec_from_file_location("allneg01run", EXP / "ALLNEG-01/run.py")
    A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)          # SAFE-04 / VOX2-01 / T3ENC-01 on sys.path
    import rel_sweep5 as R
    import run as T                                                               # T3ENC-01/run.py
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))
    t0 = time.time()
    E4, EE = e6144(R)
    held, subsets = split(R, s)
    fs = R.make_fold(s)                                                           # en_ext files (ext_held of SAFE-04 fold s) + ext_pool
    files = {("v4", "ng"): R.v4_file(held, False, np.random.RandomState(3000 + s)), ("v4", "g"): R.v4_file(held, True, np.random.RandomState(3000 + s)),
             ("en_ext", "ng"): fs["files"][("en_ext", "ng")], ("en_ext", "g"): fs["files"][("en_ext", "g")]}
    r8 = np.random.RandomState(8000 + s)
    for sg, prot in ((False, "ng"), (True, "g")):                                 # fusion-fit set: ext_pool persons, never seen by base or T3
        files[("fit_en", prot)] = R.ext_file(fs["ext_pool"], "english", sg, r8)
    assert not set(fs["ext_held"]) & set(fs["ext_pool"])
    D = pd.concat([pd.DataFrame(dict(split=s, fold=s, data=data, protocol=prot, y=d["y"], fi=d["fi"], vi=d["vi"], pf=d["pf"], pv=d["pv"]))
                   for (data, prot), d in files.items()], ignore_index=True)
    R.log(f"split {s}: held {len(held)} v4 persons; files " + ", ".join(f"{a} {b} {len(d['y'])}" for (a, b), d in files.items()))
    F4t, V4t, FEt, VEt = T.t3("face", "v4"), T.t3("voice", "v4"), T.t3("face", "ext"), T.t3("voice", "ext")
    FX, VX, SX = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    R.FACE["t3f"], R.VOICE["t3v"] = (F4t, FEt), (V4t, VEt)
    is4 = lambda data: data == "v4"
    for N in NS:
        persons = subsets[N]
        fold = dict(s=s, v4_train=persons, v4tr=np.where(np.isin(R.V4S, persons))[0], ext_pool=[], v4_held=held)
        variants = [(f"N{N}", R.N_NETS)] + ([(f"N{N}x5", 5)] if N == NS[-1] else [])
        for tag, nets in variants:                                                # ---- base: s007 (VGG / BTC) and s010B (VGG / ECAPA-6144)
            for mem, arrs, X4, vsel in (("s007", None, None, lambda data: None),
                                        ("s010B", (R.V4["vgg"], E4, None, None), dict(R.V4, btc=E4), lambda data: E4 if is4(data) else EE)):
                TF, TV, TS, _ = R.train_rows(fold, ext=False, arrays=arrs)
                ep, bpe = R.epochs_for(len(TS))
                units = [R.train_one(TF, TV, TS, seed=1000 * s + N + 100 * i, epochs=ep) for i in range(nets)]
                sc = R.L0(R.Member(units, TF, TV), fold["v4tr"], X4=X4)
                S_ = [sc.streams(d if vsel(data) is None else dict(d, btc=vsel(data)[d["vi"]])) for (data, prot), d in files.items()]
                for k in ("deep0", "cca", "prod0") + (("ib", "age") if mem == "s007" else ()):
                    D[f"{tag}_{mem}_{k}"] = np.concatenate([x[k] for x in S_])
                R.log(f"split {s} {tag} {mem}: {len(set(TS))} persons, {len(TS)} rows, {ep * bpe} steps, {nets} nets [{time.time() - t0:.0f}s]")
        # ---- T3 head on the N v4 persons + VoxCeleb2 (no v1/v2), P512, 3 nets; + t3_relg with anchors = the N persons
        TF, TV, TS, _ = R.train_rows(fold, ext=False, arrays=(F4t, V4t, FEt, VEt))
        TF, TV, TS = np.concatenate([TF, FX]), np.concatenate([TV, VX]), np.concatenate([TS, SX])
        bpe = max(1, len(TS) // (512 * 2)); ep = max(1, int(round(R.STEPS / bpe)))
        units = [R.train_pk(TF, TV, TS, seed=1000 * s + N + 100 * i, epochs=ep, loss_fn=A.make_loss(0.0), P=512, K=2) for i in range(R.N_NETS)]
        sc = R.L0(R.Member(units, TF, TV), fold["v4tr"], X4=dict(R.V4, vgg=F4t, btc=V4t))
        gf, gv = R.id_rows(fold, "t3f", "face"), R.id_rows(fold, "t3v", "voice")
        rel = R.RelMulti([R.Side([a for a, _ in gf], [b for _, b in gf], colnorm=True)], [R.Side([a for a, _ in gv], [b for _, b in gv], colnorm=True)])
        amale = np.array([R.GMAP.get(p) == "m" for p in rel.ids]); keep = lambda vm: (amale[None, :] == vm[:, None])
        tr = fold["v4tr"]; rr = np.random.RandomState(0); rf_, rv_ = tr[rr.randint(len(tr), size=R.N_REF)], tr[rr.randint(len(tr), size=R.N_REF)]
        zg = R.Std(rel.score([F4t[rf_]], [V4t[rv_]], R.V4S[rf_], R.V4S[rv_], keep=keep(R.VGEN4[rv_])))
        deep, cca, relg = [], [], []
        for (data, prot), d in files.items():
            f, v = (F4t if is4(data) else FEt)[d["fi"]], (V4t if is4(data) else VEt)[d["vi"]]
            St = sc.streams(dict(d, vgg=f, btc=v)); deep.append(St["deep0"]); cca.append(St["cca"])
            relg.append(zg(rel.score([f], [v], keep=keep((R.VGEN4 if is4(data) else R.VGENE)[d["vi"]]))))
        D[f"N{N}_t3_deep"], D[f"N{N}_t3_cca"], D[f"N{N}_t3_relg"] = np.concatenate(deep), np.concatenate(cca), np.concatenate(relg)
        R.log(f"split {s} N{N} T3: {len(set(TS))} identities ({N} v4 + VoxCeleb2), {len(TS)} rows, {ep * bpe} steps [{time.time() - t0:.0f}s]")
        D.to_parquet(R.SP / f"out/safe04/{TAG}_fold{s}.parquet")
    R.log(f"split {s} done [{time.time() - t0:.0f}s]")


def report():
    import glob
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    os.environ["SAFE04_NOMAIN"] = "1"; sys.path.insert(0, str(EXP / "SAFE-04"))
    from lab_fuse import SP, eer
    D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SP / f"out/safe04/{TAG}_fold*.parquet")))], ignore_index=True)
    D["cell"] = D.data + " " + D.protocol
    CELLS = ["en_ext g", "v4 g", "en_ext ng", "v4 ng"]
    tags = [f"N{N}" for N in NS] + [f"N{NS[-1]}x5"]
    rows = []
    for s, g in D.groupby("split"):
        fit, ev = g[g.data == "fit_en"], g[g.data != "fit_en"]
        for tag in tags:
            t3tag = tag.replace("x5", "")
            base = [f"{tag}_s007_deep0", f"{tag}_s007_cca", f"{tag}_s010B_deep0", f"{tag}_s010B_cca", f"{tag}_s007_ib", f"{tag}_s007_age"]
            t3 = [f"{t3tag}_t3_deep", f"{t3tag}_t3_cca", f"{t3tag}_t3_relg"]
            if base[0] not in g:
                continue
            sc = {}
            for name, st in (("base", base), ("base_t3", base + t3)):              # F1: fitted on the fusion-fit persons only
                z = StandardScaler().fit(fit[st].values)
                lr = LogisticRegression(C=1.0, max_iter=5000).fit(z.transform(fit[st].values), fit.y.values)
                sc[f"F1_{name}"] = lr.decision_function(z.transform(ev[st].values))
            b0 = 0.5 * (ev[f"{tag}_s007_prod0"].values + ev[f"{tag}_s010B_prod0"].values)   # F0: fixed level-0 weights
            sc["F0_base"], sc["F0_base_t3"] = b0, b0 + 0.75 * ev[t3[0]].values + 0.25 * ev[t3[1]].values
            for cell in CELLS:
                m = (ev.cell == cell).values
                r = dict(split=s, tag=tag, cell=cell)
                for k, v in sc.items():
                    r[k] = eer(v[m], ev.y.values[m])
                rows.append(r)
    T = pd.DataFrame(rows)
    for F in ("F1", "F0"):
        T[f"{F}_dEER"] = T[f"{F}_base"] - T[f"{F}_base_t3"]
    pd.set_option("display.width", 200)
    for F in ("F1", "F0"):
        print(f"\n== {F} ({'fitted on the fusion-fit persons (primary)' if F == 'F1' else 'fixed level-0 weights (robustness)'}): mean (sd) over 5 splits")
        g = T.groupby(["cell", "tag"])[[f"{F}_base", f"{F}_base_t3", f"{F}_dEER"]].agg(["mean", "std"]).round(2)
        print(g.to_string())
    # decision quantities (F1; mean of the two deciding cells en_ext g and v4 g), per split
    dec = T[T.cell.isin(["en_ext g", "v4 g"])].groupby(["split", "tag"])[["F1_base", "F1_dEER", "F0_base", "F0_dEER"]].mean().reset_index()
    pv = lambda col, tag: dec[dec.tag == tag].set_index("split")[col]
    if not {"N20", "N50"} <= set(dec.tag):
        print(f"\n(decision needs N20 and N50; present: {sorted(set(dec.tag))})"); return
    b20, b50, d20, d50 = pv("F1_base", "N20"), pv("F1_base", "N50"), pv("F1_dEER", "N20"), pv("F1_dEER", "N50")
    from scipy import stats
    def ci(x):
        x = np.asarray(x, float); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return f"{x.mean():+.2f} [{x.mean() - h:+.2f}, {x.mean() + h:+.2f}]"
    slopes = []
    for s, g in dec.groupby("split"):
        g = g[g.tag.isin([f"N{N}" for N in NS])].copy(); g["N"] = g.tag.str[1:].astype(int)
        slopes.append(np.polyfit(g.N, g.F1_dEER, 1)[0] * 10)
    print("\n== decision (F1, mean of en_ext g and v4 g; 95 % t-interval over splits)")
    print(f"  base EER change 20 -> 50 persons: {ci(b50 - b20)}  (registered: base strengthens if <= -1.0)")
    print(f"  dEER(T3) at N20 {ci(d20)} | at N50 {ci(d50)} | change {ci(d50 - d20)}; splits with change < 0: {int((d50 - d20 < 0).sum())}/5")
    print(f"  slope of dEER per +10 persons: {np.mean(slopes):+.2f} (sd {np.std(slopes, ddof=1):.2f})")
    e3, e5 = pv("F1_base", "N50"), pv("F1_base", "N50x5")
    print(f"  ensemble at N50: base EER 5 nets - 3 nets {ci(e5 - e3)}")
    sup = (b50 - b20).mean() <= -1.0 and (d50 - d20).mean() <= -1.0 and (d50 - d20 < 0).sum() >= 4
    flat = abs((d50 - d20).mean()) < 0.5 and (b50 - b20).mean() <= -1.0
    nob = (b50 - b20).mean() > -0.5
    print("  -> " + ("SUPPORTS the hypothesis" if sup else "base does NOT strengthen: look elsewhere" if nob else "does NOT support (flat T3 gain)" if flat else "inconclusive")
          + "  (no claim about N = 70: extrapolation)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            run(s)
