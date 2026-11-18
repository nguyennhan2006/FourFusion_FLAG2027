r"""VOX2-01a: does the XALL bridge get better with ~25x more identities (VoxCeleb2, FLAG_37) at FIXED architecture?

Arms (registered 06/10, before any result), all = SAFE-04's XALL bridge recipe (EXP-007 MLP on VGG fc7 + BTC ECAPA, InfoNCE,
PK 32 x 8, 3 nets), only the training identities change:
  XALL    v4 train (40) + v1/v2 pool (~120)                       = SAFE-04 XALL (must reproduce XALL_deep0: corr 1.0000)
  XV1K    XALL + 1,000 VoxCeleb2 identities (fixed random subset), 1200 steps
  XV5K    XALL + all 4,877 VoxCeleb2 identities, 1200 steps
  XV5KL   XALL + all 4,877, 3600 steps (more data may need more optimisation)
VoxCeleb2 training rows: one per voice clip, paired with a face frame of ANOTHER video of the same speaker (cycling).
Stream = deep0 (mean over nets of the z-scored bridge cosine, constants from the fold's v4 train pairs) = XALL_deep0's scale.
Registered decision (= LOSSG-01 gate, replacement mode): CA-FUSE with XALL_deep0 replaced by the arm's deep0, against the
same replacement by this run's XALL, must gain >= 1.0 EER on the mean of the g cells (>= 4/5 folds better in >= 2 g cells)
and lose no more than 0.2 in either held-out-language check. The bridge alone and the negative-pair offset are reported.
Rule status: research on labelled validation; using VoxCeleb2 in the submitted system waits for the organisers' answers
(docs/email_organizers.md, follow-up 5-6).
    FOLDS=1,2,3,4,5 python run_bridge.py ; python run_bridge.py report > report_bridge.txt
"""
import os, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
os.environ["SAFE04_NOMAIN"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Experiment/SAFE-04"))
TAG = os.environ.get("VOX_TAG", "vox2a")
V = HERE / "feats_vox2"
ARMS = {"XALL": (None, None), "XV1K": (1000, None), "XV5K": ("all", None), "XV5KL": ("all", 3600)}


def vox_rows(n_ids):
    """VGG fc7 face / BTC voice training rows of VoxCeleb2 speakers: each voice with a face of another video."""
    import pandas as pd
    fm = pd.read_csv(V / "face_meta.csv", dtype={"video": str, "utt": str})
    vm = pd.read_csv(V / "voice_meta.csv", dtype={"video": str, "utt": str})
    FV, VB = np.load(V / "face_vgg.npy").astype(np.float32), np.load(V / "voice_btc.npy").astype(np.float32)
    spk = sorted(set(fm.spk) & set(vm.spk))
    if n_ids != "all":
        spk = sorted(np.random.RandomState(0).choice(spk, n_ids, replace=False))
    keep, face_of, fvid, vvid = set(spk), fm.groupby("spk").indices, fm.video.values, vm.video.values
    Fi, Vi, S = [], [], []
    for s, idx in sorted(vm.groupby("spk").indices.items()):
        if s not in keep:
            continue
        faces = face_of[s]
        for j, v in enumerate(sorted(idx)):
            cand = [f for f in faces if fvid[f] != vvid[v]] or list(faces)
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vox2:" + s)
    return FV[Fi], VB[Vi], np.array(S)


def cov_pca(Z, k):
    """Top-k principal directions via the d x d covariance (same subspace as the SVD, for n >> d)."""
    w, U = np.linalg.eigh(Z.T.astype(np.float64) @ Z.astype(np.float64))
    return U[:, np.argsort(w)[::-1][:k]].astype(np.float32)


def patch_large(G):
    """Prep / RidgeCCA fit PCA by an SVD of the rows (minutes and several GB for ~57k x 4096). Above 30,000 rows use the
    covariance instead; below (XALL, ~19.5k rows) the original code runs, so XALL reproduces SAFE-04 exactly."""
    P0, R0 = G["Prep"]._pca, G["RidgeCCA"]._prep

    def pca(Z, k):
        if not k or k >= Z.shape[1] or len(Z) <= 30000:
            return P0(Z, k)
        return cov_pca(Z, k)

    def prep(self, X, n_pca):
        if len(X) <= 30000 or n_pca is None or n_pca >= X.shape[1]:
            return R0(self, X, n_pca)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        return mu, sd, cov_pca((X - mu) / sd, n_pca).astype(np.float64)
    G["Prep"]._pca = staticmethod(pca)
    G["RidgeCCA"]._prep = prep


def run(s):
    import pandas as pd, torch
    import rel_sweep5 as R
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))
    t0 = time.time()
    fold = R.make_fold(s); files = list(fold["files"].items())
    D = pd.concat([pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"], fi=d["fi"], vi=d["vi"])) for (data, prot), d in files],
                  ignore_index=True)
    for arm, (n_ids, steps) in ARMS.items():
        TF, TV, TS, info = R.train_rows(fold, ext=True)
        if n_ids is not None:
            F2, V2, S2 = vox_rows(n_ids)
            TF, TV, TS = np.concatenate([TF, F2]), np.concatenate([TV, V2]), np.concatenate([TS, S2])
        ep, bpe = R.epochs_for(len(TS), steps)
        units = [R.train_one(TF, TV, TS, seed=fold["s"] + 100 * i, epochs=ep) for i in range(R.N_NETS)]
        sc = R.L0(R.Member(units, TF, TV), fold["v4tr"])
        D[f"d_{arm}"] = np.concatenate([sc.streams(d)["deep0"] for (data, prot), d in files])
        R.log(f"fold {s} {arm}: {len(set(TS))} identities, {len(TS)} rows, {ep * bpe} steps [{time.time() - t0:.0f}s]")
    D.to_parquet(R.SP / f"out/safe04/{TAG}_fold{s}.parquet")
    R.log(f"fold {s} done [{time.time() - t0:.0f}s]")


def ca_replace(D, col, held=()):
    import pandas as pd
    from stream_gate import S6, _ca
    D2 = D.assign(XALL_deep0=D[col].values)
    out = pd.Series(np.nan, index=D.index)
    for s in sorted(D.fold.unique()):
        trn = D2[(D2.fold != s) & ~D2.data.isin(held)]
        te = (D2.fold == s) & (D2.data.isin(held) if held else True)
        if te.any() and len(trn):
            out[te] = _ca(trn, D2[te], S6)
    return out.values


def report():
    from stream_gate import load, _boot_gain
    from lab_fuse import table
    D = load(TAG)
    arms = [a for a in ARMS if f"d_{a}" in D]
    print(f"wiring: corr(d_XALL, SAFE-04 XALL_deep0) = {np.corrcoef(D.d_XALL, D.XALL_deep0)[0, 1]:.4f} (must be 1.0000 on this CPU)")
    cols = [f"d_{a}" for a in arms]
    T0 = table(D, {c: (lambda g, c=c: g[c].values) for c in cols})
    print("\n== bridge alone (deep0), EER mean of folds")
    print(T0.groupby("cell").mean(numeric_only=True)[cols].round(2).to_string())
    print("\n== negative-pair offset: mean bridge z on NEGATIVE pairs of new persons (ERR-03; informative only, see log 141)")
    print(D[D.y == 0].groupby("cell")[cols].mean().round(2).to_string())
    regimes = {"oof": (), "hold_urdu": ("urdu",), "hold_hindi": ("hindi",)}
    for r, held in regimes.items():
        for c in cols:
            D[f"CA[{c}]|{r}"] = ca_replace(D, c, held)
    T = table(D, {k: (lambda g, k=k: g[k].values) for k in D.columns if k.startswith("CA[")})
    gcells = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    print("\n== CA-FUSE with the bridge replaced: EER mean of folds (OOF)")
    print(T.groupby("cell").mean(numeric_only=True)[[f"CA[{c}]|oof" for c in cols]].round(2).to_string())
    print("\n== gain vs CA-FUSE[d_XALL] in the same regime (+ = better), per cell mean (folds better, folds >= 1)")
    verdict = {}
    for c in cols[1:]:
        res = {}
        for r in regimes:
            ref, col = "CA[d_XALL]|" + r, f"CA[{c}]|{r}"
            sub = T[["cell", ref, col]].dropna(); d = sub[ref] - sub[col]; g = d.groupby(sub.cell)
            m, w, w1 = g.mean(), g.apply(lambda x: (x > 0).sum()), g.apply(lambda x: (x >= 1).sum())
            gc = [k for k in m.index if k.endswith(" g")]
            res[r] = dict(mean=m.mean(), g=m[gc].mean() if gc else np.nan, g_w4=int(sum(w[k] >= 4 for k in gc)))
            print(f"  {r:>10} {c:<8}: all {m.mean():+.2f} | g {res[r]['g']:+.2f} | "
                  + " ".join(f"{k.replace(' ', '/')} {m[k]:+.2f}({w[k]},{w1[k]})" for k in m.index))
        ok = res["oof"]["g"] >= 1.0 and res["oof"]["g_w4"] >= 2 and res["hold_urdu"]["mean"] >= -0.2 and res["hold_hindi"]["mean"] >= -0.2
        lo, hi = _boot_gain(D, "CA[d_XALL]|oof", f"CA[{c}]|oof", gcells)
        verdict[c] = ok
        print(f"  oof g-cell gain {res['oof']['g']:+.2f}, 95 % person-bootstrap interval [{lo:+.2f}, {hi:+.2f}]")
        print(f"  -> {c}: {'KEEP (passes the registered gate)' if ok else 'close (fails the registered gate)'}\n")
    print("VERDICT:", {k: ("KEEP" if v else "close") for k, v in verdict.items()})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            run(s)
