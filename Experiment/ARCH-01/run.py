"""ARCH-01 — does the FORM of the bridge matter at the level the system is scored (cluster / person)?  TRUE labels.

Pre-registered arms (v4 train split only, 40 ep, InfoNCE tau .07, AdamW 1e-3 / wd 1e-2, bs 256, n=2 models):
  MLP      EXP-007 recipe (2-branch MLP 512, emb 128, drop .3) = the production bridge
  LIN128   linear two-tower, emb 128, drop .3  (= low-rank bilinear score f^T U V^T v, rank 128)
  LIN16    linear two-tower, emb 16            (rank 16)
  CCA<k>   ridge CCA alone (no training; reg 10, PCA 256 face), k in {8, 16, 32, 64}; CCA16 = the base of RCCA16
  RCCA16   z = CCA_16(x) + a * R(x): CCA fixed (fit on the split's train rows, variates standardised),
           R = (256 | 192) -> 64 -> 16 MLP (drop .3), a learnable per branch, init 0.1
  RCCA32   same with k = 32
Every arm is scored alone and fused with the production CCA-4 (w .25), as in dev_scores.
Scoring = production: full face x voice matrix (mean over models of z-scored cosines [+ CCA]), then
  sample  : M[f, v]
  cluster : 0.01 M[f, v] + 0.99 * block mean over (face cluster x voice cluster); v4 held-out 30 speakers, ArcFace /
            own ECAPA-192 clusters with thresholds calibrated on the split's train speakers
  person  : the same with TRUE identities as clusters; held-out v1 (Urdu) / v2 (Hindi) persons = unseen languages
Held-out sets are identical for every arm of a split (paired).  Results appended to results.csv (resumable)."""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); torch.set_num_threads(int(os.environ.get("ARCH_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_ext")
import flag_v2 as V
from flag_lib import RidgeCCA

SPLITS = [int(x) for x in os.environ.get("ARCH_SPLITS", "1,2,3,4,5").split(",")]
N_MODELS = int(os.environ.get("ARCH_MODELS", "2"))
EPOCHS = int(os.environ.get("ARCH_EPOCHS", "40"))
ARMS = os.environ.get("ARCH_ARMS", "MLP,LIN128,LIN16,CCA8,CCA16,CCA32,CCA64,RCCA16,RCCA32").split(",")
OUT = Path(__file__).parent / os.environ.get("ARCH_OUT", "results.csv")

store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=EPOCHS)
ARM_CFG = {"MLP": REC, "LIN128": dict(REC, head="linear"), "LIN16": dict(REC, head="linear", emb=16)}
cent = lambda Z: V.l2n(V.l2n(Z) - V.l2n(Z).mean(0))
clus = lambda Z, t: AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)
calib = lambda Z, lab: max(np.arange(.4, 1.11, .05), key=lambda t: adjusted_rand_score(lab, clus(Z, t)))
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
AF, AV = store.face("arcface", "train"), store.voice("ecapa192", "train")

# ---- external held-out files (same construction as EXT-03)
EXT = {}
for src, lang in [("v1_complete", "urdu"), ("v2_complete", "hindi")]:
    D = store.ext_source(src)
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    persons = np.array(sorted({key(x) for x in D["vm"].spk}))
    EXT[src] = dict(D=D, key=key, lang=lang, folds=np.array_split(np.random.RandomState(0).permutation(persons), 5))
    store.gmap.update(D["gen_of"])


def ext_eval_file(src, held, rng):
    E = EXT[src]; D, key = E["D"], E["key"]; vm, fm = D["vm"], D["fm"]
    rows = vm[(vm.lang == E["lang"]) & vm.spk.map(lambda x: key(x) in held)]
    pick = []
    for p, g in rows.groupby(rows.spk.map(key)):
        pick += list(rng.choice(g.index, min(len(g), 15), replace=False))
    rows = vm.loc[sorted(pick)]; fi = []
    for r in rows.itertuples():
        cand = fm[(fm.spk.map(key) == key(r.spk)) & (fm.video != r.video)].index
        fi.append(rng.choice(cand if len(cand) else fm[fm.spk.map(key) == key(r.spk)].index))
    return D["Fa"][fi], D["V"][rows.index.values], np.array([key(x) for x in rows.spk])


# ---- residual CCA
class RCCA(nn.Module):
    def __init__(self, d_f, d_v, k, hid=64, drop=0.3, a0=0.1):
        super().__init__()
        mk = lambda d: nn.Sequential(nn.Dropout(drop), nn.Linear(d, hid), nn.BatchNorm1d(hid), nn.ReLU(),
                                     nn.Dropout(drop), nn.Linear(hid, k))
        self.rf, self.rv = mk(d_f), mk(d_v)
        self.af, self.av = nn.Parameter(torch.tensor(a0)), nn.Parameter(torch.tensor(a0))

    def forward(self, f, v, cf, cv):
        return F.normalize(cf + self.af * self.rf(f), dim=1), F.normalize(cv + self.av * self.rv(v), dim=1)


class CCABase:
    """Ridge CCA fitted on training rows; raw canonical variates standardised with training stats.
    Eval sets are centred first (their own mean removed, training mean restored), like cca_proj."""
    def __init__(self, TF, TV, k):
        self.m = RidgeCCA(k=k, reg=10.0, pca_x=256, pca_y=192).fit(TF, TV)
        self.mf, self.mv = TF.mean(0), TV.mean(0)
        a, b = self._raw(TF, TV)
        self.sa, self.sb = (a.mean(0), a.std(0) + 1e-6), (b.mean(0), b.std(0) + 1e-6)

    def _raw(self, X, Y):
        m = self.m
        return ((((X - m.mx) / m.sx) @ m.Px) @ m.Wx).astype(np.float32), ((((Y - m.my) / m.sy) @ m.Py) @ m.Wy).astype(np.float32)

    def __call__(self, X, Y, centre=True):
        if centre:
            X, Y = V.centre(X, self.mf), V.centre(Y, self.mv)
        a, b = self._raw(X, Y)
        return (a - self.sa[0]) / self.sa[1], (b - self.sb[0]) / self.sb[1]


def train_rcca(TF, TV, TS, k, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    prep = V.Prep(TF, TV, 256, None); base = CCABase(TF, TV, k)
    cf, cv = base(TF, TV, centre=False)
    Ft, Vt = torch.tensor(prep.f(TF)), torch.tensor(prep.v(TV))
    Ct, Dt = torch.tensor(cf), torch.tensor(cv)
    cls = {s: i for i, s in enumerate(np.unique(TS))}; Y = torch.tensor([cls[s] for s in TS])
    net = RCCA(Ft.shape[1], Vt.shape[1], k)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n, bs = len(TS), 256
    for ep in range(EPOCHS):
        net.train(); order = rng.permutation(n)
        for s in range(0, max(1, n - bs // 2), bs):
            b = torch.tensor(order[s:s + bs])
            u, w = net(Ft[b], Vt[b], Ct[b], Dt[b])
            loss = V.infonce(u, w, Y[b], 0.07)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    mu_f, mu_v = prep.f(TF).mean(0), prep.v(TV).mean(0)

    @torch.no_grad()
    def emb(Af, Av):
        a, b = base(Af, Av)
        u, w = net(torch.tensor(V.centre(prep.f(Af), mu_f)), torch.tensor(V.centre(prep.v(Av), mu_v)),
                   torch.tensor(a), torch.tensor(b))
        return u.numpy(), w.numpy()
    return emb, (net.af.item(), net.av.item())


def cca_emb(TF, TV, k):
    base = CCABase(TF, TV, k)
    return lambda Af, Av: tuple(V.l2n(x) for x in base(Af, Av))


def block_scores(M, fi, vj, cf, cv, b=0.99):
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * M[fi, vj] + b * (Af @ M @ Av.T)[cf[fi], cv[vj]]


done = set()
if OUT.exists():
    prev = pd.read_csv(OUT); done = set(zip(prev.split, prev.arm))
for s in SPLITS:
    tr, va = V.speaker_split(store.spk, s, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]; sub = tri[::3]
    cF = clus(cent(AF[vai]), calib(cent(AF[sub]), store.spk[sub]))
    cV = clus(cent(AV[vai]), calib(cent(AV[sub]), store.spk[sub]))
    T4 = {sg: V.build_trials(store.spk[vai], store.gmap, seed=s, same_gender=sg) for sg in (False, True)}
    rng_e = np.random.RandomState(100 + s)
    files = {"v4": (store.Xf[vai], store.Xv[vai], None)}
    for src in EXT:
        files[src] = ext_eval_file(src, set(EXT[src]["folds"][s - 1]), rng_e)
    Tx = {}
    for src in EXT:
        px = files[src][2]; gm = {p: store.gmap.get(p, "u") for p in px}
        protos = [False, True] if all(g in "mf" for g in gm.values()) else [False]
        Tx[src] = {sg: V.build_trials(px, gm, seed=s, same_gender=sg) for sg in protos}
    TF, TV, TS = store.Xf[tri], store.Xv[tri], store.spk[tri]
    C4 = {f: V.cca_proj(TF, TV, Fx, Vx) for f, (Fx, Vx, _) in files.items()}
    for arm in ARMS:
        if (s, arm) in done:
            print("skip", s, arm, flush=True); continue
        t0 = time.time(); embs = {f: [] for f in files}; extra = ""
        for i in range(N_MODELS if not arm.startswith("CCA") else 1):
            seed = s + 100 * i
            if arm in ARM_CFG:
                net, prep = V.train_one(store, ARM_CFG[arm], None, seed=seed, rows=(TF, TV, TS))
                mu_f, mu_v = prep.f(TF).mean(0), prep.v(TV).mean(0)
                fn = lambda Af, Av: V.embed(net, prep, Af, Av, mu_f, mu_v)
            elif arm.startswith("RCCA"):
                fn, a = train_rcca(TF, TV, TS, int(arm[4:]), seed); extra += f" a={a[0]:.3f}/{a[1]:.3f}"
            else:
                fn = cca_emb(TF, TV, int(arm[3:]))
            for f, (Fx, Vx, _) in files.items():
                embs[f].append(fn(Fx, Vx))
        rec = []
        for f in files:
            M0 = np.mean([zm(Ef @ Ev.T) for Ef, Ev in embs[f]], 0)
            for fus, M in (("alone", M0), ("+cca4", 0.75 * M0 + 0.25 * zm(C4[f][0] @ C4[f][1].T))):
                if f == "v4":
                    for sg, (fi, vj, lab) in T4.items():
                        p = "g" if sg else "ng"
                        rec += [(fus, "v4_en", p, "sample", V.eer_from_scores(M[fi, vj], lab)),
                                (fus, "v4_en", p, "cluster", V.eer_from_scores(block_scores(M, fi, vj, cF, cV), lab))]
                else:
                    ids = pd.factorize(files[f][2])[0]; tgt = EXT[f]["lang"]
                    for sg, (fi, vj, lab) in Tx[f].items():
                        p = "g" if sg else "ng"
                        rec += [(fus, tgt, p, "sample", V.eer_from_scores(M[fi, vj], lab)),
                                (fus, tgt, p, "person", V.eer_from_scores(block_scores(M, fi, vj, ids, ids), lab))]
        df = pd.DataFrame(rec, columns=["fusion", "target", "protocol", "level", "eer"])
        df.insert(0, "arm", arm); df.insert(0, "split", s); df["sec"] = round(time.time() - t0)
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"split {s} {arm}{extra} {time.time() - t0:.0f}s | " +
              " ".join(f"{fu[0]}{t}/{p}/{l[0]} {e:.2f}" for fu, t, p, l, e in rec if l != "sample"), flush=True)
print("ALL DONE", flush=True)
