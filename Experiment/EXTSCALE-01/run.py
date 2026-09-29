"""EXTSCALE-01 — does the bridge keep improving with MORE training identities, with and without SetProto, and does that
survive in the PRODUCTION member (s007 + ImageBind + age)? TRUE labels. Follows docs/deep-research-report (28/09):
EXT-VAL70 factorial + SCALE-ID, equal optimizer budget, P x K fixed, paired splits.

What EXT-03 left open: external arms got more optimizer steps; only the bare member was scored; no identity-scaling
curve; no English held-out external persons.

Part 1  EXTVAL  (5 folds of the MAV-Celeb v1 + v2 PERSONS; fold f held out, never trained on)
        arms  R   70 v4 persons, row loss         P   70 v4, + SetProto (w 0.5)
              XR  70 v4 + v1/v2 of the other 4 folds, row loss      XP  XR + SetProto
        eval  held-out fold persons: Urdu, Hindi, English voices (up to 16 per person); the face of a positive from
              ANOTHER video; ng (+ g where genders are known or predicted)
Part 2  SCALE   (5 v4 splits: 30 persons held out; nested train subsets of the other 40)
        points v4 10 / 20 / 30 / 40, then 40 + 25 / 50 / 100 % of the v1/v2 persons of the 4 folds not held out
        arms   row loss / SetProto at every point
        eval   the 30 held-out v4 persons (16 faces + 16 voices each, different rows) AND the held-out v1/v2 fold
Every arm: sampler P x K (Part 1: 32 x 8, Part 2: 8 x 8, fixed along the curve), the SAME number of optimizer steps
(STEPS1 / STEPS2; epochs = steps / batches-per-epoch), N_NETS nets (seeds s + 100 i), EXP-007 head.
Scored like SET-13 per file: file-centred member matrix (+ 0.25 CCA-4), z over the matrix;
  member  the bridge alone          prod  z(0.5 z(member) + 0.5 z(IB)) + 0.1 z(AGE)   (the s007' of SET-13)
levels: sample (diagonal) | person (true identities as blocks, b 0.99) | cluster (ArcFace / ECAPA-192 clusters at
the production thresholds 0.75 / 0.70, b 0.99).
    python run.py [part1|part2]      -> results_part1.csv / results_part2.csv (resumable per split x arm)
"""
import os, sys, time
import numpy as np, pandas as pd, torch
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); sys.path.insert(0, str(R / "BEST/v6_eval"))
torch.set_num_threads(int(os.environ.get("XS_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = os.pathsep.join([str(R / "kaggle/output/feats_models"), str(R / "kaggle/output/feats_ext_full")])
import flag_v2 as V
from flag_lib import RidgeCCA
import pipeline as P6

PARTS = [a for a in sys.argv[1:] if a.startswith("part")] or ["part1", "part2"]
N_NETS = int(os.environ.get("XS_NETS", "3"))
STEPS1, STEPS2 = int(os.environ.get("XS_STEPS1", "1200")), int(os.environ.get("XS_STEPS2", "1500"))
FOLDS = [int(x) for x in os.environ.get("XS_FOLDS", "1,2,3,4,5").split(",")]
OUT = Path(__file__).parent
FE, FM, FX = R / "kaggle/output/feats_eval", R / "kaggle/output/feats_models", R / "kaggle/output/feats_ext_full"
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, **P6.CONFIG["recipe"])
W, CW, zm = P6.CONFIG["s007_recipe"], P6.CONFIG["cca_w"], P6.zm
TF_, TV_ = 0.75, 0.70                                            # production cluster thresholds (v4 train calibration)
OVERLAP = {"v1_complete:shaukat_tarin", "v2_complete:id0073"}   # SET-07/overlap.csv

# ============================================================================ feature tables
# domain "v4": the 6485 v4 train rows (face row i and voice row i come from the same clip)
v4 = dict(vgg=store.Xf, btc=store.Xv, ibf=np.load(FM / "face_ibv_train.npy").astype(np.float32),
          iba=np.load(FM / "voice_iba_train.npy").astype(np.float32),
          agef=np.load(FM / "face_vitageprob_train.npy") @ P6.AGE_MIDS, agev=np.load(FM / "voice_w2v2agage_train.npy")[:, 0] * 100,
          arc=store.face("arcface", "train"), ecapa=store.voice("ecapa192", "train"))
V4S = store.spk.astype(str); V4G = np.array([store.gmap.get(s, "u") for s in store.spk])

# domain "ext": MAV-Celeb v1 + v2 rows of feats_ext_full, faces and voices in separate tables
FMETA, VMETA, FA, VA, SRC = [], [], {}, {}, {}
for src in ("v1_complete", "v2_complete"):
    D = store.ext_source(src)
    SRC[src] = D
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    fm, vm = pd.read_csv(FX / f"ext_{src}_face_meta.csv"), pd.read_csv(FX / f"ext_{src}_voice_meta.csv")
    assert len(fm) == len(D["Fa"]) and len(vm) == len(D["V"]), src
    FMETA.append(fm.assign(p=fm.spk.map(key), src=src)); VMETA.append(vm.assign(p=vm.spk.map(key), src=src))
    ld = lambda d, n: np.load(d / f"ext_{src}_{n}.npy").astype(np.float32)
    for k, a in dict(vgg=D["Fa"], ibf=ld(FM, "face_ibv"), agef=ld(FM, "face_vitageprob") @ P6.AGE_MIDS, arc=ld(FE, "face_arcface")).items():
        FA.setdefault(k, []).append(a)
    gen = ld(FM, "voice_w2v2aggen")
    for k, a in dict(btc=D["V"], iba=ld(FM, "voice_iba"), agev=ld(FM, "voice_w2v2agage")[:, 0] * 100,
                     ecapa=ld(FE, "voice_ecapa192"), male=(gen[:, 1] > gen[:, 0]).astype(np.float32)).items():
        VA.setdefault(k, []).append(a)
FMETA, VMETA = pd.concat(FMETA, ignore_index=True), pd.concat(VMETA, ignore_index=True)
FA = {k: np.concatenate(v) for k, v in FA.items()}; VA = {k: np.concatenate(v) for k, v in VA.items()}
GEN = {}
for D in SRC.values():
    GEN.update(D["gen_of"])
for p, g in VMETA.assign(male=VA["male"]).groupby("p").male.mean().items():
    GEN.setdefault(p, "m" if g >= .5 else "f")
PERSONS = sorted(set(VMETA.p) - OVERLAP)
PFOLD = dict(zip(PERSONS, np.random.RandomState(0).permutation(len(PERSONS)) % 5))
store.gmap.update(GEN)


def ext_train_rows(keep, rng, cap=150):
    """Organiser-space training rows (VGG face of the same video, BTC voice) of the persons in `keep`."""
    parts = []
    for src, D in SRC.items():
        key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
        ids = [x for x in D["vm"].spk.unique() if key(x) in keep]
        if ids:
            parts.append(V.pair_ext(D, src, cap, rng, ids=ids))
    if not parts:
        return None
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


# ============================================================================ evaluation files
def ext_file(persons, lang, sg, rng, m=16):
    vm = VMETA[(VMETA.lang == lang) & VMETA.p.isin(persons)]
    Fi, Vi, Y, nf, nv = [], [], [], [], []
    for p, g in vm.groupby("p"):
        fr = FMETA[FMETA.p == p]
        if len(g) < 2 or not len(fr):
            continue
        vs = rng.permutation(g.index.values)[:m]
        for j, v in enumerate(vs):
            other = fr[fr.video != VMETA.video[v]]
            if not len(other):
                continue
            f = rng.choice(other.index.values)
            if j < len(vs) // 2:
                Fi.append(f); Vi.append(v); Y.append(1)
            else:
                nf.append(f); nv.append(v)
    nv = list(rng.permutation(nv))
    for f in rng.permutation(nf):
        pf = FMETA.p[f]
        ok = [j for j, v in enumerate(nv) if VMETA.p[v] != pf and (not sg or GEN.get(VMETA.p[v]) == GEN.get(pf))]
        if ok:
            Fi.append(f); Vi.append(nv.pop(ok[0])); Y.append(0)
    Fi, Vi = np.array(Fi, int), np.array(Vi, int)
    return dict(f={k: a[Fi] for k, a in FA.items()}, v={k: a[Vi] for k, a in VA.items()}, y=np.array(Y),
                pf=FMETA.p.values[Fi], pv=VMETA.p.values[Vi])


def v4_file(rows, sg, rng, m=16):
    Fi, Vi, Y, nf, nv = [], [], [], [], []
    for p in np.unique(V4S[rows]):
        r = rng.permutation(rows[V4S[rows] == p]); k = min(m, len(r) // 2)
        fr, vr = r[:k], r[k:2 * k]; h = k // 2
        Fi += list(fr[:h]); Vi += list(vr[:h]); Y += [1] * h; nf += list(fr[h:]); nv += list(vr[h:])
    nv = list(rng.permutation(nv))
    for f in rng.permutation(nf):
        ok = [j for j, v in enumerate(nv) if V4S[v] != V4S[f] and (not sg or V4G[v] == V4G[f])]
        if ok:
            Fi.append(f); Vi.append(nv.pop(ok[0])); Y.append(0)
    Fi, Vi = np.array(Fi, int), np.array(Vi, int)
    return dict(f={"vgg": v4["vgg"][Fi], "ibf": v4["ibf"][Fi], "agef": v4["agef"][Fi], "arc": v4["arc"][Fi]},
                v={"btc": v4["btc"][Vi], "iba": v4["iba"][Vi], "agev": v4["agev"][Vi], "ecapa": v4["ecapa"][Vi]},
                y=np.array(Y), pf=V4S[Fi], pv=V4S[Vi])


def eval_files(held_ext, v4_rows, rng):
    files = {}
    for lang in ("urdu", "hindi", "english"):
        for sg in (False, True):
            if lang == "hindi" and sg:
                continue                                   # v2 genders are only predicted: ng only, as EXT-03
            files[(lang, "g" if sg else "ng")] = ext_file(held_ext, lang, sg, rng)
    if v4_rows is not None:
        for sg in (False, True):
            files[("v4", "g" if sg else "ng")] = v4_file(v4_rows, sg, rng)
    for d in files.values():                                 # clusters do not depend on the arm: once per file
        d["cf"] = P6.clus(P6.cent(d["f"]["arc"], V.l2n), TF_)
        d["cv"] = P6.clus(P6.cent(d["v"]["ecapa"], V.l2n), TV_)
    return {k: d for k, d in files.items() if len(d["y"]) >= 20 and 0 < d["y"].mean() < 1}


# ============================================================================ training + scoring
def train_arm(TF, TV, TS, proto, P, K, steps, seed0):
    bpe = max(1, len(TS) // (P * K))
    ep = max(1, int(round(steps / bpe)))
    cfg = dict(REC, sampler="pk", pk_P=P, pk_K=K, w_proto=0.5 if proto else 0.0, epochs=ep)
    nets = [V.train_one(store, cfg, None, seed=seed0 + 100 * i, rows=(TF, TV, TS)) for i in range(N_NETS)]
    cca = RidgeCCA(**V.CCA_CFG).fit(TF, TV)
    return dict(nets=nets, cca=cca, mf=TF.mean(0), mv=TV.mean(0), steps=bpe * ep, TF=TF, TV=TV)


def score(model, d):
    Af, Av = d["f"]["vgg"], d["v"]["btc"]
    embs = [V.embed(net, prep, Af, Av, prep.f(model["TF"]).mean(0), prep.v(model["TV"]).mean(0)) for net, prep in model["nets"]]
    M = np.mean([zm(a @ b.T) for a, b in embs], 0)
    cx = model["cca"].transform_x(Af - Af.mean(0) + model["mf"]); cy = model["cca"].transform_y(Av - Av.mean(0) + model["mv"])
    M = (1 - CW) * M + CW * zm(cx @ cy.T)
    Fb, Ab = d["f"]["ibf"], d["v"]["iba"]
    ib = V.l2n(Fb - Fb.mean(0)) @ V.l2n(Ab - Ab.mean(0)).T
    S = zm((1 - W["w_ib"]) * zm(M) + W["w_ib"] * zm(ib)) + W["w_age"] * zm(-np.abs(d["f"]["agef"][:, None] - d["v"]["agev"][None, :]))
    out = {}
    cfo, cvo = pd.factorize(d["pf"])[0], pd.factorize(d["pv"])[0]
    for var, X in (("member", M), ("prod", S)):
        out[(var, "sample")] = V.eer_from_scores(np.diag(X), d["y"])
        out[(var, "person")] = V.eer_from_scores(P6.block(X, cfo, cvo, .99), d["y"])
        out[(var, "cluster")] = V.eer_from_scores(P6.block(X, d["cf"], d["cv"], .99), d["y"])
    return out


def run(part):
    f_out = OUT / f"results_{part}.csv"
    done = set()
    recs = pd.read_csv(f_out).to_dict("records") if f_out.exists() else []
    for r in recs:
        done.add((r["split"], r["arm"]))
    for s in FOLDS:
        held = [p for p in PERSONS if PFOLD[p] == s - 1]
        pool = [p for p in PERSONS if PFOLD[p] != s - 1]
        rng = np.random.RandomState(1000 + s)
        if part == "part1":
            files = eval_files(held, None, np.random.RandomState(s))
            v4_idx = np.arange(len(V4S))
            points = [("R", 70, 0.0, False), ("P", 70, 0.0, True), ("XR", 70, 1.0, False), ("XP", 70, 1.0, True)]
            P, K, steps = 32, 8, STEPS1
        else:
            tr, va = V.speaker_split(store.spk, s, n_val=30)
            v4_idx = np.where(tr)[0]
            files = eval_files(held, np.where(va)[0], np.random.RandomState(s))
            points = [(f"v4_{n}{'_e' + str(int(100 * e)) if e else ''}{'_SP' if sp else ''}", n, e, sp)
                      for n, e in [(10, 0), (20, 0), (30, 0), (40, 0), (40, .25), (40, .5), (40, 1.0)] for sp in (False, True)]
            P, K, steps = 8, 8, STEPS2
        order_v4 = list(np.random.RandomState(2000 + s).permutation(np.unique(V4S[v4_idx])))     # nested prefixes
        order_ext = list(np.random.RandomState(3000 + s).permutation(pool))
        for arm, n_v4, e_frac, sp in points:
            if (s, arm) in done:
                continue
            t0 = time.time()
            keep_v4 = set(order_v4[:n_v4])
            ridx = v4_idx[np.isin(V4S[v4_idx], list(keep_v4))]
            TF, TV, TS = store.Xf[ridx], store.Xv[ridx], V4S[ridx]
            n_ext = int(round(e_frac * len(order_ext)))
            if n_ext:
                ex = ext_train_rows(set(order_ext[:n_ext]), np.random.RandomState(4000 + s))
                TF, TV, TS = np.concatenate([TF, ex[0]]), np.concatenate([TV, ex[1]]), np.concatenate([TS, ex[2]])
            model = train_arm(TF, TV, TS, sp, P, K, steps, seed0=s)
            for (data, prot), d in files.items():
                for (var, level), e in score(model, d).items():
                    recs.append(dict(split=s, arm=arm, n_v4=n_v4, ext_frac=e_frac, n_ext_persons=n_ext, setproto=sp,
                                     train_ids=len(set(TS)), train_rows=len(TS), steps=model["steps"], data=data,
                                     protocol=prot, variant=var, level=level, n=len(d["y"]), eer=e))
            pd.DataFrame(recs).to_csv(f_out, index=False)
            print(f"{part} split {s} {arm:14s} ids {len(set(TS)):4d} rows {len(TS):6d} steps {model['steps']:5d} "
                  f"{time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(recs)


if __name__ == "__main__":
  for part in PARTS:
    T = run(part)
    piv = T.pivot_table(index=["variant", "level", "data", "protocol"], columns="arm", values="eer").round(2)
    print(f"\n==== {part}: mean EER over splits\n" + piv.to_string(), flush=True)
  print("ALL DONE", flush=True)
