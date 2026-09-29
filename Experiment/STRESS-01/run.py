"""STRESS-01 part A (docs/PLAN_EVAL.md E2): when does test-time clustering (SET-13) stop paying, or start hurting, compared
with scoring every pair alone (INDEP-01)? TRUE labels, v4 held-out persons (English), 5 splits x 40 train / 30 held out.

The member is s007' (the part that differs between the two systems): s007 (EXP-007 MLP x N_NETS + 0.25 CCA-4, trained
on the split's 40 persons) + ImageBind + age, as in SET-13 / INDEP-01. Each simulated test file is scored five ways:
  L0    INDEP-01: every pair alone, constants from 20 000 random pairs of the 40 train persons
  L1    SET-13 without clusters: file-centred member + ImageBind, z over the file's full face x voice matrix
  L2    SET-13: L1 + block mean over estimated clusters (ArcFace faces / ECAPA-192 voices, thresholds calibrated on the
        40 train persons), b = 0.99
  L2a   L2 with an ADAPTIVE voice threshold: the one whose number of voice clusters is closest to the number of face
        clusters (faces cluster almost perfectly, ARI ~0.98); no labels
  L2o   L2 with the TRUE identities as clusters (ceiling)
plus the cluster diagnostics of BEST/v6_eval (decide()), so that the `auto` rules can be read off the table.

Files are built like the dev lists: every face and every voice appears in one trial, faces and voices of a person come
from DIFFERENT clips, a fraction `pos` of each person's faces is paired with its own voices, the rest with voices of
other persons (same gender in protocol g). Scenarios (name: persons per file, samples per person, positives):
  m1 .. m32   30 persons, m = 1 / 2 / 4 / 8 / 16 / 32 faces and voices each, 50 % positives
  pos20       30 persons, m = 16, 20 % positives
  dom         one person with as many samples as it has (up to 120) + 29 persons with m = 8
  p5 / p10    5 / 10 persons, m = 16
    python run.py        -> results.csv (one row per file x arm), summary printed
"""
import os, sys, time
import numpy as np, pandas as pd, torch
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); sys.path.insert(0, str(R / "BEST/v6_eval"))
torch.set_num_threads(int(os.environ.get("STRESS_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = str(R / "kaggle/output/feats_models")
import flag_v2 as V
from flag_lib import RidgeCCA
import pipeline as P6                                     # cent, clus, block, diagnose, decide, Std, CONFIG

OUT = Path(__file__).parent / os.environ.get("STRESS_OUT", "results.csv")
N_NETS, REPS = int(os.environ.get("STRESS_NETS", "3")), int(os.environ.get("STRESS_REPS", "2"))
SPLITS = [int(x) for x in os.environ.get("STRESS_SPLITS", "1,2,3,4,5").split(",")]
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
REC = dict(V.BASE, **P6.CONFIG["recipe"])
arr = lambda n: np.load(store._find(f"{n}_train.npy")).astype(np.float32)
AF, AV = store.face("arcface", "train"), store.voice("ecapa192", "train")
IBF, IBA = arr("face_ibv"), arr("voice_iba")
AGEF, AGEV = arr("face_vitageprob") @ P6.AGE_MIDS, arr("voice_w2v2agage")[:, 0] * 100
GEN = np.array([store.gmap.get(s, "u") for s in store.spk])
W = P6.CONFIG["s007_recipe"]; CW = P6.CONFIG["cca_w"]
zm, pair = P6.zm, P6.pair
SCEN = {f"m{m}": dict(P=30, m=m, pos=.5) for m in (1, 2, 4, 8, 16, 32)}
SCEN.update(pos20=dict(P=30, m=16, pos=.2), dom=dict(P=30, m=8, pos=.5, dom=120),
            p5=dict(P=5, m=16, pos=.5), p10=dict(P=10, m=16, pos=.5))
TGRID = np.arange(.4, 1.21, .05)


def make_file(rows, spk, sc, sg, rng):
    """rows: candidate row indices (held-out persons). -> face rows, voice rows, labels."""
    persons = rng.permutation(np.unique(spk[rows]))[:sc["P"]]
    if "dom" in sc:                                        # the person with the most clips dominates the file
        persons = sorted(persons, key=lambda p: -np.sum(spk[rows] == p))
    F_, V_, Y_, nf, nv = [], [], [], [], []
    for j, p in enumerate(persons):
        r = rng.permutation(rows[spk[rows] == p])
        m = min(sc["dom"] if ("dom" in sc and j == 0) else sc["m"], len(r) // 2)
        if m < 1:
            continue
        fr, vr = r[:m], r[m:2 * m]                         # faces and voices from different clips
        h = int(round(sc["pos"] * m)) if m > 1 else int(rng.rand() < sc["pos"])
        F_ += list(fr[:h]); V_ += list(vr[:h]); Y_ += [1] * h
        nf += list(fr[h:]); nv += list(vr[h:])
    nv = list(rng.permutation(nv))
    for f in rng.permutation(nf):
        ok = [j for j, v in enumerate(nv) if spk[v] != spk[f] and (not sg or GEN[v] == GEN[f])]
        if ok:
            F_.append(f); V_.append(nv.pop(ok[0])); Y_.append(0)
    return np.array(F_), np.array(V_), np.array(Y_)


def adaptive_voice(Zv, n_face, tv):
    """Voice threshold whose cluster count is closest to the face cluster count (ties: closest to the calibrated one)."""
    best = min(TGRID, key=lambda t: (abs(P6.clus(Zv, t).max() + 1 - n_face), abs(t - tv)))
    return P6.clus(Zv, best), float(best)


recs = []
for s in SPLITS:
    t0 = time.time()
    tr, va = V.speaker_split(store.spk, s, n_val=30); tri, vai = np.where(tr)[0], np.where(va)[0]; sub = tri[::3]
    calib = lambda Z: float(max(TGRID, key=lambda t: P6.adjusted_rand_score(store.spk[sub], P6.clus(Z[sub], t))))
    tf, tv = calib(P6.cent(AF, V.l2n)), calib(P6.cent(AV, V.l2n))
    TF, TV, TS = store.Xf[tri], store.Xv[tri], store.spk[tri]
    nets = [V.train_one(store, REC, None, seed=s + 100 * i, rows=(TF, TV, TS)) for i in range(N_NETS)]
    cca = RidgeCCA(**V.CCA_CFG).fit(TF, TV)

    def emb0(net, prep, F_, V_):
        with torch.no_grad():
            u, w = net(torch.tensor(prep.f(F_), device=V.DEVICE), torch.tensor(prep.v(V_), device=V.DEVICE))
        return u.cpu().numpy(), w.cpu().numpy()

    # level 0: constants from random pairs of the 40 train persons
    rng0 = np.random.RandomState(0)
    ref = (tri[rng0.randint(len(tri), size=20000)], tri[rng0.randint(len(tri), size=20000)])
    stds, parts = [], []
    for net, prep in nets:
        u, w = emb0(net, prep, store.Xf[ref[0]], store.Xv[ref[1]])
        stds.append(P6.Std(pair(u, w))); parts.append(stds[-1](pair(u, w)))
    cref = pair(cca.transform_x(store.Xf[ref[0]]), cca.transform_y(store.Xv[ref[1]])); cstd = P6.Std(cref)
    s_ref = (1 - CW) * np.mean(parts, 0) + CW * cstd(cref)
    mf, mv = IBF[tri].mean(0), IBA[tri].mean(0)
    ib_ref = pair(V.l2n(IBF[ref[0]] - mf), V.l2n(IBA[ref[1]] - mv))
    age_ref = -np.abs(AGEF[ref[0]] - AGEV[ref[1]])
    z7, zib, zag = P6.Std(s_ref), P6.Std(ib_ref), P6.Std(age_ref)
    zmix = P6.Std((1 - W["w_ib"]) * z7(s_ref) + W["w_ib"] * zib(ib_ref))

    # level-0 score of every (held-out face, held-out voice): does not depend on the file, computed once
    U = [emb0(net, prep, store.Xf[vai], store.Xv[vai]) for net, prep in nets]
    deep = np.mean([sd(u @ w.T) for (u, w), sd in zip(U, stds)], 0)
    L0 = (1 - CW) * deep + CW * cstd(cca.transform_x(store.Xf[vai]) @ cca.transform_y(store.Xv[vai]).T)
    L0 = zmix((1 - W["w_ib"]) * z7(L0) + W["w_ib"] * zib(V.l2n(IBF[vai] - mf) @ V.l2n(IBA[vai] - mv).T)) \
        + W["w_age"] * zag(-np.abs(AGEF[vai][:, None] - AGEV[vai][None, :]))
    pos_of = {r: i for i, r in enumerate(vai)}

    for name, sc in SCEN.items():
        for sg in (False, True):
            for rep in range(REPS):
                rng = np.random.RandomState(100000 * s + 1000 * list(SCEN).index(name) + 10 * rep + sg)
                Fr, Vr, y = make_file(vai, store.spk, sc, sg, rng)
                if len(y) < 20 or y.min() == y.max():
                    continue
                # level 1: the file's own centring and z (the SET-13 member matrix, one member)
                Af, Av = store.Xf[Fr], store.Xv[Vr]
                embs = [V.embed(net, prep, Af, Av, prep.f(TF).mean(0), prep.v(TV).mean(0)) for net, prep in nets]
                M = np.mean([zm(a @ b.T) for a, b in embs], 0)
                cx, cy = V.cca_proj(TF, TV, Af, Av)
                M = (1 - CW) * M + CW * zm(cx @ cy.T)
                F_, A_ = IBF[Fr], IBA[Vr]
                ib = V.l2n(F_ - F_.mean(0)) @ V.l2n(A_ - A_.mean(0)).T
                S1 = zm((1 - W["w_ib"]) * zm(M) + W["w_ib"] * zm(ib)) + W["w_age"] * zm(-np.abs(AGEF[Fr][:, None] - AGEV[Vr][None, :]))
                cf = P6.clus(P6.cent(AF[Fr], V.l2n), tf)
                Zv = P6.cent(AV[Vr], V.l2n)
                cv = P6.clus(Zv, tv)
                cva, ta = adaptive_voice(Zv, cf.max() + 1, tv)
                cfo, cvo = pd.factorize(store.spk[Fr])[0], pd.factorize(store.spk[Vr])[0]
                arms = {"L0": L0[[pos_of[f] for f in Fr], [pos_of[v] for v in Vr]], "L1": np.diag(S1),
                        "L2": P6.block(S1, cf, cv, .99), "L2a": P6.block(S1, cf, cva, .99), "L2o": P6.block(S1, cfo, cvo, .99)}
                d = P6.diagnose(cf, cv)
                auto, why = P6.decide(d)
                arms["AUTO"] = arms["L2"] if auto == "set13" else arms["L0"]
                top = np.bincount(pd.factorize(store.spk[np.r_[Fr, Vr]])[0]).max() / (2 * len(Fr))
                base = dict(split=s, scenario=name, protocol="g" if sg else "ng", rep=rep, n=len(y), pos_rate=round(y.mean(), 3),
                            persons=len(np.unique(store.spk[Fr])), true_top_share=round(float(top), 3), thr_voice=tv,
                            thr_voice_adaptive=ta, voice_clusters_adaptive=int(cva.max() + 1), auto=auto, **d)
                for arm, sco in arms.items():
                    recs.append(dict(base, arm=arm, eer=V.eer_from_scores(sco, y)))
    pd.DataFrame(recs).to_csv(OUT, index=False)
    print(f"split {s} done in {time.time() - t0:.0f}s (thresholds face {tf:.2f} voice {tv:.2f})", flush=True)

T = pd.DataFrame(recs)
piv = T.pivot_table(index=["scenario", "protocol"], columns="arm", values="eer").round(2)
piv["L2-L0"] = (piv["L2"] - piv["L0"]).round(2)
order = list(SCEN)
piv = piv.reindex(sorted(piv.index, key=lambda k: (order.index(k[0]), k[1])))
print("\nMEAN EER over splits x reps\n" + piv[["L0", "L1", "L2", "L2a", "L2o", "AUTO", "L2-L0"]].to_string())
D = T[T.arm == "L2"].merge(T[T.arm == "L0"][["split", "scenario", "protocol", "rep", "eer"]],
                           on=["split", "scenario", "protocol", "rep"], suffixes=("", "_L0"))
D["gain"] = D.eer_L0 - D.eer
print("\nfiles where clustering LOST to pair-only (L2 > L0 + 1):", int((D.gain < -1).sum()), "of", len(D))
cols = ["face_median_size", "voice_median_size", "voice_face_ratio", "largest_share", "face_clusters", "voice_clusters", "persons"]
print(D.groupby("scenario")[cols + ["gain"]].mean().round(2).reindex(order).to_string())
print("ALL DONE", flush=True)
