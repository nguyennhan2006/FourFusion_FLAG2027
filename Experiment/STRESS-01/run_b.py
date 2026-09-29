"""STRESS-01 part B (docs/PLAN_EVAL.md E4): the same five scorings as part A (run.py), on persons the production members
never saw and in other languages. TRUE labels.

Members: s007 x 5 trained on ALL 70 v4 persons, i.e. the production nets cached by BEST/v6_eval (_members/s007_seed*.pt).
Test persons: MAV-Celeb v1 (Urdu + English) and v2 (Hindi + English), minus the two that overlap v4 (SET-07). Faces of a
positive come from ANOTHER video than its voice. Needs, for the v1 / v2 rows (ext_<src>_*_meta.csv of feats_ext_full):
  feats_ext_full   ext_<src>_face.npy (VGG 4096), ext_<src>_voice.npy (organiser ECAPA-192)       (FLAG_08)
  feats_models     ext_<src>_face_ibv / _vitageprob, ext_<src>_voice_iba / _w2v2agage / _w2v2aggen   (FLAG_10)
  feats_eval       ext_<src>_face_arcface, ext_<src>_voice_ecapa192                                  (FLAG_12)
Scenarios (name: language of the voices, persons per file, samples per person):
  urdu / hindi / english   30 persons, m = 16   (clustering thresholds were calibrated on English v4 persons)
  many60 / many100 / many150   60 / 100 / 150 persons (v1 + v2, English voices), m = 8   (more near neighbours)
  urdu_m2 / hindi_m2        30 persons, m = 2
Protocol g uses the known genders of v1 and, for v2, the majority w2v2 gender call of the person's voices.
    python run_b.py [--feats-eval <dir with FLAG_12 output>]   -> results_b.csv
"""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "kaggle")); sys.path.insert(0, str(R / "BEST/v6_eval"))
ap = argparse.ArgumentParser()
ap.add_argument("--feats-eval", default=str(R / "kaggle/output/feats_eval"))
ap.add_argument("--members", default=str(R / "BEST/v6_eval/_members"))
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("STRESS_THREADS", "10")))
os.environ["FLAG_FEATS_EXTRA"] = os.pathsep.join([str(R / "kaggle/output/feats_models"), str(R / "kaggle/output/feats_ext_full")])
import flag_v2 as V
from flag_lib import RidgeCCA
import pipeline as P6

FE, FM, FX = Path(a.feats_eval), R / "kaggle/output/feats_models", R / "kaggle/output/feats_ext_full"
need = [FE / f"ext_{s}_{k}.npy" for s in ("v1_complete", "v2_complete") for k in ("face_arcface", "voice_ecapa192")]
if not all(f.exists() for f in need):
    sys.exit(f"missing FLAG_12 arrays for v1/v2 (run kaggle/FLAG_12_features with MAV-Celeb v1/v2 + flag2027-ext-meta): "
             f"{[f.name for f in need if not f.exists()]}")
OUT = Path(__file__).parent / "results_b.csv"
store = V.FeatureStore(R / "kaggle_upload", feats_dir=R / "kaggle/output/feats_v2")
W, CW, zm, pair = P6.CONFIG["s007_recipe"], P6.CONFIG["cca_w"], P6.zm, P6.pair
OVERLAP = {"v1_complete:shaukat_tarin", "v2_complete:id0073"}                  # SET-07/overlap.csv

# ---------------------------------------------------------------- the v1 / v2 rows, one table per modality
F_rows, V_rows = [], []
for src in ("v1_complete", "v2_complete"):
    D = store.ext_source(src)
    key = lambda x, D=D, src=src: D["key_of"].get(x, f"{src}:{x}")
    fm, vm = pd.read_csv(FX / f"ext_{src}_face_meta.csv"), pd.read_csv(FX / f"ext_{src}_voice_meta.csv")
    ld = lambda d, n: np.load(d / f"ext_{src}_{n}.npy").astype(np.float32)
    fm = fm.assign(p=fm.spk.map(key), src=src, row=np.arange(len(fm)))
    vm = vm.assign(p=vm.spk.map(key), src=src, row=np.arange(len(vm)))
    F_rows.append((fm, dict(vgg=ld(FX, "face"), ibv=ld(FM, "face_ibv"), age=ld(FM, "face_vitageprob") @ P6.AGE_MIDS,
                            arc=ld(FE, "face_arcface"))))
    gen = ld(FM, "voice_w2v2aggen")
    V_rows.append((vm, dict(btc=ld(FX, "voice"), iba=ld(FM, "voice_iba"), age=ld(FM, "voice_w2v2agage")[:, 0] * 100,
                            ecapa=ld(FE, "voice_ecapa192"), male=gen[:, 1] > gen[:, 0])))
    for x, n in [(fm, "face"), (vm, "voice")]:
        assert len(x) == len(F_rows[-1][1]["vgg"] if n == "face" else V_rows[-1][1]["btc"]), (src, n)
FMETA = pd.concat([m for m, _ in F_rows], ignore_index=True)
VMETA = pd.concat([m for m, _ in V_rows], ignore_index=True)
FA = {k: np.concatenate([d[k] for _, d in F_rows]) for k in F_rows[0][1]}
VA = {k: np.concatenate([d[k] for _, d in V_rows]) for k in V_rows[0][1]}
GEN = {}
for src in ("v1_complete", "v2_complete"):
    GEN.update(store.ext_source(src)["gen_of"])
for p, g in VMETA.assign(male=VA["male"]).groupby("p").male.mean().items():
    GEN.setdefault(p, "m" if g >= .5 else "f")
FMETA = FMETA[~FMETA.p.isin(OVERLAP)]; VMETA = VMETA[~VMETA.p.isin(OVERLAP)]

# ---------------------------------------------------------------- production member s007 (70 v4 persons) + constants
nets = [torch.load(Path(a.members) / f"s007_seed{1 + 100 * i}.pt", weights_only=False) for i in range(5)]
TF, TV = store.Xf, store.Xv
cca = RidgeCCA(**V.CCA_CFG).fit(TF, TV)


def emb0(net, prep, F_, V_):
    with torch.no_grad():
        u, w = net(torch.tensor(prep.f(F_), device=V.DEVICE), torch.tensor(prep.v(V_), device=V.DEVICE))
    return u.cpu().numpy(), w.cpu().numpy()


rng0 = np.random.RandomState(0); n = len(store.spk)
ref = (rng0.randint(n, size=20000), rng0.randint(n, size=20000))
stds, parts = [], []
for net, prep in nets:
    r = pair(*emb0(net, prep, TF[ref[0]], TV[ref[1]])); stds.append(P6.Std(r)); parts.append(stds[-1](r))
cref = pair(cca.transform_x(TF[ref[0]]), cca.transform_y(TV[ref[1]])); cstd = P6.Std(cref)
s_ref = (1 - CW) * np.mean(parts, 0) + CW * cstd(cref)
tr_ibf = np.load(FM / "face_ibv_train.npy").astype(np.float32); tr_iba = np.load(FM / "voice_iba_train.npy").astype(np.float32)
mf, mv = tr_ibf.mean(0), tr_iba.mean(0)
tr_agef = np.load(FM / "face_vitageprob_train.npy") @ P6.AGE_MIDS
tr_agev = np.load(FM / "voice_w2v2agage_train.npy")[:, 0] * 100
ib_ref = pair(V.l2n(tr_ibf[ref[0]] - mf), V.l2n(tr_iba[ref[1]] - mv)); age_ref = -np.abs(tr_agef[ref[0]] - tr_agev[ref[1]])
z7, zib, zag = P6.Std(s_ref), P6.Std(ib_ref), P6.Std(age_ref)
zmix = P6.Std((1 - W["w_ib"]) * z7(s_ref) + W["w_ib"] * zib(ib_ref))
sub = np.arange(n)[::3]
calib = lambda Z: float(max(np.arange(.4, 1.21, .05), key=lambda t: P6.adjusted_rand_score(store.spk[sub], P6.clus(Z[sub], t))))
tf = calib(P6.cent(store.face("arcface", "train"), V.l2n)); tv = calib(P6.cent(store.voice("ecapa192", "train"), V.l2n))
print(f"thresholds (v4 train): face {tf:.2f} voice {tv:.2f}", flush=True)

SCEN = {"urdu": ("urdu", 30, 16), "hindi": ("hindi", 30, 16), "english": ("english", 30, 16),
        "many60": ("english", 60, 8), "many100": ("english", 100, 8), "many150": ("english", 150, 8),
        "urdu_m2": ("urdu", 30, 2), "hindi_m2": ("hindi", 30, 2)}


def make_file(lang, P, m, sg, rng):
    """Voices in `lang`; each positive's face from another video. -> face rows, voice rows (into FA / VA), labels."""
    vm = VMETA[VMETA.lang == lang]
    cand = [p for p, g in vm.groupby("p") if len(g) >= 2 and (FMETA.p == p).any()]
    Fi, Vi, Y, nf, nv = [], [], [], [], []
    for p in rng.permutation(cand)[:P]:
        vs = rng.permutation(vm.index[vm.p == p].values)[:m]
        fr = FMETA[FMETA.p == p]
        h = len(vs) // 2 if len(vs) > 1 else int(rng.rand() < .5)
        for j, v in enumerate(vs):
            other = fr[fr.video != VMETA.video[v]]
            f = rng.choice((other if len(other) else fr).index.values)
            if j < h and len(other):
                Fi.append(f); Vi.append(v); Y.append(1)
            else:
                nf.append(f); nv.append(v)
    nv = list(rng.permutation(nv))
    for f in rng.permutation(nf):
        pf = FMETA.p[f]
        ok = [j for j, v in enumerate(nv) if VMETA.p[v] != pf and (not sg or GEN.get(VMETA.p[v]) == GEN.get(pf))]
        if ok:
            Fi.append(f); Vi.append(nv.pop(ok[0])); Y.append(0)
    return np.array(Fi), np.array(Vi), np.array(Y)


recs = []
for name, (lang, P, m) in SCEN.items():
    for sg in (False, True):
        for rep in range(a.reps):
            t0 = time.time()
            rng = np.random.RandomState(1000 * list(SCEN).index(name) + 10 * rep + sg)
            Fi, Vi, y = make_file(lang, P, m, sg, rng)
            if len(y) < 20 or y.min() == y.max():
                continue
            Af, Av = FA["vgg"][Fi], VA["btc"][Vi]
            # L0: every pair alone
            deep = np.mean([sd(pair(*emb0(net, prep, Af, Av))) for (net, prep), sd in zip(nets, stds)], 0)
            s0 = (1 - CW) * deep + CW * cstd(pair(cca.transform_x(Af), cca.transform_y(Av)))
            L0 = zmix((1 - W["w_ib"]) * z7(s0) + W["w_ib"] * zib(pair(V.l2n(FA["ibv"][Fi] - mf), V.l2n(VA["iba"][Vi] - mv)))) \
                + W["w_age"] * zag(-np.abs(FA["age"][Fi] - VA["age"][Vi]))
            # L1 / L2: the SET-13 member matrix of this file
            embs = [V.embed(net, prep, Af, Av, prep.f(TF).mean(0), prep.v(TV).mean(0)) for net, prep in nets]
            M = np.mean([zm(u @ w.T) for u, w in embs], 0)
            cx, cy = V.cca_proj(TF, TV, Af, Av)
            M = (1 - CW) * M + CW * zm(cx @ cy.T)
            Fb, Ab = FA["ibv"][Fi], VA["iba"][Vi]
            ib = V.l2n(Fb - Fb.mean(0)) @ V.l2n(Ab - Ab.mean(0)).T
            S1 = zm((1 - W["w_ib"]) * zm(M) + W["w_ib"] * zm(ib)) + W["w_age"] * zm(-np.abs(FA["age"][Fi][:, None] - VA["age"][Vi][None, :]))
            cf = P6.clus(P6.cent(FA["arc"][Fi], V.l2n), tf)
            Zv = P6.cent(VA["ecapa"][Vi], V.l2n); cv = P6.clus(Zv, tv)
            ta = min(np.arange(.4, 1.21, .05), key=lambda t: (abs(P6.clus(Zv, t).max() - cf.max()), abs(t - tv)))
            cva = P6.clus(Zv, ta)
            pf, pv = FMETA.p.loc[Fi].values, VMETA.p.loc[Vi].values
            arms = {"L0": L0, "L1": np.diag(S1), "L2": P6.block(S1, cf, cv, .99), "L2a": P6.block(S1, cf, cva, .99),
                    "L2o": P6.block(S1, pd.factorize(pf)[0], pd.factorize(pv)[0], .99)}
            d = P6.diagnose(cf, cv); auto, why = P6.decide(d)
            arms["AUTO"] = arms["L2"] if auto == "set13" else arms["L0"]
            base = dict(scenario=name, lang=lang, protocol="g" if sg else "ng", rep=rep, n=len(y), pos_rate=round(y.mean(), 3),
                        persons=len(set(pf)), ari_face=round(P6.adjusted_rand_score(pf, cf), 3),
                        ari_voice=round(P6.adjusted_rand_score(pv, cv), 3), thr_voice_adaptive=round(float(ta), 2), auto=auto, **d)
            for arm, sc in arms.items():
                recs.append(dict(base, arm=arm, eer=V.eer_from_scores(sc, y)))
            pd.DataFrame(recs).to_csv(OUT, index=False)
    print(f"{name} done", flush=True)

T = pd.DataFrame(recs)
piv = T.pivot_table(index=["scenario", "protocol"], columns="arm", values="eer").round(2)
piv["L2-L0"] = (piv["L2"] - piv["L0"]).round(2)
piv = piv.reindex(sorted(piv.index, key=lambda k: (list(SCEN).index(k[0]), k[1])))
print("\nMEAN EER over reps\n" + piv[["L0", "L1", "L2", "L2a", "L2o", "AUTO", "L2-L0"]].to_string())
cols = ["persons", "ari_face", "ari_voice", "face_clusters", "voice_clusters", "voice_face_ratio", "largest_share",
        "face_median_size", "voice_median_size"]
print("\n" + T[T.arm == "L2"].groupby("scenario")[cols].mean().round(2).reindex(list(SCEN)).to_string())
print("ALL DONE", flush=True)
