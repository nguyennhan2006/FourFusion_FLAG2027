"""Evaluation-phase pipeline v6 (docs/PLAN_EVAL.md, E1): SET-13, its no-cluster version and INDEP-01 from ONE set of
trained members, chosen per test file.

    python pipeline.py --data <organiser data> --feats <dir> [<dir> ...] --out <dir>
                       [--mode auto|set13|noclus|indep] [--file-mode <file>=<mode> ...] [--n-models 15]
                       [--members-from <dir>] [--reference name=zip ...]

  set13  SET-13, CodaBench 21.01 on dev (BEST/v5_set13): per-file centring and z-scores, ImageBind + age on the s007
         member, clusters (ArcFace faces, ECAPA-192 voices, thresholds calibrated on train speakers), block mean b = 0.99,
         rank fusion per cell.
  noclus SET-13 without the clusters (b = 0; = v5 --b 0): file-centred and z-scored, every trial scored from its own cell
         of the matrix. The fallback when the clusters of a file are wrong (STRESS-01: >= INDEP-01 in almost every
         scenario, also small files and files dominated by one person).
  indep  INDEP-01, CodaBench 26.70 on dev (Experiment/INDEP-01): every pair scored on its own; every constant comes from
         20 000 random train pairs; no statistic of the test file and no clusters. A pair's score does not change when
         the rest of the file changes (checked by a self-test on every run, invariance.json). For a ruling that forbids
         using other test samples.
  auto   set13 for every file. STRESS-01 part A (200 simulated files, true labels): clustering beat pair-only scoring in
         every scenario (1 to 32 samples per person, 5 to 30 persons, one dominant person); the files it lost were tiny
         (30-120 trials) and no diagnostic predicted them. A file whose voices look over-merged (far fewer voice clusters
         than face clusters, the SET-04 failure) is FLAGGED in decisions.csv, not switched: switch it by hand with
         --file-mode <file>=noclus after reading the diagnostics (STRESS-01 part B will say whether to automate it).

Members (s007 / s010B / r2mix, EXP-007 recipe, all 70 train speakers) are trained ONCE and serve both modes. Each
(member, seed) is cached in <out>/members/ (or --members-from): seeds 1, 101, 201, ...; the first 5 are exactly the
submitted members, so --n-models 5 reproduces SET-13 / INDEP-01 and --n-models 15 (log 69: same expectation, no seed
lottery) reuses them.

--feats: every folder holding arrays by name (flag2027-feats-eval from kaggle/FLAG_12_features, or feats_v2 +
feats_models): face_arcface_*, voice_ecapa192_*, voice_ecapa6144_*, face_ibv_*, voice_iba_*, face_vitageprob_*,
voice_w2v2agage_*, for train and for every test file.

Outputs in --out: submission.zip (the chosen mode per file), submission_set13.zip, submission_noclus.zip,
submission_indep.zip, decisions.csv (per file: cluster diagnostics, mode used, warning), invariance.json, config.json.
"""
import argparse, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kaggle"))

CONFIG = {
    "recipe": dict(head="mlp", drop=0.3, emb=128, epochs=40),              # EXP-007
    "members": {"s007": dict(),                                             # VGG + organiser ECAPA-192
                "s010B": dict(voice="ecapa6144"),                           # VGG + speechbrain ECAPA 6144
                "r2mix": dict(voice="ecapa192", voice_view="mix")},         # VGG + ECAPA-192, Bangla-length crops
    "seeds": "1 + 100 i",
    "cca_w": 0.25,
    "cells": {"English": ["s007", "s010B"], "Bangla": ["r2mix", "s007"]},
    "cluster_face": "arcface", "cluster_voice": "ecapa192", "b": 0.99,
    "s007_recipe": dict(w_ib=0.5, w_age=0.1),                               # SET-13 = INDEP-01 weights
    "ref_pairs": 20000,                                                     # INDEP: constants from random train pairs
    # auto = set13 everywhere (STRESS-01 A). Dropped after STRESS-01 A, both cost EER: "largest cluster > 25 % of the
    # file -> indep" (2.6 to 5.7 lost on files dominated by one person) and "median cluster size < 2 -> indep"
    # (2 to 6 lost at one sample per person). Kept as a WARNING only:
    "auto": dict(warn_voice_face_ratio=0.8),  # far fewer voice clusters than face clusters: voices of people merged?
}
AGE_MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)   # centres of the 9 FairFace age bins
zm = lambda M: (M - M.mean()) / (M.std() + 1e-9)
rank = lambda s: rankdata(s) / (len(s) + 1)
pair = lambda U, W: np.sum(U * W, 1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ clusters (as SET-02)
def cent(Z, l2n):
    return l2n(l2n(Z) - l2n(Z).mean(0))


def clus(Z, t):
    return AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=t).fit_predict(Z)


def block(M, cf, cv, b):
    M = M.astype(np.float64)
    Af = np.zeros((cf.max() + 1, len(cf))); Af[cf, np.arange(len(cf))] = 1; Af /= Af.sum(1, keepdims=True)
    Av = np.zeros((cv.max() + 1, len(cv))); Av[cv, np.arange(len(cv))] = 1; Av /= Av.sum(1, keepdims=True)
    return (1 - b) * np.diag(M) + b * (Af @ M @ Av.T)[cf, cv]


def diagnose(cf, cv):
    """Cluster diagnostics of one test file, no labels. Sizes are per SAMPLE (the size of the cluster it sits in)."""
    sf, sv = np.bincount(cf), np.bincount(cv)
    return dict(rows=len(cf), face_clusters=len(sf), voice_clusters=len(sv),
                voice_face_ratio=round(len(sv) / len(sf), 3),
                face_median_size=float(np.median(sf[cf])), voice_median_size=float(np.median(sv[cv])),
                face_singletons=int((sf == 1).sum()), voice_singletons=int((sv == 1).sum()),
                largest_share=round(float(max(sf.max(), sv.max()) / len(cf)), 3))


def decide(d):
    """-> (mode, note). Always set13 (STRESS-01 A); over-merged-looking voices are flagged for a manual decision."""
    r = CONFIG["auto"]
    if d["voice_face_ratio"] < r["warn_voice_face_ratio"]:
        return "set13", (f"WARNING voice/face clusters {d['voice_face_ratio']} < {r['warn_voice_face_ratio']}: voices of "
                         f"different people may be merged; check, and if so rerun with --file-mode <file>=noclus")
    return "set13", "clusters look normal"


# ============================================================================ members
def train_members(V, store, n, cache: Path):
    """{member: (cfg, [(net, prep), ...])}; each (member, seed) is trained once and cached as <member>_seed<s>.pt."""
    import torch
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, extra in CONFIG["members"].items():
        cfg = dict(V.BASE, **CONFIG["recipe"], **extra)
        nets = []
        for i in range(n):
            seed = 1 + 100 * i                                              # = flag_v2.dev_scores(seed0=1)
            f = cache / f"{name}_seed{seed}.pt"
            if f.exists():
                nets.append(torch.load(f, weights_only=False))
            else:
                t = time.time()
                nets.append(V.train_one(store, cfg, np.arange(len(store.spk)), seed=seed))
                torch.save(nets[-1], f)
                log(f"trained {name} seed {seed} in {time.time() - t:.0f}s")
        out[name] = (cfg, nets)
    log("members:", {k: len(v[1]) for k, v in out.items()})
    return out


class Std:
    """z-score with constants measured on the train reference pairs only."""
    def __init__(self, ref):
        self.m, self.s = float(np.mean(ref)), float(np.std(ref) + 1e-9)

    def __call__(self, x):
        return (x - self.m) / self.s


class Scorer:
    """Both scorings of one member. l1_matrix: SET-13's full face x voice matrix of a test file (file-centred, z over the
    matrix; = flag_v2.dev_scores(matrix_path=...)). pairs: INDEP-01's score of row-aligned (face, voice) pairs."""

    def __init__(self, V, store, cfg, nets, ref):
        from flag_lib import RidgeCCA
        self.V, self.store, self.cfg, self.nets = V, store, cfg, nets
        self.Xf, self.Xv = store.face(cfg["face"], "train"), store.voice(cfg["voice"], "train")
        self.cca = RidgeCCA(**V.CCA_CFG).fit(self.Xf, self.Xv)
        self.stds, parts = [], []
        for net, prep in nets:
            r = pair(*[e[i] for e, i in zip(self.embed0(net, prep, self.Xf, self.Xv), ref)])
            self.stds.append(Std(r)); parts.append(self.stds[-1](r))
        cref = pair(self.cca.transform_x(self.Xf)[ref[0]], self.cca.transform_y(self.Xv)[ref[1]])
        self.cstd = Std(cref)
        w = CONFIG["cca_w"]
        self.ref = (1 - w) * np.mean(parts, 0) + w * self.cstd(cref)       # this member's score on the reference pairs

    def embed0(self, net, prep, F_, V_):
        """The member's embeddings with NO test-set centring (train standardisation / PCA only)."""
        import torch
        with torch.no_grad():
            u, w = net(torch.tensor(prep.f(F_), device=self.V.DEVICE), torch.tensor(prep.v(V_), device=self.V.DEVICE))
        return u.cpu().numpy(), w.cpu().numpy()

    def rows(self, split):
        return self.store.face(self.cfg["face"], split), self.store.voice(self.cfg["voice"], split)

    def pairs(self, Af, Av):
        w = CONFIG["cca_w"]
        deep = np.mean([sd(pair(*self.embed0(net, prep, Af, Av))) for (net, prep), sd in zip(self.nets, self.stds)], 0)
        return (1 - w) * deep + w * self.cstd(pair(self.cca.transform_x(Af), self.cca.transform_y(Av)))

    def l1_matrix(self, split):
        V = self.V
        Af, Av = self.rows(split)
        embs = [V.embed(net, prep, Af, Av, prep.f(self.Xf).mean(0), prep.v(self.Xv).mean(0)) for net, prep in self.nets]
        M = np.mean([zm(Ef @ Ev.T) for Ef, Ev in embs], 0)
        cx, cy = V.cca_proj(self.Xf, self.Xv, Af, Av)
        M = (1 - CONFIG["cca_w"]) * M + CONFIG["cca_w"] * zm(cx @ cy.T)
        return M.astype(np.float16).astype(np.float64)                     # dev_scores stored fp16: kept for identity


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="organiser data (train/ + test files: faces/voices csv + lists)")
    ap.add_argument("--feats", required=True, nargs="+", help="folders with the named arrays (FLAG_12 output, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "set13", "noclus", "indep"])
    ap.add_argument("--file-mode", nargs="*", default=[], help="per-file override, e.g. gender/Bangla=noclus")
    ap.add_argument("--n-models", type=int, default=15)
    ap.add_argument("--members-from", help="folder with cached <member>_seed<s>.pt (default: <out>/members)")
    ap.add_argument("--reference", nargs="*", default=[], help="name=zip: rank correlation of our zip 'name' with it")
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    os.environ["FLAG_FEATS_EXTRA"] = os.pathsep.join(a.feats[1:])
    import flag_v2 as V
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(CONFIG, mode=a.mode, n_models=a.n_models, feats=a.feats), indent=1))
    store = V.FeatureStore(a.data, feats_dir=a.feats[0])
    V.polarity_preflight()
    FILES = list(V.CELLS)                                                    # [(protocol, language)]

    def arr(name, split):
        f = store._find(f"{name}_{split.replace('/', '_')}.npy")
        assert f is not None, f"{name}_{split} not found in {a.feats}"
        return np.load(f).astype(np.float32)

    # 1. members, trained once for both modes
    M = train_members(V, store, a.n_models, Path(a.members_from) if a.members_from else out / "members")
    rng = np.random.RandomState(0)
    REF = (rng.randint(len(store.spk), size=CONFIG["ref_pairs"]), rng.randint(len(store.spk), size=CONFIG["ref_pairs"]))
    SC = {m: Scorer(V, store, cfg, nets, REF) for m, (cfg, nets) in M.items()}

    # ImageBind + age: level 1 uses the file's own means, level 0 the train means
    Fi_tr, Ai_tr = arr("face_ibv", "train"), arr("voice_iba", "train")
    mf, mv = Fi_tr.mean(0), Ai_tr.mean(0)
    age_f = lambda split: arr("face_vitageprob", split) @ AGE_MIDS
    age_v = lambda split: arr("voice_w2v2agage", split)[:, 0] * 100
    ib0 = lambda F_, A_: pair(V.l2n(F_ - mf), V.l2n(A_ - mv))
    R = CONFIG["s007_recipe"]
    ib_ref = ib0(Fi_tr[REF[0]], Ai_tr[REF[1]])
    age_ref = -np.abs(age_f("train")[REF[0]] - age_v("train")[REF[1]])
    z7, zib, zag = Std(SC["s007"].ref), Std(ib_ref), Std(age_ref)
    mix_ref = (1 - R["w_ib"]) * z7(SC["s007"].ref) + R["w_ib"] * zib(ib_ref); zmix = Std(mix_ref)
    z7b = Std(zmix(mix_ref) + R["w_age"] * zag(age_ref))
    zo = {m: Std(SC[m].ref) for m in SC}

    def indep(split, rows=None):
        """INDEP-01 score of every trial of `split` (or of the trials `rows` only), higher = same."""
        take = (lambda X: X) if rows is None else (lambda X: X[rows])
        lang = split.split("/")[1]
        comp = {}
        for m in CONFIG["cells"][lang]:
            s = SC[m].pairs(*map(take, SC[m].rows(split)))
            if m == "s007":
                ib = ib0(take(arr("face_ibv", split)), take(arr("voice_iba", split)))
                age = -np.abs(take(age_f(split)) - take(age_v(split)))
                s = z7b(zmix((1 - R["w_ib"]) * z7(s) + R["w_ib"] * zib(ib)) + R["w_age"] * zag(age))
            else:
                s = zo[m](s)
            comp[m] = s
        return np.mean([comp[m] for m in CONFIG["cells"][lang]], 0)

    # 2. clusters (thresholds from train speakers only) + diagnostics
    thr = {}
    sub = np.arange(len(store.spk))[::3]
    for kind, name, get in [("face", CONFIG["cluster_face"], store.face), ("voice", CONFIG["cluster_voice"], store.voice)]:
        Z = cent(get(name, "train"), V.l2n)
        thr[kind] = float(max(np.arange(.4, 1.21, .05), key=lambda t: adjusted_rand_score(store.spk[sub], clus(Z[sub], t))))
    log(f"cluster thresholds (train-calibrated): face {thr['face']:.2f} voice {thr['voice']:.2f}")

    # 3. both scorings, the decision per file
    S13, S1, S0, CHOSEN, rows = {}, {}, {}, {}, []
    override = dict(s.split("=", 1) for s in a.file_mode)
    assert set(override.values()) <= {"set13", "noclus", "indep"}, override
    for k in FILES:
        c = "/".join(k)
        cf = clus(cent(store.face(CONFIG["cluster_face"], c), V.l2n), thr["face"])
        cv = clus(cent(store.voice(CONFIG["cluster_voice"], c), V.l2n), thr["voice"])
        mats = {m: SC[m].l1_matrix(c) for m in CONFIG["cells"][k[1]]}
        F_, A_ = arr("face_ibv", c), arr("voice_iba", c)
        ib = V.l2n(F_ - F_.mean(0)) @ V.l2n(A_ - A_.mean(0)).T
        age = -np.abs(age_f(c)[:, None] - age_v(c)[None, :])
        mats["s007"] = zm((1 - R["w_ib"]) * zm(mats["s007"]) + R["w_ib"] * zm(ib.astype(np.float64))) \
            + R["w_age"] * zm(age.astype(np.float64))
        S13[k] = np.mean([rank(block(mats[m], cf, cv, CONFIG["b"])) for m in CONFIG["cells"][k[1]]], 0)
        S1[k] = np.mean([rank(np.diag(mats[m])) for m in CONFIG["cells"][k[1]]], 0)
        S0[k] = indep(c)
        d = diagnose(cf, cv)
        mode, why = decide(d) if a.mode == "auto" else (a.mode, "forced by --mode")
        if c in override:
            mode, why = override[c], f"--file-mode (auto said: {why})"
        CHOSEN[k] = {"set13": S13, "noclus": S1, "indep": S0}[mode][k]
        rows.append(dict(file=c, mode=mode, why=why, **d))
        log(f"{c:18s} {mode:6s} | {d['face_clusters']} face / {d['voice_clusters']} voice clusters for {d['rows']} rows"
            f" | median size {d['face_median_size']:.0f} / {d['voice_median_size']:.0f} | {why}")
    D = pd.DataFrame(rows); D.to_csv(out / "decisions.csv", index=False)

    # 4. self-test: an INDEP score must not depend on the other trials (score a random half of one file again)
    k = FILES[0]; c = "/".join(k)
    half = np.sort(np.random.RandomState(1).choice(len(S0[k]), len(S0[k]) // 2, replace=False))
    diff = float(np.abs(indep(c, half) - S0[k][half]).max())
    inv = dict(file=c, rows=len(half), max_abs_diff=diff, passed=diff < 1e-4)
    (out / "invariance.json").write_text(json.dumps(inv, indent=1))
    log(f"invariance self-test on {c}: max |diff| {diff:.2e} -> {'PASSED' if inv['passed'] else 'FAILED'}")
    assert inv["passed"], "an INDEP score changed when half of the file was removed"

    # 5. zips
    zips = {"set13": out / "submission_set13.zip", "noclus": out / "submission_noclus.zip",
            "indep": out / "submission_indep.zip", "chosen": out / "submission.zip"}
    for name, sc in [("set13", S13), ("noclus", S1), ("indep", S0), ("chosen", CHOSEN)]:
        V.write_zip(store, zips[name], sc); V.check_zip(store, zips[name])
    for spec in a.reference:
        name, ref = spec.split("=", 1)
        for fn in V.NAMES.values():
            x = pd.read_csv(zipfile.ZipFile(zips[name]).open(fn), sep=" ", header=None)[1]
            y = pd.read_csv(zipfile.ZipFile(ref).open(fn), sep=" ", header=None)[1]
            log(f"{name:6s} {fn:45s} rank-corr with {Path(ref).name}: {x.corr(y, method='spearman'):.4f}")
    log("chosen per file:", dict(zip(D.file, D["mode"])), "->", zips["chosen"])


if __name__ == "__main__":
    main()
