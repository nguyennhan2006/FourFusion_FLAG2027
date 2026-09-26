"""Generate the two Kaggle notebooks from cell lists (keeps the JSON valid and easy to regenerate)."""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = (HERE / "flag_lib.py").read_text(encoding="utf-8")


def nb(cells, gpu=True):
    return {
        "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                   **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"},
                     "accelerator": "GPU" if gpu else "None"},
        "nbformat": 4, "nbformat_minor": 5,
    }


# flag_lib.py itself contains triple-quoted docstrings, so inlining it as a Python string literal
# terminates the literal early and corrupts the cell. Embed it base64-encoded instead.
_B64 = base64.b64encode(LIB.encode("utf-8")).decode("ascii")
_B64_WRAPPED = "\n".join(_B64[i:i + 100] for i in range(0, len(_B64), 100))
WRITE_LIB = (
    "# Writes flag_lib.py next to the notebook so the rest of the cells can import it.\n"
    "import base64\n"
    "_SRC = '''\n" + _B64_WRAPPED + "\n'''\n"
    "open('flag_lib.py', 'wb').write(base64.b64decode(''.join(_SRC.split())))\n"
    "print('flag_lib.py written:', len(open('flag_lib.py', encoding='utf-8').read()), 'chars')\n"
)

SETUP = '''import os, sys, time, json, itertools, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch

# flag_lib searches recursively, so any nesting Kaggle applies to the uploaded dataset works.
os.environ["FLAG_DATA"] = "/kaggle/input"

import flag_lib as L
print("train csv found at:", L.find_file("train/train_English_faces.csv"))
print("device:", L.DEVICE, "| GPUs:", torch.cuda.device_count())
Xf, Xv, spk, gmap, vgrp = L.load_train()
print("train:", Xf.shape, Xv.shape, len(set(spk)), "speakers |", len(np.unique(vgrp)), "unique voice clips")
SEEDS = [1, 2, 3]
'''

SPLIT_PREP = '''# Pre-compute the face PCA for each split once (the only slow CPU step); cached inside flag_lib.Prep.
t0 = time.time()
SPLITS = {}
for s in SEEDS:
    tr, va = L.speaker_split(spk, s)
    SPLITS[s] = (np.where(tr)[0], np.where(va)[0])
    L.Prep(Xf[SPLITS[s][0]], Xv[SPLITS[s][0]], key=f"{len(SPLITS[s][0])}_{int(SPLITS[s][0].sum())}")
ALL = np.arange(len(spk))
L.Prep(Xf, Xv, key=f"{len(ALL)}_{int(ALL.sum())}")
print("PCA ready in", round(time.time() - t0), "s")

TRIALS = {}
for s in SEEDS:
    _, vai = SPLITS[s]
    for sg in [False, True]:
        TRIALS[(s, sg)] = L.build_trials(spk[vai], gmap, seed=s, n_pos=3000, n_neg=3000, same_gender=sg)
'''

EVAL_HELPER = '''# Aligners and the linear CCA are fit once per split and reused by every config.
ALIGNERS, CCAS = {}, {}
for s in SEEDS:
    tri, _ = SPLITS[s]
    pr = L.Prep(Xf[tri], Xv[tri], key=f"{len(tri)}_{int(tri.sum())}")
    ALIGNERS[s] = (L.Aligner(pr.f(Xf[tri])), L.Aligner(pr.v(Xv[tri])), L.Aligner(Xf[tri]), L.Aligner(Xv[tri]))
    CCAS[s] = L.RidgeCCA(k=4, reg=1.0, pca_x=128).fit(Xf[tri], Xv[tri])

def eval_cfg(cfg, seeds_per_split=1, use_cca=True, alpha=0.0, cca_weight=0.25):
    """Mean internal EER over the 3 speaker-disjoint splits -> (int_ng, int_g).

    Selection metric is int_g. alpha > 0 applies partial CORAL on top of mean centering.
    cca_weight is the CCA's share of the fused score (EXP-007: ~0.25 beats equal weighting).
    """
    out = {"no_gender": [], "gender": []}
    for s in SEEDS:
        tri, vai = SPLITS[s]
        aF, aV, aFraw, aVraw = ALIGNERS[s]
        embs = []
        for i in range(seeds_per_split):
            net, prep, _ = L.train_model(Xf, Xv, spk, gmap, vgrp, tri, cfg, seed=s + 100 * i)
            embs.append(L.embed_one(net, prep, Xf[vai], Xv[vai], alF=aF, alV=aV, alpha=alpha))
        cca_emb = ((CCAS[s].transform_x(aFraw(Xf[vai], alpha)), CCAS[s].transform_y(aVraw(Xv[vai], alpha)))
                   if use_cca else None)
        for sg in [False, True]:
            fi, vj, lab = TRIALS[(s, sg)]
            sc = L.fuse_scores(embs, fi, vj, cca=cca_emb, cca_weight=cca_weight)
            out["gender" if sg else "no_gender"].append(L.eer_from_scores(sc, lab))
    return float(np.mean(out["no_gender"])), float(np.mean(out["gender"]))

print("smoke test:", np.round(eval_cfg(dict(epochs=5)), 2))
'''

SWEEP = '''# Local 3-split evidence (EXP-005): dedup_clip=False beat True, tau 0.05 and bs 512 beat the defaults,
# and hid=1024 led on ONE seed but not over three -> everything stays in the grid, nothing is assumed.
GRID = []
for hid in [512, 1024]:
    for emb in [64, 128, 256]:
        for ep in [25, 40, 60]:
            GRID.append(dict(hid=hid, emb=emb, epochs=ep))
for tau in [0.03, 0.05, 0.1]:
    GRID.append(dict(hid=512, emb=128, epochs=40, tau=tau))
for bs in [512, 1024]:
    GRID.append(dict(hid=512, emb=128, epochs=40, bs=bs))
for drop in [0.3, 0.7]:
    GRID.append(dict(hid=512, emb=128, epochs=40, drop=drop))
for wd in [1e-3, 1e-1]:
    GRID.append(dict(hid=512, emb=128, epochs=40, wd=wd))
GRID.append(dict(hid=512, emb=128, epochs=40, dedup_clip=True))   # the assumption EXP-005 overturned
print(len(GRID), "configs x", len(SEEDS), "splits")

rows = []
t0 = time.time()
for i, cfg in enumerate(GRID):
    ng, g = eval_cfg(cfg)
    rows.append(dict(**{**L.BASE_CFG, **cfg}, int_ng=round(ng, 2), int_g=round(g, 2)))
    pd.DataFrame(rows).to_csv("sweep.csv", index=False)
    print(f"[{i+1}/{len(GRID)}] {cfg} -> int_g {g:.2f} int_ng {ng:.2f} ({round(time.time()-t0)}s)", flush=True)

pd.DataFrame(rows).sort_values("int_g").head(15)
'''

BEST_CFG = '''KEYS = ("hid", "emb", "epochs", "tau", "drop", "wd", "bs", "dedup_clip", "lr")
best = pd.read_csv("sweep.csv").sort_values("int_g").iloc[0].to_dict()
BEST = {}
for k in KEYS:
    if k in best and pd.notna(best[k]):
        BEST[k] = bool(best[k]) if k == "dedup_clip" else (int(best[k]) if k in ("hid", "emb", "epochs", "bs") else float(best[k]))
print("best cfg:", BEST, "-> int_g", best["int_g"])
'''

AUG_SWEEP = '''# Internal is English-only, so an augmentation can only be shown NOT to hurt here; its Bangla
# benefit has to be measured on CodaBench. Keep anything within ~0.3 of the best.
AUG_GRID = [dict(aug="none", aug_p=0.0)]
for p in [0.1, 0.3, 0.5]:
    AUG_GRID += [dict(aug="noise", aug_p=p), dict(aug="dropdim", aug_p=p),
                 dict(aug="shift", aug_p=p), dict(aug="mixup", aug_p=p)]
AUG_GRID += [dict(aug="both", aug_p=0.3)]

arows = []
for a in AUG_GRID:
    ng, g = eval_cfg(dict(BEST, **a), seeds_per_split=3)
    arows.append(dict(**a, int_ng=round(ng, 2), int_g=round(g, 2)))
    pd.DataFrame(arows).to_csv("aug_sweep.csv", index=False)
    print(arows[-1], flush=True)
pd.DataFrame(arows).sort_values("int_g")
'''

ALPHA = '''# EXP-004 showed alpha=1 (full CORAL) costs +10 EER, because a dev file's covariance IS its speaker
# structure. This checks whether a small alpha buys language robustness without that cost.
prows = []
for alpha in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]:
    ng, g = eval_cfg(BEST, seeds_per_split=3, alpha=alpha)
    prows.append(dict(alpha=alpha, int_ng=round(ng, 2), int_g=round(g, 2)))
    print(prows[-1], flush=True)
pd.DataFrame(prows).to_csv("alpha.csv", index=False)
BEST_ALPHA = float(pd.DataFrame(prows).sort_values("int_g").iloc[0].alpha)
print("best alpha:", BEST_ALPHA, "(0.0 = keep mean centering only)")
'''

ENSEMBLE = '''# EXP-007: equal-weight fusion over n+1 members silently shrinks the CCA's weight as n grows
# (1/4 at n=3 -> 1/11 at n=10), which looked like "bigger ensembles are worse". Give the CCA an
# explicit weight w instead; the optimum sits near w=0.25-0.30 at every n.
erows = []
for n in [3, 5, 10]:
    for w in [0.0, 0.15, 0.25, 0.3, 0.4]:
        ng, g = eval_cfg(BEST, seeds_per_split=n, cca_weight=w)
        erows.append(dict(nseed=n, w=w, int_ng=round(ng, 2), int_g=round(g, 2)))
        print(erows[-1], flush=True)
    pd.DataFrame(erows).to_csv("ensemble.csv", index=False)
edf = pd.DataFrame(erows).sort_values("int_g")
N_BEST, W_BEST = int(edf.iloc[0].nseed), float(edf.iloc[0].w)
print("best ensemble:", N_BEST, "seeds, CCA weight", W_BEST)
edf.head(10)
'''

SUBMIT_CORE = '''devs = {k: L.load_dev(*k) for k in L.NAMES}
prep_all = L.Prep(Xf, Xv, key=f"{len(ALL)}_{int(ALL.sum())}")
cca = L.RidgeCCA(k=4, reg=1.0, pca_x=128).fit(Xf, Xv)
alF_all, alV_all = L.Aligner(prep_all.f(Xf)), L.Aligner(prep_all.v(Xv))
alFraw, alVraw = L.Aligner(Xf), L.Aligner(Xv)

def make_submission(cfg, n_seeds, tag, use_cca=True, alpha=0.0, cca_weight=0.25):
    """Trains n_seeds models on all 70 speakers, fuses with the CCA, writes a CodaBench zip."""
    nets = [L.train_model(Xf, Xv, spk, gmap, vgrp, ALL, cfg, seed=1 + 100 * i)[:2] for i in range(n_seeds)]
    scores = {}
    for k, (a, b, t) in devs.items():
        embs = [L.embed_one(net, prep, a, b, alF=alF_all, alV=alV_all, alpha=alpha) for net, prep in nets]
        cca_emb = (cca.transform_x(alFraw(a, alpha)), cca.transform_y(alVraw(b, alpha))) if use_cca else None
        ii = np.arange(len(t))
        scores[k] = L.fuse_scores(embs, ii, ii, cca=cca_emb, cca_weight=cca_weight)
    p = L.write_submission(f"submission_{tag}.zip", scores, devs)
    print("wrote", p, flush=True)
    return p
'''

SUBMIT_SWEEP = SUBMIT_CORE + '''
BEST_AUG = pd.read_csv("aug_sweep.csv").sort_values("int_g").iloc[0].to_dict()
CFG_FINAL = dict(BEST, aug=BEST_AUG["aug"], aug_p=float(BEST_AUG["aug_p"]))
print("final cfg:", CFG_FINAL)

# Use what the sweeps actually chose, not hard-coded numbers.
make_submission(BEST, N_BEST, f"best_n{N_BEST}_w{W_BEST}", cca_weight=W_BEST)
if (CFG_FINAL.get("aug", "none") != "none"):     # only if an augmentation really won
    make_submission(CFG_FINAL, N_BEST, f"best_aug_n{N_BEST}_w{W_BEST}", cca_weight=W_BEST)
make_submission(BEST, N_BEST, f"best_n{N_BEST}_nocca", use_cca=False)
if BEST_ALPHA > 0:
    make_submission(BEST, N_BEST, f"best_n{N_BEST}_alpha{BEST_ALPHA}", alpha=BEST_ALPHA, cca_weight=W_BEST)
print(sorted(Path(".").glob("submission_*.zip")))
'''

SUBMIT_MANUAL = SUBMIT_CORE + '''
BEST = {k: v for k, v in CFG_FINAL.items() if k not in ("aug", "aug_p")}
print("final cfg:", CFG_FINAL)
make_submission(CFG_FINAL, N_SEEDS, "final_aug", alpha=BEST_ALPHA)
make_submission(BEST, N_SEEDS, "final_noaug", alpha=BEST_ALPHA)
make_submission(CFG_FINAL, N_SEEDS, "final_aug_nocca", use_cca=False, alpha=BEST_ALPHA)
print(sorted(Path(".").glob("submission_*.zip")))
'''

CHECK = '''# Sanity check every zip before downloading: 4 files, right order, right ids, no NaN.
for zp in sorted(Path(".").glob("submission_*.zip")):
    zf = zipfile.ZipFile(zp)
    ok = sorted(zf.namelist()) == sorted(L.NAMES.values())
    detail = {}
    for k, fn in L.NAMES.items():
        d = pd.read_csv(zf.open(fn), sep=" ", header=None, names=["pid", "s"])
        detail["/".join(k)] = dict(n=len(d), ids_ok=bool((d.pid.values == devs[k][2].pair_id.values).all()),
                                   nan=int(d.s.isna().sum()))
    print(zp.name, "files_ok" if ok else "FILES WRONG")
    print("   ", detail)
'''

INTRO = """# FLAG 2027 — GPU sweep (Kaggle T4 x2)

Baseline to beat: **EXP-003c = 31.34** (per cell 25.60 / 27.88 / 34.01 / 37.87).

Settled locally, so not re-litigated here:
* InfoNCE (symmetric, SupCon positives) beats CE+OPL (the FOP loss) and beats linear CCA.
* Fusing the deep ensemble with a 4-dim linear CCA helps.
* Per-file **mean** centering helps; full CORAL destroys identity (41.36) — section 3 tests partial CORAL.
* AS-norm looks good internally but **hurts** on CodaBench — scores stay raw cosine.
* Dedup-per-voice-clip was assumed to help; a 3-split sweep said otherwise, so it stays in the grid.

Selection metric is `int_g` = EER on gender-constrained internal trials over 3 speaker-disjoint splits.
Internal contains English only, so a Bangla gain can only be confirmed by submitting.

**Settings:** Accelerator = GPU T4 x2, Internet off. Just add the dataset — the loader finds it by filename.
"""

SUBMIT_INTRO = """# FLAG 2027 — final submission builder

Use after the sweep notebook: paste the winning config, train an ensemble on all 70 speakers,
write the zips. Nothing is selected here; this notebook only rebuilds and packages.
"""

MANUAL = '''# Paste the config chosen by the sweep notebook.
CFG_FINAL = dict(hid=512, emb=128, epochs=40, tau=0.05, drop=0.5, wd=1e-2, bs=512,
                 dedup_clip=False, aug="noise", aug_p=0.3)
N_SEEDS = 10
BEST_ALPHA = 0.0   # > 0 only if the sweep found partial CORAL helps
'''

sweep_cells = [
    ("markdown", INTRO),
    ("code", WRITE_LIB), ("code", SETUP), ("code", SPLIT_PREP), ("code", EVAL_HELPER),
    ("markdown", "## 1. Architecture / optimisation sweep"), ("code", SWEEP), ("code", BEST_CFG),
    ("markdown", "## 2. Augmentation sweep (targets the Bangla gap)"), ("code", AUG_SWEEP),
    ("markdown", "## 3. Partial CORAL (alpha)"), ("code", ALPHA),
    ("markdown", "## 4. Ensemble size"), ("code", ENSEMBLE),
    ("markdown", "## 5. Submissions"), ("code", SUBMIT_SWEEP), ("code", CHECK),
]

submit_cells = [
    ("markdown", SUBMIT_INTRO),
    ("code", WRITE_LIB), ("code", SETUP), ("code", SPLIT_PREP), ("code", MANUAL),
    ("code", SUBMIT_MANUAL), ("code", CHECK),
]

for name, cells in [("FLAG_01_sweep.ipynb", sweep_cells), ("FLAG_02_submit.ipynb", submit_cells)]:
    (HERE / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", name, len(cells), "cells")
