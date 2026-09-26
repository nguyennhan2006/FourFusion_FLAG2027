"""Rebuild the 28.69 submission (FUSE-03) from its four component submissions. No training, no extra data.

    python build_best.py            -> submission_best_28.69.zip (+ check against the submitted zip if present)

The whole recipe is BEST_CONFIG below; the code only interprets it. Every component zip is lower = same;
ranks are taken on "same-ness" (= -score) and written back as lower = same.

    no_gender/English : mean(rank c007, rank c010B)                                         CB 23.61
    gender/English    : mean(rank c007, rank c010B)                                         CB 29.12
    no_gender/Bangla  : mean(rank c003, rank r2_mix)                                        CB 26.88
    gender/Bangla     : w(d) rank c003 + (1 - w(d)) rank r2_mix, w(d) = sigmoid(0.5 ln d - 1.5),
                        d = duration (s) of the trial's voice clip                           CB 35.15

The gate is adaptive REWEIGHTING, not a proven duration effect: w(d) goes 0.24 (2 s) -> 0.33 (4.7 s, Bangla
median) -> 0.44 (12 s) -- it gives c003 MORE weight on long clips, where r2_mix alone is strongest (FUSE-03 /
DUR-02 in EXPERIMENT_LOG). Read it as "lean ~0.3/0.7 towards r2_mix".
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
C = HERE / "components"
FILES = {"no_gender/English": "no_gender/sub_score_v4_English_heard.txt",
         "no_gender/Bangla": "no_gender/sub_score_v4_Bangla_unheard.txt",
         "gender/English": "gender/sub_score_v4_English_heard.txt",
         "gender/Bangla": "gender/sub_score_v4_Bangla_unheard.txt"}

BEST_CONFIG = {
    "no_gender/English": {"type": "rank_fusion", "systems": ["c007", "c010B"], "weights": [0.5, 0.5]},
    "gender/English": {"type": "rank_fusion", "systems": ["c007", "c010B"], "weights": [0.5, 0.5]},
    "no_gender/Bangla": {"type": "rank_fusion", "systems": ["c003", "r2_mix"], "weights": [0.5, 0.5]},
    # w_first = sigmoid(a ln d + b); a, b fitted out-of-fold on SEL-01b dev pseudo-labels (declared)
    "gender/Bangla": {"type": "duration_gate", "systems": ["c003", "r2_mix"], "a": 0.5, "b": -1.5,
                      "fit_on": "dev_pseudo_labels"},
}

# Transductive policy (see docs/PLAN_V4.md §3). Scores must come from component systems trained on labelled
# train data only; dev pseudo-labels may pick systems and fit a handful of fusion parameters, nothing more.
ALLOWED_TYPES = {"rank_fusion", "duration_gate"}
ALLOWED_FIT = {None, "dev_pseudo_labels"}
MAX_FITTED_PARAMS = 2


def check_policy(cfg):
    n_fit = 0
    for cell, c in cfg.items():
        assert c["type"] in ALLOWED_TYPES, f"{cell}: {c['type']} is not an allowed combiner"
        assert c.get("fit_on") in ALLOWED_FIT, f"{cell}: fit_on={c.get('fit_on')!r}"
        assert all((C / f"{s}.zip").exists() for s in c["systems"]), f"{cell}: scores must come from component zips"
        n_fit += 2 if c.get("fit_on") else 0
    assert n_fit <= MAX_FITTED_PARAMS, f"{n_fit} parameters fitted on dev pseudo-labels"


def read(comp, fn):
    d = pd.read_csv(zipfile.ZipFile(C / f"{comp}.zip").open(fn), sep=" ", header=None, names=["pid", "s"])
    return d.pid.values, -d.s.values                     # higher = same


def rank(s):
    return rankdata(s) / (len(s) + 1)


def durations(cell):
    """Voice-clip durations for one dev file, from its pair list order (== row order of every zip)."""
    protocol, lang = cell.split("/")
    dur = pd.read_csv(C / "wav_durations.csv")
    dmap = dict(zip(dur.name, dur.dur))
    txt = HERE.parents[1] / "kaggle_upload" / "dev" / f"{protocol}_{lang}_test.txt"
    t = pd.read_csv(txt, sep=" ", header=None, names=["pid", "voice", "face"])
    return t.pid.values, np.array([dmap[f"dev_set/{protocol}/{v}"] for v in t.voice])


def combine(cell, c):
    reads = [read(s, FILES[cell]) for s in c["systems"]]
    pid = reads[0][0]
    assert all((p == pid).all() for p, _ in reads)
    R = np.stack([rank(s) for _, s in reads])
    if c["type"] == "rank_fusion":
        return pid, np.asarray(c["weights"]) @ R
    dpid, d = durations(cell)
    assert (dpid == pid).all()
    w = 1 / (1 + np.exp(-(c["a"] * np.log(np.maximum(d, 0.5)) + c["b"])))
    return pid, w * R[0] + (1 - w) * R[1]


check_policy(BEST_CONFIG)
out = HERE / "submission_best_28.69.zip"
with zipfile.ZipFile(out, "w") as zo:
    for cell, c in BEST_CONFIG.items():
        pid, s = combine(cell, c)
        assert np.isfinite(s).all()
        zo.writestr(FILES[cell], "\n".join(f"{p} {-v:.6f}" for p, v in zip(pid, s)) + "\n")   # back to lower = same
print("wrote", out)

ref = HERE.parents[1] / "Experiment/FUSE-03/out/submission_FUSE03.zip"
if ref.exists():
    for fn in FILES.values():
        a = pd.read_csv(zipfile.ZipFile(out).open(fn), sep=" ", header=None)[1]
        b = pd.read_csv(zipfile.ZipFile(ref).open(fn), sep=" ", header=None)[1]
        print(f"{fn:45s} rank-corr with submitted FUSE-03: {a.corr(b, method='spearman'):.6f}")
