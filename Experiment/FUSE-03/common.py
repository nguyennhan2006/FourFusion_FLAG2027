"""Shared loaders for FUSE-03/04 and DUR-02: per-cell scores of submitted systems, pseudo-labels, durations."""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "EDA-000_raw"))
from eda_utils import load_dev, eer_from_scores  # noqa: E402

NB3 = E.parent / "kaggle/NB3"
SRC = {"003c": E / "EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip",
       "007": E / "EXP-007_fusion/out/submission_EXP007_n10_w0.25.zip",
       "010B": E / "EXP-010_feat/out/submission_newfeat.zip",
       "r2_192": NB3 / "r2_ecapa192.zip", "r2_mix": NB3 / "r2_ecapa192_mix.zip", "r2_wavlm": NB3 / "r2_wavlm.zip"}
CELLS = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
CB_NOW = {("no_gender", "English"): 23.61, ("no_gender", "Bangla"): 26.88,
          ("gender", "English"): 29.12, ("gender", "Bangla"): 36.10}
rank = lambda s: rankdata(s) / (len(s) + 1)


def cell_data(k, members, labels="arc"):
    """-> dict of higher-is-same rank arrays, labels, known mask, face-cluster groups, durations, txt."""
    t = load_dev(*k)[2]; fn = CELLS[k]; R = {}
    for m in members:
        d = pd.read_csv(zipfile.ZipFile(SRC[m]).open(fn), sep=" ", header=None, names=["pid", "s"])
        assert (d.pid.values == t.pair_id.values).all()
        R[m] = rank(-d.s.values)
    L = np.load(E / f"SEL-01_selector/pseudo_labels_{labels}.npz")
    tk = f"{k[0]}_{k[1]}"
    dur = pd.read_csv(E / "EDA-001_flag_questions/wav_durations.csv")
    dmap = dict(zip(dur.name, dur.dur))
    d = np.array([dmap[f"dev_set/{k[0]}/{v}"] for v in t.voice])
    return dict(R=R, y=L[tk + "_lab"], known=L[tk + "_known"], fc=L[tk + "_fc"], dur=d, t=t)


def eer(s, y):
    return eer_from_scores(s, y)


def group_folds(fc, n=5, seed=0):
    """Folds over pseudo-identities (face clusters): a person is never in train and val at once."""
    ids = np.unique(fc); rng = np.random.RandomState(seed); rng.shuffle(ids)
    return [np.isin(fc, part) for part in np.array_split(ids, n)]
