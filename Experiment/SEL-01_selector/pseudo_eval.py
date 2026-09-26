"""SEL-01b as a library: local dev evaluation with pseudo-identity labels (ArcFace faces + own ECAPA-192 voices).

    python pseudo_eval.py A.zip                 # pseudo-EER per cell + coverage + identity-bootstrap 95% CI
    python pseudo_eval.py A.zip B.zip           # also A - B per cell with CI and P(A better)

    from pseudo_eval import evaluate, compare, cell_labels, group_folds

Labels are built by run.py (SEL_LABELS=arc -> pseudo_labels_arc.npz); this module only reads them.
Track record: Spearman 0.95-0.996 vs CodaBench on 9 systems, |pseudo - CB| <= 0.9 EER per cell over 3 submissions.

Allowed uses (docs/PLAN_V4.md §3), because the dev set is the test set of this phase:
  * choosing among a few pre-registered systems            yes
  * fitting <= 2 fusion parameters, out-of-fold             yes (declare as transductive)
  * training any model on these labels                      NO
  * producing submission scores from pseudo-identity / trial-list structure    NO (protocol leakage)
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "EDA-000_raw"))
from eda_utils import load_dev, eer_from_scores  # noqa: E402

CELLS = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
         ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
         ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
SHORT = {("no_gender", "English"): "ng/En", ("no_gender", "Bangla"): "ng/Bn",
         ("gender", "English"): "g/En", ("gender", "Bangla"): "g/Bn"}
_PAIRS = {}


def cell_labels(k, which="arc"):
    """-> (known mask, 0/1 label, face-cluster id) for every row of dev cell k, in pair-list order."""
    L = np.load(HERE / f"pseudo_labels_{which}.npz")
    t = f"{k[0]}_{k[1]}"
    return L[t + "_known"].astype(bool), L[t + "_lab"], L[t + "_fc"]


def zip_scores(path, k):
    """Higher = same scores of one cell of a submission zip (zips are lower = same), row-checked."""
    if k not in _PAIRS:
        _PAIRS[k] = load_dev(*k)[2].pair_id.values
    d = pd.read_csv(zipfile.ZipFile(path).open(CELLS[k]), sep=" ", header=None, names=["pid", "s"])
    assert (d.pid.values == _PAIRS[k]).all(), f"{path}: row order differs from the {k} pair list"
    return -d.s.values


def _boot(fc, fn, n, seed=0):
    """Resample pseudo-identities (face clusters), not trials: trials of one person are not independent."""
    ids = np.unique(fc)
    by = {c: np.where(fc == c)[0] for c in ids}
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        idx = np.concatenate([by[c] for c in rng.choice(ids, len(ids))])
        v = fn(idx)
        if v is not None:
            out.append(v)
    return np.array(out)


def evaluate(scores, which="arc", n_boot=300):
    """scores: submission zip path, or {cell: higher-is-same array}. -> DataFrame, one row per cell."""
    rows = []
    for k in CELLS:
        s = zip_scores(scores, k) if isinstance(scores, (str, Path)) else np.asarray(scores[k])
        known, y, fc = cell_labels(k, which)
        s, y, fc = s[known], y[known], fc[known]
        b = _boot(fc, lambda i: eer_from_scores(s[i], y[i]) if 0 < y[i].mean() < 1 else None, n_boot)
        rows.append(dict(cell=SHORT[k], pseudo_eer=round(eer_from_scores(s, y), 2), coverage=round(known.mean(), 3),
                         ci_lo=round(np.percentile(b, 2.5), 2), ci_hi=round(np.percentile(b, 97.5), 2)))
    df = pd.DataFrame(rows)
    return pd.concat([df, pd.DataFrame([dict(cell="overall", pseudo_eer=round(df.pseudo_eer.mean(), 2))])],
                     ignore_index=True)


def compare(a, b, which="arc", n_boot=500):
    """A - B pseudo-EER per cell (negative = A better), identity-bootstrap CI and P(A better)."""
    rows = []
    for k in CELLS:
        sa = zip_scores(a, k) if isinstance(a, (str, Path)) else np.asarray(a[k])
        sb = zip_scores(b, k) if isinstance(b, (str, Path)) else np.asarray(b[k])
        known, y, fc = cell_labels(k, which)
        sa, sb, y, fc = sa[known], sb[known], y[known], fc[known]
        d = _boot(fc, lambda i: (eer_from_scores(sa[i], y[i]) - eer_from_scores(sb[i], y[i]))
                  if 0 < y[i].mean() < 1 else None, n_boot)
        rows.append(dict(cell=SHORT[k], delta=round(eer_from_scores(sa, y) - eer_from_scores(sb, y), 2),
                         ci_lo=round(np.percentile(d, 2.5), 2), ci_hi=round(np.percentile(d, 97.5), 2),
                         P_A_better=round(float((d < 0).mean()), 2)))
    return pd.DataFrame(rows)


def group_folds(fc, n=5, seed=0):
    """Folds over pseudo-identities for out-of-fold fitting: a person is never in fit and held-out at once."""
    ids = np.unique(fc)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    return [np.isin(fc, part) for part in np.array_split(ids, n)]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--labels=")]
    which = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--labels=")), "arc")
    for p in args:
        print(f"\n{p}  (labels: {which})\n" + evaluate(p, which).to_string(index=False))
    if len(args) == 2:
        print(f"\nA - B (negative = A better)\n" + compare(*args, which=which).to_string(index=False))
