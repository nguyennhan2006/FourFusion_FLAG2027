"""FUSE-02 candidate: English cells = FUSE-01 (CB 23.61 / 29.12) verbatim; Bangla cells = rank(003c)+rank(r2_ecapa192_mix)."""
import sys, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "EDA-000_raw")); from eda_utils import load_dev
F01 = E / "FUSE-01/out/submission_FUSE01.zip"
SRC = {"003c": E / "EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip", "r2_mix": E.parent / "kaggle/NB3/r2_ecapa192_mix.zip"}
CELLS = [("no_gender", "English", "no_gender/sub_score_v4_English_heard.txt", None),
         ("no_gender", "Bangla", "no_gender/sub_score_v4_Bangla_unheard.txt", ["003c", "r2_mix"]),
         ("gender", "English", "gender/sub_score_v4_English_heard.txt", None),
         ("gender", "Bangla", "gender/sub_score_v4_Bangla_unheard.txt", ["003c", "r2_mix"])]
out = Path(__file__).parent / "out/submission_FUSE02.zip"
with zipfile.ZipFile(out, "w") as zo:
    for p, l, fn, mem in CELLS:
        if mem is None:
            zo.writestr(fn, zipfile.ZipFile(F01).read(fn)); continue
        t = load_dev(p, l)[2]; R = []
        for m in mem:
            d = pd.read_csv(zipfile.ZipFile(SRC[m]).open(fn), sep=" ", header=None, names=["pid", "s"])
            assert (d.pid.values == t.pair_id.values).all()
            R.append(rankdata(-d.s.values) / (len(d) + 1))
        zo.writestr(fn, "\n".join(f"{a} {-b:.6f}" for a, b in zip(t.pair_id, np.mean(R, 0))) + "\n")
with zipfile.ZipFile(out) as zf:
    assert sorted(zf.namelist()) == sorted(c[2] for c in CELLS)
    for p, l, fn, _ in CELLS:
        d = pd.read_csv(zf.open(fn), sep=" ", header=None, names=["pid", "s"]); t = load_dev(p, l)[2]
        assert len(d) == len(t) and (d.pid.values == t.pair_id.values).all() and np.isfinite(d.s).all()
        L = np.load(E / "SEL-01_selector/pseudo_labels_arc.npz"); kn, y = L[f"{p}_{l}_known"], L[f"{p}_{l}_lab"]
        from eda_utils import eer_from_scores
        print(f"{p}/{l}: pseudo {eer_from_scores(-d.s.values[kn], y[kn]):.2f}")
print("wrote + verified", out)
