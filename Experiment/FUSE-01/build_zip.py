"""Build the FUSE-01 candidate: rank(007)+rank(010B) for the two English cells, 003c verbatim for Bangla."""
import zipfile, numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.stats import rankdata
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "EDA-000_raw"))
from eda_utils import load_dev
SRC = {"003c": E / "EXP-003_deep/out/submission_EXP003_C_ens3+k4_file_raw.zip",
       "007": E / "EXP-007_fusion/out/submission_EXP007_n10_w0.25.zip",
       "010B": E / "EXP-010_feat/out/submission_newfeat.zip"}
PLAN = {("no_gender", "English", "no_gender/sub_score_v4_English_heard.txt"): ["007", "010B"],
        ("no_gender", "Bangla", "no_gender/sub_score_v4_Bangla_unheard.txt"): ["003c"],
        ("gender", "English", "gender/sub_score_v4_English_heard.txt"): ["007", "010B"],
        ("gender", "Bangla", "gender/sub_score_v4_Bangla_unheard.txt"): ["003c"]}
out = Path(__file__).parent / "out/submission_FUSE01.zip"
with zipfile.ZipFile(out, "w") as zo:
    for (p, l, fn), mem in PLAN.items():
        if len(mem) == 1:
            zo.writestr(fn, zipfile.ZipFile(SRC[mem[0]]).read(fn)); continue
        t = load_dev(p, l)[2]; R = []
        for m in mem:
            d = pd.read_csv(zipfile.ZipFile(SRC[m]).open(fn), sep=" ", header=None, names=["pid", "s"])
            assert (d.pid.values == t.pair_id.values).all()
            R.append(rankdata(-d.s.values) / (len(d) + 1))        # zips are lower = same -> rank of "same-ness"
        s = np.mean(R, 0)
        zo.writestr(fn, "\n".join(f"{a} {-b:.6f}" for a, b in zip(t.pair_id, s)) + "\n")   # back to lower = same
# verify
with zipfile.ZipFile(out) as zf:
    assert sorted(zf.namelist()) == sorted(f for *_, f in PLAN)
    for (p, l, fn), _ in PLAN.items():
        d = pd.read_csv(zf.open(fn), sep=" ", header=None, names=["pid", "s"]); t = load_dev(p, l)[2]
        assert len(d) == len(t) and (d.pid.values == t.pair_id.values).all() and np.isfinite(d.s).all()
print("wrote + verified", out)
