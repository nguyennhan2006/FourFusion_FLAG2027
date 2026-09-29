"""EXT-02 — PLAN_V4 A2 (bilingual validation with true labels) + A3 (pretrain -> fine-tune), local CPU.

    python run.py v3_train                      # full: 5 folds x 5 arms x 3 models
    python run.py v1_complete v2_complete          # once feats_ext/ from FLAG_08 is downloaded
    python run.py v3_train --smoke              # 1 model, 2 folds, 300 trials: plumbing only

Features: kaggle/NB5 (v3, round 1) and every folder listed in FLAG_FEATS_EXTRA (e.g. the downloaded feats_ext).
Identities flagged in kaggle/overlap.csv (overlap with v4 train/dev) are excluded.
Output: out/ext02_<src>.csv (per fold / arm / language, resumable) and out/ext02_summary.csv.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import pandas as pd  # noqa: E402
import flag_v2 as V  # noqa: E402
import flag_bilingual as B  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
smoke = "--smoke" in sys.argv
store = V.FeatureStore(ROOT / "kaggle_upload", feats_dir=ROOT / "kaggle" / "NB5")
ov = ROOT / "kaggle" / "overlap.csv"
exclude = tuple(pd.read_csv(ov).query("overlap").key) if ov.exists() else ()
out = HERE / "out"; out.mkdir(exist_ok=True)
kw = dict(n_folds=2, n_models=1, n_trials=300) if smoke else {}
R = []
for src in args or ["v3_train"]:
    csv = out / (f"ext02_{src}_smoke.csv" if smoke else f"ext02_{src}.csv")
    R.append(B.run(store, src, exclude=exclude, out_csv=csv, log=V.log, **kw))
S = B.summarise(pd.concat(R))
print("\n" + S.to_string(index=False))
if not smoke:
    S.to_csv(out / "ext02_summary.csv", index=False)
