r"""NEW (FourFusion reproduction repo, stage safe04_lab, level "core"): the columns of SAFE-04's streams_fold<s>.parquet that the
13.54 chain still uses, computed with the UNCHANGED code of Experiment/SAFE-04/lab_streams.py (imported, not edited).

lab_streams.run_fold(s) (lines 116-138) does, per fold: make_fold(s); train the XALL and CTRL bridges (arm_ctrl); score them (L0);
Extra(fold["v4tr"], fold["ext_pool"]); one row per validation trial with the keys (fold, data, protocol, y, pf, pv, fi, vi), the
bridge streams XALL_* / CTRL_* and the Extra streams (acl, lb, iba2, ibal, gen, pf_m, pv_m). This script runs the same calls
WITHOUT the bridge training: the keys come from the same make_fold(s) (seeded RandomState(2000 + s / 3000 + s / 4000 + s), no
global RNG), and Extra uses its own RandomState(0) and a deterministic lbfgs logistic regression, so neither depends on the
skipped training. With NOREL=1 + CLEANCOMP=1 the bridge columns are not used by the 13.54 chain (cleancomp_tables replaces
XALL_*; CTRL_* feed no fuser), so the "core" table carries every streams column that matters for 13.54: keys, pf_m, pv_m.

    SAFE04_SCRATCH=<sp> FLAG_LOCAL=1 FLAG_IN=<parent of the eda bundle> FLAG_OUT=<sp>/out FOLDS=1,2,3,4,5 python ff_core_streams.py
-> <sp>/out/safe04/core_streams_fold<s>.parquet (same row order and key columns as streams_fold<s>.parquet)"""
import os, sys, time
from pathlib import Path

os.environ["SAFE04_NOMAIN"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import lab_streams as L                                   # unchanged: executes build_l0_notebooks ENV / LOAD / LIB / DATA


def core_fold(s, out):
    t0 = time.time()
    fold = L.make_fold(s)
    ex = L.Extra(fold["v4tr"], fold["ext_pool"])
    rows = []
    for (data, prot), d in fold["files"].items():          # same loop and DataFrame construction as lab_streams.run_fold
        df = pd.DataFrame(dict(fold=s, data=data, protocol=prot, y=d["y"], pf=d["pf"], pv=d["pv"], fi=d["fi"], vi=d["vi"]))
        E = ex.streams(L.trial_extra(data, d))
        for k, v in E.items():
            df[k] = v
        rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    D.to_parquet(out)
    L.log(f"core fold {s}: {len(D)} trials -> {out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    od = Path(os.environ["SAFE04_SCRATCH"]) / "out" / "safe04"; od.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",") if x]:
        print(f"[ff_core_streams] fold {s} start", flush=True)
        core_fold(s, od / f"core_streams_fold{s}.parquet")
