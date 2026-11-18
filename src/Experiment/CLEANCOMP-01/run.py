r"""CLEANCOMP-01 (README.md): the SAFE-04 fold columns of F1's two older Bangla parts, learned again under LID_STRICT=1.
    LID_STRICT=1 VOX_LID=<FLAG_42 lid_summary.json> FOLDS=1 python run.py        (lab_entry.py CLEANCOMP-01; table cleancomp01_fold<s>)
Columns written (names of the old tables): XALL_deep0 / XALL_cca / XALL_ib / XALL_age (S6 bridge, lab_streams XALL), relc_fia_eb (rel),
relg_fia_eb (relg), anP512_deep / anP512_cca (v11's T3 head, BIGB-01 P512), t3_relg. Same code, same seeds as the old tables; the only
change is the list: VoxCeleb2 rows of lid_strict.json 'vox2' (T3ENC-01 dropped_speakers), MAV voice files of 'mav_files' (MAV_DROP in
ext_rows, id_rows). Without LID_STRICT=1 the job refuses to run."""
import os, sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

os.environ["SAFE04_NOMAIN"] = "1"
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
TAG = "cleancomp01"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run(s):
    assert os.environ.get("LID_STRICT") == "1" and os.environ.get("VOX_LID"), "CLEANCOMP-01 runs under LID_STRICT=1 (+ VOX_LID)"
    t0 = time.time()
    A = _load("allneg01run", "ALLNEG-01/run.py")                 # puts SAFE-04, VOX2-01, T3ENC-01 on sys.path
    import torch
    import rel_sweep5 as R
    import rel_sweep2 as R2
    from run_bridge import patch_large
    patch_large(vars(R))
    torch.set_num_threads(int(os.environ.get("SAFE04_THREADS", "10")))
    log = R.log
    od = R.SP / "out/safe04"; od.mkdir(parents=True, exist_ok=True)
    log(f"CLEANCOMP-01 fold {s}: MAV_DROP {len(R.MAV_DROP)} files, {int(R.VMETA.path.isin(R.MAV_DROP).sum())} matched in VMETA")
    if not R.VMETA.path.isin(R.MAV_DROP).sum():
        log(f"VMETA.path sample {R.VMETA.path.head(2).tolist()} | MAV_DROP sample {sorted(R.MAV_DROP)[:2]}")
    assert R.MAV_DROP and R.VMETA.path.isin(R.MAV_DROP).sum() > 0, "MAV_DROP list does not match VMETA.path"
    fold = R.make_fold(s)
    # 1. S6 bridge streams (lab_streams run_fold, XALL arm only)
    m = R.arm_ctrl(fold, ext=True)[0]
    sc = R.L0(m, fold["v4tr"])
    cols = {k: [] for k in ("deep0", "cca", "ib", "age")}
    for (data, prot), d in fold["files"].items():
        S_ = sc.streams(d)
        for k in cols:
            cols[k].append(np.asarray(S_[k]))
    X = pd.DataFrame({f"XALL_{k}": np.concatenate(v) for k, v in cols.items()})
    log(f"fold {s} XALL trained + scored [{time.time() - t0:.0f}s]")
    # 2. rel / relg (rel_sweep2 / rel_sweep5 as the old tables; id_rows drops the MAV_DROP voice files)
    R2.VARIANTS = {"relc_fia_eb": R2.VARIANTS["relc_fia_eb"]}
    R2.run2(s); R.run5(s)
    rel = pd.read_parquet(od / f"relonly_multi_fold{s}.parquet"); relg = pd.read_parquet(od / f"relonly_g_fold{s}.parquet")
    log(f"fold {s} rel / relg [{time.time() - t0:.0f}s]")
    # 3. v11's T3 head (BIGB-01 P512) and t3_relg (T3ENC-01, no bridge arm)
    A.TAG, A.DECISION, A.ARMS = f"{TAG}_p512", "P512", {"P512": dict(P=512, K=2)}
    A.run(s)
    T = _load("t3enc01run", "T3ENC-01/run.py")
    T.TAG, T.ARMS = f"{TAG}_t3r", {}
    T.run(s)
    P = pd.read_parquet(od / f"{TAG}_p512_fold{s}.parquet"); Q = pd.read_parquet(od / f"{TAG}_t3r_fold{s}.parquet")
    assert len(P) == len(Q) == len(X) == len(rel) and (P.y.values == Q.y.values).all() and (P.vi.values == Q.vi.values).all()
    assert (rel.y.values == P.y.values).all() and (relg.y.values == P.y.values).all()
    D = pd.concat([P[["fold", "data", "protocol", "y", "fi", "vi"]].reset_index(drop=True), X,
                   rel[["relc_fia_eb"]].reset_index(drop=True), relg[["relg_fia_eb"]].reset_index(drop=True),
                   P[["anP512_deep", "anP512_cca"]].reset_index(drop=True), Q[["t3_relg"]].reset_index(drop=True)], axis=1)
    assert np.isfinite(D.select_dtypes("number").values).all(), "NaN in the CLEANCOMP-01 table"
    for f in (f"relonly_multi_fold{s}", f"relonly_g_fold{s}", f"{TAG}_p512_fold{s}", f"{TAG}_t3r_fold{s}"):
        (od / f"{f}.parquet").unlink()                      # helper tables: only the merged one travels
    D.to_parquet(od / f"{TAG}_fold{s}.parquet")
    log(f"fold {s} done: {len(D)} trials, columns {list(D.columns[6:])} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    if os.environ.get("CC_MODE") == "final":
        sys.path.insert(0, str(HERE)); import final; final.main(); sys.exit(0)
    for s in [int(x) for x in os.environ.get("FOLDS", "1").split(",")]:
        run(s)
