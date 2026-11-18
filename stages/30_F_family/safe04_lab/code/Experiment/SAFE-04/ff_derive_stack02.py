r"""NEW (FourFusion reproduction repo, stage safe04_lab): the two STACK-02 products that the e2e-dev bundle keeps, from the streams
tables of this run, with the original formulas:
  stack02_rows.parquet  columns ["fold", "data", "y", "fp", "vp"] only: kaggle/build_e2e_dev_bundle.py safe04_tables() line 91 keeps
                        exactly these five columns of Experiment/STACK-02/export.py's table, so the candidate columns of the ten
                        other experiments (export.py / pools.py TABLES) are never needed;
                        fp = export.py line 35 (v4 rows: V4S[fi]; MAV rows: FMETA.p[fi]),
                        vp = stream_gate.py load() line 49 (v4 rows: V4S[vi]; MAV rows: VMETA.p[vi]);
                        row order = export.py's D = stream_gate.load(...) = the streams tables concatenated fold 1..5.
  fold_persons.json     export.py lines 41-42: {fold: sorted(set(fp) | set(vp))}, json.dumps(..., indent=0) (LF line ends: the bundle
                        holds the LF copy, sha256 43e77e8c...; the repository file Experiment/STACK-02/fold_persons.json is CRLF).
V4S / FMETA / VMETA are the module globals of lab_streams (the same objects export.py and stream_gate.py import from rel_sweep5).

    SAFE04_SCRATCH=<sp> FLAG_LOCAL=1 FLAG_IN=<...> FLAG_OUT=<sp>/out FF_STREAMS_TAG=streams|core_streams FOLDS=1,2,3,4,5 python ff_derive_stack02.py"""
import os, sys, json
from pathlib import Path

os.environ["SAFE04_NOMAIN"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import lab_streams as L

SP = Path(os.environ["SAFE04_SCRATCH"])
TAG = os.environ.get("FF_STREAMS_TAG", "streams")
FOLDS = [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",") if x]
D = pd.concat([pd.read_parquet(SP / f"out/safe04/{TAG}_fold{s}.parquet") for s in FOLDS], ignore_index=True)
v4m = D.data.values == "v4"
D["fp"] = np.where(v4m, L.V4S[np.clip(D.fi.values, 0, len(L.V4S) - 1)], L.FMETA.p.values[np.clip(D.fi.values, 0, len(L.FMETA) - 1)])
D["vp"] = np.where(v4m, L.V4S[np.clip(D.vi.values, 0, len(L.V4S) - 1)], L.VMETA.p.values[np.clip(D.vi.values, 0, len(L.VMETA) - 1)])
D[["fold", "data", "y", "fp", "vp"]].to_parquet(SP / "out/safe04/stack02_rows.parquet")
persons = {int(s): sorted(set(D.fp[D.fold == s]) | set(D.vp[D.fold == s])) for s in sorted(D.fold.unique())}
with open(SP / "out/safe04/fold_persons.json", "w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(persons, indent=0))
print(f"[ff_derive_stack02] rows {len(D)} | persons per fold {({s: len(v) for s, v in persons.items()})}", flush=True)
