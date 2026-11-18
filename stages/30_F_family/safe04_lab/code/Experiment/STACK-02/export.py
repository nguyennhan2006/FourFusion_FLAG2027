r"""STACK-02 step 1: one row-aligned table of every candidate column + the face / voice person of every row, so the
runner does not import the SAFE-04 arrays. Same alignment checks as HEADROOM-01 (y / vi per fold). No score is computed.
    python export.py
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
os.environ["SAFE04_NOMAIN"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "SAFE-04"))
from rel_sweep5 import V4S, FMETA, make_fold
from stream_gate import load, S6, CO
from lab_fuse import SP
from pools import TABLES, POOLS

D = load("attrconf01")
for tag in TABLES:
    cols = sorted({c for f in POOLS.values() for c in f["cols"] if f["table"] == tag})
    parts = []
    for s in sorted(D.fold.unique()):
        p = pd.read_parquet(SP / f"out/safe04/{tag}_fold{s}.parquet")
        base = D[D.fold == s]
        assert len(p) == len(base), (tag, s)
        if {"y", "vi"} <= set(p.columns):
            assert (p.y.values == base.y.values).all() and (p.vi.values == base.vi.values).all(), (tag, s)
        parts.append(p[cols].reset_index(drop=True))
    add = pd.concat(parts, ignore_index=True)
    for c in cols:
        if c not in D.columns:
            D[c] = add[c].values
        else:
            assert np.allclose(D[c].values, add[c].values, equal_nan=True), (tag, c)
v4m = D.data.values == "v4"
D["fp"] = np.where(v4m, V4S[np.clip(D.fi.values, 0, len(V4S) - 1)], FMETA.p.values[np.clip(D.fi.values, 0, len(FMETA) - 1)])
need = S6 + CO + [c for f in POOLS.values() for c in f["cols"]]
for c in need:
    assert np.isfinite(D[c].values).all(), f"{c} not fully covered"
keep = ["fold", "data", "protocol", "cell", "y", "fi", "vi", "fp", "vp", "agegap_c"] + list(dict.fromkeys(need))
D[keep].to_parquet(SP / "out/safe04/stack02_rows.parquet")
persons = {int(s): sorted(set(D.fp[D.fold == s]) | set(D.vp[D.fold == s])) for s in sorted(D.fold.unique())}
(HERE / "fold_persons.json").write_text(json.dumps(persons, indent=0))
print(f"rows {len(D)} | columns {len(keep)} | candidate columns {sum(len(f['cols']) for f in POOLS.values())} in {len(POOLS)} families")
print("persons per fold (both pair sides):", {s: len(v) for s, v in persons.items()})
