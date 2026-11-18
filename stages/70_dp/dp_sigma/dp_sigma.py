"""Stage dp_sigma (new code, FourFusion 13.54 reproduction repo).

The two scale constants of the final English blend (stage final_blend):

    English value = ours / SIG[protocol] - r * z_DP          (r = 1.0; SIG = 5.0983 no_gender, 3.8376 gender)

SIG[protocol] is the standard deviation of the base score U = F1s' (out of fold) + RFX (out of fold) over the English pairs of the
HELD-OUT TRAIN persons of the SAFE-04 validation (cell "v4 ng" for the no_gender file, "v4 g" for the gender file), averaged over the
three RFX seed replicates c0r / c0s / c0t (FLAG_213 p1 / p2, FLAG_216) and rounded to 4 decimals. No Evaluation data is involved.

In the original work the two numbers were typed by hand from the log of the Kaggle run
phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4 (Experiment/FUSENET-01/run.py, FUSENET_MODE=dumpv4, log line
"dumpv4 <cell> <replicate>: ... sd_all <numpy std, ddof 0>") and passed as XJF_SIG="5.0983,3.8376". This script recomputes them from
the table that run wrote (xjfdump_v4.parquet, 12 896 pairs) with the same statistic, in the same row order:

    std = g["U_<rep>"].values.astype(float).std()      (numpy, ddof = 0; rows of one cell in table order, as V.groupby("cell"))
    SIG[protocol] = mean of the three replicates, written with 4 decimals ("%.4f")

    python dp_sigma.py --dump xjfdump_v4.parquet --out sigma.json [--expect 5.0983,3.8376] [--strict]

sigma.json["xjf_sig"] is the exact string final_blend / blend.py parse (same parse as the XJF_SIG variable: float() of each part).
ddof = 1 would give 5.0987 / 3.8379, which are NOT the constants of the submission."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = {"no_gender": "v4 ng", "gender": "v4 g"}          # zip protocol -> SAFE-04 cell (English pairs of held-out v4 train persons)
REPLICATES = ["c0r", "c0s", "c0t"]                         # RFX c5rl seed replicates 4-6, 7-9, 10-12 (FLAG_213 p1, p2, FLAG_216)
SOURCE_SHA256 = "9f0a5af468d1f973e1ff33264ff10239cf21f1813a0f370740e8c2d8e02f7a09"   # xjfdump_v4.parquet of the original run
VARIANT_1354 = "5.0983,3.8376"


def sha256(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="DP blend scale constants (SIG) from the held-out-train table xjfdump_v4.parquet")
    ap.add_argument("--dump", required=True, help="xjfdump_v4.parquet written by FLAG_219 (FUSENET_MODE=dumpv4)")
    ap.add_argument("--out", required=True, help="sigma.json to write")
    ap.add_argument("--expect", default=None, help=f"expected 'no_gender,gender' string, e.g. {VARIANT_1354} (variant 13.54)")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the recomputed string differs from --expect")
    a = ap.parse_args()

    p = Path(a.dump)
    src_sha = sha256(p)
    d = pd.read_parquet(p)
    reps = sorted(c[2:] for c in d.columns if c.startswith("U_"))
    assert reps == REPLICATES, f"replicate columns {reps} != {REPLICATES}"
    cells = d["cell"].astype(str).to_numpy(dtype=str)              # plain numpy strings (pandas 3 string columns index badly)
    assert set(CELLS.values()) <= set(cells), sorted(set(cells))
    per, full, npairs, npos = {}, {}, {}, {}
    for prot, cell in CELLS.items():
        g = d[cells == cell]                                          # rows of one cell, table order (= V.groupby("cell") of FLAG_219)
        y = g["y"].values.astype(bool)
        per[prot] = {r: float(g[f"U_{r}"].values.astype(float).std()) for r in REPLICATES}
        full[prot] = float(np.mean([per[prot][r] for r in REPLICATES]))
        npairs[prot], npos[prot] = int(len(g)), int(y.sum())
        print(f"{prot:9s} (cell '{cell}', {len(g)} pairs, {int(y.sum())} same-person): "
              + "  ".join(f"sd {r} {per[prot][r]:.6f}" for r in REPLICATES) + f"  -> mean {full[prot]:.6f}", flush=True)
    xjf_sig = ",".join(f"{full[k]:.4f}" for k in ("no_gender", "gender"))
    ok = None if a.expect is None else (xjf_sig == a.expect)
    res = {
        "xjf_sig": xjf_sig,
        "rounded": dict(zip(("no_gender", "gender"), map(float, xjf_sig.split(",")))),
        "full": full,
        "per_replicate": per,
        "n_pairs": npairs,
        "n_same_person": npos,
        "statistic": "numpy std (ddof=0) of U_<replicate> = F1s'|oof + R_<replicate>|oof over the rows of one cell, mean of 3 replicates, %.4f",
        "cells": CELLS,
        "replicates": REPLICATES,
        "source": {"file": p.name, "sha256": src_sha, "sha256_of_original_run": SOURCE_SHA256, "same_as_original": src_sha == SOURCE_SHA256,
                   "rows": int(len(d))},
        "expected": a.expect,
        "matches_expected": ok,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"SIG = {xjf_sig}   (source sha256 {src_sha[:16]}..., {'same as' if src_sha == SOURCE_SHA256 else 'DIFFERENT from'} the original table)",
          flush=True)
    if a.expect is not None:
        print("OK: equals the expected constants" if ok else f"MISMATCH: expected {a.expect}", flush=True)
        if a.strict and not ok:
            sys.exit(1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
