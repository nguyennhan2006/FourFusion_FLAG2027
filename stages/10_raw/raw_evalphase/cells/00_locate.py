print("[cell] raw_evalphase: locate and check the organisers' Evaluation zip", flush=True)
# New cell (FourFusion reproduction repo). Input = the organisers' evaluation_phase.zip as released on 11/11/2026
# (artifact raw/btc_evaluation_original = FF_DATA/00_raw/btc_evaluation/evaluation_phase.zip). Only that exact file name is used,
# so a copy of eval_set.zip kept in the same folder is ignored.
import hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path

OUT = Path("/kaggle/working")
STAGE_DIR = Path(os.environ.get("FF_STAGE_DIR", "."))         # repo folder of this stage (build_evalphase.py, expected/)
BTC_NAME, BTC_BYTES = "evaluation_phase.zip", 472428761
BTC_SHA256 = "c9efa2100dc54188fbd9739970a98c1cd966985d9c6f7d46148e5a3af96fc9f9"
EVAL_SET_SHA256_SUBMITTED = "a70762a3fb52af6bbf52c14a86c79a4d074ec319b78d6ffedcdf195d71f9381e"   # zip bytes carry the build time


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


hits = sorted(p for p in Path("/kaggle/input").rglob(BTC_NAME) if p.is_file())
assert hits, f"attach the organisers' {BTC_NAME} (artifact raw/btc_evaluation_original)"
BTC = hits[0]
REP = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "btc_zip": str(BTC), "btc_bytes": BTC.stat().st_size,
       "btc_sha256": sha256_file(BTC)}
REP["btc_ok"] = REP["btc_bytes"] == BTC_BYTES and REP["btc_sha256"] == BTC_SHA256
print(json.dumps(REP, indent=1), flush=True)
assert REP["btc_ok"], "this is not the organisers' evaluation_phase.zip of 11/11/2026 (size / sha256 differ)"
assert (STAGE_DIR / "build_evalphase.py").is_file(), f"FF_STAGE_DIR must point at stages/10_raw/raw_evalphase (now {STAGE_DIR})"
