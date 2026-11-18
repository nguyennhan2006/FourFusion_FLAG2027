print("[cell] l152 compare: this run vs the original FLAG_152 output (new cell, report only)", flush=True)
# New cell (FourFusion reproduction repo). MMS-LID-4017 runs in fp16 on two GPUs (not bit-reproducible). Its votes are only reported by
# LID-AUDIT-01 (the 58 dropped MAV files come from Whisper alone), but LID-AUDIT-01/run.py needs mms.parquet with one row per MAV file.
import json
import pandas as pd
_ref = {"rows": 40271, "flagged_files": 2528, "flagged_speakers": {"mav_v1": 62, "mav_v2": 80}, "read_errors": 0,
        "mms_parquet_sha256_original": "f9a436f81ce0521b7d8ab709ee9e7a48ac9546285a09c7f63853e40f9332f052"}
_Q = pd.read_parquet(OUT / "mms.parquet")
_fl = _Q.m_flag.fillna(False).astype(bool) if "m_flag" in _Q else pd.Series(False, index=_Q.index)
_got = {"rows": int(len(_Q)), "flagged_files": int(_fl.sum()), "flagged_speakers": _Q[_fl].groupby("src").spk.nunique().to_dict(),
        "read_errors": int(_Q.err.notna().sum()) if "err" in _Q else 0}
_cmp = {k: {"original": _ref[k], "this_run": v, "same": v == _ref[k]} for k, v in _got.items()}
(Path("/kaggle/working") / "ff_l152_compare.json").write_text(json.dumps(_cmp, indent=1))
print(json.dumps(_cmp, indent=1), flush=True)
