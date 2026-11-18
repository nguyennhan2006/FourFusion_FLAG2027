print("[cell] l150 compare: this run vs the original FLAG_150 output (new cell, report only)", flush=True)
# New cell (FourFusion reproduction repo). Whisper-large-v3 runs in fp16 on two GPUs: a rerun is close to, not bit-identical with,
# the original. The 13.54 chain uses the STORED output of the original run (FF_DATA/02_intermediate/lid_whisper =
# Experiment/LIDSTRICT-01/out/{lid_whisper.json, mav.parquet}); this cell reports how close a rerun is.
import hashlib, json
import pandas as pd
_dg = lambda xs: hashlib.sha256("\n".join(sorted(map(str, xs))).encode()).hexdigest()
_ref = {"flagged": {"vox2": [29, "4c44d8f0db12c3c76f59f84eeb9fca7dd8a0226882065c94362b776c80035e8c"],
                    "vb": [43, "5eb443bc31a25304a2eb0cb6bbe549b81e1ab56036ed6f77e657fc5dbb0412d7"],
                    "mav_v1": [46, "c131034d0827c23685b13476414691094cc8abb6cf59495d8f18e97214831acb"],
                    "mav_v2": [71, "0af22f2d0df03ddfc73e7a56123f8cacaf00e28127527f24116f4243c193cf21"]},
        "clips": {"cache": 152031, "mav": 40271},
        "mav_w_files": [58, "76e7f50dfd9f6ac3a482d465abdab0da06210b2d2defb159f4de9481d200e220"]}
_J = json.loads((OUT / "lid_whisper.json").read_text())
_M = pd.read_parquet(OUT / "mav.parquet")
_W = _M[_M.w_flag.fillna(False).astype(bool)] if "w_flag" in _M else _M.iloc[:0]
_files = [f"{s[-2:]}/voices/{p}/{l}/{v}/{f}" for s, p, l, v, f in _W[["src", "spk", "lang", "video", "file"]].itertuples(index=False)]
_got = {"flagged": {k: [len(v), _dg(v)] for k, v in _J["flagged"].items()}, "clips": _J["clips"], "mav_w_files": [len(_files), _dg(_files)]}
_cmp = {"flagged": {k: {"original": v, "this_run": _got["flagged"].get(k), "same": _got["flagged"].get(k) == v} for k, v in _ref["flagged"].items()},
        "clips": {"original": _ref["clips"], "this_run": _got["clips"], "same": _got["clips"] == _ref["clips"]},
        "mav_w_files (Whisper-flagged MAV files = lid_strict.json mav_files)": {"original": _ref["mav_w_files"], "this_run": _got["mav_w_files"],
                                                                              "same": _got["mav_w_files"] == _ref["mav_w_files"]}}
(Path("/kaggle/working") / "ff_l150_compare.json").write_text(json.dumps(_cmp, indent=1))
print(json.dumps(_cmp, indent=1), flush=True)
