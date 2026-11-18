print("[cell] r08 verify: row tables byte-identical to ngnnguynchn/flag2027-ext-meta-p (new cell)", flush=True)
# New cell (FourFusion reproduction repo), after the original extraction cell. The four meta tables (+ the v1 speaker table) depend
# only on the file listing, so they must be byte-identical to the tables the whole system was keyed on. The arrays are GPU output
# (VGG-Face fc7 fp16, ECAPA fp32): only their shapes are checked here; their sha256 on the team machine is in stage.json for reference.
import hashlib, json, os
import numpy as np

EXPECTED_META = {
    "ext_v1_complete_face_meta.csv": "7b385161cfb9bcf998110b4f969c7268aaddb8c5567482d2888cc617f92601ae",
    "ext_v1_complete_voice_meta.csv": "3fac359f6a5b4fcab53032296c063fdf3adb73ba587a5107507c67982164635e",
    "ext_v2_complete_face_meta.csv": "9db02bfc517ddf5ea5e52d0c54d937d0887d87038ad969b24172c86a37534c23",
    "ext_v2_complete_voice_meta.csv": "1e22f7667044afe9bbcfa228d66ab4931f76a5df239ecc905e2bc31fa766378b",
    "ext_v1_complete_speakers.csv": "46da520173c773a66f9a9910e2d82d31a20b31db7d675a67532f87ea2b1dd4d9",
}
EXPECTED_SHAPES = {"ext_v1_complete_voice.npy": [19556, 192], "ext_v1_complete_face.npy": [7580, 4096],
                   "ext_v2_complete_voice.npy": [20715, 192], "ext_v2_complete_face.npy": [8953, 4096]}
_rep = {"meta": {}, "shapes": {}}
for _n, _d in EXPECTED_META.items():
    _p = OUTX / _n
    _h = hashlib.sha256(_p.read_bytes()).hexdigest() if _p.exists() else None
    _rep["meta"][_n] = {"sha256": _h, "ok": _h == _d}
for _n, _s in EXPECTED_SHAPES.items():
    _p = OUTX / _n
    _g = list(np.load(_p, mmap_mode="r").shape) if _p.exists() else None
    _rep["shapes"][_n] = {"shape": _g, "ok": _g == _s}
_rep["ok"] = all(v["ok"] for v in _rep["meta"].values()) and all(v["ok"] for v in _rep["shapes"].values())
(WORKDIR / "ff_r08_check.json").write_text(json.dumps(_rep, indent=1))
print(json.dumps(_rep, indent=1), flush=True)
if not _rep["ok"]:
    _msg = "r08: the MAV row tables differ from ngnnguynchn/flag2027-ext-meta-p (see ff_r08_check.json)"
    assert os.environ.get("FF_R08_ALLOW_MISMATCH") == "1", _msg + "; set FF_R08_ALLOW_MISMATCH=1 to keep them anyway"
    print("WARNING:", _msg, flush=True)
