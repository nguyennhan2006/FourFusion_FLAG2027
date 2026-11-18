print("[cell] r08 guard: MAV input layout and POSIX path order (new cell, changes nothing)", flush=True)
# New cell (FourFusion reproduction repo), inserted before the original extraction cell. The row tables ext_*_meta.csv are fully
# determined by the file listing, so three things must hold exactly as in the original Kaggle run:
#  1. POSIX path order: flag_extract.DirArchive lists files with sorted(Path.rglob("*")); pathlib sorts case-SENSITIVELY on Linux and
#     case-insensitively on Windows -> another voice row order (the original ran on Kaggle, Linux).
#  2. no zip of a MAV release under /kaggle/input: find_attached_source() returns an attached zip before the extracted folder; a zip
#     gives member names with the 'v1/' prefix and zip order instead of the sorted folder listing -> other path column and row order.
#  3. exactly the two extracted roots .../mavceleb_v1_complete-001/v1 and .../mavceleb_v2_complete-002/v2, and no '/v1/' or '/v2/'
#     segment above the mount point (find_attached_source matches the root by the '/v1/' / '/v2/' text of its path).
import os, re
from pathlib import Path

_IN = Path("/kaggle/input")
assert os.name == "posix" or os.environ.get("FF_ALLOW_NONPOSIX") == "1", (
    "r08_mav_rows must run on Linux (Kaggle / Docker): pathlib sorts paths case-insensitively on Windows, which changes the voice "
    "row order of ext_*_voice_meta.csv. Set FF_ALLOW_NONPOSIX=1 only for a smoke test.")
_zips, _roots = [], []
for _dp, _dn, _fn in os.walk(_IN, followlinks=True):
    if "faces" in _dn and "voices" in _dn:
        _roots.append(Path(_dp))
    _dn[:] = [d for d in _dn if d not in ("faces", "voices")]          # do not walk the 700 000 media files
    _zips += [Path(_dp) / f for f in _fn if f.lower().endswith(".zip")
              and re.search(r"v1_complete|v2_complete|v3_tran|v3_train", f.lower().replace("-", "_"))]
assert not _zips, (f"zip(s) of a MAV-Celeb release under /kaggle/input: {[str(z) for z in _zips]}. Mount only the EXTRACTED dataset "
                   "datasets/chnnhnnguyn/external-data-mavceleb (stage raw_mavceleb), never the zips.")
_roots = sorted(_roots)
print("MAV roots:", [str(r) for r in _roots], flush=True)
assert len(_roots) == 2 and _roots[0].as_posix().endswith("mavceleb_v1_complete-001/v1") \
    and _roots[1].as_posix().endswith("mavceleb_v2_complete-002/v2"), "expected exactly .../mavceleb_v1_complete-001/v1 and .../mavceleb_v2_complete-002/v2"
for _p in (_IN, _IN.resolve()):
    assert not re.search(r"/v[12]/", _p.as_posix().lower().replace("-", "_") + "/"), \
        f"the mount point path {_p} contains a /v1/ or /v2/ segment: find_attached_source would match the wrong root"
print("guard OK", flush=True)
