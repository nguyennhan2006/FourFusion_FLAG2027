print("[cell] raw_mavceleb: locate and check the two original MAV-Celeb zips", flush=True)
# New cell (FourFusion reproduction repo). Input = the organisers' MAV-Celeb releases mavceleb_v1_complete-001.zip and
# mavceleb_v2_complete-002.zip (artifact raw/mavceleb_v1_v2_zips). Output = exactly what Kaggle showed for the public dataset
# chnnhnnguyn/external-data-mavceleb, which is the two zips auto-extracted into folders named after them:
#   mavceleb_v1_complete-001/{meta_v1.csv, v1/faces/..., v1/voices/...}   mavceleb_v2_complete-002/v2/{faces,voices}/...
# The two parent folders are REQUIRED as they are: the stage r08_mav_rows locates v1 / v2 by the '/v1/' and '/v2/' folders, and the
# RFS LoRA C1 run (stage t163l_orig) stops at the first v1 / v2 folder it meets (kaggle/build_flag163.py:93). No zip may stay inside
# the dataset folder: flag_extract.find_attached_zip (r08) would read a zip before the extracted folder and change the row tables.
import hashlib, json, os, shutil, time, zipfile
from pathlib import Path

OUT = Path("/kaggle/working")
DS = OUT / "external-data-mavceleb"                  # becomes the export datasets/chnnhnnguyn/external-data-mavceleb
STATE = OUT / "_ff_raw_mavceleb"                     # resume markers + report, outside the dataset folder
SMOKE = os.environ.get("FF_SMOKE") == "1"            # smoke: no full sha256, only the first members of each zip
SKIP_SHA = SMOKE or os.environ.get("RAW_MAV_SKIP_SHA256") == "1"
ZIPS = {   # zip name: (bytes, sha256 measured 18/11/2026 on external_data/mavceleb/ of the team machine, expected file counts)
    "mavceleb_v1_complete-001.zip": (12245623417, "2082a9ee5f7d3634f687d682e98cd223606a5b01debc90c3e7ab566a9f86512b",
                                     {"jpg": 302492, "wav": 19556, "csv": 1, "files": 322049}),
    "mavceleb_v2_complete-002.zip": (13477476090, "daff513527d7cea698243594a18795cab6d47aea64d64c3c74035154abd6246c",
                                     {"jpg": 371626, "wav": 20715, "csv": 0, "files": 392341}),
}
META_V1_SHA256 = "3b311af585dd3e02050c551bcc889cbc47f742f7893e6ac48281d0ca85a8da53"   # mavceleb_v1_complete-001/meta_v1.csv


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


STATE.mkdir(parents=True, exist_ok=True)
REP = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": SMOKE, "zips": {}}
SRC = {}
for name, (size, digest, _) in ZIPS.items():
    hits = sorted(p for p in Path("/kaggle/input").rglob(name) if p.is_file())
    assert hits, f"attach {name} (artifact raw/mavceleb_v1_v2_zips = FF_DATA/00_raw/mavceleb_v1_v2_zips/)"
    SRC[name] = hits[0]
    r = {"path": str(hits[0]), "bytes": hits[0].stat().st_size, "bytes_ok": hits[0].stat().st_size == size}
    if not SKIP_SHA:
        t = time.time()
        r["sha256"] = sha256_file(hits[0])
        r["sha256_ok"] = r["sha256"] == digest
        r["sha256_seconds"] = round(time.time() - t)
    REP["zips"][name] = r
    print(name, r, flush=True)
    assert r["bytes_ok"] and r.get("sha256_ok", True), f"{name} is not the MAV-Celeb release the system used"
