print("[cell] raw_mavceleb_check: check a SAVED copy of the extracted MAV-Celeb v1 / v2 dataset (counts, layout, meta_v1.csv, no zip)", flush=True)
# New cell (FourFusion reproduction repo). Raw data is always taken from a saved copy (configs/modes.json always_saved_classes), so
# stage raw_mavceleb (which checks its own fresh extraction) does not run when FF_DATA already holds the dataset. This stage checks
# the copy that every MAV consumer will mount as datasets/chnnhnnguyn/external-data-mavceleb: an incomplete copy (for example the
# team package folder, which holds only mavceleb_v1_complete-001/{meta_v1.csv, v1/faces}) must stop the run here instead of
# silently changing the row tables (r08), the LoRA C1 data (t163l_orig) or the language-ID lists (l150, l152).
# Same expectations as raw_mavceleb/cells/02_check.py (central directories of the two organisers' zips, read 18/11/2026).
import hashlib, json, os, time
from pathlib import Path

IN = Path("/kaggle/input/datasets/chnnhnnguyn/external-data-mavceleb")
OUT = Path("/kaggle/working")
SMOKE = os.environ.get("FF_SMOKE") == "1"            # smoke: layout, meta_v1.csv and stray zips only; no full file count
EXPECTED = {   # extracted folder: (version folder, file counts of the release)
    "mavceleb_v1_complete-001": ("v1", {"jpg": 302492, "wav": 19556, "csv": 1, "files": 322049}),
    "mavceleb_v2_complete-002": ("v2", {"jpg": 371626, "wav": 20715, "csv": 0, "files": 392341}),
}
META_V1_SHA256 = "3b311af585dd3e02050c551bcc889cbc47f742f7893e6ac48281d0ca85a8da53"   # mavceleb_v1_complete-001/meta_v1.csv


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


rep, bad = {"checked": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": SMOKE, "mount": str(IN), "releases": {}}, []
assert IN.is_dir(), f"{IN} is not mounted (configs/data_map.json: FF_DATA/00_raw/mavceleb_v1_v2)"
extra = sorted(p.name for p in IN.iterdir() if p.name not in EXPECTED)
rep["unexpected_top_level"] = extra                  # reported; a zips/ folder is excluded from the mount by data_map 'exclude'
for stem, (ver, cnt) in EXPECTED.items():
    root = IN / stem
    r = {"present": root.is_dir(), "layout": {d: (root / ver / d).is_dir() for d in ("faces", "voices")}}
    if not r["present"] or not all(r["layout"].values()):
        bad.append(f"{stem}: missing or incomplete ({ver}/faces and {ver}/voices must sit directly under {root})")
    elif not SMOKE:
        t = time.time()
        got = {"jpg": 0, "wav": 0, "csv": 0, "files": 0}
        for dp, dn, fn in os.walk(root):
            for f in fn:
                got["files"] += 1
                ext = f.rsplit(".", 1)[-1].lower()
                if ext in got:
                    got[ext] += 1
        r.update(counts=got, expected=cnt, counts_ok=got == cnt, walk_seconds=round(time.time() - t))
        if got != cnt:
            bad.append(f"{stem}: file counts {got} differ from the release {cnt}")
    rep["releases"][stem] = r
    print(stem, r, flush=True)
meta = IN / "mavceleb_v1_complete-001/meta_v1.csv"
rep["meta_v1_sha256"] = sha256_file(meta) if meta.is_file() else None
if rep["meta_v1_sha256"] != META_V1_SHA256:
    bad.append("mavceleb_v1_complete-001/meta_v1.csv missing or different")
stray = []
for dp, dn, fn in os.walk(IN):
    dn[:] = [d for d in dn if d not in ("faces", "voices")]       # media folders hold no zip; keeps the walk short
    stray += [str(Path(dp) / f) for f in fn if f.lower().endswith(".zip")]
rep["zips_inside"] = stray
if stray:
    bad.append(f"zip(s) inside the dataset folder (r08 would read them instead of the folders): {stray[:3]}")
rep["ok"] = not bad
rep["problems"] = bad
(OUT / "raw_mavceleb_check.json").write_text(json.dumps(rep, indent=1))
assert not bad, "raw_mavceleb_check: the saved MAV-Celeb copy is not the one the system read:\n  " + "\n  ".join(bad)
print("raw_mavceleb_check: OK (saved MAV-Celeb v1 / v2 copy complete, Kaggle layout)", flush=True)
