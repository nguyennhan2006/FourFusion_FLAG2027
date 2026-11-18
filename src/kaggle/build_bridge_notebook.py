"""Generate FLAG_87_bridge.ipynb (14/10): copy the FACELORA-03 inputs that live only in onnhqunhh's PRIVATE notebook outputs into a
PRIVATE dataset of the same account, server side (no download to the local machine), so the adapters can run on other accounts:
  vox_frames_png.zip (FLAG_45), frames/ (FLAG_63: frames_manifest.json + vb/*.zip + vox2b/*.zip), feats_facelora3_b/facelora_ckpt.pt
  (FLAG_65 B3, part 1). The frame zips are renamed *.zipdata so Kaggle does not unpack them (the FLAG_65 notebook reads both names).
Runs on CPU as the notebook's owner; kagglehub authenticates inside a Kaggle notebook. Then, locally:
  python kaggle/kapi.py privatize onnhqunhh/<slug>      (private + every env account as reader)
    python build_bridge_notebook.py
"""
import build_eval_notebooks as B12

SLUG = "flag2027-lora-frames-p"

CODE = r'''
import os, sys, json, time, shutil
from pathlib import Path
IN = Path("/kaggle/input")
HANDLE = f"onnhqunhh/__SLUG__"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
import kagglehub
log("kagglehub", kagglehub.__version__)
'''.replace("__SLUG__", SLUG)

STAGE = r'''
# stage = symlinks with the published names (no copy); every source must be found once
def find(name, parent=None):
    hits = sorted(p for p in IN.rglob(name) if parent is None or p.parent.name == parent)
    assert hits, f"{name} ({parent}) not found: attach the FLAG_45 / FLAG_63 / FLAG_65-b outputs"
    return hits[0]


stage = Path("/tmp/stage"); shutil.rmtree(stage, ignore_errors=True)
z37, man = find("vox_frames_png.zip"), find("frames_manifest.json")
ck = find("facelora_ckpt.pt", "feats_facelora3_b")
links = {stage / "vox_frames_png.zipdata": z37, stage / "frames/frames_manifest.json": man,
         stage / "feats_facelora3_b/facelora_ckpt.pt": ck}
for part in ("vb", "vox2b"):
    for zf in sorted((man.parent / part).glob("*.zip")):
        links[stage / f"frames/{part}/{zf.stem}.zipdata"] = zf
for dst, src in links.items():
    dst.parent.mkdir(parents=True, exist_ok=True); os.symlink(src, dst)
m = json.loads(man.read_text())
n = {p: len(list((stage / f"frames/{p}").glob("*.zipdata"))) for p in ("vb", "vox2b")}
gb = sum(Path(s).stat().st_size for s in links.values()) / 2 ** 30
log(f"staged {len(links)} files, {gb:.2f} GB | chunk zips {n} vs manifest {{'vb': {m['parts']['vb']['zips']}, 'vox2b': {m['parts']['vox2b']['zips']}}}")
assert n["vb"] == m["parts"]["vb"]["zips"] and n["vox2b"] == m["parts"]["vox2b"]["zips"], "missing frame zips"
'''

UPLOAD = r'''
# upload as a NEW private dataset of this account (kagglehub: private by default); privatize locally afterwards
t0 = time.time()
kagglehub.dataset_upload(HANDLE, str(stage), version_notes="FACELORA-03 inputs: FLAG_45 + FLAG_63 frames, FLAG_65-b checkpoint (14/10)")
log(f"uploaded to {HANDLE} [{time.time() - t0:.0f}s]")
'''


if __name__ == "__main__":
    B12.nb([("markdown", "# FLAG 2027 — NB-87: bridge, FACELORA-03 frames + B3 checkpoint -> private dataset (CPU)"),
            ("code", CODE), ("code", STAGE), ("code", UPLOAD)], "FLAG_87_bridge.ipynb")
