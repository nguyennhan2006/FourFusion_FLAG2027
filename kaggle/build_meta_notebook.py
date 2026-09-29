"""Generate FLAG_09_ext_meta.ipynb: CPU-only pass over MAV-Celeb v1/v2/v3 zips for what NB-5 did not keep.

NB-5 (FLAG_08) extracts features and deletes each zip, so two things never left Kaggle:
  * the zips' own metadata files (identity list, GENDER) -> needed for gender-style cells in EXT-02 (PLAN_V4 A2)
  * per-utterance pitch (F0) -> ATTR-01 falsification test on v1/v2 (PLAN_V4 B)
No GPU, no model weights. Output is a few MB.
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def embed_lib(*names):
    parts = ["# Writes the helper modules next to this notebook (no .py upload needed).", "import base64"]
    for n in names:
        b = base64.b64encode((HERE / n).read_text(encoding="utf-8").encode()).decode()
        b = "\n".join(b[i:i + 100] for i in range(0, len(b), 100))
        parts.append(f"open({n!r}, 'wb').write(base64.b64decode(''.join('''\n{b}\n'''.split())))")
    parts.append(f"print('written:', {list(names)!r})")
    return "\n".join(parts) + "\n"


def nb(cells, path):
    d = {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                    **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
         "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11"}},
         "nbformat": 4, "nbformat_minor": 5}
    (HERE / path).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", path, len(cells), "cells")


INTRO = """
# FLAG 2027 — NB-6: MAV-Celeb v1/v2/v3 metadata + pitch (CPU only)

NB-5 extracted features from each external zip and deleted it. This notebook downloads the zips again and keeps
only what NB-5 threw away:

| output | used by |
|---|---|
| `ext_meta/<source>/…` + `ext_<source>_zip_listing.csv` — every non-media file of the zip (identity list, gender, readme) | EXT-02 gender-style cells (PLAN_V4 A2) |
| `ext_<source>_voice_f0.csv` — log-F0 median / IQR / voiced fraction per wav (YIN) | ATTR-01 on v1/v2 (PLAN_V4 B) |

**Settings:** Accelerator **None (CPU)**, **Internet ON**. No input dataset needed.
Expected: download dominates (~25 GB in total, one zip at a time, each deleted after use); F0 ~15–25 min per large zip.
"""

ENV = '''
import os, sys
from pathlib import Path
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
WORKDIR = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
print("LOCAL" if LOCAL else "KAGGLE", "| working", WORKDIR)
'''

RUN = '''
import shutil, time
import pandas as pd
import flag_extract as X
OUT = WORKDIR / "feats_ext_meta"; OUT.mkdir(exist_ok=True)
TMP = Path(os.environ.get("FLAG_TMP", "/tmp/flag_ext")); TMP.mkdir(parents=True, exist_ok=True)
print("free disk at", TMP, round(shutil.disk_usage(TMP).free / 1e9, 1), "GB")
# Drive "Quota exceeded": paste the id of your own copy here; it is tried first, the organiser's id second.
DRIVE_IDS = {}
SOURCES = {"mini_v3": None} if LOCAL else {k: ([DRIVE_IDS[k], fid] if k in DRIVE_IDS else fid, gb)
                                           for k, (fid, gb) in X.EXT_SOURCES.items()}
for name, spec in SOURCES.items():
    if (OUT / f"ext_{name}_voice_f0.csv").exists():
        print("skip (exists)", name); continue
    t0 = time.time()
    try:
        zp = Path(os.environ["FLAG_EXT_MINI"]) if LOCAL else X.fetch_zip(spec[0], TMP / f"{name}.zip", min_free_gb=spec[1] * 1.2 + 2)
        X.extract_ext_meta(zp, name, OUT)
    except (AssertionError, RuntimeError, OSError) as e:        # Drive quota / disk: skip this source, keep going
        print(f"SKIPPED {name}: {type(e).__name__}: {e}")
        continue
    if not LOCAL and (TMP / f"{name}.zip").exists():
        (TMP / f"{name}.zip").unlink()                         # only after a successful pass
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
'''

CHECK = '''
# What came out: metadata files per source (look for gender / identity lists) and F0 sanity.
for name in SOURCES:
    lst = OUT / f"ext_{name}_zip_listing.csv"
    if not lst.exists():
        print(name, ": nothing"); continue
    L_ = pd.read_csv(lst)
    print(f"\\n== {name}: {len(L_)} non-media members")
    print(L_.sort_values("size", ascending=False).head(15).to_string(index=False))
    for p in sorted((OUT / "ext_meta" / name).rglob("*"))[:10]:
        if p.is_file() and p.suffix.lower() in (".csv", ".txt", ".tsv", ".json", ".md") and p.stat().st_size < 2e5:
            print(f"--- {p.relative_to(OUT)} (first lines)")
            print("".join(open(p, encoding="utf-8", errors="replace").readlines()[:5]))
    f = OUT / f"ext_{name}_voice_f0.csv"
    if f.exists():
        F0 = pd.read_csv(f)
        print(f"F0: {len(F0)} wav, valid {F0.log_f0_median.notna().mean():.1%}, median {float(F0.log_f0_median.median()):.3f} (log Hz)")
print("\\nDownload: feats_ext_meta/")
'''

nb([("markdown", INTRO), ("code", ENV), ("code", embed_lib("flag_extract.py")),
    ("markdown", "## 1. Download each zip, keep metadata + F0, delete the zip"), ("code", RUN),
    ("markdown", "## 2. What came out"), ("code", CHECK)], "FLAG_09_ext_meta.ipynb")
