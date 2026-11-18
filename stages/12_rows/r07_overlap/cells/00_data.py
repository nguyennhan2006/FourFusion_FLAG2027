print("[cell] r07: unpack the organiser feature csv files of the train set into a scratch folder (no dev file unless R07_WITH_DEV=1)", flush=True)
# New cell (FourFusion reproduction repo). overlap.py reads the organiser CSV features through flag_v2.FeatureStore, which finds
# train_English_{faces,voices}.csv, train_English.txt and meta_file_train_set.csv by name anywhere under its data root. The original
# run (27/09/2026, team machine) used kaggle_upload/ = the same files unpacked from train_set.zip (sha256 identical, checked 18/11).
# Default for the 13.54 chain: --train-only (the dev-based secondary rule is skipped; no dev file is read). R07_WITH_DEV=1 reruns the
# original rule, which also unpacks the dev feature csv files from dev_set.zip.
# Everything this stage writes except r07_check.json goes to a scratch folder OUTSIDE /kaggle/working, deleted by the next cell:
# the organisers' files must not end up in a saved run output, and overlap.csv (one row per MAV person with its nearest v4 train
# identity) is an identity-linkage table whose use is a pending team decision (docs/COMPLIANCE_NOTES.md).
import os, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

STAGE_DIR = Path(os.environ.get("FF_STAGE_DIR", "."))
RAW = Path("/kaggle/input/datasets/ngnnguynchn/flag2027-raw-v4-p")
R08 = Path("/kaggle/input/notebooks/ff/r08_mav_rows")
WORK = Path("/kaggle/working")
SCRATCH = Path(tempfile.mkdtemp(prefix="ff_r07_"))
DATA = SCRATCH / "r07_data"
WITH_DEV = os.environ.get("R07_WITH_DEV") == "1"
(DATA / "train").mkdir(parents=True)
with zipfile.ZipFile(RAW / "train_set.zip") as z:
    for m in ("train_set/train_set/train_English.txt", "train_set/features/faces/train_English_faces.csv",
              "train_set/features/voices/train_English_voices.csv"):
        (DATA / "train" / Path(m).name).write_bytes(z.read(m))
shutil.copy2(RAW / "meta_file_train_set.csv", DATA / "train" / "meta_file_train_set.csv")
if WITH_DEV:
    with zipfile.ZipFile(RAW / "dev_set.zip") as z:
        for m in z.namelist():                        # flag_lib.load_dev: <Lang>_test_{faces,voices}.csv + <Lang>_test.txt
            if ("/features/" in m and m.endswith(("_faces.csv", "_voices.csv"))) or m.endswith("_test.txt"):
                q = DATA / "dev" / m
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_bytes(z.read(m))
hits = sorted(R08.rglob("ext_v1_complete_voice.npy"))
assert hits, "attach the output of stage r08_mav_rows (feats_ext/ext_v1_complete_voice.npy ...)"
FEATS_EXT = hits[0].parent
print("scratch", SCRATCH, "| data", sorted(p.relative_to(DATA).as_posix() for p in DATA.rglob("*") if p.is_file()),
      "| feats_ext", FEATS_EXT, flush=True)
