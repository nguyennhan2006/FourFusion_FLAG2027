print("[cell] r31: rebuild the MAV row tables from the extracted release and compare them byte for byte (CPU)", flush=True)
# New cell (FourFusion reproduction repo). Optional evidence stage: check_rows.py (this folder) lists the MAV-Celeb release with the
# code that really ran in FLAG_08 (listing A) and with the EXTRAW cell of FLAG_31 (listing B) and compares the four meta tables with the
# sha256 of ngnnguynchn/flag2027-ext-meta-p. A few minutes on a local disk; on Kaggle's input file system the listing alone is slow
# (the original FLAG_08 log: about 1.6 h to list v1). Run it before the 6-10 GPU hours of r08_mav_rows.
import os, subprocess, sys
from pathlib import Path

STAGE_DIR = Path(os.environ.get("FF_STAGE_DIR", "."))
MAV = Path("/kaggle/input/datasets/chnnhnnguyn/external-data-mavceleb")
OUT = Path("/kaggle/working/r31")
cmd = [sys.executable, "-u", str(STAGE_DIR / "check_rows.py"), "--mav-root", str(MAV), "--out", str(OUT),
       "--only", os.environ.get("R31_LISTINGS", "A,B")]
if os.environ.get("FF_ALLOW_NONPOSIX") == "1":
    cmd.append("--allow-nonposix")
print("run:", " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
assert rc == 0 or os.environ.get("FF_SMOKE") == "1", "r31: the rebuilt MAV row tables differ from ngnnguynchn/flag2027-ext-meta-p (see r31/r31_report.json)"
print("r31: done (exit code", rc, ")", flush=True)
