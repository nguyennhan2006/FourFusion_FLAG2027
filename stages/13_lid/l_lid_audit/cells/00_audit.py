print("[cell] l_lid_audit: LID-AUDIT-01 votes per MAV voice file -> mav_w.json (the 58 Whisper-flagged files)", flush=True)
# New cell (FourFusion reproduction repo). Runs lid_audit_run.py (= Experiment/LID-AUDIT-01/run.py, paths as environment variables)
# on FLAG_150's mav.parquet (Whisper + VoxLingua votes) and FLAG_152's mms.parquet (MMS votes, reported only). mav_w.json depends on
# the Whisper votes alone: with the stored FLAG_150 output it is byte-identical (CRLF line ends, as written on the team's Windows
# machine) to the file lid_strict.json was built from.
import hashlib, json, os, subprocess, sys
from pathlib import Path

STAGE_DIR = Path(os.environ.get("FF_STAGE_DIR", "."))
L150 = Path("/kaggle/input/notebooks/duongphihoang/flag-2027-nb-150-lidwhisper")
L152 = Path("/kaggle/input/notebooks/phantrnanna/flag-2027-nb-152-mmslid")
OUT = Path("/kaggle/working/lid_audit")


def guard_action():
    """stage.json guard_policy.on_mismatch[FF_MODE] ('warn' | 'fail'); FF_SMOKE=1 or FF_ALLOW_MISMATCH=1 -> 'warn'.
    In FF_MODE=train FLAG_150 may have been rerun (Whisper on GPU is not bit-identical)."""
    if os.environ.get("FF_SMOKE") == "1" or os.environ.get("FF_ALLOW_MISMATCH") == "1":
        return "warn", "FF_SMOKE=1 or FF_ALLOW_MISMATCH=1"
    mode = os.environ.get("FF_MODE", "")
    try:
        pol = json.loads((STAGE_DIR / "stage.json").read_text(encoding="utf-8"))["guard_policy"]["on_mismatch"]
    except Exception:
        pol = {"train": "warn"}
    return pol.get(mode, "fail"), f"stage.json guard_policy.on_mismatch[{mode or 'FF_MODE unset'}]"


def one(root, name, why):
    hits = sorted(p for p in root.rglob(name) if p.is_file())
    assert hits, f"{name} not found under {root} ({why})"
    return hits[0]


env = dict(os.environ, FF_MAV_PARQUET=str(one(L150, "mav.parquet", "FLAG_150 output lidw/")),
           FF_MMS_PARQUET=str(one(L152, "mms.parquet", "FLAG_152 output mmslid/")), FF_OUT=str(OUT))
cmd = [sys.executable, "-u", str(STAGE_DIR / "lid_audit_run.py")]
print("run:", " ".join(cmd), "| inputs", env["FF_MAV_PARQUET"], env["FF_MMS_PARQUET"], flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
for line in p.stdout:
    print(line, end="", flush=True)
assert p.wait() == 0, "lid_audit_run.py failed"
MAV_W_SHA256 = "62919289f7ca53b4359ad20834d8c5079ba33e8490cbe8ae72a206272f20eb4a"   # Experiment/LID-AUDIT-01/out/mav_w.json (CRLF)
b = (OUT / "out/mav_w.json").read_bytes()
crlf = b if b"\r\n" in b else b.replace(b"\n", b"\r\n")
rep = {"sha256": hashlib.sha256(b).hexdigest(), "sha256_crlf": hashlib.sha256(crlf).hexdigest(),
       "n_files": json.loads(b)["n_files"]}
rep["same_as_original"] = rep["sha256_crlf"] == MAV_W_SHA256
action, why = ("pass", "sha256 equal") if rep["same_as_original"] else guard_action()
rep["guard"] = {"action": action, "why": why, "FF_MODE": os.environ.get("FF_MODE", "")}
(OUT / "l_lid_audit_check.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)
if action == "warn":
    print("!! WARNING: mav_w.json differs from the file lid_strict.json was built from (" + why + ")", flush=True)
assert action != "fail", ("mav_w.json differs from the one lid_strict.json was built from (FLAG_150 rerun?): use the stored "
                          "FLAG_150 output or set FF_ALLOW_MISMATCH=1 to keep it (" + why + ")")
