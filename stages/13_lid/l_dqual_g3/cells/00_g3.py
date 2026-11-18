print("[cell] l_dqual_g3: DQUAL-01 layer G3 (mixed identities) for VoxCeleb2 and VoxBlink, CPU", flush=True)
# New cell (FourFusion reproduction repo). dqual_g3.py (this folder) = the G3 part of Experiment/DQUAL-01/run.py on the sources of
# Experiment/EDA-NEWDATA/run.py; it fails when the G3 lists differ from Experiment/DQUAL-01/persons.parquet (expected.json holds the
# counts 454 / 1 388 and the sha256 of the sorted speaker lists). What a mismatch does is set in stage.json guard_policy:
# warn in FF_MODE=train (rebuilt FLAG_41 / FLAG_42 arrays are not bit-identical) or with FF_ALLOW_MISMATCH=1 / FF_SMOKE=1, else fail.
import json, os, subprocess, sys
from pathlib import Path

STAGE_DIR = Path(os.environ.get("FF_STAGE_DIR", "."))
F41 = Path("/kaggle/input/notebooks/ff/f41_t3enc_vox2")              # FLAG_41 feats_vox2t3 (the four T3 arrays of the FLAG_37 rows)
KEYS = Path("/kaggle/input/datasets/ngnnguynchn/flag2027-vox2-keys")   # FLAG_37 key lists
L42 = Path("/kaggle/input/notebooks/nguynlcnhtho/flag-2027-nb-42")     # FLAG_42 lid_summary.json (VOX_LID)
C46 = Path("/kaggle/input/notebooks/ff/c46_voxblink")                  # FLAG_46 feats_vb (meta + T3 arrays)
OUT = Path("/kaggle/working/dqual_g3")


def folder_with(root, names, why):
    """The folder under root that holds every file in names (saved copies may nest it differently)."""
    for p in sorted(root.rglob(names[0])):
        if all((p.parent / n).is_file() for n in names):
            return p.parent
    raise FileNotFoundError(f"no folder with {names} under {root} ({why})")


def guard_action():
    """stage.json guard_policy.on_mismatch[FF_MODE] ('warn' | 'fail'); FF_SMOKE=1 or FF_ALLOW_MISMATCH=1 -> 'warn'."""
    if os.environ.get("FF_SMOKE") == "1" or os.environ.get("FF_ALLOW_MISMATCH") == "1":
        return "warn", "FF_SMOKE=1 or FF_ALLOW_MISMATCH=1"
    mode = os.environ.get("FF_MODE", "")
    try:
        pol = json.loads((STAGE_DIR / "stage.json").read_text(encoding="utf-8"))["guard_policy"]["on_mismatch"]
    except Exception:
        pol = {"train": "warn"}
    return pol.get(mode, "fail"), f"stage.json guard_policy.on_mismatch[{mode or 'FF_MODE unset'}]"


T3 = ["face_farl16.npy", "face_clipL.npy", "voice_sam100p.npy", "voice_sam100f.npy"]
cmd = [sys.executable, "-u", str(STAGE_DIR / "dqual_g3.py"),
       "--vox2-arrays", str(folder_with(F41, T3, "FLAG_41 output feats_vox2t3")),
       "--vox2-keys", str(folder_with(KEYS, ["face_meta.csv", "voice_meta.csv"], "flag2027-vox2-keys")),
       "--vox2-lid-summary", str(folder_with(L42, ["lid_summary.json", "voice_lid_probs.npy"], "FLAG_42 output feats_vox2lid") / "lid_summary.json"),
       "--vb", str(folder_with(C46, T3 + ["face_meta.csv", "voice_meta.csv"], "FLAG_46 output feats_vb")),
       "--out", str(OUT), "--expected", str(STAGE_DIR / "expected.json")]
ACTION, WHY = guard_action()
print("on a G3 mismatch:", ACTION, "(" + WHY + ")", flush=True)
if ACTION == "warn":
    cmd.append("--allow-mismatch")
print("run:", " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
assert p.wait() == 0, ("dqual_g3.py failed (see the MISMATCH line above): use the stored FLAG_41 / FLAG_42 / FLAG_46 outputs "
                       "(FF_DATA/02_intermediate, configs/data_map.json) or set FF_ALLOW_MISMATCH=1 (" + WHY + ")")
_rp = OUT / "dqual_g3_report.json"
_r = json.loads(_rp.read_text())
_ok = all((_r.get("match_persons_parquet") or {}).values())
_r["guard"] = {"action": "pass" if _ok else ACTION, "why": "G3 lists equal to persons.parquet" if _ok else WHY,
               "FF_MODE": os.environ.get("FF_MODE", "")}
_rp.write_text(json.dumps(_r, indent=1))
if not _ok:
    print("!! WARNING: G3 lists differ from Experiment/DQUAL-01/persons.parquet: lid_strict.json will differ too (" + WHY + ")", flush=True)
