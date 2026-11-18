"""Copy the stored per-pair score files of mode (c) and the 20 DP feature arrays into a folder laid out for FF_DATA (the Drive
folder), and check every copy against its recorded sha256. Team use: the sources are the team package and the team machine.

    python tools/stage_scores_and_dp_features.py --dest <folder> --team-archive <drive holding FLAG2027_submit/> \
        --project <team project root, holds Experiment/ and kaggle/output/>

  * mode (c) inputs: the ten files of assembly/expected.json 'inputs' (their 'team_copy' paths) -> <dest>/05_scores/<file name>;
    these are the files assembly/assemble_exact.py and the assembly stages read from FF_DATA/05_scores;
  * DP features: the 20 files of stages/70_dp/dp_feats_layout/dp_features.json (their 'team_machine_copy' paths)
    -> <dest>/02_intermediate/dp_features/<target>; stage dp_train finds this folder by its structure (gap G13).
Exits with code 1 when a source is missing or a copy does not match.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def copy_and_check(src, dst, wanted):
    """Copy src to dst (unless dst already holds a matching file) and return True when dst's sha256 is one of `wanted`."""
    if not (dst.is_file() and sha256_of_file(dst) in wanted):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return sha256_of_file(dst) in wanted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--team-archive", required=True, type=Path, help="folder that holds FLAG2027_submit/")
    ap.add_argument("--project", required=True, type=Path, help="team project root (Experiment/, kaggle/output/)")
    args = ap.parse_args()
    failed = 0

    # 1. the ten per-pair score files of mode (c)
    inputs = json.loads((REPO / "assembly" / "expected.json").read_text(encoding="utf-8"))["inputs"]
    for name, spec in inputs.items():
        team_copy = spec["team_copy"].split(" (or ")[0]                       # first location named
        src = (args.team_archive / team_copy.replace("<team archive>/", "")) if team_copy.startswith("<team archive>") \
            else (args.project / team_copy)
        wanted = set(spec.get("sha256_any_of") or [spec["sha256"]])
        if not src.is_file():
            print(f"--- 05_scores/{name}: source missing ({src})", flush=True); failed += 1; continue
        ok = copy_and_check(src, args.dest / "05_scores" / name, wanted)
        print(f"{'OK ' if ok else 'BAD'} 05_scores/{name} <- {src}", flush=True); failed += not ok

    # 2. the 20 DP feature arrays
    spec = json.loads((REPO / "stages" / "70_dp" / "dp_feats_layout" / "dp_features.json").read_text(encoding="utf-8"))["files"]
    for f in spec:
        src = args.project / f["team_machine_copy"]
        dst = args.dest / "02_intermediate" / "dp_features" / f["target"]
        if not src.is_file():
            print(f"--- dp_features/{f['target']}: source missing ({src})", flush=True); failed += 1; continue
        ok = copy_and_check(src, dst, {f["sha256"]})
        print(f"{'OK ' if ok else 'BAD'} 02_intermediate/dp_features/{f['target']} <- {src}", flush=True); failed += not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
