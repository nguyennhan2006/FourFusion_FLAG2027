"""Run stage dp_train (DP branch: 10 seeds x two towers) on a local machine, without the runner, and check its result.

Stage dp_train (stages/70_dp/dp_train) normally runs through `runner/ffrun.py`, which mounts its inputs under a Kaggle-like
/kaggle/input view. This script does the same work with plain paths given on the command line:

    python tools/dp_train_local.py \
        --train-zip   <organisers' train_set.zip> \
        --train-meta  <organisers' meta_file_train_set.csv> \
        --lid-strict  <lid_strict.json, version 2> \
        --dp-feats    <folder with feats_v2/ feats_models/ feats_eval/ feats_ext_full/> \
        --eval-feats  <folder with the DP Evaluation features (output of stage dp_extract_eval)> \
        --eval-zip    <eval_set.zip of the Evaluation phase> \
        --out         <empty output folder>

Steps:
  1. print the sha256 of every input file and compare it with the value used for the submission;
  2. run stages/70_dp/dp_train/xjf.py with the paths in its environment variables (DP_FEATS, DP_OUT, DP_TRAIN_ZIP, DP_TRAIN_META,
     DP_LID_STRICT, DP_EVAL_FEATS) and eval_set.zip as its argument; it writes out/xjf_scores.npz, out/consts.json, out/report.txt
     and out/dp_towers.pt (the ten tower pairs);
  3. compare consts.json and the arrays of xjf_scores.npz with the submission run (values in stages/70_dp/dp_train/stage.json
     'verify') and write dp_train_check.json next to out/.

CPU only; about 12 minutes. Bit-identical scores are expected only on the machine of the original run (other CPUs / BLAS builds
give small differences).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STAGE_DIR = REPO / "stages" / "70_dp" / "dp_train"

# sha256 of the input files used for the submission
SUBMISSION_INPUT_SHA256 = {
    "train_set.zip": "1803739f059860d5aede064839a17b01c6297f3160ab80cc9d3c929b27ae2ad9",
    "meta_file_train_set.csv": "fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030",
    "lid_strict.json": "f4065711e01e6284",          # prefix; the full value is stage.json variant.lid_strict_sha256
    "eval_set.zip": "a70762a3fb52af6bbf52c14a86c79a4d074ec319b78d6ffedcdf195d71f9381e",
}


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-zip", required=True, type=Path)
    ap.add_argument("--train-meta", required=True, type=Path)
    ap.add_argument("--lid-strict", required=True, type=Path)
    ap.add_argument("--dp-feats", required=True, type=Path)
    ap.add_argument("--eval-feats", required=True, type=Path)
    ap.add_argument("--eval-zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    stage = json.loads((STAGE_DIR / "stage.json").read_text(encoding="utf-8"))
    expected_inputs = dict(SUBMISSION_INPUT_SHA256, **{"lid_strict.json": stage["variant"]["lid_strict_sha256"]})

    # 1. inputs
    print("[1] inputs", flush=True)
    inputs = {"train_set.zip": args.train_zip, "meta_file_train_set.csv": args.train_meta,
              "lid_strict.json": args.lid_strict, "eval_set.zip": args.eval_zip}
    input_report = {}
    for name, path in inputs.items():
        assert path.is_file(), f"{name}: {path} is not a file"
        got = sha256_of_file(path)
        same = got == expected_inputs[name]
        input_report[name] = {"path": str(path), "sha256": got, "as_submission": same}
        print(f"  {name}: {path} sha256 {got[:16]}... {'(as used for the submission)' if same else '(DIFFERS)'}", flush=True)
    for name in ("feats_v2", "feats_models", "feats_eval", "feats_ext_full"):
        assert (args.dp_feats / name).is_dir(), f"--dp-feats {args.dp_feats} has no {name}/"
    assert (args.eval_feats / "ids_no_gender_English.txt").is_file(), f"--eval-feats {args.eval_feats} has no ids_no_gender_English.txt"
    eval_feats_sha256 = {f.name: sha256_of_file(f) for f in sorted(args.eval_feats.iterdir())
                         if f.is_file() and f.suffix in (".npy", ".txt")}

    # 2. run xjf.py
    print("[2] xjf.py", flush=True)
    out_dir = args.out / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ,
               DP_FEATS=str(args.dp_feats), DP_OUT=str(out_dir), DP_TRAIN_ZIP=str(args.train_zip),
               DP_TRAIN_META=str(args.train_meta), DP_LID_STRICT=str(args.lid_strict), DP_EVAL_FEATS=str(args.eval_feats),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", str(STAGE_DIR / "xjf.py"), str(args.eval_zip)]
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)

    # 3. compare with the submission run
    print("[3] compare with the submission run", flush=True)
    verify = {v["file"]: v for v in stage["verify"]}
    consts = json.loads((out_dir / "consts.json").read_text(encoding="utf-8"))
    want_consts = verify["out/consts.json"]["values"]
    consts_same = all(list(consts[k]) == list(want_consts[k]) for k in want_consts)
    print(f"  consts.json: {consts} -> {'identical' if consts_same else 'DIFFERENT from ' + str(want_consts)}", flush=True)
    scores = np.load(out_dir / "xjf_scores.npz")
    arrays_same = {}
    for key, want in verify["out/xjf_scores.npz"]["arrays_sha256"].items():
        arrays_same[key] = key in scores.files and sha256_of_array(scores[key]) == want
        print(f"  xjf_scores.npz {key}: {'identical' if arrays_same[key] else 'DIFFERENT'}", flush=True)
    towers = out_dir / "dp_towers.pt"
    assert towers.is_file(), "out/dp_towers.pt was not written"
    report = {"inputs": input_report, "eval_feats_sha256": eval_feats_sha256, "consts_identical": consts_same,
              "arrays_identical": arrays_same, "all_identical": consts_same and all(arrays_same.values()),
              "dp_towers.pt": {"bytes": towers.stat().st_size, "sha256": sha256_of_file(towers)}}
    (args.out / "dp_train_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("RESULT:", "bit-identical to the submission run" if report["all_identical"] else "differs from the submission run",
          "| dp_towers.pt sha256", report["dp_towers.pt"]["sha256"], flush=True)


if __name__ == "__main__":
    main()
