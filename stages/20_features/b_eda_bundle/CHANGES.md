# CHANGES: stage `b_eda_bundle`

Original code: kaggle/build_eda_bundle.py (in git, unchanged since a89a071; blob 5921e8b0) (local-script). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | build_eda_bundle.py: line 7 | `path` | `import json, time, zipfile, io ⏎ ` -> `import json, time, zipfile, io ⏎ import os  # FF path (CHANGES.md): F…` | needed by the two changes below | none |
| 2 | build_eda_bundle.py: line 12 | `path` | `R = Path(__file__).resolve().parents[1] ⏎ ` -> `R = Path(os.environ["FF_REPO_ROOT"]) if os.environ.get("FF_REPO_ROOT"…` | the script resolved every input from the team's repository root; the stage gives a stand-in root built from the input artifacts | none |
| 3 | build_eda_bundle.py: line 52 | `wiring` | `manifest = {"created": time.strftime("%Y-%m-%d %H:%M"), "arrays"` -> `manifest = {"created": os.environ.get("FF_EDA_CREATED") or time.strft…` | the manifest stamp is a version check: kaggle/build_lab_bundle.py writes eda_bundle_created = '2026-09-29 08:06' into lab-data and the notebooks 164f, 112s-f0s, 112s-f1s, 184, 50cc, 112f0sx, 112f1sx assert equality; the stage sets the original stamp | none on any array |

## Other differences from the original run (inputs, wiring, environment)

- **wiring**: kaggle/output/feats_v2 (FLAG_04) is linked to the f12_eval_feats folder: the script reads only face_arcface_train.npy and voice_ecapa192_train.npy from it, which FLAG_12 wrote byte-identical (sha256 55491b54... / b3bd0a64..., team copies, 18/11/2026). FLAG_04 itself is kept as the optional stage f04_v2_ref.
- **path**: kaggle_upload/train is rebuilt from the raw organiser data (members of train_set.zip + meta_file_train_set.csv) with CRC32 checks against the team's copies.
- **upload**: The original zip was uploaded by hand as ngnnguynchn/flag2027-eda-bundle-p; here it is unpacked into eda_bundle_dataset/ (= the dataset content) and exported under the same artifact id.
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude). Today runner/ffcore.py build_vin applies only the configs/data_map.json exclude, so the archive is still visible; the code never unpacks or reads it (ZIP_PREFIXES / train-only index). With raw-v4 mounted unpacked (a hand-attached Kaggle dataset), find_splits still reads the dev list files (pd.read_csv of the list and an existence check of its media) before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded. Mount raw-v4 as the zipped organiser release (FF_DATA/00_raw/btc_mavceleb_v4).

## Notes

- v4 arrays stay float32, MAV-Celeb arrays float16 (as the original); BEST/v7_safe/safe_lib.py load_ext reads them back through float16 -> float32.
- expected_members.json = sha256 of every file of the original dataset (package E: copy, re-downloaded from Kaggle 12/10/2026).

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `bundle_report.json` `vs_original/missing` = []: every member of the original dataset (expected_members.json) is in the new bundle.
- verify (warning only): `bundle_report.json` `vs_original/extra`: no member beyond the original's (warning only).
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).
