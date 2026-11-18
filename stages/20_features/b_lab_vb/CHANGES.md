# CHANGES: stage `b_lab_vb`

Original code: kaggle/build_lab_bundle.py vb (not in git) (local-script). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | build_lab_bundle.py: line 17 | `path` | `import io, json, sys, time, zipfile ⏎ ` -> `import io, json, sys, time, zipfile ⏎ import os  # FF path (CHANGES.m…` | needed by the change below | none |
| 2 | build_lab_bundle.py: line 22 | `path` | `R = Path(__file__).resolve().parents[1] ⏎ ` -> `R = Path(os.environ["FF_REPO_ROOT"]) if os.environ.get("FF_REPO_ROOT"…` | the script resolved every input from the team's repository root; the stage gives a stand-in root built from the input artifacts | none |

## Other differences from the original run (inputs, wiring, environment)

- **upload**: The original zip was uploaded by hand as the private dataset ngnnguynchn/flag2027-lab-vb-p; here it is unpacked into lab_vb_dataset/ and exported under the same artifact id.

## Notes

- VoxBlink-derived arrays: CC BY-NC-SA 4.0, no redistribution; the artifact must stay private (FF_DATA shared only with the organisers).

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `bundle_report.json` `vs_original/missing` = []: every member of the original dataset (expected_members.json) is in the new bundle.
- verify (warning only): `bundle_report.json` `vs_original/extra`: no member beyond the original's (warning only).
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).
