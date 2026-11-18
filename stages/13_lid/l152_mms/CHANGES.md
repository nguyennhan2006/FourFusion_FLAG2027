# CHANGES: `l152_mms`

Stage l152_mms (13_lid): FLAG_152: MMS-LID-4017 language ID on every MAV voice file (third vote, reported only; LID-AUDIT-01 requires the table).

Origin: exact-package (`E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/02_flag-2027-nb-152-mmslid.ipynb`).

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | new cells/99_compare.py (after the last original cell) | compares row count / flagged files with the original output (report only) | show how close a rerun is | none | new |

## Notes

- 13.54: the consumers (l_dqual_g3 / l_lid_audit / l_lid_strict) need the STORED output of this stage for the exact filter list; a rerun is not bit-identical (GPU). See their `guard_policy` (FF_MODE=train warns, infer fails) and the request to WP-A for `default_saved: true` on this artifact (rebuild then with FF_FORCE=l152_mms).
- Role 'wiring': the MMS votes do not enter lid_strict.json (only the Whisper votes do), but lid_audit_run.py merges mms.parquet 1:1 with mav.parquet and asserts equal row counts.

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
