# CHANGES: `raw_mavceleb`

Stage raw_mavceleb (10_raw): Extract the original MAV-Celeb v1 / v2 zips exactly as the Kaggle dataset chnnhnnguyn/external-data-mavceleb.

Origin: new.

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | cells/00_locate.py .. 02_check.py (whole stage) | new: zip sha256 check, zipfile extraction into the folder named after each zip, file-count / layout check | the dataset had no producing code (Kaggle's own auto-extraction) | none (same file names and bytes as Kaggle's extraction) | new |

## Notes

- A saved copy of the extracted dataset (raw class, always taken from FF_DATA: this stage does not run then) is checked by the new stage `raw_mavceleb_check` (outputs.verifies), scheduled before the first MAV consumer. It stops a partial copy such as the team package folder (mavceleb_v1_complete-001/{meta_v1.csv, v1/faces} only).
- Zip sha256 (measured 18/11/2026, enforced in cells/00_locate.py): mavceleb_v1_complete-001.zip 2082a9ee5f7d3634f687d682e98cd223606a5b01debc90c3e7ab566a9f86512b, mavceleb_v2_complete-002.zip daff513527d7cea698243594a18795cab6d47aea64d64c3c74035154abd6246c (to be copied into configs/data_map.json and configs/expected_sha256.json by WP-A, DATA.md by WP-K). mavceleb_v1_train-004.zip is not needed.
- Both parent folders are kept as they are (mavceleb_v1_complete-001/, mavceleb_v2_complete-002/): stage r08_mav_rows finds the roots by their '/v1/' and '/v2/' folders, and the RFS LoRA C1 run (t163l_orig, kaggle/build_flag163.py:93) stops at the first v1 / v2 folder it meets, so a different layout could change what C1 learned (DESIGN §4.1, G8).
- Release content (central directory read 18/11/2026): v1 = meta_v1.csv + v1/ (302 492 jpg, 19 556 wav; 2 337 directory entries); v2 = v2/ (371 626 jpg, 20 715 wav; 2 767 directory entries). mavceleb_v1_train-004.zip is not used.

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
