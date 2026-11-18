# CHANGES: `r31_mav_rows_check`

Stage r31_mav_rows_check (12_rows): Optional: rebuild the MAV row tables without encoders (FLAG_08 listing + FLAG_31 EXTRAW listing) and compare byte for byte.

Origin: new.

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | check_rows.py + cells/00_check_rows.py | new script; the listing / meta-table code is copied verbatim (ast.get_source_segment) from the executed FLAG_08 module and the EXTRAW cell | close DESIGN G21 / S1 gap 1: prove on CPU that the raw MAV release reproduces the row order of the four ext-meta tables before the GPU stage r08 | none (report only) | new |

## Notes

- DESIGN gives the builder kaggle/build_raw_notebook.py:68-240 (cell EXTRAW): that cell also runs both encoders and needs the FLAG_12 helper cells; the row tables do not depend on the encoders, so only the listing and table code is reused.
- check_rows.py sha256 is recorded in original/SOURCE.json (stage_files); the full builder is kept in original/ for reading.

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
