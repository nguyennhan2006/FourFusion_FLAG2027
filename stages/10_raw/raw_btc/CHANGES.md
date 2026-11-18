# CHANGES: `raw_btc`

Stage raw_btc (10_raw): Check the organisers' MAV-Celeb v4 train / dev release (no producing code).

Origin: new.

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | cells/00_check.py (whole stage) | new verification cell (no original code: organisers' data) | prove that FF_DATA holds the exact release the system read | none (reads only) | new |

## Notes

- The stage has no artifact output the chain consumes: raw data is always taken from FF_DATA (DESIGN §2.4); raw_btc only checks it.
- dev_set.zip is reported if present and never required: no stage of the 13.54 chain reads dev data (DESIGN §2.8).

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
