# CHANGES: `raw_evalphase`

Stage raw_evalphase (10_raw): Organisers' Evaluation zip -> eval_set.zip (*_valid -> *_test, 416 list lines fixed) = dataset flag2027-evalphase-p.

Origin: local-script (`Experiment/EVALPATH-U/build_evalphase.py`).

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | build_evalphase.py | copied unchanged (paths were already command-line arguments) | original local script | none | none |
| 2 | cells/00_locate.py .. 03_unpack.py | new cells: locate + sha256 the organisers' zip, run the script, compare every member with the uploaded eval_set.zip, unpack the dataset view | run the local script inside the stage runner and prove the output equals the dataset the system read | none | new |

## Notes

- The original command was `python -I Experiment/EVALPATH-U/build_evalphase.py <btc zip> <out zip>` (usage line of the script).
- Member content is deterministic; the zip bytes are not (writestr uses the current time, and the deflate stream may depend on the zlib version). Hence the member-by-member check.
- Kaggle auto-extracted the uploaded eval_set.zip: the package copy E:/.../datasets/ngnnguynchn__flag2027-evalphase-p.zip (sha256 411c9f16...) holds evaluation_phase/... with 4 669 files and no zip.

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
