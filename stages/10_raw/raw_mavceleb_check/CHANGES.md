# CHANGES: `raw_mavceleb_check`

Stage raw_mavceleb_check (10_raw): Check a saved copy of the extracted MAV-Celeb v1 / v2 dataset (chnnhnnguyn/external-data-mavceleb): counts, layout, meta_v1.csv, no zip.

Origin: new.

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | cells/00_check.py (whole stage) | new verification cell (no original code: organisers' data) | raw data is always taken from a saved copy under FF_DATA, so raw_mavceleb (which checks only its own fresh extraction) does not run then; a partial copy (for example the team package folder: v1 faces only) would silently change the MAV row tables (r08), the LoRA C1 data (t163l_orig) and the language-ID lists (l150, l152). This stage fails before any consumer runs | none (reads only) | new |

## Notes

- Declared through `outputs.verifies` (as raw_btc does for the organisers' v4 release): the runner schedules it before the first stage that mounts `datasets/chnnhnnguyn/external-data-mavceleb` from a saved copy.
- Smoke (FF_SMOKE=1): layout, meta_v1.csv and stray zips only; the full file count (about 714 000 files) is skipped.
- Expected counts and the meta_v1.csv sha256 are those of `raw_mavceleb/expected.json` (central directories of the two organisers' zips, read 18/11/2026).

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells` and `runner/check_stage.py`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`.
