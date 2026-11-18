# CHANGES: `l150_whisper`

Stage l150_whisper (13_lid): FLAG_150: Whisper-large-v3 language ID on every cached 3 s training clip + every MAV voice file (+ VoxLingua107 on MAV).

Origin: exact-package (`E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/01_flag-2027-nb-150-lidwhisper.ipynb`).

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | new cells/99_compare.py (after the last original cell) | compares the flagged lists (counts + digests) and the Whisper-flagged MAV files with the original output (report only) | show how close a rerun is | none | new |

## Notes

- 13.54: the consumers (l_dqual_g3 / l_lid_audit / l_lid_strict) need the STORED output of this stage for the exact filter list; a rerun is not bit-identical (GPU). See their `guard_policy` (FF_MODE=train warns, infer fails) and the request to WP-A for `default_saved: true` on this artifact (rebuild then with FF_FORCE=l150_whisper).
- Package notebook = `kaggle kernels pull` of the run with the secret value redacted by release/factor_enfuse/make_package.py (cell 0 now holds only a comment); HF_TOKEN is read from the environment if set (public models: optional).
- The notebook embeds the 'risk speaker' order (VoxCeleb2 / VoxBlink speaker ids, read from the stored VoxLingua probabilities when the notebook was built): ids only, no media.
- Kept as in the run: pip install of speechbrain inside cell 5 (pip_policy original).

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
