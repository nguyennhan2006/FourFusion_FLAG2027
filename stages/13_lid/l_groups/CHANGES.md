# CHANGES: `l_groups`

Stage l_groups (13_lid): Evidence: language groups (DSEL-01, SCALE-CC-01) and genders of the VoxCeleb2 speakers.

Origin: local-script (`Experiment/DSEL-01/make_groups.py + Experiment/SCALE-CC-01/make_groups.py`).

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | dsel01_make_groups.py (copy of Experiment/DSEL-01/make_groups.py) | LD and the output folder from FF_VOX2LID / FF_OUT (defaults = original paths) | team-machine paths | none | path |
| 2 | scalecc01_make_groups.py (copy of Experiment/SCALE-CC-01/make_groups.py) | DSEL-01 script, FLAG_53 folder, FLAG_42 labels, vox2_meta.csv and output folder from FF_DSEL01_SCRIPT / FF_FLAG53_DIR / FF_FLAG42_LABELS / FF_VOX2_META / FF_OUT | team-machine paths | none | path |
| 3 | cells/00_groups.py | new: run both scripts, compare the outputs with the original files (line ends normalised to CRLF: the originals were written on Windows) | run inside the stage runner | none | new |
| 4 | cells/00_groups.py | the mismatch guard reads `guard_policy.on_mismatch[FF_MODE]` from stage.json: `warn` in FF_MODE=train or with FF_ALLOW_MISMATCH=1 / FF_SMOKE=1, `fail` in infer / assemble / unset; the decision goes into groups/l_groups_check.json (`guard`) | same policy as the other 13_lid guards: a rebuild with FLAG_42 re-run on GPU must not dead-stop, a reproduction from the stored output must | none (only decides between stopping and warning) | new |

## Notes

- value_counts() decides ties of the majority language; a pandas version that orders ties differently would change the group of a tied speaker. The check reports it (FF_ALLOW_MISMATCH=1 keeps the output).

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
