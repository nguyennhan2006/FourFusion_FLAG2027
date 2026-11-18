# CHANGES: `l_lid_audit`

Stage l_lid_audit (13_lid): LID-AUDIT-01: three LID votes per MAV voice file -> mav_w.json (58 Whisper-flagged files, file-level rule).

Origin: local-script (`Experiment/LID-AUDIT-01/run.py`).

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | lid_audit_run.py (copy of Experiment/LID-AUDIT-01/run.py) | the two inputs and the output folder from FF_MAV_PARQUET / FF_MMS_PARQUET / FF_OUT (defaults = original paths); FF_OUT/out created | team-machine paths | none | path |
| 2 | cells/00_audit.py | new: run the script, compare mav_w.json with the original (CRLF form) | run inside the stage runner | none | new |
| 3 | cells/00_audit.py | the mismatch guard reads `guard_policy.on_mismatch[FF_MODE]` from stage.json: `warn` in FF_MODE=train or with FF_ALLOW_MISMATCH=1 / FF_SMOKE=1, `fail` in infer / assemble / unset; the decision goes into lid_audit/l_lid_audit_check.json (`guard`) | the old guard failed in every mode unless the undocumented FF_ALLOW_MISMATCH=1 was set, so a full rebuild (FF_MODE=train, LID inputs re-run on GPU) would dead-stop here | none (only decides between stopping and warning) | new |

## Notes

- **13.54 policy for the Bengali-speech filter** (machine-readable: `guard_policy` in the stage.json of l_dqual_g3, l_lid_audit and l_lid_strict, read by their cells; FF_MODE is set by the runner): the exact list f4065711... is rebuilt only from the STORED outputs of FLAG_42 (l42_vox2_lid), FLAG_150 (l150_whisper), FLAG_152 (l152_mms), FLAG_41 (f41_t3enc_vox2) and the collected FLAG_53 / FLAG_46 data. A mismatch fails in FF_MODE=infer / assemble (and when FF_MODE is unset); in FF_MODE=train (those GPU inputs may have been re-run and are not bit-identical) it warns, the run continues as a recipe-level rebuild with the new list, and the check file records the decision. FF_ALLOW_MISMATCH=1 forces the warning in any mode (stage.json env_doc). Requested from WP-A: `default_saved: true` in configs/data_map.json for the outputs of l42 / l150 / l152 / f41 so FF_MODE=train keeps the stored copies by default (a rebuild then needs FF_FORCE).
- mav_w.json lists the files with a Whisper vote (user decision 3/11 14:30: Whisper votes only); MMS / VoxLingua votes are reported only. With the stored FLAG_150 output the file is identical (the original was written on Windows: CRLF).

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
