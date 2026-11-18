# CHANGES: `l_dqual_g3`

Stage l_dqual_g3 (13_lid): DQUAL-01 layer G3 only (bottom 5 % within-person consistency) for VoxCeleb2 and VoxBlink.

Origin: new.

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | dqual_g3.py (whole stage) | new reduced copy of DQUAL-01: only layer G3, only VoxCeleb2 / VoxBlink; no AVSpeech / LRS3 (NEW_DIR), no AUXP-01 CC head (CC_MODELS), no v4 / MAV sources (rel_sweep5), no G1 / G2 / G4 | DESIGN G12: the original run.py needs sources the 13.54 system must not use; lid_strict.json reads G3 of VoxCeleb2 / VoxBlink only (merge.py:47-49) | none on G3 (each source's quantile is computed within the source) | new |
| 2 | dqual_g3.py sources() | drop list = FLAG_42 lid_summary speakers_dropped_ids; LID_STRICT not applied | = T3ENC-01/run.py dropped_speakers() on 14/10, before lid_strict.json existed (it is built from this table) | none | new |
| 3 | cells/00_g3.py | the mismatch guard reads `guard_policy.on_mismatch[FF_MODE]` from stage.json: `warn` in FF_MODE=train or with FF_ALLOW_MISMATCH=1 / FF_SMOKE=1, `fail` in infer / assemble / unset; the decision goes into dqual_g3/dqual_g3_report.json (`guard`) | the old cell passed --allow-mismatch only with FF_ALLOW_MISMATCH=1 / FF_SMOKE=1, so the script failed in every mode unless the undocumented FF_ALLOW_MISMATCH=1 was set, so a full rebuild (FF_MODE=train, LID inputs re-run on GPU) would dead-stop here | none (only decides between stopping and warning) | new |

## Notes

- **13.54 policy for the Bengali-speech filter** (machine-readable: `guard_policy` in the stage.json of l_dqual_g3, l_lid_audit and l_lid_strict, read by their cells; FF_MODE is set by the runner): the exact list f4065711... is rebuilt only from the STORED outputs of FLAG_42 (l42_vox2_lid), FLAG_150 (l150_whisper), FLAG_152 (l152_mms), FLAG_41 (f41_t3enc_vox2) and the collected FLAG_53 / FLAG_46 data. A mismatch fails in FF_MODE=infer / assemble (and when FF_MODE is unset); in FF_MODE=train (those GPU inputs may have been re-run and are not bit-identical) it warns, the run continues as a recipe-level rebuild with the new list, and the check file records the decision. FF_ALLOW_MISMATCH=1 forces the warning in any mode (stage.json env_doc). Requested from WP-A: `default_saved: true` in configs/data_map.json for the outputs of l42 / l150 / l152 / f41 so FF_MODE=train keeps the stored copies by default (a rebuild then needs FF_FORCE).
- Saved copy (skip_with 02_intermediate/dqual_g3): either `persons_g3.parquet` (this stage's output) or Experiment/DQUAL-01/persons.parquet itself; l_lid_strict accepts both names.
- Exactness check without shipping data: expected.json holds the counts and the sha256 of the sorted speaker lists (all ids and G3 fails) per source, computed 18/11 from Experiment/DQUAL-01/persons.parquet; the script fails on a mismatch unless FF_ALLOW_MISMATCH=1 (cell) / --allow-mismatch.

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
