# CHANGES: `l42_vox2_lid`

Stage l42_vox2_lid (13_lid): FLAG_42: VoxLingua107 language ID of the 38 345 FLAG_37 VoxCeleb2 clips (12 s centre crop, streamed from the HF tars).

Origin: builder-regenerated (`kaggle/FLAG_42_vox2_lid.ipynb`).

| # | where | old -> new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | new cells/99_compare.py (after the last original cell) | compares lid_summary counts / drop list digest and file sha256 with the original output (report only) | show how close a rerun is | none | new |

## Notes

- 13.54: the consumers (l_dqual_g3 / l_lid_audit / l_lid_strict) need the STORED output of this stage for the exact filter list; a rerun is not bit-identical (GPU). See their `guard_policy` (FF_MODE=train warns, infer fails) and the request to WP-A for `default_saved: true` on this artifact (rebuild then with FF_FORCE=l42_vox2_lid).
- The FLAG_42 cell finds the key lists with IN.rglob('face_meta.csv') / ('voice_meta.csv') and takes the first hit: mount only flag2027-vox2-keys (or the FLAG_37 output), nothing else with such files.
- Rule in the notebook = the OLD rule (top-1 bn or p(bn) >= 0.2); the strict rule (bn + as) is applied later by merge.py (stage l_lid_strict) on the stored probabilities.
- Kept as in the run: `!pip -q install -U "speechbrain>=1.1.1" av` (pip_policy original).

## How stage.ipy is built

Recipe `ff-stage-ipy/1` (same convention as `release/flag2027_e2e/build_notebook.py:stage_cells`): take the cells listed in `stage.json` `build.sequence` in that order (`orig`: index of the cell in the original notebook's `nb["cells"]` list, markdown cells counted, and the referenced cell must be a non-empty code cell, as in `runner/check_stage.py`; CRLF -> LF; `new`: a file of `cells/`), apply every `patches` entry to the original cells only (each must match exactly `count` times in total), and join every cell as `cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n'`. The sha256 of the result is `build.stage_ipy_sha256`. Original code cells not in the sequence are the ones matched by `drop_cells` (regex on the cell title = its `print("[cell] ...")` text or first line).
