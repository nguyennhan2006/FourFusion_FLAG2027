# CHANGES: `c53_vox2b`

Baseline: `original/flag-2027-nb-53-scaleid.ipynb` (exact-executed-copy; run `nguynlcnhtho/flag-2027-nb-53-scaleid`).

How `stage.ipy` is made: release/flag2027_e2e/build_notebook.py stage_cells(): code cells of the original notebook in order, empty ones skipped, CRLF -> LF, cells whose title (first line, or the name in a leading print("[cell]/[boot] ..."), cut to 90 chars) matches a drop_cells regex removed, then every patch applied in list order (each must match exactly `count` times), then ''.join(cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n' for cell in cells), written UTF-8 with LF. No global patch is assumed.

Original code cells: 12; kept: 6; dropped: 6; patches: 5.

## Dropped cells (`drop_cells`)

| notebook cell | code cell | title |
|---|---|---|
| 9 | 7 | `# embedded code (= the local files, md5 below) -> /kaggle/working/lab; data folders linked` |
| 11 | 8 | `# resume: a previous version's e2e_out/ (same experiment) -> finished jobs and their files` |
| 13 | 9 | `# JOBS: every SAFE-04 fold + the final head (started second, so it overlaps the folds); at` |
| 15 | 10 | `# registered gate, computed here (Experiment/E2E/e2e_lib.gate through the experiment's rep` |
| 17 | 11 | `# final fuser (CA-FUSE refit on every SAFE-04 row with the new stream) -> submission.zip; ` |
| 19 | 12 | `prog.update(gpu=GPU, torch=torch.__version__, lab_data=MAN["created"], e2e=E2E_MAN["create` |

Reason: the SCALE-ID-01 experiment (lab code, SAFE-04 folds, gate, dev submission zip) is not part of the 13.54 chain; only part 0 (the extraction) is (design notes, internal). Type: wiring / dev-exclusion.

## Patches (in application order)

| # | id | where (notebook cell / code cell) | type | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | `drop_experiment_inputs` | 1 / 1 | dev-exclusion | the dropped SCALE-ID-01 experiment needed eda-bundle + e2e-dev (dev-derived tables); part 0 does not: stop requiring them | none |
| 2 | `hf_revision` | 7 / 6 | pin | HF revision parameter (G27) + token header for the gated repo; unset = original strings | none |
| 3 | `auth_remote_r` | 7 / 6 | url | Remote(): resolve each part with the token (ProgramComputer/voxceleb is gated now); no HF_TOKEN = original request | none |
| 4 | `auth_remote_q` | 7 / 6 | url | Remote._get(): re-resolve an expired signed URL with the token (same reason) | none |
| 5 | `auth_meta` | 7 / 6 | url | vox2_meta.csv of the gated repo: send the token | none |

Exact old / new text of every patch: `stage.json` → `patches`. Reverse check: replacing every `new` by its `old` in `stage.ipy` gives back the kept original cells byte for byte (verified when the stage was built, 18/11/2026).

## Deliberately unchanged

- `!pip` lines (unpinned packages as in the run): `pip_policy` original; the pinned environment specification pins versions (FF_PIP=pinned; Docker optional).
- Selection / sampling code and every threshold: verbatim.

## Inputs (original run -> this stage)

| original attachment (push log line 13) | here | reason | type | effect on numbers |
|---|---|---|---|---|
| `ngnnguynchn/flag2027-lab-data` | `datasets/ngnnguynchn/flag2027-lab-data-p`, mounted as `datasets/ngnnguynchn/flag2027-lab-data` | FLAG_37 keys through the lab manifest roots; the private copy of the same bundle, rebuilt by `b_lab_data` | wiring | none |
| `berryyusilee/flag2027-eda-bundle` | not attached | needed only by the dropped SCALE-ID-01 experiment (dev-derived tables); patch 1 removes the presence check | dev-exclusion | none |
| `hieuhau/flag2027-e2e-dev` | not attached | same (frozen SAFE-04 tables, dev-derived) | dev-exclusion | none |

## Evidence file in `original/`

`original/flag-2027-nb-53-scaleid.part0.log` = the run log cut after part 0 (first 267 lines verbatim + the closing `]`): the cut-off
events belong to the dropped experiment (lab code listing, SAFE-04 folds, dev-scoring output of SCALE-ID-01). The full log's size,
md5 and sha256 are in `original/SOURCE.json` (`full_log`). The notebook in `original/` is unchanged (byte-exact, md5 = push record).

## Runner metadata changed after review (18/11/2026; `stage.json` only, no code change)

| change | reason | effect on numbers |
|---|---|---|
| `skip_with` kept, `skip_with_requires` added | `FF_DATA/01_collected/voxceleb2_flag53` is a saved copy of the part of the kernel output that downstream reads only when `feats_vox2b/` also holds the two listing parquets (copied from frames-index `vox2b/`); without them a forced `c_frames_index` stops at its `find()` | none |
| `env` pins `FLAG_LOCAL="0"` and `FLAG_SMOKE="0"` | a value left in the caller's environment could switch the notebook into its local / smoke mode during a real run (the code compares with `== "1"`); `smoke_env` still sets `"1"` under `FF_SMOKE=1` | none |
| `private: true` | aligned with `configs/data_map.json` (VoxCeleb2-derived data; private FF_DATA share) | none |
| `modes.train` rewritten | a re-collection must force the whole collect closure in one run (fresh `FF_WORK`, `FF_USE_SAVED=collected`); forcing one stage alone mixes new rows with the old frames / audio cache | none |
| `verify` expect uses nested `lid` | the runner compares dict keys literally (no dotted paths) | none |
| `license` on every `external_sources` entry, licence notes on `models` | input for `LICENSES.md` (WP-K); uncertain items marked 'not re-checked' | none |
| internal references labelled | design notes / team logs are internal provenance, not shipped; `CHUA CHAC` -> `unverified` | none |
