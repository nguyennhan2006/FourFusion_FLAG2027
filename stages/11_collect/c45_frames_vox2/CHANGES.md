# CHANGES: `c45_frames_vox2`

Baseline: `original/flag-2027-nb-45-facelora-farl.ipynb` (exact-executed-copy; run `onnhqunhh/flag-2027-nb-45-facelora-farl`).

How `stage.ipy` is made: release/flag2027_e2e/build_notebook.py stage_cells(): code cells of the original notebook in order, empty ones skipped, CRLF -> LF, cells whose title (first line, or the name in a leading print("[cell]/[boot] ..."), cut to 90 chars) matches a drop_cells regex removed, then every patch applied in list order (each must match exactly `count` times), then ''.join(cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n' for cell in cells), written UTF-8 with LF. No global patch is assumed.

Original code cells: 10; kept: 4; dropped: 6; patches: 5.

## Dropped cells (`drop_cells`)

| notebook cell | code cell | title |
|---|---|---|
| 6 | 4 | `# ============================================================================ audio + bat` |
| 8 | 5 | `# ============================================================================ T3ENC-01 en` |
| 12 | 7 | `# ============================================================================ the adapted` |
| 14 | 8 | `# ============================================================================ running` |
| 16 | 9 | `# ============================================================================ the adapted` |
| 18 | 10 | `man = {"created": time.strftime("%Y-%m-%d %H:%M"), "splits": {k: len(v["wav"]) for k, v in` |

Reason: only the frame cache of FLAG_45 belongs to the 13.54 chain (design notes, internal); the dropped cells serve the FaRL LoRA adapter, which is not used. Type: wiring.

## Patches (in application order)

| # | id | where (notebook cell / code cell) | type | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | `train_only` | 1 / 1 | dev-exclusion | dev never read; the stage reads organiser TRAIN data only for the verbatim data cell and copies the train list + meta into its output | none |
| 2 | `train_zip_only` | 1 / 1 | dev-exclusion | without it the data cell would unpack dev_set.zip too (SPLIT_FILTER only filters afterwards) | none |
| 3 | `hf_revision` | 10 / 6 | pin | HF revision parameter (G27) + token header; unset = original strings | none |
| 4 | `auth_resolve` | 10 / 6 | url | resolve(): the gated repo needs the token | none |
| 5 | `frames_only` | 10 / 6 | wiring | frames-only stage: the LoRA FaRL training of FLAG_45 is not part of the 13.54 chain (design notes, internal); it shares one notebook cell with the frame fetching, so drop_cells cannot remove it | none (vox_frames_png.zip is written before this point) |

Exact old / new text of every patch: `stage.json` → `patches`. Reverse check: replacing every `new` by its `old` in `stage.ipy` gives back the kept original cells byte for byte (verified when the stage was built, 18/11/2026).

## Deliberately unchanged

- `!pip` lines (unpinned packages as in the run): `pip_policy` original; the pinned environment specification pins versions (FF_PIP=pinned; Docker optional).
- Selection / sampling code and every threshold: verbatim.

## Inputs (original run -> this stage)

| original attachment (push log line 30) | here | reason | type | effect on numbers |
|---|---|---|---|---|
| `trngsnam/flag-input` (older public copy of the organisers' v4 data) | `datasets/ngnnguynchn/flag2027-raw-v4-p`, mounted as `datasets/trngsnam/flag-input` | same organiser files (sha256 checked by `raw_btc`); the verbatim data cell needs an organiser list with media; only `train_set.zip` is unpacked (patches) | wiring / dev-exclusion | none |
| `chnnhnnguyn/external-data-mavceleb` | not attached | MAV-Celeb rows are optional in the kept data cell (`EXT_SOURCES` sources are skipped when absent) and serve only the dropped LoRA / MAV cells | wiring | none (frames do not use MAV) |
| `nguynchnngn/flag2027-ext-meta` | not attached | MAV row tables, read only by the dropped cells | wiring | none |
| `ngnnguynchn/flag2027-lab-data` | `datasets/ngnnguynchn/flag2027-lab-data-p`, mounted as `datasets/ngnnguynchn/flag2027-lab-data` | FLAG_37 keys, FLAG_41 arrays (row-count `assert`), FLAG_42 `lid_summary.json` (`assert`); the private copy of the same bundle, rebuilt by `b_lab_data` | wiring | none |

Organiser files in the output: the verbatim data cell copies `train_English.txt` to `feats_facelora/index_train.txt` and
`meta_file_train_set.csv` to `feats_facelora/` (`stage.ipy` lines 142-146). `scratch` skips top-level names only, so these copies stay
in the (private) artifact. No dev file is read or copied.

## Runner metadata changed after review (18/11/2026; `stage.json` only, no code change)

| change | reason | effect on numbers |
|---|---|---|
| `skip_with` removed | the runner serves `skip_with` as a saved copy of the stage's KERNEL output (`runner/ffcore.py` `skip_with_view`, `locate`); `FF_DATA/01_collected/...` holds the exported dataset, not the kernel layout. The stage is skipped by default anyway (collect stages run only when forced) | none |
| `env` pins `FLAG_LOCAL="0"` and `FLAG_SMOKE="0"` | a value left in the caller's environment could switch the notebook into its local / smoke mode during a real run (the code compares with `== "1"`); `smoke_env` still sets `"1"` under `FF_SMOKE=1` | none |
| `private: true` | aligned with `configs/data_map.json` (VoxCeleb2 face PNGs + organiser train-file copies; private FF_DATA share) | none |
| `modes.train` rewritten | a re-collection must force the whole collect closure in one run (fresh `FF_WORK`, `FF_USE_SAVED=collected`); forcing one stage alone mixes new rows with the old frames / audio cache | none |
| `verify` = size of `vox_frames_png.zip` (1 576 077 666 B, informational) | the former prose `expect` on a zip could only raise a JSON error | none |
| `license` on every `external_sources` entry, licence notes on `models` | input for `LICENSES.md` (WP-K); uncertain items marked 'not re-checked' | none |
| internal references labelled | design notes / team logs are internal provenance, not shipped; `CHUA CHAC` -> `unverified` | none |
