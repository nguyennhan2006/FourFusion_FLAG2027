# CHANGES: `c87_frames_bridge`

Baseline: `original/flag-2027-nb-87-bridge.ipynb` (exact-executed-copy; run `onnhqunhh/flag-2027-nb-87-bridge`).

How `stage.ipy` is made: release/flag2027_e2e/build_notebook.py stage_cells(): code cells of the original notebook in order, empty ones skipped, CRLF -> LF, cells whose title (first line, or the name in a leading print("[cell]/[boot] ..."), cut to 90 chars) matches a drop_cells regex removed, then every patch applied in list order (each must match exactly `count` times), then ''.join(cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n' for cell in cells), written UTF-8 with LF. No global patch is assumed.

Original code cells: 3; kept: 2; dropped: 1; patches: 4.

## Dropped cells (`drop_cells`)

| notebook cell | code cell | title |
|---|---|---|
| 3 | 3 | `# upload as a NEW private dataset of this account (kagglehub: private by default); privati` |

Reason: the upload to Kaggle is replaced by the runner's export of /kaggle/working/lora_frames. Type: upload.

## Patches (in application order)

| # | id | where (notebook cell / code cell) | type | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | `no_kagglehub` | 1 / 1 | upload | the upload is replaced by the runner's export; kagglehub may be absent outside Kaggle | none |
| 2 | `stage_in_working` | 2 / 2 | upload | the dataset must be left in /kaggle/working for the runner to export it (exports: lora_frames) | none |
| 3 | `ckpt_optional` | 2 / 2 | wiring | see original_inputs_not_attached: the FLAG_65-b checkpoint is outside the 13.54 chain | none (file never read downstream) |
| 4 | `real_files` | 2 / 2 | upload | symlinks into the stage's virtual /kaggle/input would dangle after the stage; hard link (no extra space) else copy | none |

Exact old / new text of every patch: `stage.json` → `patches`. Reverse check: replacing every `new` by its `old` in `stage.ipy` gives back the kept original cells byte for byte (verified when the stage was built, 18/11/2026).

## Deliberately unchanged

- `!pip` lines (unpinned packages as in the run): `pip_policy` original; the pinned environment specification pins versions (FF_PIP=pinned; Docker optional).

## Inputs (original run -> this stage)

| original attachment (push log line 246) | here | reason | type | effect on numbers |
|---|---|---|---|---|
| kernel `onnhqunhh/flag-2027-nb-45-facelora-farl` | same id (output of `c45_frames_vox2`) | `vox_frames_png.zip` | — | none |
| kernel `onnhqunhh/flag-2027-nb-63-frames` | same id (output of `c63_frames`) | `frames/` | — | none |
| kernel `onnhqunhh/flag-2027-nb-65-facelora3-b` | not attached (`original_inputs_not_attached`) | its `feats_facelora3_b/facelora_ckpt.pt` has no producing stage and is never read by the 13.54 chain; the runner mounts only `inputs`; patch 3 makes the file optional | wiring | none; the stage output has 723 files instead of the published 724 |

c87 needs the kernel outputs of c45 and c63, which are kept nowhere (only renamed copies inside the published dataset): it runs only
in the same run as `c45_frames_vox2` and `c63_frames`.

## Runner metadata changed after review (18/11/2026; `stage.json` only, no code change)

| change | reason | effect on numbers |
|---|---|---|
| `skip_with` removed | the runner serves `skip_with` as a saved copy of the stage's KERNEL output (`runner/ffcore.py` `skip_with_view`, `locate`); `FF_DATA/01_collected/...` holds the exported dataset, not the kernel layout. The stage is skipped by default anyway (collect stages run only when forced) | none |
| `modes.train` rewritten | a re-collection must force the whole collect closure in one run (fresh `FF_WORK`, `FF_USE_SAVED=collected`); forcing one stage alone mixes new rows with the old frames / audio cache | none |
| `optional_inputs` -> `original_inputs_not_attached`; byte-for-byte claim dropped | the runner never reads optional inputs; the FLAG_65-b checkpoint cannot reach the stage | none |
| `verify`: the entry without `file` removed; the manifest hash marked informational | an entry without `file` only reported 'missing None' | none |
| internal references labelled | design notes / team logs are internal provenance, not shipped; `CHUA CHAC` -> `unverified` | none |
