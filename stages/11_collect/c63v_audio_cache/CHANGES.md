# CHANGES: `c63v_audio_cache`

Baseline: `original/FLAG_63v_audio_cache.ipynb` (builder-regenerated; run `duongphihoang/flag-2027-nb-63v-audio-cache`).

How `stage.ipy` is made: release/flag2027_e2e/build_notebook.py stage_cells(): code cells of the original notebook in order, empty ones skipped, CRLF -> LF, cells whose title (first line, or the name in a leading print("[cell]/[boot] ..."), cut to 90 chars) matches a drop_cells regex removed, then every patch applied in list order (each must match exactly `count` times), then ''.join(cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n' for cell in cells), written UTF-8 with LF. No global patch is assumed.

Original code cells: 7; kept: 7; dropped: 0; patches: 3.

## Patches (in application order)

| # | id | where (notebook cell / code cell) | type | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | `hf_revision_vb` | 6 / 4 | pin | HF revision parameter (G27) for the 8 VoxBlink part repos; unset = original URL | none |
| 2 | `hf_revision_vox` | 8 / 5 | pin | HF revision parameter (G27); unset = original URL | none |
| 3 | `resume_from_env` | 2 / 2 | path | AUDIO_TIME_LIMIT_H (default 10.5) stops the run with a partial `audio_cache/`; the original resumed by attaching its own output, which the runner cannot mount (its `/kaggle/input` view holds only the declared inputs). `AUDIO_RESUME_FROM` = a folder that contains the earlier `audio_cache/`. Unset = the original search | none |

Exact old / new text of every patch: `stage.json` → `patches`. Reverse check: replacing every `new` by its `old` in `stage.ipy` gives back the kept original cells byte for byte (verified when the stage was built, 18/11/2026).

## Upload

Upload switch(es) ['AUDIO_UPLOAD'] must stay unset (`neutralize_upload`); the code is unchanged.

## Deliberately unchanged

- `!pip` lines (unpinned packages as in the run): `pip_policy` original; the pinned environment specification pins versions (FF_PIP=pinned; Docker optional).
- Selection / sampling code and every threshold: verbatim.
- The cache key layout (bare `video` id) and the old per-source LID rule: required for the ORIGINAL LoRA C1 of 13.54.

## Runner metadata changed after review (18/11/2026; `stage.json` only, no code change)

| change | reason | effect on numbers |
|---|---|---|
| `skip_with` removed | the runner serves `skip_with` as a saved copy of the stage's KERNEL output (`runner/ffcore.py` `skip_with_view`, `locate`); `FF_DATA/01_collected/...` holds the exported dataset, not the kernel layout. The stage is skipped by default anyway (collect stages run only when forced) | none |
| `env` pins `FLAG_LOCAL="0"` and `FLAG_SMOKE="0"` | a value left in the caller's environment could switch the notebook into its local / smoke mode during a real run (the code compares with `== "1"`); `smoke_env` still sets `"1"` under `FF_SMOKE=1` | none |
| `modes.train` rewritten | a re-collection must force the whole collect closure in one run (fresh `FF_WORK`, `FF_USE_SAVED=collected`); forcing one stage alone mixes new rows with the old frames / audio cache | none |
| `post_checks`: the manifest's own `complete` flag(s) == true (fatal) | a run stopped by the time limit must not be accepted; outside Kaggle raise the limit in the caller's environment (e.g. 48) | none |
| `verify` expect: nested `wiring` (vb mean cosines), prose `files` moved to notes | the runner compares dict keys literally | none |
| `license` on every `external_sources` entry, licence notes on `models` | input for `LICENSES.md` (WP-K); uncertain items marked 'not re-checked' | none |
| internal references labelled | design notes / team logs are internal provenance, not shipped; `CHUA CHAC` -> `unverified` | none |
