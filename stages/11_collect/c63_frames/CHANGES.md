# CHANGES: `c63_frames`

Baseline: `original/flag-2027-nb-63-frames.ipynb` (exact-executed-copy; run `onnhqunhh/flag-2027-nb-63-frames`).

How `stage.ipy` is made: release/flag2027_e2e/build_notebook.py stage_cells(): code cells of the original notebook in order, empty ones skipped, CRLF -> LF, cells whose title (first line, or the name in a leading print("[cell]/[boot] ..."), cut to 90 chars) matches a drop_cells regex removed, then every patch applied in list order (each must match exactly `count` times), then ''.join(cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n' for cell in cells), written UTF-8 with LF. No global patch is assumed.

Original code cells: 6; kept: 6; dropped: 0; patches: 5.

## Patches (in application order)

| # | id | where (notebook cell / code cell) | type | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | `hf_revision_vb` | 6 / 4 | pin | HF revision parameter (G27) for the 8 VoxBlink part repos; unset = original URL | none |
| 2 | `hf_revision_vox2` | 8 / 5 | pin | HF revision parameter (G27) + token header for the gated repo; unset = original string | none |
| 3 | `auth_remote_r` | 8 / 5 | url | Remote(): resolve each part with the token (ProgramComputer/voxceleb is gated now); no HF_TOKEN = original request | none |
| 4 | `auth_remote_q` | 8 / 5 | url | Remote._get(): re-resolve an expired signed URL with the token (same reason) | none |
| 5 | `resume_from_env` | 2 / 2 | path | FRAMES_TIME_LIMIT_H (default 11) stops the run with a partial `frames/`; the original resumed by attaching its own output, which the runner cannot mount (its `/kaggle/input` view holds only the declared inputs). `FRAMES_RESUME_FROM` = a folder that contains the earlier `frames/`. Unset = the original search | none |

Exact old / new text of every patch: `stage.json` → `patches`. Reverse check: replacing every `new` by its `old` in `stage.ipy` gives back the kept original cells byte for byte (verified when the stage was built, 18/11/2026).

## Deliberately unchanged

- `!pip` lines (unpinned packages as in the run): `pip_policy` original; the pinned environment specification pins versions (FF_PIP=pinned; Docker optional).
- Selection / sampling code and every threshold: verbatim.

## Runner metadata changed after review (18/11/2026; `stage.json` only, no code change)

| change | reason | effect on numbers |
|---|---|---|
| `skip_with` removed | the runner serves `skip_with` as a saved copy of the stage's KERNEL output (`runner/ffcore.py` `skip_with_view`, `locate`); `FF_DATA/01_collected/...` holds the exported dataset, not the kernel layout. The stage is skipped by default anyway (collect stages run only when forced) | none |
| `env` pins `FLAG_LOCAL="0"` and `FLAG_SMOKE="0"` | a value left in the caller's environment could switch the notebook into its local / smoke mode during a real run (the code compares with `== "1"`); `smoke_env` still sets `"1"` under `FF_SMOKE=1` | none |
| `modes.train` rewritten | a re-collection must force the whole collect closure in one run (fresh `FF_WORK`, `FF_USE_SAVED=collected`); forcing one stage alone mixes new rows with the old frames / audio cache | none |
| `post_checks`: the manifest's own `complete` flag(s) == true (fatal) | a run stopped by the time limit must not be accepted; outside Kaggle raise the limit in the caller's environment (e.g. 48) | none |
| `verify` marked informational | a re-collection is not byte-identical | none |
| `license` on every `external_sources` entry, licence notes on `models` | input for `LICENSES.md` (WP-K); uncertain items marked 'not re-checked' | none |
| internal references labelled | design notes / team logs are internal provenance, not shipped; `CHUA CHAC` -> `unverified` | none |
