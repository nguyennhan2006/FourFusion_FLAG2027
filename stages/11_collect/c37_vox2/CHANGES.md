# CHANGES: `c37_vox2`

Baseline: `original/fork-of-flag-2027-nb-37.ipynb` (exact-executed-copy; run `unknown slug (Kaggle 'fork of flag-2027-nb-37', executed 2026-10-06; run before kapi logged pushes)`).

How `stage.ipy` is made: release/flag2027_e2e/build_notebook.py stage_cells(): code cells of the original notebook in order, empty ones skipped, CRLF -> LF, cells whose title (first line, or the name in a leading print("[cell]/[boot] ..."), cut to 90 chars) matches a drop_cells regex removed, then every patch applied in list order (each must match exactly `count` times), then ''.join(cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n' for cell in cells), written UTF-8 with LF. No global patch is assumed.

Original code cells: 9; kept: 9; dropped: 0; patches: 4.

## Patches (in application order)

| # | id | where (notebook cell / code cell) | type | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | `imagebind_url` | 6 / 4 | url | Meta's ImageBind link is dead since 12/11/2026; pinned HF mirror of the same file (sha256 d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157), as release/flag2027_e2e GLOBAL_PATCHES and the 12/11 builder | none (same weights; FLAG_37's ImageBind arrays are not used by 13.54 anyway) |
| 2 | `hf_revision` | 9 / 6 | pin | HF revision parameter (gap G27) + the token header used by the two patches below; with FF_HF_REVS / HF_TOKEN unset every string equals the original | none |
| 3 | `auth_meta` | 9 / 6 | url | vox2_meta.csv of the gated repo: send the token (no HF_TOKEN = original anonymous request) | none |
| 4 | `auth_resolve` | 11 / 7 | url | resolve(): signed CDN URL of vox2_mp4_<k>.zip from the gated repo: send the token (the range reads on the signed URL need none) | none |

Exact old / new text of every patch: `stage.json` → `patches`. Reverse check: replacing every `new` by its `old` in `stage.ipy` gives back the kept original cells byte for byte (verified when the stage was built, 18/11/2026).

## Deliberately unchanged

- `!pip` lines (unpinned packages as in the run): `pip_policy` original; the pinned environment specification pins versions (FF_PIP=pinned; Docker optional).
- Selection / sampling code and every threshold: verbatim.
- Model downloads other than ImageBind (speechbrain, ViT-age, w2v2 age/gender, organiser ECAPA, VGG-Face): unpinned as in the run; their arrays are not used by 13.54.

## Runner metadata changed after review (18/11/2026; `stage.json` only, no code change)

| change | reason | effect on numbers |
|---|---|---|
| `skip_with` removed | the runner serves `skip_with` as a saved copy of the stage's KERNEL output (`runner/ffcore.py` `skip_with_view`, `locate`); `FF_DATA/01_collected/...` holds the exported dataset, not the kernel layout. The stage is skipped by default anyway (collect stages run only when forced) | none |
| `env` pins `FLAG_LOCAL="0"` | a value left in the caller's environment could switch the notebook into its local / smoke mode during a real run (the code compares with `== "1"`); `smoke_env` still sets `"1"` under `FF_SMOKE=1` | none |
| `private: true` | aligned with `configs/data_map.json` (VoxCeleb2-derived data; private FF_DATA share) | none |
| `modes.train` rewritten | a re-collection must force the whole collect closure in one run (fresh `FF_WORK`, `FF_USE_SAVED=collected`); forcing one stage alone mixes new rows with the old frames / audio cache | none |
| `verify`: `broken.image` dropped | it is a list of 11 failed reads in `feats_manifest.json`; the runner compares values literally | none |
| `smoke_prerequisite` added | smoke needs the notebook's local mode (`FLAG_LOCAL=1` skips the `!pip` cell): install the two pip lines first; no Docker needed | none |
| `license` on every `external_sources` entry, licence notes on `models` | input for `LICENSES.md` (WP-K); uncertain items marked 'not re-checked' | none |
| internal references labelled | design notes / team logs are internal provenance, not shipped; `CHUA CHAC` -> `unverified` | none |

`vox2_meta.csv` is downloaded into `FLAG_CACHE` (not saved in the output): the FF_DATA copy `raw/voxceleb2_vox2_meta` is raw input with no producing stage by design (see `stage.json` notes). Deliberately not changed into a `save` patch: c37 never runs for 13.54 and `l_groups` reads the FF_DATA copy either way.
