# CHANGES: `c46_voxblink`

Baseline: `original/flag-2027-nb-46.ipynb` (exact-executed-copy; run `unknown slug (2 sessions, 9-10/10/2026; session 2 attached session 1's output datasetVersion 20511943)`).

How `stage.ipy` is made: release/flag2027_e2e/build_notebook.py stage_cells(): code cells of the original notebook in order, empty ones skipped, CRLF -> LF, cells whose title (first line, or the name in a leading print("[cell]/[boot] ..."), cut to 90 chars) matches a drop_cells regex removed, then every patch applied in list order (each must match exactly `count` times), then ''.join(cell.rstrip('\n') + '\n# <<<FF-CELL>>>\n' for cell in cells), written UTF-8 with LF. No global patch is assumed.

Original code cells: 9; kept: 9; dropped: 0; patches: 2; inserted cells: 1.

## Patches (in application order)

| # | id | where (notebook cell / code cell) | type | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | `hf_revision` | 10 / 6 | pin | HF revision parameter (G27) for the 10 VoxBlink repos (annotations, progress map, 8 media parts); unset = original URLs (FF_PartURL.format returns exactly the original string when no revision is given) | none |
| 2 | `resume_from_env` | 10 / 6 | path | the original run needed 2 Kaggle sessions (session 2 resumed from session 1's attached output); under the runner the stage has no inputs, so a second session reaches the first session's `feats_vb/` through `VB_RESUME_FROM` (a folder that contains it). Unset = the original search (attached inputs only) | none |

Exact old / new text of every patch: `stage.json` → `patches`. Reverse check: replacing every `new` by its `old` in `stage.ipy` gives back the kept original cells byte for byte (verified when the stage was built, 18/11/2026).

## Deliberately unchanged

- `!pip` lines (unpinned packages as in the run): `pip_policy` original; the pinned environment specification pins versions (FF_PIP=pinned; Docker optional).
- Selection / sampling code and every threshold: verbatim.
- VoxBlink requests stay anonymous, as in the run (the mirror is not gated).

## Inserted cells (`insert_cells`)

| at | cell | why | effect on numbers |
|---|---|---|---|
| end | `print("[cell] ff: completeness marker for the runner post_check", flush=True)` | reads `feats_vb/feats_manifest.json` and writes `ff_completeness.json` (`complete` = `shards_done == shards_needed`) next to `feats_vb/`; `post_checks` make `complete == true` fatal, so a run stopped by `VB_TIME_LIMIT_H` (the original stopped after 473 / 705 shards in session 1) is never accepted as a finished collection | none (adds one small file outside `feats_vb/`) |

## Exactness caveat

Only session 2 of the original run is kept. The CSVs that define the data (`selection.csv`, `timestamps.csv`, `shard_listing.csv`)
and 473 / 705 shard parts were written by session 1, whose code is not kept; session 2 copied and reused them. Session-1 code is
presumed identical (speaker counts 15 252 - 45 = 15 207; builder output dated before the run), not verified: see `stage.json`
`origin.exactness_note`.

## Runner metadata changed after review (18/11/2026; `stage.json` only, no code change)

| change | reason | effect on numbers |
|---|---|---|
| `env` pins `FLAG_LOCAL="0"` | a value left in the caller's environment could switch the notebook into its local / smoke mode during a real run (the code compares with `== "1"`); `smoke_env` still sets `"1"` under `FF_SMOKE=1` | none |
| `modes.train` rewritten | a re-collection must force the whole collect closure in one run (fresh `FF_WORK`, `FF_USE_SAVED=collected`); forcing one stage alone mixes new rows with the old frames / audio cache | none |
| `compute.minutes_original` = null (`minutes_measured_session2` 306, `minutes_estimated_session1` 630) | session 1 has no record; the former 936 mixed a measurement and an estimate | none |
| `run_policy`: 33 failed videos = session 2 only | `STATS` is reset at session start; session-1 failures are not recorded | none |
| `verify` expect uses nested `lid` | the runner compares dict keys literally (no dotted paths) | none |
| `post_checks`: `ff_completeness.json` `complete == true` (fatal) | a truncated `feats_vb/` must not flow into `b_lab_vb`, `c_frames_index`, `l_lid_strict`, `l_dqual_g3`, `c63v` | none |
| `env_doc`: `VB_TIME_LIMIT_H` (set e.g. 48 outside Kaggle, in the caller's environment) and `VB_RESUME_FROM` | one long run outside Kaggle, or two sessions | none |
| `smoke_prerequisite` added (replaces the Docker-only todo) | install the two pip lines of the install cell first; no Docker needed | none |
| `license` on every `external_sources` entry, licence notes on `models` | input for `LICENSES.md` (WP-K); uncertain items marked 'not re-checked' | none |
| internal references labelled | design notes / team logs are internal provenance, not shipped; `CHUA CHAC` -> `unverified` | none |

## Comment edit (20/11/2026)

Patches 2..4 of `stage.json` (ids `comment_team_label_*`) change comment text only: another team's label is replaced by a neutral wording (public naming rule, `docs/NAMING.md`). No code, path, parameter or number changes; the original notebook in `original/` is unchanged.
- comment now reads `A teammate's VoxBlink pipeline (CB` (1x; old text in the `stage.json` patch)
- comment now reads `encoder family of a teammate's VoxBlink system` (1x; old text in the `stage.json` patch)
- comment now reads `(commit 9fecd6c, as in the teammate pipeline)` (1x; old text in the `stage.json` patch)
- `stage.ipy` sha256 1f1bd43ddd61ab09... -> e3324e972eff554c...
