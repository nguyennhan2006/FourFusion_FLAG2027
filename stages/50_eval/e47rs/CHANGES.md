# e47rs: changes from the original run

**One code change** (dev-exclusion, fix round 18/11). `stage.ipy` = the 9 code cells of `original/03_flag-2027-nb-47rs-screen2-eval.ipynb` with the `stage.json` patch `no_dev_zip` (sha256 `6e5f3456c81b8ac69da0a7c8d094d78bca43af16976019510c236d59d8c574e1`). Without it, the code cells are byte-identical to `release/flag2027_e2e/code/e47rs.ipy` (sha256 `a29f667d4ac22ab85edd5b4b93d2224be729d6370c7b96342892ec9203868704`, the code of the run).

| # | where | old → new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | code cell 1/9 "01 what to run" (`stage.ipy` line 21) | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")` → `ZIP_PREFIXES = ("train_set", "eval", "test")` plus a comment (a `stage.json` patch, id `no_dev_zip`, `count` 1, `in_release: false`; same text as the patch of `f47b_screen2`) | the repository mounts `datasets/ngnnguynchn/flag2027-raw-v4-p` as the organisers' files in `FF_DATA/00_raw/btc_mavceleb_v4` (`train_set.zip`, `dev_set.zip`, meta csv), not as the tree Kaggle had auto-extracted for the original run (log: `…/flag2027-raw-v4-p/train_set/train_set/train_set/train_English.txt`, no `unpacking` line). No train list is then found, so the verbatim fallback `unpack_zips()` runs, and without the patch it unpacks `dev_set.zip` too (9 757 members, ~990 MB) into `/kaggle/working/raw`. With the patch the dev archive is never opened (DESIGN §2.8). Found by the WP-G review of 18/11. | none: `train_set.zip` is unpacked by the same fallback (same train list and media as the auto-extracted copy, other path; the log gains an `unpacking …/train_set.zip` line) and the verbatim last cell removes `raw/` (`shutil.rmtree(WORKDIR / "raw")`), so the output holds no organiser media; dev lists were already skipped by `ONLY_UNDER` | dev-exclusion |

Original run: `thanhhuyen11/flag-2027-nb-47rs-screen2-eval`, pushed by hand 11/11/2026 15:53 (UTC+7) (kapi_runs line 1443, pushed
notebook md5 `8efafbebda567e67b07d4839ddd7c9e6`); the team-package copy in `original/` has the same code cells.

## Decisions that are not code changes

| item | decision | effect on numbers |
|---|---|---|
| export dropped | `release/flag2027_e2e/stages.py:100` re-exported `feats_screen2/` as `datasets/ngnnguynchn/flag2027-feats-screen2-p`. This repository keeps one producer per artifact: that dataset is produced by stage `f47b_screen2` (`20_features`), the run whose output the RFX training actually read. The equivalence (same 21 files, cos 1.000000, not byte-identical) is recorded in `stage.json` `alt_export`. | none on the 13.54 zip (this stage's own output feeds e112f1sx and e163i as a kernel artifact) |
| mounts | default (`mount` = artifact id): the log shows the new Kaggle layout `/kaggle/input/datasets/<owner>/<slug>` | none |
| dev data | `ONLY_UNDER = 'evaluation_phase'` (code cell 01): dev lists are not read; with change 1, `dev_set.zip` is never opened | none |
| producer preference | DESIGN §4.8 still says `e47rs` exports `feats-screen2-p`; this repository does not (row 'export dropped'). `configs/pipeline_1354.json` `producer_preference` still names `e47rs` for `infer` / `assemble`; it has no effect today (`producer_of()` uses the preference only among several candidates, and `f47b_screen2` is the only producer), but it contradicts this stage. WP-A: remove that entry or make it `{default: f47b_screen2}`; WP-K: note the deviation from DESIGN §4.8. | none |

## Comment edit (20/11/2026)

Patches 1..5 of `stage.json` (ids `comment_team_label_*`) change comment text only: another team's label is replaced by a neutral wording (public naming rule, `docs/NAMING.md`). No code, path, parameter or number changes; the original notebook in `original/` is unchanged.
- comment now reads `A teammate's VoxBlink pipeline (CB` (1x; old text in the `stage.json` patch)
- comment now reads `encoder family of a teammate's VoxBlink system` (1x; old text in the `stage.json` patch)
- comment now reads `encoder family of a teammates VoxBlink s` (1x; old text in the `stage.json` patch)
- comment now reads `(commit 9fecd6c, as in the teammate pipeline)` (2x; old text in the `stage.json` patch)
- comment now reads `A teammate's voice branch` (1x; old text in the `stage.json` patch)
- `stage.ipy` sha256 2d667b0ad5ecf8ab... -> 6e5f3456c81b8ac6...
