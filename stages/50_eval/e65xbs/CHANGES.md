# e65xbs: changes from the original run

**One code change** (dev-exclusion, fix round 18/11). `stage.ipy` = the 13 code cells of `original/01_flag-2027-nb-65xbs-b3sx-eval.ipynb` with the `stage.json` patch `no_dev_zip` (sha256 `35ce6900f6c12725e5c1233b76ed50a314ff93612d9c3cce039b512c77fd5637`). Without it, the code cells are byte-identical to `release/flag2027_e2e/code/e65xbs.ipy` (sha256 `b129385cf497762f5e6ee2b520f0416da844b210ee8b1e2d16032f5c8755a149`, the code of the run). No patch of `release/flag2027_e2e/stages.py` applies (its global ImageBind URL patch has no hit in this notebook).

| # | where | old → new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | code cell 1/13 "01 what to run" (`stage.ipy` line 21) | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")` → `ZIP_PREFIXES = ("train_set", "eval", "test")` plus a comment (a `stage.json` patch, id `no_dev_zip`, `count` 1, `in_release: false`; same text as the patch of `f47b_screen2`) | the repository mounts `datasets/ngnnguynchn/flag2027-raw-v4-p` as the organisers' files in `FF_DATA/00_raw/btc_mavceleb_v4` (`train_set.zip`, `dev_set.zip`, meta csv), not as the tree Kaggle had auto-extracted for the original run (log: `…/flag2027-raw-v4-p/train_set/train_set/train_set/train_English.txt`, no `unpacking` line). No train list is then found, so the verbatim fallback `unpack_zips()` runs, and without the patch it unpacks `dev_set.zip` too (9 757 members, ~990 MB) into `/kaggle/working/raw`. With the patch the dev archive is never opened (DESIGN §2.8). Found by the WP-G review of 18/11. | none: `train_set.zip` is unpacked by the same fallback (same train list and media as the auto-extracted copy, other path; the log gains an `unpacking …/train_set.zip` line) and the verbatim last cell removes `raw/` (`shutil.rmtree(WORKDIR / "raw")`), so the output holds no organiser media; dev lists were already skipped by `ONLY_UNDER` | dev-exclusion |

Original run: `thanhhuyen11/flag-2027-nb-65xbs-b3sx-eval`, pushed 11/11/2026 20:46 (UTC+7) after `kaggle/patch_evalphase_checks.py`
(kapi_runs line 1466, pushed notebook md5 `736f0364842831fcaef1e246180a0c8e`); the team-package copy in `original/` has the same code
cells.

## Decisions that are not code changes

| item | decision | effect on numbers |
|---|---|---|
| mounts | every dataset input has `mount = <slug>`: the log shows that this run saw the old Kaggle layout `/kaggle/input/<slug>` (three datasets printed; the other six assumed the same). The code walks `/kaggle/input` recursively. | none expected |
| inputs | the nine datasets of the run, as pushed (`original/kernel-metadata.json`), including the four only used to index the adapter's sampling pool (cell 06; no training in this run) | none |
| dev data | `ONLY_UNDER = 'evaluation_phase'` (code cell 01) skips every list outside the Evaluation data except `train_English.txt`; with change 1, `dev_set.zip` of `flag2027-raw-v4-p` is mounted but never opened | none |
| role | wiring only: the 13.54 numbers do not depend on this stage (see `stage.json` `role`) | none |
| output content | the output folder holds every file copied from the adapter dataset (code cell 10), including the VoxBlink face array `vb_face_clip*.npy` and `facelora_state.pt` (adapter learned with VoxBlink frames): declared in `stage.json` `outputs.contains` (`voxblink: true`); the run output is private storage only. WP-A: set `voxblink: true` on this run-output entry of `configs/data_map.json`. | none |

## Comment edit (20/11/2026)

Patches 1..3 of `stage.json` (ids `comment_team_label_*`) change comment text only: another team's label is replaced by a neutral wording (public naming rule, `docs/NAMING.md`). No code, path, parameter or number changes; the original notebook in `original/` is unchanged.
- comment now reads `A teammate's VoxBlink pipeline (CB` (1x; old text in the `stage.json` patch)
- comment now reads `encoder family of a teammate's VoxBlink system` (1x; old text in the `stage.json` patch)
- comment now reads `encoder family of a teammates VoxBlink s` (1x; old text in the `stage.json` patch)
- `stage.ipy` sha256 0a8f1ed64d83bd85... -> 35ce6900f6c12725...
