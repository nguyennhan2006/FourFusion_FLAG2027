# e65xcs: changes from the original run

**One code change** (dev-exclusion, fix round 18/11). `stage.ipy` = the 13 code cells of `original/02_flag-2027-nb-65xcs-c5sx-eval.ipynb` with the `stage.json` patch `no_dev_zip` (sha256 `e948aca445dc5d1a1b577973b550745a62c46e2a3070c5fef76b1faadd11f6ec`). Without it, the code cells are byte-identical to `release/flag2027_e2e/code/e65xcs.ipy` (sha256 `88fc960d7509dcbea17fca44c0227b9df653d08f57f29abb18f3012ccda26e97`, the code of the run). No patch of `release/flag2027_e2e/stages.py` applies (its global ImageBind URL patch has no hit in this notebook).

| # | where | old → new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | code cell 1/13 "01 what to run" (`stage.ipy` line 21) | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")` → `ZIP_PREFIXES = ("train_set", "eval", "test")` plus a comment (a `stage.json` patch, id `no_dev_zip`, `count` 1, `in_release: false`; same text as the patch of `f47b_screen2`) | the repository mounts `datasets/ngnnguynchn/flag2027-raw-v4-p` as the organisers' files in `FF_DATA/00_raw/btc_mavceleb_v4` (`train_set.zip`, `dev_set.zip`, meta csv), not as the tree Kaggle had auto-extracted for the original run (log: `…/flag2027-raw-v4-p/train_set/train_set/train_set/train_English.txt`, no `unpacking` line). No train list is then found, so the verbatim fallback `unpack_zips()` runs, and without the patch it unpacks `dev_set.zip` too (9 757 members, ~990 MB) into `/kaggle/working/raw`. With the patch the dev archive is never opened (DESIGN §2.8). Found by the WP-G review of 18/11. | none: `train_set.zip` is unpacked by the same fallback (same train list and media as the auto-extracted copy, other path; the log gains an `unpacking …/train_set.zip` line) and the verbatim last cell removes `raw/` (`shutil.rmtree(WORKDIR / "raw")`), so the output holds no organiser media; dev lists were already skipped by `ONLY_UNDER` | dev-exclusion |

Original run: `thanhhuyen11/flag-2027-nb-65xcs-c5sx-eval`, last push 11/11/2026 22:05 (UTC+7) of a retry loop that pushed the same
notebook nine times (kapi_runs lines 1468, 1469, 1471–1473, 1476, 1479–1481, all md5 `e79a5615742b2320a13cfb19c2f09d62`); the
team-package copy in `original/` has the same code cells.

## Decisions that are not code changes

| item | decision | effect on numbers |
|---|---|---|
| mounts | every dataset input has `mount = <slug>` (old Kaggle layout seen in the log; three datasets printed, the other six assumed the same). The code walks `/kaggle/input` recursively. | none expected |
| inputs | the nine datasets of the run, as pushed (`original/kernel-metadata.json`) | none |
| dev data | `ONLY_UNDER = 'evaluation_phase'` (code cell 01): dev lists are not read; with change 1, `dev_set.zip` is never opened | none |
| output content | the output folder holds every file copied from the adapter dataset (code cell 10), including the VoxBlink face array `vb_face_clip*.npy` and `facelora_state.pt` (adapter learned with VoxBlink frames): declared in `stage.json` `outputs.contains` (`voxblink: true`); the run output is private storage only. WP-A: set `voxblink: true` on this run-output entry of `configs/data_map.json`. | none |
