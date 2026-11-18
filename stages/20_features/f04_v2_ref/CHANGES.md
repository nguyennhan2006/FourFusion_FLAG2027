# CHANGES: stage `f04_v2_ref`

Original code: kaggle/flag-2027-04.ipynb (git blob ccde784c, commit 8544b3b) (exact-executed-copy). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 4, line 3 | `dev-exclusion` | `N_CROPS = 3                      # duration-matched crops per train u…` -> `N_CROPS = 0                      # FF dev-exclusion (CHANGES.md #1): …` | the crop step chooses crop lengths from the duration distribution of the dev Bangla clips (DESIGN §4.5: n_crops=0 is mandatory) | none on voice_ecapa192_train / voice_ecapa6144_train (the crop arrays voice_ecapa*crop_train are no longer written; no stage reads them) |
| 2 | cell 4, line 9 | `dev-exclusion` | `root = X.media_root(IN, WORKDIR / "raw") ⏎ S = X.index_splits(root) ⏎ ` -> `# --- FF dev-exclusion (CHANGES.md #2): index ONLY the organiser trai…` | the original unpacked every input zip and indexed the 4 dev lists; now only train_English.txt, same rows / order | none on train arrays |
| 3 | cell 5, line 2 | `dev-exclusion` | `CONS = X.check_batch_consistency(ecapa, S, n=12 if LOCAL else 24)` -> `CONS = X.check_batch_consistency(ecapa, {"no_gender/English": S["trai…` | the batched-vs-single check read 24 dev English clips; it now reads 24 train clips | none (check only) |
| 4 | cell 6, line 3 | `path` | `given = pd.read_csv(sorted(IN.rglob("train_English_voices.csv"))[0], …` -> `given = pd.read_csv(sorted(list(IN.rglob("train_English_voices.csv"))…` | the original raw dataset was attached unpacked; with the zipped organiser data the csv only exists after unpacking (under `root`) | none |
| 5 | cell 8, lines 1-4 | `dev-exclusion` | `SSL_COS = {} ⏎ for model_id, short in SSL: ⏎ try: ⏎ SSL_COS[short] = X.extract_ssl(S, OUT, model_id, short)` -> `class _FFTrainAsBangla(dict)` (answers S["no_gender/Bangla"] with the train list) + `X.extract_ssl(_FFTrainAsBangla(S), OUT, model_id, short)` | the embedded flag_extract.extract_ssl (sha256 75fd7ac2...) runs its 4-clip fp16 self-check on S["no_gender/Bangla"] before its per-split loop; with the train-only index (#2) that lookup raised KeyError, cell 8's try/except caught it and no SSL array was written (the original wrote voice_wavlm_train and voice_xlsr_train, (6485, 25, 1024)); same idea as #3 and as f10_models #3 | none on numbers: the SSL train arrays are produced again; the self-check (logged, ssl_fp16_cos) reads 4 train clips instead of 4 dev Bangla clips |

## Other differences from the original run (inputs, wiring, environment)

- **wiring**: The two SSL encoders (wavlm-large, xls-r-300m) are kept as in the original although no stage reads them; change #5 is what keeps them running under the train-only index (before it they failed with a caught KeyError).
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude). Today runner/ffcore.py build_vin applies only the configs/data_map.json exclude, so the archive is still visible; the code never unpacks or reads it (ZIP_PREFIXES / train-only index). With raw-v4 mounted unpacked (a hand-attached Kaggle dataset), find_splits still reads the dev list files (pd.read_csv of the list and an existence check of its media) before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded. Mount raw-v4 as the zipped organiser release (FF_DATA/00_raw/btc_mavceleb_v4).

## Notes

- Optional stage: run it only to show that FLAG_04 and FLAG_12 give the same train ArcFace / ECAPA arrays.

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats/feats_manifest.json` `files` = {"face_arcface_detok_train.npy": [6485], "face_arcface_train.npy": [6485, 512], "voice_ecapa192_train.npy": [6485, 192], "voice_ecapa6144_train.npy": [6485, 6144]}: the ArcFace / ECAPA train arrays of the original run with their shapes (from kaggle/output/feats_v2/feats_manifest.json (created 2026-09-25 13:15)); the crop arrays (n_crops = 0) and the optional SSL arrays are not required.
- verify (informational): `feats/face_arcface_train.npy` sha256 vs the saved copy `55491b54180e116c...`.
- verify (informational): `feats/voice_ecapa192_train.npy` sha256 vs the saved copy `b3bd0a6430412e28...`.
- verify (informational): `feats/voice_ecapa6144_train.npy` sha256 vs the saved copy `195490e23b3c7bbf...`.
- verify (warning only): `feats/feats_manifest.json` `files`: optional SSL train arrays of the original run (warning only; the original catches an SSL failure).
