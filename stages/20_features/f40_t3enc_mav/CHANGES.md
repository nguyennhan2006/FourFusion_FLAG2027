# CHANGES: stage `f40_t3enc_mav`

Original code: Experiment/T3ENC-01/flag-2027-nb-40-t3enc-01.ipynb (exact-executed-copy). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 1, line 18 | `dev-exclusion` | `SPLIT_FILTER = None            # e.g. ["no_gender/English"]; None = e…` -> `SPLIT_FILTER = ["train"]      # FF dev-exclusion: organiser train lis…` | no dev list is indexed or encoded (DESIGN §2.8; as TRAIN_ONLY of release/flag2027_e2e/stages.py:151) | none on the train / MAV-Celeb / auxiliary arrays (each split is encoded on its own); the dev arrays and dev index files are no longer written |
| 2 | cell 1, line 19 | `dev-exclusion` | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")             #…` -> `ZIP_PREFIXES = ("train_set", "eval", "test")                        #…` | with zipped raw data the original unpacked dev_set.zip too (then SPLIT_FILTER dropped it); now the dev archive is never read | none |
| 3 | cell 2, line 5 | `pin` | `    !pip -q install --no-deps "git+https://github.com/wenet-e2e/wespe…` -> `    # FF environment patch (CHANGES.md): the team's own pip fix of 29…` | on the current Kaggle image the openai CLIP build fails (its setup.py needs pkg_resources) and, in this shared pip call, takes wespeaker down with it (BEST/v1977/README.md §6). Same packages and commits (wespeaker@9fecd6c, CLIP@d50d76d), one pip call each; CLIP from a git clone at d50d76d when the pip build fails | none (same code at the same commits) |

## Other differences from the original run (inputs, wiring, environment)

- **dev-exclusion** (environment, no code change): stage.json env pins `FLAG_SPLITS=train`. The original code lets FLAG_SPLITS override the patched SPLIT_FILTER in a smoke / local run; the runner applies the stage env after the caller's environment, so no outside value can turn dev extraction back on. Same value as the patched default: effect on numbers none.
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude). Today runner/ffcore.py build_vin applies only the configs/data_map.json exclude, so the archive is still visible; the code never unpacks or reads it (ZIP_PREFIXES / train-only index). With raw-v4 mounted unpacked (a hand-attached Kaggle dataset), find_splits still reads the dev list files (pd.read_csv of the list and an existence check of its media) before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded. Mount raw-v4 as the zipped organiser release (FF_DATA/00_raw/btc_mavceleb_v4).
- **dev-exclusion** (consequence of SPLIT_FILTER): the encoder self-checks (`self_check`: batch-vs-single and fp16-vs-fp32 cosine > 0.995 on the 3 longest clips of the first non-train split, else of train) now run on train clips; the original took a dev list. Effect on numbers: none (check only). Check behaviour on train clips: untested. A failing check puts the encoder in FAILED and skips it without stopping the run; the post_check on feats_manifest.json `files` below turns that into a stage failure.

## Notes

- FaRL here is epoch 16 (FARL16_URL, 652 165 542 bytes) = the FaRL of the 13.54 system; f10_models' FaRL is epoch 64 (wiring only).

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats_t3enc/feats_manifest.json` `files` = 12 entries: every v4-train and MAV-Celeb v1 / v2 array of the original run with its shape (12 arrays; values from Experiment/T3ENC-01/feats_t3enc/feats_manifest.json (created 2026-10-07 16:01), dev entries left out). A failed encoder (recorded in 'failed' and skipped) or a skipped MAV-Celeb source (ExtRows error only logged) leaves an array out, so the stage fails here instead of hours later downstream.
- post_check (fatal): `feats_t3enc/feats_manifest.json` `external` = {"v1_complete": [7580, 19556], "v2_complete": [8953, 20715]}: MAV-Celeb v1 / v2 [face rows, voice rows] of the original manifest.
- post_check (fatal): `feats_t3enc/feats_manifest.json` `splits/train` = 6485: organiser train rows.
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).
