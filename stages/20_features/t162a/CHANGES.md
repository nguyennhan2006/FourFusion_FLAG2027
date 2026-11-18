# CHANGES: stage `t162a`

Original code: E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/11_flag-2027-nb-162a-rfs.ipynb (exact-package). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 1, line 4 | `upload` | `os.environ['RFS_UPLOAD'] = 'nguynchnngn1/flag2027-feats-rfs-a-p'` -> `# FF: no upload to a Kaggle dataset (the output becomes an artifact).…` | the original run uploaded its output to a Kaggle dataset; here the output is a pipeline artifact (as release/flag2027_e2e/stages.py noup) | none |
| 2 | cell 2, line 20 | `dev-exclusion` | `SPLIT_FILTER = None            # e.g. ["no_gender/English"]; None = e…` -> `SPLIT_FILTER = ["train"]      # FF dev-exclusion: organiser train lis…` | no dev list is indexed or encoded (DESIGN §2.8; as TRAIN_ONLY of release/flag2027_e2e/stages.py:151) | none on the train / MAV-Celeb / auxiliary arrays (each split is encoded on its own); the dev arrays and dev index files are no longer written |
| 3 | cell 2, line 21 | `dev-exclusion` | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")             #…` -> `ZIP_PREFIXES = ("train_set", "eval", "test")                        #…` | with zipped raw data the original unpacked dev_set.zip too (then SPLIT_FILTER dropped it); now the dev archive is never read | none |
| 4 | cell 6, line 193; cell 7, line 20 | `url` | `https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth` -> `https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade…` | Meta's link to imagebind_huge.pth is dead since 12/11/2026; same file (sha256 d6f6c22b...2157) on a pinned Hugging Face mirror, as release/flag2027_e2e/stages.py GLOBAL_PATCHES | none (same file) |

## Other differences from the original run (inputs, wiring, environment)

- **dev-exclusion**: ZIP_PREFIXES patch is an addition to the release patches (all lanes).
- **dev-exclusion** (environment, no code change): stage.json env pins `FLAG_SPLITS=train`. The original code lets FLAG_SPLITS override the patched SPLIT_FILTER in a smoke / local run; the runner applies the stage env after the caller's environment, so no outside value can turn dev extraction back on. Same value as the patched default: effect on numbers none.
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude). Today runner/ffcore.py build_vin applies only the configs/data_map.json exclude, so the archive is still visible; the code never unpacks or reads it (ZIP_PREFIXES / train-only index). With raw-v4 mounted unpacked (a hand-attached Kaggle dataset), find_splits still reads the dev list files (pd.read_csv of the list and an existence check of its media) before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded. Mount raw-v4 as the zipped organiser release (FF_DATA/00_raw/btc_mavceleb_v4).
- **dev-exclusion** (consequence of SPLIT_FILTER): the encoder self-checks (`self_check`: batch-vs-single and fp16-vs-fp32 cosine > 0.995 on the 3 longest clips of the first non-train split, else of train) now run on train clips; the original took a dev list. Effect on numbers: none (check only). Check behaviour on train clips: untested. A failing check puts the encoder in FAILED and skips it without stopping the run; the post_check on feats_manifest.json `files` below turns that into a stage failure.

## Notes

- The RF core flag_final_core_092.py is embedded (zlib + base64) and checked by sha256 (25abe24d...); the ImageBind URL lives in the notebook's URL table, not in the core.
- Release's T2 run reproduced 16.18 with the corrected C1; that does not concern this stage (no LoRA here).

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats_rfs_a/feats_manifest.json` `files` = 20 entries: every v4-train and MAV-Celeb v1 / v2 array of the original run with its shape (20 arrays; values from package E nguynchnngn1__flag2027-feats-rfs-a-p.zip :: feats_manifest.json (created 2026-11-03 16:38), dev entries left out). A failed encoder (recorded in 'failed' and skipped) or a skipped MAV-Celeb source (ExtRows error only logged) leaves an array out, so the stage fails here instead of hours later downstream.
- post_check (fatal): `feats_rfs_a/feats_manifest.json` `external` = {"v1_complete": [7580, 19556], "v2_complete": [8953, 20715]}: MAV-Celeb v1 / v2 [face rows, voice rows] of the original manifest.
- post_check (fatal): `feats_rfs_a/feats_manifest.json` `splits/train` = 6485: organiser train rows.
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).
