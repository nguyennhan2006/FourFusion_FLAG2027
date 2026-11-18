# CHANGES: stage `f14_av`

Original code: kaggle/output/flag-2027-nb-14-other-image-audio-models (1).ipynb (exact-executed-copy). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 1, line 20 | `dev-exclusion` | `SPLIT_FILTER = None                                      # e.g. {"tra…` -> `SPLIT_FILTER = ["train"]                                 # FF dev-exc…` | no dev list is indexed or encoded (DESIGN §2.8) | none on train / MAV-Celeb arrays; dev arrays are no longer written |
| 2 | cell 5, line 42 | `url` | `https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth` -> `https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade…` | Meta's link to imagebind_huge.pth is dead since 12/11/2026; same file (sha256 d6f6c22b...2157) on a pinned Hugging Face mirror, as release/flag2027_e2e/stages.py GLOBAL_PATCHES | none (same file) |
| 3 | cell 3, line 102 | `dev-exclusion` | `            if (inv["has_faces"] and inv["has_voices"] and (inv["has_…` -> `            if lname.startswith("dev_set"):      # FF dev-exclusion (…` | the archive auto-extraction would unpack dev_set.zip too | none |

## Other differences from the original run (inputs, wiring, environment)

- **wiring**: The optional input flag2027-feats-models (for the ImageBind rows of the zero-shot probe) is not attached: that probe row is skipped; no array changes.
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude). Today runner/ffcore.py build_vin applies only the configs/data_map.json exclude, so the archive is still visible; the code never unpacks or reads it (ZIP_PREFIXES / train-only index). With raw-v4 mounted unpacked (a hand-attached Kaggle dataset), find_splits still reads the dev list files (pd.read_csv of the list and an existence check of its media) before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded. Mount raw-v4 as the zipped organiser release (FF_DATA/00_raw/btc_mavceleb_v4).
- **wiring** (saved view): skip_with now also points at the frozen lab-data copy of this output (`02_intermediate/lab_data/lab/kaggle/output/feats_av`, 19 files, and the package-E layout of the same folder), as DESIGN §4.5 says. Effect on numbers: none (b_lab_data packs those same files).

## Notes

- Wiring only: kept so that the lab-data bundle can be rebuilt with the same file list.

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats_av/feats_manifest.json` `files` = 19 entries: the 19 arrays b_lab_data packs (b_lab_data/expected_members.json lab/kaggle/output/feats_av/*: AudioCLIP, ImageBind-align, LanguageBind, MAV-Celeb row selections) with their shapes from kaggle/output/feats_av/feats_manifest.json (created 2026-09-29 09:05). A failed encoder or a skipped MAV-Celeb source leaves one out, so the stage fails here.
- verify (warning only): `feats_av/feats_manifest.json` `files`: all 28 arrays of the original run (CLIP B/32, B/16, Wav2CLIP included; no stage reads those three).
- post_check (fatal): `feats_av/feats_manifest.json` `external` = {"v1_complete": [1400, 2229], "v2_complete": [1680, 2626]}: MAV-Celeb v1 / v2 [face rows, voice rows] of the original manifest.
- post_check (fatal): `feats_av/feats_manifest.json` `splits/train` = 6485: organiser train rows.
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).
