# CHANGES: stage `f103c_tap01`

Original code: kaggle/FLAG_103c_tap01_voice.ipynb (builder-regenerated). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 1, line 19 | `dev-exclusion` | `SPLIT_FILTER = None            # e.g. ["no_gender/English"]; None = e…` -> `SPLIT_FILTER = ["train"]      # FF dev-exclusion: organiser train lis…` | no dev list is indexed or encoded (DESIGN §2.8; as TRAIN_ONLY of release/flag2027_e2e/stages.py:151) | none on the train / MAV-Celeb / auxiliary arrays (each split is encoded on its own); the dev arrays and dev index files are no longer written |
| 2 | cell 1, line 20 | `dev-exclusion` | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")             #…` -> `ZIP_PREFIXES = ("train_set", "eval", "test")                        #…` | with zipped raw data the original unpacked dev_set.zip too (then SPLIT_FILTER dropped it); now the dev archive is never read | none |

## Other differences from the original run (inputs, wiring, environment)

- **wiring**: In the original, the v4 / MAV-Celeb arrays of the dataset were copied from the failed first run's output (resume); here they are computed by the r2 code. The first run's notebook (md5 436c24d6...) is not available, so the equality of its split / MAV code with r2's is not verified; those arrays only fill the samS4 slot that v_cat4 drops.
- **dev-exclusion** (environment, no code change): stage.json env pins `FLAG_SPLITS=train`. The original code lets FLAG_SPLITS override the patched SPLIT_FILTER in a smoke / local run; the runner applies the stage env after the caller's environment, so no outside value can turn dev extraction back on. Same value as the patched default: effect on numbers none.
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude). Today runner/ffcore.py build_vin applies only the configs/data_map.json exclude, so the archive is still visible; the code never unpacks or reads it (ZIP_PREFIXES / train-only index). With raw-v4 mounted unpacked (a hand-attached Kaggle dataset), find_splits still reads the dev list files (pd.read_csv of the list and an existence check of its media) before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded. Mount raw-v4 as the zipped organiser release (FF_DATA/00_raw/btc_mavceleb_v4).
- **dev-exclusion** (consequence of SPLIT_FILTER): the encoder self-checks (`self_check`: batch-vs-single and fp16-vs-fp32 cosine > 0.995 on the 3 longest clips of the first non-train split, else of train) now run on train clips; the original took a dev list. Effect on numbers: none (check only). Check behaviour on train clips: untested. A failing check puts the encoder in FAILED and skips it without stopping the run; the post_check on feats_manifest.json `files` below turns that into a stage failure.
- **environment** (unverified assumption): The pushed notebook almost certainly carried a `kapi --env` cell that kaggle/kapi.py does not log (cmd_push inserts smoke_cell(envs, '# run parameters (kapi --env) ...') but the kapi_runs.jsonl line holds only time, account, kernel, source, notebook_md5, gpu, smoke, secrets, datasets, kernels, ok). Evidence: the code reads TAP_UPLOAD (stage.ipy line 891) only from the environment and the output dataset dnq1209/flag2027-feats-tap01-voice-p exists (the dataset of the package-E copy; whether it was created by the upload branch or by hand is not recorded). Other environment knobs the code reads (TAP_AUX_MAX and the other os.environ.get knobs of the notebook) may have been set in the same cell; this stage assumes their defaults. Unverified: re-applying the kapi transform with or without such a cell does not give the pushed md5 e192b1b76ee4 locally.

## Notes

- samS4 does not enter any score: it only decides which auxiliary rows the CA head trains on (rows with NaN are dropped).

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats_tap01_voice/feats_manifest.json` `files` = 9 entries: every v4-train and MAV-Celeb v1 / v2 array of the original run with its shape (9 arrays; values from package E dnq1209__flag2027-feats-tap01-voice-p.zip :: feats_manifest.json (created 2026-10-29 17:43), dev entries left out). A failed encoder (recorded in 'failed' and skipped) or a skipped MAV-Celeb source (ExtRows error only logged) leaves an array out, so the stage fails here instead of hours later downstream.
- post_check (fatal): `feats_tap01_voice/feats_manifest.json` `external` = {"v1_complete": [7580, 19556], "v2_complete": [8953, 20715]}: MAV-Celeb v1 / v2 [face rows, voice rows] of the original manifest.
- post_check (fatal): `feats_tap01_voice/feats_manifest.json` `splits/train` = 6485: organiser train rows.
- verify strict (fatal when missing): 9 auxiliary-source files (list in stage.json).
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).

## Comment edit (20/11/2026)

Patches 2..4 of `stage.json` (ids `comment_team_label_*`) change comment text only: another team's label is replaced by a neutral wording (public naming rule, `docs/NAMING.md`). No code, path, parameter or number changes; the original notebook in `original/` is unchanged.
- comment now reads `A teammate's VoxBlink pipeline (CB` (1x; old text in the `stage.json` patch)
- comment now reads `encoder family of a teammate's VoxBlink system` (1x; old text in the `stage.json` patch)
- comment now reads `(commit 9fecd6c, as in the teammate pipeline)` (1x; old text in the `stage.json` patch)
- `stage.ipy` sha256 0b02be12f5ded69f... -> 46635c55ebef0a1a...
