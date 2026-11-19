# CHANGES: stage `f12_eval_feats`

Original code: git show a89a071:kaggle/FLAG_12_features.ipynb (builder-regenerated). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 1, line 18 | `dev-exclusion` | `SPLIT_FILTER = None            # e.g. ["no_gender/English"]; None = e…` -> `SPLIT_FILTER = ["train"]      # FF dev-exclusion: organiser train lis…` | no dev list is indexed or encoded (DESIGN §2.8; as TRAIN_ONLY of release/flag2027_e2e/stages.py:151) | none on the train / MAV-Celeb / auxiliary arrays (each split is encoded on its own); the dev arrays and dev index files are no longer written |
| 2 | cell 1, line 19 | `dev-exclusion` | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")             #…` -> `ZIP_PREFIXES = ("train_set", "eval", "test")                        #…` | with zipped raw data the original unpacked dev_set.zip too (then SPLIT_FILTER dropped it); now the dev archive is never read | none |
| 3 | cell 5, line 189 | `url` | `https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth` -> `https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade…` | Meta's link to imagebind_huge.pth is dead since 12/11/2026; same file (sha256 d6f6c22b...2157) on a pinned Hugging Face mirror, as release/flag2027_e2e/stages.py GLOBAL_PATCHES | none (same file) |
| 4 | cell 5, line 116 | `pin` | `self.proc = ViTImageProcessor.from_pretrained(self.ID)` -> `self.proc = ViTImageProcessor.from_pretrained(self.ID, revision="4b5a…` | ViT-age: Hugging Face revision pinned (the original resolved the then-current `main`); the pinned revision is the one in the team's HF cache and in the RF core / RFX manifest (S8 E10, DESIGN §9.2) | none if the repository did not change since the run |
| 5 | cell 5, line 117 | `pin` | `self.m = ViTForImageClassification.from_pretrained(self.ID)` -> `self.m = ViTForImageClassification.from_pretrained(self.ID, revision=…` | ViT-age: Hugging Face revision pinned (the original resolved the then-current `main`); the pinned revision is the one in the team's HF cache and in the RF core / RFX manifest (S8 E10, DESIGN §9.2) | none if the repository did not change since the run |
| 6 | cell 5, line 163 | `pin` | `self.proc = Wav2Vec2Processor.from_pretrained(self.ID)` -> `self.proc = Wav2Vec2Processor.from_pretrained(self.ID, revision="bad2…` | audeering: Hugging Face revision pinned (the original resolved the then-current `main`); the pinned revision is the one in the team's HF cache and in the RF core / RFX manifest (S8 E10, DESIGN §9.2) | none if the repository did not change since the run |
| 7 | cell 5, line 164 | `pin` | `self.m = AgeGenderModel.from_pretrained(self.ID)` -> `self.m = AgeGenderModel.from_pretrained(self.ID, revision="bad2737f52…` | audeering: Hugging Face revision pinned (the original resolved the then-current `main`); the pinned revision is the one in the team's HF cache and in the RF core / RFX manifest (S8 E10, DESIGN §9.2) | none if the repository did not change since the run |
| 8 | cell 5, line 68 | `pin` | `        kw = {} ⏎         try: ⏎             from speechbrain.utils.f…` -> `        kw = {"revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"}…` | speechbrain/spkrec-ecapa-voxceleb: Hugging Face revision pinned (the original resolved the then-current `main`); the pinned revision is the one in the team's HF cache and in the RF core / RFX manifest (S8 E10, DESIGN §9.2) (from_hparams(revision=...)) | none if the repository did not change since the run |

## Other differences from the original run (inputs, wiring, environment)

- **wiring**: The optional reference inputs (flag2027-feats-v2, flag2027-feats-models) of the check cell are not attached: check_vs_reference.csv stays empty. The optional resume input is not attached.
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude). Today runner/ffcore.py build_vin applies only the configs/data_map.json exclude, so the archive is still visible; the code never unpacks or reads it (ZIP_PREFIXES / train-only index). With raw-v4 mounted unpacked (a hand-attached Kaggle dataset), find_splits still reads the dev list files (pd.read_csv of the list and an existence check of its media) before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded. Mount raw-v4 as the zipped organiser release (FF_DATA/00_raw/btc_mavceleb_v4).
- **dev-exclusion** (consequence of SPLIT_FILTER): the encoder self-checks (`self_check`: batch-vs-single and fp16-vs-fp32 cosine > 0.995 on the 3 longest clips of the first non-train split, else of train) now run on train clips; the original took a dev list. Effect on numbers: none (check only). Check behaviour on train clips: untested. A failing check puts the encoder in FAILED and skips it without stopping the run; the post_check on feats_manifest.json `files` below turns that into a stage failure.

## Notes

- Today's builder (kaggle/build_eval_notebooks.py, modified after a89a071) adds broken-file handling, a [boot] line and the ImageBind URL (+58/-10 lines). The a89a071 version is used because it is the closest to the run (the run's manifests list no broken file, so the extra handling would not change the arrays).
- The 'FLAG_12 loop' (cells DATA12 / AUDIO12 / ENC12) is the shared data loop of FLAG_40 / 47b / 103c / 160a / 162a / 50cc: v4 and test lists use the whole clip, MAV-Celeb is cut to the centre 12 s (137 MAV v1 files at 44.1 kHz are resampled).
- Two different ECAPA models: speechbrain/spkrec-ecapa-voxceleb (this stage, arrays ecapa192 / ecapa6144, DP only) vs yangwang825/ecapa-tdnn-vox2 (the organisers' ECAPA space, stage r08_mav_rows).

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats_eval/feats_manifest.json` `files` = 19 entries: every v4-train and MAV-Celeb v1 / v2 array of the original run with its shape (19 arrays; values from kaggle/output/feats_eval/feats_manifest.json (created 2026-09-29 04:37), dev entries left out). A failed encoder (recorded in 'failed' and skipped) or a skipped MAV-Celeb source (ExtRows error only logged) leaves an array out, so the stage fails here instead of hours later downstream.
- post_check (fatal): `feats_eval/feats_manifest.json` `external` = {"v1_complete": [7580, 19556], "v2_complete": [8953, 20715]}: MAV-Celeb v1 / v2 [face rows, voice rows] of the original manifest.
- post_check (fatal): `feats_eval/feats_manifest.json` `splits/train` = 6485: organiser train rows.
- verify (informational): `feats_eval/face_arcface_train.npy` sha256 vs the saved copy `55491b54180e116c...`.
- verify (informational): `feats_eval/voice_ecapa192_train.npy` sha256 vs the saved copy `b3bd0a6430412e28...`.
- verify (informational): `feats_eval/voice_ecapa6144_train.npy` sha256 vs the saved copy `195490e23b3c7bbf...`.
- verify (informational): `feats_eval/ext_v1_complete_face_arcface.npy` sha256 vs the saved copy `e80283e1ee273581...`.
- verify (informational): `feats_eval/ext_v2_complete_face_arcface.npy` sha256 vs the saved copy `c1e99f58e41aef4b...`.
- verify (informational): `feats_eval/ext_v1_complete_voice_ecapa192.npy` sha256 vs the saved copy `e56824753c4fc9e1...`.
- verify (informational): `feats_eval/ext_v2_complete_voice_ecapa192.npy` sha256 vs the saved copy `611ec9c5f71eaef6...`.
- verify (informational): `feats_eval/ext_v1_complete_voice_ecapa6144.npy` sha256 vs the saved copy `7735d1111a1e57c1...`.
- verify (informational): `feats_eval/ext_v2_complete_voice_ecapa6144.npy` sha256 vs the saved copy `b906a9328c8a9ffd...`.
- verify (informational): `feats_eval/face_ibv_train.npy` sha256 vs the saved copy `1d4103653769481d...`.
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).
