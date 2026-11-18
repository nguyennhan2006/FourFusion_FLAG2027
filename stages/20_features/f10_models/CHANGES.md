# CHANGES: stage `f10_models`

Original code: git show a89a071:kaggle/FLAG_10_models.ipynb (builder-regenerated). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 4, line 1 | `url, pin` | `import flag_extract as X, flag_models as M ⏎ OUT = WORKDIR / "feats_m…` -> `# --- FF reproduction patch (stage f10_models, CHANGES.md #1): edit t…` | rewrite flag_models.py / agegender.py on disk before import: ImageBind URL (dead link, same file) + revisions of ViT-age (4b5acc34), AdaFace (54f602a0), audeering (bad2737f); Hugging Face revision pinned (the original resolved the then-current `main`); the pinned revision is the one in the team's HF cache and in the RF core / RFX manifest (S8 E10, DESIGN §9.2) | URL: none (same file); pins: none if the repository did not change since the run (not verified: CHƯA CHẮC) |
| 2 | cell 4, line 9 | `dev-exclusion` | `S = M.v4_index(SCAN, WORKDIR / "raw") ⏎ ` -> `# --- FF dev-exclusion (CHANGES.md #2): index ONLY the organiser trai…` | the original indexes and (zipped data) unpacks the 4 dev lists; now only train_English.txt is indexed, same rows / order | none on any train / MAV-Celeb array; the dev arrays are no longer written |
| 3 | cell 5, line 3 | `dev-exclusion` | `voice_probe = [X.load_audio(S["no_gender/Bangla"]["root"] / p) for p …` -> `voice_probe = [X.load_audio(S["train"]["root"] / p) for p in S["train…` | the fp16 self-check of the voice encoders read 4 dev Bangla clips; it now reads 4 train clips | none (the check only asserts cos > 0.995 between fp16 and fp32) |
| 4 | cell 3, line 6 | `pin` | `    !pip -q install --no-deps "insightface==2.0" "git+https://github.…` -> `    # FF environment patch (CHANGES.md #4): one pip call per package,…` | the shared pip call fails on the current Kaggle image (openai CLIP build); split into one call per package + CLIP fallback clone (the team's own fix, BEST/v1977/README.md §6) | none expected (same packages); CLIP fallback pinned at d50d76d, the commit used by the team's other notebooks; the original resolved an unpinned openai/CLIP HEAD on 27-28/09/2026 (not verified to be d50d76d; todo) |

## Other differences from the original run (inputs, wiring, environment)

- **wiring**: feats_models came from two runs of two notebook versions (EXPERIMENT_LOG 58 / 59): run 1 (pre-fix) wrote vitage / farl / ibv / iba, run 2 (fixed) resumed and wrote siglip2 / adaface / w2v2ag. This stage is one run of the fixed version (commit a89a071), which writes all seven encoders; the code paths of vitage / farl / ibv / iba are the same in both versions as far as the log says (the fix concerned transformers 5.x in siglip2 / adaface / w2v2ag).
- **wiring**: The optional resume input (an earlier output of this notebook, dataset flag2027-feats-models) is not attached.
- **wiring**: Inputs as the original: raw v4 data, MAV-Celeb v1 / v2, flag2027-ext-meta (row order of the MAV-Celeb arrays).
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude). Today runner/ffcore.py build_vin applies only the configs/data_map.json exclude, so the archive is still visible; the code never unpacks or reads it (ZIP_PREFIXES / train-only index). With raw-v4 mounted unpacked (a hand-attached Kaggle dataset), find_splits still reads the dev list files (pd.read_csv of the list and an existence check of its media) before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded. Mount raw-v4 as the zipped organiser release (FF_DATA/00_raw/btc_mavceleb_v4).

## Notes

- FaRL here is epoch 64 (flag_models.FaRL); it only feeds the `rel` streams removed by NOREL and is NOT the FaRL of the 13.54 system (epoch 16, stage f40_t3enc_mav / f41_t3enc_vox2). It must still be produced: BEST/v7_safe/safe_lib.py load_ext asserts the eda-bundle files exist.
- The notebook text calls ArcFace 'R100'; insightface buffalo_l is w600k_r50 (ResNet-50, WebFace600K).

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats_models/feats_manifest.json` `files` = 30 entries: every v4-train and MAV-Celeb v1 / v2 array of the original run with its shape (30 arrays; values from kaggle/output/feats_models/feats_manifest.json (created 2026-09-28 01:18), dev entries left out). A failed encoder (recorded in 'failed' and skipped) or a skipped MAV-Celeb source (ExtRows error only logged) leaves an array out, so the stage fails here instead of hours later downstream.
- post_check (fatal): `feats_models/feats_manifest.json` `external` = ["v1_complete", "v2_complete"]: both MAV-Celeb sources were read (original manifest: ['v1_complete', 'v2_complete']).
- verify (informational): `feats_models/face_adaface_train.npy` sha256 vs the saved copy `c806596c394b1c94...`.
- verify (informational): `feats_models/ext_v1_complete_face_adaface.npy` sha256 vs the saved copy `d160e51b74941f56...`.
- verify (informational): `feats_models/ext_v2_complete_face_adaface.npy` sha256 vs the saved copy `0844ca8185750016...`.
- verify (informational): `feats_models/face_ibv_train.npy` sha256 vs the saved copy `1d4103653769481d...`.
- verify (informational): `feats_models/voice_iba_train.npy` sha256 vs the saved copy `1a369c2dc6488429...`.
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).
