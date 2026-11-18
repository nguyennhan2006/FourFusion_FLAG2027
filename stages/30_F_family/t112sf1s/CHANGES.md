# CHANGES: t112sf1s (exact-package; executed notebook from package E (E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/) `notebooks/1_train/`)

## 1. Repository copy of the notebook: embedded DEV score zip removed (type `dev-exclusion`; numbers: none)

- **Where:** `original/07_flag-2027-nb-112s-f1s.redacted.ipynb` (and therefore `stage.ipy`), code cell 3, the string `CODE_B64 = "..."`.
- **Old → new:** the base64 zip of the run's code had 39 members; the repository copy has 38: the member
  `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (91 043 bytes, sha256 `b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d`)
  is removed. Every other member is copied as raw zip entries (same bytes, same CRC, same ZipInfo); the rest of the notebook file is
  byte-identical (`_common/redact_embedded_dev_zip.py`, deterministic: re-running it on the exact notebook gives sha256 `8111272408dfbe8ea9b608e258fcbd35f0e5afab932bfec38436f2c3104f2f12`).
- **Why:** that member holds per-pair scores on the organisers' DEV lists; the published repository must not contain dev data (user rule 6,
  DESIGN 2.8 / 9.4). The byte-exact notebook stays at `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/07_flag-2027-nb-112s-f1s.ipynb` (md5 `2544147f1422427211a70e63425076f7`, sha256 `d03ab9af1ff42b742e315858d7093848fe268f39de83c7325e76b541b6adee27`) and belongs in the private FF_DATA store.
- **Effect:** none on any array, model, table or score. The only reader of the file is `zip_ct()` of the embedded
  `Experiment/CLEANTUNE-01/run.py` (line 561: `FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None`),
  and `FACELORA-03/run.py cmp_zip()` returns `None` for a missing file (line 406). In this run the line `correlation per file with the CB 20.18 zip: {...}` that `zip_ct()` prints (it is in the log of the run) becomes `... None`, and so does the same line in `e2e_out/cleantune01_zip_<arm>.txt`; the dev zip, the fuser JSON and the model file are unchanged.
  The notebook's `CODE_MD5` dict still lists the member; its check loop iterates over the members present in the zip, so it still passes.

## 2. Export narrowed to the model file (type `dev-exclusion`; numbers: none)

- The original dataset `ngnnguynchn/flag2027-evalu-f1s-p` (`sat_push.sh evalu-heads`, kaggle/sat_push.sh:1409-1418) held `ct01r2_final_models.pt` AND the dev zip
  `submission_cleantune01_F1_CA_dev.zip` (Kaggle unpacked it into a folder). The export keeps `ct01r2_final_models.pt` only (as the release
  stage `t112sf1s`). The dev zip is read by the Evaluation notebooks only in R11 mode (`SPLIT == "dev"`); the 13.54 chain runs them with
  E2E_SPLIT=eval. No later stage reads the kernel artifact `notebooks/locnguyentruongan/flag-2027-nb-112s-f1s`; the Evaluation path (e112f1sx) reads the export.

## 3. Code cells

Otherwise verbatim: building stage.ipy from the EXACT notebook gives the sha256 of release/flag2027_e2e/code/t112sf1s.ipy (checked 2026-11-18);
until 2026-11-18 the repository stage.ipy differed from it only inside the `CODE_B64` string; it now also carries the train-only patches and the inserted cell of section 6 (the release code is the executed behaviour, = FF_TRAIN_ONLY=0).

## 4. Evidence kept with this stage

Bengali-voice leak fix of 3/11/2026 12:15 (EXPERIMENT_LOG entry 346): the embedded Experiment/VOICELORA-01/run.py has md5 6d0dfe2bdf72d2204b32dd1285129207 (= the fixed file), whose line 166 reads `drop = set(json.loads((VB / "lid_summary.json").read_text())["speakers_dropped_ids"]) | T.strict_drop("vb")   # 3/11: LIDSTRICT-01, as vb_rows4`; the run was pushed after the fix; log of this run: `A0 PASSED`; `A0 audit (ct01r2): VoxCeleb2 speakers 5387 (FLAG_37 + FLAG_53), VoxBlink 13471; Bengali-flagged among them 0`; `VOICELORA-01 VoxBlink rows: 59826 rows, 13471 speakers`; `LID_STRICT MAV voice files dropped from training rows: 58 of 58`.

## 5. Runner

Runner-level only (not a change of this stage's code): the runner replaces the strings /kaggle/input and /kaggle/working by the stage's own input view and working folder, and prints a `[cell] <stage> k/n` line before each cell (DESIGN 2.4).

Runner-level requirement (2026-11-18 fix; not a change of this stage's code): `stage.json` declares `runner_requirements.literal_kaggle_input: true`. The modules decoded from `CODE_B64` at run time (listed in `runner_requirements.why`, with file and line) are written to /kaggle/working/lab and find their inputs by scanning the LITERAL path /kaggle/input, which the string rewrite above cannot reach. With the flag, the runner points /kaggle/input at this stage's own input view on Linux (symlink for the duration of the stage; `FF_LITERAL_KAGGLE=mounted` when a container already mounts it). On Kaggle the runner cannot change /kaggle/input and only warns: run this stage in a session with exactly its inputs attached (as the original run). Without the flag a Linux / Docker run would fail the asserts (lid_strict.json, adapter arrays, feats_vox2b, CT blocks), and a Kaggle orchestrator session would let the modules pick the session's own attachments (e.g. a saved original instead of an artifact rebuilt in the session).

## 6. Train-only route, default (type `dev-exclusion`; numbers: none) - 2026-11-18 fix

- **Why:** DESIGN 2.8, and mode (a) consistency: WP-D's f47b_screen2 and the t153b / t153c adapters are now rebuilt train-only, so the dev-list arrays that the executed run scored (`final` → FACELORA-03 / -05 `patch_test`, CLEANTUNE-01 `patch_test_cat4`) no longer exist in a rebuild from raw data.
- **Inserted cell** (`insert_cells`, at index 4, before cell 05; opens with `print("[cell] ...")`): `TRAIN_ONLY = os.environ.get("FF_TRAIN_ONLY", "1") != "0"` and it writes `lab/ff_train_only_entry.py` = `lab_entry.py` with `e2e_lib.components()` returning `{}` (the embedded e2e_lib is imported unchanged and only that attribute is replaced in the subprocess). Embedded files and their md5 checks are untouched.
- **Patches** (stage.json `patches` 0-3): cell 05 `if os.environ.get("CT_F1") == "1":` → `... and not TRAIN_ONLY:` (the F0 dev zip is read only by the dev zip step); cell 06 the command list `("audit", "final", "zip")` → `("audit", "final")` in train-only mode, `lab_entry.py` → `ff_train_only_entry.py` in train-only mode, and the final submission-zip block accepts 0 zips in train-only mode (`if ZP:`).
- **Effect:** none on `ct01r2_final_models.pt`: T3RECIPE-01 `final()` (embedded run.py lines 196-214) trains on the train / auxiliary rows and saves the models with torch.save before it calls `E.components()` (lines 215-224); with `{}` the loop scores nothing and `ct01r2_final_dev.npz` holds only `info`. No dev zip, no fuser JSON of the dev zip; the kernel output has no `submission.zip`. The A0 audit is unchanged. `FF_TRAIN_ONLY=0` gives the executed behaviour (needs the dev-list arrays of un-patched t153b / t153c / f47b_screen2, or the frozen originals, and dev_set.zip).
- **Still present, not opened:** the e2e bundle's `components_dev.npz` (cell 02 checks it exists; `e2e_manifest.json` is read for its split tag).

The input `notebooks/locnguyentruongan/flag-2027-nb-112s-f0s` is read only with FF_TRAIN_ONLY=0 (cell 05, F0 dev zip); it stays in `inputs` so the DAG order of the executed chain is unchanged.

Licence (2026-11-18): the head trains with 13 471 VoxBlink speakers; its model file is VoxBlink-derived (CC BY-NC-SA 4.0): private FF_DATA only, never in git (`stage.json` `licence`).
