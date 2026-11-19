# CHANGES: t112sf0s (exact-package; executed notebook from package E (E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/) `notebooks/1_train/`)

## 1. Repository copy of the notebook: embedded DEV score zip removed (type `dev-exclusion`; numbers: none)

- **Where:** `original/06_flag-2027-nb-112s-f0s.redacted.ipynb` (and therefore `stage.ipy`), code cell 3, the string `CODE_B64 = "..."`.
- **Old → new:** the base64 zip of the run's code had 39 members; the repository copy has 38: the member
  `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (91 043 bytes, sha256 `b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d`)
  is removed. Every other member is copied as raw zip entries (same bytes, same CRC, same ZipInfo); the rest of the notebook file is
  byte-identical (`_common/redact_embedded_dev_zip.py`, deterministic: re-running it on the exact notebook gives sha256 `b3bdbca555617b0e3500c219eecf5c445d29457557b42128dd81403ac0ae61f6`).
- **Why:** that member holds per-pair scores on the organisers' DEV lists; the published repository must not contain dev data (user rule 6,
  DESIGN 2.8 / 9.4). The byte-exact notebook stays at `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/06_flag-2027-nb-112s-f0s.ipynb` (md5 `db61a33b7cb7ba8aa9d738da7fcbdc35`, sha256 `7a08e671e70b5b8033ea32173c9c80b2c6853afe1a4aaa9f2ba27c979090baa4`) and belongs in the private FF_DATA store.
- **Effect:** none on any array, model, table or score. The only reader of the file is `zip_ct()` of the embedded
  `Experiment/CLEANTUNE-01/run.py` (line 561: `FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None`),
  and `FACELORA-03/run.py cmp_zip()` returns `None` for a missing file (line 406). In this run the line `correlation per file with the CB 20.18 zip: {...}` that `zip_ct()` prints (it is in the log of the run) becomes `... None`, and so does the same line in `e2e_out/cleantune01_zip_<arm>.txt`; the dev zip, the fuser JSON and the model file are unchanged.
  The notebook's `CODE_MD5` dict still lists the member; its check loop iterates over the members present in the zip, so it still passes.

## 2. Export narrowed to the model file (type `dev-exclusion`; numbers: none)

- The original dataset `ngnnguynchn/flag2027-evalu-f0s-p` (`sat_push.sh evalu-heads`, kaggle/sat_push.sh:1409-1418) held `ct01g1_final_models.pt` AND the dev zip
  `submission_cleantune01_F0_dev.zip` (Kaggle unpacked it into a folder). The export keeps `ct01g1_final_models.pt` only (as the release
  stage `t112sf0s`). The dev zip is read by the Evaluation notebooks only in R11 mode (`SPLIT == "dev"`); the 13.54 chain runs them with
  E2E_SPLIT=eval. The kernel artifact `notebooks/locnguyentruongan/flag-2027-nb-112s-f0s` still holds the whole /kaggle/working of the run: t112sf1s reads `submission_cleantune01_F0_dev.zip` (English bytes of its dev zip) from it.

## 3. Code cells

Otherwise verbatim: building stage.ipy from the EXACT notebook gives the sha256 of release/flag2027_e2e/code/t112sf0s.ipy (checked 2026-11-18);
until 2026-11-18 the repository stage.ipy differed from it only inside the `CODE_B64` string; it now also carries the train-only patches and the inserted cell of section 6 (the release code is the executed behaviour, = FF_TRAIN_ONLY=0).

## 4. Evidence kept with this stage

Bengali-voice leak fix of 3/11/2026 12:15 (EXPERIMENT_LOG entry 346): the embedded Experiment/VOICELORA-01/run.py has md5 6d0dfe2bdf72d2204b32dd1285129207 (= the fixed file), whose line 166 reads `drop = set(json.loads((VB / "lid_summary.json").read_text())["speakers_dropped_ids"]) | T.strict_drop("vb")   # 3/11: LIDSTRICT-01, as vb_rows4`; the run was pushed after the fix; log of this run: `A0 PASSED`; `A0 audit (ct01g1): VoxCeleb2 speakers 5504 (FLAG_37 + FLAG_53), VoxBlink 13471; Bengali-flagged among them 0`.

## 5. Runner

Runner-level only (not a change of this stage's code): the runner replaces the strings /kaggle/input and /kaggle/working by the stage's own input view and working folder, and prints a `[cell] <stage> k/n` line before each cell (DESIGN 2.4).

Runner-level requirement (2026-11-18 fix; not a change of this stage's code): `stage.json` declares `runner_requirements.literal_kaggle_input: true`. The modules decoded from `CODE_B64` at run time (listed in `runner_requirements.why`, with file and line) are written to /kaggle/working/lab and find their inputs by scanning the LITERAL path /kaggle/input, which the string rewrite above cannot reach. With the flag, the runner points /kaggle/input at this stage's own input view on Linux (symlink for the duration of the stage; `FF_LITERAL_KAGGLE=mounted` when a container already mounts it). On Kaggle the runner cannot change /kaggle/input and only warns: run this stage in a session with exactly its inputs attached (as the original run). Without the flag a Linux / Docker run would fail the asserts (lid_strict.json, adapter arrays, feats_vox2b, CT blocks), and a Kaggle orchestrator session would let the modules pick the session's own attachments (e.g. a saved original instead of an artifact rebuilt in the session).

## 6. Train-only route, default (type `dev-exclusion`; numbers: none) - 2026-11-18 fix

- **Why:** DESIGN 2.8, and mode (a) consistency: WP-D's f47b_screen2 and the t153b / t153c adapters are now rebuilt train-only, so the dev-list arrays that the executed run scored (`final` → FACELORA-03 / -05 `patch_test`, CLEANTUNE-01 `patch_test_cat4`) no longer exist in a rebuild from raw data.
- **Inserted cell** (`insert_cells`, at index 4, before cell 05; opens with `print("[cell] ...")`): `TRAIN_ONLY = os.environ.get("FF_TRAIN_ONLY", "1") != "0"` and it writes `lab/ff_train_only_entry.py` = `lab_entry.py` with `e2e_lib.components()` returning `{}` (the embedded e2e_lib is imported unchanged and only that attribute is replaced in the subprocess). Embedded files and their md5 checks are untouched.
- **Patches** (stage.json `patches` 0-3): cell 05 `if os.environ.get("CT_F1") == "1":` → `... and not TRAIN_ONLY:` (the F0 dev zip is read only by the dev zip step); cell 06 the command list `("audit", "final", "zip")` → `("audit", "final")` in train-only mode, `lab_entry.py` → `ff_train_only_entry.py` in train-only mode, and the final submission-zip block accepts 0 zips in train-only mode (`if ZP:`).
- **Effect:** none on `ct01g1_final_models.pt`: T3RECIPE-01 `final()` (embedded run.py lines 196-214) trains on the train / auxiliary rows and saves the models with torch.save before it calls `E.components()` (lines 215-224); with `{}` the loop scores nothing and `ct01g1_final_dev.npz` holds only `info`. No dev zip, no fuser JSON of the dev zip; the kernel output has no `submission.zip`. The A0 audit is unchanged. `FF_TRAIN_ONLY=0` gives the executed behaviour (needs the dev-list arrays of un-patched t153b / t153c / f47b_screen2, or the frozen originals, and dev_set.zip).
- **Still present, not opened:** the e2e bundle's `components_dev.npz` (cell 02 checks it exists; `e2e_manifest.json` is read for its split tag).

## 7. skip_with (type `wiring`; numbers: none)

`skip_with` = `04_run_outputs/flag-2027-nb-112s-f0s` only (was also `03_checkpoints/_wiring_b0s_head` first). The runner takes the first existing folder as the saved copy of the KERNEL output (runner/ffcore.py 541-563, 582-586); the kernel output holds `e2e_out/submission_cleantune01_F0_dev.zip`, which t112sf1s needs with FF_TRAIN_ONLY=0, while the evalu-f0s-p folder holds only the model file. `03_checkpoints/_wiring_b0s_head` stays the saved copy of the export (configs/data_map.json).

Licence (2026-11-18): the head trains with 13 471 VoxBlink speakers; its model file is VoxBlink-derived (CC BY-NC-SA 4.0): private FF_DATA only, never in git (`stage.json` `licence`).

## Requirement 1: frozen wiring inputs removed (19/11/2026 (requirement-1 closure))

- input `datasets/ngnnguynchn/flag2027-cc5-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (t3recipe02_fold*.parquet) are read only by DSEL-02/run.py with_cc5() and DSEL-03/run.py attach(), which are reached only from the DSEL-02 / DSEL-03 / SCALE-CC-01 / AUXP-01 report commands; this stage runs CLEANTUNE-01 / FACTOR-01 / FUSENET-01 commands, which never call them, and no cell checks the dataset's presence. No producing stage exists (old run FLAG_57).
- input `datasets/ngnnguynchn/flag2027-dsel02-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (dsel02_fold*.parquet) are read only by DSEL-02/run.py and DSEL-03/run.py report paths, which this stage's commands never call; no presence check. No producing stage exists (old run FLAG_58).
- input `datasets/ngnnguynchn/flag2027-facelora03-tables-p` dropped (stage.json `inputs_dropped`): attached in the original run; real runs only copy facelora03_B3_fold*.parquet into SP/out/safe04 (no assert) and nothing reads the copies: the only reader is the smoke stand-in (FL3Z_SMOKE / FLAG_SMOKE), which now takes the key columns of the SAFE-04 streams tables instead (patch of the same date). No producing stage exists (old runs FLAG_86 / 86b).
- input `datasets/ngnnguynchn/flag2027-lab-avfull-p` dropped (stage.json `inputs_dropped`): attached in the original run only to satisfy the lab presence check OPTIONAL_DIRS['Experiment/CAFUSE-EN/feats_av_full'] = ['CLEANTUNE-01']; its content (AudioCLIP arrays) is read only by EN-T3-01 table_en() via CLEANTUNE-01 fit_en_clean(), called from the zip command when F0_ZIP is unset, i.e. only in e112f0sx (F0 English). The presence check is made optional by the patch of the same date. No producing stage exists (team package labavfull/, FLAG_32 run 2).
- patch (smoke): `pd.read_parquet(SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet")` -> `pd.read_parquet(SP / "out/safe04" / f"streams_fold{s}.parquet")[["fold", "data", "protocol", "y", "fi", "vi", "XALL_deep0"]].rename(columns={"XALL_deep0": "trB3"})`. Why: smoke only (FL3Z_SMOKE / FLAG_SMOKE): the stand-in tables were noisy copies of FACELORA-03's B3 tables (datasets/ngnnguynchn/flag2027-facelora03-tables-p, no producing stage). They now start from the SAFE-04 streams table of the same fold (frozen bundle / stage safe04_lab, already in SP/out/safe04): same 7 406 rows and the same key columns fold, data, protocol, y, fi, vi (checked on fold 1 against the team copies, 19/11); XALL_deep0 plays the role of trB3, whose columns the stand-in drops anyway. Effect on numbers: none in real runs (smoke stand-ins only)
- patch (wiring): `"Experiment/CAFUSE-EN/feats_av_full": ["CLEANTUNE-01"]}` -> `"Experiment/CAFUSE-EN/feats_av_full": []}`. Why: the lab link loop asserts that every OPTIONAL_DIRS folder listed for the running experiment is attached; feats_av_full (AudioCLIP arrays of datasets/ngnnguynchn/flag2027-lab-avfull-p, no producing stage) is read only by the F0 English zip path (e112f0sx), which this stage never runs, so the folder becomes optional here. Effect on numbers: none (the folder is never opened by this stage's commands)
- evidence: `repo_map/build/REQ1_GAPS.md` (team archive); `python runner/check_stage.py` rebuilds this stage

## SAFE-04 bundle without dev wiring accepted (type `wiring`; numbers: none) - added 19/11/2026

- **Where:** the e2e bundle search (`e2e = [... components_{SPLIT}.npz ...]`).
- **Old:** a bundle qualified only with `components_dev.npz` (dev-list arrays of the frozen dev wiring, no producing stage).
- **New:** a bundle with a `safe04/` folder qualifies too. Stage `safe04_lab` writes such a bundle (manifest without dev lists) when the dev wiring is not attached.
- **Why:** lineage of the 13.54 zip from raw data. With `FF_TRAIN_ONLY=1` (default) this stage reads only `safe04/` and the manifest's `created` field; the dev scoring that reads `components_dev.npz` runs only with `FF_TRAIN_ONLY=0`. When `components_dev.npz` exists the same bundle is chosen as before.
