# CHANGES: cleancomp_tables (local-script `Experiment/CLEANCOMP-01/build_tables.py` turned into a stage)

## 1. Input read from the dataset folder instead of the local zip (type `path`; numbers: none)

- build_tables.py iterates over `kaggle/flag2027-e2e-dev.zip`; the stage iterates over the attached `e2edev/` folder (as the release stage
  `cleancomp_tables`, release/flag2027_e2e/stages.py:386-415). Same NEW columns, same alignment asserts, same CLEANCOMP.txt text. On Kaggle
  the three inner `ref/submission_v*.zip` are unpacked folders: they are copied as found (the original dataset flag2027-cleancomp-e2e-p shows the
  same unpacked layout). Parquet tables are re-serialised by `to_parquet` exactly as in build_tables.py.

## 2. build_dev.py not run (type `dev-exclusion`; numbers: none for 13.54)

- `kaggle/cleancomp_chain.sh` also ran `Experiment/CLEANCOMP-01/build_dev.py` (local arrays kaggle/output/feats_ext_full, kaggle_upload,
  BEST/v6_eval/_members): it rewrote `components_dev.npz` (the four DEV lists scored with the strict XALL member and v11 head), added
  `CLEANCOMP_dev.txt` and a manifest note. This stage keeps the e2e-dev versions of these three files.
- Effect: only the dev scoring of t112sf1s (its dev zip) sees the old dev components; `ct01r2_final_models.pt`, the SAFE-04 tables used by the
  F1s' fuser (e50cc forwards `safe04/` only, kind == "e2e-dev"), and every Evaluation score are unaffected. The frozen dataset (with build_dev.py's
  files) stays the default for 13.54 (`skip_with`).

## 3. Output layout and messages (type `path`; numbers: none)

- Output folder `ds/e2edev/` (the dataset = `ds/`, so the artifact has `e2edev/...` at its root like the original zip; the release stage exported
  the `e2edev` folder itself as the root). Readers locate the bundle by `e2e_manifest.json`, so both layouts work.
- English cell title; one extra `print` of the inputs and of the file count.
