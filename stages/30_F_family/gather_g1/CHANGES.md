# CHANGES: gather_g1 (new stage = the local gather step `sat_push.sh 111sg`)

- **Original step:** `gather_tables ct01g1 <dataset> <3 runs>` (kaggle/sat_push.sh:52-66) downloaded `ct01g1_fold[0-9].parquet` of the three runs
  (`kapi.py output --pattern`), required exactly 5 files, zipped them as `tables/<name>` and created the private dataset `ngnnguynchn/flag2027-ct02s-g1-tables-p` (`kapi.py dscreate`).
- **Stage:** copies the five files (byte for byte, `shutil.copy2`) from the three kernel artifacts to `ds/tables/`; the export makes `ds/` the artifact
  `datasets/ngnnguynchn/flag2027-ct02s-g1-tables-p` (same layout as the original dataset). Type `wiring` (download / upload replaced by artifacts); numbers: none.
- Excerpt of the original script: `original/sat_push.sh.excerpt` (with the sha256 of the whole file).
- **Smoke (2026-11-18, type `wiring`):** under FF_SMOKE=1 the stage accepts fewer than five tables and prints the missing folds (the t154_g1 smoke runs make one fold each); a full run still requires exactly five. t112sf0s / t112sf1s fill missing folds with their smoke stand-ins.
