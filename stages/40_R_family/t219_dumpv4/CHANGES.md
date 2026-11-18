# CHANGES: `t219_dumpv4` (phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4)

Source of `stage.ipy`: `original/flag-2027-nb-219-fusenet-dumpv4.redacted.ipynb`, a redacted copy of the exact notebook `NOP_BTC_2026-11-20/repo/_staging_backup/_PRIVATE_NOT_FOR_PUBLICATION/kapi_flag-2027-nb-219-fusenet-dumpv4_a1nyz4zf/flag-2027-nb-219-fusenet-dumpv4.ipynb` (md5 `6ad5b31c797b6323120eec0cc4dd727d`, sha256 `7a84b0ccaad7beea633e7cbd291c91067e1b44515493c151409a05622bac22d8`; exactness `exact-staging`; kapi_runs line 1887, pushed md5 `6ad5b31c797b6323120eec0cc4dd727d`).
Rebuild rule: every non-empty code cell, CRLF -> LF, the patches below in this order (each must match exactly `count` times), cells joined by `\n# <<<FF-CELL>>>\n`; result sha256 `8f0de824b34a79178a728a9e4b56d04e31c9453e86e41e221411bbc6de9cbb59` (9 cells). Same rule as `release/flag2027_e2e/build_notebook.py` (checked: it reproduces `release/flag2027_e2e/code/t163b.ipy` byte for byte when given that package's two patches).

## A. Changes made when copying the exact notebook into `original/` (redaction)

1. **notebook cell 4 (0-based, markdown counted), `CODE_B64` literal** - type `dev-exclusion`.
   - old: base64 of the code zip (366708 characters, zip sha256 `41a6e0b66b14fc53c25ddeac6b4d98ff1ccda1dca39ca710ff59eb811aa4a3f3`) with the member `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (91043 bytes, sha256 `b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d`, md5 `ae902d1c7e87783bf88e345b459e39f2`).
   - new: the same zip without that member; every other member is copied byte for byte (local records kept, central directory re-written; zip sha256 `1b3c2de9b49625706711a9981e76470ac9a071efba3d6b2340c716ef284e9f8b`, literal 288260 characters).
   - why: dev-phase score zip; the repo must not carry dev data (user rule 6, DESIGN 9.4).
   - effect on numbers: none: the cell checks the md5 of each member that is present; the only reader of that file is CLEANTUNE-01 `zip_ct()` (dev zip comparison), which this stage never calls; `CODE_MD5` / `EXP_CODE` still name its md5 and are only printed / compared for resume.

## B. Patches (`stage.json` -> `patches`, applied in this order)

2. **notebook cell 11 (0-based index in the notebook, markdown cells counted), line 13 of that cell** - type `bugfix`.
   - old (1 hit): `for n in ("fusenet_report.txt", "fusenet_summary.json"):\n    shutil.copy(SP / "out/safe04" / n, OUTD / n)\n`
   - new: `for n in ("fusenet_report.txt", "fusenet_summary.json"):\n    if os.environ.get("FUSENET_MODE") == "dumpv4" and not (SP / "out/safe04" / n).exists():   # FF bugfix (CHANGES.md): dumpv4 writes xjfdump_* and returns before the report; the original run stopped here with FileNotFoundError after its outputs were written\n        continue\n    shutil.copy(SP / "out/safe04" / n, OUTD / n)\n`
   - why: in FUSENET_MODE=dumpv4, Experiment/FUSENET-01/run.py report() writes xjfdump_v4.parquet / xjfdump_persons.json / xjfdump_report.txt and returns without fusenet_report.txt; the original run (16/11 04:00 UTC) therefore ended with FileNotFoundError at this copy, AFTER its outputs were written (log line 'FileNotFoundError ... fusenet_report.txt'), and Kaggle marked it ERROR. Skipping the two missing report files lets the stage finish; nothing that is computed changes.
   - effect on numbers: none.
   - side effect (added by the fixer 18/11): the rest of the 'RFSTUNE-01 report' cell (one print) and the whole 'summary' cell now run; in the original run they never ran because this cell died. The summary cell rewrites e2e_out/progress.json with five more keys (gpu, torch, lab_data, e2e, finished), so this stage's progress.json differs from the saved one (Experiment/XJF-01/out/dumpv4/e2e_out/progress.json, sha256 f28b4783..., keys code_md5 / done / exp / files / seconds / split; listed in original/OUTPUTS.sha256). No number changes; only the xjfdump_* files are read downstream (WP-I dp_sigma reads xjfdump_v4.parquet).

## C. Kept as in the original run (on purpose)

- Inputs: exactly the datasets / kernels the run attached (`stage.json` -> `inputs`, from the run's kernel metadata), including the ones attached but not read.
- Every other line of every cell, the embedded code (`original/decoded/`, byte-exact, CRLF kept) and the `[boot]` / `[cell]` lines that kapi inserted at push time.
- pip calls (`pip_policy: original`).

## D. Changes in `stage.json` outside the code (fixer, 18/11/2026, review of WP-F)

- `runner_requirements.literal_kaggle_input` = `true` (new key; why: the lab modules decoded from CODE_B64 scan the literal /kaggle/input: kaggle/build_l0_notebooks.py line 143 (LOAD cell: the run log reads 'LID_STRICT MAV voice files dropped from training rows: 58 of 58 listed | /kaggle/input/datasets/ngnnguynchn/flag2027-lid-strict-p/lid_strict.json'; e2e_lib.mav_flagged uses that list to choose the rows of the trRFS fits that build U, so it enters sigma), Experiment/T3ENC-01/run.py 33-34, CLEANTUNE-01/run.py 168-182 and 850, FACELORA-03/run.py 53-54, SCALE-CC-01/run.py 32-33, DSEL-02/run.py 53-54, DSEL-03/run.py 75-76, VOICELORA-01/run.py 48-49, AUXP-01/run.py 34, IBLORA-01/run.py 30 (same modules as WP-G's e112f1sx)).
- `notes` / `todo`: identity-linkage constant OVERLAP of the embedded lab code (user decision before publication, DESIGN G17), dev_data flag for e2e-dev-p (WP-A), private storage of the exact notebook.
- `notes` / `todo`: sigma_source layout (lab/ excluded: dev zip), t3s01-tables-p enters sigma (correction), OVERLAP, data_map todos for WP-A / WP-E.

## E. Input t3s01-tables-p replaced by the t3s01_trca export (stage.json only; integration fixer r1, 18/11/2026)

- type: `wiring`
- where: stage.json `inputs`: `datasets/ngnnguynchn/flag2027-t3s01-tables-p` -> `{"artifact": "datasets/ff/flag2027-t3s01-trca-p", "mount": "datasets/ngnnguynchn/flag2027-t3s01-tables-p"}`; recorded under `inputs_replaced`. No code change: the gather cell (rglob `ct01r2?*_fold[1-5].parquet`, faces farl / clip / b3s / c5s) sees the same path and file names
- old: the frozen FLAG_167 T3S-01 tables (no producing stage; the FLAG_167 notebooks are lost)
- new: the tables made by the new stage `t3s01_trca` from `ct02s-r2ca-tables-p` (gather_r2ca <- t155s <- raw data): the row keys, and `trCA` in the farl tables
- why: INTEGRATION_r1 B3 (user requirement 1). The dumpv4 path reads only the keys and `trCA` (`merge_t3s`, then `FUSENET-01 report`: `F1S = S6 + V11_T3 + ["trCA"]`); the `trT3S_*` columns of the frozen tables are not read on that path. Keys and `trCA` equal the frozen tables exactly (WP-F light check 18/11; `t3s01_trca` re-checks it when the frozen copy is under FF_DATA)
- effect on numbers: none expected (sigma uses U = F1s|oof + R|oof); not run yet (todo: compare xjfdump_v4.parquet with 9f0a5af4...)

