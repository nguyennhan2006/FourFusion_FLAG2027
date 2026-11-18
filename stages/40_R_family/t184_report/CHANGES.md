# CHANGES: `t184_report` (phantrnhuyntrn/flag-2027-nb-184-factor-report)

Source of `stage.ipy`: `original/18_flag-2027-nb-184-factor-report.redacted.ipynb`, a redacted copy of the exact notebook `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/18_flag-2027-nb-184-factor-report.ipynb` (md5 `c7b54a775ce9c57bd7b16285931339f2`, sha256 `b92faf9a9fc01682b21e78e75efabc98abaca0071fecb5f2ce650b41aad39cf6`; exactness `exact-package`; kapi_runs line 1549, pushed md5 `3bcb7ce1495f194756b9314ce32da588`).
Rebuild rule: every non-empty code cell, CRLF -> LF, the patches below in this order (each must match exactly `count` times), cells joined by `\n# <<<FF-CELL>>>\n`; result sha256 `7689f728a007dece7e20a410b87f42eda7f37188088ff37b2a62e7125cae40e0` (9 cells). Same rule as `release/flag2027_e2e/build_notebook.py` (checked: it reproduces `release/flag2027_e2e/code/t163b.ipy` byte for byte when given that package's two patches).

## A. Changes made when copying the exact notebook into `original/` (redaction)

1. **notebook cell 4 (0-based, markdown counted), `CODE_B64` literal** - type `dev-exclusion`.
   - old: base64 of the code zip (366708 characters, zip sha256 `225a250aa6b3ec356aca63b6eab471c7e0797fd6c1eba58ace15f459ce848dff`) with the member `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (91043 bytes, sha256 `b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d`, md5 `ae902d1c7e87783bf88e345b459e39f2`).
   - new: the same zip without that member; every other member is copied byte for byte (local records kept, central directory re-written; zip sha256 `604f650d74fda24c36d41092a180b088796bc4cd6d4330b8571d6d2bfe2dec17`, literal 288260 characters).
   - why: dev-phase score zip; the repo must not carry dev data (user rule 6, DESIGN 9.4).
   - effect on numbers: none: the cell checks the md5 of each member that is present; the only reader of that file is CLEANTUNE-01 `zip_ct()` (dev zip comparison), which this stage never calls; `CODE_MD5` / `EXP_CODE` still name its md5 and are only printed / compared for resume.

The second source `original/staging/flag-2027-nb-184-factor-report.redacted.ipynb` (exact push staging copy, md5 `3bcb7ce1495f194756b9314ce32da588` = kapi_runs) got the same redaction and gives the same `stage.ipy`.

## B. Patches (`stage.json` -> `patches`, applied in this order)

none

## C. Kept as in the original run (on purpose)

- Inputs: exactly the datasets / kernels the run attached (`stage.json` -> `inputs`, from the run's kernel metadata), including the ones attached but not read.
- Every other line of every cell, the embedded code (`original/decoded/`, byte-exact, CRLF kept) and the `[boot]` / `[cell]` lines that kapi inserted at push time.
- pip calls (`pip_policy: original`).

## D. Changes in `stage.json` outside the code (fixer, 18/11/2026, review of WP-F)

- `runner_requirements.literal_kaggle_input` = `true` (new key; why: the lab modules decoded from CODE_B64 scan the literal /kaggle/input: kaggle/build_l0_notebooks.py line 143 (LOAD cell, lid_strict.json fallback), Experiment/T3ENC-01/run.py lines 33-34 (lid_strict.json), CLEANTUNE-01/run.py lines 168-182 and 850 (_find / _find_all), FACELORA-03/run.py 53-54, SCALE-CC-01/run.py 32-33, DSEL-02/run.py 53-54, DSEL-03/run.py 75-76, VOICELORA-01/run.py 48-49, AUXP-01/run.py 34, IBLORA-01/run.py 30 (same modules as WP-G's e112f1sx)).
- `notes` / `todo`: identity-linkage constant OVERLAP of the embedded lab code (user decision before publication, DESIGN G17), dev_data flag for e2e-dev-p (WP-A), private storage of the exact notebook.

## E. Frozen t3s01-tables-p kept as saved only (stage.json only; integration fixer r1, 18/11/2026)

- type: `variant` (resolution rule only)
- where: stage.json input `datasets/ngnnguynchn/flag2027-t3s01-tables-p`: `"resolve": "saved_only"`
- why: since 18/11 the only producer-like stage for these tables (`t3s01_trca`, for t219_dumpv4) makes keys + trCA only, under its own id; this evidence stage keeps the frozen tables with their `trT3S_*` columns. Documented requirement-1 exception (evidence, not in the 13.54 numbers)
- effect on numbers: none

