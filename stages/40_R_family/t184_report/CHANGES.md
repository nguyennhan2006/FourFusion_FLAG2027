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
- `notes`: identity-linkage constant OVERLAP of the embedded lab code (disclosed in docs/COMPLIANCE_NOTES.md §4), dev_data flag for e2e-dev-p, private storage of the exact notebook.

## E. Frozen t3s01-tables-p kept as saved only (stage.json only; integration fixer r1, 18/11/2026)

- type: `variant` (resolution rule only)
- where: stage.json input `datasets/ngnnguynchn/flag2027-t3s01-tables-p`: `"resolve": "saved_only"`
- why: since 18/11 the only producer-like stage for these tables (`t3s01_trca`, for t219_dumpv4) makes keys + trCA only, under its own id; this evidence stage keeps the frozen tables with their `trT3S_*` columns. Documented requirement-1 exception (evidence, not in the 13.54 numbers)
- effect on numbers: none

## Requirement 1: frozen wiring inputs removed (19/11/2026 (requirement-1 closure))

- input `datasets/ngnnguynchn/flag2027-cc5-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (t3recipe02_fold*.parquet) are read only by DSEL-02/run.py with_cc5() and DSEL-03/run.py attach(), which are reached only from the DSEL-02 / DSEL-03 / SCALE-CC-01 / AUXP-01 report commands; this stage runs CLEANTUNE-01 / FACTOR-01 / FUSENET-01 commands, which never call them, and no cell checks the dataset's presence. No producing stage exists (old run FLAG_57).
- input `datasets/ngnnguynchn/flag2027-dsel02-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (dsel02_fold*.parquet) are read only by DSEL-02/run.py and DSEL-03/run.py report paths, which this stage's commands never call; no presence check. No producing stage exists (old run FLAG_58).
- input `datasets/ngnnguynchn/flag2027-facelora03-tables-p` dropped (stage.json `inputs_dropped`): attached in the original run; real runs only copy facelora03_B3_fold*.parquet into SP/out/safe04 (no assert) and nothing reads the copies: the only reader is the smoke stand-in (FL3Z_SMOKE / FLAG_SMOKE), which now takes the key columns of the SAFE-04 streams tables instead (patch of the same date). No producing stage exists (old runs FLAG_86 / 86b).
- input `datasets/ngnnguynchn/flag2027-lab-avfull-p` dropped (stage.json `inputs_dropped`): attached in the original run only to satisfy the lab presence check OPTIONAL_DIRS['Experiment/CAFUSE-EN/feats_av_full'] = ['CLEANTUNE-01']; its content (AudioCLIP arrays) is read only by EN-T3-01 table_en() via CLEANTUNE-01 fit_en_clean(), called from the zip command when F0_ZIP is unset, i.e. only in e112f0sx (F0 English). The presence check is made optional by the patch of the same date. No producing stage exists (team package labavfull/, FLAG_32 run 2).
- input `datasets/ngnnguynchn/flag2027-feats-scout-wsvrdn-p` dropped (stage.json `inputs_dropped`): attached in the original run but never opened by this stage: CLEANTUNE-01 _scout_mav() / _scout_test() are called only when training rows are built (mav_extra) or test lists are scored (final / final_t3s); the FACTOR-01 / FUSENET-01 report commands of this stage only call merge_t3s, _lr, _nested_design and stack02. No producing stage exists.
- patch (smoke): `pd.read_parquet(SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet")` -> `pd.read_parquet(SP / "out/safe04" / f"streams_fold{s}.parquet")[["fold", "data", "protocol", "y", "fi", "vi", "XALL_deep0"]].rename(columns={"XALL_deep0": "trB3"})`. Why: smoke only (FL3Z_SMOKE / FLAG_SMOKE): the stand-in tables were noisy copies of FACELORA-03's B3 tables (datasets/ngnnguynchn/flag2027-facelora03-tables-p, no producing stage). They now start from the SAFE-04 streams table of the same fold (frozen bundle / stage safe04_lab, already in SP/out/safe04): same 7 406 rows and the same key columns fold, data, protocol, y, fi, vi (checked on fold 1 against the team copies, 19/11); XALL_deep0 plays the role of trB3, whose columns the stand-in drops anyway. Effect on numbers: none in real runs (smoke stand-ins only)
- patch (wiring): `"Experiment/CAFUSE-EN/feats_av_full": ["CLEANTUNE-01"]}` -> `"Experiment/CAFUSE-EN/feats_av_full": []}`. Why: the lab link loop asserts that every OPTIONAL_DIRS folder listed for the running experiment is attached; feats_av_full (AudioCLIP arrays of datasets/ngnnguynchn/flag2027-lab-avfull-p, no producing stage) is read only by the F0 English zip path (e112f0sx), which this stage never runs, so the folder becomes optional here. Effect on numbers: none (the folder is never opened by this stage's commands)
- evidence: `repo_map/build/REQ1_GAPS.md` (team archive); `python runner/check_stage.py` rebuilds this stage

