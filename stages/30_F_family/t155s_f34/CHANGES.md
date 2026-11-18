# CHANGES: t155s_f34 (exact-pulled; `release/flag2027_e2e/src_nb/flag-2027-nb-155s-ct02ca-f34/`)

## 1. Repository copy of the notebook: embedded DEV score zip removed (type `dev-exclusion`; numbers: none)

- **Where:** `original/flag-2027-nb-155s-ct02ca-f34.redacted.ipynb` (and therefore `stage.ipy`), code cell 3, the string `CODE_B64 = "..."`.
- **Old → new:** the base64 zip of the run's code had 39 members; the repository copy has 38: the member
  `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (91 043 bytes, sha256 `b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d`)
  is removed. Every other member is copied as raw zip entries (same bytes, same CRC, same ZipInfo); the rest of the notebook file is
  byte-identical (`_common/redact_embedded_dev_zip.py`, deterministic: re-running it on the exact notebook gives sha256 `e73a01227a876459ae06eb347e235840a0c7a528cb1a7d87a9b10edabb910496`).
- **Why:** that member holds per-pair scores on the organisers' DEV lists; the published repository must not contain dev data (user rule 6,
  DESIGN 2.8 / 9.4). The byte-exact notebook stays at `d:/Sinh viên CNhan/FLAG_2027_FourFusion/release/flag2027_e2e/src_nb/flag-2027-nb-155s-ct02ca-f34/flag-2027-nb-155s-ct02ca-f34.ipynb` (md5 `c520c8a1ab7bad8689c24217f5f7af70`, sha256 `517e38661df5c51bcfd5938fbacf0816da0d9f2e1b6b16678c794375ae79b2f9`) and belongs in the private FF_DATA store.
- **Effect:** none on any array, model, table or score. The only reader of the file is `zip_ct()` of the embedded
  `Experiment/CLEANTUNE-01/run.py` (line 561: `FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None`),
  and `FACELORA-03/run.py cmp_zip()` returns `None` for a missing file (line 406). In this run `zip_ct()` is never called (no zip / report job), so not even a printed line changes.
  The notebook's `CODE_MD5` dict still lists the member; its check loop iterates over the members present in the zip, so it still passes.

## 2. Code cells

Otherwise verbatim: building stage.ipy from the EXACT notebook gives the sha256 of release/flag2027_e2e/code/t155s_f34.ipy (checked 2026-11-18);
the repository stage.ipy differs from it only inside the `CODE_B64` string.

## 3. Evidence kept with this stage

Bengali-voice leak fix of 3/11/2026 12:15 (EXPERIMENT_LOG entry 346): the embedded Experiment/VOICELORA-01/run.py has md5 6d0dfe2bdf72d2204b32dd1285129207 (= the fixed file), whose line 166 reads `drop = set(json.loads((VB / "lid_summary.json").read_text())["speakers_dropped_ids"]) | T.strict_drop("vb")   # 3/11: LIDSTRICT-01, as vb_rows4`; the run was pushed after the fix; log of this run: `VOICELORA-01 VoxBlink rows: 59826 rows, 13471 speakers`.

## 4. Runner

Runner-level only (not a change of this stage's code): the runner replaces the strings /kaggle/input and /kaggle/working by the stage's own input view and working folder, and prints a `[cell] <stage> k/n` line before each cell (DESIGN 2.4).

Runner-level requirement (2026-11-18 fix; not a change of this stage's code): `stage.json` declares `runner_requirements.literal_kaggle_input: true`. The modules decoded from `CODE_B64` at run time (listed in `runner_requirements.why`, with file and line) are written to /kaggle/working/lab and find their inputs by scanning the LITERAL path /kaggle/input, which the string rewrite above cannot reach. With the flag, the runner points /kaggle/input at this stage's own input view on Linux (symlink for the duration of the stage; `FF_LITERAL_KAGGLE=mounted` when a container already mounts it). On Kaggle the runner cannot change /kaggle/input and only warns: run this stage in a session with exactly its inputs attached (as the original run). Without the flag a Linux / Docker run would fail the asserts (lid_strict.json, adapter arrays, feats_vox2b, CT blocks), and a Kaggle orchestrator session would let the modules pick the session's own attachments (e.g. a saved original instead of an artifact rebuilt in the session).

Smoke note (2026-11-18): the verbatim env cell (cell 01, `os.environ['LAB_FOLDS'] = ...`) runs AFTER the runner has applied `smoke_env` (LAB_FOLDS=1), so a smoke run of this stage trains fold 3 (FL3Z_SMOKE keeps the first fold of the env cell's list; harmless), not fold 1 as the original smoke push did (that push had no env cell). Full runs are unaffected. Not patched (logic fidelity; the runner has no smoke-only patch key); recorded in `stage.json` `smoke_note`.

Licence (2026-11-18): the fold heads train with VoxBlink speakers (VOICELORA-01 rows); the fold tables hold scores only, the training inputs are VoxBlink-derived (CC BY-NC-SA 4.0): private FF_DATA only (`stage.json` `licence`).

## Requirement 1: frozen wiring inputs removed (19/11/2026 (requirement-1 closure))

- input `datasets/ngnnguynchn/flag2027-cc5-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (t3recipe02_fold*.parquet) are read only by DSEL-02/run.py with_cc5() and DSEL-03/run.py attach(), which are reached only from the DSEL-02 / DSEL-03 / SCALE-CC-01 / AUXP-01 report commands; this stage runs CLEANTUNE-01 / FACTOR-01 / FUSENET-01 commands, which never call them, and no cell checks the dataset's presence. No producing stage exists (old run FLAG_57).
- input `datasets/ngnnguynchn/flag2027-dsel02-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (dsel02_fold*.parquet) are read only by DSEL-02/run.py and DSEL-03/run.py report paths, which this stage's commands never call; no presence check. No producing stage exists (old run FLAG_58).
- input `datasets/ngnnguynchn/flag2027-facelora03-tables-p` dropped (stage.json `inputs_dropped`): attached in the original run; real runs only copy facelora03_B3_fold*.parquet into SP/out/safe04 (no assert) and nothing reads the copies: the only reader is the smoke stand-in (FL3Z_SMOKE / FLAG_SMOKE), which now takes the key columns of the SAFE-04 streams tables instead (patch of the same date). No producing stage exists (old runs FLAG_86 / 86b).
- input `datasets/ngnnguynchn/flag2027-lab-avfull-p` dropped (stage.json `inputs_dropped`): attached in the original run only to satisfy the lab presence check OPTIONAL_DIRS['Experiment/CAFUSE-EN/feats_av_full'] = ['CLEANTUNE-01']; its content (AudioCLIP arrays) is read only by EN-T3-01 table_en() via CLEANTUNE-01 fit_en_clean(), called from the zip command when F0_ZIP is unset, i.e. only in e112f0sx (F0 English). The presence check is made optional by the patch of the same date. No producing stage exists (team package labavfull/, FLAG_32 run 2).
- patch (smoke): `pd.read_parquet(SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet")` -> `pd.read_parquet(SP / "out/safe04" / f"streams_fold{s}.parquet")[["fold", "data", "protocol", "y", "fi", "vi", "XALL_deep0"]].rename(columns={"XALL_deep0": "trB3"})`. Why: smoke only (FL3Z_SMOKE / FLAG_SMOKE): the stand-in tables were noisy copies of FACELORA-03's B3 tables (datasets/ngnnguynchn/flag2027-facelora03-tables-p, no producing stage). They now start from the SAFE-04 streams table of the same fold (frozen bundle / stage safe04_lab, already in SP/out/safe04): same 7 406 rows and the same key columns fold, data, protocol, y, fi, vi (checked on fold 1 against the team copies, 19/11); XALL_deep0 plays the role of trB3, whose columns the stand-in drops anyway. Effect on numbers: none in real runs (smoke stand-ins only)
- patch (wiring): `"Experiment/CAFUSE-EN/feats_av_full": ["CLEANTUNE-01"]}` -> `"Experiment/CAFUSE-EN/feats_av_full": []}`. Why: the lab link loop asserts that every OPTIONAL_DIRS folder listed for the running experiment is attached; feats_av_full (AudioCLIP arrays of datasets/ngnnguynchn/flag2027-lab-avfull-p, no producing stage) is read only by the F0 English zip path (e112f0sx), which this stage never runs, so the folder becomes optional here. Effect on numbers: none (the folder is never opened by this stage's commands)
- evidence: `repo_map/build/REQ1_GAPS.md` (team archive); `python runner/check_stage.py` rebuilds this stage

