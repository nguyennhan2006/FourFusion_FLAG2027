# CHANGES: t154_g1_f12 (builder output kept on disk, NOT the executed notebook)

## 1. Repository copy of the notebook: embedded DEV score zip removed (type `dev-exclusion`; numbers: none)

- **Where:** `original/FLAG_110_ct01_g1.redacted.ipynb` (and therefore `stage.ipy`), code cell 3, the string `CODE_B64 = "..."`.
- **Old → new:** the base64 zip of the run's code had 39 members; the repository copy has 38: the member
  `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (91 043 bytes, sha256 `b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d`)
  is removed. Every other member is copied as raw zip entries (same bytes, same CRC, same ZipInfo); the rest of the notebook file is
  byte-identical (`_common/redact_embedded_dev_zip.py`, deterministic: re-running it on the exact notebook gives sha256 `bc058ea7929eb4529cebb5a49e84c72b4c6d444c8aa0ea8236aaadadd99810a1`).
- **Why:** that member holds per-pair scores on the organisers' DEV lists; the published repository must not contain dev data (user rule 6,
  DESIGN 2.8 / 9.4). The byte-exact notebook stays at `d:/Sinh viên CNhan/FLAG_2027_FourFusion/kaggle/FLAG_110_ct01_g1.ipynb` (md5 `56dde966a7bdbba769a7c34d0c50f116`, sha256 `56c0ff6259c1f99d8a789b943465f9a57cdaa82a15b8e3f6d55663abfbf89619`) and belongs in the private FF_DATA store.
- **Effect:** none on any array, model, table or score. The only reader of the file is `zip_ct()` of the embedded
  `Experiment/CLEANTUNE-01/run.py` (line 561: `FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None`),
  and `FACELORA-03/run.py cmp_zip()` returns `None` for a missing file (line 406). The fold jobs never call `zip_ct()` in this notebook (as t155s).
  The notebook's `CODE_MD5` dict still lists the member; its check loop iterates over the members present in the zip, so it still passes.

## 2. Source is a builder output, not the executed notebook (type `wiring`; numbers: unknown, unverified)

- The executed notebook of `nguynthanhnhnn/flag-2027-nb-154-ct02-g1-f12` (pushed 2026-11-03 19:36, md5 `06af774b0f509e3fc540e083ee6cb08d`) was never pulled; its kapi staging folder is empty.
- original/ holds `kaggle/FLAG_110_ct01_g1.ipynb` (mtime 2026-11-03 22:57, after the pushes; md5 `56dde966a7bdbba769a7c34d0c50f116`), redacted. Its embedded code
  files equal those of the executed t155s / t112s runs (same 39 md5); its cells may differ from the executed ones (the kapi transform emulation does not give
  the pushed md5). The run-parameter cell kapi inserted is replaced by `env` in stage.json: `LAB_FOLDS=1,2`, `LID_STRICT=1`, `T3R_QF=1`, `T3R_EMA=0.9995` (2026-11-18 fix: the executed pushes had the last two keys: kaggle/keodiem_chain.sh:24-25 calls sat_push.sh lane 110 without QE and kaggle/sat_push.sh:373 appends `${QE-;T3R_QF=1;T3R_EMA=0.9995}`; docs/RUNS.md:408 / EXPERIMENT_LOG entry 356 (3/11 15:35) put them on by default before the g1 pushes of 19:22-19:37; the original tables ct01g1_fold*.parquet carry the columns trB0c_nf / _nv / _ema / _ema_nf / _ema_nv that the embedded T3RECIPE-01/run.py lines 139-148 writes only under these keys. The tr<arm> columns do not depend on them (t3r.py line 451, RUNS.md:408).) Only the cell-level differences between the 22:57 builder output and the executed notebooks remain unverified.
- These stages only rebuild the WIRING dataset `flag2027-ct02s-g1-tables-p` (B0s head of t112sf0s); the 13.54 chain uses the frozen copy by default.

## 3. Runner

Runner-level only (not a change of this stage's code): the runner replaces the strings /kaggle/input and /kaggle/working by the stage's own input view and working folder, and prints a `[cell] <stage> k/n` line before each cell (DESIGN 2.4).

Runner-level requirement (2026-11-18 fix; not a change of this stage's code): `stage.json` declares `runner_requirements.literal_kaggle_input: true`. The modules decoded from `CODE_B64` at run time (listed in `runner_requirements.why`, with file and line) are written to /kaggle/working/lab and find their inputs by scanning the LITERAL path /kaggle/input, which the string rewrite above cannot reach. With the flag, the runner points /kaggle/input at this stage's own input view on Linux (symlink for the duration of the stage; `FF_LITERAL_KAGGLE=mounted` when a container already mounts it). On Kaggle the runner cannot change /kaggle/input and only warns: run this stage in a session with exactly its inputs attached (as the original run). Without the flag a Linux / Docker run would fail the asserts (lid_strict.json, adapter arrays, feats_vox2b, CT blocks), and a Kaggle orchestrator session would let the modules pick the session's own attachments (e.g. a saved original instead of an artifact rebuilt in the session).
