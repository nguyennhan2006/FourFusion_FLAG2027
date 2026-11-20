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

## Comment edit (20/11/2026)

Patches 3..3 of `stage.json` (ids `comment_team_label_*`) change comment text only: another team's label is replaced by a neutral wording (public naming rule, `docs/NAMING.md`). No code, path, parameter or number changes; the original notebook in `original/` is unchanged.
- comment now reads `the amendment of 11/10` (1x; old text in the `stage.json` patch)
- `stage.ipy` sha256 52c3589a65e03f2b... -> 152a7207829d7e87...
