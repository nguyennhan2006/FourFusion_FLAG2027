# e112f1sx: changes from the original run

Original run: `thanhhuyen11/flag-2027-nb-112f1sx-eval` (CPU, no internet), pushed 12/11/2026 00:52 (UTC+7) by
`kaggle/evalrun_chain.sh` (kapi_runs line 1485, pushed notebook md5 `681eb47b0d70589baae3e5295565880f`); its zip was downloaded at
01:02 to `Experiment/EVALPATH-U/eval_out/f1s_eval.zip`.
Exact executed notebook (team package `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/2_eval/06_flag-2027-nb-112f1sx-eval.ipynb`):
sha256 `4b746a07ce9b72e12cc253e364e5b4b5400a2c230e479e4e6eaccaf12b6b74ad`, same code cells as the pushed bytes. It is **not** in this
repository (change 1); `original/` holds the redacted copy `06_flag-2027-nb-112f1sx-eval_redacted.ipynb`
(sha256 `1d3ee6080b1a7fcbc9201d20cf2bbb1fb7983306b83d84d66a711e4fb27e1628`).

| # | where | old → new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | code cell 3/8 "03 embedded code" (notebook cell 4), the `CODE_B64 = "…"` line | the base64 code zip (39 members, zip sha256 `d40f3b8606531936a45a815c01b5345ab42f1f343d1ebbaf262619fa15469cd3`) → the same zip without member `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (md5 `ae902d1c7e87783bf88e345b459e39f2`; 38 members kept byte for byte, zip sha256 `b70f04635b0c5d9b299cec99828ef5478da5bf31db5dac76919015a50aa2f53a`), plus one `# FF-REDACTED …` comment line before `CODE_B64` | the removed member is a dev-phase score zip: dev data must not be published. Its only reader is `Experiment/CLEANTUNE-01/run.py` line 635 (embedded copy md5 `634b708ef21f63d4366f0bafc6daaaaa`), `ref = FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None`; this run sets `E2E_SPLIT = 'eval'` (code cell 01). The md5 check of the cell loops over the members present, each still equal to its `CODE_MD5` entry. | none: the dev branch is never taken | dev-exclusion |

Change 1 is applied to the notebook in `original/` (the old text is dev data); the zip is rebuilt by raw copy of the kept members
(deterministic, no recompression). Check with the exact notebook:

```
python stages/50_eval/_tools/redact_dev_refs.py e112f1sx <exact 06_flag-2027-nb-112f1sx-eval.ipynb> \
       --check stages/50_eval/e112f1sx/original/06_flag-2027-nb-112f1sx-eval_redacted.ipynb
```

Code built from the **exact** notebook is `release/flag2027_e2e/code/e112f1sx.ipy` (sha256
`54ef89e453ee3a7b3a19ba00ea758cb1b0b0e5f7aa1dbbd0500211ab5371dc53`); `stage.ipy` here (sha256
`a197100fc4b96a6aff5c7c46503c063c41c16750e53ed55fd0e00ee93b3f4ac9`) differs from it only in that line (and the inserted comment).

## Decisions that are not code changes

| item | decision | effect on numbers |
|---|---|---|
| inputs | the thirteen datasets and five kernel outputs of the run, as pushed (`original/kernel-metadata.json`), including the wiring datasets the verbatim lab code checks for (DESIGN G18) and `flag2027-feats-tap01-voice-p` (only a row mask) | none |
| run flags | set by the notebook itself (code cell 01: `E2E_SPLIT=eval`, `LID_STRICT=1`, `CLEANCOMP=1`, `NOREL=1`; code cell 02: `TAG=ct01r2`, `CT_GROUP=r2`, `BN42=1`, `CT_F1=1`, defaults `FINAL_ARM=CA`, `CT_FINAL_SEEDS=10`); `stage.json` `env` is empty | none |
| fuser refit | the F1s′ Bangla fuser is fitted inside this run (sklearn logistic regression on the SAFE-04 out-of-fold tables); a re-run on another library build may give slightly different coefficients (see `Experiment/E2E/e2e_lib.py` `check_v11_fuser` docstring); the saved JSON (sha256 `c4737367…`) is the weight used by the 13.54 zip | as stated |
| saved output | the saved run output (`FF_DATA/04_run_outputs/flag-2027-nb-112f1sx-eval`) holds `lab/Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (the decoded unredacted code zip): private storage only | none |
| literal `/kaggle/input` (fix round 18/11) | `runner_requirements`: `literal_kaggle_input`, `kaggle_strict: true`, `kaggle: separate session with exactly these inputs`. The modules decoded from `CODE_B64` take the first hit of their own scan of the literal `/kaggle/input`; in one Kaggle orchestrator session that scan sees the orchestrator's attachments (saved copies, dev-list arrays of `feats-screen2-p`), not the outputs made in the session. Run this stage on Linux / Docker (symlink) or in a dedicated Kaggle session with exactly its inputs attached; the runner refuses it otherwise (`runner_requirements.kaggle_strict`). Injecting `CT_DIRS` was not done: `CT_DIRS` lists folders that hold the file directly, while the files sit deep in the virtual input, and it covers only CLEANTUNE-01 lookups. | none |
| wiring inputs (G18) | `cc5-oof-p`, `dsel02-oof-p`, `facelora03-tables-p`, `lab-avfull-p` were never read by this stage and are dropped (19/11, see below) | none |
| output content | the saved output holds `lab/` with the dev score zip of CLEAN-EN-01 (decoded from the unredacted `CODE_B64`) and a copy of the head trained with auxiliary VoxBlink rows: `stage.json` `outputs.contains` (`dev_data`, `voxblink` true); private storage only. WP-A: set `voxblink: true` on this run-output entry of `configs/data_map.json`. | none |

## Requirement 1: frozen wiring inputs removed (19/11/2026 (requirement-1 closure))

- input `datasets/ngnnguynchn/flag2027-cc5-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (t3recipe02_fold*.parquet) are read only by DSEL-02/run.py with_cc5() and DSEL-03/run.py attach(), which are reached only from the DSEL-02 / DSEL-03 / SCALE-CC-01 / AUXP-01 report commands; this stage runs CLEANTUNE-01 / FACTOR-01 / FUSENET-01 commands, which never call them, and no cell checks the dataset's presence. No producing stage exists (old run FLAG_57).
- input `datasets/ngnnguynchn/flag2027-dsel02-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (dsel02_fold*.parquet) are read only by DSEL-02/run.py and DSEL-03/run.py report paths, which this stage's commands never call; no presence check. No producing stage exists (old run FLAG_58).
- input `datasets/ngnnguynchn/flag2027-facelora03-tables-p` dropped (stage.json `inputs_dropped`): attached in the original run; real runs only copy facelora03_B3_fold*.parquet into SP/out/safe04 (no assert) and nothing reads the copies: the only reader is the smoke stand-in (FL3Z_SMOKE / FLAG_SMOKE), which now takes the key columns of the SAFE-04 streams tables instead (patch of the same date). No producing stage exists (old runs FLAG_86 / 86b).
- input `datasets/ngnnguynchn/flag2027-lab-avfull-p` dropped (stage.json `inputs_dropped`): attached in the original run only to satisfy the lab presence check OPTIONAL_DIRS['Experiment/CAFUSE-EN/feats_av_full'] = ['CLEANTUNE-01']; its content (AudioCLIP arrays) is read only by EN-T3-01 table_en() via CLEANTUNE-01 fit_en_clean(), called from the zip command when F0_ZIP is unset, i.e. only in e112f0sx (F0 English). The presence check is made optional by the patch of the same date. No producing stage exists (team package labavfull/, FLAG_32 run 2).
- patch (smoke): `pd.read_parquet(SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet")` -> `pd.read_parquet(SP / "out/safe04" / f"streams_fold{s}.parquet")[["fold", "data", "protocol", "y", "fi", "vi", "XALL_deep0"]].rename(columns={"XALL_deep0": "trB3"})`. Why: smoke only (FL3Z_SMOKE / FLAG_SMOKE): the stand-in tables were noisy copies of FACELORA-03's B3 tables (datasets/ngnnguynchn/flag2027-facelora03-tables-p, no producing stage). They now start from the SAFE-04 streams table of the same fold (frozen bundle / stage safe04_lab, already in SP/out/safe04): same 7 406 rows and the same key columns fold, data, protocol, y, fi, vi (checked on fold 1 against the team copies, 19/11); XALL_deep0 plays the role of trB3, whose columns the stand-in drops anyway. Effect on numbers: none in real runs (smoke stand-ins only)
- patch (wiring): `"Experiment/CAFUSE-EN/feats_av_full": ["CLEANTUNE-01"]}` -> `"Experiment/CAFUSE-EN/feats_av_full": []}`. Why: the lab link loop asserts that every OPTIONAL_DIRS folder listed for the running experiment is attached; feats_av_full (AudioCLIP arrays of datasets/ngnnguynchn/flag2027-lab-avfull-p, no producing stage) is read only by the F0 English zip path (e112f0sx), which this stage never runs, so the folder becomes optional here. Effect on numbers: none (the folder is never opened by this stage's commands)
- evidence: `repo_map/build/REQ1_GAPS.md` (team archive); `python runner/check_stage.py` rebuilds this stage

