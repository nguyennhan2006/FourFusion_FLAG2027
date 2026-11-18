# e112f0sx: changes from the original run

Original run: `thanhhuyen11/flag-2027-nb-112f0sx-eval` (CPU, no internet), pushed 12/11/2026 00:41 (UTC+7) by
`kaggle/evalrun_chain.sh` (kapi_runs line 1484, pushed notebook md5 `a4c88727916dc7a9bb990ec4a59bbd02`).
Exact executed notebook (team package `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/2_eval/05_flag-2027-nb-112f0sx-eval.ipynb`):
sha256 `0e0aeae47716e6a7dfa658ad560c33c2ccbcd88dd6c8790c0c6bd43cc00daae6`, same code cells as the pushed bytes. It is **not** in this
repository (change 1); `original/` holds the redacted copy `05_flag-2027-nb-112f0sx-eval_redacted.ipynb`
(sha256 `842b338f3036db6550ee8e7aa59ad4d206faab1ccb14ba686cb6e22beea9808e`).

| # | where | old → new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | code cell 3/8 "03 embedded code" (notebook cell 4), the `CODE_B64 = "…"` line | the base64 code zip (39 members, zip sha256 `a5b7e4dc9e522cdc4e3e8bf4bb2751ba77357a5a39b511177067848529bd518a`) → the same zip without member `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (md5 `ae902d1c7e87783bf88e345b459e39f2`; 38 members kept byte for byte, zip sha256 `9f198dcd077c9c9176e55cccc33c5e138fec5f1a16ff8ae8fedf97497c58aada`), plus one `# FF-REDACTED …` comment line before `CODE_B64` | the removed member is a dev-phase score zip: dev data must not be published. Its only reader is `Experiment/CLEANTUNE-01/run.py` line 635, `ref = FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None`; this run sets `E2E_SPLIT = 'eval'` (code cell 01). The cell's md5 check loops over the members present, each still equal to its `CODE_MD5` entry; `CODE_MD5` keeps the removed file's line (printed only). | none: the dev branch is never taken | dev-exclusion |

Change 1 is applied to the notebook in `original/` (the old text is dev data). The zip is rebuilt by copying the local headers,
compressed data and central-directory records of the 38 kept members (only offsets change; no recompression), so the result is
deterministic. Check with the exact notebook:

```
python stages/50_eval/_tools/redact_dev_refs.py e112f0sx <exact 05_flag-2027-nb-112f0sx-eval.ipynb> \
       --check stages/50_eval/e112f0sx/original/05_flag-2027-nb-112f0sx-eval_redacted.ipynb
```

Code built from the **exact** notebook is `release/flag2027_e2e/code/e112f0sx.ipy` (sha256
`e9a29a4c818d4a535af4454b326417d3c4ff08171b1242ea5679c462cf67c62b`); `stage.ipy` here (sha256
`6b7984a8d4df9c668fd21b97b0311ec2f5902c998c7e9a7204522f7d6cb7c938`) differs from it only in that line (and the inserted comment).

## Decisions that are not code changes

| item | decision | effect on numbers |
|---|---|---|
| role | wiring only: e112f1sx requires exactly one F0 zip of the split (English bytes that a_factor recomputes) | none |
| inputs | the eleven datasets and two kernel outputs of the run, as pushed (`original/kernel-metadata.json`), including the wiring datasets the verbatim lab code checks for (DESIGN G18) | none |
| run flags | set by the notebook itself (code cell 01: `E2E_SPLIT=eval`, `LID_STRICT=1`, `CLEANCOMP=1`, `NOREL=1`; code cell 02: `CT_GROUP=g1`, `BN42=1`, defaults `FINAL_ARM=B0c`, `CT_FINAL_SEEDS=5`); `stage.json` `env` is empty | none |
| saved output | the saved run output (`FF_DATA/04_run_outputs/flag-2027-nb-112f0sx-eval`) holds `lab/Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (the decoded unredacted code zip): private storage only | none |
| literal `/kaggle/input` (fix round 18/11) | `runner_requirements`: `literal_kaggle_input`, `kaggle_strict: true`, `kaggle: separate session with exactly these inputs`. The modules decoded from `CODE_B64` take the first hit of their own scan of the literal `/kaggle/input`; in one Kaggle orchestrator session that scan sees the orchestrator's attachments (saved copies, dev-list arrays of `feats-screen2-p`), not the outputs made in the session. Run this stage on Linux / Docker (symlink) or in a dedicated Kaggle session with exactly its inputs attached; the runner should refuse it otherwise (WP-A todo). Injecting `CT_DIRS` was not done: `CT_DIRS` lists folders that hold the file directly, while the files sit deep in the virtual input, and it covers only CLEANTUNE-01 lookups. | none |
| wiring inputs (G18) | `cc5-oof-p`, `dsel02-oof-p`, `facelora03-tables-p`, `lab-avfull-p` have no producing stage (frozen wiring datasets, presence / row counts only; `stage.json` todo, inputs flagged `wiring_g18`) | none |
| output content | the saved output holds `lab/` with the dev score zip of CLEAN-EN-01 (decoded from the unredacted `CODE_B64`) and a copy of the head trained with auxiliary VoxBlink rows: `stage.json` `outputs.contains` (`dev_data`, `voxblink` true); private storage only. WP-A: set `voxblink: true` on this run-output entry of `configs/data_map.json`. | none |

## Requirement 1: frozen wiring inputs removed (19/11/2026 (requirement-1 closure))

- input `datasets/ngnnguynchn/flag2027-cc5-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (t3recipe02_fold*.parquet) are read only by DSEL-02/run.py with_cc5() and DSEL-03/run.py attach(), which are reached only from the DSEL-02 / DSEL-03 / SCALE-CC-01 / AUXP-01 report commands; this stage runs CLEANTUNE-01 / FACTOR-01 / FUSENET-01 commands, which never call them, and no cell checks the dataset's presence. No producing stage exists (old run FLAG_57).
- input `datasets/ngnnguynchn/flag2027-dsel02-oof-p` dropped (stage.json `inputs_dropped`): attached in the original run but never read by this stage: its tables (dsel02_fold*.parquet) are read only by DSEL-02/run.py and DSEL-03/run.py report paths, which this stage's commands never call; no presence check. No producing stage exists (old run FLAG_58).
- input `datasets/ngnnguynchn/flag2027-facelora03-tables-p` dropped (stage.json `inputs_dropped`): attached in the original run; real runs only copy facelora03_B3_fold*.parquet into SP/out/safe04 (no assert) and nothing reads the copies: the only reader is the smoke stand-in (FL3Z_SMOKE / FLAG_SMOKE), which now takes the key columns of the SAFE-04 streams tables instead (patch of the same date). No producing stage exists (old runs FLAG_86 / 86b).
- patch (smoke): `pd.read_parquet(SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet")` -> `pd.read_parquet(SP / "out/safe04" / f"streams_fold{s}.parquet")[["fold", "data", "protocol", "y", "fi", "vi", "XALL_deep0"]].rename(columns={"XALL_deep0": "trB3"})`. Why: smoke only (FL3Z_SMOKE / FLAG_SMOKE): the stand-in tables were noisy copies of FACELORA-03's B3 tables (datasets/ngnnguynchn/flag2027-facelora03-tables-p, no producing stage). They now start from the SAFE-04 streams table of the same fold (frozen bundle / stage safe04_lab, already in SP/out/safe04): same 7 406 rows and the same key columns fold, data, protocol, y, fi, vi (checked on fold 1 against the team copies, 19/11); XALL_deep0 plays the role of trB3, whose columns the stand-in drops anyway. Effect on numbers: none in real runs (smoke stand-ins only)
- evidence: `repo_map/build/REQ1_GAPS.md` (team archive); `python runner/check_stage.py` rebuilds this stage

