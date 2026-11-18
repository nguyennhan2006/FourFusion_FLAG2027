# a_factor: changes from the original step

Original step: CPU, team machine, 13/11/2026 12:00 (UTC+7), `kaggle/rfx_chain.sh` state 3 (log `original/rfx_chain_2026-11-13.log`).
It built `Experiment/POSTEVAL-01/zips/posteval_FACTOR_U_c5rl.zip` (sha256 `23702a4295b7189c…`) and
`posteval_FACTOR_ENFUSE_c5rl.zip` (sha256 `9cf5bbe4e3af1418…`, Evaluation 16.99, the base of three of the four files of 13.54).
The exact command list is in `original/SOURCE.json` (`commands_as_run`).

**Script code: no change.** The six files in `tree/` are byte copies (sha256 in `stage.json` `tree_sha256`):

| file in `tree/` | copied from | note |
|---|---|---|
| `Experiment/EVALPATH-U/make_eval_zip.sh` | same path | unchanged since 3/11; the file that ran |
| `Experiment/EVALPATH-U/check_eval_zip.py` | same path | unchanged since 3/11; the file that ran |
| `Experiment/XTEAM-03/run.py` | `release/fourfusion_flag2027/Experiment/XTEAM-03/run.py` (3/11) | **the version that ran on 13/11.** The repository file was edited on 15/11 (FUSEDIM-02: optional `XT_W`, unset = unchanged); the diff between the two is exactly that edit. Today's file is kept for reference as `original/XTEAM-03_run_current_2026-11-15.py`; both give the same bytes with `XT_W` unset. |
| `Experiment/ENFUSE-U/run.py` | same path | today's file: `ENF_RFS` / `ENF_U` / `ENF_OUT` existed on 13/11; FUSEDIM-02 added the optional `ENF_W` on 15/11 (unset here). No byte copy of the 13/11 version exists; cell 5 asserts that the output equals the independent `rebuild_final.py`. |
| `Experiment/E2E/e2e_lib.py` | same path | md5 `f8efa929…` = the copy embedded in the 112f1sx Evaluation run; unchanged since 5/11 |
| `release/factor_enfuse/rebuild_final.py` | same path | = team package `rebuild/rebuild_final.py` |

## Changes in how the commands are run

| # | where | old (13/11) → new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | working directory (cells 2–4) | the repository root (outputs written into `Experiment/POSTEVAL-01/zips/`, `Experiment/ENFUSE-U/out/`, `Experiment/POSTEVAL-01/manifest.json`) → a copy of the repository layout under `/kaggle/working/tree`, holding the six scripts and the six inputs at the repository paths the run read | the step must not overwrite repository files (`ENFUSE-U/run.py` overwrites its output zip and edits the POSTEVAL manifest); DESIGN G16 "write to a new path" | none (same scripts, same relative paths) | path |
| 2 | `Experiment/POSTEVAL-01/manifest.json` (cell 2) | the repository manifest with earlier entries → a fresh `{"zips": {}}` | both the freeze snippet and `ENFUSE-U/run.py` only add an entry under `"zips"` | none (bookkeeping file only) | path |
| 3 | both `check_eval_zip.py` calls (cells 3–4) | `--peer Experiment/EVALPATH-U/eval_out/submission_U_eval.zip` (the U′ Evaluation zip) → `--peer Experiment/EVALPATH-U/eval_out/f1s_eval.zip Experiment/FACTOR-01/out/rfsfxc5rl_eval.zip` | the U′ zip belongs to another system and is not an input of this chain; `--peer` only checks that file names and pair-id sets agree, which the two input zips check as well | none (a check, not a computation) | wiring |
| 4 | environment of the subprocesses (cell 3) | inherited → inherited minus `XT_W`, `ENF_W`, `ENF_RFS`, `ENF_U`, `ENF_OUT`, `E2E_DIR`, `E2E_SPLIT`, `NOREL`; `PATH` prefixed with the folder of the running Python | 13/11 had `XT_W` / `ENF_W` unset (they did not exist yet): guard the 1 : 1 variant; `ENFUSE-U/run.py` sets `E2E_DIR` / `E2E_SPLIT` / `NOREL` itself; `make_eval_zip.sh` calls `python` | none | variant |
| 5 | `--counts` (cells 3–4), only when `FF_SMOKE=1` | `--counts "$CNT"` → no count check | smoke runs of the upstream stages hold fewer pairs | none in normal runs | wiring |
| 6 | cell 5 (new) | — → `rebuild_final.py <six inputs> FACTOR_ENFUSE_c5rl_rebuilt.zip <ENFUSE zip of cell 4>` must print "OK: identical"; member sha256 of both zips compared with the 13/11 zips (asserted under `FF_MODE=assemble` / `FF_STRICT=1`); optional Spearman report against the 13/11 zips when attached | cross-check and verification only; the zips of cells 3–4 are not modified | none | — |
| 7 | `score_dirs()` (cell 1; integration fixer r1, 18/11/2026) | `FF_DATA/05_scores`, `$FF_SCORES` → the same, then `<repo>/assembly/inputs` (and the not-found message names it) | `assembly/inputs/README.md` promises the fallback, and a_fvbind / dp_sigma / final_blend implement it (INTEGRATION_r1 M5); stage.ipy sha256 7717041d... → fd37c904... | none (lookup order only; the files are checked by sha256 / members downstream) | path |

The four input-locating rules of cell 1 (output of the original kernel at its original relative path; else a single attached file with
the flat name of DESIGN 5.3; else `FF_DATA/05_scores`; else `FF_SCORES`) are new code that only decides where the six files are read.
