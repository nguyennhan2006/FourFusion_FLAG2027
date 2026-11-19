# Checks (19/11/2026)

## 1. Code and graph

- `python runner/check_stage.py` on all 87 stages: every `stage.ipy` rebuilds from its `original/` notebook plus the listed patches
  (sha256 compare): 53 EXACT and 4 EXACT (recipe) for the stages copied from executed notebooks (exact-package 23, exact-pulled 5,
  exact-staging 8, exact-executed-copy 12, builder-regenerated 9), 14 local-script stages whose listed files match their sha256, and
  16 new-code stages that follow the `[cell]` rule; 0 errors.
- `python runner/ffrun.py dag --compare-skeleton`: 87 stages, no cycle, no duplicate producer, no unresolved or forbidden input, no
  variant error, no skeleton difference; section LINEAGE: the 73 stages the zip depends on, 0 gaps.
- `python runner/ffrun.py plan --assume-saved`: infer 19 stages (= `configs/modes.json`), assemble 5, train 58 (default) / 74
  (the mode (a) command of the README) / 82 (with the collect stages); none blocked.
- `python runner/ffrun.py plan` with `FF_DATA` = the uploaded Drive folder (local copy of batches 1, 2–3 and 4): assemble 5
  stages, infer 18, train 57; the only missing inputs are the organisers' `00_raw/btc_mavceleb_v4/` and
  `00_raw/btc_evaluation/evaluation_phase.zip`, which are not uploaded. Every file of the 13 checkpoint artifacts and all 19
  dataset zips under `datasets/` match their sha256 in `configs/data_map.json`; batches 2–3 match their `MANIFEST.sha256` (183 files).
- `python runner/selftest.py`: 8/8 cases; `python tools/build_docs.py --check`: generated docs up to date.
- Every `.py` parses, every `.json` loads; a secret scan of the tree finds no high-severity item.

## 2. Results reproduced

| what | how | result |
|---|---|---|
| mode (c): the submitted zip from the ten stored per-pair score files | `python assembly/assemble_exact.py --inputs <FF_DATA>/05_scores` | the four member files are byte-identical to `xjf_r10_trsig.zip`; with the submitted ZipInfo fields the whole zip has the submitted sha256 `b4354eaf…` |
| DP branch retrained | stage `dp_train` on the team's CPU machine (`tools/dp_train_local.py`) | the constants and the English scores are bit-identical to the submission run; `dp_towers.pt` sha256 `b2fe1287…` |
| DP scoring from the saved towers | stage `dp_score` | the scores of all four files are bit-identical to the submitted DP arrays |

GPU arithmetic is not bit-identical across runs, so modes (a) and (b) are compared per file with Spearman's correlation
(`runner/verify.py`), and mode (a) retrains on a non-deterministic GPU (recipe-level rebuild).

## 3. Inputs taken as stored copies

Mode (a) rebuilds the chain from raw and collected data. The following inputs are taken as stored copies in every mode.

| artifact | consumers | role |
|---|---|---|
| `ff-frozen/flag2027-e2e-dev-p` (frozen SAFE-04 tables) | default for the identity-classification mixer, XALL bridge, T3 head | pair keys and folds of the mixer tables; `safe04_lab` (`FF_FORCE=safe04_lab`) rebuilds them |
| `ff-frozen/flag2027-e2e-dev-wiring-p` | `safe04_lab` | dev-phase members of the frozen bundle that the verbatim lab code checks for; not in the 13.54 numbers |
| `lab-avfull` | `e112f0sx` | AudioCLIP features (FLAG_32) for the F0 English path; not in the 13.54 zip |
| `members` | `e50cc` | v6 member seeds of the retired v9 / v12 English streams; not in the 13.54 numbers |
| original LoRA C1 (`onnhqunhh/flag2027-rfs-lora-p`) | every RFX consumer (`resolve: saved_only`) | defines the 13.54 variant (`docs/VARIANTS.md`); `t163l_orig` retrains it as a check only |
| embedded configuration files | identity-classification family, `e50cc` | `CLEANTUNE-01/ct_round2.json` selects the CA recipe (`I = C5CAT4`, `S_A = [T, AIp]`; written by CLEANTUNE-01 `report()` from the round-1 tables of FLAG_110); `STACK-02/eligible.json` is a hand-written registration; the four v9 / v11 / v12 fuser JSONs of `e50cc` are constants of retired systems |

By default the stored LID outputs (l42, l150, l152, lid-strict), f41, `f14_av` and `ct02s-g1-tables` are also taken saved although a
producer exists; `FF_USE_SAVED=none` with the `FF_FORCE` set of `configs/modes.json` → `full_rebuild_from_raw` rebuilds them.

## 4. Pinning

- Pretrained models: `tools/fetch_models.py` fetches every model at a pinned revision with a sha256 check; the runner seeds the caches
  from `FF_DATA/06_pretrained` (`FF_MODELS`; `FF_HF_OFFLINE=1` keeps the pins offline). The verbatim code installs the ImageBind code
  from GitHub; `FF_PIP=pinned` pins commit `53680b02…`.
- Hugging Face dataset sources of the collect stages are read at `resolve/main`; `FF_HF_REVS` pins a commit.
- Environment: Kaggle base images pinned by digest and pip pins in `docker/`; the DP branch ran on the team's CPU machine (Python
  3.14, `docker/requirements-dp-local.txt`).
