# Verification status (19/11/2026)

What has been checked, at which level, and what has not. Nothing in this repository has been run end to end in its repository form
yet; the honest status of each mode is below.

## 1. Levels

| level | meaning | applies to |
|---|---|---|
| L0 static | every `stage.ipy` rebuilds from its `original/` notebook + the listed patches (sha256 compare), scripts match their recorded sha256, graph / plan consistency, syntax | all stages |
| L1 byte-exact | the four member files of the submitted zip are reproduced byte for byte (sha256) | mode (c) |
| L2 inference | Evaluation media + released checkpoints; per-file Spearman ≥ 0.9999 against the submitted zip (expected; GPU is not bit-deterministic) | mode (b) |
| L3 recipe | retraining from raw data; per-file Spearman ≥ 0.99 proposed, never measured | mode (a) |

## 2. L0 (done in the repository, 19/11)

- `python runner/check_stage.py` on all 93 stages: 56 EXACT and 4 EXACT (recipe) for the stages copied from executed notebooks
  (exact-package 26, exact-pulled 5, exact-staging 8, exact-executed-copy 12, builder-regenerated 9), 14 local-script stages whose
  listed files match their sha256, 19 new-code stages that follow the `[cell]` rule; 0 errors. (The nine stages that embed
  `rfs_train.py` were "EXACT with problems" until 18/11: their literal `/kaggle/input` is now declared unreachable / redirected,
  `runner/README.md`.)
- `python runner/ffrun.py dag --compare-skeleton`: 93 stages, no cycle, no duplicate producer, no unresolved or forbidden input, no
  static variant error, no skeleton difference.
- `python runner/ffrun.py plan --assume-saved`: infer 19 stages (= `configs/modes.json`), assemble 5, train 58 (default) / 71
  (`full_rebuild_from_raw`) / 79 (with the collect stages); none blocked.
- Every `.py` parses, every `.json` loads; `tools/secret_scan.py` on the tree before publication.

## 3. Modes

| mode | status | what is missing |
|---|---|---|
| (c) assemble (`python assembly/assemble_exact.py` or `ffrun.py run --mode assemble`) | Logic traced end to end (INTEGRATION_r1 §3). Shown on the team machine with the team's scripts: `blend.py` (r = 1.0, `XJF_SIG="5.0983,3.8376"`) on the two stored ENFUSE zips + `xjf_scores.npz` gives the four member files of `xjf_r10_trsig.zip` byte for byte (repo map S8, check K1); `rebuild_final.py` rebuilds FACTOR_ENFUSE from the six stored files (rebuild report of 14/11). The repository script `assembly/assemble_exact.py` chains these steps; **run 19/11 on the ten files staged for the Drive (`FF_DATA/05_scores/`, copied and checked by `tools/stage_scores_and_dp_features.py`): the four members are identical, and the zip rewritten with the submitted ZipInfo fields has the submitted sha256 `b4354eaf…`.** The ten inputs exist on the team machine with the expected sha256 (re-hashed 18/11). | The ten inputs are staged for the Drive folder (`05_scores/`) but not uploaded yet; not in `assembly/inputs/` of the public tree. Whole-zip sha256 differs by design (zip header time); members are compared. |
| (b) infer | Plan complete (19 stages, all four files); variant guard pins the original C1 artifact and sigma / r. Never run. | 19/11: `thanhhuyen11/flag2027-rfsfxfvb-art-p` and the output of `thanhhuyen11/flag-2027-nb-192-fvbx3` downloaded (G4 closed on the team side; `lora_C1.pt` = `b97372e3…`); `dp_train` run once on the team machine (`tools/dp_train_local.py`; G5 closed): constants and the six English arrays bit-identical to the submission run, and `dp_score.py` on the new `dp_towers.pt` (`b2fe1287…`) bit-identical for all four files. Still to do: upload these, the other checkpoints and the 20 DP feature arrays (G13) to the Drive folder. |
| (a) train | Plan complete back to raw / collected data except the requirement-1 exceptions (§4). Never run (≈ 150–170 GPU-hours by the original runs). | Retraining is not deterministic; the original LoRA C1 is always taken saved by design (`docs/VARIANTS.md`). |

## 4. Requirement 1 (start from raw data): exceptions

`python runner/ffrun.py dag` prints the current list. On 19/11:

| artifact | consumers | in the 13.54 numbers? | why frozen / what would close it |
|---|---|---|---|
| `ff-frozen/flag2027-e2e-dev-p` (= frozen SAFE-04 tables `ngnnguynchn/flag2027-e2e-dev-p`) | default for the F1s′ mixer, XALL bridge, T3 head (`default_saved`); `safe04_compare`, `safe04_lab_core` | yes (pair keys / folds of the mixer tables) | original scratch lost 1/11 (G1); `safe04_lab` (force_only) rebuilds the tables; rebuilt tables may differ slightly |
| `ff-frozen/flag2027-e2e-dev-wiring-p` | `safe04_lab` | no | dev-phase members of the frozen bundle that the verbatim lab code checks for |
| `cc5-oof`, `dsel02-oof`, `facelora03-tables`, `feats-scout-wsvrdn`, `best-v7` | none since 19/11 | no | **closed 19/11**: dropped from every consumer (`inputs_dropped` with the reason) because no code of the chain reads them; `best-v7`'s AdaFace arrays are byte-identical to the f10_models outputs in `lab-data`; the smoke stand-in of the F stages now starts from the SAFE-04 streams table (evidence: team archive `repo_map/build/REQ1_GAPS.md`) |
| `lab-avfull` | `e112f0sx` only (F0 English zip path) | no (F0 English reaches `f1s_eval.zip`, whose English pairs ENFUSE reads only for the pair-id check) | the presence check of the eleven other consumers was made optional on 19/11 (`OPTIONAL_DIRS` patch); producer outside the graph: FLAG_32 run 2 (AudioCLIP on MAV-Celeb v1 / v2) |
| `members` | `e50cc` | no (v6 members feed `en7[:, :4]`, i.e. the retired v9 / v12 English streams) | old package; `P6.train_members` can retrain them (not byte-identical; not wired as a stage) |
| `trngsnam/flag2027-feats-peface01-pe0-p` | `t183_p1`, `t183_p2` (evidence) | no | arms `pel` / `ALL*` of the FACTOR-01 lab read it; dropped 18/11 from every non-evidence stage (never read there) |
| `ngnnguynchn/flag2027-t3s01-tables-p` | `t184_report` (evidence) | no | FLAG_167 runs lost; since 18/11 `t219_dumpv4` (sigma) reads `datasets/ff/flag2027-t3s01-trca-p` made by the new stage `t3s01_trca` from `ct02s-r2ca-tables-p` |
| 7 embedded files without producer stage | F1s′ family, `e50cc` | only `CLEANTUNE-01/ct_round2.json` (it selects the CA recipe: `I = C5CAT4`, `S_A = [T, AIp]`) | configuration files: `ct_round2.json` was written by CLEANTUNE-01 `report()` (FLAG_110r) from the round-1 tables of FLAG_110 (≈ 55–60 GPU-hours, not a stage); `STACK-02/eligible.json` is a hand-written registration, read at import only; `VBSEL-01/vox2_sa_speakers.json` is never read (its producer logic, EDA-VB §4 over the l42 output, reproduces its bytes); the four v9 / v11 / v12 fuser JSONs of `e50cc` are frozen constants of retired systems (producers in the team repository: CAFUSE-01, CAFUSE-EN, BIGB-01, EN-T3-01) |

Also taken frozen by default although a producer exists (rebuild with `FF_USE_SAVED=none` and the `FF_FORCE` set of
`configs/modes.json` → `full_rebuild_from_raw`): the stored LID outputs (l42, l150, l152, lid-strict) and f41, the SAFE-04 tables,
`f14_av` and `ct02s-g1-tables` (class wiring), the original LoRA C1 (always, by design).

## 5. Pinning still open

- Model downloads inside verbatim code without a revision: ImageBind code (`pip install git+…/ImageBind`, HEAD; commit
  `53680b02…` only with `FF_PIP=pinned`), `speechbrain>=1.1.1`, `yangwang825/ecapa-tdnn-vox2` (r08 / RFS), VoxLingua107 (l42,
  c46, c53), MMS-LID-4017 (l152), Whisper-large-v3 (l150). `tools/fetch_models.py` fetches every one at a pinned revision with a
  sha256 check, and the runner seeds the caches from `FF_DATA/06_pretrained` (`FF_MODELS`; `FF_HF_OFFLINE=1` makes the pins hold
  offline). The models are not a stage input of the graph: `models_fetch` writes `pretrained/models`, which the runner uses
  implicitly.
- Hugging Face dataset sources of the collect stages are read at `resolve/main` (commits at submission time not recorded, G27).
- Environment: Kaggle base images pinned by digest, pip pins in `docker/`; the GPU image was never built, no `pip freeze` lock exists
  (`docker/README.md`); the DP branch ran on the team's CPU machine (Python 3.14, `docker/requirements-dp-local.txt`).

## 6. Planned checks (after 20/11, team decision)

One inference run of the 13.54 variant (e112f1sx, e163i, e163i_fvb, assembly; ~1.5 GPU-hours) with the Spearman criterion (G20); a
smoke run without `dev_set.zip` (`FF_NO_DEV=1`, G11); `t3s01_trca` + `t219_dumpv4` against `xjfdump_v4.parquet`; the wiring patches
of G18; a rebuild of the SAFE-04 tables (`safe04_lab`, G1); a re-extraction of the DP Evaluation voice features with batch size 1
(max |dz| and Spearman of the DP scores against the submitted arrays; `docs/COMPLIANCE_NOTES.md` §3).

Fixed after the second integration review (19/11, `repo_map/build/INTEGRATION_r2.md`): B1 (`t192_fvbx` and `t189_b2` now unpack
`train_set.zip` inside their input view when no extracted view is mounted, as `t163b` / `t163l_orig` do; `dev_set.zip` excluded);
M3 (one LoRA C1 policy: every consumer takes the saved original, the in-session lineage text of `e163i` / `variant_1354.json` was
rewritten); M4 (batched extraction disclosed in `docs/COMPLIANCE_NOTES.md` §3).
