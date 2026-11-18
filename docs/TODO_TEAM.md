# What is still open (19/11/2026)

State after the second integration review (`repo_map/build/INTEGRATION_r2.md`, team archive) and its fixes. The repository
itself is complete: 93 stages rebuild from their original notebooks / scripts (`python runner/check_stage.py`: 0 errors), the stage
graph has no cycle, unresolved input or variant error (`python runner/ffrun.py dag --compare-skeleton`), the plans of modes (b) and
(c) match `configs/modes.json`, and `python tools/secret_scan.py .` finds 0 high-severity items. What is open is data that lives
outside git, and checks that need a GPU run.

## 1. Before the organisers can run each mode

| # | item | needed by | how |
|---|---|---|---|
| 1 | **Staged 19/11 and mode (c) run on them: byte-identical (upload left).** Ten per-pair score files of mode (c) (≈ 18 MB, list and sha256 in `assembly/inputs/README.md`) | (c) | copy them to `FF_DATA/05_scores/` on the Drive, or into `assembly/inputs/` of a private copy of the repository (`.gitignore` keeps them out of the public tree) |
| 2 | Drive link | all | populate the Drive folder (`DATA.md`; batch 1 = raw / collected data, batch 2 = checkpoints, batch 3 = intermediates and per-pair scores), share it only with the organisers' addresses, and put the link at the top of `README.md` |
| 3 | **Done 19/11 (downloaded; upload left).** RFX′ artifact `thanhhuyen11/flag2027-rfsfxfvb-art-p` and the output of `thanhhuyen11/flag-2027-nb-192-fvbx3` (exist only on Kaggle, DESIGN G4) | (b) gender/Bangla | commands in `tools/PULL_MISSING_RUNS.md` §2; store as `FF_DATA/03_checkpoints/rfx_fvb/` and `FF_DATA/04_run_outputs/flag-2027-nb-192-fvbx3/`; record the sha256 in `configs/data_map.json` and regenerate `CHECKPOINTS.md` (`python tools/build_docs.py`) |
| 4 | **Done 19/11 (made, bit-identical; upload left).** `dp_towers.pt` (DESIGN G5: the original DP script saved scores only) | (b) English | run stage `dp_train` once on the team machine (≈ 12 CPU minutes, `python runner/ffrun.py run --only dp_train`), check that its English scores equal `xjf_scores.npz` (`686c1188…`), upload to `FF_DATA/03_checkpoints/dp_towers/` |
| 5 | **Staged 19/11 (`tools/stage_scores_and_dp_features.py`, sha256 checked; upload left).** The 20 DP feature arrays (DESIGN G13) | (b) English | upload from the team machine to the Drive path given in `DATA.md` |
| 6 | **Staged 19/11 (`tools/stage_checkpoints.py`: 13 / 13 match; upload left).** Checkpoints of `CHECKPOINTS.md` §2 whose status is "saved copy expected on Drive" | (b) | upload the team copies (batch 2) |

## 2. Decisions recorded as exceptions (accept or close)

- Requirement 1 ("every intermediate has a producing stage"): eight wiring datasets and seven embedded data files have no producer
  (`docs/VERIFICATION.md` §4). None of them changes a number of 13.54 except the frozen SAFE-04 tables and the three embedded files
  that shape row sets. Closing them needs the G18 wiring patches (drop presence / row-count checks) and one run to prove them.
- The original LoRA C1 is always taken saved (`docs/VARIANTS.md`); a C1 re-learned by `t163l_orig` is evidence only.

## 3. Checks planned after 20/11

As listed in `docs/VERIFICATION.md` §6: one inference run of the 13.54 variant (≈ 1.5 GPU-hours) with the Spearman criterion; a smoke
run without `dev_set.zip` (`FF_NO_DEV=1`); `t3s01_trca` + `t219_dumpv4` against `xjfdump_v4.parquet`; the G18 wiring patches; a
rebuild of the SAFE-04 tables; the DP voice re-extraction with batch size 1. The FVBIND unpack fix of 19/11 (`t189_b2`,
`t192_fvbx`) is verified by `check_stage.py` only; the first non-smoke run will exercise it.

## 4. Smaller items

- Stage `todo` lists still name the work package that once owned them; several are already done (INTEGRATION_r2 "Minor").
- ImageBind code: some stages pin commit `53680b02…` in code, others install HEAD unless `FF_PIP=pinned`.
- `compute.minutes_original` is unknown for `t213_p1`, `t213_p2`, `t216_p3`, `c46`, so the mode (a) time estimate is low.
- Machine paths of the team (provenance and verbatim code) remain in about 150 files; Kaggle account names appear in artifact ids.
