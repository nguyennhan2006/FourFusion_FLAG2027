# CHANGES: t189_b2

Stage code = the 6 code cells of the exact notebook thanhhuyen11/flag-2027-nb-189-fvbind-b2-ib0 (kapi_runs:1580, md5 4a5e41949fc3...).

Every change against the code of the original run (fields: type, where, old, new, why, effect on numbers). The runner also maps `/kaggle/input` and `/kaggle/working` to the stage's own folders and prints a `[cell]` line before every cell. The mapping does not touch the code below except through change 3 (the embedded loader reads `RFS_XDIRS`, which the mapping rewrites).

## 1. organisers' data removed from the repository copy of the notebook

- type: `organiser-data-exclusion`
- where: original/flag-2027-nb-189-fvbind-b2-ib0.redacted.ipynb, setup cell, tuple ("train/gender.csv", <base64>) (byte offset 5603 of the exact file)
- old: a 228-character gzip + base64 literal = the organisers' meta_file_train_set.csv (sha256 fd7991a3...)
- new: `""` (empty string): the cell writes an empty rfs/train/gender.csv
- why: user rule 6: no organisers' data in the repository. The literal is exactly gzip.compress(csv, compresslevel=9, mtime=0) in base64, so `python stages/60_fvbind/restore_exact.py --csv <meta_file_train_set.csv> --out <dir>` rebuilds the byte-exact notebook (md5 checked)
- effect on numbers: none: only assemble.py (RFS_MODE=B, stage t163b) reads train/gender.csv; this stage never runs mode B

## 2. ImageBind code pinned to a commit

- type: `pin`
- where: setup cell (notebook cell 2 = stage.ipy cell 1; RFS_MODE=L branch, not executed here) and FVBIND cell (notebook cell 5 = stage.ipy cell 4; executed); 2 occurrences
- old: `pip -q install --no-deps git+https://github.com/facebookresearch/ImageBind`
- new: `pip -q install --no-deps git+https://github.com/facebookresearch/ImageBind@53680b02d7e37b19b124fa37bae4b6c98c38f5be`
- why: the run installed GitHub HEAD without a commit; 53680b02d7e37b19b124fa37bae4b6c98c38f5be is main on 2026-11-05 (release/docker/Dockerfile:22-23). NOT VERIFIED that HEAD was unchanged on 14-15/11; the model weights are fixed by the sha256 check of imagebind_huge.pth; the wiring check of the cell (LoRA = 0 reproduces stored v_ibv, cos >= 0.99) still guards the result
- effect on numbers: none expected

## 3. RFS_XDIRS set to the literal input root, so the runner's path mapping reaches the embedded loader

- type: `path`
- where: kapi parameter cell (notebook cell 0 = stage.ipy cell 0), one line added after `os.environ['FVB_ARMS'] = 'B2,IB0'` (patch 1 of stage.json)
- old: (no line) RFS_XDIRS unset
- new: `os.environ['RFS_XDIRS'] = '/kaggle/input'` + comment
- why: `_xload` of the gzip+base64-embedded rfs_train.py (original/embedded/train/rfs_train.py:86) scans `RFS_XDIRS` or else the literal `/kaggle/input` for the FACTOR-01 extra blocks (f_c5s, f_farl, v_ibv: the frozen face anchor of B2 and the rows of its wiring check, reached through fvbind.py:179 `R.load_extra(E, V4, ['f_c5s', 'f_farl', 'v_ibv'])`). The runner rewrites `/kaggle/input` only in the text of stage.ipy, never inside code decoded at run time, and this stage does not declare `runner_requirements.literal_kaggle_input`. Without the line, on Linux / Windows the blocks are not found (AssertionError 'FACTOR-01: ... not found'); in a Kaggle orchestrator session the loader would index the whole notebook input (every attached FF_DATA folder, but not artifacts made earlier in the same session under FF_WORK) and merge same-name copies row-wise. With the line, the runner's rewrite turns the value into the stage's own input view, which holds exactly the inputs of the original run; the subprocesses (`run()` / fvbind.py) inherit `os.environ`.
- effect on numbers: none. On Kaggle the value equals the default root of `_xload`, so the original notebook behaves the same; under the runner the loader sees the same files as the original run.

## 4. input datasets/trngsnam/flag2027-feats-peface01-pe0-p dropped

- type: `wiring`
- where: stage.json inputs (no code change); the dataset stays in original/kernel-metadata.json as evidence of the original attachment
- old: attached (only indexed by the .npy scan of `_xload`)
- new: not attached; recorded under `inputs_dropped` in stage.json
- why: never read. `_xload` selects files by exact name (vox_face_clipC5.npy, face_farl16.npy, voice_rfibv.npy, ...) and this dataset holds only `*_peL*.npy` (block f_pel, not requested by this stage), `bn42_clean.json`, `facelora_state.pt` and `train_log.json` (13 members, listing of E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/datasets/trngsnam__flag2027-feats-peface01-pe0-p.zip); no cell of the stage looks these names up (`find_all` asks only ext.npz, v4.npz, lid_strict.json, ct01r2_fold1.parquet, lora_C1.pt, v4_extra.npz, tar0000.npz). It has no producing stage (configs/data_map.json class wiring, producer null: a walk back to raw data could not pass it, user requirement 1) and it holds Evaluation-list features (face_peL_{gender,no_gender}_*.npy) that a training stage has no reason to mount.
- effect on numbers: none

## 5. working-folder copies declared scratch (stage.json only)

- type: `wiring`
- where: stage.json `scratch: ["work", "imagebind_huge.pth"]` (no code change)
- old: everything left in the working folder became the kernel output, including work/lora_C1.pt, work/v4_extra.npz (copies of the LoRA dataset) and imagebind_huge.pth (4.5 GB download)
- new: the runner does not collect these three items
- why: nothing downstream reads them (t192_fvbx reads only fvb_out/fvbB2_state.pt and fvb_out/fvbB2_emb.npz; it downloads its own imagebind_huge.pth into its working folder); dropping them saves 4.5 GB per run and keeps the fake lora_C1.pt of the smoke branch out of the collected output, where the variant file check (every lora_C1.pt must be the original C1) would refuse it
- effect on numbers: none

## 6. runner requirement declared: literal /kaggle/input not needed (stage.json only; integration fixer r1, 18/11/2026)

- type: `path`
- where: stage.json `runner_requirements = {"literal_kaggle_input": false, "why": ...}`
- old: no declaration; the runner's payload scan found the literal root of `rfs_train.py _xload` and treated the stage as literal (refused on Windows, separate Kaggle session needed)
- new: explicit `false` with the reason; runner/ffcore.py `is_literal` now honours an explicit, reasoned `false` (`FF_LITERAL_HONOUR_FALSE=0` restores detection)
- why: INTEGRATION_r1 M1. The only literal use is `_xload` (embedded rfs_train.py:86), redirected by the RFS_XDIRS patch (change 3).
- effect on numbers: none

## 7. original LoRA C1 input resolved as saved only (stage.json only; integration fixer r1, 18/11/2026)

- type: `variant`
- where: stage.json input `datasets/onnhqunhh/flag2027-rfs-lora-p`: `"resolve": "saved_only"` (no code change)
- old: the input followed configs/data_map.json `default_saved`; with `FF_FORCE=t163l_orig` the runner preferred the C1 retrained in the same session, so RFX' (this chain) could get a different C1 than RFX (WP-F stages use `saved_only`)
- new: always the saved ORIGINAL C1 (`FF_DATA/03_checkpoints/rfs_lora_c1_original`, `lora_C1.pt` b97372e3...), exactly as t163p_c5rl, t213_p*, t216_p3
- why: INTEGRATION_r1 M2 (one C1 for both Bangla files of 13.54)
- effect on numbers: none for the 13.54 chain (the original run attached this very dataset)

## 8. raw-v4: unpack train_set.zip in the input view when no extracted view is mounted (19/11/2026 (integration fixer r2))

- type: `path`
- where: stage.ipy cell 2, after `def find_all(name)`; stage.json patch 3; raw-v4 input gets `"exclude": ["dev_set.zip", "dev_set"]`
- old: the os.walk for `train_English.txt` next to `faces/` assumed the extracted Kaggle view; with FF_DATA (zips as delivered) modes (a) and (b) stopped at `assert v4, "thieu raw-v4 ..."` (hidden in smoke, the walk is under `if not SMOKE`)
- new: when `train_English.txt` is absent and one `train_set.zip` is present, unpack it beside itself in the input view (dev_set.zip excluded, never opened); nothing happens when an extracted view is mounted
- why: INTEGRATION_r2 B1; the same block as WP-F's t163b / t163l_orig
- effect on numbers: none

## Not changed (deliberately)

- FVB_ARMS=B2,IB0 (kapi parameter cell kept verbatim; IB0 runs after B2 is saved, about 8 minutes, not needed for 13.54).
- The time rule of train_big (steps cut by measured s/step) is kept: a re-training on other hardware may run another number of steps (original: 584). Optional knob FVB_STEPS=584, not set.
- The MAV root of the gate-1 streams (first v1 / v2 folder found) is kept: gate-1 MAV tables are not part of 13.54.
- pip -q install iopath ftfy regex fvcore (unpinned) kept: environment pinning belongs to docker/ (FF_PIP=pinned).
- Checkpoint download from the pinned HuggingFace mirror PKU-Alignment/AnyRewardModel@40c7e2ade237848a194b2b77517da024c4ec8532 kept (already the live mirror; sha256 prefix d6f6c22b checked by the cell).
