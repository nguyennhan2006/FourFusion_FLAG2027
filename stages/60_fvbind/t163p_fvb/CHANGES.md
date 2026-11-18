# CHANGES: t163p_fvb

Stage code = the 6 code cells of the exact notebook thanhhuyen11/flag-2027-nb-163-rfsfxfvb-p (kapi_runs:1605, md5 308d8715ee7d...).

Every change against the code of the original run (fields: type, where, old, new, why, effect on numbers). The runner also maps `/kaggle/input` and `/kaggle/working` to the stage's own folders and prints a `[cell]` line before every cell. The mapping does not touch the code below except through change 4 (the embedded loader reads `RFS_XDIRS`, which the mapping rewrites).

## 1. organisers' data removed from the repository copy of the notebook

- type: `organiser-data-exclusion`
- where: original/flag-2027-nb-163-rfsfxfvb-p.redacted.ipynb, setup cell, tuple ("train/gender.csv", <base64>) (byte offset 5436 of the exact file)
- old: a 228-character gzip + base64 literal = the organisers' meta_file_train_set.csv (sha256 fd7991a3...)
- new: `""` (empty string): the cell writes an empty rfs/train/gender.csv
- why: user rule 6: no organisers' data in the repository. The literal is exactly gzip.compress(csv, compresslevel=9, mtime=0) in base64, so `python stages/60_fvbind/restore_exact.py --csv <meta_file_train_set.csv> --out <dir>` rebuilds the byte-exact notebook (md5 checked)
- effect on numbers: none: only assemble.py (RFS_MODE=B, stage t163b) reads train/gender.csv; this stage never runs mode B

## 2. upload to Kaggle disabled

- type: `upload`
- where: kapi parameter cell (notebook cell 0 = stage.ipy cell 0)
- old: `os.environ['RFS_UPLOAD'] = 'thanhhuyen11/flag2027-rfsfxfvb-art-p'`
- new: a comment line (see the patch in stage.json)
- why: the runner never uploads to Kaggle; rfs_art/ becomes the artifact datasets/thanhhuyen11/flag2027-rfsfxfvb-art-p through stage.json exports; with RFS_UPLOAD unset the upload cell prints and skips
- effect on numbers: none

## 3. ImageBind code pinned to a commit

- type: `pin`
- where: setup cell (notebook cell 2 = stage.ipy cell 1; RFS_MODE=L branch); 1 occurrence; inert in this run (mode P never executes it), pinned for a uniform rule
- old: `pip -q install --no-deps git+https://github.com/facebookresearch/ImageBind`
- new: `pip -q install --no-deps git+https://github.com/facebookresearch/ImageBind@53680b02d7e37b19b124fa37bae4b6c98c38f5be`
- why: the run installed GitHub HEAD without a commit; 53680b02d7e37b19b124fa37bae4b6c98c38f5be is main on 2026-11-05 (release/docker/Dockerfile:22-23). NOT VERIFIED that HEAD was unchanged on 14-15/11; the model weights are fixed by the sha256 check of imagebind_huge.pth
- effect on numbers: none expected

## 4. RFS_XDIRS set to the literal input root, so the runner's path mapping reaches the embedded loader

- type: `path`
- where: kapi parameter cell (notebook cell 0 = stage.ipy cell 0), one line added after `os.environ['RFS_CFG'] = '{"add": ["f_c5s", "v_rl", "v_fvb"]}'` (patch 2 of stage.json)
- old: (no line) RFS_XDIRS unset
- new: `os.environ['RFS_XDIRS'] = '/kaggle/input'` + comment
- why: `_xload` of the gzip+base64-embedded rfs_train.py (original/embedded/train/rfs_train.py:89) scans `RFS_XDIRS` or else the literal `/kaggle/input` for the FACTOR-01 extra blocks (f_c5s, v_rl, v_fvb, reached through load_extra for RFS_CFG add, rfs_train.py:521 / 547 / 565 / 707 (mode P and its selfcheck)). The runner rewrites `/kaggle/input` only in the text of stage.ipy, never inside code decoded at run time, and this stage does not declare `runner_requirements.literal_kaggle_input`. Without the line, on Linux / Windows the blocks are not found (AssertionError 'FACTOR-01: ... not found'); in a Kaggle orchestrator session the loader would index the whole notebook input (every attached FF_DATA folder, but not artifacts made earlier in the same session under FF_WORK) and merge same-name copies row-wise. With the line, the runner's rewrite turns the value into the stage's own input view, which holds exactly the inputs of the original run; the subprocesses (`run()` / fvbind.py) inherit `os.environ`.
- effect on numbers: none. On Kaggle the value equals the default root of `_xload`, so the original notebook behaves the same; under the runner the loader sees the same files as the original run.

## 5. input datasets/trngsnam/flag2027-feats-peface01-pe0-p dropped

- type: `wiring`
- where: stage.json inputs (no code change); the dataset stays in original/kernel-metadata.json as evidence of the original attachment
- old: attached (only indexed by the .npy scan of `_xload`)
- new: not attached; recorded under `inputs_dropped` in stage.json
- why: never read. `_xload` selects files by exact name (vox_face_clipC5.npy, voice_rn293.npy, voice_ecapa1024.npy, voice_fvb.npy, ...) and this dataset holds only `*_peL*.npy` (block f_pel, not requested by this stage), `bn42_clean.json`, `facelora_state.pt` and `train_log.json` (13 members, listing of E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/datasets/trngsnam__flag2027-feats-peface01-pe0-p.zip); no cell of the stage looks these names up (`find_all` asks only ext.npz, v4.npz, lid_strict.json, ct01r2_fold1.parquet, lora_C1.pt, v4_extra.npz, tar0000.npz). It has no producing stage (configs/data_map.json class wiring, producer null: a walk back to raw data could not pass it, user requirement 1) and it holds Evaluation-list features (face_peL_{gender,no_gender}_*.npy) that a training stage has no reason to mount.
- effect on numbers: none

## 6. smoke branch uses the mounted original LoRA C1 instead of a stand-in

- type: `variant`
- where: inputs cell (notebook cell 3 = stage.ipy cell 2), `elif SMOKE:` branch, the line `torch.save({"params": [torch.zeros(2)]}, WORK / "lora_C1.pt")` (patch 3 of stage.json)
- old: the smoke branch always wrote a stand-in lora_C1.pt
- new: it copies the first mounted lora_C1.pt (the original C1 of datasets/onnhqunhh/flag2027-rfs-lora-p); the stand-in only when none is mounted
- why: stage_release copies work/lora_C1.pt into rfs_art/lora_C1.pt (the exported artifact); the runner's variant file check refuses any lora_C1.pt that is not the original C1, so the declared smoke run could never pass. Mode P only copies the file, never loads it. Same patch as stage t163p_c5rl (WP-F)
- effect on numbers: none (FLAG_SMOKE=1 path only)

## 7. runner requirement declared: literal /kaggle/input not needed (stage.json only; integration fixer r1, 18/11/2026)

- type: `path`
- where: stage.json `runner_requirements = {"literal_kaggle_input": false, "why": ...}`
- old: no declaration; the runner's payload scan found the literal root of `rfs_train.py _xload` and treated the stage as literal (refused on Windows, separate Kaggle session needed)
- new: explicit `false` with the reason; runner/ffcore.py `is_literal` now honours an explicit, reasoned `false` (`FF_LITERAL_HONOUR_FALSE=0` restores detection)
- why: INTEGRATION_r1 M1. The only literal use is `_xload` (embedded rfs_train.py:89), redirected by the RFS_XDIRS patch (change 4).
- effect on numbers: none

## 8. original LoRA C1 input resolved as saved only (stage.json only; integration fixer r1, 18/11/2026)

- type: `variant`
- where: stage.json input `datasets/onnhqunhh/flag2027-rfs-lora-p`: `"resolve": "saved_only"` (no code change)
- old: the input followed configs/data_map.json `default_saved`; with `FF_FORCE=t163l_orig` the runner preferred the C1 retrained in the same session, so RFX' (this chain) could get a different C1 than RFX (WP-F stages use `saved_only`)
- new: always the saved ORIGINAL C1 (`FF_DATA/03_checkpoints/rfs_lora_c1_original`, `lora_C1.pt` b97372e3...), exactly as t163p_c5rl, t183_p*, t213_p*, t216_p3
- why: INTEGRATION_r1 M2 (one C1 for both Bangla files of 13.54)
- effect on numbers: none for the 13.54 chain (the original run attached this very dataset)

## Not changed (deliberately)

- RFS_MODE=P and RFS_CFG={"add": ["f_c5s", "v_rl", "v_fvb"]} (kapi parameter cell) kept verbatim.
- Inputs kept as attached (except the never-read peface01 dataset, change 5), including the original LoRA C1 dataset onnhqunhh/flag2027-rfs-lora-p (variant 13.54) and the datasets that are only indexed; kaggle/rfx_chain.sh (LORAD line 11 = fixed C1, no FLAG_192 kernel on the P push) is NOT used.
