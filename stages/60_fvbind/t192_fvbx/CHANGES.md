# CHANGES: t192_fvbx

Stage code = the 5 code cells of the exact notebook thanhhuyen11/flag-2027-nb-192-fvbx3 (kapi_runs:1603, md5 ba933cdea97e...).

Every change against the code of the original run (fields: type, where, old, new, why, effect on numbers). The runner also maps `/kaggle/input` and `/kaggle/working` to the stage's own folders and prints a `[cell]` line before every cell; neither touches the code below.

## 1. organisers' data removed from the repository copy of the notebook

- type: `organiser-data-exclusion`
- where: original/flag-2027-nb-192-fvbx3.redacted.ipynb, setup cell, tuple ("train/gender.csv", <base64>) (byte offset 5074 of the exact file)
- old: a 228-character gzip + base64 literal = the organisers' meta_file_train_set.csv (sha256 fd7991a3...)
- new: `""` (empty string): the cell writes an empty rfs/train/gender.csv
- why: user rule 6: no organisers' data in the repository. The literal is exactly gzip.compress(csv, compresslevel=9, mtime=0) in base64, so `python stages/60_fvbind/restore_exact.py --csv <meta_file_train_set.csv> --out <dir>` rebuilds the byte-exact notebook (md5 checked)
- effect on numbers: none: only assemble.py (RFS_MODE=B, stage t163b) reads train/gender.csv; this stage never runs mode B

## 2. ImageBind code pinned to a commit

- type: `pin`
- where: setup cell (notebook cell 1 = stage.ipy cell 0; RFS_MODE=L branch, not executed here) and extraction cell (notebook cell 4 = stage.ipy cell 3; executed); 2 occurrences
- old: `pip -q install --no-deps git+https://github.com/facebookresearch/ImageBind`
- new: `pip -q install --no-deps git+https://github.com/facebookresearch/ImageBind@53680b02d7e37b19b124fa37bae4b6c98c38f5be`
- why: the run installed GitHub HEAD without a commit; 53680b02d7e37b19b124fa37bae4b6c98c38f5be is main on 2026-11-05 (release/docker/Dockerfile:22-23). NOT VERIFIED that HEAD was unchanged on 14-15/11; the model weights are fixed by the sha256 check of imagebind_huge.pth; the wiring check of fvb_extract.py (32 v4 rows vs fvbB2_emb.npz, cos >= 0.999) still guards the result
- effect on numbers: none expected

## 3. working-folder copies declared scratch (stage.json only)

- type: `wiring`
- where: stage.json `scratch: ["work", "imagebind_huge.pth"]` (no code change)
- old: everything left in the working folder became the kernel output, including work/lora_C1.pt, work/v4_extra.npz (copies of the LoRA dataset) and imagebind_huge.pth (4.5 GB download)
- new: the runner does not collect these three items
- why: nothing downstream reads them (t163p_fvb reads the bundle part of fvbx_out/ and finds the LoRA dataset copy of lora_C1.pt / v4_extra.npz first anyway, datasets/ sorting before notebooks/; e163i_fvb reads only fvbx_out/eval/); dropping them saves 4.5 GB per run and keeps the fake lora_C1.pt of the smoke branch out of the collected output, where the variant file check (every lora_C1.pt must be the original C1) would refuse it
- effect on numbers: none

## 4. RFS_XDIRS set to the literal input root (defence in depth; integration fixer r1, 18/11/2026)

- type: `path`
- where: stage.ipy cell 0, after line 3 (`import os, sys, json, gzip, base64, subprocess, shutil, time`): `os.environ["RFS_XDIRS"] = "/kaggle/input"`; stage.json patch 1; stage.ipy sha256 7b509174... (81 894 bytes; before f480a31a...)
- old: RFS_XDIRS unset (the embedded `_xload` would scan the literal /kaggle/input)
- new: RFS_XDIRS = the stage text's /kaggle/input, which the runner rewrites to the stage's input view (same line as WP-F's t163p_c5rl patch)
- why: INTEGRATION_r1 M1. `_xload` (embedded rfs_train.py:89) is reached only through `_xpart` / `load_extra`, which fvb_extract.py never calls, so the line is inert here; it keeps the three FVBIND stages and the WP-F stages on one rule. On Kaggle, as is, the value equals the original default
- effect on numbers: none

## 5. runner requirement declared: literal /kaggle/input not needed (stage.json only; integration fixer r1, 18/11/2026)

- type: `path`
- where: stage.json `runner_requirements = {"literal_kaggle_input": false, "why": ...}`
- old: no declaration; the runner's payload scan found the literal root of `rfs_train.py _xload` and treated the stage as literal (refused on Windows, separate Kaggle session needed)
- new: explicit `false` with the reason; runner/ffcore.py `is_literal` now honours an explicit, reasoned `false` (`FF_LITERAL_HONOUR_FALSE=0` restores detection)
- why: INTEGRATION_r1 M1. The only literal use (embedded rfs_train.py:89) is unreachable in this stage, see the previous change.
- effect on numbers: none

## 6. original LoRA C1 input resolved as saved only (stage.json only; integration fixer r1, 18/11/2026)

- type: `variant`
- where: stage.json input `datasets/onnhqunhh/flag2027-rfs-lora-p`: `"resolve": "saved_only"` (no code change)
- old: the input followed configs/data_map.json `default_saved`; with `FF_FORCE=t163l_orig` the runner preferred the C1 retrained in the same session, so RFX' (this chain) could get a different C1 than RFX (WP-F stages use `saved_only`)
- new: always the saved ORIGINAL C1 (`FF_DATA/03_checkpoints/rfs_lora_c1_original`, `lora_C1.pt` b97372e3...), exactly as t163p_c5rl, t213_p*, t216_p3
- why: INTEGRATION_r1 M2 (one C1 for both Bangla files of 13.54)
- effect on numbers: none for the 13.54 chain (the original run attached this very dataset)

## 7. raw-v4: unpack train_set.zip in the input view when no extracted view is mounted (19/11/2026 (integration fixer r2))

- type: `path`
- where: stage.ipy cell 1, after `def find_all(name)`; stage.json patch 3; raw-v4 input gets `"exclude": ["dev_set.zip", "dev_set"]`
- old: the os.walk for `train_English.txt` next to `faces/` assumed the extracted Kaggle view; with FF_DATA (zips as delivered) modes (a) and (b) stopped at `assert v4, "thieu raw-v4 ..."` (hidden in smoke, the walk is under `if not SMOKE`)
- new: when `train_English.txt` is absent and one `train_set.zip` is present, unpack it beside itself in the input view (dev_set.zip excluded, never opened); nothing happens when an extracted view is mounted
- why: INTEGRATION_r2 B1; the same block as WP-F's t163b / t163l_orig
- effect on numbers: none

## Not changed (deliberately)

- No --env (as the run). FVB_ONLY=eval is available in fvb_extract.py but was not used by the run.
- Path discovery kept verbatim: the FLAG_47rs index is found by '47rs' in the path and the Evaluation root by a folder named flag2027-evalphase-p; the runner's mount ids keep both true.
- The 'best effort' eval part (a failure is only logged) kept; stage.json verify requires the four eval files (strict: a missing file is fatal).
- No RFS_XDIRS line (unlike t189_b2 / t163p_fvb): fvb_extract.py imports fvbind.py but never calls `FV.main()` or `load_extra`, so the embedded `_xload` is not reached. (Superseded 18/11 by the defence-in-depth RFS_XDIRS line above; the analysis still holds: the line is inert in this stage.)
