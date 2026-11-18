# CHANGES: `t213_p1` (phantrnhuyntrn/flag-2027-nb-213-seedrep-p1)

Source of `stage.ipy`: `original/flag-2027-nb-213-seedrep-p1.redacted.ipynb`, a redacted copy of the exact notebook `NOP_BTC_2026-11-20/repo/_staging_backup/kapi_flag-2027-nb-213-seedrep-p1_61811_n4/flag-2027-nb-213-seedrep-p1.ipynb` (md5 `f8528ee4010fbffc97af216e5ea101ef`, sha256 `a2a25534f290585c51b748a6d4f07316e09c280cd5f60f4e62e004f026293cfd`; exactness `exact-staging`; kapi_runs line 1691, pushed md5 `f8528ee4010fbffc97af216e5ea101ef`).
Rebuild rule: every non-empty code cell, CRLF -> LF, the patches below in this order (each must match exactly `count` times), cells joined by `\n# <<<FF-CELL>>>\n`; result sha256 `8ef6ff4e4e3c595ccc28a8817e0324c05ea7fd3b725655cb5d18590a8c0a3e27` (6 cells). Same rule as `release/flag2027_e2e/build_notebook.py` (checked: it reproduces `release/flag2027_e2e/code/t163b.ipy` byte for byte when given that package's two patches).

## A. Changes made when copying the exact notebook into `original/` (redaction)

1. **notebook cell 2 (0-based, markdown counted), `("train/gender.csv", ...)` literal** - type `redaction` (organisers' data).
   - old: gzip + base64 literal, 228 characters (sha256 of the literal `5af7e8b8d87b0d1d4aa4f59c495150e2f305b45236466f8f31930678578e372c`), decoding to 560 bytes with sha256 `fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030` = the organisers' `meta_file_train_set.csv` (70 lines `idNNN,f|m`).
   - new: the placeholder string `FF-REDACTED:organisers-file:meta_file_train_set.csv:560-bytes:sha256=fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030`.
   - why: the repo must not carry organisers' data (user rule 6); every FLAG_163-family notebook embeds this file as `train/gender.csv`.
   - effect on numbers: none (patches 'redaction' below remove the entry and, in `RFS_MODE=B` only, copy the same bytes from the organisers' file with a sha256 check; no other mode reads the file).

## B. Patches (`stage.json` -> `patches`, applied in this order)

2. **notebook cell 2 (0-based index in the notebook, markdown cells counted), line 8 of that cell** - type `redaction`.
   - old (1 hit): `("train/gender.csv", "FF-REDACTED:organisers-file:meta_file_train_set.csv:560-bytes:sha256=fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030"), `
   - new: ``
   - why: the embedded copy of the organisers' meta_file_train_set.csv is not distributed with the repo (user rule 6); its entry is removed from the embedded-file loop (the literal was already replaced by a placeholder in original/*.redacted.ipynb, see SOURCE.json).
   - effect on numbers: none.

3. **notebook cell 2 (0-based index in the notebook, markdown cells counted), line 9 of that cell** - type `redaction`.
   - old (1 hit): `    p = CODE / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(gzip.decompress(base64.b64decode(blob)))\n`
   - new: `    p = CODE / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(gzip.decompress(base64.b64decode(blob)))\nif MODE == "B":   # FF redaction (CHANGES.md): train/gender.csv was an embedded copy of the organisers' meta_file_train_set.csv; the same bytes come from the raw-v4 input (only assemble.py, RFS_MODE=B, reads the file)\n    _ff_g = [q for q in sorted(Path("/kaggle/input").rglob("... (843 chars)`
   - why: puts back train/gender.csv in RFS_MODE=B from the organisers' file itself (sha256 asserted = the embedded copy), so assemble.py reads the same 560 bytes; in modes L / P / LAB no code reads train/gender.csv (checked in every embedded rfs_train.py version), so nothing is written.
   - effect on numbers: none.

4. **notebook cell 2 (0-based index in the notebook, markdown cells counted), line 3 of that cell** - type `path`.
   - old (1 hit): `import os, sys, json, gzip, base64, subprocess, shutil, time\n`
   - new: `import os, sys, json, gzip, base64, subprocess, shutil, time\nos.environ["RFS_XDIRS"] = "/kaggle/input"   # FF path (CHANGES.md): root that rfs_train.py _xload scans for the extra blocks of RFS_CFG (e.g. f_c5s, v_rl); the runner rewrites this literal to the stage's own input view (original run: RFS_XDIRS unset -> the same /kaggle/input)\n`
   - why: the decoded rfs_train.py looks up the extra feature blocks (RFS_CFG 'add', e.g. f_c5s / v_rl) with _xload, whose root is RFS_XDIRS or, when unset, the literal /kaggle/input (line 83 of the P / FACTOR-lab versions, line 96 of the 213 / 216 versions). Code decoded at run time is out of reach of the runner's /kaggle/input rewrite, so on Linux / Docker the lookup found nothing and on a Kaggle orchestration session it scanned every attachment of the session. Setting RFS_XDIRS in the stage text (which the runner rewrites to the stage's input view; the subprocess inherits os.environ) points it at exactly the stage's inputs on every machine; run as is on Kaggle it is the original default.
   - effect on numbers: none (the same root: the datasets the run attached = the stage's inputs).
   - added: 18/11/2026 (fixer, review of WP-F).

5. **notebook cell 3 (0-based index in the notebook, markdown cells counted), line 22 of that cell** - type `variant`.
   - old (1 hit): `    torch.save({"params": [torch.zeros(2)]}, WORK / "lora_C1.pt")          # smoke: lora gia (luot L that chay tren Kaggle GPU)\n`
   - new: `    _ff_c1 = find_all("lora_C1.pt")   # FF variant guard (CHANGES.md): in smoke mode too, use the mounted ORIGINAL LoRA C1 (modes P / LAB only copy this file), so no stand-in lora_C1.pt reaches the artifact\n    if _ff_c1: shutil.copy(_ff_c1[0], WORK / "lora_C1.pt")\n    else: torch.save({"params": [torch.zeros(2)]}, WORK / "lora_C1.pt")          # smoke: lora gia (luot L that chay tren Kaggle GPU... (403 chars)`
   - why: with FLAG_SMOKE=1 the original code writes a stand-in lora_C1.pt ({'params': [zeros(2)]}) that modes P / LAB copy into the output (rfs_art/ or work/); the runner's variant check (configs/variant_1354.json file_checks, every lora_C1.pt of every output = b97372e3...) then stopped every smoke run with VariantError. In modes P / LAB the file is only copied (stage_release: shutil.copy + sha256 in the manifest), never loaded, so copying the mounted original C1 (always present: input resolve saved_only + variant.require_files) changes nothing else in smoke.
   - effect on numbers: none (FLAG_SMOKE=1 path only; a real run never reaches this branch).
   - added: 18/11/2026 (fixer, review of WP-F).

## C. Kept as in the original run (on purpose)

- Inputs: exactly the datasets / kernels the run attached (`stage.json` -> `inputs`, from the run's kernel metadata), including the ones attached but not read.
- Every other line of every cell, the embedded code (`original/decoded/`, byte-exact, CRLF kept) and the `[boot]` / `[cell]` lines that kapi inserted at push time.
- pip calls (`pip_policy: original`).
- The ORIGINAL LoRA C1 (`onnhqunhh/flag2027-rfs-lora-p`, `lora_C1.pt` sha256 `b97372e3...`); the substitutions to the fixed C1 made for the 16.18 system (`release/flag2027_e2e/stages.py:234-320, 526-527`) and the `lorafix1` defaults of today's scripts are NOT applied (see `../C1_VARIANTS.md`).

## D. Changes in `stage.json` outside the code (fixer, 18/11/2026, review of WP-F)

- `runner_requirements.literal_kaggle_input` = `false` (new key; why: the only literal /kaggle/input read in the decoded code is rfs_train.py _xload (line 83 / 96), redirected to the stage's input view by the RFS_XDIRS path patch of the setup cell; every other path is in the stage text (rewritten by the runner) or passed as an argument (grep of original/decoded/: flag_final_core_092.py mentions /kaggle/input in a docstring only)).
- `post_checks`: removed `{"file": "work/lora_C1.pt", "sha256": "b97372e3..."}`. The runner reads every post_check as a JSON file (runner/ffcore.py verify_stage), so this entry failed fatally on the binary torch file of every real run. The same hash stays in `verify` (check `exact`) and is now also enforced on the input (next items).
- `inputs` -> `datasets/onnhqunhh/flag2027-rfs-lora-p`: `"resolve": "saved_only"` (runner: only a saved / attached copy counts, whatever FF_MODE / FF_USE_SAVED say; the producer t163l_orig is not scheduled for it). `variant.require_files`: `lora_C1.pt` sha256 `b97372e3...` on that input (runner build_vin, before the stage runs, smoke included). Reason: in mode (a) a re-trained C1 can never equal b97372e3 (GPU), and 13.54 is defined by the original file (DESIGN 2.7); see ../README.md 'LoRA C1 policy'.
- `compute.gpus` 1 -> 2 (+ `gpus_note`): the runs had Kaggle T4 x2 (logs 'gpus 2', machine_shape NvidiaTeslaT4); the code uses one device, so numbers do not depend on it.
- `title`: 'DP (Different Profile)' written out at first use (user rule 5); XJF appears only in file paths, mapping in docs/NAMING.md.

## E. Input datasets/trngsnam/flag2027-feats-peface01-pe0-p dropped (stage.json only; integration fixer r1, 18/11/2026)

- type: `wiring`
- where: stage.json `inputs` (entry removed; recorded under `inputs_dropped`); no code change
- old: attached as in the original run (only indexed by the .npy scan of `_xload`)
- new: not attached
- why: INTEGRATION_r1 M7 / B3. Never read: `_xload` (decoded rfs_train.py:96) selects files by exact name for the requested blocks only (ARMS c0r / c0s = add [f_c5s, v_rl], seeds 4-9); the dataset holds only `*_peL*.npy` (block `f_pel`), `bn42_clean.json`, `facelora_state.pt`, `train_log.json`. Same drop as WP-H (t163p_fvb, t189_b2). It has no producing stage, so it was a requirement-1 gap with no use. Kept in t183_p1 / t183_p2 (evidence stages whose arms request `f_pel`).
- effect on numbers: none
