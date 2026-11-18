# 40_R_family: pair-contrastive family RFX, original LoRA C1, sources of the DP (Different Profile) scale constants

Group of the reproduction tree for the displayed Evaluation submission **13.54** (team FourFusion, CodaBench submission 1054704 =
`xjf_r10_trsig.zip`, sha256 `b4354eaf988b2b4ea8c4e1c3fa44d7bcf631ea5cb3cd635ab2965021fe82e507`). Built on 18/11/2026 by work package WP-F
(`repo_map/DESIGN.md` section 4.7, section 6) and revised the same day after review (fixer pass: section 'LoRA C1 policy' and the `D.`
sections of every `CHANGES.md`). Nothing in this folder was executed: it is the exact code of the original Kaggle runs plus the documented
changes. "DP" = the English branch DP (Different Profile); its internal name XJF appears only in file names and paths (mapping:
`docs/NAMING.md`).

## What the group produces

| stage | original Kaggle run | exactness | kind / role | machine, time of the run | output used downstream | skip with (saved, `FF_DATA/`) | patches |
|---|---|---|---|---|---|---|---|
| `t163b` | ngnnguynchn/flag-2027-nb-163-rfs-b | exact-package | data / score | CPU, 4 min | `datasets/ngnnguynchn/flag2027-rfs-bundle-p` | 02_intermediate/rfs_bundle | 5 |
| `t163l_orig` | onnhqunhh/flag-2027-nb-163-rfs-l | exact-package | train / score | GPU T4 x2, 66 min; **force_only** | `datasets/onnhqunhh/flag2027-rfs-lora-p` (ORIGINAL LoRA C1) | 03_checkpoints/rfs_lora_c1_original | 9 |
| `t163p_c5rl` | ngnnguynchn/flag-2027-nb-163-rfsfxc5rl-p | exact-package | train / score | GPU T4 x2 (1 used), 9 min | `datasets/ngnnguynchn/flag2027-rfsfxc5rl-art-p` (RFX c5rl) | 03_checkpoints/rfx_c5rl | 5 |
| `t183_p1` | phantrnhuyntrn/flag-2027-nb-183-factor-p1 | exact-package | train / evidence (optional) | GPU T4 x2 (1 used), 43 min | kernel output (tables `rfx*lab_fold*`) | 04_run_outputs/flag-2027-nb-183-factor-p1 | 4 |
| `t183_p2` | phantrnhuyntrn/flag-2027-nb-183-factor-p2 | exact-package | train / evidence (optional) | GPU T4 x2 (1 used), 50 min | kernel output | 04_run_outputs/flag-2027-nb-183-factor-p2 | 4 |
| `t184_report` | phantrnhuyntrn/flag-2027-nb-184-factor-report | exact-package | train / evidence (optional) | CPU, 5 min | `e2e_out/factor_report.txt` (c5rl chosen on validation) | 04_run_outputs/flag-2027-nb-184-factor-report | 0 |
| `t213_p1` | phantrnhuyntrn/flag-2027-nb-213-seedrep-p1 | exact-staging | train / score | GPU T4 x2 (1 used), not logged | tables `rfrc0rlab_fold1..5` (c5rl, seeds 4-6) | 04_run_outputs/flag-2027-nb-213-seedrep-p1 | 4 |
| `t213_p2` | phantrnhuyntrn/flag-2027-nb-213-seedrep-p2 | exact-staging | train / score | GPU T4 x2 (1 used), not logged | tables `rfrc0slab_fold1..5` (seeds 7-9) | 04_run_outputs/flag-2027-nb-213-seedrep-p2 | 4 |
| `t216_p3` | phantrnhuyntrn/flag-2027-nb-216-seedrep3 | exact-staging | train / score | GPU T4 x2 (1 used), not logged | tables `rfrc0tlab_fold1..5` (seeds 10-12) | 04_run_outputs/flag-2027-nb-216-seedrep3 | 4 |
| `t219_dumpv4` | phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4 | exact-staging | train / score | CPU, 2 min | `sp/out/safe04/xjfdump_v4.parquet` (sha256 `9f0a5af4...`) | 03_checkpoints/sigma_source | 1 |

Order: `t163b` -> `t163l_orig` (saved file, see 'LoRA C1 policy') -> `t163p_c5rl` (-> `e163i` in 50_eval); `t163b` + `t163l_orig` -> `t213_p1`, `t213_p2`, `t216_p3` -> `t219_dumpv4`
(-> `dp_sigma` in 70_dp, sigma = 5.0983 / 3.8376); `t163b` + `t163l_orig` -> `t183_p1`, `t183_p2` -> `t184_report` (evidence only).
`t163b` and `onnhqunhh/flag2027-rfs-lora-p` are also inputs of the FVBIND stages (60_fvbind).

Role in the four files of the zip: RFX c5rl gives the R logit of `no_gender/...Bangla_unheard.txt` and of the two English files (through
FACTOR_ENFUSE, then the DP blend); sigma (from `t213_*`, `t216_p3`, `t219_dumpv4`) divides the English base score in the DP blend.

## LoRA C1 policy (all run modes; fixer 18/11)

The 13.54 variant is defined by one file: the ORIGINAL `lora_C1.pt` (sha256 `b97372e3504a131d...`, `datasets/onnhqunhh/flag2027-rfs-lora-p`,
`FF_DATA/03_checkpoints/rfs_lora_c1_original/`). A GPU re-train of the same recipe is not byte-identical, so it could never pass the 13.54
guard (`configs/variant_1354.json` file_checks, DESIGN 2.7). Rules:

1. The six consumers (`t163p_c5rl`, `t183_p1`, `t183_p2`, `t213_p1`, `t213_p2`, `t216_p3`) mount the C1 with `"resolve": "saved_only"`: in
   every mode (a) / (b) / (c) and in smoke runs they take the saved file (or the attached Kaggle dataset), never a re-train; when it is
   missing the runner reports `saved copy required (resolve: saved_only): put it at FF_DATA/03_checkpoints/rfs_lora_c1_original`.
   `variant.require_files` checks `lora_C1.pt` = `b97372e3...` on that input before the stage runs, and the `verify` entry (check `exact`)
   checks the byte copy the code makes (`rfs_art/lora_C1.pt` or `work/lora_C1.pt`); both hold in every mode. Mode (a) therefore rebuilds
   everything from raw data except this one variant-defining file (`docs/VERIFICATION.md` should say so).
2. `t163l_orig` is `force_only`: it runs only with `FF_FORCE=t163l_orig`, as a provenance check of the recipe (cache-key bug kept). A forced
   re-train is accepted on its post_checks: `ff_lora_drycheck.json` (210 clips / 70 persons / all `mav:v1`, before training) and
   `ff_lora_meta.json` (48 params, 1 200 steps, batch 64, 210 clips, 70 persons, 0 skipped, read from the new file; the values of the saved
   file were read from its pickle on 18/11). Its output is not substituted into the 13.54 chain: the consumers keep `resolve: saved_only`.
   Open for WP-A: `configs/variant_1354.json` file_checks reject the re-trained export before the post_checks run (exemption requested in
   `t163l_orig/stage.json -> variant.outputs_file_checks`).
3. post_checks are JSON checks only (the runner reads every post_check file as JSON); the hash checks live in `verify` and
   `variant.require_files`. `_wpf_tools.py check` enforces rules 1-3.
4. In smoke runs (`FLAG_SMOKE=1`) the original code wrote a stand-in `lora_C1.pt` that modes P / LAB copy into the output; a `variant` patch
   now copies the mounted original instead (the file is only copied in these modes, never loaded), so the variant guard also passes in smoke.

Sigma does not depend on C1 numerically (`t219_dumpv4` uses only the `t3` / `nf` / `nv` streams of the three replicates), but the replicate
runs mounted the original C1 and keep the same guard.

## How the decoded code finds its inputs (`runner_requirements`)

Every stage states `runner_requirements.literal_kaggle_input` with a reason. P / LAB stages: the only literal `/kaggle/input` read of the
decoded `rfs_train.py` is `_xload` (extra blocks `f_c5s`, `v_rl`, ...); a `path` patch sets `RFS_XDIRS` to `/kaggle/input` in the stage text,
which the runner rewrites to the stage's own input view, so they run on any OS (`false`). `t163b`, `t163l_orig`: every path is an argument
(`false`). `t184_report`, `t219_dumpv4`: the lab modules decoded from `CODE_B64` scan `/kaggle/input` in about ten places (LOAD cell
`lid_strict.json` fallback, CLEANTUNE-01 `_find`, T3ENC-01, ...): `true`, as WP-G's `e112f1sx`; on Linux the runner points `/kaggle/input` at
the stage's input view, on Kaggle run them in a session with exactly their inputs attached.

## Raw-v4 mount of `t163b` / `t163l_orig`

The original runs saw `ngnnguynchn/flag2027-raw-v4-p` extracted by Kaggle (`train_set/train_set/train_English.txt` next to `faces/`,
`voices/`; `train_set/features/*/*.csv`). `FF_DATA/00_raw/btc_mavceleb_v4/` keeps the organisers' files packed. A `path` patch (cell
'inputs') unpacks `train_set.zip` only, beside itself in the stage's own input view, when `train_English.txt` is absent; `dev_set.zip`
stays packed and is never opened. With an extracted train-only view mounted (Kaggle, or a future WP-B export) the block does nothing.

## Files of a stage

```
<stage>/stage.json        metadata (DESIGN 2.2): origin, inputs (artifact ids of the original run), outputs, patches, verify, post_checks, todo
<stage>/stage.ipy         code run by the runner: original code cells + patches, separated by "\n# <<<FF-CELL>>>\n"
<stage>/CHANGES.md        every change versus the exact notebook, with reason and effect on numbers
<stage>/original/         *.redacted.ipynb (rebuild source), the run log (E: package), kernel-metadata.json (push staging), OUTPUTS.sha256,
                          SOURCE.json (where each file comes from, md5 / sha256, kapi_runs line), decoded/ (embedded code, byte-exact)
```

`stage.ipy` = every non-empty code cell of `origin.original_notebook`, CRLF -> LF, the `patches` applied in order (each must match exactly
`count` times), joined by `"\n# <<<FF-CELL>>>\n"`. This is the rule of `release/flag2027_e2e/build_notebook.py`; with that package's two
patches it reproduces `release/flag2027_e2e/code/t163b.ipy` byte for byte (checked 18/11).

## Why the notebooks in `original/` are redacted copies

The exact notebooks cannot be published as they are:

1. Every FLAG_163-family notebook (B, L, P and the RFS-LAB runs 183, 213, 216) embeds, as `train/gender.csv`, a gzip + base64 copy of the
   organisers' file `meta_file_train_set.csv` (560 bytes, sha256 `fd7991a3353b9f90...`). The literal is replaced by the placeholder
   `FF-REDACTED:organisers-file:meta_file_train_set.csv:...`; two `redaction` patches remove the entry and, in `RFS_MODE=B` only (the
   only reader, `assemble.py --gender_csv`), copy the organisers' file from the raw-v4 mount after checking its sha256.
2. The two FLAG_174-family report notebooks (184, 219) carry the dev-phase score zip `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip`
   inside their code zip `CODE_B64`. The member is removed without recompressing the other members (`dev-exclusion`); no code path of
   these two stages reads it.

Only string literals change; every other byte of the notebook is the same. Each redacted file is re-created exactly from its exact copy by

```
python stages/40_R_family/_wpf_tools.py redact <stage> <exact notebook>   # md5 of the exact copy must equal SOURCE.json -> derived_from.md5
```

(checked 18/11 for all 14 exact copies: E: package notebooks 14-19 and the push staging copies of 183 p1/p2, 184, 163-rfsfxc5rl-p, 213
p1/p2, 216, 219; the staging md5 equal `kaggle/kapi_runs.jsonl`). The exact copies stay outside the published repo: the E: package
(`E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/`) and `repo/_staging_backup/` (184 and 219 in its git-ignored
`_PRIVATE_NOT_FOR_PUBLICATION/`). `_staging_backup/` as a whole must not be published either: its FLAG_163-family notebooks also embed the
organisers' file.

## Self-check

```
python stages/40_R_family/_wpf_tools.py check
```

rebuilds every `stage.ipy` and compares it with the file and with `stage.json -> stage_ipy_sha256`, compiles every cell, checks the sha256 of
every file listed in `SOURCE.json` / `DECODED.json`, checks that no notebook copy still holds the organisers' file or the dev zip, applies the
LoRA C1 variant guard (forbidden inputs; `resolve: saved_only` + `variant.require_files` on the C1 input; `t163l_orig` force_only; JSON-only
post_checks; `runner_requirements` stated) and scans every file for key-like strings. Result on 18/11 after the fixer pass: `ALL OK`.

## Code embedded in the runs (`original/decoded/`)

| file | sha256 (first 16) | bytes | used by | note |
|---|---|---|---|---|
| `train/rfs_train.py` | `98aa07ae529c0afc` | 31 635 (489 lines, CRLF) | `t163b`, `t163l_orig` | L run: `lora_data()` line 406, VoxCeleb2 cache-key bug kept |
| `train/rfs_train.py` | `2e700a31c9171657` | 50 690 (743 lines) | `t183_p1`, `t183_p2` | FACTOR-01 lab |
| `train/rfs_train.py` | `ab65b842f48a0179` | 56 041 (816 lines) | `t163p_c5rl` | P run: split 375, oof 474, proj 500, fuse 612, release 636, selfcheck 659 |
| `train/rfs_train.py` | `e31a25e18e5a6b3d` | 69 269 (1 002 lines) | `t213_p1`, `t213_p2` | lab; `RFS_LORA_FIX` opt-in, off, not used in LAB mode |
| `train/rfs_train.py` | `e3e9440277f00da0` | 69 409 (1 003 lines) | `t216_p3` | = 213 version + one line (`self.last_streams`, FUSEARCH-03) |
| `train/assemble.py` | `b46dd180c21e1583` | 7 027 | B run and all later runs | = `Experiment/RFS-01/train/assemble.py`; the L run embeds the older `432bdbf943e0530d` (differs only in the v4 CSV columns) |
| `ref/flag_final_core_092.py` | `25abe24db6fbe863` | 69 260 | every run | = `Experiment/RFS-01/ref/flag_final_core_092.py` |
| `ref/manifest_final.json` | `7c11399a63514f40` | 3 628 | every run | manifest of the RF-092 reference artifact |
| 61 lab files (`Experiment/...`, `kaggle/build_l0_notebooks.py`, `lab_entry.py`) | see `t219_dumpv4/original/decoded/DECODED.json` | | `t219_dumpv4` | md5-identical to the run's own `lab/` folder in `Experiment/XJF-01/out/dumpv4/lab/` |

`Experiment/RFS-01/train/rfs_train.py` of the repository (sha256 `2f79b0045db8bdc5...`) is none of these and defaults to the fixed LoRA
(`RFS_LORA_FIX=1`, line 959): never use it to replay 13.54. See `C1_VARIANTS.md`.

## Inputs from other groups

- 10_raw / 12_rows: `ngnnguynchn/flag2027-raw-v4-p` (t163b, t163l_orig; must contain the organisers' `meta_file_train_set.csv` for t163b),
  `chnnhnnguyn/external-data-mavceleb` (t163l_orig; two parent folders `mavceleb_v1_complete-001/v1`, `mavceleb_v2_complete-002/v2`),
  `ngnnguynchn/flag2027-ext-meta-p` (t163b).
- 13_lid: `ngnnguynchn/flag2027-lid-strict-p` (every stage except t163b).
- 11_collect: `duongphihoang/flag2027-audio-cache-p` (t163l_orig; mounted, 0 clips used because of the key bug).
- 20_features: `feats-rfs-{a,b,c}`, `feats-whv3-{a,b,c}` (t163b); `feats-screen2`, `aux-voice-feats` (block `v_rl`), `lab-data`, `lab-vb`,
  `feats-vox2b`, `feats-tap01-voice`, `eda-bundle` (lab / report stages, some attached but not read).
- 30_F_family: `quynhnhu82/flag2027-feats-ls-c5s-p` (block `f_c5s`), `ngnnguynchn/flag2027-ct02s-r2ca-tables-p` (SAFE-04 pair tables of the
  lab stages), `phanvnhtrng/flag2027-feats-ls-b3s-p`, the frozen `ngnnguynchn/flag2027-e2e-dev-p` and the wiring datasets `cc5-oof`,
  `dsel02-oof`, `facelora03-tables`, `lab-avfull`, `feats-scout-wsvrdn`, `trngsnam/flag2027-feats-peface01-pe0-p` (the last one only for
  t183_p1 / t183_p2 since 18/11: dropped from t163p_c5rl, t213_p*, t216_p3 as never read, integration fixer r1).
- `ngnnguynchn/flag2027-t3s01-tables-p` (T3S-01, FLAG_167) is **not** wiring for sigma: its four job tables give `ct01r2_fold<s>`
  (row keys + `trCA`), from which `t219_dumpv4` builds the F1s' out-of-fold score inside U. Since 18/11 (integration fixer r1) t219
  mounts `datasets/ff/flag2027-t3s01-trca-p` instead, made by the new stage `t3s01_trca` from `ct02s-r2ca-tables-p` (keys + trCA, identical
  to the frozen copy by WP-F's light check; the unused `trT3S_*` columns are absent), at the original mount path. The frozen dataset is
  kept only for `t184_report` (evidence, `resolve: saved_only`).

## Licences and privacy of what this group mounts and exports

| artifact | stage | content | sharing |
|---|---|---|---|
| `ngnnguynchn/flag2027-rfs-bundle-p` | `t163b` (export) | `ext.npz` holds VoxBlink rows (features of VoxBlink faces / voices) | VoxBlink CC BY-NC-SA: private only (FF_DATA shared with the organisers), never public |
| `onnhqunhh/flag2027-rfs-lora-p` | `t163l_orig` (export) | LoRA adapter on ImageBind-huge + `v_lora` of the v4 train voices | ImageBind weights CC BY-NC 4.0 (the adapter is a derivative); `v_lora` derives from organisers' data: private |
| `ngnnguynchn/flag2027-rfsfxc5rl-art-p` | `t163p_c5rl` (export) | projector trained on features incl. VoxBlink rows, + a copy of the C1 | VoxBlink CC BY-NC-SA and ImageBind CC BY-NC 4.0: private only |
| `duongphihoang/flag2027-audio-cache-p` | `t163l_orig` (input) | 3 s audio of VoxCeleb2 / VoxBlink | VoxBlink media: private only |
| kernel outputs of 183 / 213 / 216 / 184 / 219 | (outputs) | SAFE-04 pair scores of train persons (derived from the above) | private (FF_DATA) |

`configs/data_map.json` carries the matching `private` / `voxblink` flags; licences in `LICENSES.md` (WP-K). Nothing listed here is in the repo.

## Open items

- `t163l_orig`: the dry check and the folder sort have never run (no compute in this workflow); the first run must print
  `FF lora dry check: OK`. The ImageBind code is pip-installed from GitHub HEAD as in the run; the Docker pin `53680b02...` is not proven
  equal to HEAD of 2026-11-03.
- `t213_p1`, `t213_p2`, `t216_p3`: run outputs survive only as the copies gathered by `t219_dumpv4`
  (`Experiment/XJF-01/out/dumpv4/sp/out/safe04/rfr{c0r,c0s,c0t}lab_fold*.parquet`); lay them out under `FF_DATA/04_run_outputs/<slug>/rfs_art/`
  or download the kernel outputs if Kaggle still has them. Run times were not logged.
- `t184_report`, `t219_dumpv4`: exact notebooks only in private storage (dev zip inside).
- Private storage of every exact original (staging copies of 163p, 183 p1 / p2, 184, 213 p1 / p2, 216, 219; the E: package already holds
  notebooks 14-19): copy to `FF_DATA/_private_originals/<stage id>/` (or `$FF_PRIVATE/<stage id>/`; `runner/check_stage.py` looks there),
  record each md5, point `original/SOURCE.json -> derived_from` at it. Until then the only exact copies of 213 / 216 are in
  `repo/_staging_backup/` (and volatile Temp).
- `repo/_staging_backup/` holds organisers' data (the `train/gender.csv` blob in the FLAG_163-family notebooks) and has no root ignore
  rule (only its `_PRIVATE_NOT_FOR_PUBLICATION/` is ignored): WP-A / WP-K must add `/_staging_backup/` to a root `.gitignore` or move
  the folder to private storage before `repo/` becomes the main branch (outside WP-F's write scope).
- WP-A: exempt the outputs of a forced `t163l_orig` from the variant file_checks; optionally `default_saved: true` on the C1 dataset
  (needed by WP-H's C1 consumers in mode (a) unless they also use `resolve: saved_only`).
- WP-A / WP-E: producer + class for `t3s01-tables-p` (above); `dev_data: true` on `e2e-dev-p`; exclude `lab/` (dev zip) from the saved
  `sigma_source` (`t219_dumpv4/stage.json` notes).
- User decision before publication: the identity-linkage constant `OVERLAP` = {`v1_complete:shaukat_tarin`, `v2_complete:id0073`} in
  the embedded lab code of `t184_report` / `t219_dumpv4` (`kaggle/build_l0_notebooks.py` lines 92, 1309; DESIGN G17; WP-B
  `12_rows/r07_overlap`; `docs/COMPLIANCE_NOTES.md`).
- New code of the fixer pass never executed (no compute in this workflow): RFS_XDIRS and smoke-C1 patches, the raw-v4 unpack, the
  ImageBind sha256 check and `ff_lora_meta.json` of `t163l_orig`.
- `t219_dumpv4`, `t184_report` mount the frozen `e2e-dev` dataset, which also holds dev-phase components whose presence the original code
  asserts (wiring, DESIGN 2.8 / G18).
