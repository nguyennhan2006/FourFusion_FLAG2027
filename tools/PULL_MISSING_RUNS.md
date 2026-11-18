# Pulling the runs and artifacts that are still missing (team only)

These commands need the Kaggle API with the team's credentials, so they are **not** run by any build script of this repository
(DESIGN 9.3, gaps G2 / G4 / G15). Run them on the team machine, check the md5 / SHA-256 given here, then put each result where the
last column says. Never copy `~/.kaggle/kaggle.json` or any token into this repository; run `python tools/secret_scan.py` on every
pulled notebook before it goes into a stage's `original/` folder.

`md5 (kapi_runs)` = `notebook_md5` of the **last** push of that kernel in the team log `kaggle/kapi_runs.jsonl` (line given): the
pulled notebook must have this md5 to count as the exact code that ran (`kapi.py` inserts its run-parameter / `[boot]` lines before
pushing, so the pulled file is the pushed file).

## 1. Exact notebooks to pull (`kaggle kernels pull -p <dir> -m <owner/slug>`)

`-m` also writes `kernel-metadata.json` (datasets, kernel inputs, GPU, `docker_image`). After pulling: `md5sum <dir>/*.ipynb`,
compare, copy the notebook + `kernel-metadata.json` into the stage's `original/`, record the md5 / SHA-256 and the kapi line in
`original/SOURCE.json`, switch `origin.exactness` from `builder-regenerated` to `exact-pulled`, and re-run
`python runner/check_stage.py <stage>`.

| stage | kernel | md5 (kapi_runs) | kapi line | why missing | destination |
|---|---|---|---|---|---|
| `f47b_screen2` | `thanhthaoph/flag-2027-nb-47b-voice-screen` | `8eeb5c674f054d4412a4b279ed291bd0` | 53 | no staging folder | `stages/20_features/f47b_screen2/original/` |
| `f63w_aux_voice` | `hthlin/flag-2027-nb-63w-aux-voice` | `5a9d3bc9c869502c1e11929ef6abddd1` | 361 | pushed with `--secret` (staging deleted) | `stages/20_features/f63w_aux_voice/original/` |
| `f103c_tap01` | `dnq1209/flag-2027-nb-103c-tap01-voice-r2` | `e192b1b76ee448bc253137e788bc9a88` | 711 | staging folder empty | `stages/20_features/f103c_tap01/original/` |
| `f103c_tap01` (first run, its output is an input) | `dnq1209/flag-2027-nb-103c-tap01-voice` | `436c24d646af2966dc3f42e6e1f30bce` | 691 | staging folder empty | `stages/20_features/f103c_tap01/original/` |
| `c63v_audio_cache` | `duongphihoang/flag-2027-nb-63v-audio-cache` | `0123c6d09eff6865fe797966243cc292` | 342 | pushed with `--secret` | `stages/11_collect/c63v_audio_cache/original/` |
| `l42_vox2_lid` | `nguynlcnhtho/flag-2027-nb-42` | (ran before the log started) | — | never logged | `stages/13_lid/l42_vox2_lid/original/` |
| `f41_t3enc_vox2` | `nguynlcnhtho/flag-2027-nb-41` (unverified: slug) | (ran before the log started) | — | executed copy in the repo, slug unconfirmed | compare with `Experiment/T3ENC-01/flag-2027-nb-41.ipynb` |
| `c53_vox2b` | `nguynlcnhtho/flag-2027-nb-53-scaleid` | `ef50745327c79dbae1c08f154aa6ca4f` | 13 | none: `kaggle/FLAG_53_scaleid.ipynb` already has this md5 | (only to confirm) |
| `c45_frames_vox2` | `onnhqunhh/flag-2027-nb-45-facelora-farl` | `a0558cc91281bb8b9c93616d5da59d06` | 30 | no staging folder | `stages/11_collect/c45_frames_vox2/original/` |
| `c63_frames` | `onnhqunhh/flag-2027-nb-63-frames` | `dd0c8f9e11cb28f3055dcc64464a99b9` | 73 | no staging folder | `stages/11_collect/c63_frames/original/` |
| `c87_frames_bridge` | `onnhqunhh/flag-2027-nb-87-bridge` | `a69638cba7e86f757b38a51e93dc109c` | 246 | no staging folder | `stages/11_collect/c87_frames_bridge/original/` |
| `t154_g1_f12` | `nguynthanhnhnn/flag-2027-nb-154-ct02-g1-f12` | `06af774b0f509e3fc540e083ee6cb08d` | 1174 | staging folders empty | `stages/30_F_family/t154_g1_f12/original/` |
| `t154_g1_f34` | `hdonqu/flag-2027-nb-154-ct02-g1-f34` | `f7ef6ccd24d5c9d2dc9ef597340b34b7` | 1176 | staging folders empty | `stages/30_F_family/t154_g1_f34/original/` |
| `t154_g1_f5` | `thanhtruc02/flag-2027-nb-154-ct02-g1-f5` | `585e01231290ef64e7780bc2dc4d2089` | 1177 | staging folders empty | `stages/30_F_family/t154_g1_f5/original/` |
| `t189_b2` (backup of the staging copy) | `thanhhuyen11/flag-2027-nb-189-fvbind-b2-ib0` | `4a5e41949fc3aff3dffea9341e902b1b` | 1580 | staging copy exists (`_staging_backup/`) | compare only |
| `t192_fvbx` (backup) | `thanhhuyen11/flag-2027-nb-192-fvbx3` | `ba933cdea97efea2d241f849dca1a27b` | 1603 | staging copy exists | compare only |
| `t163p_fvb` (backup) | `thanhhuyen11/flag-2027-nb-163-rfsfxfvb-p` | `308d8715ee7df933aaf4881d8aea0ea1` | 1605 | staging copy exists | compare only |
| `e163i_fvb` (backup) | `thanhhuyen11/flag-2027-nb-163i-rfsfxfvb-eval` | `354351113d5ed7f4350be800c4b0b79d` | 1609 | staging copy exists | compare only |

`t219_dumpv4` (`phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4`, md5 `6ad5b31c797b6323120eec0cc4dd727d`, line 1887) was deleted on
Kaggle; its exact notebook survives only in `_staging_backup/_PRIVATE_NOT_FOR_PUBLICATION/` (it embeds a dev score zip).

Shell template (one kernel):

```bash
k=thanhthaoph/flag-2027-nb-47b-voice-screen; d=pulled/${k#*/}
mkdir -p "$d" && kaggle kernels pull -p "$d" -m "$k" && md5sum "$d"/*.ipynb && python tools/secret_scan.py "$d"
```

## 2. Artifacts that exist only on Kaggle (download to the Drive folder `FF_DATA`)

Do not delete these kernels / datasets on Kaggle before the copy is checked (output is lost with the kernel).

| artifact | command | put under `FF_DATA/` | check |
|---|---|---|---|
| output of `thanhhuyen11/flag-2027-nb-192-fvbx3` (the `v_fvb` block) | `kaggle kernels output thanhhuyen11/flag-2027-nb-192-fvbx3 -p <dir>` | `04_run_outputs/flag-2027-nb-192-fvbx3/` | record SHA-256 (`python tools/drive_manifest.py make FF_DATA --only 04_run_outputs`) |
| dataset `thanhhuyen11/flag2027-rfsfxfvb-art-p` (RFX′) | `kaggle datasets download -d thanhhuyen11/flag2027-rfsfxfvb-art-p -p <dir> --unzip` | `03_checkpoints/rfx_fvb/` | its `lora_C1.pt` must have SHA-256 `b97372e3504a131d0a969e478a855868095c67db4b3c0c133ef770c00e40991c` (ORIGINAL LoRA C1) |
| outputs of `phantrnhuyntrn/flag-2027-nb-213-seedrep-p1`, `-p2`, `phantrnhuyntrn/flag-2027-nb-216-seedrep3` (if still there) | `kaggle kernels output <kernel> -p <dir>` | `04_run_outputs/<slug>/` | tables `rfrc0{r,s,t}lab_fold*.parquet` = copies in `Experiment/XJF-01/out/dumpv4/sp/out/safe04/` |
| outputs of `nguyenchonnhan/flag-2027-nb-155s-ct02ca-f12`, `-f34`, `locnguyentruongan/flag-2027-nb-155s-ct02ca-f5` (if still there) | `kaggle kernels output <kernel> -p <dir>` | `04_run_outputs/<slug>/` | `ct01r2_fold*.parquet` = members of `flag2027-ct02s-r2ca-tables-p` |
| outputs of `nguynlcnhtho/flag-2027-nb-164-cleancomp01-f123`, `dngxunqun/flag-2027-nb-164-cleancomp01-f45` (if still there) | `kaggle kernels output <kernel> -p <dir>` | `04_run_outputs/<slug>/` | `cleancomp01_fold*.parquet` |
| output of `dnq1209/flag-2027-nb-103c-tap01-voice` (input of the r2 run) | `kaggle kernels output dnq1209/flag-2027-nb-103c-tap01-voice -p <dir>` | `04_run_outputs/flag-2027-nb-103c-tap01-voice/` | — |

Everything else in `configs/data_map.json` is in the team package `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/` (`datasets/`,
`outputs/`, `pretrained/`) or on the team machine (paths in the `status` field of each entry); `python tools/drive_manifest.py
datamap <FF_DATA>` lists what is present.

## 3. Image digest of the runs (gap G15)

`kaggle kernels pull -m` writes `docker_image` and `machine_shape` into `kernel-metadata.json`. Read it for the Evaluation runs
(`thanhhuyen11/flag-2027-nb-{65xbs-b3sx,65xcs-c5sx,47rs-screen2,50cc-e2e,112f0sx,112f1sx,163i-rfsfxc5rl}-eval`) and the FVBIND chain
(`…-189-fvbind-b2-ib0`, `…-192-fvbx3`, `…-163-rfsfxfvb-p`, `…-163i-rfsfxfvb-eval`), and add the digests to `docker/README.md`.
Known so far: GPU `gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01`
(R11 runs of 5/11 and the five pulled runs 155s / 164 of early November). GPU image Python 3.13.16, CPU image Python 3.13.18
(every kept CPU log: t163b, t184, t219, e112f0sx, e112f1sx); the September / October runs used older images (torch 2.11.0+cu128 /
2.10.0+cu128): per-stage table `docker/ENV_PER_STAGE.md`. Unverified: the CPU digest of the 11–16/11 runs.

## 4. On the team machine (no Kaggle API)

- SHA-256 of the two original MAV-Celeb zips (`sha256sum`), into `configs/data_map.json` (`raw/mavceleb_v1_v2_zips`) and `configs/expected_sha256.json` (`raw`).
- Run `dp_train` once (about 12 min CPU; deterministic on this machine, S8 K6) to create the DP tower checkpoint (gap G5), check its English `z` equals `Experiment/XJF-01/out/bak_english_only/xjf_scores.npz`, upload to `FF_DATA/03_checkpoints/dp_towers/`.
- `python assembly/assemble_exact.py` (mode (c)), or `python runner/ffrun.py run --mode assemble` with `FF_DATA` set.
- Optional: extract the pip lock of the CPU image built on 7/11: `docker run --rm fourfusion:cpu cat /opt/fourfusion/docker/requirements.lock.txt > docker/requirements.lock.txt` (that image was built from the U′ package; it is the closest existing lock).
- Upload `voice_lid_probs.npy` of FLAG_42 (`Experiment/T3ENC-01/feats_vox2lid/`), `Experiment/SCALE-VB/results_flag46.zip` (FLAG_46, VoxBlink: private) and the DP feature arrays (`kaggle/output/feats_{v2,models,eval,ext_full}`) to the paths given in `configs/data_map.json`.
