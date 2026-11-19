# CHANGES: stage `f32_aclip_full`

Original code: Experiment/CAFUSE-EN/flag-2027-nb-32.ipynb (exact-executed-copy: the executed notebook of FLAG_32 run 2, 04/10/2026, kept in `original/`). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`), and applies configs/pipeline_1354.json `runtime_rewrites` (the dead ImageBind URL of the unused IBAligned class); that is not a code change.

## Which run made `datasets/ngnnguynchn/flag2027-lab-avfull-p` (certainty: high)

- The 8 npy files of the frozen dataset (package E: zip sha256 b5febc05...) are byte-identical to `Experiment/CAFUSE-EN/feats_av_full/` and to the members of `Experiment/CAFUSE-EN/results_flag32.zip`, the zip that this notebook's last cell makes (`NB_32_feats_av.zip`; member times 10:31-11:46 = the run's log times).
- The printed log of this notebook lists the same 8 files with the same shapes ((7580, 1024), (8953, 1024), (19556, 1024), (20715, 1024)); its manifest (2026-10-04 11:46) has 2 more files, the dev arrays `face/voice_aclip_no_gender_English.npy`, which the dataset does not hold.
- Not the original: `flag-2027-nb-32_run1_lfs.ipynb` (run 1, failed after 75 s on the git-lfs pointer of the AudioCLIP BPE vocabulary, EXPERIMENT_LOG 123) and `kaggle/FLAG_32_aclip_ext_full.ipynb` (regenerated 12/11/2026 by the builder with broken-audio / short-clip handling; never run for these arrays).
- The MAV-Celeb row tables of the original run were found by name (dataset not printed). `datasets/ngnnguynchn/flag2027-ext-meta-p` has the same row counts, and this run's face arrays equal FLAG_14's (stage f14_av, which reads ext-meta-p) at FLAG_14's selected rows with cosine 1.00000: same row order.
- Packaging into `labavfull/` (15/10/2026 18:06) was a separate ad-hoc step: stage `b_lab_avfull`.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 0, line 18 | `dev-exclusion` | `SPLIT_FILTER = ["no_gender/English"]   # the data cell needs one…` -> `SPLIT_FILTER = ["train"]   # FF dev-exclusion (CHANGES.md #1): …` | the original encoded the organisers' dev list only because the data cell asserts that one organiser split exists; no dev list is indexed or encoded here (DESIGN §2.8); the train list of raw-v4 takes that role, as in f12_eval_feats / f14_av | none on the 8 MAV-Celeb arrays (each split / source is encoded on its own; eval mode, no state between jobs). The 2 dev arrays are no longer written; `face_aclip_train.npy` / `voice_aclip_train.npy` (6 485 rows), `index_train.txt`, `meta_file_train_set.csv` are written instead and not packed |
| 2 | cell 0, line 19 | `dev-exclusion` | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")` -> `ZIP_PREFIXES = ("train_set", "eval", "test")   # FF dev-exclusion…` | with the zipped organiser release the data cell unpacks every zip with these prefixes, dev_set.zip included (same patch as f12_eval_feats #2) | none |
| 3 | cell 2 (data) | `smoke` | after `S = {k: v for k, v in S.items() if k in SPLIT_FILTER}` add `if os.environ.get("FF_SMOKE") == "1": … v["wav"][:8], v["jpg"][:8]` | FF_SMOKE=1 runs on 8 rows of the organiser split | none in real runs (the runner sets FF_SMOKE=0) |
| 4 | cell 2 (data) | `smoke` | before `rows.fsel, rows.vsel = fsel, vsel` add `if os.environ.get("FF_SMOKE") == "1": fsel, vsel = fsel[:8], vsel[:8]` | FF_SMOKE=1 encodes 8 face and 8 voice rows per MAV-Celeb source | none in real runs (full arange(N) selection, saved as `ext_*_rows.npy`) |

## Other differences from the original run (inputs, wiring, environment)

- **inputs**: the original attached `trngsnam/flag-input` (organisers' dev list) and an unprinted dataset holding the four `ext_*_meta.csv`; here `flag2027-raw-v4-p` (train list) and `flag2027-ext-meta-p` (made by stage r08_mav_rows) replace them. MAV-Celeb is the same artifact (`chnnhnnguyn/external-data-mavceleb`, extracted layout).
- **dev-exclusion** (input view, pending runner support): the raw-v4 input asks the runner not to mount `dev_set.zip` / `dev_set` (stage.json inputs[].exclude), as f12_eval_feats / f14_av. With raw-v4 mounted unpacked, find_splits still reads the dev list files before SPLIT_FILTER drops them; no dev clip or image is decoded or encoded.
- **environment**: the original ran on the Kaggle image of early October 2026 (Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1). Its pip cell is kept (`pip_policy: original`): onnxruntime-gpu, insightface 2.0, ImageBind from git (pinned by the runner with FF_PIP=pinned), wav2clip; only AudioCLIP is used.
- **models**: `AudioCLIP-Full-Training.pt` comes from `FLAG_CACHE` (tools/fetch_models.py `audioclip_full`, sha256 2441d35b...; group `wiring`) or from the GitHub release URL. The AudioCLIP code is cloned from GitHub (master, depth 1, unpinned, as in the original run); the BPE vocabulary is fetched from media.githubusercontent.com (fallback openai/CLIP) and checked by md5 933b7abb....
- **outputs**: the original's last cell (zip of feats_av as `NB_32_feats_av.zip`, Vietnamese comments) is kept; the zip is part of the kernel artifact.

## Notes

- Wiring only: the arrays feed `b_lab_avfull` -> `datasets/ngnnguynchn/flag2027-lab-avfull-p` -> e112f0sx (EN-T3-01 table_en on the F0 English path), which reaches f1s_eval.zip but not the 13.54 zip.
- A re-run is not bit-identical (GPU, fp16 autocast for faces; AudioCLIP voice arrays drifted between Kaggle images: FLAG_14 vs this run cosine median 0.974). The row files are deterministic (arange(N)).

## Run checks (stage.json post_checks / verify)

- post_check (fatal): `feats_av/feats_manifest.json` `external` = {"v1_complete": [7580, 19556], "v2_complete": [8953, 20715]}: every MAV-Celeb row (a skipped source is only logged by the code).
- post_check (fatal): `feats_av/feats_manifest.json` `failed` = {} (the original printed `failed: none`).
- post_check (fatal): `feats_av/feats_manifest.json` `splits/train` = 6485.
- verify (warning only): `feats_manifest.json` `files` = the 8 MAV-Celeb arrays with the original shapes + the 2 train arrays.
- verify (informational): sha256 of the 8 files vs the original run's files (= the lab-avfull members). b_lab_avfull enforces the comparison (guard policy).
