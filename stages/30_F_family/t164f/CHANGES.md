# CHANGES: t164f (exact-package; executed notebook `05_flag-2027-nb-164f-cleancomp01-final.ipynb`)

The code cells are verbatim (stage.ipy sha256 = release/flag2027_e2e/code/t164f.ipy, checked 2026-11-18).

## 1. Export without the dev streams (type `dev-exclusion`; numbers: none)

- **Where:** `stage.json` `outputs.exports` (no code change). **Old:** the whole folder `lab_out/final` (4 files, as the original dataset `ngnnguynchn/flag2027-cleancomp-final-p`). **New:** a file map with `s007_xall.pt`, `final_info.json`, `v11_head/v11_t3head.pt`.
- **Why:** `v11_head/t3_streams.npz` holds per-pair streams of the new v11 head on the four DEV lists (final.py lines 57-68, from components_dev.npz). No 13.54 consumer reads it: e50cc reuses only `v11_t3head.pt` and deletes / regenerates `t3_streams.npz` (stages/50_eval/e50cc/stage.ipy lines 126-129). The file stays in the kernel output (FF_WORK only, never in git).
- **Not changed:** the dev scoring itself (inside the embedded final.py, a module of CODE_B64; cell 05 still asserts the four files of `lab_out/final`). It runs after both models are saved, so no trained weight depends on it; it needs the bundle's dev wiring file components_dev.npz (stage.json `dev_usage`).

Runner-level only (not a change of this stage's code): the runner replaces the strings /kaggle/input and /kaggle/working by the stage's own input view and working folder, and prints a `[cell] <stage> k/n` line before each cell (DESIGN 2.4).

Runner-level requirement (2026-11-18 fix; not a change of this stage's code): `stage.json` declares `runner_requirements.literal_kaggle_input: true`. The modules decoded from `CODE_B64` at run time (listed in `runner_requirements.why`, with file and line) are written to /kaggle/working/lab and find their inputs by scanning the LITERAL path /kaggle/input, which the string rewrite above cannot reach. With the flag, the runner points /kaggle/input at this stage's own input view on Linux (symlink for the duration of the stage; `FF_LITERAL_KAGGLE=mounted` when a container already mounts it). On Kaggle the runner cannot change /kaggle/input and only warns: run this stage in a session with exactly its inputs attached (as the original run). Without the flag a Linux / Docker run would fail the asserts (lid_strict.json, adapter arrays, feats_vox2b, CT blocks), and a Kaggle orchestrator session would let the modules pick the session's own attachments (e.g. a saved original instead of an artifact rebuilt in the session).

## 2. Dev scoring only when components_dev.npz is attached (type `dev-exclusion`; numbers: none) - added 19/11/2026

- **Where:** a new cell before `final-artefacts` (`stage.json` `insert_cells`), the `need` list of `final-artefacts` (patch), and `stage.json` input `datasets/ngnnguynchn/flag2027-e2e-dev-p` (`"optional": true`).
- **Old:** final.py (embedded, decoded at run time) always scored the four dev lists after saving the models, reading `components_dev.npz` (dev-list file of the frozen e2e-dev bundle, no producing stage).
- **New:** when no `components_dev.npz` is attached (mode (a) from raw data), the new cell removes final.py lines 59-67 from the decoded copy on disk (exact-text check first) and the `need` list no longer asks for `v11_head/t3_streams.npz`. With the original inputs the cell changes nothing.
- **Why:** lineage of the 13.54 zip from raw data, and no dev data in the 13.54 chain. `s007_xall.pt` and `v11_t3head.pt` are saved before the removed block; `t3_streams.npz` is read by no 13.54 stage (section 1 above).
