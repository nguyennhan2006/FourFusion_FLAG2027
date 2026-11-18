# dp_feats_layout: changes from the original

Origin: `new`. There is no original code to change. On the team machine, `xjf.py` and `extract_eval.py` read 20 arrays and tables directly from `<team repo>/kaggle/output/{feats_v2, feats_models, feats_eval, feats_ext_full}/`. This stage rebuilds that folder layout, `dp_feats/<sub-folder>/<file>`, from the outputs of the stages that make the files. Every file name and sub-folder is the one the scripts expect, so no script has to change its file names.

| sub-folder | files | producing stage (run in the team's work) |
|---|---|---|
| `feats_v2/` | `face_arcface_train.npy`, `voice_ecapa192_train.npy`, `voice_ecapa6144_train.npy` | `f12_eval_feats` (FLAG_12 `feats_eval/`). The arrays first came from FLAG_04; FLAG_12 re-made the same three files with identical sha256. FLAG_04 also wrote cropped variants whose crop lengths were chosen from the durations of dev files. DP never used those, and taking the FLAG_12 copies means FLAG_04 need not run at all |
| `feats_models/` | `face_adaface_train.npy`, `ext_v{1,2}_complete_face_adaface.npy` | `f10_models` (FLAG_10) |
| `feats_eval/` | `ext_v{1,2}_complete_face_arcface.npy`, `ext_v{1,2}_complete_voice_ecapa192.npy`, `ext_v{1,2}_complete_voice_ecapa6144.npy` | `f12_eval_feats` (FLAG_12, MAV-Celeb voices cut to the middle 12 s) |
| `feats_ext_full/` | `ext_v{1,2}_complete_{face,voice}_meta.csv`, `ext_v{1,2}_complete_face.npy` (VGG-Face fc7), `ext_v{1,2}_complete_voice.npy` (organisers'-type ECAPA, float32) | `r08_mav_rows` (the full FLAG_08 run of git commit `8544b3b`; its folder `feats_ext/` was renamed `feats_ext_full` on the team machine) |

`dp_features.json` lists every file with the sha256 of the bytes used for the submission. These were measured on the team machine on 18/11; the total is about 0.9 GB.

## Behaviour

- In each producing artifact, a file is found by its exact name. If the name occurs more than once, the copy inside the producing run's own folder (`feats_eval`, `feats_models` or `feats_ext*`) is taken. A file that is still ambiguous stops the stage.
- Files are hard-linked when the artifact store allows it, and copied otherwise.
- The sha256 values go to `dp_feats/dp_feats_manifest.json` and are compared with `dp_features.json`. Re-running the producing GPU stages gives slightly different arrays, so a difference is reported and is fatal only with `DP_FEATS_STRICT=1`.
- *Fixer, 18/11.* `DP_FEATS_STRICT` now defaults to 1 (stop on a byte difference) except in `FF_MODE=train`, where the producing GPU stages are re-run; it is no longer set to 0 in `stage.json` → `env`, and an explicit value still wins. Shape and dtype of every `.npy` file are checked against `dp_features.json` in all modes. Reason: in modes (b) and (c) the arrays come from saved copies, and a copy of the wrong bundle (the empty 6485 × 0 placeholder of `flag2027-lab-data-p`, or float16 copies of `flag2027-eda-bundle-p`) was laid out with only a NOTE; `xjf.py` then trained for about 11 minutes before failing, and float16 substitutes would not fail at all.
- *Fixer, 18/11.* The downstream stages (`dp_extract_eval`, `dp_train`) no longer need `dp_feats_manifest.json`: they find the folder by its structure and check the 20 sha256 values themselves when the manifest is absent.

## Known traps (not used here)

- `kaggle/output/feats_ext/` (150 utterances per speaker, 12 s cut, `kaggle/local_extract_ext.py`) is a different table.
- The MAV-Celeb arrays inside `flag2027-eda-bundle-p` store the organisers'-type ECAPA as float16, and `flag2027-lab-data-p` holds empty (N × 0) ECAPA-6144 arrays. Neither can replace these files.

## To do

- **G13.** The 20 files used for 13.54 exist today only on the team machine. Upload them as `FF_DATA/02_intermediate/dp_features/dp_feats/<sub-folder>/<file>` so that this stage, and the stages that need it, can be skipped with the stored arrays (`skip_with`). No manifest is needed in the upload.
- **WP-A, data_map.** Give `notebooks/ff/dp_feats_layout` a file list (the 20 targets and sha256 values of `dp_features.json`), give `notebooks/ff/f12_eval_feats` the real hashes of the arrays the DP branch reads (`face_arcface_train.npy` `55491b54…`, `voice_ecapa192_train.npy` `b3bd0a64…`, `voice_ecapa6144_train.npy` `195490e2…`), and mark the lab-data hash `91f05874…` (an empty placeholder) as such.
