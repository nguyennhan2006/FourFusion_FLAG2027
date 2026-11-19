# dp_extract_eval: changes from the original

Origin: `local-script`. The team ran `Experiment/XJF-01/extract_eval.py` on its own Windows machine, CPU only, on 2026-11-16 from 01:22:37 to 01:56:45 (+07:00). The log is in `original/extract.log`. The command was `python Experiment/XJF-01/extract_eval.py <eval_set.zip> <unzip folder> English`; the second argument is not in the log and only sets where the media are unzipped. Byte-exact copies of the files that ran are in `original/`; their sha256 values are in `original/SOURCE.json`.

| file in this folder | made from | changed? |
|---|---|---|
| `extract_eval.py` | `original/extract_eval.py` (sha256 `dad9866b…`) + script patches EE1–EE7 | yes, 7 patches |
| `flag_extract.py` | `original/flag_extract.py` = `kaggle/flag_extract.py` (sha256 `7c227c4c…`, unchanged since 27/09) | no |
| `flag_models.py` | `original/flag_models.py` = `kaggle/flag_models.py` as of 12/11 22:03, which is before the run (sha256 `9172b68e…`) + patch FM1 | yes, 1 patch |
| `stage.ipy` | new runner glue: 4 cells, each opening with `print("[cell] …")` | new |
| `models.json`, `expected_outputs.json` | new: pins and sha256 values measured on the team machine | new |

The patches are listed in `stage.json` → `script_patches` as exact `old` → `new` strings with match counts. `runner/check_stage.py` can rebuild every patched file from `original/`.

## Script patches

| id | where | old → new | reason | effect on the numbers | type |
|---|---|---|---|---|---|
| EE1 | `extract_eval.py` line 6 | `import io, json, sys, time, zipfile` → adds `os` | needed by EE2–EE7 | none | path |
| EE2 | line 11 | `sys.path.insert(0, str(REPO / "kaggle"))` → `os.environ.get("DP_CODE", …)` | the helper modules sit in the stage folder | none: same module files | path |
| EE3 | line 16 | `OUT = HERE / "feats_evalphase"` → `Path(os.environ.get("DP_EVAL_FEATS", …))` | the stage folder is read-only, so output goes to the stage's working folder | none | path |
| EE4 | line 17 | `KO = REPO / "kaggle" / "output"` → `Path(os.environ.get("DP_FEATS", …))` | the reference train arrays for the self-check come from stage `dp_feats_layout` | none | path |
| EE5 | line 36 | `zipfile.ZipFile(REPO / "Input" / "train_set.zip")` → `os.environ.get("DP_TRAIN_ZIP", …)` | the organisers' `train_set.zip` is read from `FF_DATA/00_raw`; only the 20 self-check rows are used | none | path |
| EE6 | lines 74–75 | the `except` block of the AdaFace section now ends with `raise` | the original swallowed any AdaFace error and went on. `xjf.py` would then fail later because the `face_adaface_*` arrays are missing (S7 gap 10, DESIGN WP-I) | none: on 16/11 the run took the success path (`extract.log` 01:31:53 "AdaFace check", 01:38:59 and 01:46:14 "adaface … done") | bugfix |
| EE7 | line 78 | `X.Ecapa(KO / "_ecapa_sb")` → `Path(os.environ.get("DP_ECAPA_SB", …))` | the SpeechBrain savedir is prepared by stage cell 2 from the pinned revision | none: same five files, sha256 checked | path |
| FM1 | `flag_models.py`, `AdaFace.__init__`, line 123 | `snapshot_download(self.ID)` → `snapshot_download(self.ID, revision=os.environ.get("FF_ADAFACE_REVISION", "54f602a0737bd1ee4a4e7e9fd089a485f397fefd"))` | the original took whatever revision was latest. On 16/11 the team machine's Hugging Face cache held only snapshot `54f602a0…` (`refs/main` = `54f602a0…`), so that is the revision that ran | none if the hub files are unchanged; the stage checks `pretrained_model/model.pt` sha256 `e312d792…` | pin |

Every default path stays the original one: a patched script started from its original place in the team repository behaves exactly as before.

## Stage glue (`stage.ipy`, new code)

1. **Inputs.** Locates `eval_set.zip` (dataset `ngnnguynchn/flag2027-evalphase-p`), `train_set.zip` (dataset `ngnnguynchn/flag2027-raw-v4-p`) and `dp_feats/` (stage `dp_feats_layout`), and prints the sha256 values. If a dataset is mounted unpacked, an equivalent zip with the same member paths is rebuilt in a temporary folder. Only the probed top folder is packed (`evaluation_phase/` or `train_set/`), never `dev_set/`.
   - *Fixer, 18/11.* Cell 1 first checks that the Python packages of this stage can be imported (insightface, onnxruntime, speechbrain, omegaconf, huggingface_hub, torch, torchaudio, …) and otherwise stops with the list and the pinned requirement files (`docker/requirements-dp-local.txt`, `docker/requirements-kaggle-pinned.txt`). `pip_policy` stays `skip`, because the runner has no per-stage requirement subset; before, a stage run alone (`FF_ONLY`) failed late at an import.
   - *Fixer, 18/11.* `eval_set.zip` is taken first from the output of stage `raw_evalphase` (new input `notebooks/ff/raw_evalphase`, the zip the DP branch read), then from `DP_EVAL_ZIP` or the extracted dataset view. Unless it is byte-identical to the submitted file (`a70762a3…`), every member is checked against `stages/10_raw/raw_evalphase/expected/eval_set_members.sha256`; a differing or missing member stops the stage (`DP_ALLOW_INPUT_MISMATCH=1` to go on). Before, a zip re-built from the extracted view, which is the normal case in mode (b), was never checked.
   - *Fixer, 18/11.* The temporary folder is `FF_WORK/tmp/ff_dp/<stage>` (or `DP_TMP`) and is emptied at every run. Before, it sat in the system temp folder, and rebuilt zips, unpacked media (`.done` marker of `extract_eval.py`) or model copies of an earlier run (another input, a smoke run, another `FF_DATA`) were silently reused.
   - *Fixer, 18/11.* `dp_feats/` is found by its structure (the one folder under the `dp_feats_layout` input that holds all 20 files of `dp_features.json`; `DP_FEATS_DIR` overrides) instead of through `dp_feats_manifest.json`, which only `dp_feats_layout` writes. A saved copy uploaded without the manifest (gap G13) now works and is checked file by file against the sha256 values.
2. **Pinned models** (`models.json`):
   - insightface `buffalo_l`: checked by the sha256 of `det_10g.onnx` and `w600k_r50.onnx`.
   - AdaFace: Hugging Face `minchul/cvlface_adaface_ir101_webface12m` at `54f602a0…`.
   - SpeechBrain: `speechbrain/spkrec-ecapa-voxceleb` at `0f99f2d0…`. The savedir is the 5 files of that snapshot, with `label_encoder.txt` saved as `label_encoder.ckpt`. SpeechBrain 1.1 reuses files that already exist in the savedir.

   A copy found under `FF_DATA/06_pretrained` is used first; otherwise the pinned version is downloaded.
3. **Run.** Runs `extract_eval.py <eval_set.zip> <tmp unzip folder> English` with the paths given through the environment. `CUDA_VISIBLE_DEVICES=""` is set unless `DP_DEVICE=gpu`, because the original run was CPU only. The output is streamed line by line.
4. **Checks.** The self-check thresholds are new checks; the script itself only records the values:
   - ArcFace and AdaFace minimum cosine ≥ 0.999 on 20 train rows (16/11 run: 0.9999995);
   - ECAPA minimum cosine ≥ 0.99 (16/11 run: 0.99939).

   The sha256 values of the outputs are compared with the 16/11 run and reported; they are equal only on the same machine and libraries.
   - *Fixer, 18/11.* `expected_outputs.json` keyed the self-check file as `verify_english.json`, the name the team gave its copy after the later Bangla run wrote `verify.json` again. `extract_eval.py` writes `verify.json`, so that file was never compared. The key is now `verify.json` (same sha256 `9ef21fed…`, with a note); `original/verify_english.json` keeps the team's name.

## Not changed

Feature definitions, face alignment, model calls, batching, file names and row order are those of the original scripts and modules.

**Batch padding (disclosed, kept verbatim).** The ECAPA voice embeddings of the Evaluation clips are computed in batches packed by duration (`flag_extract.run_batched` → `pack_batches`, at most 240 padded seconds and 32 clips per batch; `extract_eval.py` lines 89–90). Each batch is zero-padded to its longest clip and passed with relative lengths (`Ecapa.__call__`, `flag_extract.py` lines 165–178). The embedding of one clip can therefore vary in the last digits with the lengths of the other test clips in its batch, through convolution context at the padded end and the pooling. No label and no statistic of the test files enters, but strictly a pair's DP score is not a function of that pair alone (rule level 0). The 13.54 features were made this way (`original/extract.log`). The code is kept verbatim for fidelity; the point goes into `docs/COMPLIANCE_NOTES.md` (WP-K). Optional check after 20/11: re-extract with `max_bs=1` and report max |Δz| and the Spearman correlation of z_DP against the submission arrays.

## Missing packages installed at the pinned versions (19/11/2026)

- A fresh Kaggle session has no `onnxruntime`, `insightface` or `speechbrain`; the package check stopped the stage there (Kaggle check
  run `dngxunqun/ff1354-chk-b1b-4b6be30`). New: unless `FF_PIP=skip`, the missing packages of the check list are installed at the
  versions of `docker/requirements-dp-local.txt` (`insightface==2.0` with `--no-deps`, as on the team machine), then checked again.
  Nothing changes where the packages are already present (team machine, Docker image).
