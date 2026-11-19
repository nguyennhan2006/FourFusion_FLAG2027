# dp_train: changes from the original

Origin: `local-script`. The team ran `Experiment/XJF-01/xjf.py` on its own Windows machine, CPU, on 2026-11-16:

- **02:11:06 to 02:21:32 (+07:00).** Only the English Evaluation features existed. This run's `out/xjf_scores.npz` was used for the submission zip; the log is `original/xjf.log`.
- **21:52 to 22:03.** The same script, after the Bangla features had been extracted. The losses, EERs and constants are identical, and the English scores are bit-identical. Log: `original/xjf_bn.log`.

`original/` holds byte-exact copies (sha256 values in `original/SOURCE.json`): `xjf.py` (sha256 `cb93cd52…`, last modified 02:10:52), the two logs, the report and `consts.json` of the 02:11 run.

The DP branch, for Different Profile, is called XJF inside the team repository; see `docs/NAMING.md`. It trains on the organisers' FLAG train set (70 persons, 6 485 pairs) and on MAV-Celeb v1 / v2 (154 persons). The 58 MAV-Celeb voice files flagged as Bengali speech in `lid_strict.json` (key `mav_files`: 13 in v1, 45 in v2) are dropped, and the script asserts that each one was found and dropped. The training step reads no dev data and no Evaluation data; the Evaluation features are only scored at the end of the script, with constants fixed on the train set. This sentence describes the script's data flow only. It does not claim that the branch raises no compliance question: see "Compliance disclosure" below.

## Script patches

Exact strings and match counts are in `stage.json` → `script_patches`.

| id | where in `xjf.py` | old → new | reason | effect on the numbers | type |
|---|---|---|---|---|---|
| XJ1 | line 11 | adds `os` to the imports | needed by XJ2–XJ7 | none | path |
| XJ2 | line 17 | `KO = REPO / "kaggle" / "output"` → `Path(os.environ.get("DP_FEATS", …))` | the feature arrays come from stage `dp_feats_layout`, with the same sub-folders and file names as `kaggle/output/` | none | path |
| XJ3 | line 18 | `OUT = HERE / "out"` → `Path(os.environ.get("DP_OUT", …))` | the stage folder is read-only | none | path |
| XJ4 | line 49 | `REPO / "Input" / "train_set.zip"` → `os.environ.get("DP_TRAIN_ZIP", …)` | the organisers' zip is read from `FF_DATA/00_raw` | none | path |
| XJ5 | line 52 | `REPO / "Input" / "meta_file_train_set.csv"` → `os.environ.get("DP_TRAIN_META", …)` | as XJ4 | none | path |
| XJ6 | line 60 | `REPO / "Experiment" / "LIDSTRICT-01" / "lid_strict.json"` → `os.environ.get("DP_LID_STRICT", …)` | the list comes from stage `l_lid_strict` (dataset `ngnnguynchn/flag2027-lid-strict-p`, version 2) | none: the stage requires sha256 `f4065711…` | path |
| XJ7 | line 85 | `FE = HERE / "feats_evalphase"` → `Path(os.environ.get("DP_EVAL_FEATS", …))` | the Evaluation features come from stage `dp_extract_eval` | none | path |
| XJ8 | after line 167 (final models) | adds `torch.save(...)` of `out/dp_towers.pt`, holding the state dicts of the 10 tower pairs (seeds 0, 1000, …, 9000), the input dimensions, the centring vectors `MU_F` / `MU_V` and the out-of-fold constants `CONST` | the original kept the trained towers only in memory, so no checkpoint existed (gap G5). The organisers ask for checkpoints, and stage `dp_score` scores from them | none: `train()` re-seeds torch and numpy itself; `state_dict()` and `torch.save` draw no random number and change no model or input. The line is printed rather than logged, so `report.txt` keeps its content | save |

Every default path stays the original one.

## Stage glue (`stage.ipy`, new code)

1. **Inputs.** Locates `train_set.zip` and `meta_file_train_set.csv` (organisers), `lid_strict.json`, `dp_feats/`, the DP Evaluation features and `eval_set.zip`. A mounted-unpacked dataset is re-zipped in a temporary folder with the same member paths, packing only the probed top folder. The sha256 values are printed; the run stops if `lid_strict.json` is not the version-2 list, unless `DP_ALLOW_INPUT_MISMATCH=1`.
   - *Fixer, 18/11.* `eval_set.zip` is taken first from the output of stage `raw_evalphase` (new input `notebooks/ff/raw_evalphase`, the zip the DP branch read), then from `DP_EVAL_ZIP`'s path or the extracted dataset view. Unless it is byte-identical to the submitted file (`a70762a3…`), every member is checked against `stages/10_raw/raw_evalphase/expected/eval_set_members.sha256`; a differing or missing member stops the run (`DP_ALLOW_INPUT_MISMATCH=1` to go on). Before, a re-zipped view was never checked.
   - *Fixer, 18/11.* `dp_feats/` is found by its structure: the one folder under the `dp_feats_layout` input that holds all 20 files of `dp_feats_layout/dp_features.json` (`DP_FEATS_DIR` overrides). Before, the stage needed `dp_feats_manifest.json`, which only `dp_feats_layout` writes, so a saved copy uploaded as documented (the 20 arrays, gap G13) made the stage fail. Without the manifest the 20 files are checked by sha256.
   - *Fixer, 18/11.* The temporary folder is `FF_WORK/tmp/ff_dp/dp_train` (or `DP_TMP`) and is emptied at every run. Before, it sat in the system temp folder and a rebuilt zip from an earlier run (another input, a smoke run) was silently reused.
2. **Run.** Runs `xjf.py <eval_set.zip>`; the output is streamed line by line.
3. **Comparison.** Compares the new `consts.json` with the constants of the submission run, and the arrays of `xjf_scores.npz` with the submission run through the sha256 of `ndarray.tobytes()`. The result goes to `dp_train_check.json`. Bit identity was observed only on the team machine (S8 K6); on other hardware or library versions small differences are expected.
   - *Fixer, 18/11.* `dp_train_check.json` also records `eval_feats_sha256`, the sha256 of the DP Evaluation feature files this run scored. Stage `dp_score` uses it to decide whether its own scores must be bit-identical to the ones written here.
   - *Fixer, 18/11.* The summary no longer prints the mean / standard deviation of z on the Evaluation files (it printed them for information only). Score statistics of the test files are not printed by new code; see "Compliance disclosure", item 5.

## Compliance disclosure

To be carried into `docs/COMPLIANCE_NOTES.md` (WP-K) and into the system description. The data flow of this stage follows the letter of the rules: the external data are declared, no dev or Evaluation data enter training, and the constants are fixed on train persons. The team has nevertheless identified the following risk and history, and a reader of this package must know them.

1. **The branch is weak on unseen persons.** On held-out FLAG train persons, the DP branch alone reaches only EER 25.67 (no_gender) and 34.64 (gender) (`original/xjf.log`, lines 17–18). The base system reaches about 15 / 22 on the same kind of persons. Yet adding the branch lowered the English Evaluation EERs from 13.78 / 22.12 (FACTOR_ENFUSE alone) to 10.20 / 12.98.
2. **Likely cause: recording overlap.** The most plausible explanation is that some recordings of the English Evaluation file are also in MAV-Celeb v1 / v2, on which the branch trains. A team study that simulated overlap on held-out data (team folder `Experiment/DPSTUDY-01/`) found a large gain of the fused system (+17.6 EER points on the MAV-Celeb English same-gender cell) only when the same video / recording is in training. When only the person overlaps, through other videos, the result equals that for unseen persons (the fusion then costs about 6 EER points). The team has not checked the overlap directly, because that requires matching identities across datasets, which the challenge rules forbid.
3. **Held-out test of the fusion.** A test written before running it, which applied the same fusion to held-out train persons, made the system worse (v4 gender −5.65, v4 no_gender −4.91 EER points, better in 0 of 5 folds; team folder `Experiment/XJF-01/`, README section 6).
4. **The organisers have not been asked.** A question about MAV-Celeb v1 / v2 as training data was drafted on 16/11 (`docs/BTC_QUESTION_2026-11-16.md` in the team repository) and had not been sent as of 18/11. The team regards this as a grey zone for the organisers to decide.
5. **Submission order.** `xjf_r10.zip` was submitted first: its two scale constants, 5.8 / 2.9, had been read off the score scale of the test files, which the rules do not allow. It scored 13.54 (16/11 10:31, CodaBench 1054532). Only afterwards were the constants replaced by values measured on held-out train persons (5.0983 / 3.8376, stage `dp_sigma`). `xjf_r10_trsig.zip` was then built (11:08) and submitted (12:02, CodaBench 1054704); it also scored 13.54 on average. The team made ten Evaluation submissions in all (`docs/EVAL_SUBMISSIONS.md` in the team repository); choosing among them with Evaluation scores runs against FAQ question 11 of the organisers.

## To do

- **G5.** The tower checkpoint does not exist yet. Run this stage once on the team machine (about 12 min on CPU) and check that `dp_train_check.json` says `all_identical: true`. Then upload `out/dp_towers.pt`, `out/xjf_scores.npz` and `out/consts.json` to `FF_DATA/03_checkpoints/dp_towers/out/`.

## Comment edit (19/11/2026)

- `xjf.py` lines 1–2: the docstring now says the recipe came from a teammate's report; it no longer names that report's internal branch label. Comment only: no code line changed, no effect on the numbers. `original/` keeps the byte-exact text.
