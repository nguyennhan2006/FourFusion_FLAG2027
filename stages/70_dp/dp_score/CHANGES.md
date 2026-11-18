# dp_score: changes from the original

Origin: `new`. In the original work there was no separate scoring step: `xjf.py` scored the Evaluation trials in memory, right after training the final towers (original lines 166–173). Stage `dp_train` now also saves the towers (`out/dp_towers.pt`, patch XJ8), and `dp_score.py` scores from that checkpoint. Mode (b), inference with a stored checkpoint, therefore needs no re-training.

`dp_score.py` copies these definitions verbatim from `xjf.py` (sha256 `cb93cd52…`), so the arithmetic is the same:

| xjf.py lines | copied |
|---|---|
| 28–45 | `l2`, `csv_feats`, `face_cat`, `voice_cat` |
| 94 | `T` (centring and conversion to a tensor) |
| 100–105 | `Tower` |
| 131–133 | `cos` (mean cosine over the 10 seeds) |
| 166–173 | scoring loop, re-written around the loaded towers: `z = (cos − mean) / sd` with the out-of-fold constants stored in the checkpoint |

New behaviour:

- the checkpoint is loaded with `torch.load(..., weights_only=True)`, falling back to plain `torch.load` on old torch versions;
- input dimensions are checked;
- `--compare` reports, per Evaluation file, whether the ids are equal, whether z is bit-identical, max |Δz| and Spearman against another `xjf_scores.npz`.

Expected result: in the same environment as `dp_train`, the scores are bit-identical to those `xjf.py` wrote; `stage.ipy` cell 3 checks this when both files are present. With a stored checkpoint on other hardware, or with re-extracted Evaluation features (mode b), the scores differ in the last digits. The downstream check is Spearman, in `final_verify`.

## Stage glue (`stage.ipy`, new code)

1. **Inputs.** Locates `dp_towers.pt` (kernel `notebooks/ff/dp_train`, or `DP_TOWERS`), any `xjf_scores.npz` next to it, the DP Evaluation features (`notebooks/ff/dp_extract_eval`) and `eval_set.zip` (re-zipped from an unpacked mount if needed).
2. **Run.** Runs `dp_score.py`, with `--compare` when dp_train's own scores are present; the output is streamed.
3. **Summary.** In mode train, a difference between the two scorings is flagged.

### Fixer changes (18/11)

- **A difference from the verbatim script is fatal when the inputs are the same.** `dp_train` now records in `dp_train_check.json` the sha256 of the DP Evaluation feature files it scored (`eval_feats_sha256`). When those equal the files this stage reads, z must be bit-identical to `dp_train`'s own `out/xjf_scores.npz` (written by the verbatim `xjf.py`); otherwise the stage stops (`DP_ALLOW_INPUT_MISMATCH=1` to go on). Without `dp_train_check.json`, the same rule applies in `FF_MODE=train`. With other features (for example the stored checkpoint of the team machine and re-extracted features in mode (b)) a difference is expected and reported. Before, a difference only printed a warning, and only in `FF_MODE=train`. Reason: `final_blend` reads this stage's scores, and the check guarantees that they are the verbatim script's scores bit for bit whenever both exist (see `final_blend/CHANGES.md`).
- **Clear stop without a checkpoint.** When the `dp_train` artifact holds no `dp_towers.pt` (for instance a saved copy with only the original `xjf_scores.npz` / `consts.json`, gap G5), the stage stops with a message that names G5, `FF_FORCE=dp_train` and the upload path, instead of a bare assertion.
- **Saved output for mode (c).** `skip_with` now also lists the flat file `05_scores/xjf_scores.npz`, so that the runner can use the flat layout of DESIGN 5.3 as this stage's saved output.
- **No score statistics of the test files.** `dp_score.py` no longer prints the mean / standard deviation of z per Evaluation file, only the number of trials scored.
- *Fixer, 18/11.* `eval_set.zip` is taken first from the output of stage `raw_evalphase` (new input `notebooks/ff/raw_evalphase`, the zip the DP branch read), then from `DP_EVAL_ZIP` or the extracted dataset view. Unless it is byte-identical to the submitted file (`a70762a3…`), every member is checked against `stages/10_raw/raw_evalphase/expected/eval_set_members.sha256`; a differing or missing member stops the stage (`DP_ALLOW_INPUT_MISMATCH=1` to go on). Before, a zip re-built from the extracted view, which is the normal case in mode (b), was never checked.
- *Fixer, 18/11.* The temporary folder is `FF_WORK/tmp/ff_dp/<stage>` (or `DP_TMP`) and is emptied at every run. Before, it sat in the system temp folder, and rebuilt zips, unpacked media (`.done` marker of `extract_eval.py`) or model copies of an earlier run (another input, a smoke run, another `FF_DATA`) were silently reused.

## To do

- `dp_towers.pt` does not exist until `dp_train` has run once (G5).
- For mode (c) through the runner, `FF_DATA/05_scores/dp/` (or the flat `FF_DATA/05_scores/xjf_scores.npz`, `skip_with`) must hold the original `xjf_scores.npz` (sha256 `686c1188…`, or `6a039faf…` for the copy that also has Bangla keys) as the stored output of this stage. WP-A owns the data_map row `notebooks/ff/dp_score` → `05_scores/dp/`. `assembly/assemble_exact.py` does not need this row.
