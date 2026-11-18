# dp_sigma: changes from the original

Origin: `new`. In the original work the two scale constants of the final English blend were **typed by hand** into the submission command, `XJF_SIG="5.0983,3.8376"`, from the log of the Kaggle CPU run `phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4` (16/11). That run was deleted after an error in a copy step that came after the values were printed; its output was downloaded. No script wrote these two numbers. This stage replaces the manual step with `dp_sigma.py`.

What the constants are: SIG[protocol] is the standard deviation of the base score U = F1s′ (out of fold) + RFX (out of fold) over the English pairs of **held-out train persons** of the 5-fold person-disjoint validation SAFE-04:

- cell `v4 ng` (6 448 pairs) for the no_gender file;
- cell `v4 g` (6 448 pairs) for the gender file.

The value is the mean over the three RFX seed replicates `c0r`, `c0s`, `c0t` (Kaggle runs FLAG_213 p1, FLAG_213 p2 and FLAG_216), written with 4 decimals. No Evaluation data is involved.

The statistic is the one printed on the log line `dumpv4 <cell> <replicate>: … sd_all …` of `Experiment/FUSENET-01/run.py` (`FUSENET_MODE=dumpv4`; annotated excerpt in `evidence/fusenet_dumpv4_branch.py.txt`; the byte-exact file, sha256 `df6d739e…`, is kept in stage `t219_dumpv4` as `original/decoded/Experiment/FUSENET-01/run.py`): `g["U_<rep>"].values.astype(float).std()`, numpy with ddof 0, over the rows of one cell in table order. `dp_sigma.py` applies it to the table written by that run, `xjfdump_v4.parquet` (12 896 rows, sha256 `9f0a5af4…`).

| check | value |
|---|---|
| log, cell `v4 ng` (4 decimals) | 5.1007, 5.0980, 5.0961 → mean 5.0983 |
| log, cell `v4 g` (4 decimals) | 3.8331, 3.8481, 3.8314 → mean of the rounded values 3.8375; the full-precision mean 3.837570 gives 3.8376, the value used |
| layer-map recomputation (S7 D11/D12, S8 K3) | 5.098270 / 3.837570 → `5.0983,3.8376` |
| ddof = 1 (not used) | 5.0987 / 3.8379 |

The constants do not depend on the LoRA C1 variant: U uses only the RFX streams `t3`, `nf` and `nv`, without the ImageBind stream. In any case the three seed runs attached the original C1 dataset `onnhqunhh/flag2027-rfs-lora-p` (DESIGN N4).

## Behaviour of the stage

- **Input.** The table is taken, in this order, from `DP_SIGMA_DUMP`; the mounted output of the FLAG_219 run (`notebooks/phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4`, made by stage `t219_dumpv4` in mode (a), or its saved copy `FF_DATA/03_checkpoints/sigma_source` in mode (b)); then the stored copies of mode (c): flat `<FF_DATA entry>/05_scores/xjfdump_v4.parquet` (the layout of DESIGN 5.3, `assembly/inputs/README.md` and `assemble_exact.py`), `<FF_DATA entry>/03_checkpoints/sigma_source/…`, and `assembly/inputs/xjfdump_v4.parquet`. In mode (c) the runner mounts the FLAG_219 output only when a saved copy of class `checkpoint` is allowed (`FF_USE_SAVED=all`); the fallback makes the stage work with the flat layout and with `FF_USE_SAVED=collected` or `none` as well. Reading the stored table in mode (c) is the definition of that mode (DESIGN 5.3 lists it among the ten stored inputs).
- **Output.** `sigma.json["xjf_sig"]` is the exact string that `blend.py` parses, the same way as `XJF_SIG`.
- **Strictness.** `--expect 5.0983,3.8376` is always passed (`DP_SIGMA_EXPECT`). In `FF_MODE=assemble` the stage is strict by default: `dp_sigma.py --strict` stops on a mismatch, and the stage also stops when the table is not the original one (sha256 `9f0a5af4…`). In modes train / infer a mismatch is reported only, because a rebuilt SAFE-04 table may move the 4th decimal; `final_blend` keeps the pinned `XJF_SIG` of variant 13.54 there. An explicit `DP_SIGMA_STRICT=0` / `1` overrides the default. (Before: `DP_SIGMA_STRICT=0` was set in `stage.json` → `env`, so the check was never enforced through the runner; the `verify` entry used a key, `json_field`, that `runner/ffcore.py` does not read. It now uses `key` / `equals`.)
- The sha256 of the input table is compared with the original run's and reported.
- `evidence/` holds the two annotated excerpts used in this file (log lines and the code branch). They are not byte-exact copies; `original/SOURCE.json` gives the sha256 and the in-repository location of the full verbatim sources.
