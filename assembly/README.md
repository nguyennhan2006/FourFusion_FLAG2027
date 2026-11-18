# Mode (c): rebuild the submitted zip byte for byte from stored scores

Submission: CodaBench **1054704**, `xjf_r10_trsig.zip`, sha256 `b4354eaf988b2b4ea8c4e1c3fa44d7bcf631ea5cb3cd635ab2965021fe82e507`, Evaluation EER **13.54**:

| file | EER |
|---|---|
| no_gender / English | 10.20 |
| no_gender / Bangla | 11.97 |
| gender / English | 12.98 |
| gender / Bangla | 18.99 |

This is the only mode that reproduces the submission **byte for byte**, file by file. Mode (b), inference from the raw Evaluation media, and mode (a), re-training, re-run GPU stages; those are not bit-deterministic and are judged by Spearman. Mode (c) needs a CPU with Python ≥ 3.10, numpy, pandas, scipy and pyarrow, and takes a few seconds. It needs no GPU and no network.

```
python assembly/assemble_exact.py                     # inputs from $FF_DATA/05_scores (each FF_DATA entry), else assembly/inputs/
python assembly/assemble_exact.py --inputs /path/to/05_scores --out /tmp/ff_assembly
```

The script exits with code 0 and prints `OK: the four members of … equal the submission` when everything matches. A report is written to `<out>/assembly_report.json`.

## Inputs: ten stored files

Each file was made by a stage of the chain. Every sha256 is in `expected.json` → `inputs`, and the script checks each one before it starts (`--no-check-inputs` skips the check).

| file | sha256 (first 16) | made by (stage / original Kaggle run) |
|---|---|---|
| `f1s_eval.zip` | `1b43b14ff763bd0e` | `e112f1sx`: F1s′ scores of the four Evaluation files (identity-classification family) |
| `cleantune01_fuser_bangla_CA.json` | `c4737367e787da6d` | `e112f1sx`: weights of the F1s′ Bangla fuser |
| `ct01r2_final_eval.npz` | `bbad53ed8e320662` | `e112f1sx`: scores of the CA head |
| `components_eval.npz` | `2cd160ae456447c0` | `e50cc`: the other F1s′ streams and conditions |
| `e2e_manifest.json` | `b0ba27c2982f9171` | `e50cc` |
| `rfx_eval.zip` | `1fa53218c9501ef3` | `e163i`: RFX (pair-matching family) with the **original** ImageBind-audio LoRA C1 |
| `rfsfxfvb_eval.zip` | `45044189ae4b228b` | `e163i_fvb`: RFX′ = RFX + FVBIND voice block B2, original LoRA C1 |
| `xjf_scores.npz` | `686c1188ce2cd5c6` (or `6a039fafe5c0f820`, which also has Bangla keys; same English arrays) | `dp_train`: DP (Different Profile) branch scores (file name keeps the internal name XJF; see `docs/NAMING.md`) |
| `xjfdump_v4.parquet` | `9f0a5af468d1f973` | `t219_dumpv4`: scores of held-out **train** pairs, used for the scale constants |
| `consts.json` (optional) | `356242de47674dfb` | `dp_train`: out-of-fold z constants, cross-check only |

The files are not in the repository; see `inputs/README.md`.

## Steps

1. **FACTOR_ENFUSE.** `rebuild_final.py` on the five F1s′ files + `rfx_eval.zip` gives `posteval_FACTOR_ENFUSE_c5rl.zip`.
   - Bangla files: −(F1s′ + RFX) for each pair.
   - English files: −(F1s′ Bangla fuser applied to the English pairs) + RFX value.
   - All four members must equal the stored zip (sha256 `9cf5bbe4…`).
   - `rebuild_final.py` is a verbatim copy of the team package script (`release/factor_enfuse/rebuild_final.py` = `rebuild/rebuild_final.py` of package FACTOR_ENFUSE_c5rl, sha256 `0878da7f…`). On 14/11 it reproduced all four members of the stored zip ("OK: identical").
2. **FVBIND_ENFUSE.** The same script with `rfx_eval.zip` := `rfsfxfvb_eval.zip` gives `posteval_FVBIND_ENFUSE_fvb.zip`.
   - Only its `gender/…Bangla_unheard.txt` member enters the final zip; it must have sha256 `9a619736…`.
   - The other three members are compared with the stored zip (`bb0e0175…`) and reported.
   - The stored zip was made by the team's two-step chain (`make_eval_zip.sh` + `XTEAM-03/run.py`, then `ENFUSE-U/run.py`), which uses the same per-pair sum `float(rfx) + float(f1s)` in F1s′ pair order with `%.8f`.
3. **Scale constants.** `stages/70_dp/dp_sigma/dp_sigma.py` on `xjfdump_v4.parquet` gives `5.0983,3.8376`: numpy std (ddof 0) of the base score over held-out train pairs, mean of three seed replicates. It must be equal.
4. **Final blend.** `stages/80_final/final_blend/blend.py 1.0` with `XJF_SIG=5.0983,3.8376` gives `zips/xjf_r10_trsig.zip`.
   - English = FACTOR_ENFUSE English / SIG − 1.0 × z_DP.
   - Bangla = bytes of the base zips.
   - The four members must have the sha256 values of the submission (`expected.json` → `target.members`).
5. **For information only.** The four members are re-written with the header fields of the submitted zip: time 2026-11-16 11:08:08, made on Windows (`create_system = 0`), `external_attr = 0o600 << 16`, deflate, default level. The whole-file sha256 is then compared with `b4354eaf…`.
   - It can only be equal when the deflate library produces the same bytes. The original was written by Python 3.14.0 on Windows, whose `zlib` module is **zlib-ng 2.2.4** (compatible with zlib 1.3.1).
   - Other zlib builds may compress differently. The member contents, which are what the organisers score, are the same either way.

The runner reaches the same result through the stages `a_factor`, `a_fvbind`, `dp_sigma`, `final_blend` and `final_verify`, with `FF_MODE=assemble`. This script is the stand-alone shortcut and calls the same `dp_sigma.py` and `blend.py`.

**Layout for the runner.** The runner accepts the same flat layout as this script: the ten files directly in `<FF_DATA entry>/05_scores/` (or in `assembly/inputs/`). In `FF_MODE=assemble` every assembly stage looks there itself when the runner has not mounted a saved copy of the producing run: `a_factor` and `a_fvbind` for the Evaluation score files, `dp_sigma` for `xjfdump_v4.parquet` (also under `<FF_DATA entry>/03_checkpoints/sigma_source/`), and `final_blend` for `xjf_scores.npz` (also under `<FF_DATA entry>/05_scores/dp/`). The per-run folders of `configs/data_map.json` (for example `05_scores/dp/`, `03_checkpoints/sigma_source/`) work as well; they are needed only for the runner's plan to show those inputs as available. This holds for every `FF_USE_SAVED` level. In mode (c) the stages stop unless the DP arrays, the scale constants and the stored table equal the originals.

## What it does not show

Mode (c) shows that the submitted numbers follow from the stored per-pair scores by the documented formulas. It does not re-compute those scores; modes (a) and (b) do, see the top-level README. The stored files come from the original runs (Kaggle notebooks and the team machine). Their provenance is recorded in each stage's `original/SOURCE.json`.
