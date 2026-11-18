# final_verify: changes from the original

Origin: `new`. In the original work the zip was checked by hand: member sha256 values, and `python Experiment/EVALPATH-U/check_eval_zip.py <zip>` for the format.

| file | origin | changed? |
|---|---|---|
| `check_eval_zip.py` | verbatim copy of `Experiment/EVALPATH-U/check_eval_zip.py` (sha256 `bde629bc…`; byte-exact copy in `original/`) | no |
| `verify_zip.py` | new: member sha256 values against `assembly/expected.json`, whole-zip sha256 (for information), and per-file Spearman / max \|difference\| against the submitted zip when it is available | new |
| `stage.ipy` | new runner glue (3 cells, each opening with `print("[cell] …")`) | new |
| `expected.json` | identical copy of `assembly/expected.json` (same generator, same bytes), so the stage still works when a code package holds only `runner/`, `stages/` and `configs/` | new |

## Criteria

- **Format, always required.** Four files, 392 / 802 / 416 / 716 pairs, finite values, non-constant scores (`check_eval_zip.py`), plus unique pair ids (`verify_zip.py`).
- **Mode `assemble`, or `FF_VERIFY_STRICT=1`.** All four members must be byte-identical to submission 1054704 (sha256 values in `assembly/expected.json` → `target.members`).
- **Modes `train` / `infer`.** Re-run GPU stages are not bit-deterministic. Member identity and Spearman against the submitted zip are **reported**; the target in DESIGN §5.2 is ≥ 0.9999 per file. `verify_zip.py --strict-spearman` can enforce it.
- The sha256 of the whole zip is informational. `zipfile` writes the current time into each member header, so a rebuilt zip has another sha256 even when the members are identical. `assembly/assemble_exact.py` step 5 tries the submitted header fields.

## To do

- The submitted zip is not copied into the repository while Q-D (public or private) is open. Place it at `FF_DATA/05_scores/xjf_r10_trsig.zip`, or set `FF_REFERENCE_ZIP`, to get the Spearman comparison.
- That reference zip is read straight from `FF_DATA/<entry>/05_scores/` and is not declared as an input (`stage.json` → `reads_undeclared`). Declaring it now would block the stage when the optional zip is absent: the file has no producing stage, so the runner does not treat it as a soft input. WP-A: add a data_map entry (class `score`, path `05_scores/xjf_r10_trsig.zip`, private) and declare the input once the runner supports optional inputs.
