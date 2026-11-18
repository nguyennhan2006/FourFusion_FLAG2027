# final_blend: changes from the original

Origin: `local-script`. The submitted zip was written on the team's Windows machine by `Experiment/XJF-01/blend.py` at 2026-11-16 11:08:08 (+07:00); that time is stored in the zip headers. The command, recorded in the team's experiment README section 6, was:

    XJF_SIG="5.0983,3.8376" python Experiment/XJF-01/blend.py 1.0        (XJF_BASE unset)

`original/blend.py` is a byte-exact copy of the file as it is now (sha256 `ece86b2a…`). It was modified at 17:08 the same day, after the submission. The edit added the `XJF_BASE=lf` branch (lines 17–22 and the `_lf` parts of the zip names) for a later zip, `xjf_r10_trsig_lf.zip`, built on the key-fixed LoRA C1 of another system and never submitted. With `XJF_BASE` unset, the code path is the one of the 11:08 run: the layer maps recomputed all four members of the submitted zip byte for byte with this logic (S7 §0.1, S8 K1).

Recipe of the four files (the zip stores −score; lower means same person):

| file | value |
|---|---|
| `no_gender/…English_heard.txt` | (value of `posteval_FACTOR_ENFUSE_c5rl.zip`) / 5.0983 − 1.0 × z_DP |
| `gender/…English_heard.txt` | (value of `posteval_FACTOR_ENFUSE_c5rl.zip`) / 3.8376 − 1.0 × z_DP |
| `no_gender/…Bangla_unheard.txt` | value of `posteval_FACTOR_ENFUSE_c5rl.zip`, re-written with `%.8f`, so the same bytes |
| `gender/…Bangla_unheard.txt` | value of `posteval_FVBIND_ENFUSE_fvb.zip`, re-written with `%.8f`, so the same bytes |

z_DP is the DP branch score (Different Profile, internal name XJF; see `docs/NAMING.md`), standardised with out-of-fold constants of the FLAG train set. 5.0983 and 3.8376 are fixed constants measured on held-out train persons; stage `dp_sigma` recomputes them. No constant or weight of this step is computed from the test files: the formula has no part fitted at test time. That statement covers this step only; the history of the two constants and the open question about the DP branch are in "Compliance disclosure" below.

## Script patches

Exact strings and match counts are in `stage.json` → `script_patches`.

| id | where in `blend.py` | old → new | reason | effect on the numbers | type |
|---|---|---|---|---|---|
| BL1 | line 8 | adds `json` to the imports | needed by BL5 | none | path |
| BL2 | line 14 | `PE = REPO / "Experiment" / "POSTEVAL-01" / "zips"` → `Path(os.environ.get("DP_POSTEVAL_DIR", …))` | the FACTOR_ENFUSE zip is made by stage `a_factor` | none: same file name; members checked | path |
| BL3 | line 16 | `REPO / "Experiment" / "FVBIND-01" / "zips" / …` → `Path(os.environ.get("DP_FVBIND_DIR", …)) / …` | the FVBIND_ENFUSE zip is made by stage `a_fvbind` | none | path |
| BL4 | after line 20 | adds `assert not LF` | `XJF_BASE=lf` would take the base of another, unsubmitted system; `configs/variant_1354.json` forbids it | none for 13.54: the submission was made with `XJF_BASE` unset | variant |
| BL5 | line 28 | when `XJF_SIG` is unset, read the same string from `sigma.json` (`DP_SIGMA_JSON`, stage `dp_sigma`); if neither is set, stop | the constants can come from `dp_sigma`; the original fallback 5.8 / 2.9 was read off the score scale of the test files (run 9, not allowed) and must never be used silently | none for 13.54: `XJF_SIG="5.0983,3.8376"` takes precedence exactly as before | variant |
| BL6 | line 31 | `np.load(HERE / "out" / "xjf_scores.npz")` → `os.environ.get("DP_SCORES", …)` | the DP scores come from stage `dp_score`, or the stored file in mode (c) | none | path |
| BL7 | line 44 | `(HERE / "zips").mkdir(...)` → `ZD = Path(os.environ.get("DP_ZIP_DIR", …))` | the stage folder is read-only | none | path |
| BL8 | line 45 | default weights `(0.5, 1.0, 2.0)` → `(1.0,)`, used only when no argument is given | without an argument the original wrote three weight variants; only r = 1.0 was submitted, and the team decided not to produce the others | none: the submission command passed `1.0` | variant |
| BL9 | line 47 | `HERE / "zips" / (…)` → `ZD / (…)` | output folder of BL7 | none: same zip name, `xjf_r10_trsig.zip` | path |
| BL10 | line 58 | `print("wrote", p.relative_to(REPO))` → `print("wrote", p)` | the zip is no longer under the team repo root, so `relative_to` would raise | none | path |

The formula (line 55 of the original: `o / SIG[protocol] - r * z`), the `%.8f` writing (line 57), the member order and the refusal to overwrite an existing zip (lines 48–49) are unchanged.

## Stage glue (`stage.ipy`, new code)

1. **Inputs.**
   - Locates `posteval_FACTOR_ENFUSE_c5rl.zip` (kernel `notebooks/ff/a_factor`), `posteval_FVBIND_ENFUSE_fvb.zip` (`notebooks/ff/a_fvbind`), `xjf_scores.npz` (`notebooks/ff/dp_score`) and `sigma.json` (`notebooks/ff/dp_sigma`). Each can be overridden with `DP_FACTOR_ZIP`, `DP_FVBIND_ZIP`, `DP_SCORES_NPZ` or `DP_SIGMA_FILE`.
   - Copies the two base zips under their canonical names into a temporary folder.
   - Variant guard: `XJF_BASE` must be unset; for `FF_VARIANT=1354` (the default), `XJF_SIG` must be `5.0983,3.8376` and r must be 1.0.
   - Prints whether the base members equal the stored ones.
2. **Run.** Runs `blend.py 1.0`; the output is streamed.
3. **Members.** Writes the member sha256 values of the new zip to `final_blend_members.json`, compared with the submission.

The reference values are read from `expected.json` in this folder, an identical copy of `assembly/expected.json` produced by the same generator.

### Fixer changes (18/11)

- **Mode (c) through the runner reads the flat layout.** Before, the DP scores were looked up only in the mounted output of `dp_score`. In `FF_MODE=assemble` that stage cannot run (kind `infer`), so the runner mounts its output only when a saved copy exists; with the ten stored files placed flat in `FF_DATA/05_scores/`, as DESIGN 5.3 and `assembly/inputs/README.md` document, the stage failed. Now, in `FF_MODE=assemble` only, when the mount has no `xjf_scores.npz`, the stage reads `<FF_DATA entry>/05_scores/xjf_scores.npz`, then `<FF_DATA entry>/05_scores/dp/xjf_scores.npz`, then `assembly/inputs/xjf_scores.npz`, which is the search order of `a_factor`, `a_fvbind` and `assemble_exact.py`. In that mode the `ids_*` / `z_*` arrays must equal the stored original (sha256 of `ndarray.tobytes()`, `expected.json` → `inputs.xjf_scores.npz.arrays_sha256`), and the constants that `dp_sigma` recomputed from the stored table must equal the pinned `XJF_SIG`; otherwise the stage stops. In modes train / infer a stored copy is never read by this stage. `final_blend_members.json` records where each input came from. Type: wiring; effect on the numbers: none.
- **Routing of the DP scores (deviation from the DESIGN 4.10 / 4.11 graph).** DESIGN draws `dp_train` → `final_blend`. The stage keeps reading `dp_score`, so that one path serves modes (a) and (b) alike: in mode (b) `dp_train` does not run and only `dp_score` has scores of the re-extracted Evaluation features. Fidelity to the script that ran is enforced in `dp_score` instead: when `dp_train` wrote its own `out/xjf_scores.npz` (verbatim `xjf.py`) from the same DP Evaluation features, which `dp_train_check.json` → `eval_feats_sha256` shows, `dp_score` stops unless its z arrays are bit-identical to those. So the values that reach `blend.py` are the verbatim script's values bit for bit whenever both exist. Before, a difference only printed a warning, and only in `FF_MODE=train`. Type: wiring; effect on the numbers: none.
- **Temporary folder.** `FF_WORK/tmp/ff_dp/final_blend` (or `DP_TMP`), emptied at every run.

## Compliance disclosure

The full text is in `stages/70_dp/dp_train/CHANGES.md`, section "Compliance disclosure", and goes into `docs/COMPLIANCE_NOTES.md` (WP-K). In short: the DP branch alone is weak on held-out train persons (EER 25.67 / 34.64); the English Evaluation gain most likely comes from recordings of the English Evaluation file that are also in MAV-Celeb v1 / v2 (not checked by identity matching, which the rules forbid); the same fusion made the system worse on held-out train persons; the organisers have not been asked yet. Two points concern this step directly.

- **History of the constants.** The first zip of this recipe, `xjf_r10.zip`, used 5.8 / 2.9, read off the score scale of the test files (not allowed; the comment on line 26 of `blend.py` says so). It was submitted and scored 13.54 (16/11 10:31, CodaBench 1054532) before the constants were replaced by 5.0983 / 3.8376, measured on held-out train persons, and before `xjf_r10_trsig.zip` was built (11:08) and submitted (12:02, CodaBench 1054704). Patch BL5 makes sure the old constants can no longer be used silently.
- **Diagnostic printout.** `blend.py` line 43 (verbatim) prints, for each English Evaluation file, the Spearman correlation between the base score and z_DP and the mean / standard deviation of z_DP. The printout uses no label and feeds no value, so it is kept verbatim. It is disclosed because the disallowed 5.8 / 2.9 came from score statistics of the test files, the kind of number this line prints. New code in the DP stages no longer prints score statistics of the test files.
