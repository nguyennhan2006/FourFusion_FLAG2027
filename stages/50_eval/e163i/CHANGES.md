# e163i: changes from the original run

**Code changes: none.** `stage.ipy` = the 6 code cells of `original/07_flag-2027-nb-163i-rfsfxc5rl-eval.ipynb`, verbatim
(sha256 `0948c0c00f67bdba2a3e86918e10edd10531395d9c4c3d737754bfefdc06be11`, byte-identical to
`release/flag2027_e2e/code/e163i.ipy`). The notebook already downloads ImageBind from the pinned Hugging Face mirror (set at push time
by `kaggle/patch_163i_eval.py`) and checks the sha256 of all four checkpoints it downloads.

Original run: `thanhhuyen11/flag-2027-nb-163i-rfsfxc5rl-eval`, pushed 13/11/2026 11:20 (UTC+7) by `kaggle/rfx_chain.sh` state 2
(kapi_runs line 1553, pushed notebook md5 `cc1ec0fe56e910c92d2ef3146329a58a`); the team-package copy in `original/` has the same code
cells. The patches applied before that push (`patch_163i_eval.py --data-root …`, `patch_163i_factor.py`) are part of the original code;
`kaggle/patch_163i_dataroot.py` (added 15/11) was not applied and is not applied here.

## Decisions that are not code changes

| item | decision | effect on numbers |
|---|---|---|
| variant (13.54) | the RFX artifact input is `datasets/ngnnguynchn/flag2027-rfsfxc5rl-art-p` with the ORIGINAL LoRA C1 (`lora_C1.pt` sha256 `b97372e3…`). The substitution of `release/flag2027_e2e/stages.py:526-527` (C1FIX artifact of system 16.18 mounted at this path) is NOT applied. `stage.json` `variant` lists the files the runner must check and the artifacts it must refuse. | this is what makes the stage the 13.54 one |
| `DATA_ROOT` mount | `DATA_ROOT` is the constant `/kaggle/input/datasets/ngnnguynchn/flag2027-evalphase-p`; the input `datasets/ngnnguynchn/flag2027-evalphase-p` keeps the default mount (= artifact id), which the runner maps to exactly that path. Its content must be `eval_set.zip` extracted (top folder `evaluation_phase/`), as Kaggle shows a dataset zip. | none |
| inputs | `chnnhnnguyn/external-data-mavceleb` and `ngnnguynchn/flag2027-raw-v4-p` stay attached as in the run, although they may be unused once `DATA_ROOT` is set (DESIGN G25; to test by smoke later) | none |
| runner rewrite inside `CORE_SOURCE` | under `ffrun` the runner's string rewrite of `/kaggle/input` (`runner/ffcore.py` `stage_code`, `count_sub("/kaggle/input", vin)`) also hits one docstring line of the `CORE_SOURCE` literal (core line 1220: 'a recursive search of /kaggle/input with the 28.9 GB external tree'). The core module written to disk (`module_path.write_bytes(CORE_SOURCE.encode())`) then differs from the one that ran, so `FLAG_RF_E2E/run_report.json` `code_sha256` of a re-run is not `3cad7b8a83d3817c…` (the value of the saved run_report and of `stage.json` `origin.builder`). Not a change of `stage.ipy`; WP-A may skip rewrites inside `CORE_SOURCE`. | none: docstring only, no assert compares `code_sha256` |
| variant pins in train mode | `variant.require_files` (sha256 of `lora_C1.pt`, `manifest_final.json`) now carry `applies_to: [saved, attached]`; an artifact made in the session (FF_MODE=train re-learns C1 and the projector) is checked by provenance (`variant.made_check`: made by `t163p_c5rl` from `t163l_orig`'s 210-clip / 70-person C1, `RFS_LORA_FIX` unset). Needs the runner to honour `applies_to` (WP-A todo). | none |
| C1 policy (19/11, INTEGRATION_r2 M3) | `variant.made_check` rewritten: every C1 consumer is `resolve: saved_only` (r1 fixer), so an artifact made in the session carries the saved original `lora_C1.pt` (`b97372e3...`); the earlier requirement "C1 of a `t163l_orig` run made in the same session" was unreachable and is removed. `configs/variant_1354.json` `file_checks[lora_c1].rule` says the same. | none |
| markdown | markdown cell 0 is the teammate's text about their RF-092 notebook (LoRA "C1c", 19 122 clips / 5 929 persons). Kept verbatim; it does not describe the artifact used here (original C1: 210 MAV-Celeb clips of 70 persons). | none (not executed) |
