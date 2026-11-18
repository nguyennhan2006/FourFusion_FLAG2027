# CHANGES: e163i_fvb

Stage code = the 6 code cells of the exact notebook thanhhuyen11/flag-2027-nb-163i-rfsfxfvb-eval (kapi_runs:1609, md5 354351113d5e...). The notebook holds no organisers' data, so original/ has the byte-exact copy.

Every change against the code of the original run (fields: type, where, old, new, why, effect on numbers). The runner also maps `/kaggle/input` and `/kaggle/working` to the stage's own folders and prints a `[cell]` line before every cell. The mapping rewrites every occurrence in the stage text, including one inside the CORE_SOURCE string literal (see 'Runner rewrite inside CORE_SOURCE' below); it changes no executed path of the code below.

## 1. artifact folder pinned

- type: `wiring`
- where: config cell (notebook cell 1 = stage.ipy cell 0), line `ART = None ...`
- old: `ART = None           # None = tự tìm thư mục có manifest_final.json` (first manifest_final.json returned by rglob)
- new: `ART = Path('/kaggle/input/datasets/thanhhuyen11/flag2027-rfsfxfvb-art-p')` + comment
- why: the attached FLAG_192 kernel output also holds rfs/ref/manifest_final.json (RF-092 reference manifest written by its setup cell), so the original choice depends on directory order; the run resolved the RFX' artifact (log: 'artifacts: /kaggle/input/datasets/thanhhuyen11/flag2027-rfsfxfvb-art-p', 'FACTOR-01 extra blocks: [f_c5s] [v_rl, v_fvb]'). A wrong choice fails loudly (no final_params.npz)
- effect on numbers: none (the same artifact as the run)

## 2. ImageBind code pinned to a commit

- type: `pin`
- where: setup cell (notebook cell 2 = stage.ipy cell 1), the conditional pip call
- old: `pip('--no-deps', 'git+https://github.com/facebookresearch/ImageBind.git')`
- new: `pip('--no-deps', 'git+https://github.com/facebookresearch/ImageBind.git@53680b02d7e37b19b124fa37bae4b6c98c38f5be')`
- why: unpinned HEAD in the run; 53680b02d7e37b19b124fa37bae4b6c98c38f5be = main on 2026-11-05 (release/docker/Dockerfile); the install only runs when the image has no imagebind package. The string inside CORE_SOURCE (an ImportError message) is not touched
- effect on numbers: none expected

## Not changed (deliberately)

- DATA_ROOT hard-coded to /kaggle/input/datasets/ngnnguynchn/flag2027-evalphase-p (the runner mounts the dataset there).
- Checkpoint URLs (HF mirror of ImageBind pinned to a commit; resnet221 / voxblink2 'resolve/main' and the Oxford VGG-Face URL) kept: every download is checked against the sha256 table of the config cell.
- pip('--no-deps', '-U', 'speechbrain>=1.1.1') and the other conditional installs kept (environment pinning: docker/).
- CORE_SOURCE (the RF core, sha256 3cad7b8a...) untouched in stage.ipy: the run prints its sha256.

## Runner rewrite inside CORE_SOURCE (not a change of stage.ipy)

- type: `path` (applied at run time by the runner, runner/ffcore.py stage_code)
- where: CORE_SOURCE string literal (stage.ipy line 43), one occurrence of `/kaggle/input` in the docstring of locate_roots (= original/embedded/code/flag_e2e_core.py:1220, '... a recursive search of /kaggle/input with the 28.9 GB external tree ...')
- old / new: `/kaggle/input` -> the stage's input view (the same rewrite as for the rest of the stage text; release/flag2027_e2e/orch.py:212 does the same)
- effect: FLAG_RF_E2E/code/flag_e2e_core.py, the printed 'Code SHA256' and run_report.json code_sha256 of a runner run differ from 3cad7b8a...; the expected line 'Code SHA256: 3cad7b8a...' in stage.json log_expectations is therefore marked 'original run only'
- effect on numbers: none (docstring only)
