# CHANGES: stage `f63w_aux_voice`

Original code: kaggle/FLAG_63w_aux_voice_feats.ipynb (builder-regenerated). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

none

## Other differences from the original run (inputs, wiring, environment)

- **upload**: VFEAT_UPLOAD is not set (the notebook reads it from the environment; the original run had it set, so it uploaded the dataset). The runner also clears it (neutralize_upload).
- **wiring**: HF_TOKEN is read from the environment / Kaggle secret by the runner (the original got it from a kapi.py secret cell); never printed or written.
- **environment** (unverified assumption): The pushed notebook almost certainly carried a `kapi --env` cell that kaggle/kapi.py does not log (cmd_push inserts smoke_cell(envs, '# run parameters (kapi --env) ...') but the kapi_runs.jsonl line holds only time, account, kernel, source, notebook_md5, gpu, smoke, secrets, datasets, kernels, ok). Evidence: the code reads VFEAT_UPLOAD only from the environment and the output dataset hthlin/flag2027-aux-voice-feats-p exists (created 16/10 17:55 UTC at the end of the run, made private afterwards: docs/RUNS.md:615, docs/DATASETS.md:72). Other environment knobs the code reads (AUDIO_TIME_LIMIT_H (default 10.5), AUDIO_THREADS, AUDIO_TAR_THREADS, AUDIO_SMOKE_*) may have been set in the same cell; this stage assumes their defaults. Unverified: re-applying the kapi transform with or without such a cell does not give the pushed md5 5a9d3bc9c869 locally.

## Notes

- Cuts the centre 12 s of each clip (different from the 3 s audio cache used by the RFS bundle): kept as in the system.

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats_aux_voice/vfeat_manifest.json` `complete` = true: the original upload gate (stage.ipy: `if UP and not SMOKE and man["complete"] and ok`): coverage >= 0.97 for every source; the upload itself is neutralised, so the gate is enforced here.
- post_check (fatal): `feats_aux_voice/vfeat_manifest.json` `arrays` = 6 entries: rows / wanted rows per source and encoder of the original vfeat_manifest.json (package E hthlin__flag2027-aux-voice-feats-p.zip, created 2026-10-16 17:55).
- verify (warning only): `feats_aux_voice/vfeat_manifest.json` `arrays`: finite rows equal to the original's (warning only: the Hugging Face sources may have changed).
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).

## Comment edit (20/11/2026)

Patches 0..1 of `stage.json` (ids `comment_team_label_*`) change comment text only: another team's label is replaced by a neutral wording (public naming rule, `docs/NAMING.md`). No code, path, parameter or number changes; the original notebook in `original/` is unchanged.
- comment now reads `A teammate\'s VoxBlink pipeline (CB` (1x; old text in the `stage.json` patch)
- comment now reads `encoder family of a teammate\'s VoxBlink system` (1x; old text in the `stage.json` patch)
- `stage.ipy` sha256 c2018004ec012d87... -> 7e78c56a94b9336a...
