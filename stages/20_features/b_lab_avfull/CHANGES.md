# CHANGES: stage `b_lab_avfull`

Original code: none kept (exactness `new`). The dataset `datasets/ngnnguynchn/flag2027-lab-avfull-p` was packed on 15/10/2026 18:06 by an ad-hoc script in a Claude scratchpad of the team machine (deleted with the Temp folder) and uploaded at 18:07 as a private Kaggle dataset (kaggle/kapi_runs.jsonl line 300). Stage code: `stage.ipy` (3 new cells, each starting with `print("[cell] ...", flush=True)`).
The runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory; that is not a code change.

## What the stage rebuilds

| member | made from |
|---|---|
| `labavfull/Experiment/CAFUSE-EN/feats_av_full/ext_v{1,2}_complete_{face,voice}_{aclip,rows}.npy` (8 files) | byte copies of the files of the same name in the output of stage `f32_aclip_full` (FLAG_32 run 2) |
| `labavfull/lab_manifest.json` | `{"created", "kind": "data-avfull", "files": {path: bytes}, "note"}` with the original note text, json indent 1: the keys of the original manifest and of kaggle/build_lab_bundle.py build_vb() (stage b_lab_vb) |

The layout (top folder `labavfull/`, repository-relative paths below it) is the one the consumers' OPTIONAL_DIRS entry `CAFUSE-EN/feats_av_full` expects (e112f0sx).

## Differences from the original dataset

- `labavfull/lab_manifest.json` carries the build time (`created`), so it is never byte-identical (the original says 2026-10-15 18:06); it is not compared.
- The 8 npy files are compared by sha256 with `expected_members.json` (read 19/11/2026 from the package E: zip, sha256 b5febc05...). With the frozen f32 output (or the f32 skip_with view = the frozen `feats_av_full` folder) they are identical (local test 19/11/2026: 8 / 8 identical, guard pass). After a GPU re-run of FLAG_32 the AudioCLIP arrays differ; the row files must not.

## Guard (stage.json `guard_policy`, read by cell 3)

| difference in | FF_MODE=train | FF_MODE=infer / assemble / unset |
|---|---|---|
| `*_aclip.npy` | warn (FLAG_32 is a GPU run, not bit-identical when re-run) | fail |
| `*_rows.npy` | fail (arange(N) of the ext-meta tables: other rows = misaligned arrays) | fail |

`FF_ALLOW_MISMATCH=1` or `FF_SMOKE=1` turn every failure into a warning. Same policy family as l_groups / l_dqual_g3 / l_lid_audit / l_lid_strict. The decision and the list of differing files are written to `bundle_report.json` (`guard`).

## Run checks (stage.json post_checks / verify)

- cell 2 (fatal outside smoke): row files int64 arange(N) with N = 7 580 / 19 556 / 8 953 / 20 715 (ext-meta rows); AudioCLIP arrays float16 (N, 1024) and finite.
- post_check (fatal): `bundle_report.json` `vs_original/missing` = [] and `structure/ok` = true.
- verify (warning only): `vs_original/extra` = [], `vs_original/differ` = [].

## Notes

- Private: features of MAV-Celeb v1 / v2 (organisers' data, challenge terms); shared only with the organisers through FF_DATA, never in the repository.
- Wiring only: e112f0sx reads these arrays on the F0 English path (EN-T3-01 table_en), which reaches f1s_eval.zip but not the 13.54 zip.
