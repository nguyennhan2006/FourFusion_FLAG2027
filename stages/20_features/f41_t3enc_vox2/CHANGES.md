# CHANGES: stage `f41_t3enc_vox2`

Original code: Experiment/T3ENC-01/flag-2027-nb-41.ipynb (exact-executed-copy). Stage code: `stage.ipy`. Change types follow DESIGN.md §2.5.
Besides the changes below, the runner points every `/kaggle/input` and `/kaggle/working` string at the stage's own input view and working directory (runner/ffcore.py, as release/flag2027_e2e/orch.py `_stage_code`); that is not a code change.

## Code changes

| # | where | type | old -> new (full text in stage.json) | why | effect on numbers |
|---|---|---|---|---|---|
| 1 | cell 2, line 5 | `pin` | `    !pip -q install --no-deps "git+https://github.com/wenet-e2e/wespe…` -> `    # FF environment patch (CHANGES.md): the team's own pip fix of 29…` | on the current Kaggle image the openai CLIP build fails (its setup.py needs pkg_resources) and, in this shared pip call, takes wespeaker down with it (BEST/v1977/README.md §6). Same packages and commits (wespeaker@9fecd6c, CLIP@d50d76d), one pip call each; CLIP from a git clone at d50d76d when the pip build fails | none (same code at the same commits) |
| 2 | cell 6, line 6 | `url` | `    with requests.get(ZIP_URL.format(k), headers={"Range": "bytes=0-0…` -> `    with requests.get(ZIP_URL.format(k), headers={"Range": "bytes=0-0…` | ProgramComputer/voxceleb is gated today (403 for a token without accepted terms, 18/11/2026); the original resolved the zip URL without a token. With HF_TOKEN set, the resolve request carries it (requests drops the header on the redirect to the CDN host); same files | none (same files; without HF_TOKEN the request is the original one) |

## Other differences from the original run (inputs, wiring, environment)

- **wiring**: The key lists are read from datasets/ngnnguynchn/flag2027-vox2-keys (face_meta.csv, voice_meta.csv) instead of the FLAG_37 output dataset hthlin/flag2027-feats-vox2 that the run attached: the two files are byte-identical (sha256 abf8cb53... / 3db779d3..., S2 A08). The code finds them by name (rglob face_meta.csv / voice_meta.csv).
- **wiring**: Media are streamed from Hugging Face (gaunernst/voxceleb2-dev-wds tars; ProgramComputer/voxceleb vox2_mp4_{2..6}.zip by HTTP range). `resolve/main` is not pinned (G27).

## Notes

- Deterministic given the keys: frames are the middle frame of the keyed track, voices the keyed clip (the tar order only changes the visiting order).

## Run checks (stage.json post_checks / verify; fixer 18/11/2026)

- post_check (fatal): `feats_vox2t3/feats_manifest.json` `faces` = 29251: FLAG_37 face keys (original manifest 2026-10-07 15:51).
- post_check (fatal): `feats_vox2t3/feats_manifest.json` `voices` = 38345: FLAG_37 voice keys.
- post_check (fatal): `feats_vox2t3/feats_manifest.json` `coverage` = {"face_farl16": 1.0, "face_clipL": 1.0, "voice_sam100p": 1.0, "voice_sam100f": 1.0}: every key encoded, as in the original (a lower coverage changes the rows l_dqual_g3 / l_lid_strict see).
- Comparisons the runner cannot express (row cosine, NaN pattern, finite-row masks) are kept as text in stage.json `verify_notes` for runner/verify.py / a person; they no longer sit in `verify` (where an entry without a file logged `WARN verify: missing None`).
