# e50cc: changes from the original run

Original run: `thanhhuyen11/flag-2027-nb-50cc-e2e-eval`, pushed by hand 11/11/2026 20:46–20:47 (UTC+7) after
`kaggle/patch_evalphase_checks.py` (kapi_runs line 1467, pushed notebook md5 `b34646dd394b6d52ede3b0b5279029ac`).
Exact executed notebook (team package `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/2_eval/04_flag-2027-nb-50cc-e2e-eval.ipynb`):
sha256 `49022347acefbc24eb43a2370e263d7aa6c081438c18bc0a23d62db5ef7abad1`, same code cells as the pushed bytes. It is **not** in this
repository (change 2); `original/` holds the redacted copy
`04_flag-2027-nb-50cc-e2e-eval_redacted.ipynb` (sha256 `377fcb83c7dd3a1d6ca79c770e102a4a158fb5cbad0fe1a5bf5d83b06d821369`).

| # | where | old → new | why | effect on numbers | type |
|---|---|---|---|---|---|
| 1 | code cell 9/34 "09 encoders", line 193 (`IB_URL = …`) and code cell 20/34 "20 encoders", line 43 (`torch.hub.download_url_to_file(…)`) | `https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth` → `https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth` (a `stage.json` patch, `count` 2) | Meta's link is dead since 12/11/2026; the commit-pinned mirror serves the same checkpoint (sha256 `d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157`, the hash e163i asserted for this URL on 13/11). Same substitution as `release/flag2027_e2e/stages.py` lines 24–26. This notebook does not check the hash itself. | none (same file) | url |
| 2 | code cell 32/34 "32 the submitted zips (dev references)" (notebook cell 41) | 3 036 lines that embedded three base64 zips and wrote them to `repo/reference/` with an md5 assert (`submission_CA_EN_BN.zip` md5 `f6d77418f3a44f2273ab5fa921d8059d`, `submission_v11.zip` md5 `bc741c4a56c310c7e0b134c35a64a8db`, `submission_v12.zip` md5 `a7323cc4a6a26feff0a111745f27694a`) → a 7-line comment that records them; the cell keeps its first four lines (`REF_ZIPS = {}`) and its last line | the three zips are the team's dev-phase score files (pair ids and scores of the dev trial lists): dev data must not be published. They are only compared with this run's own zips when the run is on dev: code cell 33, `for name, ref in (REF_ZIPS.items() if is_dev else [])`, with `is_dev = any(k.startswith("no_gender/") for k in S) and not ONLY_UNDER`; `ONLY_UNDER = 'evaluation_phase'` (code cell 02), so `is_dev` is False. `REF_ZIPS` has no other reader (searched in every cell). | none: the Evaluation branch never iterates `REF_ZIPS` (the redacted cell leaves it empty; the original filled it with three paths that the Evaluation branch does not read) | dev-exclusion |
| 3 | code cell 2/34 "02 what to run" (`stage.ipy` line 36) | `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")` → `ZIP_PREFIXES = ("train_set", "eval", "test")` plus a comment (a `stage.json` patch, id `no_dev_zip`, `count` 1, `in_release: false`; same text as the patch of `f47b_screen2`) | the repository mounts `datasets/ngnnguynchn/flag2027-raw-v4-p` as the organisers' files in `FF_DATA/00_raw/btc_mavceleb_v4` (`train_set.zip`, `dev_set.zip`, meta csv), not as the tree Kaggle had auto-extracted for the original run (log: `…/flag2027-raw-v4-p/train_set/train_set/train_set/train_English.txt`, no `unpacking` line). No train list is then found, so the verbatim fallback `unpack_zips()` runs, and without the patch it unpacks `dev_set.zip` too (9 757 members, ~990 MB) into `/kaggle/working/raw`. With the patch the dev archive is never opened (DESIGN §2.8). Found by the WP-G review of 18/11. | none: `train_set.zip` is unpacked by the same fallback (same train list and media as the auto-extracted copy, other path; the log gains an `unpacking …/train_set.zip` line) and the verbatim last cell removes `raw/` (`shutil.rmtree(WORKDIR / "raw")`), so the output holds no organiser media; dev lists were already skipped by `ONLY_UNDER` | dev-exclusion |

Change 2 is applied to the notebook in `original/`, not as a `stage.json` patch, because the old text is dev data. It is reproducible
and checkable by anyone who holds the exact notebook:

```
python stages/50_eval/_tools/redact_dev_refs.py e50cc <exact 04_flag-2027-nb-50cc-e2e-eval.ipynb> \
       --check stages/50_eval/e50cc/original/04_flag-2027-nb-50cc-e2e-eval_redacted.ipynb
```

The tool checks the exact sha256, decodes the three payloads and checks their md5 against the cell's own asserts, re-serialises the
notebook exactly as the Kaggle-pulled copy (only that cell's source changes), and compares bytes. Code built from the **exact** notebook
with change 1 is `release/flag2027_e2e/code/e50cc.ipy` (sha256 `9889e2a9a1d989a1c5c40101f2cf65cf926d1a13eda419fd184d13f31b455956`);
`stage.ipy` here (sha256 `4f9f4fb7c876adb869e4a2a4ba5a1379793236cd862eb2bea84233862067e925`) differs from it only in code cell 32 (change 2) and in the `ZIP_PREFIXES` line
(change 3; before change 3 it was `6dddeed22d5292d44b51c726fdff5b0d23c8277cfc660273ed1a8bf98e3ee34a`). `_tools/selfcheck_50_eval.py` verifies both statements
when the exact notebook is given (team-private check; the release code is compared with the patches that release also applies, i.e. without change 3).

## Decisions that are not code changes

| item | decision | effect on numbers |
|---|---|---|
| inputs | the nine datasets of the run, as pushed (`original/kernel-metadata.json`): `members-p` and `best-v7-p` are frozen wiring datasets without a producing stage (DESIGN G18): `members-p` = v6 member seeds of s007 / s010B, feeding only the English v9 / v12 arrays `en7` / `enco` that no 13.54 fuser reads; `best-v7-p` = the old `s007_xall.pt` (overridden by `XALL_PREFER=cleancomp`) + MAV-Celeb AdaFace arrays (only `rel` / `relg`, 0 under `NOREL`), and it ALSO holds dev-phase score files (`best_v7/submission_REL2_E0_B3/*/sub_score_v4_*.txt`) that are mounted but never read | none |
| run flags | set by the notebook itself in code cell 01 (`LID_STRICT=1`, `CLEANCOMP=1`, `XALL_PREFER=cleancomp`, `NOREL=1`); `stage.json` `env` is empty | none |
| smoke | the original notebook has no smoke mode; `smoke_env` is empty, as in `release/flag2027_e2e` | — |
| saved output | the saved output of the run (`FF_DATA/04_run_outputs/flag-2027-nb-50cc-e2e-eval`) still holds `repo/reference/*.zip` written by the unredacted cell: private storage only; declared in `stage.json` `outputs.contains` (`dev_data: true`). | none |
| literal `/kaggle/input` | `runner_requirements`: `literal_kaggle_input`, `kaggle_strict: true`, `kaggle: separate session with exactly these inputs` (the decoded v11 head job finds `lid_strict.json` by scanning the literal path); run on Linux / Docker or in a dedicated Kaggle session | none |
| dev-selection disclosure | the XALL bridge `s007_xall.pt` (in the numbers) follows the 's007' recipe, originally chosen per cell with dev pseudo-labels (FUSE-01 / SEL-01b); retrained on train / clean data in CLEANCOMP-01 (`t164f`); F1s′ fuser refitted on SAFE-04 validation (`e112f1sx`). Declared in `stage.json` notes for COMPLIANCE_NOTES. | none |
| embedded fuser JSONs (integration fixer r1, 18/11/2026) | the four frozen fuser files written by cells 28-31 (`BEST/v7_safe/cafuse_fuser.json` 41482021..., `BEST/v7_safe/fuser_en.json` 8c042484..., `Experiment/BIGB-01/cafuse_t3_fuser_v11.json` ec8a201e..., `Experiment/EN-T3-01/fuser_en_t3.json` fd549d18...; raw strings at stage.ipy:3553 / 3750 / 3956 / 4240) are now registered: stage.json `embedded_data`, configs/data_map.json `embedded/...` (class embedded, requirement-1 exception). No code change. They feed only `d['v9']`, `d['v11']`, `d['v12']`; `CAFuse.conditions` (the 7 conditions) does not read the model values (INTEGRATION_r1 B3) |

## Requirement 1: frozen wiring inputs removed (19/11/2026 (requirement-1 closure))

- input `datasets/ngnnguynchn/flag2027-best-v7-p` dropped (stage.json `inputs_dropped`): attached in the original run for s007_xall.pt and the MAV-Celeb AdaFace arrays. XALL_PREFER=cleancomp makes input_dir() take s007_xall.pt from datasets/ngnnguynchn/flag2027-cleancomp-final-p (stage t164f); the AdaFace arrays ext_v{1,2}_complete_face_adaface.npy are byte-identical (sha256 d160e51b... / 0844ca81...) to the f10_models outputs that datasets/ngnnguynchn/flag2027-lab-data-p already mounts (lab/kaggle/output/feats_models/), and under NOREL they are only row-counted (relational bridge skipped). Its dev-phase score files were never read. No producing stage exists.
- evidence: `repo_map/build/REQ1_GAPS.md` (team archive); `python runner/check_stage.py` rebuilds this stage

## v6 members optional (type `wiring`; numbers: none on 13.54) - added 19/11/2026

- **Where:** last cell (driver), the line `assert v6_members is not None, ...`; `stage.json` input `datasets/ngnnguynchn/flag2027-members-p` gets `"optional": true`.
- **Old:** the stage stopped when the saved v6 members (s007 / s010B seeds, no producing stage in this repository) were not attached.
- **New:** when they are absent, the empty folder `members_v6/` is handed to `P6.train_members` (BEST/v6_eval/pipeline.py, the code that made them), which trains each missing (member, seed) from this run's train features and caches it there.
- **Why:** lineage of the 13.54 zip from raw data (`python runner/ffrun.py dag`, section LINEAGE). The members feed only `System.en()` -> `en7[:, :4]` (English v9 / v12 streams); `a_factor` / `a_fvbind` read `b3`, `co`, `t3` and the pair ids of `components_eval.npz`. With the members attached the run is byte-for-byte the original.
