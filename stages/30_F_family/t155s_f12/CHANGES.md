# CHANGES: t155s_f12 (exact-pulled; `release/flag2027_e2e/src_nb/flag-2027-nb-155s-ct02ca-f12/`)

## 1. Repository copy of the notebook: embedded DEV score zip removed (type `dev-exclusion`; numbers: none)

- **Where:** `original/flag-2027-nb-155s-ct02ca-f12.redacted.ipynb` (and therefore `stage.ipy`), code cell 3, the string `CODE_B64 = "..."`.
- **Old → new:** the base64 zip of the run's code had 39 members; the repository copy has 38: the member
  `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip` (91 043 bytes, sha256 `b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d`)
  is removed. Every other member is copied as raw zip entries (same bytes, same CRC, same ZipInfo); the rest of the notebook file is
  byte-identical (`_common/redact_embedded_dev_zip.py`, deterministic: re-running it on the exact notebook gives sha256 `69fe740b48fca42bad69d9edf2a63f53d03648a2f3aa7c3e26c4dc6b9834b48c`).
- **Why:** that member holds per-pair scores on the organisers' DEV lists; the published repository must not contain dev data (user rule 6,
  DESIGN 2.8 / 9.4). The byte-exact notebook stays at `d:/Sinh viên CNhan/FLAG_2027_FourFusion/release/flag2027_e2e/src_nb/flag-2027-nb-155s-ct02ca-f12/flag-2027-nb-155s-ct02ca-f12.ipynb` (md5 `ede88e3ca65c00f5d51df903dcab85c2`, sha256 `2bb4b7031642d7904e21bc65ebf9c367958f21f828d56dba88be7a5801ffce74`) and belongs in the private FF_DATA store.
- **Effect:** none on any array, model, table or score. The only reader of the file is `zip_ct()` of the embedded
  `Experiment/CLEANTUNE-01/run.py` (line 561: `FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None`),
  and `FACELORA-03/run.py cmp_zip()` returns `None` for a missing file (line 406). In this run `zip_ct()` is never called (no zip / report job), so not even a printed line changes.
  The notebook's `CODE_MD5` dict still lists the member; its check loop iterates over the members present in the zip, so it still passes.

## 2. Code cells

Otherwise verbatim: building stage.ipy from the EXACT notebook gives the sha256 of release/flag2027_e2e/code/t155s_f12.ipy (checked 2026-11-18);
the repository stage.ipy differs from it only inside the `CODE_B64` string.

## 3. Evidence kept with this stage

Bengali-voice leak fix of 3/11/2026 12:15 (EXPERIMENT_LOG entry 346): the embedded Experiment/VOICELORA-01/run.py has md5 6d0dfe2bdf72d2204b32dd1285129207 (= the fixed file), whose line 166 reads `drop = set(json.loads((VB / "lid_summary.json").read_text())["speakers_dropped_ids"]) | T.strict_drop("vb")   # 3/11: LIDSTRICT-01, as vb_rows4`; the run was pushed after the fix; log of this run: `VOICELORA-01 VoxBlink rows: 59826 rows, 13471 speakers`.

## 4. Runner

Runner-level only (not a change of this stage's code): the runner replaces the strings /kaggle/input and /kaggle/working by the stage's own input view and working folder, and prints a `[cell] <stage> k/n` line before each cell (DESIGN 2.4).

Runner-level requirement (2026-11-18 fix; not a change of this stage's code): `stage.json` declares `runner_requirements.literal_kaggle_input: true`. The modules decoded from `CODE_B64` at run time (listed in `runner_requirements.why`, with file and line) are written to /kaggle/working/lab and find their inputs by scanning the LITERAL path /kaggle/input, which the string rewrite above cannot reach. With the flag, the runner points /kaggle/input at this stage's own input view on Linux (symlink for the duration of the stage; `FF_LITERAL_KAGGLE=mounted` when a container already mounts it). On Kaggle the runner cannot change /kaggle/input and only warns: run this stage in a session with exactly its inputs attached (as the original run). Without the flag a Linux / Docker run would fail the asserts (lid_strict.json, adapter arrays, feats_vox2b, CT blocks), and a Kaggle orchestrator session would let the modules pick the session's own attachments (e.g. a saved original instead of an artifact rebuilt in the session).

Licence (2026-11-18): the fold heads train with VoxBlink speakers (VOICELORA-01 rows); the fold tables hold scores only, the training inputs are VoxBlink-derived (CC BY-NC-SA 4.0): private FF_DATA only (`stage.json` `licence`).
