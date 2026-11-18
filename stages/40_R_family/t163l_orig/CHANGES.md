# CHANGES: `t163l_orig` (onnhqunhh/flag-2027-nb-163-rfs-l)

Source of `stage.ipy`: `original/15_flag-2027-nb-163-rfs-l.redacted.ipynb`, a redacted copy of the exact notebook `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/15_flag-2027-nb-163-rfs-l.ipynb` (md5 `0e80fd17a307fa522eb87663821848f9`, sha256 `d2603b9c8bfe37c52ce5f444868bb31e235a9a22b7aa60228b083a2b0aca100e`; exactness `exact-package`; kapi_runs line 1315, pushed md5 `72b75238af56b4fc55a2d998b29c92b6`).
Rebuild rule: every non-empty code cell, CRLF -> LF, the patches below in this order (each must match exactly `count` times), cells joined by `\n# <<<FF-CELL>>>\n`; result sha256 `d956a0144620f92dca3c2decc6a2c7c2ab7a056f152cf3ea116b10e2b858fe74` (6 cells). Same rule as `release/flag2027_e2e/build_notebook.py` (checked: it reproduces `release/flag2027_e2e/code/t163b.ipy` byte for byte when given that package's two patches).

## A. Changes made when copying the exact notebook into `original/` (redaction)

1. **notebook cell 2 (0-based, markdown counted), `("train/gender.csv", ...)` literal** - type `redaction` (organisers' data).
   - old: gzip + base64 literal, 228 characters (sha256 of the literal `5af7e8b8d87b0d1d4aa4f59c495150e2f305b45236466f8f31930678578e372c`), decoding to 560 bytes with sha256 `fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030` = the organisers' `meta_file_train_set.csv` (70 lines `idNNN,f|m`).
   - new: the placeholder string `FF-REDACTED:organisers-file:meta_file_train_set.csv:560-bytes:sha256=fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030`.
   - why: the repo must not carry organisers' data (user rule 6); every FLAG_163-family notebook embeds this file as `train/gender.csv`.
   - effect on numbers: none (patches 'redaction' below remove the entry and, in `RFS_MODE=B` only, copy the same bytes from the organisers' file with a sha256 check; no other mode reads the file).

## B. Patches (`stage.json` -> `patches`, applied in this order)

2. **notebook cell 0 (0-based index in the notebook, markdown cells counted), line 5 of that cell** - type `upload`.
   - old (1 hit): `os.environ['RFS_UPLOAD'] = 'onnhqunhh/flag2027-rfs-lora-p'`
   - new: `# FF: upload disabled, the output folder becomes a runner artifact (CHANGES.md). Original: os.environ['RFS_UPLOAD'] = 'onnhqunhh/flag2027-rfs-lora-p'`
   - why: the original run uploaded its output folder to the Kaggle dataset onnhqunhh/flag2027-rfs-lora-p with kagglehub; here the runner keeps it as an artifact (cell 'upload' then sees RFS_UPLOAD unset and skips).
   - effect on numbers: none.

3. **notebook cell 4 (0-based index in the notebook, markdown cells counted), line 6 of that cell** - type `url`.
   - old (1 hit): `https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth`
   - new: `https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth`
   - why: Meta's ImageBind link is dead since 12/11/2026; pinned Hugging Face mirror of the same file (imagebind_huge.pth sha256 d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157).
   - effect on numbers: none.

4. **notebook cell 2 (0-based index in the notebook, markdown cells counted), line 8 of that cell** - type `redaction`.
   - old (1 hit): `("train/gender.csv", "FF-REDACTED:organisers-file:meta_file_train_set.csv:560-bytes:sha256=fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030"), `
   - new: ``
   - why: the embedded copy of the organisers' meta_file_train_set.csv is not distributed with the repo (user rule 6); its entry is removed from the embedded-file loop (the literal was already replaced by a placeholder in original/*.redacted.ipynb, see SOURCE.json).
   - effect on numbers: none.

5. **notebook cell 2 (0-based index in the notebook, markdown cells counted), line 9 of that cell** - type `redaction`.
   - old (1 hit): `    p = CODE / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(gzip.decompress(base64.b64decode(blob)))\n`
   - new: `    p = CODE / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(gzip.decompress(base64.b64decode(blob)))\nif MODE == "B":   # FF redaction (CHANGES.md): train/gender.csv was an embedded copy of the organisers' meta_file_train_set.csv; the same bytes come from the raw-v4 input (only assemble.py, RFS_MODE=B, reads the file)\n    _ff_g = [q for q in sorted(Path("/kaggle/input").rglob("... (843 chars)`
   - why: puts back train/gender.csv in RFS_MODE=B from the organisers' file itself (sha256 asserted = the embedded copy), so assemble.py reads the same 560 bytes; in modes L / P / LAB no code reads train/gender.csv (checked in every embedded rfs_train.py version), so nothing is written.
   - effect on numbers: none.

6. **notebook cell 4 (0-based index in the notebook, markdown cells counted), line 10 of that cell** - type `variant`.
   - old (1 hit): `        if Path(d).name in ("v1", "v2") and {"faces", "voices"} <= set(dirs): mav = Path(d).parent; break\n`
   - new: `        dirs.sort()   # FF determinism (CHANGES.md): visit folders in name order so mavceleb_v1_complete-001/v1 is found before mavceleb_v2_complete-002/v2, as in the original run (210 clips of the 70 MAV-Celeb v1 persons)\n        if Path(d).name in ("v1", "v2") and {"faces", "voices"} <= set(dirs): mav = Path(d).parent; break\n`
   - why: os.walk order depends on the file system; the original Kaggle run took the MAV-Celeb v1 parent folder (its LoRA C1 saw 210 clips of 70 persons = the 70 v1 identities x 3 videos; v2 has 84 identities), so every v2/... wav path failed to load and was skipped silently. Sorting reproduces that choice on any mount; the dry check below verifies the outcome.
   - effect on numbers: none if the original run found v1 first (evidence: 70 persons = MAV v1); otherwise this restores the original data selection.

7. **notebook cell 4 (0-based index in the notebook, markdown cells counted), line 18 of that cell** - type `variant`.
   - old (1 hit): `    run(["lora", "--bundle", str(BUNDLE), "--work", str(WORK), "--lid", LID, "--ib_ckpt", str(ck), "--cache", str(tars[0].parent.parent), "--mav", str(mav), "--... (180 chars)`
   - new: `    if not SMOKE:   # FF variant guard (CHANGES.md): dry run of lora_data before training, no weights touched\n        print("[cell] FF dry check: LoRA C1 clip list (expected 210 clips, 70 persons, all MAV-Celeb v1, 0 VoxCeleb2)", flush=True)\n        _ff_src = '''import collections, json, sys, types, importlib.util\ntrain, bundle, lid, cache, mav, out = sys.argv[1:7]\nspec = importlib.util.spec_f... (2109 chars)`
   - why: DESIGN 4.7 / G8: before training, run the original lora_data() (same code, same arguments, cap=None) in a separate process and require the original selection: 210 clips, 70 persons, all MAV-Celeb v1, 0 VoxCeleb2 (the VoxCeleb2 cache key bug of rfs_train.py line 406 is kept on purpose). Always runs, except with FLAG_SMOKE=1 (the notebook's synthetic path; the runner's smoke_env of this stage does not set it, so a runner smoke run is a real-data mini run and the dry check runs). Writes /kaggle/working/ff_lora_drycheck.json (not part of the exported dataset). Fixer 18/11: the FF_LORA_DRYCHECK=0 opt-out was removed (it skipped the file that the post_check requires).
   - effect on numbers: none (extra read-only pass; the training subprocess is unchanged).
   - modified: 18/11/2026 (fixer, review of WP-F): opt-out FF_LORA_DRYCHECK removed.

8. **notebook cell 3 (0-based index in the notebook, markdown cells counted), line 3 of that cell** - type `path`.
   - old (1 hit): `def find_all(name):\n    return sorted(p for p in IN.rglob(name)) if IN.exists() else []\n`
   - new: `def find_all(name):\n    return sorted(p for p in IN.rglob(name)) if IN.exists() else []\nif MODE in ("B", "L") and not find_all("train_English.txt"):   # FF path (CHANGES.md): FF_DATA keeps the organisers' train_set.zip packed (Kaggle served the dataset extracted); unpack train_set.zip only, next to itself; dev_set.zip is never opened\n    import zipfile as _ff_zf\n    _ff_tz = find_all("train_se... (734 chars)`
   - why: the original run read ngnnguynchn/flag2027-raw-v4-p as Kaggle served it, extracted (train_set/train_set/train_English.txt next to faces/ and voices/, train_set/features/{faces,voices}/*.csv). FF_DATA/00_raw/btc_mavceleb_v4/ keeps the organisers' files as delivered (train_set.zip, dev_set.zip, meta_file_train_set.csv) and the runner does not unzip raw data, so Q('train_English.txt') (mode B) / the os.walk for train_English.txt + faces/ (mode L) found nothing. When train_English.txt is absent, the block unpacks train_set.zip ONLY, beside itself inside the stage's own input view (real directories made by the runner; FF_DATA is not written), giving the Kaggle layout; dev_set.zip stays packed and is never opened (dev exclusion, DESIGN 2.8). It does nothing when an extracted view is mounted (Kaggle, or a WP-B extracted train-only view).
   - effect on numbers: none (same files as the extracted Kaggle view: members of the organisers' train_set.zip, sha256 1803739f... checked by raw_btc).
   - added: 18/11/2026 (fixer, review of WP-F).

9. **notebook cell 4 (0-based index in the notebook, markdown cells counted), line 6 of that cell** - type `url`.
   - old (1 hit): `        _t.hub.download_url_to_file("https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_hu... (240 chars)`
   - new: `        _t.hub.download_url_to_file("https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth", str(ck) + ".part", progress=False); os.replace(str(ck) + ".part", ck)\n    _ff_h = __import__("hashlib").sha256()   # FF url (CHANGES.md): the pinned mirror must serve Meta's imagebind_huge.pth byte for byte\n    with open(ck, "rb"... (685 chars)`
   - why: completes the url change above: the Hugging Face mirror replaces Meta's dead link, so the file is checked against the sha256 of Meta's imagebind_huge.pth (d6f6c22b..., the value release/docker/ff.py IB_SHA checks) before it is loaded into the LoRA C1 training; a cached W/imagebind_huge.pth is checked too (S4 section 3 item 12: 'kiem sha256 truoc khi nap').
   - effect on numbers: none (check only; stops the stage if the mirror served another file).
   - added: 18/11/2026 (fixer, review of WP-F).

10. **notebook cell 4 (0-based index in the notebook, markdown cells counted), line 20 of that cell** - type `variant`.
   - old (1 hit): `    for n in ("lora_C1.pt", "v4_extra.npz"): shutil.copy(WORK / n, OUTD / n)\n`
   - new: `    for n in ("lora_C1.pt", "v4_extra.npz"): shutil.copy(WORK / n, OUTD / n)\n    _ff_m = torch.load(OUTD / "lora_C1.pt", map_location="cpu")   # FF variant guard (CHANGES.md): training metadata of this C1 for the post_check (original file: 48 params, 1200 steps, batch 64, 210 clips, 70 persons, 0 skipped)\n    (W / "ff_lora_meta.json").write_text(json.dumps({"n_params": len(_ff_m["params"]), **{k... (485 chars)`
   - why: a re-trained C1 is not byte-identical (GPU), so its sha256 cannot be compared with b97372e3...; what can be compared is the metadata that rfs_train.py stores in the file (torch.save at line 438 of the embedded copy: params, steps, batch, clips, persons, skipped). The new post_check ff_lora_meta.json requires the values of the original file (read from its pickle on 18/11: 48 params, steps 1200, batch 64, clips 210, persons 70, skipped 0). Written to /kaggle/working, not into the exported dataset rfs_lora/.
   - effect on numbers: none (reads the saved file after training; nothing is changed).
   - added: 18/11/2026 (fixer, review of WP-F).

## C. Kept as in the original run (on purpose)

- Inputs: exactly the datasets / kernels the run attached (`stage.json` -> `inputs`, from the run's kernel metadata), including the ones attached but not read.
- Every other line of every cell, the embedded code (`original/decoded/`, byte-exact, CRLF kept) and the `[boot]` / `[cell]` lines that kapi inserted at push time.
- pip calls (`pip_policy: original`).
- The ORIGINAL LoRA C1 (`onnhqunhh/flag2027-rfs-lora-p`, `lora_C1.pt` sha256 `b97372e3...`); the substitutions to the fixed C1 made for the 16.18 system (`release/flag2027_e2e/stages.py:234-320, 526-527`) and the `lorafix1` defaults of today's scripts are NOT applied (see `../C1_VARIANTS.md`).
- The VoxCeleb2 cache-key bug of the embedded `rfs_train.py` (line 406): it is what the 13.54 LoRA C1 was trained with.

## D. Changes in `stage.json` outside the code (fixer, 18/11/2026, review of WP-F)

- `runner_requirements.literal_kaggle_input` = `false` (new key; why: mode L: rfs_train.py lora gets --bundle / --lid / --cache / --mav / --v4root / --ib_ckpt as arguments and the dry-check subprocess the same paths; the decoded rfs_train.py (98aa07ae) contains no /kaggle path (grep); all lookups are in the stage text).
- `force_only: true` (+ `force_only_why`): the stage runs only with FF_FORCE=t163l_orig; see ../README.md 'LoRA C1 policy'.
- `post_checks`: added `ff_lora_meta.json` = {n_params 48, steps 1200, batch 64, clips 210, persons 70, skipped 0} (patch 'variant' above writes it). `variant.outputs_file_checks`: exemption request for WP-A.
- Input substitution, type `wiring` (recorded in `notes` too): the bundle input `ngnnguynchn/flag2027-rfs-bundle-p` is the output of the repo's
  `t163b` = FLAG_163 B version 5 (`assemble.py` `b46dd180...`, `v4.npz` `f_xf` with 4 096 columns). The original L run (pushed 2026-11-04 00:37,
  `kaggle/kapi_runs.jsonl` line 1315) read the earlier bundle of B version 4 (B pushed 00:31, md5 `3e9eef0c...`; built by `assemble.py`
  `432bdbf9...`, the copy embedded in the L notebook, which keeps the 4 097 / 193-column CSVs); B was re-pushed at 02:10 (line 1327, md5
  `d2ed4036...`). The two `assemble.py` differ only in lines 71-74 (the `rd()` column cut of the v4 CSVs, i.e. `v4.npz` `f_xf` / `v_xv`).
  Effect on numbers: none. Mode L reads `ext.npz` (`lora_data`, `face_by_person`; the same code and inputs in both versions), `V4["v_path"]`
  (for `v_lora`) and the presence of `f_age` (present in both versions, so the age branch of line 441 is not taken); it never reads `f_xf` /
  `v_xv`.
- `notes`: runner smoke of this stage = real-data mini run (FLAG_SMOKE not set; the dry check runs; the `ff_lora_meta.json` post_check fails by
  construction with 3 steps, not fatal in smoke).
