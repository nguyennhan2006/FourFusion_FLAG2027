# CHANGES: `t163b` (ngnnguynchn/flag-2027-nb-163-rfs-b)

Source of `stage.ipy`: `original/14_flag-2027-nb-163-rfs-b.redacted.ipynb`, a redacted copy of the exact notebook `E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/1_train/14_flag-2027-nb-163-rfs-b.ipynb` (md5 `4fd83ff36e12b0d36eb2ee6e22c10866`, sha256 `50c147da0094dd8bd6fb4f51cf8387f1d9ad9718e0bad64617a8c4fe04865df7`; exactness `exact-package`; kapi_runs line 1327, pushed md5 `d2ed403618c620d6f253981b43cbdbf5`).
Rebuild rule: every non-empty code cell, CRLF -> LF, the patches below in this order (each must match exactly `count` times), cells joined by `\n# <<<FF-CELL>>>\n`; result sha256 `09ab431d415b3338cd3c5d36bd9d09f5ed83a0ba3f6aa8a596b49245324df8d8` (6 cells). Same rule as `release/flag2027_e2e/build_notebook.py` (checked: it reproduces `release/flag2027_e2e/code/t163b.ipy` byte for byte when given that package's two patches).

## A. Changes made when copying the exact notebook into `original/` (redaction)

1. **notebook cell 2 (0-based, markdown counted), `("train/gender.csv", ...)` literal** - type `redaction` (organisers' data).
   - old: gzip + base64 literal, 228 characters (sha256 of the literal `5af7e8b8d87b0d1d4aa4f59c495150e2f305b45236466f8f31930678578e372c`), decoding to 560 bytes with sha256 `fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030` = the organisers' `meta_file_train_set.csv` (70 lines `idNNN,f|m`).
   - new: the placeholder string `FF-REDACTED:organisers-file:meta_file_train_set.csv:560-bytes:sha256=fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030`.
   - why: the repo must not carry organisers' data (user rule 6); every FLAG_163-family notebook embeds this file as `train/gender.csv`.
   - effect on numbers: none (patches 'redaction' below remove the entry and, in `RFS_MODE=B` only, copy the same bytes from the organisers' file with a sha256 check; no other mode reads the file).

## B. Patches (`stage.json` -> `patches`, applied in this order)

2. **notebook cell 0 (0-based index in the notebook, markdown cells counted), line 5 of that cell** - type `upload`.
   - old (1 hit): `os.environ['RFS_UPLOAD'] = 'ngnnguynchn/flag2027-rfs-bundle-p'`
   - new: `# FF: upload disabled, the output folder becomes a runner artifact (CHANGES.md). Original: os.environ['RFS_UPLOAD'] = 'ngnnguynchn/flag2027-rfs-bundle-p'`
   - why: the original run uploaded its output folder to the Kaggle dataset ngnnguynchn/flag2027-rfs-bundle-p with kagglehub; here the runner keeps it as an artifact (cell 'upload' then sees RFS_UPLOAD unset and skips).
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

6. **notebook cell 3 (0-based index in the notebook, markdown cells counted), line 3 of that cell** - type `path`.
   - old (1 hit): `def find_all(name):\n    return sorted(p for p in IN.rglob(name)) if IN.exists() else []\n`
   - new: `def find_all(name):\n    return sorted(p for p in IN.rglob(name)) if IN.exists() else []\nif MODE in ("B", "L") and not find_all("train_English.txt"):   # FF path (CHANGES.md): FF_DATA keeps the organisers' train_set.zip packed (Kaggle served the dataset extracted); unpack train_set.zip only, next to itself; dev_set.zip is never opened\n    import zipfile as _ff_zf\n    _ff_tz = find_all("train_se... (734 chars)`
   - why: the original run read ngnnguynchn/flag2027-raw-v4-p as Kaggle served it, extracted (train_set/train_set/train_English.txt next to faces/ and voices/, train_set/features/{faces,voices}/*.csv). FF_DATA/00_raw/btc_mavceleb_v4/ keeps the organisers' files as delivered (train_set.zip, dev_set.zip, meta_file_train_set.csv) and the runner does not unzip raw data, so Q('train_English.txt') (mode B) / the os.walk for train_English.txt + faces/ (mode L) found nothing. When train_English.txt is absent, the block unpacks train_set.zip ONLY, beside itself inside the stage's own input view (real directories made by the runner; FF_DATA is not written), giving the Kaggle layout; dev_set.zip stays packed and is never opened (dev exclusion, DESIGN 2.8). It does nothing when an extracted view is mounted (Kaggle, or a WP-B extracted train-only view).
   - effect on numbers: none (same files as the extracted Kaggle view: members of the organisers' train_set.zip, sha256 1803739f... checked by raw_btc).
   - added: 18/11/2026 (fixer, review of WP-F).

## C. Kept as in the original run (on purpose)

- Inputs: exactly the datasets / kernels the run attached (`stage.json` -> `inputs`, from the run's kernel metadata), including the ones attached but not read.
- Every other line of every cell, the embedded code (`original/decoded/`, byte-exact, CRLF kept) and the `[boot]` / `[cell]` lines that kapi inserted at push time.
- pip calls (`pip_policy: original`).

## D. Changes in `stage.json` outside the code (fixer, 18/11/2026, review of WP-F)

- `runner_requirements.literal_kaggle_input` = `false` (new key; why: mode B: assemble.py receives every path as an argument (--rfs_a ... --rawv4 --gender_csv); the decoded assemble.py / rfs_train.py (98aa07ae) contain no /kaggle path (grep); all lookups are in the stage text, which the runner rewrites).
- `notes` / `todo`: raw-v4 unpack (patch 'path' above) replaces the requirement 'raw-v4 mount laid out as on Kaggle'.
