# CHANGES: t153b (exact-package; executed notebook `03_flag-2027-nb-153b-b3s.ipynb`)

## 1. Upload to a Kaggle dataset disabled (type `upload`; numbers: none)

- **Where:** code cell 1 (the run-parameter cell inserted by kapi --env), line `os.environ['FL3_UPLOAD'] = 'phanvnhtrng/flag2027-feats-ls-b3s-p'`.
- **Old → new:** the line becomes the comment `# ff: no upload to a Kaggle dataset (the stage output becomes a runner artifact). Original: os.environ['FL3_UPLOAD'] = 'phanvnhtrng/flag2027-feats-ls-b3s-p'`
  (patch `count` = 1). `neutralize_upload: ["FL3_UPLOAD"]` also tells the runner to keep the variable unset.
- **Why:** the run pushed its arrays to the private dataset `phanvnhtrng/flag2027-feats-ls-b3s-p` with kagglehub (cell 14: `UP = os.environ.get("FL3_UPLOAD", "")`;
  empty → `no upload`). In the reproduction the stage output becomes the artifact `datasets/phanvnhtrng/flag2027-feats-ls-b3s-p` (`exports`, the 13 files of the original dataset).
- **Effect:** none (cell 14 only uploads; it never changes an array). Same replacement as the release stage `t153b`
  (release/flag2027_e2e/stages.py `noup`), only the comment text is English.

## 2. Train-only: no dev list is embedded (type `dev-exclusion`; numbers: none) - 2026-11-18 fix

- **Where:** code cell 2 (settings), two lines; patches 2 and 3 of `stage.json` (same text as WP-D's `f47b_screen2`).
- **Old → new:** `SPLIT_FILTER = None` → `SPLIT_FILTER = ["train"]`; `ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")` → `("train_set", "eval", "test")` (dev_set.zip is never unpacked).
- **Why:** DESIGN 2.8 (every stage of the chain reads only the train set). The executed run embedded the four dev lists with the trained adapter (log: gender/Bangla 1468, no_gender/Bangla 1406, gender/English 982, no_gender/English 1008 rows); their only readers were the dev scoring steps of t112sf0s / t112sf1s, which are now skipped by default (FF_TRAIN_ONLY, their CHANGES.md). e65xbs / e65xcs copy only the non-dev files of this artifact (stages/50_eval/e65xbs/stage.ipy line 7009).
- **Effect:** none on the adapter (trained on VoxCeleb2 + VoxBlink faces only, before any list is embedded) and none on the train / MAV-Celeb / VoxCeleb2 / VoxBlink arrays (each split is embedded on its own). The export loses the four dev-list arrays `face_clipB3_{gender,no_gender}_{English,Bangla}.npy` (the original dataset had 13 members). `self_check()` takes its six batching / fp16 check images from the train list instead of the first dev list (a numeric self-test only). To restore the executed behaviour, revert both patches (and run t112s* with FF_TRAIN_ONLY=0).

## 3. Organiser data layout (type `path`; numbers: none expected, unverified)

- On Kaggle the organiser zips were auto-extracted (log: `.../flag2027-raw-v4-p/train_set/train_set/train_set/train_English.txt`). FF_DATA holds `00_raw/btc_mavceleb_v4/train_set.zip` / `dev_set.zip`, so `find_splits` finds no list and the notebook's own `unpack_zips()` fallback extracts train_set.zip into WORKDIR/raw (removed at the end, cell 12). No code change; the fallback was never exercised by a recorded run: the first smoke run must show the split `train` with 6485 rows (`stage.json` verify).

## 4. Not changed, on purpose

- Internet downloads of cell 06 / cell 03 (FaRL ep16 GitHub release, ModelScope WeSpeaker zips, `pip install git+https://github.com/openai/CLIP.git@d50d76d...`,
  HF `openai/clip-vit-large-patch14@32bd642`): kept verbatim (`compute.internet = true`); offline route in `stage.json` `flags.offline_option`.
- `!pip -q install ftfy regex` (cell 03) is unpinned in the executed notebook and kept verbatim (`pip_policy: original`); the versions must come from the environment specification (`stage.json` `pip_note`).
- Licence: the exports hold VoxBlink face features (`vb_face_clipB3.npy`) and an adapter trained on VoxBlink frames (CC BY-NC-SA 4.0): private FF_DATA only, never in git (`stage.json` `licence`).

## 5. Runner

Runner-level only (not a change of this stage's code): the runner replaces the strings /kaggle/input and /kaggle/working by the stage's own input view and working folder, and prints a `[cell] <stage> k/n` line before each cell (DESIGN 2.4).
