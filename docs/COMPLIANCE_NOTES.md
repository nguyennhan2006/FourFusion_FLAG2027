# Compliance notes (FLAG 2027 rules, organisers' FAQ)

What the 13.54 chain does with respect to the challenge rules, and what the team declares. Stage-level evidence is in each stage's
`stage.json` / `CHANGES.md`; this file collects it. Public names follow `docs/NAMING.md` (the English branch is "DP (Different
Profile)").

## 1. Organisers' dev data

- **No stage trains, fits or selects on dev data.** The feature runs that originally also extracted dev media are restricted to the
  train split: `SPLIT_FILTER = ["train"]` / `TRAIN_ONLY` patches (f10, f12, f14, f40, f47b, t160a, t162a), `n_crops=0` for the optional
  FLAG_04 reference (its crop lengths had been chosen from dev Bangla clip lengths), the dev embeddings
  `ngnnguynchn/flag2027-rf092-dev-emb-p` are dropped from `t162a` and refused by the variant guard, and `r07_overlap` has a
  `--train-only` flag. Each change is typed `dev-exclusion` in the stage's `CHANGES.md`.
- **Mount filter.** 14 stages of `20_features` declare `exclude: ["dev_set.zip", "dev_set"]` on the organisers' raw data
  (`datasets/ngnnguynchn/flag2027-raw-v4-p`); since 18/11 the runner honours this per-input filter for every source
  (`runner/RUNTIME_CHANGES.md` #15). `FF_NO_DEV=1` keeps every listed dev file out of every stage.
- **Artifacts that still contain dev files** (flag `dev_data` in `configs/data_map.json`): the saved copies of
  `ngnnguynchn/flag2027-e2e-dev-p` (frozen SAFE-04 bundle; its dev-phase members are also registered as
  `ff-frozen/flag2027-e2e-dev-wiring-p`), the face LoRA datasets `phanvnhtrng/flag2027-feats-ls-b3s-p` and
  `quynhnhu82/flag2027-feats-ls-c5s-p` (their saved copies hold face arrays of the dev lists; a train-only rebuild has none),
  `cleancomp-e2e-p`, `cleancomp-final-p` (dev per-pair T3 streams), `evalu-f0s-p`, `evalu-f1s-p`, and the saved outputs of
  `112s-f0s`, `112f0sx`, `112f1sx` and `219-fusenet-dumpv4` (its `lab/` folder). They are mounted only because the verbatim code checks
  for these files or because the same dataset also carries train-side tables. The dev files do not enter any number of 13.54. They
  are stored only in the private data folder (`FF_DATA`), never in this repository.
- **Disclosure carried from the old systems.** The XALL bridge `s007_xall.pt` of F1s′ follows the "s007" recipe, which was chosen per
  cell with dev pseudo-labels in September (FUSE-01 / SEL-01b); the bridge itself was retrained on train / clean data for 13.54
  (`stages/50_eval/e50cc/CHANGES.md`, row "dev-selection disclosure").

## 2. Bengali voice

- **Rule applied to every voice the chain learns from (version 2, "LID_STRICT", 2/11/2026):** a clip is flagged when its top
  language is Bengali or Assamese, or p(bn) + p(as) ≥ 0.2 (VoxLingua107); every clip of a flagged person is dropped (by person). Lists:
  `lid_strict.json` (sha256 `f4065711…`; VoxCeleb2 608 and VoxBlink 1 637 persons, plus 58 MAV-Celeb voice files flagged by
  Whisper-large-v3), stage `l_lid_strict`. `BN42=1` applies the earlier FLAG_42 Bengali list to every row source. Faces of these
  persons may be used; their voices are not.
- **VoxBlink** persons with `spk2lan = bengali` / Bangladesh were removed before download (stage `c46_voxblink`).
- **Leak fixed before the 13.54 runs.** On 3/11/2026 12:15 a Bengali-voice leak in the training rows (`VOICELORA-01` `vb_rows_vl`
  without `strict_drop("vb")`) was fixed; the F1s′ runs of this chain embed the fixed file (md5 `6d0dfe2bdf72d2204b32dd1285129207`)
  and log `A0 PASSED` (e.g. `stages/30_F_family/t112sf1s/CHANGES.md`).
- **Pretrained encoders** are used frozen, as allowed by the organisers' answer of 17/10 (even when their pre-training data contain
  Bengali); they are listed in `LICENSES.md` and must be declared in the system description.
- The 58 MAV-Celeb files flagged "Bengali" were later listened to (LISTEN-01): none holds readable Bengali speech; dropping them is
  harmless and kept.

## 3. Score normalisation, cohorts, test-time adaptation

- `NOREL=1` in every run of the chain: the streams `rel` / `relg` / `t3_relg` (Z-norm against train anchor persons) are not used, as
  the FAQ forbids cohort normalisation even with train-only cohorts (NOREL-01).
- No CSLS, no test-time adaptation, no statistic of a test file, no use of the trial lists to choose or mix systems. Every score of a
  pair depends only on that pair and on constants fixed from train / external data.
- The DP scale constants sigma = 5.0983 / 3.8376 are the standard deviations of the base score on held-out **train** trials (stage
  `dp_sigma`, from `xjfdump_v4.parquet`). The language-specific mixers (English vs Bangla) are fixed before the test, as allowed.
- Batched feature extraction (disclosed for completeness): the DP branch computes the ECAPA voice embeddings of the Evaluation clips
  in zero-padded batches packed by duration (`stages/70_dp/dp_extract_eval/flag_extract.py` `run_batched`, at most 240 padded
  seconds / 32 clips per batch; relative lengths `wav_lens` are passed to SpeechBrain, so padding is masked in the feature
  normalisation and pooling). A clip's embedding can therefore differ in the last digits with the other clips of its batch. No label,
  trial list or statistic of a test file enters, and the clip order is the organisers' list order. Kept verbatim.

## 4. Identity overlap list

The F1s′ family drops two MAV-Celeb identities that are also persons of the organisers' v4 train set (constant `OVERLAP` in the
embedded code). Making that list links identities across MAV-Celeb and the organisers' data, so it is disclosed here. Stage `r07_overlap`
(force_only, `FF_FORCE=r07_overlap`) shows how it was derived and keeps no linkage table. The list is part of the F1s′ family code (the SAFE-04
lab code, also embedded in the stage `t219_dumpv4`).

## 5. DP (Different Profile) branch: training data and risk (must be in the system description)

The data flow of the DP branch follows the letter of the rules: its external data are declared, no dev or Evaluation data enter
training, and its constants are fixed on train persons. It trains on the organisers' FLAG train set (70 persons, 6 485 pairs) and on
MAV-Celeb v1 / v2 (154 persons), with the 58 Bengali-flagged MAV-Celeb voice files dropped (stage `dp_train`). The team has identified
the following risk and history (text of `stages/70_dp/dp_train/CHANGES.md`, "Compliance disclosure"):

1. **The branch is weak on unseen persons.** On held-out FLAG train persons the DP branch alone reaches only EER 25.67 (no_gender) and
   34.64 (gender); the base system reaches about 15 / 22 on the same kind of persons. Yet adding the branch lowered the English
   Evaluation EERs from 13.78 / 22.12 (FACTOR_ENFUSE alone) to 10.20 / 12.98.
2. **Likely cause: recording overlap.** The most plausible explanation is that some recordings of the English Evaluation file are also
   in MAV-Celeb v1 / v2, on which the branch trains. A team study that simulated overlap on held-out data (DPSTUDY-01) found a large
   gain of the fused system (+17.6 EER points on the MAV-Celeb English same-gender cell) only when the same video / recording is in
   training; when only the person overlaps through other videos, the fusion costs about 6 EER points. The team has not checked the
   overlap directly, because that requires matching identities across datasets, which the rules forbid.
3. **Held-out test of the fusion.** A test written before running it, which applied the same fusion to held-out train persons, made
   the system worse (v4 gender −5.65, v4 no_gender −4.91 EER points, better in 0 of 5 folds).
4. **Grey zone.** The team regards the use of MAV-Celeb v1 / v2 as training data of this branch as a question for the organisers to
   decide.
5. **Submission order.** `xjf_r10.zip` was submitted first: its two scale constants, 5.8 / 2.9, had been read off the score scale of
   the test files, which the rules do not allow. It scored 13.54 (16/11 10:31, CodaBench 1054532). The constants were then replaced by
   values measured on held-out train persons (5.0983 / 3.8376); `xjf_r10_trsig.zip` was built (11:08) and submitted (12:02, CodaBench
   1054704, the displayed submission) and also scored 13.54 on average. The team made ten Evaluation submissions in all; choosing among
   them with Evaluation scores runs against FAQ question 11.

## 6. LoRA C1: the true training statement

The ImageBind-audio LoRA C1 of 13.54 learned **210 MAV-Celeb v1 clips of 70 persons**; because of a cache-key bug no VoxCeleb2 clip
was matched (`stages/40_R_family/C1_VARIANTS.md`). The field `declared` of the RFX artifact's `manifest_final.json` ("cleaned
VoxCeleb2 + MAV clips") is wrong for this variant; the statement above is the true one.

## 7. External data

| source | use | access in this repository |
|---|---|---|
| MAV-Celeb v4 (organisers: train set, Evaluation media) | train / evaluation | `FF_DATA/00_raw/` (private Drive folder), never in git |
| MAV-Celeb v1 / v2 (public release) | extra persons for every family and the DP branch | `FF_DATA/00_raw/mavceleb_v1_v2*` |
| VoxCeleb2 (Hugging Face mirrors `gaunernst/voxceleb2-dev-wds`, `ProgramComputer/voxceleb`) | extra persons (faces + voices), Bengali-filtered | collected copies in `FF_DATA/01_collected/` (collection code in `stages/11_collect/`) |
| VoxBlink (Hugging Face mirror `adollahamini1998/VoxBlink-Video-clean-8-*`; licence CC BY-NC-SA) | extra persons, Bengali-filtered | private only (`FF_DATA/01_collected/`, shared with the organisers' e-mail); no VoxBlink media, feature or weight is in this repository |
| pretrained models | see `LICENSES.md` | `FF_DATA/06_pretrained/` via `tools/fetch_models.py` |

AVSpeech and LRS3 are **not** used by the 13.54 chain (team rule of 28/10; the G3 quality layer was rebuilt without them, stage
`l_dqual_g3`).

## 8. What stays outside this repository

Organisers' data (train / dev / Evaluation media and lists, the meta file), dev score zips, VoxBlink material, credentials, and the
byte-exact Kaggle push folders that embed some of them (`_staging_backup/`, kept by the team; the `original/` folders here hold
redacted copies checked against them). The tree was scanned for secrets before publication; the root `.gitignore` refuses
data, checkpoints and the backup folder.
