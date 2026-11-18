# FourFusion — FLAG 2027 reproduction repository (Evaluation 13.54)

Code to rebuild the Evaluation submission of team **FourFusion** to the FLAG 2027 Grand Challenge (ICASSP 2027: does a face image
and a voice clip belong to the same person; MAV-Celeb v4).

| target | value |
|---|---|
| displayed submission | CodaBench submission **1054704** (user `minhnhien2811`) |
| file | `xjf_r10_trsig.zip`, sha256 `b4354eaf988b2b4ea8c4e1c3fa44d7bcf631ea5cb3cd635ab2965021fe82e507` |
| Evaluation EER | **13.54** = 10.20 (no_gender/English) · 11.97 (no_gender/Bangla) · 12.98 (gender/English) · 18.99 (gender/Bangla) |

**Team FourFusion**, University of Information Technology, VNU-HCM: Nguyen Chon Nhan, Dang Minh Nhien, Vu Nguyen Anh Dung, Nguyen
Huu Bao Loc; advisor Nguyen Duy Tam Anh.

**Data and checkpoints** (Google Drive folder `FourFusion_FLAG2027_data/`, shared with the organisers only because it holds their
data and VoxBlink-derived material): link to be added by the team. Open items: [`docs/TODO_TEAM.md`](docs/TODO_TEAM.md).

Read also: [`docs/COMPLIANCE_NOTES.md`](docs/COMPLIANCE_NOTES.md) (rules, dev data, Bengali voice, and the disclosure about the DP
branch), [`docs/VERIFICATION.md`](docs/VERIFICATION.md) (what has and has not been checked), [`DATA.md`](DATA.md),
[`CHECKPOINTS.md`](CHECKPOINTS.md), [`LICENSES.md`](LICENSES.md), [`docs/NAMING.md`](docs/NAMING.md) (internal names).

## 1. The system

Each file of the zip stores −score per trial (lower = same person, as the organisers' scorer expects).

| file | value | how it is produced |
|---|---|---|
| `no_gender/sub_score_v4_Bangla_unheard.txt` | −(logit F1s′ + logit RFX) | FACTOR_ENFUSE (c5rl), member sha256 `36056e7e…` |
| `gender/sub_score_v4_Bangla_unheard.txt` | −(logit F1s′ + logit RFX′) | FVBIND_ENFUSE (fvb), member `9a619736…` |
| `no_gender/…English_heard.txt`, `gender/…English_heard.txt` | (FACTOR_ENFUSE English score) / sigma[protocol] − 1.0 × z_DP, sigma = 5.0983 / 3.8376 | final blend, members `d45f8502…`, `ead7e177…` |

- **F1s′**: identity-classification family (face LoRA adapters on CLIP ViT-L/14, FaRL, frozen speaker encoders; CA head, XALL bridge,
  T3 v11 head) fused by a logistic mixer fitted on the team's SAFE-04 validation tables (train persons only).
- **RFX**: pair-contrastive family (RFS recipe retrained on the team's cleaned data) with the extra blocks `f_c5s` and `v_rl`; it
  contains the ImageBind-audio LoRA **C1 in its original version** (cache-key bug: 210 MAV-Celeb v1 clips of 70 persons).
  **RFX′** adds the voice block B2 of FVBIND-01 and is used only for the gender/Bangla file.
- **DP (Different Profile)** branch, English files only: two towers × 10 seeds on face / voice embeddings (ArcFace, AdaFace, ECAPA),
  trained on the FLAG train set and MAV-Celeb v1 / v2; sigma are the standard deviations of the base score on held-out train trials.
  See the disclosure in `docs/COMPLIANCE_NOTES.md` §5.
- Variant guard: `docs/VARIANTS.md` (the 16.18 system with a fixed C1 and other close systems are refused by the runner).

## 2. Repository layout

```
runner/        ffrun.py (command line), ffcore.py (engine), check_stage.py, verify.py, build_kaggle_notebook.py, selftest.py
configs/       data_map.json (artifact id -> FF_DATA path), modes.json, variant_1354.json, pipeline_1354.json, expected_sha256.json
stages/        91 stages in 11 groups (10_raw ... 80_final; + 2 environment stages in tools/stages/00_env); each: stage.json,
               stage.ipy (cells of the run), CHANGES.md, original/
assembly/      mode (c): assemble_exact.py + expected.json; inputs/ (empty in the public repository)
notebooks/     Kaggle orchestration notebooks FourFusion_1354_{infer,train,assemble}.ipynb (generated)
docker/        environment specification (Kaggle base image digests, pinned pip sets, reference Dockerfile; Docker is not required)
tools/         fetch_models.py (pinned pretrained models), drive_manifest.py, secret_scan.py, build_docs.py, PULL_MISSING_RUNS.md
src/           read-only copies of the team's builders and modules (SRC_MANIFEST.json); not needed to run
docs/          STAGES.md, VERIFICATION.md, COMPLIANCE_NOTES.md, VARIANTS.md, NAMING.md, CHANGES_FROM_ORIGINAL.md
```

Every stage keeps the **code that actually ran** (the executed Kaggle notebook, or the local script), copied verbatim; the only
changes are listed with their reason in the stage's `CHANGES.md` (merged in `docs/CHANGES_FROM_ORIGINAL.md`), plus the run-time
rewrites of the runner (`runner/RUNTIME_CHANGES.md`: `/kaggle/input` → the stage's own input view, upload switches off, …).

## 3. Running

All three modes use one data root, `FF_DATA`, with the layout of the team's Drive folder (`DATA.md`; shared with the organisers),
and one work folder, `FF_WORK`. The runner itself needs only a recent Python 3 (the team used 3.13 / 3.14); the stages need the
Kaggle image environment (`docker/README.md`; `FF_PIP=pinned` applies the pinned package set). `python runner/ffrun.py plan ...` shows what would run and where every input comes from, without data.

### (c) Byte-exact zip from the stored per-pair scores — CPU, under a minute

```bash
python assembly/assemble_exact.py --inputs <FF_DATA>/05_scores --out /tmp/ff_c
# or through the runner (stages a_factor, a_fvbind, dp_sigma, final_blend, final_verify)
FF_DATA=<FF_DATA> FF_WORK=<work> python runner/ffrun.py run --mode assemble
python runner/verify.py zip <work>/art/notebooks/ff/final_blend/zips/xjf_r10_trsig.zip --expected xjf_r10_trsig
```

Expected: the four member files have the sha256 of `assembly/expected.json` (the whole-zip sha256 differs because the zip header
stores the write time). Inputs: the ten files of `assembly/inputs/README.md` (per-pair scores of our models, our fuser weights,
the DP scores and the sigma source table).

### (b) Inference from the organisers' Evaluation media + released checkpoints — about 12 GPU-hours (T4) + 1 CPU-hour

```bash
FF_DATA=<FF_DATA> FF_WORK=<work> python runner/ffrun.py plan --mode infer
FF_DATA=<FF_DATA> FF_WORK=<work> python runner/ffrun.py run  --mode infer          # 19 stages
```

Starts from `evaluation_phase.zip` (stage `raw_evalphase`), extracts features of the four Evaluation lists with the saved
checkpoints, scores with F1s′ / RFX / RFX′ / DP and assembles the zip. Expected: per-file Spearman ≥ 0.9999 against the submitted
zip (GPU arithmetic is not bit-identical). Needs the checkpoints of `CHECKPOINTS.md` and some train-side intermediate tables
(`DATA.md`), because the verbatim code computes train-side constants inside the runs.

### (a) Rebuild from the raw and collected data — about 150–170 GPU-hours (T4) + 10 CPU-hours

```bash
# default: from raw / collected data, but the stored LID outputs, the frozen SAFE-04 tables and the original LoRA C1 are taken saved
FF_DATA=<FF_DATA> FF_WORK=<work> python runner/ffrun.py run --mode train --only <stage,...>     # one session per call
# widest rebuild (configs/modes.json -> full_rebuild_from_raw): 71 stages in the plan
FF_USE_SAVED=none FF_FORCE=raw_mavceleb,safe04_lab,f14_av,gather_g1 FF_DATA=<FF_DATA> FF_WORK=<work> \
    python runner/ffrun.py run --mode train --only <stage,...>
# add the collect stages to FF_FORCE to re-collect VoxCeleb2 / VoxBlink from Hugging Face (79 stages; needs HF_TOKEN)
```

Expected: a recipe-level reproduction, not byte-identical (GPU training is not deterministic; VoxCeleb2 sampling depends on the
network, so 13.54 uses the collected copies). The original LoRA C1 is always the stored one: it defines the 13.54 variant. A small set
of frozen artifacts has no producing stage in this repository (wiring datasets checked only for presence by the verbatim code, the
frozen SAFE-04 tables, a few embedded files): `docs/VERIFICATION.md` §4 lists them with their role.

### On Kaggle

Use `notebooks/FourFusion_1354_{assemble,infer,train}.ipynb` (built by `runner/build_kaggle_notebook.py`): attach the code bundle
and the Drive-layout data, set `FF_ONLY` per session (a session lasts at most 12 hours). Stages whose decoded code reads the literal
`/kaggle/input` need a session with exactly their inputs attached (`runner/README.md`).

## 4. Hardware and time (original runs)

GPU stages ran on Kaggle with NVIDIA Tesla T4 (one or two per run); small steps ran on Kaggle CPU sessions or on the team's CPU
machine. Per-stage minutes are in `docs/STAGES.md`. Mode (a) is about 45–55 hours of wall time on the critical path when enough
sessions run in parallel.

## 5. Status and what is still missing (19/11/2026)

- **L0 checks pass**: all 93 stages rebuild from their original notebooks / scripts (`python runner/check_stage.py`), the stage graph
  has no cycle or unresolved input (`python runner/ffrun.py dag`).
- **Mode (c)**: logic verified on the team machine (the four member files are reproduced byte for byte by the same scripts); the ten
  input files are not yet in `FF_DATA/05_scores/` (team decision on public / private repository).
- **Mode (b)**: not run end to end. Before it can finish, the team must (i) download two artifacts that exist only on Kaggle
  (`thanhhuyen11/flag2027-rfsfxfvb-art-p`, output of `thanhhuyen11/flag-2027-nb-192-fvbx3`; commands in
  `tools/PULL_MISSING_RUNS.md`), (ii) run `dp_train` once on its machine to save the DP towers (`dp_towers.pt`, ~12 CPU minutes), and
  (iii) upload the DP feature arrays and the checkpoints to the Drive folder.
  The FVBIND stages (`t189_b2`, `t192_fvbx`) unpack the organisers' `train_set.zip` in their input view when no extracted copy
  is mounted (fixed 19/11 after the second integration review; `dev_set.zip` is never opened).
- **Mode (a)**: never run.

## 6. Compliance in one paragraph

No organisers' dev data is used for training, fitting or selection; no score normalisation, cohort, CSLS or test-time adaptation;
Bengali voices are removed from every voice source by person (version-2 rule); pretrained encoders are frozen and declared;
VoxBlink material and the organisers' data are never in this repository. The English gain of 13.54 comes from the DP branch, which
trains on MAV-Celeb v1 / v2; the team's own studies suggest the gain comes from recordings shared between the English Evaluation file
and MAV-Celeb v1 / v2, and the organisers have not been asked yet. Details and the submission history: `docs/COMPLIANCE_NOTES.md`.
