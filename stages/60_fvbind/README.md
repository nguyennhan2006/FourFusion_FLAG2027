# Group 60_fvbind: FVBIND voice block B2, RFX′ and the Bangla same-gender file

## What this group contributes

The displayed Evaluation submission of team FourFusion (CodaBench submission 1054704, Evaluation EER 13.54 =
10.20 · 11.97 · 12.98 · 18.99; zip file `xjf_r10_trsig.zip`, sha256 `b4354eaf988b2b4ea8c4e1c3fa44d7bcf631ea5cb3cd635ab2965021fe82e507`)
holds four score files. This group produces exactly one of them:

| file of the final zip | sha256 | value per pair |
|---|---|---|
| `gender/sub_score_v4_Bangla_unheard.txt` (716 pairs, EER 18.99) | `9a6197360fa2c1ddb8c7039730df67b3ace221a635a10abfe51f3ec72c38f392` | −(logit F1s′ + logit RFX′), pair order of F1s′, printed `%.8f` |

- **F1s′** is the identity-classification family (group 30 / 50, file `f1s_eval.zip` of stage `e112f1sx`).
- **RFX′** is the pair-contrastive family RFX (c5rl configuration) with one more voice block, `v_fvb`, made by this group:
  `v_fvb = head(ImageBind_B2(voice)) − μ_v`, where B2 is the ImageBind-huge audio tower with a rank-8 LoRA and a linear head 1024 → 1280,
  trained with symmetric InfoNCE against a frozen per-person face anchor [CLIP-L + C5s adapter ‖ FaRL ep16] (FVBIND-01, arm B2).
- The final blend (group 80, stage `final_blend`) copies this member unchanged from `posteval_FVBIND_ENFUSE_fvb.zip`, the output of stage
  `a_fvbind`. The other Bangla file of the final zip comes from FACTOR_ENFUSE (group 50); the two English files come from the
  DP (Different Profile) branch blended with FACTOR_ENFUSE (groups 70 / 80).

## How this file was selected (declaration)

This file did not enter the 13.54 zip through our pre-registered validation rule, and we declare how it was chosen:

- FVBIND failed its pre-registered validation gate 2 (RFX′ against RFX on the same-gender cells of our SAFE-04 validation folds:
  −0.39, 1 of 5 folds; `t163p_fvb/original/evidence/gate2_report_194/fvbg2_report.txt`, line 15:
  `GATE 2: FAIL -> keep FACTOR_ENFUSE, paper only`).
- `posteval_FVBIND_ENFUSE_fvb.zip` was nevertheless submitted to the Evaluation phase (CodaBench submission 1053013, 15/11/2026,
  cell EERs 14.29 · 12.22 · 22.12 · 18.99).
- The team then composed a zip cell by cell from Evaluation scores already measured (reference BEST4, stage `a_best4_ref`): the
  same-gender Bangla file was taken from FVBIND ENFUSE because its measured EER on that cell (18.99) was lower than that of FACTOR_ENFUSE
  (20.11) (`a_best4_ref/original/make_probes.py`: `KNOWN` at lines 36-38, `BEST_NOW` at line 112). The 13.54 zip keeps this file.
- Choosing a file by its Evaluation score is against question 11 of the organisers' FAQ (choose the system before submitting; no
  selection or tuning on Evaluation scores). See also `docs/COMPLIANCE_NOTES.md`.

## Stages

| id | what | original Kaggle run (exactness) | device, time of the run | output artifact | saved copy |
|---|---|---|---|---|---|
| `t189_b2` | train the B2 voice block (584 steps, 2 048 persons per step, GradCache) | `thanhhuyen11/flag-2027-nb-189-fvbind-b2-ib0` (exact-staging, md5 `4a5e4194…`) | T4, about 6.3 h | `fvb_out/fvbB2_state.pt` (`d691a449…`), `fvbB2_emb.npz` (`bf6d03a3…`) | `FF_DATA/03_checkpoints/fvbind_b2/` |
| `t192_fvbx` | extract `v_fvb` for every bundle voice row, the v4 train voices and the four Evaluation lists | `thanhhuyen11/flag-2027-nb-192-fvbx3` (exact-staging, md5 `ba933cde…`) | T4, about 2.7 h | `fvbx_out/` | `FF_DATA/04_run_outputs/flag-2027-nb-192-fvbx3/` |
| `t163p_fvb` | train RFX′ (projector, 3 seeds; probes; fusers), `RFS_MODE=P`, `add [f_c5s, v_rl, v_fvb]` | `thanhhuyen11/flag-2027-nb-163-rfsfxfvb-p` (exact-staging, md5 `308d8715…`) | T4, 10 min | dataset `thanhhuyen11/flag2027-rfsfxfvb-art-p` | `FF_DATA/03_checkpoints/rfx_fvb/` |
| `e163i_fvb` | score the four Evaluation lists with RFX′ from the original media | `thanhhuyen11/flag-2027-nb-163i-rfsfxfvb-eval` (exact-staging, md5 `35435111…`) | T4, 34 min | `FLAG_RF_E2E/submission.zip` = `rfsfxfvb_eval.zip` (`45044189…`) | `FF_DATA/05_scores/` |
| `a_fvbind` | zips U (F1s′ + RFX′) and ENFUSE; one-step cross-check | local scripts (byte-exact copies in `a_fvbind/tree/`) | CPU, seconds | `posteval_FVBIND_U_fvb.zip` (`b61c84ed…`), `posteval_FVBIND_ENFUSE_fvb.zip` (`bb0e0175…`) | rebuilt by mode (c) |
| `a_best4_ref` | optional reference BEST4 (16.72); not an input of the final zip | new minimal assembler | CPU, seconds | `fourfusion_BEST4_v1.zip` (`eb58ba77…`) | — |

Each stage folder holds `stage.json` (definition, inputs, outputs, checks, notes), `stage.ipy` (the code the runner executes: the
original cells, cells separated by `# <<<FF-CELL>>>`), `CHANGES.md` (every change against the original code, with the reason) and
`original/` (the notebook of the run, its `kernel-metadata.json`, the run log when available, decoded copies of the embedded code under
`original/embedded/`, evidence runs under `original/evidence/`, and `SOURCE.json` with md5 / sha256 and the `kaggle/kapi_runs.jsonl` line).

## Run modes for this group

- **(c) byte-exact assembly** (`FF_MODE=assemble`): `a_fvbind` on the six saved files (`f1s_eval.zip`, `rfsfxfvb_eval.zip`,
  `components_eval.npz`, `e2e_manifest.json`, `cleantune01_fuser_bangla_CA.json`, `ct01r2_final_eval.npz`). The driver checks the input
  sha256 and the member sha256 of both zips against the 15/11/2026 originals (fatal in this mode). CPU, seconds.
- **(b) inference from the original Evaluation media + saved checkpoints** (`FF_MODE=infer`): `t192_fvbx` (or its saved output) →
  `e163i_fvb` → `a_fvbind`. Needs the saved B2 state and the RFX′ artifact. GPU re-runs agree closely but not bit for bit.
- **(a) full re-training** (`FF_MODE=train`): `t189_b2` → `t192_fvbx` → `t163p_fvb` → `e163i_fvb` → `a_fvbind`. Approximate only:
  GPU training is not deterministic, and the registered time rule of `t189_b2` sets the number of steps from the measured speed.

## The 13.54 variant

- Every stage attaches the **original** LoRA C1 dataset `onnhqunhh/flag2027-rfs-lora-p` where the run did. Only `t163p_fvb` and
  `e163i_fvb` use it (it is copied into the RFX′ artifact and drives the `ib` stream); the check after `t163p_fvb` requires
  `rfs_art/lora_C1.pt` sha256 `b97372e3504a131d0a969e478a855868095c67db4b3c0c133ef770c00e40991c`. The fixed C1 datasets of the 16.18 system
  must never be attached. Today's push scripts of the team (`kaggle/sat_push.sh` cases 189 / 192 / 193, `kaggle/rfx_chain.sh`) point to
  the fixed C1 and are not used here.
- In mode (a) the stages must consume the **saved** original C1 (`FF_DATA/03_checkpoints/rfs_lora_c1_original`): a re-trained C1 has
  other bytes and is refused by the variant guard of the runner (every consumer resolves it `saved_only`).
- `NOREL`, `LID_STRICT`, `BN42` are not read by these notebooks: the Bengali-voice filter acts through the content of `lid_strict.json`
  and through the audio cache, both built upstream (group 13, group 11).

## Organisers' data: redacted notebook copies

The notebooks of the FLAG_163 family (`t189_b2`, `t192_fvbx`, `t163p_fvb` and three evidence runs) embed the organisers' file
`meta_file_train_set.csv` (written as `train/gender.csv`; read only in bundle-assembly mode, which these stages never run). The repository
does not redistribute organisers' data, so the copies under `original/` have exactly that one base64 literal removed; `stage.ipy` is built
from those copies. With your copy of the organisers' file, the byte-exact notebooks (md5 as in `kaggle/kapi_runs.jsonl`) are rebuilt by

```
python stages/60_fvbind/restore_exact.py --csv <FF_DATA>/00_raw/btc_mavceleb_v4/meta_file_train_set.csv --out <folder outside the repo>
```

The script writes the fixed gzip header of the original literal itself, so the gzip module of the interpreter does not matter; the
deflate stream comes from the interpreter's zlib. Verified 7 / 7 byte-exact with CPython 3.14.0 on Windows (zlib-ng 2.2.4, zlib API
1.3.1) on 18/11/2026. On a failure the script says whether the deflate stream differs (another zlib build); use that interpreter then.
The byte-exact notebooks are also kept in the private data folder (`FF_DATA/_private_originals/<stage id>/`).

The report notebook of gate 2 (`thanhhuyen11/flag-2027-nb-194-fvbg2-report`) embeds a dev score zip and is not in the repository at all
(pointer in `t163p_fvb/original/evidence/gate2_report_194/SOURCE.json`).

## Private data

The audio cache, the VoxBlink feature bundles, the output of `t192_fvbx` and the RFX′ artifact contain VoxBlink-derived rows
(CC BY-NC-SA, no redistribution): they live only in the private data folder shared with the organisers, never in this repository.
The same holds for the B2 checkpoint `fvb_out/fvbB2_state.pt` and the whole kernel output of `t189_b2`: B2 was trained mostly on
VoxBlink audio (36 589 of the 52 750 training clips) and its stored constants μ_f / μ_v are means over the training persons and clips.

Licences of the pretrained models: ImageBind-huge (weights and code) is CC BY-NC 4.0, so every model derived from it (the B2 voice
block, the LoRA C1 of the `ib` stream, the RFX′ artifact) is for non-commercial use only.

