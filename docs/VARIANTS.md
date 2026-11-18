# The 13.54 variant and how the runner guards it

The displayed Evaluation submission is CodaBench 1054704 = `xjf_r10_trsig.zip`
(sha256 `b4354eaf988b2b4ea8c4e1c3fa44d7bcf631ea5cb3cd635ab2965021fe82e507`; Evaluation 13.54 = 10.20 · 11.97 · 12.98 · 18.99 for
no_gender/English, no_gender/Bangla, gender/English, gender/Bangla). Several close systems exist in the team repository; this file
says what makes the chain of this repository that one and not another.

## 1. What defines 13.54

| item | 13.54 value | where it is fixed / checked |
|---|---|---|
| LoRA C1 of the ImageBind audio tower (inside RFS / RFX / RFX′) | the **original** one, trained with the cache-key bug on 210 MAV-Celeb v1 clips of 70 persons (0 VoxCeleb2 clips): `datasets/onnhqunhh/flag2027-rfs-lora-p` (copy `datasets/ngnnguynchn/flag2027-backup-rfs-lora-p`), `lora_C1.pt` sha256 `b97372e3504a131d0a969e478a855868095c67db4b3c0c133ef770c00e40991c` | `configs/variant_1354.json` `required.lora_c1` and `file_checks`; every consumer stage mounts it with `resolve: saved_only` (t163p_c5rl, t183_p*, t213_p*, t216_p3, t163p_fvb, t189_b2, t192_fvbx); `stages/40_R_family/C1_VARIANTS.md` |
| gender/Bangla file | from the FVBIND ENFUSE path: −(logit F1s′ + logit RFX′), member sha256 `9a6197360fa2c1dd…` | stages `t189_b2` → `t192_fvbx` → `t163p_fvb` → `e163i_fvb` → `a_fvbind` |
| no_gender/Bangla file | FACTOR_ENFUSE (c5rl): −(logit F1s′ + logit RFX), member `36056e7e…` | stages `e112f1sx`, `e163i` → `a_factor` |
| both English files | `blend.py`: (FACTOR_ENFUSE English score / sigma[protocol]) − r × z_DP with sigma = 5.0983 (no_gender) / 3.8376 (gender) and r = 1.0; members `d45f8502…`, `ead7e177…` | `stages/80_final/final_blend` asserts `XJF_SIG = 5.0983,3.8376` and r = 1.0 under `FF_VARIANT=1354`; sigma is recomputed by `stages/70_dp/dp_sigma` from `xjfdump_v4.parquet` |
| flags of the runs | `NOREL=1` (no train-person cohort normalisation), `LID_STRICT=1` (Bengali-voice filter version 2), `BN42=1`, `CLEANCOMP=1`, as each original run set them | kept verbatim in the stage code / env (e50cc, e112f0sx, e112f1sx, t112s*, t153b / c, t155s*, t219) |

## 2. Close systems that must not leak in

| system | difference | guard |
|---|---|---|
| 16.18 (C1FIX-01, "FACTOR_ENFUSE + fixed LoRA") | LoRA C1 retrained with the key fixed (`ngnnguynchn/flag2027-c1fix-lora-p`, RFX artifact `ngnnguynchn/flag2027-c1fix-rfx-art-p`) | `configs/variant_1354.json` `forbidden_artifacts`; the runner stops a stage of the chain that mounts one |
| LORAFIX-01 / CELL4-01 `lf` | team's key-fixed LoRA `ngnnguynchn/flag2027-rfs-lorafix1-p`, RFX `ngnnguynchn/flag2027-rfsc4lf-art-p` | same list; environment `RFS_LORA_FIX=1` forbidden, `RFS_LORA_FIX=0` set by default for every stage |
| `xjf_r10_trsig_lf.zip` | English / Bangla from the `lf` system | environment `XJF_BASE=lf` forbidden; `final_blend` asserts it is unset |
| `xjf_r10.zip` (CodaBench 1054532, also 13.54) | DP scale constants 5.8 / 2.9 read off the test-score scale (not allowed) | `final_blend` asserts 5.0983 / 3.8376 (constants from held-out train persons); see `docs/COMPLIANCE_NOTES.md` |
| RF-092 dev embeddings | `ngnnguynchn/flag2027-rf092-dev-emb-p` (dev data) | forbidden artifact; dropped from `t162a` |

Today's team scripts default to the fixed C1 in several places (`release/flag2027_e2e/stages.py`, `kaggle/rfx_chain.sh`,
`kaggle/sat_push.sh`, `Experiment/RFS-01/train/rfs_train.py` with `RFS_LORA_FIX` defaulting to 1); the stages of this repository run
the code that actually ran (embedded copies without the switch, or with the switch off). The list with line numbers is in
`stages/40_R_family/C1_VARIANTS.md`.

## 3. What the runner checks

- Before a stage: forbidden artifacts / environment values (`configs/variant_1354.json`), per-stage `variant.require_files`
  (e.g. `e163i` requires `lora_C1.pt` = `b97372e3…` and `manifest_final.json` in the RFX artifact).
- After a stage: every `lora_C1.pt` found in an artifact of the chain must have the original sha256, or descend from a forced
  `t163l_orig` run (lineage rule; such an output is never mounted by the chain because its consumers are `saved_only`).
- `python runner/selftest.py` exercises these rules on a toy repository with the real configuration.
