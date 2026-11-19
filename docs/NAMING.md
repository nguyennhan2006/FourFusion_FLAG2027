# Names used in this repository

Public text (README, documentation, stage titles, system description) uses the descriptive names in the left column. Internal team
names survive only where changing them would change bytes that are checked: file and folder names, environment variables, artifact
ids of the original Kaggle runs, and verbatim code. This file is the mapping.

## The English branch: DP (Different Profile) = internal name XJF

| public name | internal name (team repository) | where the internal name still appears |
|---|---|---|
| DP (Different Profile) branch | XJF, XJF-01 | `Experiment/XJF-01/...` paths in provenance; script names `xjf.py`, `blend.py` of `Experiment/XJF-01/`; files `xjf_scores.npz`, `xjfdump_v4.parquet`, `xjf_r10_trsig.zip` (the submitted file name); environment variables `XJF_SIG`, `XJF_BASE`; column / function names inside verbatim code |
| DP scale constants sigma = 5.0983 (no_gender) / 3.8376 (gender) | `XJF_SIG="5.0983,3.8376"` | `stages/80_final/final_blend`, `configs/variant_1354.json` |
| DP scores (z of the two English files) | `xjf_scores.npz` (`z_*`, `cos_*`, `ids_*`) | `stages/70_dp/*`, `assembly/` |
| sigma source table (U on held-out v4 train trials) | `xjfdump_v4.parquet` (FUSENET-01 `dumpv4`) | `stages/40_R_family/t219_dumpv4`, `stages/70_dp/dp_sigma` |

The submitted zip keeps its file name `xjf_r10_trsig.zip` (CodaBench submission 1054704), because its sha256
`b4354eaf988b2b4ea8c4e1c3fa44d7bcf631ea5cb3cd635ab2965021fe82e507` is the reference of this repository.

## Other internal names

| name in this repository | meaning |
|---|---|
| F1s′ (F1s prime, `F1s'`) | the identity-classification family: the CA head (stage `t112sf1s`; face LoRA C5s on CLIP ViT-L/14 and FaRL ep16, voice "CAT4" = frozen speaker encoders listed in that stage) + the XALL bridge + the T3 v11 head (stage `t164f`), fused by a logistic mixer fitted on SAFE-04 tables (stage `e112f1sx`); "prime" = rebuilt without the train-person cohort normalisation (`NOREL=1`) |
| RFX | the pair-contrastive family (RFS recipe of RF-092 retrained on the team's cleaned data) + the extra blocks `f_c5s` (face LoRA C5s) and `v_rl` (ResNet293 ‖ ECAPA-1024); artifact `ngnnguynchn/flag2027-rfsfxc5rl-art-p` |
| RFX′ (RFX prime) | RFX + the voice block `v_fvb` (FVBIND-01 B2: ImageBind-audio LoRA aligned to the face anchor); used only for the gender/Bangla file |
| FACTOR_ENFUSE (c5rl) | zip of F1s′ + RFX: Bangla = −(logit F1s′ + logit RFX), English = Bangla mixer of F1s′ applied to the English pairs + RFX; Evaluation 16.99 on its own |
| FVBIND_ENFUSE (fvb) | the same with RFX′; only its gender/Bangla member enters 13.54 |
| LoRA C1 (original) | low-rank adapter of the ImageBind audio tower inside RFS / RFX, trained with a cache-key bug on 210 MAV-Celeb v1 clips of 70 persons; `lora_C1.pt` sha256 `b97372e3…`; see `docs/VARIANTS.md` |
| SAFE-04 | the team's 5-fold validation protocol on train persons only (v4 held-out persons, MAV-Celeb v1 / v2 English, Urdu, Hindi cells); the frozen tables are `ngnnguynchn/flag2027-e2e-dev-p` (the name "e2e-dev" is historical: the SAFE-04 tables are built from train persons, but the frozen bundle also carries dev-phase files that the verbatim code checks for; see `docs/COMPLIANCE_NOTES.md` §1) |
| LID_STRICT, BN42, NOREL, CLEANCOMP | flags of the runs: Bengali-voice filter version 2 (`lid_strict.json`), the FLAG_42 Bengali list applied to every row source, no cohort normalisation, the clean component tables |
| `ff`, `ff-frozen` (artifact owners) | ids created by this repository: `notebooks/ff/<stage>` = output of a new stage; `datasets/ff/...` = a dataset made by a new stage; `datasets/ff-frozen/...` = a frozen copy registered under a second id to keep the graph acyclic |
| `_staging_backup/` (mentioned in provenance) | the team's byte-exact backup of the Kaggle push folders (`C:/Users/ASUS/AppData/Local/Temp/kapi_<slug>_<random>/`); kept **outside** this repository because some copies embed the organisers' meta file or dev score zips; the `original/` folder of each stage holds the redacted copy that was checked against it |
| `<team archive>` | the team's storage (USB disk `E:/FLAG2027_submit/...` and the team repository) holding the package `FACTOR_ENFUSE_c5rl/` with the exact notebooks, outputs and datasets of 26 runs |
| Kaggle account names in artifact ids | the accounts the original runs used (team members and helpers); kept because the verbatim code and `configs/data_map.json` key on the original ids |
