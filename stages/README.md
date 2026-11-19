# stages/ — the 13.54 chain, group by group

Each folder `stages/<group>/<stage>/` is one step of the chain that made `xjf_r10_trsig.zip`: `stage.json` (inputs, outputs,
device, checks), `stage.ipy` (the cells that run, separated by `# <<<FF-CELL>>>`), `CHANGES.md` (every difference from the code
that ran, with its reason) and `original/` (the executed notebook or script, redacted where it embedded the organisers' data).
The groups follow the data flow from raw data to the submitted file. The full stage list with devices and run times is
[`docs/STAGES.md`](../docs/STAGES.md); `python runner/ffrun.py dag` prints the graph.

| group | part of the system | what its stages make |
|---|---|---|
| `00_env` | environment | check of the Kaggle base image and pinned packages; download of every pretrained model at a pinned revision |
| `10_raw` | input data | the organisers' MAV-Celeb v4 train / Evaluation data and the MAV-Celeb v1 / v2 zips, checked and unpacked |
| `11_collect` | auxiliary data | VoxCeleb2 and VoxBlink identities, face frames and 3 s audio cache (run only when forced; mode (a) uses the collected copies) |
| `12_rows` | row tables | MAV-Celeb row tables and the train-only identity-overlap list |
| `13_lid` | Bengali filter | language identification of every auxiliary and MAV voice (Whisper, MMS, VoxLingua) and the strict exclusion list; person groups |
| `20_features` | frozen encoders | face and voice features of the train, auxiliary and MAV rows, and the lab bundles the families read |
| `30_F_family` | identity-classification family (F1s′) | face adapters B3s / C5s, the SAFE-04 tables, the CA / bridge / T3 heads per fold and the F1s′ mixer |
| `40_R_family` | pair-contrastive family (RFX) | the RFS bundle, the original LoRA C1, the RFX projector (c5rl) and the three seed replicas that give the DP scale constants |
| `50_eval` | Evaluation scoring | features and scores of the four Evaluation lists for F1s′ and RFX; `a_factor` assembles the FACTOR_ENFUSE zip |
| `60_fvbind` | RFX′ | RFX with the FVBIND B2 voice block; `a_fvbind` gives the same-gender Bangla file |
| `70_dp` | DP branch | DP features, training, the scale constants σ = 5.098 / 3.838 and the DP scores of the Evaluation lists |
| `80_final` | submission | the per-file blend into `xjf_r10_trsig.zip` and its byte check |

Helper scripts kept next to the stages they serve (not stages themselves): `30_F_family/_common/` (group check, redaction of
embedded dev zips), `40_R_family/_wpf_tools.py` (group check, redaction, decoding of embedded code), `50_eval/_tools/` (redaction of
dev references, group self-check), `60_fvbind/restore_exact.py` (rebuilds the byte-exact notebooks from the redacted copies when the
organisers' metadata file is present).
