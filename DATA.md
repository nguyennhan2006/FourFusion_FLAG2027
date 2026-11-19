# Data

No data is stored in this repository. Every input lives under one data root, `FF_DATA`, whose layout is the layout of the team's
Google Drive folder `FourFusion_FLAG2027_data/` (shared only with the organisers' e-mail, because it holds the organisers' data and
VoxBlink material). `configs/data_map.json` maps every artifact id of the original Kaggle runs (`datasets/<owner>/<slug>`,
`notebooks/<owner>/<slug>`) to a folder of `FF_DATA`; the runner builds each stage's virtual `/kaggle/input` from it, so the verbatim
code reads the paths it read on Kaggle.

## 1. Layout of `FF_DATA`

```
FF_DATA/                                   # = Drive "FourFusion_FLAG2027_data/"
├── 00_raw/
│   ├── btc_mavceleb_v4/                   # organisers: train_set.zip, meta_file_train_set.csv (+ dev_set.zip, never read by the chain)
│   ├── btc_evaluation/                    # organisers: evaluation_phase.zip ; eval_set.zip (built by stage raw_evalphase)
│   ├── mavceleb_v1_v2/                    # MAV-Celeb v1 / v2 extracted with the exact Kaggle layout mavceleb_v1_complete-001/v1/..., mavceleb_v2_complete-002/v2/...
│   └── mavceleb_v1_v2_zips/               # the two original zips (input of stage raw_mavceleb; a SIBLING folder on purpose)
├── 01_collected/                          # data the team collected from public Hugging Face sources (VoxBlink: private)
├── 02_intermediate/<name>/                # one folder per intermediate dataset; _wiring/<slug>/ for datasets the code only checks for
├── 03_checkpoints/<name>/                 # learned weights / tables (CHECKPOINTS.md)
├── 04_run_outputs/<run slug>/             # /kaggle/working of runs that later runs attach as kernel input
├── 05_scores/                             # the ten per-pair score files of mode (c) (assembly/inputs/README.md)
└── 06_pretrained/                         # public pretrained models + SOURCES.json (python tools/fetch_models.py --dest FF_DATA/06_pretrained)
```

The MAV-Celeb layout is load-bearing: the original LoRA C1 run took the first `v1` / `v2` folder it met (DESIGN 4.1,
`stages/40_R_family/C1_VARIANTS.md`). On Kaggle, `FF_DATA=auto` makes every attached folder holding `00_raw/` … `06_pretrained/` a data
root; several roots can be given with the path separator.

## 2. Raw data (start of mode (a))

| data | sha256 | stage that checks / prepares it |
|---|---|---|
| organisers' `train_set.zip` (MAV-Celeb v4, 1 428 601 179 B) | `1803739f059860d5aede064839a17b01c6297f3160ab80cc9d3c929b27ae2ad9` | `raw_btc` |
| organisers' `meta_file_train_set.csv` | `fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030` | `raw_btc` |
| organisers' `dev_set.zip` (989 881 860 B; not read by the chain, `docs/COMPLIANCE_NOTES.md` §1) | `f29cc4139bc3e40bf9e9cd6533b6822e41296de972589d85ac71953cd1a92983` | `raw_btc` (optional) |
| organisers' `evaluation_phase.zip` (472 428 761 B) | `c9efa2100dc54188fbd9739970a98c1cd966985d9c6f7d46148e5a3af96fc9f9` | `raw_evalphase` |
| `eval_set.zip` = `datasets/ngnnguynchn/flag2027-evalphase-p` (`*_valid` → `*_test`, 416 lines of `gender/English_valid.txt` repaired, originals kept under `btc_original/`) | `a70762a3fb52af6bbf52c14a86c79a4d074ec319b78d6ffedcdf195d71f9381e` | `raw_evalphase` (deterministic) |
| MAV-Celeb v1 / v2 zips (12 245 623 417 B / 13 477 476 090 B) | `2082a9ee5f7d3634…` / `daff513527d7cea6…` (measured 18/11; `configs/expected_sha256.json`) | `raw_mavceleb` (extract), `raw_mavceleb_check` (count an extracted copy) |

Collected data (`01_collected/`) counts as raw for 13.54: VoxCeleb2 keys / rows of FLAG_37 and FLAG_53, VoxBlink-clean of FLAG_46,
the PNG face frames, the 3-second audio cache. The collection code is in `stages/11_collect/` (re-collection needs a Hugging Face
token with the `ProgramComputer/voxceleb` terms accepted; sampling depends on the network, so 13.54 uses the collected copies).
Hugging Face sources are read at `resolve/main`; a commit can be pinned with `FF_HF_REVS`.

<!-- BEGIN GENERATED: raw -->
| artifact id | class | path under FF_DATA | producing stage | private | sha256 (zip / file) | consumers |
|---|---|---|---|---|---|---|
| `datasets/chnnhnnguyn/external-data-mavceleb` | raw | `00_raw/mavceleb_v1_v2` | raw_mavceleb | yes |  | e163i, e163i_fvb, e47rs, e65xbs, e65xcs, f103c_tap01 … |
| `datasets/ngnnguynchn/flag2027-evalphase-p` | raw | `00_raw/btc_evaluation/flag2027-evalphase-p` | raw_evalphase | yes | `411c9f1682b05d56…` | dp_extract_eval, dp_score, dp_train, e163i, e163i_fvb, e47rs … |
| `datasets/ngnnguynchn/flag2027-raw-v4-p` | raw | `00_raw/btc_mavceleb_v4` | - | yes | `dev_set.zip` `f29cc4139bc3e40b…` | b_eda_bundle, c45_frames_vox2, dp_extract_eval, dp_train, e163i, e163i_fvb … |
| `raw/btc_evaluation_original` | raw | `00_raw/btc_evaluation/evaluation_phase.zip` | - | yes | `evaluation_phase.zip` `c9efa2100dc54188…` | raw_evalphase |
| `raw/mavceleb_v1_v2_zips` | raw | `00_raw/mavceleb_v1_v2_zips` | - | yes | `mavceleb_v1_complete-001.zip` `2082a9ee5f7d3634…` | raw_mavceleb |
| `datasets/duongphihoang/flag2027-audio-cache-p` | collected | `01_collected/audio_cache_3s` | c63v_audio_cache | yes | `4c0de5060ea84e0b…` | f103c_tap01, l150_whisper, t160a, t160b, t160c, t162a … |
| `datasets/ngnnguynchn/flag2027-feats-vox2b-p` | collected | `01_collected/voxceleb2_flag53` | c53_vox2b | yes | `01b870c6197c3069…` | c63v_audio_cache, e112f0sx, e112f1sx, e65xbs, e65xcs, f103c_tap01 … |
| `datasets/ngnnguynchn/flag2027-frames-index-p` | collected | `01_collected/frames_index` | c_frames_index | yes |  | c63_frames, c63v_audio_cache, f63w_aux_voice |
| `datasets/ngnnguynchn/flag2027-vox2-keys` | collected | `01_collected/voxceleb2_flag37_keys` | c37_vox2 | yes | `face_meta.csv` `abf8cb53cd224156…` | b_lab_data, f41_t3enc_vox2, l42_vox2_lid, l_dqual_g3 |
| `datasets/onnhqunhh/flag2027-lora-frames-p` | collected | `01_collected/face_frames_png` | c87_frames_bridge | yes | `20cf33248449b40d…` | e65xbs, e65xcs, t153b, t153c, t162a, t162b … |
| `notebooks/ff/c46_voxblink` | collected | `01_collected/voxblink_flag46` | c46_voxblink | yes |  | b_lab_vb, c_frames_index, l_dqual_g3, l_lid_strict |
| `notebooks/nguynlcnhtho/flag-2027-nb-53-scaleid` | collected | `01_collected/voxceleb2_flag53` | c53_vox2b | yes |  | c_frames_index |
| `notebooks/onnhqunhh/flag-2027-nb-45-facelora-farl` | collected | - | c45_frames_vox2 | yes |  | c87_frames_bridge |
| `notebooks/onnhqunhh/flag-2027-nb-63-frames` | collected | - | c63_frames | yes |  | c87_frames_bridge |
| `raw/voxceleb2_vox2_meta` | collected | `01_collected/voxceleb2_meta/vox2_meta.csv` | - | yes | `vox2_meta.csv` `9a4f437fcf78b803…` | l_groups |
<!-- END GENERATED: raw -->

## 3. Every other artifact

Classes (`configs/data_map.json` → `classes`): intermediate / checkpoint = a producing stage exists in this repository; run_output =
a run's working folder that later runs attach; score = mode (c) files; wiring = only needed because the verbatim code checks for it
(not in the 13.54 numbers); embedded = small data files carried inside stage code; forbidden = refused by the 13.54 variant guard.
Inputs taken as stored copies in every mode are listed by `python runner/ffrun.py dag` and in `docs/VERIFICATION.md` §3.

<!-- BEGIN GENERATED: intermediate -->
| artifact id | class | path under FF_DATA | producing stage | private | sha256 (zip / file) | consumers |
|---|---|---|---|---|---|---|
| `datasets/dnq1209/flag2027-feats-tap01-voice-p` | intermediate | `02_intermediate/feats_tap01_voice` | f103c_tap01 | yes | `a15691aad2f55ea6…` | e112f1sx, t112sf1s, t155s_f12, t155s_f34, t155s_f5, t184_report … |
| `datasets/ff/flag2027-t3s01-trca-p` | intermediate | - | t3s01_trca | no |  | t219_dumpv4 |
| `datasets/hthlin/flag2027-aux-voice-feats-p` | intermediate | `02_intermediate/aux_voice_feats` | f63w_aux_voice | yes | `7e879ff26825f8cc…` | e112f1sx, t112sf1s, t155s_f12, t155s_f34, t155s_f5, t163p_c5rl … |
| `datasets/ngnnguynchn/flag2027-e2e-dev-p` | intermediate | `02_intermediate/e2e_dev_frozen` | safe04_lab | yes | `0795818f7c1f2cfa…` | cleancomp_tables, safe04_compare, t112sf0s, t154_g1_f12, t154_g1_f34, t154_g1_f5 … |
| `datasets/ngnnguynchn/flag2027-eda-bundle-p` | intermediate | `02_intermediate/eda_bundle` | b_eda_bundle | yes | `a710be3f63d03959…` | e112f0sx, e112f1sx, e50cc, safe04_lab, safe04_lab_core, t112sf0s … |
| `datasets/ngnnguynchn/flag2027-ext-meta-p` | intermediate | `02_intermediate/ext_meta` | r08_mav_rows | yes | `61ff3019285043ff…` | e47rs, e65xbs, e65xcs, f103c_tap01, f10_models, f12_eval_feats … |
| `datasets/ngnnguynchn/flag2027-feats-screen2-p` | intermediate | `02_intermediate/feats_screen2` | f47b_screen2 | yes | `28cc84869d38136d…` | t112sf1s, t155s_f12, t155s_f34, t155s_f5, t163p_c5rl, t163p_fvb … |
| `datasets/ngnnguynchn/flag2027-lab-data-p` | intermediate | `02_intermediate/lab_data` | b_lab_data | yes | `lab/lab_manifest.json` `8062f5e938fc30d4…` | c45_frames_vox2, c53_vox2b, c63v_audio_cache, e112f0sx, e112f1sx, e50cc … |
| `datasets/ngnnguynchn/flag2027-lab-vb-p` | intermediate | `02_intermediate/lab_vb` | b_lab_vb | yes | `e1a55d95bc47e6c5…` | c63v_audio_cache, e112f0sx, e112f1sx, e65xbs, e65xcs, f103c_tap01 … |
| `datasets/ngnnguynchn/flag2027-lid-strict-p` | intermediate | `02_intermediate/lid_strict` | l_lid_strict | yes | `78a6026c52cece60…` | dp_train, e112f0sx, e112f1sx, e50cc, t112sf0s, t112sf1s … |
| `datasets/ngnnguynchn/flag2027-rfs-bundle-p` | intermediate | `02_intermediate/rfs_bundle` | t163b | yes | `f5f38463f682d102…` | t163l_orig, t163p_c5rl, t163p_fvb, t183_p1, t183_p2, t189_b2 … |
| `datasets/nguynchnngn1/flag2027-feats-rfs-a-p` | intermediate | `02_intermediate/feats_rfs_a` | t162a | yes | `6e086e0f8fd796e6…` | t163b, t163p_c5rl, t163p_fvb, t183_p1, t183_p2, t189_b2 … |
| `datasets/phanththui/flag2027-feats-rfs-b-p` | intermediate | `02_intermediate/feats_rfs_b` | t162b | yes | `0d8bb909a74eaff2…` | t163b, t163p_c5rl, t163p_fvb, t183_p1, t183_p2, t189_b2 … |
| `datasets/phanvnnnnn6789/flag2027-feats-rfs-c-p` | intermediate | `02_intermediate/feats_rfs_c` | t162c | yes | `e3972feae13303de…` | t163b, t163p_c5rl, t163p_fvb, t183_p1, t183_p2, t189_b2 … |
| `datasets/phuonggmaii/flag2027-feats-whv3-c-p` | intermediate | `02_intermediate/feats_whv3_c` | t160c | yes | `785811ad0ce294b2…` | t163b |
| `datasets/thanhhuyen11/flag2027-feats-whv3-b-p` | intermediate | `02_intermediate/feats_whv3_b` | t160b | yes | `9170a05d93e5c328…` | t163b |
| `datasets/tranthuytien2511/flag2027-feats-whv3-a-p` | intermediate | `02_intermediate/feats_whv3_a` | t160a | yes | `f43b9ae9a323c23a…` | t163b |
| `notebooks/duongphihoang/flag-2027-nb-150-lidwhisper` | intermediate | `02_intermediate/lid_whisper` | l150_whisper | yes |  | l_lid_audit, l_lid_strict |
| `notebooks/ff/dp_feats_layout` | intermediate | `02_intermediate/dp_features` | dp_feats_layout | yes |  | dp_extract_eval, dp_train |
| `notebooks/ff/f04_v2_ref` | intermediate | `02_intermediate/feats_v2` | f04_v2_ref | yes |  |  |
| `notebooks/ff/f10_models` | intermediate | `02_intermediate/feats_models` | f10_models | yes | `ext_v1_complete_face_adaface.npy` `d160e51b74941f56…` | b_eda_bundle, b_lab_data, dp_feats_layout |
| `notebooks/ff/f12_eval_feats` | intermediate | `02_intermediate/feats_eval` | f12_eval_feats | yes | `voice_ecapa6144_train.npy (lab-data copy)` `91f0587426deb020…` | b_eda_bundle, b_lab_data, dp_feats_layout, safe04_lab |
| `notebooks/ff/f40_t3enc_mav` | intermediate | `02_intermediate/feats_t3enc` | f40_t3enc_mav | yes |  | b_lab_data |
| `notebooks/ff/f41_t3enc_vox2` | intermediate | `02_intermediate/feats_vox2t3` | f41_t3enc_vox2 | yes |  | b_lab_data, l_dqual_g3 |
| `notebooks/ff/l_dqual_g3` | intermediate | `02_intermediate/dqual_g3` | l_dqual_g3 | yes |  | l_lid_strict |
| `notebooks/ff/l_groups` | intermediate | `02_intermediate/vox2_groups` | l_groups | yes |  |  |
| `notebooks/ff/l_lid_audit` | intermediate | `02_intermediate/lid_audit` | l_lid_audit | yes |  | l_lid_strict |
| `notebooks/ff/r08_mav_rows` | intermediate | `02_intermediate/feats_ext_full` | r08_mav_rows | yes |  | b_eda_bundle, dp_feats_layout, r07_overlap |
| `notebooks/ff/raw_evalphase` | intermediate | `00_raw/btc_evaluation` | raw_evalphase | yes | `eval_set.zip` `a70762a3fb52af6b…` | dp_extract_eval, dp_score, dp_train |
| `notebooks/nguynlcnhtho/flag-2027-nb-42` | intermediate | `02_intermediate/feats_vox2lid` | l42_vox2_lid | yes | `lid_summary.json` `4d05cb34fc0fa669…` | l_dqual_g3, l_groups, l_lid_strict |
| `notebooks/phantrnanna/flag-2027-nb-152-mmslid` | intermediate | `02_intermediate/lid_mms` | l152_mms | yes |  | l_lid_audit |
| `notebooks/dngxunqun/flag-2027-nb-164-cleancomp01-f45` | run_output | `04_run_outputs/flag-2027-nb-164-cleancomp01-f45` | t164_f45 | yes |  | cleancomp_tables |
| `notebooks/dnq1209/flag-2027-nb-103c-tap01-voice` | run_output | `04_run_outputs/flag-2027-nb-103c-tap01-voice` | - | yes |  |  |
| `notebooks/hdonqu/flag-2027-nb-154-ct02-g1-f34` | run_output | `04_run_outputs/flag-2027-nb-154-ct02-g1-f34` | t154_g1_f34 | yes |  | gather_g1 |
| `notebooks/locnguyentruongan/flag-2027-nb-112s-f0s` | run_output | `04_run_outputs/flag-2027-nb-112s-f0s` | t112sf0s | yes |  | t112sf1s |
| `notebooks/locnguyentruongan/flag-2027-nb-155s-ct02ca-f5` | run_output | `04_run_outputs/flag-2027-nb-155s-ct02ca-f5` | t155s_f5 | yes |  | gather_r2ca |
| `notebooks/nguyenchonnhan/flag-2027-nb-155s-ct02ca-f12` | run_output | `04_run_outputs/flag-2027-nb-155s-ct02ca-f12` | t155s_f12 | yes |  | gather_r2ca |
| `notebooks/nguyenchonnhan/flag-2027-nb-155s-ct02ca-f34` | run_output | `04_run_outputs/flag-2027-nb-155s-ct02ca-f34` | t155s_f34 | yes |  | gather_r2ca |
| `notebooks/nguynlcnhtho/flag-2027-nb-164-cleancomp01-f123` | run_output | `04_run_outputs/flag-2027-nb-164-cleancomp01-f123` | t164_f123 | yes |  | cleancomp_tables |
| `notebooks/nguynthanhnhnn/flag-2027-nb-154-ct02-g1-f12` | run_output | `04_run_outputs/flag-2027-nb-154-ct02-g1-f12` | t154_g1_f12 | yes |  | gather_g1 |
| `notebooks/phantrnhuyntrn/flag-2027-nb-183-factor-p1` | run_output | `04_run_outputs/flag-2027-nb-183-factor-p1` | t183_p1 | yes |  | t184_report |
| `notebooks/phantrnhuyntrn/flag-2027-nb-183-factor-p2` | run_output | `04_run_outputs/flag-2027-nb-183-factor-p2` | t183_p2 | yes |  | t184_report |
| `notebooks/phantrnhuyntrn/flag-2027-nb-184-factor-report` | run_output | `04_run_outputs/flag-2027-nb-184-factor-report` | t184_report | yes |  |  |
| `notebooks/phantrnhuyntrn/flag-2027-nb-213-seedrep-p1` | run_output | `04_run_outputs/flag-2027-nb-213-seedrep-p1` | t213_p1 | yes |  | t219_dumpv4 |
| `notebooks/phantrnhuyntrn/flag-2027-nb-213-seedrep-p2` | run_output | `04_run_outputs/flag-2027-nb-213-seedrep-p2` | t213_p2 | yes |  | t219_dumpv4 |
| `notebooks/phantrnhuyntrn/flag-2027-nb-216-seedrep3` | run_output | `04_run_outputs/flag-2027-nb-216-seedrep3` | t216_p3 | yes |  | t219_dumpv4 |
| `notebooks/thanhhuyen11/flag-2027-nb-112f0sx-eval` | run_output | `04_run_outputs/flag-2027-nb-112f0sx-eval` | e112f0sx | yes |  | e112f1sx |
| `notebooks/thanhhuyen11/flag-2027-nb-112f1sx-eval` | run_output | `04_run_outputs/flag-2027-nb-112f1sx-eval` | e112f1sx | yes |  | a_factor, a_fvbind |
| `notebooks/thanhhuyen11/flag-2027-nb-163i-rfsfxc5rl-eval` | run_output | `04_run_outputs/flag-2027-nb-163i-rfsfxc5rl-eval` | e163i | yes |  | a_factor |
| `notebooks/thanhhuyen11/flag-2027-nb-163i-rfsfxfvb-eval` | run_output | `04_run_outputs/flag-2027-nb-163i-rfsfxfvb-eval` | e163i_fvb | yes |  | a_fvbind |
| `notebooks/thanhhuyen11/flag-2027-nb-192-fvbx3` | run_output | `04_run_outputs/flag-2027-nb-192-fvbx3` | t192_fvbx | yes | `fvbx_out/eval/index_gender_Bangla.txt` `997806bbce4ccbf0…` | e163i_fvb, t163p_fvb |
| `notebooks/thanhhuyen11/flag-2027-nb-47rs-screen2-eval` | run_output | `04_run_outputs/flag-2027-nb-47rs-screen2-eval` | e47rs | yes |  | e112f1sx, e163i, e163i_fvb, t192_fvbx |
| `notebooks/thanhhuyen11/flag-2027-nb-50cc-e2e-eval` | run_output | `04_run_outputs/flag-2027-nb-50cc-e2e-eval` | e50cc | yes |  | a_factor, a_fvbind, e112f0sx, e112f1sx |
| `notebooks/thanhhuyen11/flag-2027-nb-65xbs-b3sx-eval` | run_output | `04_run_outputs/flag-2027-nb-65xbs-b3sx-eval` | e65xbs | yes |  | e112f0sx, e112f1sx |
| `notebooks/thanhhuyen11/flag-2027-nb-65xcs-c5sx-eval` | run_output | `04_run_outputs/flag-2027-nb-65xcs-c5sx-eval` | e65xcs | yes |  | e112f1sx, e163i, e163i_fvb |
| `notebooks/thanhtruc02/flag-2027-nb-154-ct02-g1-f5` | run_output | `04_run_outputs/flag-2027-nb-154-ct02-g1-f5` | t154_g1_f5 | yes |  | gather_g1 |
| `notebooks/ff/a_factor` | score | - | a_factor | yes | `posteval_FACTOR_ENFUSE_c5rl.zip` `9cf5bbe4e3af1418…` | a_best4_ref, final_blend |
| `notebooks/ff/a_fvbind` | score | - | a_fvbind | yes | `posteval_FVBIND_ENFUSE_fvb.zip` `bb0e0175e9e94481…` | a_best4_ref, final_blend |
| `notebooks/ff/dp_extract_eval` | score | `05_scores/dp_eval_feats` | dp_extract_eval | yes |  | dp_score, dp_train |
| `notebooks/ff/dp_score` | score | `05_scores/dp` | dp_score | yes | `xjf_scores.npz (English only)` `686c1188ce2cd5c6…` | final_blend |
| `notebooks/ff/dp_sigma` | score | `05_scores/dp_sigma` | dp_sigma | yes |  | final_blend |
| `scores/1354_inputs` | score | `05_scores` | - | yes |  |  |
| `scores/xjf_r10_trsig_submitted` | score | `05_scores/xjf_r10_trsig.zip` | - | yes | `xjf_r10_trsig.zip` `b4354eaf988b2b4e…` |  |
| `datasets/ff-frozen/flag2027-e2e-dev-p` | wiring | `02_intermediate/e2e_dev_frozen` | - | yes |  | safe04_compare, safe04_lab_core |
| `datasets/ff-frozen/flag2027-e2e-dev-wiring-p` | wiring | `02_intermediate/_wiring/e2e_dev_wiring` | - | yes | `e2edev/components_dev.npz` `b8da75dc9f44df07…` | safe04_lab |
| `datasets/ngnnguynchn/flag2027-best-v7-p` | wiring | `02_intermediate/_wiring/flag2027-best-v7-p` | - | yes | `9bac62044510eff7…` |  |
| `datasets/ngnnguynchn/flag2027-cc5-oof-p` | wiring | `02_intermediate/_wiring/flag2027-cc5-oof-p` | - | yes | `6766081fe11978cc…` |  |
| `datasets/ngnnguynchn/flag2027-ct02s-g1-tables-p` | wiring | `02_intermediate/_wiring/flag2027-ct02s-g1-tables-p` | gather_g1 | yes | `8d09a3af0c25e96c…` | e112f0sx, t112sf0s |
| `datasets/ngnnguynchn/flag2027-dsel02-oof-p` | wiring | `02_intermediate/_wiring/flag2027-dsel02-oof-p` | - | yes | `3849c9924d2f40db…` |  |
| `datasets/ngnnguynchn/flag2027-facelora03-tables-p` | wiring | `02_intermediate/_wiring/flag2027-facelora03-tables-p` | - | yes | `8ab8e33d81317125…` |  |
| `datasets/ngnnguynchn/flag2027-feats-scout-wsvrdn-p` | wiring | `02_intermediate/_wiring/flag2027-feats-scout-wsvrdn-p` | - | yes | `badbcb545b7fcced…` |  |
| `datasets/ngnnguynchn/flag2027-lab-avfull-p` | wiring | `02_intermediate/_wiring/flag2027-lab-avfull-p` | - | yes | `b5febc059547e3c0…` | e112f0sx |
| `datasets/ngnnguynchn/flag2027-members-p` | wiring | `02_intermediate/_wiring/flag2027-members-p` | - | yes | `cba6a339c7b91d8e…` | e50cc |
| `datasets/ngnnguynchn/flag2027-t3s01-tables-p` | wiring | `02_intermediate/_wiring/flag2027-t3s01-tables-p` | - | yes | `bf9a8d1dcb7b0810…` | t184_report |
| `datasets/trngsnam/flag2027-feats-peface01-pe0-p` | wiring | `02_intermediate/_wiring/flag2027-feats-peface01-pe0-p` | - | yes | `3c147dfb1ea265e9…` | t183_p1, t183_p2 |
| `notebooks/ff/f14_av` | wiring | `02_intermediate/feats_av` | f14_av | yes |  | b_lab_data |
| `embedded/BEST/v7_safe/cafuse_fuser.json` | embedded | - | - | no | `4148202185fa0c1b…` |  |
| `embedded/BEST/v7_safe/fuser_en.json` | embedded | - | - | no | `8c042484b55b0cd5…` |  |
| `embedded/Experiment/BIGB-01/cafuse_t3_fuser_v11.json` | embedded | - | - | no | `ec8a201ef8e59864…` |  |
| `embedded/Experiment/CLEANTUNE-01/ct_round2.json` | embedded | - | - | no | `d251142dec126de8…` |  |
| `embedded/Experiment/DSEL-01/vox2_groups.json` | embedded | - | l_groups | no | `4f44f5cee3f9a3ae…` |  |
| `embedded/Experiment/EN-T3-01/fuser_en_t3.json` | embedded | - | - | no | `fd549d18e983b415…` |  |
| `embedded/Experiment/SCALE-CC-01/vox2_gender.json` | embedded | - | l_groups | no | `1d229fddee02f573…` |  |
| `embedded/Experiment/SCALE-CC-01/vox2b_groups.json` | embedded | - | l_groups | no | `7beb7ce7f67de74b…` |  |
| `embedded/Experiment/STACK-02/eligible.json` | embedded | - | - | no | `9b4fe3f56fdb1a05…` |  |
| `embedded/Experiment/STACK-02/fold_persons.json` | embedded | - | safe04_lab | no | `d76f684d719e5c4c…` |  |
| `embedded/Experiment/VBSEL-01/vox2_sa_speakers.json` | embedded | - | - | no | `eccc765365323b2e…` |  |
| `datasets/ngnnguynchn/flag2027-c1fix-lora-p` | forbidden | - | - | yes |  |  |
| `datasets/ngnnguynchn/flag2027-c1fix-rfx-art-p` | forbidden | - | - | yes |  |  |
| `datasets/ngnnguynchn/flag2027-rf092-dev-emb-p` | forbidden | - | - | yes |  |  |
| `datasets/ngnnguynchn/flag2027-rfs-lorafix1-p` | forbidden | - | - | yes |  |  |
| `datasets/ngnnguynchn/flag2027-rfsc4lf-art-p` | forbidden | - | - | yes |  |  |
| `pretrained/models` | pretrained | `06_pretrained` | models_fetch | yes |  |  |
<!-- END GENERATED: intermediate -->

## 4. Sizes and sharing

The Drive folder needs about 70–80 GB (DESIGN G24). Private material (VoxBlink and everything derived from it, the organisers'
data, run outputs that hold dev files) is shared only with the organisers' e-mail; the `private` column above says which artifacts.
`python tools/drive_manifest.py` writes `MANIFEST.sha256` for a populated `FF_DATA`.
