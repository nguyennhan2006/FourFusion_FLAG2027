# Environment of each stage's original run

Read on 2026-11-18 from the files kept in `stages/*/*/original/` (versions printed in notebook outputs and logs; `docker_image` from kernel metadata). Machine-readable copy: `configs/env_per_stage.json` (same content, used by the runner).

- **v173** = the November Kaggle image family: GPU runs print Python 3.13.16 and `torch 2.11.0+cu130`, CPU runs print Python 3.13.18 and `torch 2.11.0+cpu`. The digest `de153e6d…` (Kaggle tag v173) is confirmed only where `docker_image` is recorded (the pulled runs 155s / 164).
- **older** = September / October images: `torch 2.11.0+cu128` (collection c37 / c46 / c53, l42, f40, f41) or `torch 2.10.0+cu128` (f04, f14).
- **local** = the team machine (Python 3.14.0, `torch 2.12.1+cpu`).
- **unknown** = no version printed in the kept files.

`docker/constraints-kaggle-v173.txt` and `FF_PIP=pinned` therefore describe the **v173 family only**. The runner warns when `FF_PIP=pinned` is applied to another family; `FF_PIP_SCOPE=v173` keeps `original` for those stages.

| stage | device | family | python | torch | docker_image (digest confirmed) |
|---|---|---|---|---|---|
| `raw_btc` | cpu | unknown | - | - | - |
| `raw_evalphase` | cpu | unknown | - | - | - |
| `raw_mavceleb` | cpu | unknown | - | - | - |
| `raw_mavceleb_check` | cpu | unknown | - | - | - |
| `c37_vox2` | gpu | older | - | 2.11.0+cu128 | - |
| `c45_frames_vox2` | cpu | unknown | - | - | - |
| `c46_voxblink` | gpu | older | - | 2.11.0+cu128 | - |
| `c53_vox2b` | gpu | older | - | 2.11.0+cu128 | - |
| `c63_frames` | cpu | unknown | - | - | - |
| `c63v_audio_cache` | cpu | unknown | - | - | - |
| `c87_frames_bridge` | cpu | unknown | - | - | - |
| `c_frames_index` | cpu | unknown | - | - | - |
| `r07_overlap` | cpu | unknown | - | - | - |
| `r08_mav_rows` | gpu | unknown | - | - | - |
| `l150_whisper` | gpu | unknown | - | - | - |
| `l152_mms` | gpu | unknown | - | - | - |
| `l42_vox2_lid` | gpu | older | - | 2.11.0+cu128 | - |
| `l_dqual_g3` | cpu | unknown | - | - | - |
| `l_groups` | cpu | unknown | - | - | - |
| `l_lid_audit` | cpu | unknown | - | - | - |
| `l_lid_strict` | cpu | unknown | - | - | - |
| `b_eda_bundle` | cpu | unknown | - | - | - |
| `b_lab_data` | cpu | unknown | - | - | - |
| `b_lab_vb` | cpu | unknown | - | - | - |
| `f103c_tap01` | gpu | unknown | - | - | - |
| `f10_models` | gpu | unknown | - | - | - |
| `f12_eval_feats` | gpu | unknown | - | - | - |
| `f14_av` | gpu | older | - | 2.10.0+cu128 | - |
| `f40_t3enc_mav` | gpu | older | - | 2.11.0+cu128 | - |
| `f41_t3enc_vox2` | gpu | older | - | 2.11.0+cu128 | - |
| `f47b_screen2` | gpu | unknown | - | - | - |
| `f63w_aux_voice` | gpu | unknown | - | - | - |
| `t160a` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t160b` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t160c` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t162a` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t162b` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t162c` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `cleancomp_tables` | cpu | unknown | - | - | - |
| `gather_g1` | cpu | unknown | - | - | - |
| `gather_r2ca` | cpu | unknown | - | - | - |
| `safe04_lab` | gpu | unknown | - | - | - |
| `t112sf0s` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t112sf1s` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t153b` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t153c` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t154_g1_f12` | gpu | unknown | - | - | - |
| `t154_g1_f34` | gpu | unknown | - | - | - |
| `t154_g1_f5` | gpu | unknown | - | - | - |
| `t155s_f12` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | sha256:de153e6d3038 (yes) |
| `t155s_f34` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | sha256:de153e6d3038 (yes) |
| `t155s_f5` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | sha256:de153e6d3038 (yes) |
| `t164_f123` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | sha256:de153e6d3038 (yes) |
| `t164_f45` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | sha256:de153e6d3038 (yes) |
| `t164f` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `t163b` | cpu | v173 | 3.13.18 | - | - |
| `t163l_orig` | gpu | v173 | 3.13.16 | - | - |
| `t163p_c5rl` | gpu | v173 | 3.13.16 | - | - |
| `t213_p1` | gpu | unknown | - | - | - |
| `t213_p2` | gpu | unknown | - | - | - |
| `t216_p3` | gpu | unknown | - | - | - |
| `t219_dumpv4` | cpu | v173 | 3.13.18 | 2.11.0+cpu | - |
| `a_factor` | cpu | unknown | - | - | - |
| `e112f0sx` | cpu | v173 | 3.13.18 | 2.11.0+cpu | - |
| `e112f1sx` | cpu | v173 | 3.13.18 | 2.11.0+cpu | - |
| `e163i` | gpu | v173 | 3.13.16 | - | - |
| `e47rs` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `e50cc` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `e65xbs` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `e65xcs` | gpu | v173 | 3.13.16 | 2.11.0+cu130 | - |
| `a_fvbind` | cpu | unknown | - | - | - |
| `e163i_fvb` | gpu | v173 | 3.13.16 | - | - |
| `t163p_fvb` | gpu | unknown | - | - | - |
| `t189_b2` | gpu | v173 | 3.13.16 | - | - |
| `t192_fvbx` | gpu | unknown | - | - | - |
| `dp_extract_eval` | cpu | local | 3.14.0 | 2.12.1+cpu | - |
| `dp_feats_layout` | cpu | unknown | - | - | - |
| `dp_score` | cpu | unknown | - | - | - |
| `dp_sigma` | cpu | unknown | - | - | - |
| `dp_train` | cpu | unknown | - | - | - |
| `final_blend` | cpu | local | 3.14.0 | - | - |
| `final_verify` | cpu | unknown | - | - | - |
| `env_docker` | cpu | unknown | - | - | - |
| `models_fetch` | cpu | unknown | - | - | - |
