# Checkpoints and learned tables

Learned weights and tables of the 13.54 chain. None is stored in git; the saved copies go to `FF_DATA/03_checkpoints/<name>/`
(Drive). Mode (b) (`FF_MODE=infer`) uses the saved copies; mode (a) retrains them (GPU training is not deterministic, so a retrained
checkpoint gives close but not identical scores), except the ORIGINAL LoRA C1, which defines the 13.54 variant and is always taken
saved (`docs/VARIANTS.md`).

## 1. State on 19/11/2026

| what | state |
|---|---|
| saved copies kept by the team (package `<team archive>/FLAG2027_submit/FACTOR_ENFUSE_c5rl/` and the team repository) | face LoRA B3s / C5s, F1s′ tables and heads (cleancomp-e2e, ct02s-r2ca, cleancomp-final, evalu-f1s, evalu-f0s), original LoRA C1, RFX c5rl artifact (package); sigma source table `xjfdump_v4.parquet` (team repository); sha256 in the tables below |
| `03_checkpoints/fvbind_b2` (block B2 of RFX′) | on the team machine (`Experiment/FVBIND-01/out/fvbx_in/b2/fvb_out/`), not on Drive yet |
| `03_checkpoints/rfx_fvb` (`thanhhuyen11/flag2027-rfsfxfvb-art-p`) and the output of `thanhhuyen11/flag-2027-nb-192-fvbx3` | **only on Kaggle** (DESIGN G4); `tools/PULL_MISSING_RUNS.md` has the download commands; mode (b) cannot finish the gender/Bangla file without them |
| `03_checkpoints/dp_towers` (`dp_towers.pt`) | **does not exist yet** (DESIGN G5): the original script saved only scores. Stage `dp_train` now also saves the ten towers and constants; it must be run once on the team machine (CPU about 12 min, deterministic there) and its output uploaded. Its 20 input feature arrays (~0.9 GB, `02_intermediate/dp_features`) are only on the team machine (G13) |
| Drive folder | not populated yet (decision Q-D / Q-E of the team) |

## 2. Table (generated from `configs/data_map.json`, class checkpoint)

<!-- BEGIN GENERATED: checkpoints -->
| artifact id | path under FF_DATA | producing stage (kind, device, minutes of the original run) | files with sha256 | used by | status |
|---|---|---|---|---|---|
| `datasets/ngnnguynchn/flag2027-cleancomp-e2e-p` | `03_checkpoints/f1s_safe04_tables` | cleancomp_tables (train, CPU, 1) | zip `a36698fa78f2a3ac…` | e50cc, t112sf1s | saved copy expected on Drive |
| `datasets/ngnnguynchn/flag2027-cleancomp-final-p` | `03_checkpoints/f1s_xall_t3v11` | t164f (train, GPU, 9) | `final_info.json` `4c7ba5e1c689eafc…`; `s007_xall.pt` `e6b74a101221cfb8…`; `v11_head/t3_streams.npz` `09867a3f2b644095…` | e50cc | saved copy expected on Drive |
| `datasets/ngnnguynchn/flag2027-ct02s-r2ca-tables-p` | `03_checkpoints/f1s_ca_oof_tables` | gather_r2ca (train, CPU, 1) | `tables/ct01r2_fold1.parquet` `00cf0c5fb8dbac74…`; `tables/ct01r2_fold2.parquet` `a0e2c0338037b44d…`; `tables/ct01r2_fold3.parquet` `712ad675021874b6…` | e112f1sx, t112sf1s, t183_p1, t183_p2, t189_b2, t192_fvbx, t213_p1, t213_p2, t216_p3, t3s01_trca | saved copy expected on Drive |
| `datasets/ngnnguynchn/flag2027-evalu-f0s-p` | `03_checkpoints/_wiring_b0s_head` | t112sf0s (train, GPU, 60) | `ct01g1_final_models.pt` `1933286f1d4c7a54…` | e112f0sx | saved copy expected on Drive |
| `datasets/ngnnguynchn/flag2027-evalu-f1s-p` | `03_checkpoints/f1s_ca_head` | t112sf1s (train, GPU, 154) | `ct01r2_final_models.pt` `19285c5ca08a9454…` | e112f1sx | saved copy expected on Drive |
| `datasets/ngnnguynchn/flag2027-rfsfxc5rl-art-p` | `03_checkpoints/rfx_c5rl` | t163p_c5rl (train, GPU, 9) | `final_params.npz` `ed76bf64ab9e3b16…`; `fuse_report.json` `75f26e60777e4f87…`; `lora_C1.pt` `b97372e3504a131d…` | e163i | saved copy expected on Drive |
| `datasets/onnhqunhh/flag2027-rfs-lora-p` | `03_checkpoints/rfs_lora_c1_original` | t163l_orig (train, GPU, 66) | `lora_C1.pt` `b97372e3504a131d…`; `v4_extra.npz` `793f81bd3909597c…` | t163p_c5rl, t163p_fvb, t183_p1, t183_p2, t189_b2, t192_fvbx, t213_p1, t213_p2, t216_p3 | saved copy expected on Drive |
| `datasets/phanvnhtrng/flag2027-feats-ls-b3s-p` | `03_checkpoints/face_lora_b3s` | t153b (train, GPU, 494) | `facelora_state.pt` `b06a65c862a57958…` | e65xbs, t112sf0s, t112sf1s, t154_g1_f12, t154_g1_f34, t154_g1_f5, t155s_f12, t155s_f34, t155s_f5, t184_report, t219_dumpv4 | saved copy expected on Drive |
| `datasets/quynhnhu82/flag2027-feats-ls-c5s-p` | `03_checkpoints/face_lora_c5s` | t153c (train, GPU, 712) | `bn42_clean.json` `5dee5d7f9155fe28…`; `facelora_state.pt` `4dab964c83addaad…`; `train_log.json` `a7d8efa1cd6d0c4e…` | e65xcs, t112sf1s, t155s_f12, t155s_f34, t155s_f5, t163p_c5rl, t163p_fvb, t183_p1, t183_p2, t184_report, t189_b2, t213_p1, t213_p2, t216_p3, t219_dumpv4 | saved copy expected on Drive |
| `datasets/thanhhuyen11/flag2027-rfsfxfvb-art-p` | `03_checkpoints/rfx_fvb` | t163p_fvb (train, GPU, 10) |  | e163i_fvb | not downloaded (only on Kaggle, account thanhhuyen11) |
| `notebooks/ff/dp_train` | `03_checkpoints/dp_towers` | dp_train (train, CPU, 11) | `consts.json` `356242de47674dfb…`; `xjf_scores.npz (English only)` `686c1188ce2cd5c6…` | dp_score | does not exist yet (DESIGN G5: xjf.py never saved the two towers) |
| `notebooks/phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4` | `03_checkpoints/sigma_source` | t219_dumpv4 (train, CPU, 2) | `sp/out/safe04/xjfdump_v4.parquet` `9f0a5af468d1f973…` | dp_sigma | kernel deleted on Kaggle; team machine copy Experiment/XJF-01/out/dumpv4/ (parquet + log) |
| `notebooks/thanhhuyen11/flag-2027-nb-189-fvbind-b2-ib0` | `03_checkpoints/fvbind_b2` | t189_b2 (train, GPU, 380) | `fvb_out/fvbB2_emb.npz` `bf6d03a3aadc0e72…`; `fvb_out/fvbB2_state.pt` `d691a44933b252cf…` | t192_fvbx | team machine copy: Experiment/FVBIND-01/out/fvbx_in/b2/fvb_out/ |
<!-- END GENERATED: checkpoints -->

## 3. Other stored results that act like checkpoints

| artifact | where | role |
|---|---|---|
| `notebooks/phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4` → `xjfdump_v4.parquet` (sha256 `9f0a5af468d1f973e1ff33264ff10239cf21f1813a0f370740e8c2d8e02f7a09`, 12 896 held-out v4 train trials) | `03_checkpoints/sigma_source/` | source of sigma = 5.0983 / 3.8376 (stage `dp_sigma`); also mode (c) input |
| Evaluation run outputs `thanhhuyen11/flag-2027-nb-{65xbs,65xcs,47rs,50cc,112f0sx,112f1sx,163i}-…-eval` | `04_run_outputs/<slug>/` | let mode (b) skip a GPU stage; the per-pair files of mode (c) are extracted from them (`assembly/inputs/README.md`) |
| `notebooks/ff/dp_train` (`xjf_scores.npz` English sha256 `686c1188…`, `consts.json` `356242de…`) | `03_checkpoints/dp_towers/` | DP scores; `dp_towers.pt` to be added (G5) |

Pretrained public models are not checkpoints of this chain; they are listed with pinned revisions in `LICENSES.md` and fetched by
`tools/fetch_models.py` into `FF_DATA/06_pretrained/`.
