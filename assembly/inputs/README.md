# assembly/inputs: where the ten stored files come from

This folder is empty on purpose. Whether the repository will be public or private (decision Q-D) is still open, so the stored per-pair score files are kept in the data folder `FF_DATA/05_scores/`, which is shared only with the organisers. `assembly/assemble_exact.py` looks there first: for each entry of `FF_DATA`, it tries `<entry>/05_scores`. It falls back to this folder, so for a private copy of the repository the ten files can simply be placed here.

The runner (`FF_MODE=assemble`) uses the same flat layout: stages `a_factor`, `a_fvbind`, `dp_sigma` and `final_blend` look in `<FF_DATA entry>/05_scores/` and then in this folder when no saved copy of the producing run is mounted (see `assembly/README.md`, "Layout for the runner").

| file | bytes | sha256 | copy kept by the team (team repository / package) |
|---|---:|---|---|
| `f1s_eval.zip` | 29 188 | `1b43b14ff763bd0eac72fea505e4cfd4ff98bc41d87d49baecdfd613525a6962` | `<team archive>/FLAG2027_submit/FACTOR_ENFUSE_c5rl/rebuild/in/` (output `e2e_out/submission_cleantune01_F1_CA_eval.zip` of `thanhhuyen11/flag-2027-nb-112f1sx-eval`) |
| `rfx_eval.zip` | 32 049 | `1fa53218c9501ef38f9b1c33680fac5b7f6a0ddace4fdaffed7cd5334b9aa369` | same folder (output `FLAG_RF_E2E/submission.zip` of `thanhhuyen11/flag-2027-nb-163i-rfsfxc5rl-eval`) |
| `components_eval.npz` | 17 384 406 | `2cd160ae456447c0c47bf611f2f73480ffc5c9161e767812d42e5f380b1b56c7` | same folder (output `e2e_eval/` of `thanhhuyen11/flag-2027-nb-50cc-e2e-eval`) |
| `e2e_manifest.json` | 12 400 | `b0ba27c2982f917199d957cd42763fecfdd31c292352594a4e141105ce72bd58` | same folder (same run) |
| `cleantune01_fuser_bangla_CA.json` | 4 695 | `c4737367e787da6dc9a1e08453a9602e2ea60b830bd7a56e8d6728501e9d06a4` | same folder (output `e2e_out/` of `…-112f1sx-eval`) |
| `ct01r2_final_eval.npz` | 21 154 | `bbad53ed8e32066240f405676e7f3676dd55040573c00d1fad07b81774c3d0fc` | same folder (same run) |
| `rfsfxfvb_eval.zip` | 32 027 | `45044189ae4b228b93aaae7e6eb4015f38516bfaa8c57abf2940db7ea6590862` | `Experiment/FVBIND-01/out/eval163i/FLAG_RF_E2E/submission.zip` (output of `thanhhuyen11/flag-2027-nb-163i-rfsfxfvb-eval`) |
| `xjf_scores.npz` | 33 960 | `686c1188ce2cd5c64ba506e19f7e17de427a693878697115b43d961627172779` | `Experiment/XJF-01/out/bak_english_only/xjf_scores.npz`. Or `Experiment/XJF-01/out/xjf_scores.npz` (96 286 B, `6a039faf…`): same English arrays plus Bangla keys. Either is accepted |
| `xjfdump_v4.parquet` | 681 816 | `9f0a5af468d1f973e1ff33264ff10239cf21f1813a0f370740e8c2d8e02f7a09` | `Experiment/XJF-01/out/dumpv4/sp/out/safe04/xjfdump_v4.parquet` (output of `phantrnhuyntrn/flag-2027-nb-219-fusenet-dumpv4`) |
| `consts.json` (optional) | 136 | `356242de47674dfb0c3a245bf3c504ee9f37417e6928040313b155d6bade6324` | `Experiment/XJF-01/out/consts.json` |

None of these files holds media, features or dev data. They hold per-pair scores of our models (pair ids of the organisers' Evaluation lists), our fuser weights, and scores of our models on pairs of the organisers' train persons (`xjfdump_v4.parquet`: person ids `idNNN` of the train set). They are not copied into a public repository without the team's decision. Until that decision, the data_map rows of these files (`scores/1354_inputs`, `notebooks/ff/dp_score`) should be marked `private: true` (to do for WP-A); `sigma.json`, which holds only the two constants, may be public.
