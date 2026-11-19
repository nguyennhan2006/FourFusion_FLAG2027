# The three LoRA C1 variants, and why 13.54 needs the ORIGINAL one

LoRA C1 = a low-rank adapter (rank 8, alpha 16) on the audio tower of ImageBind-huge, trained with a symmetric InfoNCE loss so that a
voice lands near the mean ImageBind face of the same person. It feeds the `ib` stream of the pair-contrastive family (RFS / RFX):
`ib = cos(unit(f_ibf - mu_ibf), unit(v_lora - fill_lora))`. Three versions exist; only the first one is part of the displayed
Evaluation submission 13.54 (CodaBench 1054704, `xjf_r10_trsig.zip`, sha256 `b4354eaf988b2b4ea8c4e1c3fa44d7bcf631ea5cb3cd635ab2965021fe82e507`).

| variant | data actually learned | LoRA dataset (Kaggle) | `lora_C1.pt` | RFX artifacts built on it | systems |
|---|---|---|---|---|---|
| **original (cache-key bug)** | 210 clips of the 70 MAV-Celeb **v1** persons (centre 3 s of the wav, at most 3 videos per person); **0 VoxCeleb2** clips | `onnhqunhh/flag2027-rfs-lora-p` (copy: `ngnnguynchn/flag2027-backup-rfs-lora-p`) | sha256 `b97372e3504a131d0a969e478a855868095c67db4b3c0c133ef770c00e40991c`, 1 782 355 bytes; metadata `{params: 48, steps: 1200, batch: 64, clips: 210, persons: 70, skipped: 0}` (read 18/11) | `ngnnguynchn/flag2027-rfsfxc5rl-art-p` (stage `t163p_c5rl`), `thanhhuyen11/flag2027-rfsfxfvb-art-p` (stage `t163p_fvb`, group 60) | FACTOR_ENFUSE 16.99, BEST4 16.72, **13.54** |
| LORAFIX-01 L1 (team, fixed key) | 16 371 clips of 5 457 persons (VoxCeleb2 16 161 + MAV 210) | `ngnnguynchn/flag2027-rfs-lorafix1-p` (run `ngnnguynchn/flag-2027-nb-163-rfs-lfix1`, `RFS_LORA_FIX=1`) | different | `ngnnguynchn/flag2027-rfsc4lf-art-p` (CELL4-01 `lf`) | `xjf_r10_trsig_lf.zip`, `posteval_CELL4_ENFUSE_lf.zip` (not the displayed submission) |
| C1FIX-01 (teammate kit, fixed key) | 16 161 clips of 5 387 persons (VoxCeleb2 only) | `ngnnguynchn/flag2027-c1fix-lora-p` (`c1fix_train.py` in `Experiment/C1FIX-01/kit/C1FIX01_kit.zip`) | different | `ngnnguynchn/flag2027-c1fix-rfx-art-p` | 16.18 |

Source of the table: repo map S4 section 1.8, re-checked on 18/11 where marked; LORAFIX-01 / C1FIX-01 rows are taken from S4 (not re-counted).

## Where the original variant comes from (stage `t163l_orig`)

- Run `onnhqunhh/flag-2027-nb-163-rfs-l` (start 2026-11-03 17:37:31 UTC, GPU T4 x2). Log: `lora data: 210 clip, 70 nguoi (vox 0, mav 210)`, 1 200 steps, loss 3.4451 -> 0.0803.
- Code that ran: `t163l_orig/original/decoded/train/rfs_train.py` (sha256 `98aa07ae529c0afc8aa36f304595cab805b137b6386cd4d3044984fe77036562`).
  `lora_data()` line 406 builds the audio-cache key as `(v_spk, v_vid, v_utt)`; in the bundle `v_vid = "spk/video"` (`train/assemble.py`
  lines 45-46 of the embedded copy), while the cache (`duongphihoang/flag2027-audio-cache-p`) stores the bare video id: no VoxCeleb2 clip matches.
- MAV root: the notebook takes the parent of the first folder named `v1` / `v2` that holds `faces` + `voices`. On Kaggle the dataset
  `chnnhnnguyn/external-data-mavceleb` has two parent folders (`mavceleb_v1_complete-001/v1`, `mavceleb_v2_complete-002/v2`); the run took
  the v1 parent, so every `v2/...` wav path failed to load and was skipped (`load_wave` returns `None`). Evidence: 70 persons = the 70 MAV v1
  identities (MAV v2 has 84; counted in `kaggle/output/feats_ext_full/ext_v{1,2}_complete_voice_meta.csv`), 210 = 70 x 3 videos.
- The stage keeps the bug and adds two guards (see `t163l_orig/CHANGES.md`): folders visited in name order (v1 first on any mount) and a
  dry run of the original `lora_data()` that must give exactly 210 clips / 70 persons / all `mav:v1` before training starts.

## Every artifact of the 13.54 chain that carries a LoRA C1

| artifact | how the original C1 is checked |
|---|---|
| `onnhqunhh/flag2027-rfs-lora-p` | sha256 above (E: package zip `14ddbcf579e051114ad9302b6512572ee81ff26447baea4e7bdfdeb545353596`, member hashes computed 18/11) |
| `ngnnguynchn/flag2027-rfsfxc5rl-art-p` | member `lora_C1.pt` = `b97372e3...` (E: package zip `12da5398...`, computed 18/11); `manifest_final.json` lists the same sha256 |
| `t213_p1`, `t213_p2`, `t216_p3` (sigma source) | their `kernel-metadata.json` attach `onnhqunhh/flag2027-rfs-lora-p` (copied in `original/`); in the repo: input `resolve: saved_only` + `require_files`, `verify` `work/lora_C1.pt` (check `exact`); sigma uses only the `t3` / `nf` / `nv` streams, so it does not depend on C1 anyway |
| `thanhhuyen11/flag2027-rfsfxfvb-art-p` (group 60) | NOT verified here: artifact not downloaded (DESIGN G4); push staging of `t163p_fvb` attaches `onnhqunhh/flag2027-rfs-lora-p` (WP-H) |

## Places where today's code switched to a fixed C1 (do NOT use them for 13.54)

Checked on 18/11 in the repository:

- `release/flag2027_e2e/stages.py` (the 16.18 package): stage `c1fix_lora` lines 234-297 (teammate's `c1fix_train.py`, not the L run);
  `c1fix_rfx` exports `C1FIX_ART` at line 305 and mounts `C1FIX_LORA` at the path of `onnhqunhh/flag2027-rfs-lora-p` at lines 318-319;
  `e163i` mounts `C1FIX_ART` at the path of `ngnnguynchn/flag2027-rfsfxc5rl-art-p` at lines 526-527.
- `kaggle/rfx_chain.sh` line 11: `LORAD=ngnnguynchn/flag2027-rfs-lorafix1-p`.
- `kaggle/sat_push.sh`: every `LABDS=` line (984, 1005, 1028, 1075, 1095, 1125, 1144, 1166, 1198, 1228, 1248, 1293) names `lorafix1`;
  line 1168 rewrites it per account. The runs 183, 213, 216 were pushed before that change and attached the original dataset.
- `kaggle/build_flag173.py` line 6 (docstring of the lab builder) names `lorafix1`.
- `Experiment/RFS-01/train/rfs_train.py` (sha256 `2f79b0045db8bdc5...`, not in git): line 959 `RFS_LORA_FIX` defaults to `"1"` (fixed). The
  embedded versions used by the runs of this group have either no switch (L, P, FACTOR lab) or an opt-in switch that is off (213 / 216 lab).
- Documents that say "use the fixed LoRA": `release/factor_enfuse/README.md` line 136, `docs/DATASETS.md` line 323.
- `manifest_final.json` of the RFX artifact (field `declared`) says the LoRA C1 was trained on "cleaned VoxCeleb2 + MAV clips"; for the
  original C1 the true statement is "210 MAV-Celeb v1 clips of 70 persons (VoxCeleb2 clips were not matched because of a cache-key bug)".
  The system description must use the true statement.

The runner's variant guard (DESIGN 2.7) refuses `ngnnguynchn/flag2027-rfs-lorafix1-p`, `ngnnguynchn/flag2027-c1fix-lora-p`,
`ngnnguynchn/flag2027-c1fix-rfx-art-p`, `ngnnguynchn/flag2027-rfsc4lf-art-p` (and the dev-derived `ngnnguynchn/flag2027-rf092-dev-emb-p`)
in any stage of this group; every `stage.json` here lists them under `forbidden_inputs`.

## How the 13.54 chain is kept on the original C1 (fixer 18/11; details in `README.md` 'LoRA C1 policy')

- The four consumers (`t163p_c5rl`, `t213_p1`, `t213_p2`, `t216_p3`) mount `onnhqunhh/flag2027-rfs-lora-p` with
  `"resolve": "saved_only"` (the saved original, in every run mode and in smoke) and require `lora_C1.pt` = `b97372e3...` on that input
  (`variant.require_files`, checked by the runner before the stage runs). Their `verify` entry (check `exact`) compares the copy the code
  writes (`rfs_art/lora_C1.pt`, `work/lora_C1.pt`) with the same hash. `t163b` and `t219_dumpv4` do not mount a C1.
- `t163l_orig` produces the C1 but is `force_only`: mode (a) keeps the saved original (a GPU re-train is not byte-identical). A forced
  re-train is accepted on `ff_lora_drycheck.json` (210 / 70 / all `mav:v1`) and `ff_lora_meta.json` (48 params, 1 200 steps, batch 64,
  210 clips, 70 persons, 0 skipped), never on the sha256, and is not used by the consumers.
- No `post_checks` entry carries a hash: the runner reads post_checks as JSON.
