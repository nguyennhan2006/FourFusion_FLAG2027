# Stage group 30_F_family: the identity-classification family F1s′

F1s′ is the identity-classification family of the system: a logistic fuser over seven out-of-fold score streams (`XALL_deep0`,
`XALL_cca`, `XALL_ib`, `XALL_age`, `anP512_deep`, `anP512_cca`, `trCA`) and seven gender / age / language conditions, learnt on the SAFE-04
validation pairs (37 110 pairs, 5 folds split by person). It enters all four files of the evaluated submission (13.54). The fuser itself and
the Evaluation scoring are fitted later in `50_eval` (`e50cc`, `e112f1sx`); this group trains everything those stages load. No stage here
uses the ImageBind-audio LoRA C1 (the variant question of 13.54 concerns the RFX family only, `40_R_family`).

**NOREL.** The 13.54 fuser uses no train-cohort-normalised stream: `e50cc` / `e112f1sx` run with `NOREL=1`, which drops `rel`, `relg` and
`t3_relg`. The stages of this group still COMPUTE such columns as the original runs did (`t164_f123` / `t164_f45` write `relc_fia_eb`,
`relg_fia_eb` and `t3_relg` into the CLEANCOMP tables; the frozen / rebuilt e2e-dev bundle has `rel` / `relg`); they are present in the
tables and excluded downstream.

| order | stage | what | source (exactness) | role in 13.54 | machine, original minutes |
|---|---|---|---|---|---|
| 1 | `t153b` | LoRA adapter B3s on CLIP ViT-L/14 + `clipB3` of every train / auxiliary face row | executed notebook, package E (exact-package) | wiring | GPU T4 x2, 494 |
| 2 | `t153c` | LoRA adapter C5s + `clipC5` of every train / auxiliary face row | exact-package | score | GPU T4 x2, 712 |
| 3 | `safe04_lab` | rebuild every SAFE-04 reference table of the e2e-dev bundle (level full) from lab-data + eda-bundle | new (original lab scripts in `code/`) | optional (default: frozen bundle; `FF_FORCE=safe04_lab`) | GPU, several hours (unverified) |
| 3b | `safe04_lab_core` | partial rebuild (keys, pf_m / pv_m, stack02_rows, fold_persons) spliced into the frozen tables | new | evidence | CPU |
| 4 | `safe04_compare` | rebuilt (level full) vs frozen tables | new | evidence | CPU |
| 5 | `t164_f123`, `t164_f45` | CLEANCOMP-01: XALL bridge + v11 T3 head per fold (LID_STRICT=1) | executed notebook pulled from Kaggle (exact-pulled) | score | GPU, 16 / 12 |
| 6 | `cleancomp_tables` | SAFE-04 tables with the CLEANCOMP-01 columns | local script `build_tables.py` as a stage | score | CPU, 1 |
| 7 | `t164f` | final XALL member `s007_xall.pt` + v11 T3 head | exact-package | score | GPU, 9 |
| 8 | `t154_g1_f12/f34/f5`, `gather_g1` | CLEANTUNE-02 g1 fold tables (B0s head) | builder output, not the executed notebook | wiring, optional (default: frozen) | GPU |
| 9 | `t155s_f12/f34/f5` | CA head per fold → `trCA` out-of-fold | exact-pulled | score | GPU T4 x2, 78 / 79 / 69 |
| 10 | `gather_r2ca` | gather the five `ct01r2` tables | local gather step as a stage | score | CPU |
| 11 | `t112sf0s` | B0s head `ct01g1_final_models.pt` (5 seeds) | exact-package | wiring | GPU T4 x2, 60 |
| 12 | `t112sf1s` | final CA head `ct01r2_final_models.pt` (10 seeds) | exact-package | score | GPU T4 x2, 154 |

Each stage folder holds `stage.json` (inputs / outputs as original Kaggle artifact ids, flags, smoke settings, saved-artifact
fallback `skip_with`, reference sha256 in `verify`, `todo`), `stage.ipy` (the cells that run, separated by `# <<<FF-CELL>>>`),
`CHANGES.md` (every difference from the original, with the reason and the effect on numbers) and `original/` (`SOURCE.json` with md5 /
sha256 of every source; the executed notebook, its log and Kaggle `kernel-metadata.json` where available).

**Literal /kaggle/input (runner requirement).** Eleven stages (`t164_f123`, `t164_f45`, `t164f`, `t155s_f12/f34/f5`, `t112sf0s`,
`t112sf1s`, `t154_g1_f12/f34/f5`) decode lab modules from a base64 zip at run time; those modules find lid_strict.json, the adapter
arrays, feats_vox2b, the CT voice blocks and components_dev.npz by scanning the LITERAL path /kaggle/input, which the runner's string
rewrite cannot reach. Their `stage.json` declares `runner_requirements.literal_kaggle_input: true` (modules and lines in `why`). On Linux /
Docker the runner points /kaggle/input at the stage's own inputs. **On Kaggle these stages must run in a session with exactly their inputs
attached** (as the original runs); the runner can only warn there (`FF_LITERAL_STRICT=1` makes it an error).

**Dev data, stage by stage.** No trained weight of this group depends on dev data. What still touches it:
- `t153b`, `t153c`: since the 2026-11-18 fix train-only by default (`SPLIT_FILTER = ["train"]`, `dev_set.zip` never unpacked); the executed
  runs also embedded the four dev lists with the trained adapter.
- `t112sf0s`, `t112sf1s`: train-only by default (`FF_TRAIN_ONLY` unset / 1; inserted cell + patches): the final head is trained and saved,
  no dev list is scored, no dev zip is written. `FF_TRAIN_ONLY=0` restores the executed behaviour (dev scoring of the four dev lists and
  the F0 / F1 dev zip; it needs dev-list arrays that the train-only rebuilds of t153b / t153c / f47b_screen2 no longer make).
- `t164f`: reads `components_dev.npz` of the e2e bundle and scores the four dev lists with the new v11 head after both models are saved
  (verbatim, inside the embedded final.py); the resulting `v11_head/t3_streams.npz` stays in the kernel output and is not exported.
- `safe04_lab` (level full) copies the dev wiring of the frozen bundle (artifact `datasets/ff-frozen/flag2027-e2e-dev-wiring-p`:
  components_dev.npz, e2e_manifest.json, fusers/, ref/) byte for byte; `cleancomp_tables` copies the same files from the e2e-dev bundle
  into `flag2027-cleancomp-e2e-p`.
- `t155s` × 3, `t154_g1` × 3, `t112s*`, `t164f`, `cleancomp_tables`, `safe04_compare`, `safe04_lab_core`: mount an e2e bundle that holds
  those dev wiring files; the lab stages check that `components_dev.npz` exists and read `e2e_manifest.json` for its split tag; their
  fold / final jobs read only the `safe04/` tables (SAFE-04 validation pairs of train persons).
- Redaction: five executed notebooks (t155s × 3, t112sf0s, t112sf1s) and the g1 builder output carried, inside their embedded code zip, a
  file of per-pair scores on the organisers' DEV lists; the repository copies are redacted by `_common/redact_embedded_dev_zip.py`
  (byte-level, deterministic; no number changes). The byte-exact notebooks belong in the private data store.
Each `stage.json` `dev_usage` gives the details.

**Identity linkage (to disclose).** Every stage that loads the embedded `kaggle/build_l0_notebooks.py` (t164*, t155s × 3, t112s*, g1 × 3,
and the lab code of safe04_lab) removes two MAV-Celeb identities matched to v4 persons: `OVERLAP = {"v1_complete:shaukat_tarin",
"v2_complete:id0073"}` (line 92; SET-07), `PERSONS = sorted(set(VMETA.p) - OVERLAP)` (line 153). This is a link between two sources;
`stage.json` `identity_linkage` records it for `docs/COMPLIANCE_NOTES.md` (DESIGN G17); the decision on it is open.

**Licence.** Outputs of t153b / t153c (adapters trained on VoxBlink frames, `vb_face_clip*.npy`), of t112sf0s / t112sf1s (heads trained
with 13 471 VoxBlink speakers) and the training inputs of t155s are VoxBlink-derived (CC BY-NC-SA 4.0, no redistribution): private
FF_DATA only, never in git (`stage.json` `licence`).

**Check.** `python _common/check_group.py` rebuilds every `stage.ipy` from `original/` + `patches` + `insert_cells`, checks every sha256,
parses every cell, checks the `[cell]` rule of new cells, forbidden inputs and secrets (also inside the decoded CODE_B64 members of the
notebooks); `--exact` also re-makes the redacted notebooks from the exact ones when they are readable (a 'skipped' line otherwise).
