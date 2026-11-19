# CHANGES: safe04_lab_core (new stage, evidence only; split from safe04_lab on 2026-11-18)

New code only (no original): `stage.ipy` is the same text as `../safe04_lab/stage.ipy`, run with `FF_SAFE04_LEVEL=core`; the lab code
is read from `../safe04_lab/code` (verbatim copies, sha256 checked against `CODE_MANIFEST.json`). See `../safe04_lab/CHANGES.md` items
1, 3 and 4 for the script versions and the stack02_rows / fold_persons derivation.

What it does: `ff_core_streams.py` imports `lab_streams.py` unchanged and calls `make_fold` + `Extra` + `trial_extra` exactly as
`run_fold` does, without `arm_ctrl` / `L0` (keys use seeded local RandomStates and Extra its own RandomState(0) + deterministic lbfgs, so
they do not depend on the skipped bridge training). The rebuilt keys (asserted equal to the frozen tables), the Extra columns (acl, lb,
iba2, ibal, gen, pf_m, pv_m), stack02_rows and fold_persons.json are spliced into the `safe04/` tables of the frozen bundle
(`datasets/ff-frozen/flag2027-e2e-dev-p`, input with `resolve: saved_only`).

**This is a PARTIAL rebuild.** The XALL_* / CTRL_* columns of streams_fold and every relonly_multi / relonly_g / t3enc01f / bigb01f /
cafuse_en2 table stay frozen. Under NOREL=1 + CLEANCOMP=1 they feed no F1s' fuser input of 13.54 (cleancomp_tables replaces XALL_* / relc_fia_eb / relg_fia_eb / anP512_* / t3_relg; NOREL drops rel / relg / t3_relg); they feed the F0 English fuser (CTRL_ib / CTRL_age through EN-T3-01 table_en + cafuse_en2 in CLEANTUNE-01 fit_en_clean, the t112sf0s / e112f0sx zip step) and wiring checks (check_v11_fuser). Whether any 13.54 number uses the F0 English file is for WP-G to confirm (WP-E's reading: no, the FACTOR_ENFUSE English scores come from the Bangla fuser of F1s'). Either way a core output is partial and must not stand in for the bundle.
The output is evidence that the cheap part of the lab reproduces, nothing more: it goes to the kernel output `e2e_dev_core/` only, is never
exported as `flag2027-e2e-dev-p`, and copies no dev-wiring file. Smoke (`FF_SMOKE=1`): fold 1 rebuilt, folds 2-5 frozen and marked so.

Every cell opens with `print("[cell] ...", flush=True)`.

## Dev wiring optional (numbers: none) - added 19/11/2026

- **Where:** cell 1 (input check) and cell 4 (bundle packaging) of `stage.ipy`; `safe04_lab` `stage.json` input `datasets/ff-frozen/flag2027-e2e-dev-wiring-p` gets `"optional": true` (and loses `resolve: saved_only`).
- **Old:** level full required the dev wiring (dev-list files of the frozen bundle: `components_dev.npz`, `e2e_manifest.json`, `fusers/`, `ref/`; no producing stage) and copied it into the bundle.
- **New:** attached, it is copied byte for byte as before; absent, the bundle gets `e2e_manifest.json` with `kind` e2e-dev, `split` dev, `files` [] and `dev_wiring` false, and no dev-list file.
- **Why:** lineage of the 13.54 zip from raw data, and no dev data in the 13.54 chain. Readers of the bundle: the F-family training stages (bundle search accepts `safe04/` since 19/11; they read `safe04/` and the manifest's `created`), `cleancomp_tables` (copies every file), `e50cc` (copies `safe04/` from the bundle whose manifest kind is e2e-dev), `t164f` (dev scoring skipped without `components_dev.npz`).
