# CHANGES: safe04_lab (new stage; no Kaggle run of its own ever existed)

The SAFE-04 lab of October 2026 ran on the local CPU; its scratch folder was deleted on 1/11/2026, and the zip it was packed into
(`kaggle/flag2027-e2e-dev.zip`, sha256 `0257b1fe...`) is the only copy left. This stage re-runs the ORIGINAL scripts (code/, verbatim) at
level **full** and is the producer of `datasets/ngnnguynchn/flag2027-e2e-dev-p`. Since the 2026-11-18 fix the cheap partial rebuild
(level core) is a separate evidence stage, `safe04_lab_core`, so that the producer takes no table from a frozen copy. What differs from
the October procedure and why:

1. **Script versions** (type `pin`; numbers: unverified, believed none). The October file versions are not preserved. code/ holds:
   (a) the versions embedded in the executed notebooks of 3/11 (FLAG_164 / FLAG_111 runs; `code/CODE_MANIFEST.json` 'source'): every
   one equals today's repository file except `Experiment/SAFE-04/stream_gate.py` (the repository version later gained an opt-in NOREL
   branch; the 3/11 version is used); (b) four files that no executed run embeds, copied from the repository working tree, where they
   are untracked (`??`) and of unknown date: `Experiment/CAFUSE-EN/run2.py` (produces cafuse_en2_fold*, run at this level),
   `kaggle/build_e2e_dev_bundle.py` (only `safe04_tables()` is called: it selects the bundle columns), `Experiment/STACK-02/export.py`
   and `Experiment/STACK-02/pools.py` (reference only, not run). Their sha256 are in `code/CODE_MANIFEST.json`; they are unverified
   against the October 2026 versions. The known changes since October are opt-in flags (LID_STRICT / MAV_DROP, NOREL, CLEANCOMP, BN42);
   the stage removes them from the environment, because the October tables predate them and the 13.54 chain used exactly those old
   columns.
2. **Level full only; the producer takes no frozen table** (type `wiring`; 2026-11-18 fix). Every table of `e2edev/safe04/`
   (streams, relonly_multi, relonly_g, t3enc01f, bigb01f, cafuse_en2 per fold, stack02_rows, fold_persons.json) comes from the scripts
   of this run. The only producer-less input is the dev wiring of the frozen bundle, a separate artifact
   `datasets/ff-frozen/flag2027-e2e-dev-wiring-p` (the 9 members components_dev.npz, e2e_manifest.json, fusers/*.json, ref/*.zip of
   `kaggle/flag2027-e2e-dev.zip`, sha256 per file in `stage.json` `dev_wiring`; class wiring, dev data). The stage refuses a wiring
   folder that holds `safe04/` tables. Outside smoke all five folds are required. Previously the stage took the whole frozen bundle and,
   at its default level core, kept the XALL_* / CTRL_* columns and the other tables frozen. Under NOREL=1 + CLEANCOMP=1 they feed no F1s' fuser input of 13.54 (cleancomp_tables replaces XALL_* / relc_fia_eb / relg_fia_eb / anP512_* / t3_relg; NOREL drops rel / relg / t3_relg); they feed the F0 English fuser (CTRL_ib / CTRL_age through EN-T3-01 table_en + cafuse_en2 in CLEANTUNE-01 fit_en_clean, the t112sf0s / e112f0sx zip step) and wiring checks (check_v11_fuser). Whether any 13.54 number uses the F0 English file is for WP-G to confirm (WP-E's reading: no, the FACTOR_ENFUSE English scores come from the Bangla fuser of F1s'). Either way a core output is partial and must not stand in for the bundle. `notebooks/ff/f12_eval_feats` is an input of
   this stage only (no longer of the cheap level).
3. **stack02_rows reduced to the five columns the bundle keeps** (type `wiring`; numbers: none). `Experiment/STACK-02/export.py` needs the
   tables of ten other experiments (pools.py TABLES); the bundle keeps only `fold, data, y, fp, vp` (build_e2e_dev_bundle.py:91).
   `ff_derive_stack02.py` computes them with export.py's fp formula and stream_gate.load's vp formula, and fold_persons.json with
   export.py's formula (`indent=0`, LF).
4. **Run order and arms set explicitly** (type `wiring`): T3ENC-01 with `T3_ARMS=B` (T3ENC-01 README line 124, the bundle has t3B_* +
   t3_rel / t3_relg), BIGB-01 with `AN_ARMS=P256,B2x,P512,P256x2,P1024` (the five arms of the bundle's bigb01f; arm B is absent from it).
   All scripts run as `python -u <script>` (as locally); BIGB-01 originally ran through lab_entry.py on a Kaggle GPU (same code path;
   lab_entry only deletes the ECAPA-6144 placeholder stream, which BIGB-01 never reads).
5. **Real ECAPA-6144 arrays from f12_eval_feats** (type `path`). lab-data holds `(rows, 0)` placeholders; CAFUSE-EN/run2.py needs the
   real arrays, which the October lab had in kaggle/output/feats_eval (FLAG_12). The stage links the real files from the FLAG_12-round
   stage.
6. **Dev wiring copied, never recomputed** (type `dev-exclusion`). components_dev.npz, fusers/, ref/ and e2e_manifest.json come byte
   for byte from the dev-wiring artifact (build_e2e_dev_bundle.components() reads dev features: not called). fusers/v11_bangla.json is
   used downstream only as the reference of `e2e_lib.check_v11_fuser()` (an assertion on SAFE-04 table rows). A new file
   `e2edev/SAFE04_REBUILT.json` records level, folds, completeness, per-table provenance and the in-stage checks; the comparison with the
   frozen tables is the job of `safe04_compare`.
7. **Smoke** (`FF_SMOKE=1`): fold 1 only, `L0_STEPS=10`, `L0_NETS=1` (= the smoke settings of the FLAG_164 runs); the output holds the
   tables of fold 1 only, is marked `complete: false` and skips the 37 110-pair assertion (safe04_compare reports it as PARTIAL).

Every cell of stage.ipy is new and opens with `print("[cell] ...", flush=True)`; subprocess output is streamed line by line.
The same stage.ipy text serves `safe04_lab_core` (env `FF_SAFE04_LEVEL=core`; code read from `../safe04_lab/code`).
