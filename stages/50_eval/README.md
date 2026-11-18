# 50_eval: Evaluation path of F1s′ + RFX and the FACTOR_ENFUSE zip

Seven Kaggle runs on the organisers' Evaluation data (all on one account in November 2026) and one CPU step. Their product,
`posteval_FACTOR_ENFUSE_c5rl.zip`, gives three of the four files of the 13.54 submission (stage `final_blend`, group `80_final`):
the no-gender Bangla file byte for byte, and both English files before the DP (Different Profile) blend. The gender Bangla file comes
from group `60_fvbind`, which also reads the outputs of `e47rs`, `e65xcs` and `e112f1sx`.

| stage | original run (`notebooks/<account>/<slug>`) | role in 13.54 | machine, time of the run | reads (stages) |
|---|---|---|---|---|
| `e65xbs` | `thanhhuyen11/flag-2027-nb-65xbs-b3sx-eval` | wiring | Kaggle T4, 16 min | — |
| `e65xcs` | `thanhhuyen11/flag-2027-nb-65xcs-c5sx-eval` | score (CA face block, RFX `f_c5s`) | Kaggle T4, 21 min | — |
| `e47rs` | `thanhhuyen11/flag-2027-nb-47rs-screen2-eval` | score (CA voice CAT4, RFX `v_rl`) | Kaggle T4, 240 min | — |
| `e50cc` | `thanhhuyen11/flag-2027-nb-50cc-e2e-eval` | score (6 F1s′ streams, 7 conditions) | Kaggle 2 x T4, 221 min | — |
| `e112f0sx` | `thanhhuyen11/flag-2027-nb-112f0sx-eval` | wiring | Kaggle CPU, 2 min | e65xbs, e50cc |
| `e112f1sx` | `thanhhuyen11/flag-2027-nb-112f1sx-eval` | score (trCA, F1s′ fuser, f1s_eval.zip) | Kaggle CPU, 4 min | e47rs, e65xbs, e50cc, e65xcs, e112f0sx |
| `e163i` | `thanhhuyen11/flag-2027-nb-163i-rfsfxc5rl-eval` | score (RFX, ORIGINAL LoRA C1 artifact) | Kaggle T4, 31 min | e47rs, e65xcs (+ RFX artifact of `t163p_c5rl`) |
| `a_factor` | local CPU step of 13/11/2026 12:00 (`kaggle/rfx_chain.sh` state 3) | score (U and ENFUSE zips) | CPU, seconds | e112f1sx, e50cc, e163i |

Each stage folder follows DESIGN §2: `stage.json` (inputs with the `/kaggle/input` mount the run saw, flags, patches, variant guard,
hashes of the saved outputs), `stage.ipy` (the code cells, separated by `# <<<FF-CELL>>>`), `CHANGES.md` and `original/`
(the exact executed notebook of the team package, its Kaggle log, the `kernel-metadata.json` of the exact push, `SOURCE.json`).

**Code fidelity.** For `e112f0sx`, `e112f1sx` and `e163i` the code is the run's code (`e163i`: byte-identical to it; the two 112
stages: only the dev payload of the redacted notebook differs). `e65xbs`, `e65xcs`, `e47rs` and `e50cc` carry one dev-exclusion patch
(`no_dev_zip`: `"dev_set"` removed from `ZIP_PREFIXES`). The repository stores the organisers' MAV-Celeb v4 release as zips
(`FF_DATA/00_raw/btc_mavceleb_v4`), while the original Kaggle runs saw an auto-extracted tree; without the patch the verbatim fallback
`unpack_zips()` would unpack `dev_set.zip` as well. Numbers are unchanged: `train_set.zip` is unpacked into `/kaggle/working/raw`, and
the stage's verbatim last cell deletes that folder. `e50cc` also replaces the dead ImageBind URL by the pinned Hugging Face mirror (same
file). `a_factor` runs byte copies of the scripts that ran (`tree/`) in a copy of the original repository layout. Every change is listed
in the stage's `CHANGES.md`.

**Where the three literal-path stages can run.** The code that `e50cc`, `e112f0sx` and `e112f1sx` decode at run time from base64 zips
scans the literal path `/kaggle/input` (`stage.json` `runner_requirements`: `literal_kaggle_input`, `kaggle_strict`). Run these stages
on Linux or in Docker, where the runner points `/kaggle/input` at the stage's own inputs, or in a dedicated Kaggle session that has
exactly the stage's inputs attached, as the original runs did. Do not run them inside one Kaggle orchestrator session with other
attachments: the decoded modules take the first match of their own scan, which can be a saved copy or a dev-list array instead of the
array recomputed in the session. Until the runner enforces `kaggle_strict`, set `FF_LITERAL_STRICT=1`.

**Dev-selection disclosure.** The XALL streams that enter the 13.54 numbers come from `s007_xall.pt`, the XALL bridge built on the
's007' member recipe. That recipe was first chosen per cell with dev pseudo-labels (FUSE-01 / SEL-01b, flagged in the team's decision
log). The copy used here was retrained on train / clean data in CLEANCOMP-01 (stage `t164f`), and `e112f1sx` refits the F1s′ fuser on
SAFE-04 validation tables only (details in `e50cc/stage.json` notes). The v6 member seeds of `flag2027-members-p` are a different thing:
they feed only English v9 / v12 arrays that no 13.54 fuser reads.

**No dev data in this repository.** Three exact notebooks embed dev-phase score files that the Evaluation branch never reads:
`e50cc` (three reference zips of the dev submissions) and `e112f0sx` / `e112f1sx` (one dev score zip inside the base64 code zip).
`original/` holds redacted copies, produced deterministically by `_tools/redact_dev_refs.py`; each `CHANGES.md` says what was removed
and why the numbers cannot depend on it. The exact notebooks (sha256 in `stage.json` `origin.exact_notebook_sha256`) are kept in
private storage only; with them, anyone can check the redaction (`redact_dev_refs.py <stage> <exact> --check <redacted>`).
Some inputs that are mounted but never read also hold dev-phase score files: `flag2027-best-v7-p` (for `e50cc`) and `evalu-f0s-p` /
`evalu-f1s-p` (for the 112 stages). Each `stage.json` declares them under `dev_data` and declares what each run output holds under
`outputs.contains` (dev data, VoxBlink-derived arrays or weights). In the staging tree, two copies of the pushed `e50cc` notebook under
`_staging_backup/` still embed the dev zips and are not git-ignored. They must be moved before any commit (`e50cc/stage.json` todo[0]),
and the staging scan of the selfcheck fails until they are.

**13.54 variant.** `e163i` mounts `datasets/ngnnguynchn/flag2027-rfsfxc5rl-art-p`, whose `lora_C1.pt` is the ORIGINAL ImageBind-audio
LoRA C1 (sha256 `b97372e3504a131d…`), and never the C1-fix artifacts of system 16.18 (`stage.json` `variant`). The sha256 pins
apply to saved or attached copies (`applies_to`, honoured by `runner/ffcore.py`). In `FF_MODE=train` the projector is re-learned
(new `manifest_final.json`), but the C1 inside the artifact is always the saved original: every C1 consumer has
`resolve: saved_only`, so a C1 re-learned by `t163l_orig` is evidence only. A made artifact is checked by provenance
(`variant.made_check`: made in the session by `t163p_c5rl`, `lora_C1.pt` = `b97372e3…`, `RFS_LORA_FIX` unset; 19/11, INTEGRATION_r2 M3).

**Checks** (read-only, seconds):

```
# public check (everything it needs is in the repository)
python stages/50_eval/_tools/selfcheck_50_eval.py
# team / private storage only: needs the exact notebooks and release/flag2027_e2e/code, which are not published (dev data)
python stages/50_eval/_tools/selfcheck_50_eval.py --exact-dir <FACTOR_ENFUSE_c5rl/notebooks/2_eval> --release <release/flag2027_e2e/code>
```

The public form rebuilds every `stage.ipy` from `original/` + patches and checks all recorded hashes and the variant guard. It also
checks that no `ZIP_PREFIXES` line unpacks `dev_set.zip` and that no secret-like string or dev score payload is present. In the staging
tree it also scans `_staging_backup/` outside the git-ignored private folder (`--skip-staging` leaves this out; the published repository
has no such folder). The team form also re-does the redactions from the exact notebooks and compares the code with
`release/flag2027_e2e`. The `release_e2e_code` fields of each `stage.json` are provenance (`ships_with_repo: false`).
`stages/50_eval/.gitattributes` (`* -text`) keeps every hashed file byte-exact across git checkouts. Re-running the GPU stages does not reproduce bytes (GPU / library nondeterminism); `a_factor` with
the saved six inputs reproduces every member of both zips (asserted when `FF_MODE=assemble`).
