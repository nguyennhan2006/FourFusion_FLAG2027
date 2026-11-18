# runner/ — how the 13.54 chain is executed

`runner/ffrun.py` plans and runs the stages of `stages/` (and `tools/stages/`) from the raw data to the submitted zip. It is derived
from the one-notebook orchestrator of the team repository (`release/flag2027_e2e/orch.py`) and extended with a data map
(`configs/data_map.json`) so that the same code runs on a Kaggle notebook, a Linux machine or in a container.

| file | role |
|---|---|
| `ffcore.py` | the engine: stage loading, artifact lookup, planning, virtual `/kaggle/input`, per-stage process, output collection, 13.54 variant guard, checks |
| `ffrun.py` | command line: `plan`, `run`, `dag`, `locate`, `status`, `check-stage`, `verify`, `docs` |
| `check_stage.py` | rebuilds every `stage.ipy` from `original/` + the changes in `stage.json` and compares (exact / builder stages); checks the cell rule of new stages |
| `verify.py` | SHA-256 of every member of a zip against `configs/expected_sha256.json`; Spearman / largest difference between two score zips |
| `build_kaggle_notebook.py` | writes `notebooks/FourFusion_1354_{infer,train,assemble}.ipynb` and the code bundle (a Kaggle dataset) they load |

## Quick start

```bash
# (c) byte-exact zip from the saved per-pair scores (CPU, < 1 min)
FF_DATA=/data/FourFusion_FLAG2027_data FF_WORK=/work python runner/ffrun.py run --mode assemble
python runner/verify.py zip /work/art/notebooks/ff/final_blend/zips/xjf_r10_trsig.zip --expected xjf_r10_trsig

# (b) from the organisers' Evaluation media + the released checkpoints (GPU, about 12 h)
FF_DATA=/data/FourFusion_FLAG2027_data FF_WORK=/work python runner/ffrun.py plan --mode infer     # what runs, what is missing
FF_DATA=/data/FourFusion_FLAG2027_data FF_WORK=/work python runner/ffrun.py run  --mode infer

# (a) everything from the raw + collected data (GPU, about 150-170 h): one session per call, split with --only
FF_DATA=... FF_WORK=... python runner/ffrun.py run --mode train --only t153c
```

`plan --assume-saved` plans as if every artifact the mode may take from `FF_DATA` were present (useful before the data is in place).
On Kaggle use the notebooks in `notebooks/` (they call the same commands).

## Configuration

Environment variables (or the same-named options of `ffrun.py`): `FF_DATA` (data roots, `os.pathsep`-separated; the layout of
DESIGN 3 / `configs/data_map.json`; the team package layout `datasets/<owner>__<slug>[.zip]`, `outputs/<slug>/` and the Kaggle layout
`datasets/<owner>/<slug>` are also searched), `FF_WORK` (default `./ff_work`), `FF_MODE` (`train | infer | assemble`), `FF_TARGET`
(default `final_verify`), `FF_ONLY`, `FF_FORCE`, `FF_SMOKE`, `FF_USE_SAVED` (`all | collected | none`), `FF_TIME_BUDGET_H` (11.3 on
Kaggle), `FF_VARIANT` (`1354`), `FF_PIP` (`original | pinned | skip`), `FF_MODELS` (pretrained folder, `auto`, `off`), `FF_KIN`,
`FF_TMP`, `FF_KEEP_VIN`, `FF_NO_DEV=1` (never mount the organisers' dev files: `dev_files` of `configs/data_map.json`, e.g.
`dev_set.zip` of raw-v4; the smoke-without-dev test of DESIGN G11), `FF_CACHE_LINK` (`auto | symlink | copy`, how the model cache is
seeded), `FF_PIP_SCOPE` (`all | v173`, see `docker/ENV_PER_STAGE.md`), `FF_HF_OFFLINE=1` (pinned Hugging Face revisions),
`FF_SEED_HOME_CACHES=0` (do not copy CLIP / insightface models into the home folder). Literal-path stages: `FF_LITERAL_KAGGLE=mounted`,
`FF_LITERAL_STRICT=1`, `FF_LITERAL_DETECT=0` (see below). On Kaggle `FF_DATA` defaults to `auto`: every attached folder holding
`00_raw/` ... `06_pretrained/` (the Drive layout) becomes a data root.

## Where an artifact comes from

1. made by this runner: `FF_WORK/art/<id>/_DONE.json` (an artifact made in smoke mode is never used by a real run);
2. a saved copy under `FF_DATA` (path, `alt_paths`, the stage's `skip_with`, team package layout, Kaggle layout, an earlier
   `FF_WORK/art`), if its class may be taken from a saved copy in this mode (`configs/modes.json`):
   raw / collected / pretrained / wiring always; intermediate / checkpoint in `infer`; everything in `assemble`; in `train` only
   with `FF_USE_SAVED=all`, plus artifacts flagged `default_saved` (the frozen SAFE-04 tables);
3. a dataset / kernel output attached to the Kaggle notebook (matched by slug, any owner), or an earlier session's `FF_WORK`;
4. the stage that produces it (never a `collect` stage unless forced, never a `train` stage in `infer` except `dp_train`, only
   `assemble` stages in `assemble`, never a `force_only` stage unless forced).

An artifact with no producing stage must be in `FF_DATA` (the plan says where).

**Requirement 1 (start from raw data) is not met everywhere.** `python runner/ffrun.py dag` (and `plan`) print the list
"REQUIREMENT-1 GAPS": consumed artifacts that are not raw / collected / pretrained and that no stage of this repository produces
(the wiring datasets cc5-oof, dsel02-oof, facelora03-tables, lab-avfull, members, best-v7, feats-scout-wsvrdn; the frozen SAFE-04
tables and their wiring members; evidence-only: peface01-pe0 for t183_p1 / p2 and the t3s01 tables for t184_report), the artifacts
that the modes take frozen by default although a producer exists (`default_saved`: the SAFE-04 tables, the ORIGINAL LoRA C1, the
stored LID outputs; class wiring), and eleven data files carried inside stage code (`configs/data_map.json` class `embedded`, seven
without producer). Each entry carries `requirement1_gap` / `origin_run`. Mode (a) is therefore "from raw data except these frozen
exceptions" (DESIGN G1, G18). Since 18/11 (integration fixer r1) sigma no longer depends on a frozen intermediate: t219_dumpv4
reads `datasets/ff/flag2027-t3s01-trca-p`, made by stage t3s01_trca from ct02s-r2ca-tables-p. The widest rebuild is
`FF_USE_SAVED=none FF_FORCE=raw_mavceleb,safe04_lab,f14_av,gather_g1` (configs/modes.json `full_rebuild_from_raw`; 71 stages in
the plan, 79 with the optional collect stages).

## How a stage runs

- Inputs are mirrored into a private folder (directories + file symlinks; hard links or copies where symlinks are not allowed) at
  the path the original run saw (`mount`, default = the artifact id, i.e. `datasets/<owner>/<slug>`).
- `stage.ipy` cells run one by one in a fresh process (IPython `InteractiveShell`, so `!pip` and `%%writefile` work). Before each
  cell the runner prints `[cell] <stage> <k>/<n>: <first line>`; the full log goes to `FF_WORK/logs/<stage>.log` and a per-cell
  record to `FF_WORK/logs/<stage>.cells.jsonl`.
- Run-time rewrites (counted and stored in the artifact's `_DONE.json`): `/kaggle/input` and `/kaggle/working` -> the stage's own
  folders; `configs/pipeline_1354.json` `runtime_rewrites` (the dead ImageBind URL -> the pinned Hugging Face copy, same SHA-256);
  upload lines `os.environ['RFS_UPLOAD'] = '<slug>'` (and the other `upload_vars`) -> `pass  # disabled`; with `FF_PIP=pinned` the
  unpinned ImageBind git URL -> commit `53680b02…`. Kaggle credentials are never passed to a stage; `HF_TOKEN` only to stages that
  declare it (`compute.secrets`) or name it in their code.
- Embedded code: `runner/payloads.py` decodes every base64 payload of `stage.ipy`. A module default
  `os.environ.get("X", "/kaggle/input")` gets `X=<the stage's input view>` exported when the stage does not set `X` (recorded as
  `env_defaults_exported` in `_DONE.json`); any other literal `/kaggle/input` makes the stage literal (next section). The full list
  of run-time changes applied to every stage is `runner/RUNTIME_CHANGES.md`.
- Everything the stage leaves in its working folder becomes `FF_WORK/art/<kernel id>/`; `exports` are re-packed under the original
  dataset ids (a sub-folder name, or `{dst: src glob}`); links into the input view are dropped (as Kaggle's own output saving does).
- After the stage: the variant file checks (`lora_C1.pt`, next sections), then `post_checks` and `verify`. A malformed `verify`
  entry is recorded as FAIL / WARN, never raised; `expected` may be a member dict (zip), a bare SHA-256, a dotted JSON subset
  (`check: json`) or a `configs/expected_sha256.json#...` reference.
- Caution: inputs and the model cache are links where possible. A stage that opens an EXISTING input file for writing would write
  into `FF_DATA`; Kaggle inputs are read-only, and outside Kaggle the model cache is copied by default (`FF_CACHE_LINK=auto`).

## stage.json fields the runner reads

`id`, `group`, `title`, `kind` (`raw | collect | data | train | infer | assemble`), `role_in_1354`, `compute` (`device`, `gpus`,
`minutes_original`, `internet`, `secrets`), `inputs` (`artifact`, optional `mount`, optional `resolve: "saved_only"`), `outputs`
(`kernel`, `exports`, optional `verifies`: artifacts this check stage verifies, scheduled before their first consumer), `env`,
`smoke_env`, `neutralize_upload`, `pip_policy`, `variant` (`lora_c1`, `require_files`, `forbid_artifacts`), `forbidden_inputs`,
`skip_with` (saved copy of the kernel output: a folder, or flat files), `verify`, `post_checks`, `scratch`, `force_only`,
`runner_requirements.literal_kaggle_input`, `code` (file name, or `{"stage_ipy": name}`), `command` (argv list for a script-only
stage). Other fields are documentation. `patches`, `drop_cells`, `insert_cells` are applied by the work package when it writes
`stage.ipy`; `check_stage.py` re-applies them to `original/` to prove it.

`verify` entries: `sha256` (fatal for `assemble` stages or with `"check": "exact"` / `"strict": true`, informational otherwise:
a GPU re-run is not byte-identical), `saved_sha256` (informational), `members_sha256` or `members_sha256_from:
"configs/expected_sha256.json#<key>"`, `expect` / `values` (JSON subset), `key` + `equals`, `arrays_sha256` (npz arrays),
`bytes`, `spearman_ref` + `min_spearman`. `post_checks`: `file`, `json_path`, `equals` (always fatal).

## Stages that read the literal `/kaggle/input`

Some stages decode code from base64 zips at run time and that code reads the literal path `/kaggle/input`
(`runner_requirements.literal_kaggle_input`, e.g. e50cc, e112f0sx, e112f1sx, the F family; detected automatically for a stage that
does not declare it); `plan` marks them `[literal /kaggle/input: declared | detected]`, `check_stage.py` reports the undeclared ones.
An explicit `"literal_kaggle_input": false` with a non-empty `why` is honoured (integration fix 18/11, INTEGRATION_r1 M1): the stage
owner checked that every literal use is unreachable or redirected. This is the case of the nine stages that embed `rfs_train.py`
(t163p_c5rl, t183_p1 / p2, t213_p1 / p2, t216_p3, t163p_fvb, t189_b2, t192_fvbx): `_xload` falls back to the literal `/kaggle/input`
only when `RFS_XDIRS` is empty, and each of them sets `RFS_XDIRS = "/kaggle/input"` in its stage text (rewritten to the stage's
input view). `FF_LITERAL_HONOUR_FALSE=0` restores detection. The string rewrite cannot reach decoded code, so for the literal stages:
on Linux the runner points `/kaggle/input` (and `/kaggle/working`) at the stage's folders with symlinks while the stage runs (needs
a writable `/kaggle`: `sudo mkdir -p /kaggle && sudo chown $USER /kaggle`; empty directories there, as in the Docker image, are
replaced and restored); in a container that already mounts exactly the stage's inputs at `/kaggle/input`, set
`FF_LITERAL_KAGGLE=mounted`; on Windows such a stage is refused. **On Kaggle** the notebook's `/kaggle/input` cannot change:
the runner refuses the stage (StageBlocked) when any input was made in this session or is not attached at `/kaggle/input` (the
decoded code would not see it), and when the session holds attachments that are not inputs of the stage with `FF_LITERAL_STRICT=1`
(default in the generated notebooks) or `runner_requirements.kaggle_strict` (a Drive-layout dataset holding many artifacts counts as
extra; the code bundle holding the runner is allowed when it holds no score files). Run such a stage alone: a new session,
`FF_ONLY=<stage>`, exactly its inputs attached (plus the previous session's output for inputs made there). Mode (b) on Kaggle
therefore needs separate sessions for e50cc, e112f0sx, e112f1sx and t192_fvbx; on Linux or in a container one session runs the plan.

## Variant guard (configs/variant_1354.json)

Refused: any stage that mounts `ngnnguynchn/flag2027-rfs-lorafix1-p`, `…/flag2027-c1fix-lora-p`, `…/flag2027-c1fix-rfx-art-p`,
`…/flag2027-rfsc4lf-art-p` or `…/flag2027-rf092-dev-emb-p`; environment `XJF_BASE=lf` or `RFS_LORA_FIX=1`; code that sets them.
Every stage gets `RFS_LORA_FIX=0` (no-op for the original code). `stage.json` `variant.lora_c1` is free text: a value starting with
`original` or `not used` / `not_used` is accepted, a value naming a fix variant is refused (`dag` / `plan` run this check statically).

`lora_C1.pt` (DESIGN 5.1: a retrained LoRA is never bit-identical):

| where the file comes from | rule |
|---|---|
| an input taken from a saved / attached copy (any mode) | SHA-256 must be `b97372e3504a131d0a969e478a855868095c67db4b3c0c133ef770c00e40991c` (ORIGINAL) |
| any artifact in `FF_MODE=infer` / `assemble` | the same fixed SHA-256 (the chain only copies the saved LoRA) |
| made in this `FF_WORK` by the lineage root `t163l_orig` (`FF_MODE=train`, only with `FF_FORCE=t163l_orig`: `rfs-lora-p` is `default_saved`) | lineage: its dry check `ff_lora_drycheck.json` must equal 210 clips / 70 persons / all `mav:v1` (and `RFS_LORA_FIX` never 1); the new SHA-256 is recorded in `_DONE.json` (`variant_lineage`, `root_check: passed`) |
| made downstream in `FF_MODE=train` (t163p_c5rl, t189_b2, t192_fvbx, t163p_fvb, e163i, ...) | must equal a recorded lineage hash or a `lora_C1.pt` of the stage's inputs (propagation); `variant.require_files` pins with `applies_to: [saved, attached]` are skipped for made inputs; a `verify` hash on `lora_C1.pt` is downgraded to INFO when it equals the lineage hash |
| smoke runs | made files are reported only (placeholders); saved inputs still need the fixed hash |

`python runner/selftest.py` exercises these rules (and the `verify` shapes) on a toy repository with the real config.

## Self-checks after the review fixes (2026-11-18, evening)

- `python runner/selftest.py`: 8/8 cases (lineage root accepted and recorded, propagation accepted, own LoRA refused, failing dry
  check refused, non-original saved copy refused in `infer`, smoke reported only, `lora_c1` normalisation, `verify` shapes).
- `dag --compare-skeleton` on the 92 stage.json of that time: no cycle, duplicate, unresolved or forbidden input, 0 static variant
  errors, 0 skeleton differences; the REQUIREMENT-1 GAPS list is printed.
- `plan --assume-saved`: `infer` = the 19 stages of `configs/modes.json`; `assemble` = the five stages; `train` = 57 stages.
- `check_stage.py`: every exact / builder stage rebuilds; the only errors are the nine stages whose embedded code reads the literal
  `/kaggle/input` without declaring it (owners WP-F / WP-H); embedded data files match `configs/data_map.json`.

## Self-checks done when the runner was written (2026-11-18)

- `python runner/ffrun.py dag --compare-skeleton`: 90 stages, no cycle, no duplicate producer, no unresolved or forbidden input.
- `plan --assume-saved`: `infer` = the 18 stages of `configs/modes.json` (DESIGN 5.2 + `raw_btc`); `assemble` = `a_factor`,
  `a_fvbind`, `dp_sigma`, `final_blend`, `final_verify`; `train` = 63 stages, nothing blocked.
- `plan --mode assemble` against a scratch `FF_DATA` holding byte copies of the ten saved score files: the five stages, every input
  resolved (Evaluation run outputs as soft inputs, `dp_score` from `05_scores/dp`, the sigma source from `03_checkpoints/sigma_source`).
- `check_stage.py`: every exact / builder stage present at that time rebuilt byte-identically from its `original/` notebook.
- A toy two-stage chain (scratch folder) ran end to end: input mirroring, rewrites, IPython magics, `[cell]` lines, exports,
  downstream lookup, `verify` / `post_checks`, `FF_PIP=skip` shim, forbidden-artifact refusal; the two stages of `tools/stages/00_env`
  ran locally (`models_fetch` in smoke = dry listing).
- Not run: any stage of the chain itself (no Kaggle, no GPU, no training, no download).
