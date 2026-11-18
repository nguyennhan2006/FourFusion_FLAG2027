# CHANGES: a_fvbind

Kind local-script: the original steps were two shell commands on the team machine (no notebook). The stage code (stage.ipy) is new driver code; the logic is the six scripts under tree/, byte-exact copies of the team repository files.

Every change against the code of the original run (fields: type, where, old, new, why, effect on numbers). The runner also maps `/kaggle/input` and `/kaggle/working` to the stage's own folders and prints a `[cell]` line before every cell; neither touches the code below.

## 1. scripts run from a scratch copy of the original folder layout instead of the team repository root

- type: `path`
- where: stage.ipy cell 1
- old: the scripts and their inputs at fixed relative paths of the team repository (Experiment/..., release/...)
- new: the driver copies the six scripts and the six inputs to the same relative paths in a temporary folder and runs the same commands there
- why: keeps every script byte-exact (no path parameterisation needed) while making the stage independent of the team machine
- effect on numbers: none

## 2. stub Experiment/POSTEVAL-01/manifest.json in the scratch folder

- type: `wiring`
- where: stage.ipy cell 1
- old: ENFUSE-U/run.py appended the sha256 of ENF_OUT to the team's POSTEVAL-01 manifest
- new: it appends to an empty manifest {"zips": {}} in the scratch folder (copied to logs/ for audit)
- why: the script needs the file to exist when ENF_OUT is set; the team manifest is not an input of anything in the chain
- effect on numbers: none

## 3. optional knobs removed from the environment

- type: `variant`
- where: stage.ipy cell 1 (ENV)
- old: XT_W / ENF_W did not exist at the time of the build (added 15/11 09:33)
- new: XT_W, ENF_W (and stray ENF_RFS / ENF_U / ENF_OUT) are popped before each command
- why: with both unset the scripts compute the 1 : 1 sums of the 15/11 build
- effect on numbers: none

## 4. added one-step cross-check and member sha256 checks

- type: `save`
- where: stage.ipy cells 4-5
- old: -
- new: rebuild_final.py with rfx_eval.zip := rfsfxfvb_eval.zip; sha256 of every member against the 15/11 zips; a_fvbind_report.json
- why: the one-step script is the shorter route to the same zip; agreement is evidence that the reconstruction of the two hand-run commands is right; the chain output (cells 2-3) is the one kept
- effect on numbers: none

## 5. saved score files also searched in <repo>/assembly/inputs

- type: `path`
- where: stage.ipy cell 1, function flat_roots() (+ the FileNotFoundError message of find())
- old: $A_FVBIND_INPUTS, then <FF_DATA root>/05_scores, then a search of the stage's input view
- new: the same, with <repo>/assembly/inputs added after the 05_scores folders (repo = three levels above the stage folder FF_STAGE_DIR)
- why: DESIGN 5.3 names assembly/inputs/ as the place of the saved score files in a private copy of the repository (decision Q-D); assembly/assemble_exact.py already searches it
- effect on numbers: none (the input sha256 are checked as before)

## Not changed (deliberately)

- The arithmetic, pair order and %.8f printing of XTEAM-03/run.py, ENFUSE-U/run.py and rebuild_final.py.
- The --counts of the U command (392 / 802 / 416 / 716) as in the original command.
- Zip timestamps are the build time, so whole-zip sha256 differ from the originals; members are compared.
