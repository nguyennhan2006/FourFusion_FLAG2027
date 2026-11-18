# CHANGES: a_best4_ref

Kind new: a minimal assembler replacing `python Experiment/EVALPROBE-01/make_probes.py final`.

Every change against the code of the original run (fields: type, where, old, new, why, effect on numbers). The runner also maps `/kaggle/input` and `/kaggle/working` to the stage's own folders and prints a `[cell]` line before every cell; neither touches the code below.

## 1. make_probes.py replaced by its BEST_NOW path only

- type: `wiring`
- where: stage.ipy (whole)
- old: make_probes.py final: loads about 25 zips of POOL at import, then write_zip(BEST_NOW)
- new: the same load / vec / write_zip code on the two zips BEST_NOW uses (FACTOR_ENFUSE, FVBIND_ENFUSE)
- why: the other 23 zips are not needed and some are not part of the 13.54 chain (the 13.54 zip reads the two ENFUSE zips directly; BEST4 is only a reference)
- effect on numbers: none (members re-printed %.8f from %.8f sources: byte-identical, checked against the 15/11 member sha256)

## 2. saved zips also searched in <repo>/assembly/inputs; comment corrected

- type: `path`
- where: stage.ipy cell 1, function find(); header comment
- old: <FF_DATA root>/05_scores and the stage's input view; comment '16.72 = sum of four measured cells'
- new: <repo>/assembly/inputs added (from FF_STAGE_DIR, DESIGN 5.3); comment '16.72 = mean of the four measured cell EERs' ((13.78 + 11.97 + 22.12 + 18.99) / 4 = 16.715, make_probes.py KNOWN :36-38)
- why: same search order as a_fvbind; the old comment stated the arithmetic wrongly
- effect on numbers: none

## Not changed (deliberately)

- Pair order of FACTOR_ENFUSE and %.8f printing (make_probes.py write_zip).
