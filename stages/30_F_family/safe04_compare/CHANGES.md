# CHANGES: safe04_compare (new stage)

New code only (no original). Hard checks: 37 110 pairs, identical pair keys, identical stack02_rows (fold, data, y, fp, vp), identical
fold_persons.json; reported: max |diff|, Pearson r and NaN pattern of every numeric column. When the e2e-dev input is the frozen copy (safe04_lab
not run in the session; no SAFE04_REBUILT.json), the stage writes `not-a-rebuild` and passes. Every cell opens with `print("[cell] ...", flush=True)`.

2026-11-18 fix: a smoke rebuild of safe04_lab (fewer folds; `SAFE04_REBUILT.json` `complete: false`) is reported as `PARTIAL` (missing tables listed, hard checks reported, not enforced) instead of failing on the absent tables. The rebuilt bundle compared is the level-full output of safe04_lab; the level-core output (safe04_lab_core) is self-checked in its own stage.
