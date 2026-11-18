# CHANGES: `c_frames_index`

New code written for the reproduction repository (no original run code exists): every code cell starts with `print("[cell] ...", flush=True)`. Nothing to diff against. See `original/SOURCE.json` for the published dataset it reproduces.

## Deliberately unchanged

- No `!pip` line, no network.

## Runner metadata changed after review (18/11/2026; `stage.json` only, no code change)

| change | reason | effect on numbers |
|---|---|---|
| `skip_with` removed | the runner serves `skip_with` as a saved copy of the stage's KERNEL output (`runner/ffcore.py` `skip_with_view`, `locate`); `FF_DATA/01_collected/...` holds the exported dataset, not the kernel layout. The stage is skipped by default anyway (collect stages run only when forced) | none |
| `modes.train` rewritten | a re-collection must force the whole collect closure in one run (fresh `FF_WORK`, `FF_USE_SAVED=collected`); forcing one stage alone mixes new rows with the old frames / audio cache | none |
| `verify` marked informational | the stage compares the 7 sha256 itself (`FRAMES_INDEX_STRICT=1` makes a mismatch fatal) | none |
