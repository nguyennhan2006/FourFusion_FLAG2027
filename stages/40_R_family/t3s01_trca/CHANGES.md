# CHANGES: t3s01_trca (new stage, integration fixer r1, 18/11/2026)

- **Why the stage exists (INTEGRATION_r1 B3, user requirement 1):** `t219_dumpv4` (source of sigma 5.0983 / 3.8376) mounted the frozen dataset
  `ngnnguynchn/flag2027-t3s01-tables-p`, which no stage of the repository produced. Its producing runs (FLAG_167 T3S-01, five fold runs, then the
  gather step `kaggle/sat_push.sh 167g`) cannot be rebuilt: their staging folders are empty and only the builder `kaggle/FLAG_167_t3s01.ipynb` is left.
- **What it makes:** the four job tables per fold that `CLEANTUNE-01/run.py merge_t3s` expects (`ct01r2{b3s,c5s,clip,farl}_fold<s>.parquet`), with the
  row keys `fold, data, protocol, y, fi, vi` of `ct02s-r2ca-tables-p` and, in the farl tables only, its `trCA` column (as in the frozen dataset).
- **Difference from the frozen dataset (type `wiring`):** the 24 `trT3S_*` branch columns are absent. The dumpv4 path of `FUSENET-01/run.py report`
  fuses `F1S = S6 + V11_T3 + ["trCA"]` and RFX only; `trT3S_*` feed `reportt3s` / `final_t3s` / `_with_t3s`, which t219 does not call.
- **Effect on numbers:** none expected for sigma (trCA identical: WP-F light check 18/11, max |diff| 0.0 on all five folds; the stage re-checks it when the
  frozen copy is present).
- **Consumers:** `t219_dumpv4` (input `datasets/ff/flag2027-t3s01-trca-p`, mounted at `datasets/ngnnguynchn/flag2027-t3s01-tables-p`). `t184_report`
  (evidence) keeps the frozen dataset (`resolve: saved_only`).
