"""MODEL-01 summary: paired difference to CTRL per split, and the pre-registered gate (docs/OPEN_DIRECTIONS.md §1).

Gate, sample level: v4 g better by >= 0.5 in >= 4/5 splits (mean), v4 ng not worse by > 0.5,
Urdu / Hindi sample not worse by > 0.5, v4 cluster / Urdu-Hindi person not worse by > 0.5.
Delta < 0 = better. [k/n] = splits in which the arm beats CTRL."""
import sys
import pandas as pd
from pathlib import Path

r = pd.read_csv(Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "results.csv"))
r["cell"] = r.target + "/" + r.protocol + "/" + r.level
wide = r.pivot_table(index=["arm", "split"], columns="cell", values="eer")
ctrl = wide.xs("CTRL", level="arm")
rows = []
for arm in wide.index.get_level_values("arm").unique():
    w = wide.xs(arm, level="arm")
    common = w.index.intersection(ctrl.index)
    d = (w.loc[common] - ctrl.loc[common]).dropna(axis=1, how="all")      # cells this arm has (v4-only streams)
    row = {"arm": arm, "splits": len(common)}
    for c in d.columns:
        row[c] = f"{d[c].mean():+.2f} [{int((d[c] < 0).sum())}/{len(common)}]" if arm != "CTRL" else f"{w[c].mean():.2f}"
    if arm != "CTRL":
        m = d.mean()
        wins_g = int((d["v4_en/g/sample"] < 0).sum())
        guards = [c for c in d.columns if c != "v4_en/g/sample"]
        worst = max(guards, key=lambda c: m[c])
        ok = m["v4_en/g/sample"] <= -0.5 and wins_g >= 4 and m[worst] <= 0.5
        row["gate"] = "PASS" if ok else f"fail ({'g' if not (m['v4_en/g/sample'] <= -0.5 and wins_g >= 4) else worst + f' {m[worst]:+.2f}'})"
    rows.append(row)
out = pd.DataFrame(rows).set_index("arm")
order = [c for c in ["splits", "v4_en/ng/sample", "v4_en/g/sample", "urdu/ng/sample", "urdu/g/sample", "hindi/ng/sample",
                     "v4_en/ng/cluster", "v4_en/g/cluster", "urdu/ng/person", "urdu/g/person", "hindi/ng/person", "gate"]
         if c in out.columns]
pd.set_option("display.width", 250)
print(out[order].to_string())
