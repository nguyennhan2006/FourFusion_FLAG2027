"""FUSEARCH-03 (README 15/11): export the J01 fuser = one CA-FUSE on the 7 F1s' streams + raw t3 / nf / nv of RFX c5rl (oof tables
rfdc0 of FLAG_195), C 0.1, fitted on every SAFE-04 row (E.fit_fuser: no purge, MAV-flagged rows out), for the reference Evaluation
zips. Kaggle CPU (FLAG_215):  python lab_entry.py FUSEARCH-03 report  -> fusearch3_summary.json (key j01) / fusearch3_report.txt.
Nothing reads dev / CB / Evaluation."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXPD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


FA = _load("factor01", EXPD / "FACTOR-01" / "run.py")
M = FA.M
E, S6, OUT, KEY, say = M.E, M.S6, M.OUT, M.KEY, M.say
TAG = "fusearch3"
RAW = ["c0_t3", "c0_nf", "c0_nv"]


def report():
    t00 = time.time()
    CT = _load("ctrun", EXPD / "CLEANTUNE-01" / "run.py")
    CT.merge_t3s()
    assert (OUT / "rfdc0lab_fold1.parquet").exists(), "attach the FLAG_195 part 1 run (rfdc0 tables)"
    parts = []
    for s in range(1, 6):
        T = pd.read_parquet(OUT / f"ct01r2_fold{s}.parquet"); P = pd.read_parquet(OUT / f"rfdc0lab_fold{s}.parquet")
        assert len(P) == len(T) and all((P[k].astype(str).values == T[k].astype(str).values).all() for k in KEY), f"rfdc0 fold {s}: rows"
        for c in ("t3", "nf", "nv"):
            T[f"c0_{c}"] = P[f"rfdc0_{c}"].values
        parts.append(T)
    allv = pd.concat(parts)
    fill = {c: float(np.nanmean(allv[c].values)) for c in RAW}
    for s, T in enumerate(parts, 1):
        miss = {c: int(T[c].isna().sum()) for c in RAW}
        for c in RAW:
            T[c] = T[c].fillna(fill[c])
        T.to_parquet(OUT / f"{TAG}_fold{s}.parquet")
        say(f"  fold {s}: {len(T)} rows, missing RF streams filled {miss}")
    F1S = S6 + E.V11_T3 + ["trCA"]
    j01 = E.fit_fuser(TAG, F1S + RAW, C=0.1, names=F1S + ["rfx_t3", "rfx_nf", "rfx_nv"])
    j01["fill"] = {f"rfx_{c[3:]}": v for c, v in fill.items()}
    j01["note"] = "FUSEARCH-03 J01: CA-FUSE C 0.1 on every SAFE-04 row (MAV-flagged out); RFX streams = oof rfdc0 (FLAG_195, c5rl seeds 1-3)"
    say(f"== {TAG}: J01 fitted on {j01['n_trials']} trials, {len(j01['coef'])} design columns; streams {j01['streams']}; fill {j01['fill']}")
    say(f"  |coef| of the stream main effects: {dict(zip(j01['streams'], [round(abs(c), 3) for c in j01['coef'][:len(j01['streams'])]]))}")
    (OUT / f"{TAG}_summary.json").write_text(json.dumps({"j01": j01}, indent=1), encoding="utf-8")
    (OUT / f"{TAG}_report.txt").write_text("\n".join(M.TXT) + "\n", encoding="utf-8")
    say(f"{TAG} report written [{time.time() - t00:.0f}s]")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
