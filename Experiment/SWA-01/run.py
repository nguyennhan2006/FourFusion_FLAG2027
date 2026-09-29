"""SWA-01 (deep-research-report 28/09, "SEED-SWA"): does averaging the weights of the last epochs reduce the seed lottery
of the s007 member, and does it cost accuracy? TRUE labels.

Per v4 split s (40 train persons, 30 held out; the same files as EXTSCALE-01 part 2):
  raw   EXP-007 recipe (40 epochs), SEEDS nets, seeds s + 100 i
  avg   the same runs with avg_from = 30: mean of the weights after epochs 31..40, BatchNorm recomputed on the train rows
        (flag_v2.train_one; the default path is bit-identical to before, checked with 10 threads)
Each net is scored alone (the seed lottery) and as ensembles of 5 and 10 (what production does), variant `prod`
(s007 + ImageBind + age, as SET-13) and `member`, levels sample / person / cluster, on the held-out v4 persons and
on the v1/v2 persons of fold s (never trained on: this member sees v4 only).
Keep averaging if the mean does not get worse by > 0.2 and the seed SD falls by >= 20 %, or the mean improves >= 0.5.
    python run.py   -> results.csv
"""
import importlib.util, os, sys, time
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("xs", HERE.parent / "EXTSCALE-01" / "run.py")
XS = importlib.util.module_from_spec(spec); spec.loader.exec_module(XS)       # feature tables, files, scoring
V, store, RidgeCCA = XS.V, XS.store, XS.RidgeCCA
SEEDS = int(os.environ.get("SWA_SEEDS", "10"))
SPLITS = [int(x) for x in os.environ.get("SWA_SPLITS", "1,2,3,4,5").split(",")]
OUT = HERE / "results.csv"

recs = []
for s in SPLITS:
    t0 = time.time()
    tr, va = V.speaker_split(store.spk, s, n_val=30)
    tri = np.where(tr)[0]
    held = [p for p in XS.PERSONS if XS.PFOLD[p] == s - 1]
    files = XS.eval_files(held, np.where(va)[0], np.random.RandomState(s))
    TF, TV, TS = store.Xf[tri], store.Xv[tri], XS.V4S[tri]
    cca = RidgeCCA(**V.CCA_CFG).fit(TF, TV)
    nets = {"raw": [], "avg": []}
    for i in range(SEEDS):
        for arm, extra in [("raw", {}), ("avg", {"avg_from": 30})]:
            nets[arm].append(V.train_one(store, dict(XS.REC, **extra), None, seed=s + 100 * i, rows=(TF, TV, TS)))
    for arm in nets:
        groups = [(f"single{i}", [n]) for i, n in enumerate(nets[arm])]
        groups += [("ens5a", nets[arm][:5]), ("ens5b", nets[arm][5:10]), ("ens10", nets[arm][:10])]
        for name, ns in groups:
            model = dict(nets=ns, cca=cca, mf=TF.mean(0), mv=TV.mean(0), TF=TF, TV=TV)
            for (data, prot), d in files.items():
                for (var, level), e in XS.score(model, d).items():
                    recs.append(dict(split=s, arm=arm, group=name, kind=name.rstrip("0123456789ab"), data=data,
                                     protocol=prot, variant=var, level=level, eer=e))
    pd.DataFrame(recs).to_csv(OUT, index=False)
    print(f"split {s} done in {time.time() - t0:.0f}s", flush=True)

T = pd.DataFrame(recs)
S1 = T[T.kind == "single"]
a = S1.groupby(["variant", "level", "data", "protocol", "split", "arm"]).eer.agg(["mean", "std"]).reset_index()
summ = a.groupby(["variant", "level", "data", "protocol", "arm"])[["mean", "std"]].mean().unstack("arm").round(2)
print("\nsingle nets: mean EER and seed SD (mean over splits)\n" + summ.to_string())
E = T[T.kind != "single"].pivot_table(index=["variant", "level", "data", "protocol"], columns=["group", "arm"], values="eer").round(2)
print("\nensembles (mean over splits)\n" + E.to_string())
print("ALL DONE", flush=True)
