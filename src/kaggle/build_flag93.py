r"""FLAG_93 (16/10): SGW-01 (Experiment/SGW-01/README.md, registered before any result): same-gender weight grid and neighbour
("semi-hard") batches in the CC head, on the SAFE-04 folds of LAB_FOLDS; attached sgw01_fold<s>.parquet of the other runs are
gathered; with the 5 folds: report (SEL nested over W0 / W1 / W2 / H / H1 vs the in-run control B3c).
Saturday 17/10: GPU runs with --env 'LAB_FOLDS=1,2' / '3,4' / '5' (~3.1 h each), then a CPU run with --env 'LAB_FOLDS=' and the
three outputs attached (--kernels) for the report.
    python kaggle/build_flag93.py
Smoke (Kaggle CPU): --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1' -> fold 1 tiny; folds 2-5 = noisy copies of the B3 tables."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_e2e_lab as BL

EXP, TAG, NB = "SGW-01", "sgw01", "FLAG_93_sgw01.ipynb"
CODE = ["Experiment/SGW-01/run.py"] + BL.EXPS["FACELORA-03"]["code"]
OPT = dict(BL.OPTIONAL_DIRS)

ENV = BL.ENV.replace("__EXP__", EXP).replace("__TAG__", TAG).replace("__FAR__", "True").replace("__EXP_CODE__", json.dumps(CODE))
old = 'FOLDS = [int(x) for x in os.environ.get("LAB_FOLDS", "1,2,3,4,5").split(",") if x]'
assert old in ENV
ENV = ENV.replace(old, old + '\nSMOKE = os.environ.get("FL3Z_SMOKE") == "1"                       # wiring test: fold 1 only\n'
                  'FOLDS = FOLDS[:1] if SMOKE else FOLDS')

files = BL.CODE_COMMON + [f for f in CODE if f not in BL.CODE_COMMON]
blob, md5 = BL.code_blob(files)
SETUP = BL.SETUP.replace("__CODE_B64__", blob).replace("__CODE_MD5__", json.dumps(md5, indent=1)).replace(
    "__DATA_DIRS__", json.dumps(BL.L.DATA_DIRS)).replace("__OPTIONAL_DIRS__", json.dumps(OPT))

GATHER = r'''
# fold tables of the OTHER runs of this notebook (attached outputs) + FACELORA-03's B3 tables (reference; flag2027-facelora03-tables-p)
import pandas as pd
for f in sorted(IN.rglob(f"{TAG}_fold*.parquet")):
    s = f.stem.rsplit("fold", 1)[1]
    if f"fold{s}" not in prog["done"]:
        shutil.copy(f, SP / "out/safe04" / f.name); shutil.copy(f, OUTD / f.name)
        prog["done"].append(f"fold{s}"); prog["files"][f.name] = hashlib.md5(f.read_bytes()).hexdigest()
        log(f"gathered {f.name} from {f.parent}")
PROG.write_text(json.dumps(prog, indent=1))
B3T = sorted(IN.rglob("facelora03_B3_fold*.parquet"))
for f in B3T:
    shutil.copy(f, SP / "out/safe04" / f.name)
log(f"folds done {sorted(j for j in prog['done'] if j.startswith('fold'))}; FACELORA-03 B3 reference tables {len(B3T)}")
'''

STANDIN_FOLDS = r'''
# smoke only: folds not trained here = noisy copies of the B3 tables (wiring of merge / report)
if SMOKE:
    import numpy as np
    for s in range(1, 6):
        f = SP / "out/safe04" / f"{TAG}_fold{s}.parquet"
        if not f.exists():
            d = pd.read_parquet(SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet")
            rng = np.random.RandomState(s)
            for a in ("B3c", "W0", "W1", "W2", "H", "H1"):
                d[f"tr{a}"] = d["trB3"].values + 0.05 * rng.randn(len(d))
            d.drop(columns=[c for c in d.columns if c.startswith("trB3")]).to_parquet(f)
            prog["done"] = sorted(set(prog["done"]) | {f"fold{s}"})
            log(f"SMOKE: fold {s} = B3 stand-in")
'''

REPORT = r'''
# registered rule (SEL vs B3c); needs the 5 folds
GATE_TXT = OUTD / f"{TAG}_report.txt"
if all((SP / "out/safe04" / f"{TAG}_fold{s}.parquet").exists() for s in range(1, 6)):
    for cmd in ("report",):
        r = subprocess.run([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP, cmd], cwd=str(WORK), env=ENVV,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print("\n".join(l for l in r.stdout.splitlines() if "Warning" not in l))
        assert r.returncode == 0, r.stderr[-3000:]
    for s in range(1, 6):
        shutil.copy(SP / "out/safe04" / f"{TAG}_fold{s}.parquet", OUTD / f"{TAG}_fold{s}.parquet")
    shutil.copy(SP / "out/safe04" / f"{TAG}_report.txt", GATE_TXT)
else:
    print("merge / report not run: folds", [s for s in range(1, 6) if not (SP / "out/safe04" / f"{TAG}_fold{s}.parquet").exists()],
          "missing (attach the other runs' outputs and run with LAB_FOLDS=)")
'''

INTRO = """
# FLAG 2027 — NB-93: SGW-01, same-gender weight and neighbour batches in the CC head (B3 face)

**Why.** The CC head's same-gender SupCon weight (`w_sg` .5) comes unchanged from Track 3 and was never re-tuned on our head;
neighbour batches (a seed identity + its most similar same-gender identities) make the in-batch negatives the confusable people
of the g cells. Arms: **B3c** control, **W0 / W1 / W2** (w_sg 0 / 1 / 2), **H / H1** (hard_frac .5, w_sg .5 / 1); decision **SEL**
(nested). Rules in `Experiment/SGW-01/README.md` (registered before any result).
**Runs:** `--env 'LAB_FOLDS=1,2'`, `'3,4'`, `'5'` (GPU T4 x2, ~3.1 h each), then `--env 'LAB_FOLDS='` on CPU with the outputs attached.
**Inputs:** FLAG_73's 9 datasets + `nguynchnngn1/flag2027-feats-facelora3-b-p` + `ngnnguynchn/flag2027-facelora03-tables-p`. Internet OFF.
"""

cells = [("markdown", INTRO), ("code", ENV),
         ("markdown", "## 1. Code (embedded) + data links + SAFE-04 reference tables"), ("code", SETUP),
         ("markdown", "## 2. Resume (optional) + fold tables of the other runs"), ("code", BL.RESUME), ("code", GATHER),
         ("markdown", "## 3. Jobs: the folds of LAB_FOLDS, one per GPU at a time"), ("code", BL.JOBS), ("code", STANDIN_FOLDS),
         ("markdown", "## 4. Registered rule (all 5 folds)"), ("code", REPORT),
         ("markdown", "## 5. Summary"), ("code", BL.SUMMARY)]

if __name__ == "__main__":
    BL.B12.nb(cells, NB)
    print(f"{NB}: embedded {len(md5)} code files ({len(blob) / 1e3:.0f} kB base64)")
