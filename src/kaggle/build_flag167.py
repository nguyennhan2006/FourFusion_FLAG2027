r"""FLAG_167 (4/11, nhat ky 378): T3S-01 = Track-3 recipe branches (one projector per face x voice, 5 seeds) as a stream for F1s's
Bangla CA-FUSE. Experiment/T3S-01/README.md (registered before any result).
    python kaggle/build_flag167.py   -> FLAG_167_t3s01.ipynb (fold runs + CPU report) and FLAG_167f_t3s01_final.ipynb (only after the gate)
Fold runs: --env 'LAB_FOLDS=<s>;LID_STRICT=1;T3S=1;CT_R2_ONLY=CA' (GPU T4 x2): one fold, 4 face jobs (farl + the CA control, clip,
b3s, c5s), T3S_PER_GPU (2) jobs per GPU, started T3S_STAGGER (300) s apart. CPU report: LAB_FOLDS= with the 20 job tables (dataset).
Final: LAB_FOLDS=;CLEANCOMP=1 + tables + F1s's final npz / zip -> submission_t3s01_F1sT_dev.zip (English = F1s bytes).
Smoke (Kaggle CPU): bash kaggle/sat_push.sh 167 / 167f with SMOKE=1 (stand-in tables for the folds / faces not trained)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_e2e_lab as BL
import build_flag110 as B110

TAG = "ct01r2"
PARTS = ("farl", "clip", "b3s", "c5s")
ARMS_OF = {p: (("CA",) if p == "farl" else ()) + tuple(f"T3S_{p}_{v}" for v in ("sim", "r293", "ecapa", "cat4", "wsv", "rdn")) for p in PARTS}

ENV = B110.env(TAG, "r2") + r'''
os.environ["T3S"] = "1"                                            # T3S-01 (Experiment/T3S-01/README.md)
os.environ.setdefault("CT_R2_ONLY", "CA")                          # the in-run F1s control: CA only (no 'b' copy under T3S)
assert os.environ.get("LID_STRICT") == "1", "T3S-01 runs on the LID_STRICT rows (README §1)"
T3S_PARTS = [p for p in os.environ.get("T3S_PARTS", "farl,clip,b3s,c5s").split(",") if p]   # one job per face
T3S_PER_GPU = int(os.environ.get("T3S_PER_GPU", "2"))             # projector steps are launch-bound: two jobs share a GPU
T3S_STAGGER = int(os.environ.get("T3S_STAGGER", "300"))           # seconds between job starts (RAM peak of the data loading)'''


def jobs(final):
    """BL.JOBS with one job per (fold, face) or per face (final), T3S_PER_GPU slots per GPU, staggered starts."""
    j = BL.JOBS
    rep = [('JOBLIST = [f"fold{s}" for s in FOLDS[:1]] + ([] if FINAL_AFTER_REPORT else ["final"]) + [f"fold{s}" for s in FOLDS[1:]]',
            'JOBLIST = ' + ('[f"final{p}" for p in T3S_PARTS]' if final else '[f"fold{s}{p}" for s in FOLDS for p in T3S_PARTS]')
            + '   # T3S-01: one job per face'),
           ('slots = list(range(max(NGPU, 1)))', 'slots = list(range(max(NGPU, 1) * T3S_PER_GPU))   # T3S-01: slot g runs on GPU g % NGPU'),
           ('CUDA_VISIBLE_DEVICES=str(g) if NGPU else ""', 'CUDA_VISIBLE_DEVICES=str(g % NGPU) if NGPU else "", T3S_PART=name[5:]'),
           ('(["final"] if name == "final" else [])', '(["final"] if name.startswith("final") else [])'),
           ('env["FOLDS"] = name[4:]', 'env["FOLDS"] = name[4]'),
           ('SP.glob(f"out/safe04/{TAG}_{name}.parquet")', 'SP.glob(f"out/safe04/{TAG}{name[5:]}_fold{name[4]}.parquet")'),
           ("""log(f"job {name} started on {'GPU ' + str(g) if NGPU else 'CPU'}")""",
            """log(f"job {name} started on {'GPU ' + str(g % NGPU) if NGPU else 'CPU'}"); time.sleep(T3S_STAGGER)""")]
    for a, b in rep:
        assert j.count(a) == 1, a
        j = j.replace(a, b)
    return j


GATHER = r'''
# job tables ct01r2<face>_fold<s> of the OTHER runs (attached outputs / tables dataset) + FACELORA-03's B3 reference tables
import pandas as pd
for f in sorted(IN.rglob(f"{TAG}?*_fold[1-5].parquet")):
    part = f.stem[len(TAG):].split("_fold")[0]
    j = f"fold{f.stem[-1]}{part}"
    if part in ("farl", "clip", "b3s", "c5s") and j not in prog["done"] and not (SP / "out/safe04" / f.name).exists():
        shutil.copy(f, SP / "out/safe04" / f.name); shutil.copy(f, OUTD / f.name)
        prog["done"].append(j); prog["files"][f.name] = hashlib.md5(f.read_bytes()).hexdigest()
        log(f"gathered {f.name} from {f.parent}")
PROG.write_text(json.dumps(prog, indent=1))
B3T = sorted(IN.rglob("facelora03_B3_fold*.parquet"))
for f in B3T:
    shutil.copy(f, SP / "out/safe04" / f.name)
log(f"jobs done {sorted(j for j in prog['done'] if j.startswith('fold'))}; FACELORA-03 B3 reference tables {len(B3T)}")
'''

STANDIN = B110.STANDIN + r'''
if SMOKE:                                                          # every (face, fold) table not trained here
    for i, (p, arms) in enumerate(__ARMS_OF__.items()):
        standin(f"{TAG}{p}", arms, i)
'''

REPORT = r'''
# T3S-01 registered rule (README §3) on the 20 job tables -> t3s_decision.json
need = [SP / "out/safe04" / f"{TAG}{p}_fold{s}.parquet" for s in range(1, 6) for p in ("farl", "clip", "b3s", "c5s")]
if all(f.exists() for f in need):
    r = subprocess.run([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP, "report"], cwd=str(WORK), env=ENVV,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("\n".join(l for l in r.stdout.splitlines() if "Warning" not in l))
    assert r.returncode == 0, r.stderr[-3000:]
    for n in [f"{TAG}_report.txt", "t3s_decision.json"] + [f"{TAG}_fold{s}.parquet" for s in range(1, 6)]:
        shutil.copy(SP / "out/safe04" / n, OUTD / n)
    print("T3S-01 gate passed:", json.loads((OUTD / "t3s_decision.json").read_text())["passed"])
else:
    print("report not run: missing", [f.name for f in need if not f.exists()], "(attach the 20 job tables and run with LAB_FOLDS=)")
'''

ZIP = r'''
# A0 audit of the training rows, then the F1sT zip (README §3): Bangla CA-FUSE(S6 + v11 T3 + trCA + trT3S), English = F1s bytes
for cmd in ("audit", "zip"):
    r = subprocess.run([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP, cmd], cwd=str(WORK), env=ENVV,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("\n".join(l for l in r.stdout.splitlines() if "Warning" not in l))
    assert r.returncode == 0, r.stderr[-3000:]
print(sorted(p.name for p in OUTD.glob("submission_t3s01_*.zip")))
'''

INTRO = """
# FLAG 2027 — NB-167{sfx}: T3S-01, Track-3 recipe branches as a stream for F1s (LID_STRICT){what}

**What (Experiment/T3S-01/README.md, registered 4/11 before any result).** One linear projector with Track 3's trainer dict
(sub60 = SUB21A: dim 512, AAM + SupCon .5 / tau .2 + prototype .6 + same-gender .5, margin .1, 40 epochs; face dropout .70 frozen /
.55 adapter, voice .40) per face x voice pair, faces {{FaRL ep16, CLIP-L, B3s, C5s}} x voices {{sim = SimAM-R100 VoxBlink2, r293, ecapa,
cat4, wsv = WavLM-Base+-SV, rdn = ReDimNet b6}}, 5 seeds; wsv / rdn branches on v4 + MAV only (no auxiliary voices have them).
`trT3S` = logistic mix of the 24 branches nested in fold. Rule: CA-FUSE(F1s + trT3S) - F1s on g cells >= +0.3, >= 4/5 folds,
held-out Urdu / Hindi >= -0.2 (F1s = the in-run CA control). No AVSpeech / LRS3, no CSLS, no test statistic. Internet OFF.
"""

fold_cells = [("markdown", INTRO.format(sfx="", what=": fold runs (GPU, one fold each) and the CPU report")), ("code", ENV),
              ("markdown", "## 1. Code (embedded, with ct_round2.json) + data links + SAFE-04 reference tables"), ("code", B110.SETUP),
              ("markdown", "## 2. Resume (optional) + job tables of the other runs"), ("code", BL.RESUME), ("code", GATHER),
              ("markdown", "## 3. Jobs: the 4 face jobs of each fold of LAB_FOLDS"), ("code", jobs(False)),
              ("code", STANDIN.replace("__ARMS_OF__", repr(ARMS_OF))),
              ("markdown", "## 4. Registered T3S-01 rule (all 5 folds) -> t3s_decision.json"), ("code", REPORT),
              ("markdown", "## 5. Summary"), ("code", BL.SUMMARY)]
final_cells = [("markdown", INTRO.format(sfx="f", what=": final branches on every train person + the F1sT zip (only after the gate)")),
               ("code", ENV),
               ("markdown", "## 1. Code (embedded) + data links + SAFE-04 reference tables"), ("code", B110.SETUP),
               ("markdown", "## 2. Resume (optional) + the 20 job tables (tables dataset)"), ("code", BL.RESUME), ("code", GATHER),
               ("code", STANDIN.replace("__ARMS_OF__", repr(ARMS_OF))),
               ("markdown", "## 3. Jobs: the 4 face jobs (24 branches, 5 seeds, every train person; F1s's CA head is reused)"), ("code", jobs(True)),
               ("markdown", "## 4. A0 audit + F1sT zip"), ("code", ZIP),
               ("markdown", "## 5. Summary"), ("code", BL.SUMMARY)]

if __name__ == "__main__":
    BL.B12.nb(fold_cells, "FLAG_167_t3s01.ipynb")
    BL.B12.nb(final_cells, "FLAG_167f_t3s01_final.ipynb")
    print("FLAG_167_t3s01.ipynb + FLAG_167f_t3s01_final.ipynb built")
