"""Generate FLAG_48_lab.ipynb: run one SAFE-04 lab experiment (Experiment/<EXP>/run.py, default BIGB-01) fold by fold on a
Kaggle GPU with exactly the local code. The code is EMBEDDED here (base64 zip, md5 per file printed); the features come
from the dataset `flag2027-lab-data` (build_lab_bundle.py, uploaded once) + the existing `flag2027-eda-bundle`.
A new experiment = add its run.py to CODE below and rebuild this notebook; no re-upload of data.
Output: the experiment's per-fold parquet files (level-0 streams of every arm); the registered gate is scored locally.
    python build_lab_notebook.py
    FLAG_LOCAL=1 FLAG_IN=<dir with lab/ (data) and eda_bundle/> FLAG_OUT=<dir> LAB_FOLDS=1 AN_ARMS=B L0_STEPS=10 \
        python run_notebook_local.py FLAG_48_lab.ipynb
"""
import base64, hashlib, io, json, zipfile
from pathlib import Path
import build_eval_notebooks as B12

R = Path(__file__).resolve().parents[1]
CODE = ["kaggle/build_l0_notebooks.py", "Experiment/SAFE-04/lab_streams.py", "Experiment/SAFE-04/rel_sweep.py",
        "Experiment/SAFE-04/rel_sweep2.py", "Experiment/SAFE-04/rel_sweep5.py", "Experiment/SAFE-04/rel.py",
        "Experiment/VOX2-01/run_bridge.py", "Experiment/T3ENC-01/run.py", "Experiment/ALLNEG-01/run.py", "Experiment/BIGB-01/run.py",
        "Experiment/SCALE-VB/run.py", "Experiment/VBSEL-01/run.py", "Experiment/VBSEL-01/vox2_sa_speakers.json", "Experiment/ENBASE-01/run.py"]
DATA_DIRS = ["kaggle/output/feats_av", "kaggle/output/feats_eval", "kaggle/output/feats_models", "Experiment/T3ENC-01/feats_t3enc",
             "Experiment/T3ENC-01/feats_vox2t3", "Experiment/T3ENC-01/feats_vox2lid", "Experiment/VOX2-01/feats_vox2"]
OPTIONAL_DIRS = {"Experiment/SCALE-VB/feats_vb": ["SCALE-VB", "VBSEL-01"], "kaggle/output/feats_e6144": ["ENBASE-01"]}           # data folder -> experiments that need it
ENTRY = r'''"""Kaggle entry of the SAFE-04 lab: python lab_entry.py <experiment dir, e.g. BIGB-01> [args of its run.py]
Same code path as on the local machine, with one change: SAFE-04's ECAPA-6144 voice stream is a (rows, 0) placeholder in
the data bundle and is removed here, so an experiment that uses it fails loudly instead of running on empty features."""
import os, sys, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parent
run_py = ROOT / "Experiment" / sys.argv[1] / "run.py"
os.environ["SAFE04_NOMAIN"] = "1"
sys.path.insert(0, str(ROOT / "Experiment/SAFE-04"))
import rel_sweep5 as R
assert R.VOICE["e6144"][0].shape[1] == 0, "the data bundle should carry the ECAPA-6144 placeholder"
del R.VOICE["e6144"]
sys.argv = [str(run_py)] + sys.argv[2:]
runpy.run_path(str(run_py), run_name="__main__")
'''


def code_blob(extra=()):
    b = io.BytesIO(); md5 = {}
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in CODE + list(extra):
            data = (R / f).read_bytes(); z.writestr(f, data); md5[f] = hashlib.md5(data).hexdigest()
        z.writestr("lab_entry.py", ENTRY); md5["lab_entry.py"] = hashlib.md5(ENTRY.encode()).hexdigest()
    return base64.b64encode(b.getvalue()).decode(), md5


INTRO = """
# FLAG 2027 — NB-48: SAFE-04 lab experiments on a GPU (default: VBSEL-01)

**Why.** The T3-head experiments (ALLNEG-01, BIGB-01, later T3RECIPE-01 / SCALE-VB) train small heads on ~58k rows for
every fold and arm; on a busy local CPU one fold can take hours. This notebook runs **the same code** (the local
`Experiment/*/run.py` files, embedded below with their md5) on a Kaggle GPU. Nothing reads dev or test data; the registered
gate is computed locally from the output.

**Inputs (Add Input):**
- **`flag2027-lab-data`** (upload `kaggle/flag2027-lab-data.zip` once, private): T3 / VoxCeleb2 / v1-v2 features.
- **`flag2027-lab-vb`** (upload `kaggle/flag2027-lab-vb.zip` once, **private**: VoxBlink, CC BY-NC-SA, no redistribution): needed by SCALE-VB.
- **`flag2027-eda-bundle`** (already on Kaggle; must be the 2026-09-29 08:06 version, checked below).
- Optional: this notebook's previous output, to resume (finished folds are skipped).

**Settings:** GPU T4 (x1 is enough), Internet OFF is fine. `LAB_EXP` / `LAB_FOLDS` in the first cell select the run.

**Time:** VBSEL-01 (6 arms incl. the GPU control `B`; up to 124k training rows): about 15-20 min per fold on a T4, so about
1-1.5 h for 5 folds (SCALE-VB took 12 min per fold). Resumable per fold; to use two sessions at once, set `LAB_FOLDS` to `1,2,3` in one and `4,5` in the other.

**Output:** `/kaggle/working/lab_out/` -> download it (or make it a dataset) and hand it back; the gate is scored locally:
put the parquet files into the local SAFE-04 scratch `out/safe04/`, then `VOX_LID=... python Experiment/<EXP>/run.py report`.
"""

ENV = r'''
import os, sys, json, time, shutil, subprocess, hashlib, base64, io, zipfile
from pathlib import Path
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORK = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
EXP = os.environ.get("LAB_EXP", "VBSEL-01")                         # Experiment/<EXP>/run.py (registered in its README)
FOLDS = [int(x) for x in os.environ.get("LAB_FOLDS", "1,2,3,4,5").split(",")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


mans = [(m.parent, json.loads(m.read_text())) for m in sorted(IN.rglob("lab_manifest.json"))]
main = [(d, m) for d, m in mans if m.get("kind", "data") == "data"]
assert len(main) == 1, f"attach exactly one flag2027-lab-data (found {len(main)})"
DATA, MAN = main[0]
ROOTS = [d for d, _ in mans]                                           # every attached lab bundle (data, data-vb, ...)
eda = sorted(IN.rglob("eda_bundle_manifest.json"))
assert len(eda) == 1, f"attach exactly one flag2027-eda-bundle (found {len(eda)})"
created = json.loads(eda[0].read_text())["created"]
assert created == MAN["eda_bundle_created"], f"eda bundle {created} != the lab's {MAN['eda_bundle_created']}"
SP = WORK / "sp"; (SP / "out/safe04").mkdir(parents=True, exist_ok=True)
OUTD = WORK / "lab_out"; OUTD.mkdir(parents=True, exist_ok=True)
LAB = WORK / "lab"
import torch
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU: slow)"
log(f"data {[str(r) for r in ROOTS]} (main built {MAN['created']}) | eda bundle {created} | experiment {EXP} | folds {FOLDS} | GPU {GPU} | "
    f"torch {torch.__version__}")
'''

SETUP = r'''
# the embedded code (= the local files, md5 below) is written to /kaggle/working/lab; the data folders are linked next to it
CODE_B64 = "__CODE_B64__"
CODE_MD5 = __CODE_MD5__
DATA_DIRS = __DATA_DIRS__
OPTIONAL_DIRS = __OPTIONAL_DIRS__
with zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))) as z:
    for f in z.namelist():
        b = z.read(f); assert hashlib.md5(b).hexdigest() == CODE_MD5[f], f
        (LAB / f).parent.mkdir(parents=True, exist_ok=True); (LAB / f).write_bytes(b)


def link(target, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, path, target_is_directory=True)
    except OSError:                                                   # Windows without symlink rights (local smoke): junction
        subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(target)], check=True, capture_output=True)


for d in DATA_DIRS + list(OPTIONAL_DIRS):
    src = [r / d for r in ROOTS if (r / d).is_dir()]
    if not src:
        assert d not in DATA_DIRS and EXP not in OPTIONAL_DIRS.get(d, []), f"no attached lab bundle has {d} (needed by {EXP})"
        continue
    link(src[0], LAB / d)
assert (LAB / "Experiment" / EXP / "run.py").exists(), f"{EXP} is not embedded in this notebook (rebuild with it in CODE)"
for f, h in CODE_MD5.items():
    print(f"  {h}  {f}")
ENVV = dict(os.environ, SAFE04_SCRATCH=str(SP), FLAG_LOCAL="1", FLAG_IN=str(eda[0].parent.parent), FLAG_OUT=str(SP / "out"),
            VOX_LID=str(LAB / "Experiment/T3ENC-01/feats_vox2lid/lid_summary.json"), SAFE04_THREADS=str(os.cpu_count() or 4),
            PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
log(f"lab ready at {LAB}: {len(CODE_MD5)} code files, data folders linked: {[d for d in DATA_DIRS + list(OPTIONAL_DIRS) if (LAB / d).exists()]}")
'''

RESUME = r'''
# resume: a previous version's lab_out/ (same experiment) -> its finished folds are copied and skipped
PROG = OUTD / "progress.json"
prog = json.loads(PROG.read_text()) if PROG.exists() else {}
for prev in ([] if LOCAL else IN.rglob("lab_out/progress.json")):
    p = json.loads(prev.read_text())
    if p.get("exp") == EXP and not prog.get("done"):
        for f in prev.parent.glob("*_fold*.parquet"):
            shutil.copy(f, OUTD / f.name); shutil.copy(f, SP / "out/safe04" / f.name)
        prog = p; log("resume from", prev.parent, "folds done", p["done"])
prog = prog if prog.get("exp") == EXP else {"exp": EXP, "done": [], "files": {}, "seconds": {}}
prog["code_md5"] = CODE_MD5
'''

RUN = r'''
# one subprocess per fold: the local entry point, unchanged code; output streamed (warnings dropped)
for s in FOLDS:
    if s in prog["done"]:
        log(f"fold {s}: done before, skipped"); continue
    t0 = time.time(); before = set((SP / "out/safe04").glob(f"*_fold{s}.parquet"))
    p = subprocess.Popen([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP], cwd=str(WORK), env=dict(ENVV, FOLDS=str(s)),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if "Warning" not in line and "warnings.warn" not in line:
            print(line, end="", flush=True)
    if p.wait() != 0:
        print("".join(tail)); raise RuntimeError(f"{EXP} fold {s} failed")
    new = sorted(set((SP / "out/safe04").glob(f"*_fold{s}.parquet")) - before) or sorted((SP / "out/safe04").glob(f"*_fold{s}.parquet"))
    assert new, f"fold {s}: no parquet written"
    for f in new:
        shutil.copy(f, OUTD / f.name); prog["files"][f.name] = hashlib.md5(f.read_bytes()).hexdigest()
    prog["done"].append(s); prog["seconds"][str(s)] = round(time.time() - t0)
    PROG.write_text(json.dumps(prog, indent=1))
    log(f"fold {s} saved: {[f.name for f in new]} [{time.time() - t0:.0f}s]")
'''

FINAL = r'''
import pandas as pd
prog.update(gpu=GPU, torch=torch.__version__, data_built=MAN["created"], eda_bundle=created, finished=time.strftime("%Y-%m-%d %H:%M"))
PROG.write_text(json.dumps(prog, indent=1))
for f in sorted(OUTD.glob("*.parquet")):
    D = pd.read_parquet(f)
    cols = [c for c in D.columns if c.endswith(("_deep", "_cca"))]
    print(f"{f.name}: {len(D)} trials, streams {cols}, NaN {int(D[cols].isna().sum().sum())}")
print(json.dumps({k: v for k, v in prog.items() if k != "code_md5"}, indent=1))
print("NEXT: download /kaggle/working/lab_out (or Save Version -> Output -> New Dataset `flag2027-lab-out`) and hand it back.")
'''

if __name__ == "__main__":
    blob, md5 = code_blob()
    setup = SETUP.replace("__CODE_B64__", blob).replace("__CODE_MD5__", json.dumps(md5, indent=1)).replace(
        "__DATA_DIRS__", json.dumps(DATA_DIRS)).replace("__OPTIONAL_DIRS__", json.dumps(OPTIONAL_DIRS))
    cells = lambda intro, env: [("markdown", intro), ("code", env),
            ("markdown", "## 1. Code (embedded) + data links"), ("code", setup),
            ("markdown", "## 2. Resume (optional)"), ("code", RESUME),
            ("markdown", "## 3. Run the experiment fold by fold"), ("code", RUN),
            ("markdown", "## 4. Check and manifest"), ("code", FINAL)]
    B12.nb(cells(INTRO, ENV), "FLAG_48_lab.ipynb")
    intro_b = INTRO.replace("(default: VBSEL-01)", "(default: ENBASE-01)").replace(
        "- **`flag2027-lab-vb`**", "- **`flag2027-lab-e6144`** (upload `kaggle/flag2027-lab-e6144.zip` once, private): the real ECAPA-6144 "
        "arrays for the s010B analog. VoxBlink (`flag2027-lab-vb`) is NOT needed for ENBASE-01.\n- (`flag2027-lab-vb`)").replace(
        "**Time:** VBSEL-01", "**Time:** ENBASE-01: 5 splits x (4 sizes x [s007 + s010B + T3 head] + a 5-net base at 50 persons), about "
        "35-50 min per split on a T4 (the 6144-dim s010B input is the slow part), so about 3-4 h. One split per process; `LAB_FOLDS` splits it over sessions.\n**(VBSEL-01")
    env_b = ENV.replace('EXP = os.environ.get("LAB_EXP", "VBSEL-01")', 'EXP = os.environ.get("LAB_EXP", "ENBASE-01")')
    assert env_b != ENV
    B12.nb(cells(intro_b, env_b), "FLAG_48b_enbase.ipynb")
    print(f"embedded {len(md5)} code files ({len(blob) / 1e3:.0f} kB base64)")
    # FLAG_48c (12/10): ENC-SCREEN-02, the A-arm screen of FLAG_47b's voice encoders on a GPU with a same-device T3 control
    blob_c, md5_c = code_blob(extra=["Experiment/ENC-SCREEN-01/run.py", "Experiment/ENC-SCREEN-02/run.py"])
    setup_c = SETUP.replace("__CODE_B64__", blob_c).replace("__CODE_MD5__", json.dumps(md5_c, indent=1)).replace(
        "__DATA_DIRS__", json.dumps(DATA_DIRS)).replace("__OPTIONAL_DIRS__", json.dumps(OPTIONAL_DIRS))
    intro_c = """
# FLAG 2027 — NB-48c: ENC-SCREEN-02, A-arm screen of ResNet293-LM / ECAPA1024-LM (+ DINOv2 + ECAPA2) on a GPU

**Why.** Track 3's SUB21a voice branch has two encoders we never had (ResNet293-LM, ECAPA1024-LM; FLAG_47b embedded them on
our MAV-Celeb rows). ENC-SCREEN-01's cheap A-arm screen (MAV-Celeb only, InfoNCE MLP, 3 nets) decides whether one is worth a
VoxCeleb2 / VoxBlink extraction; the control `T3` runs in the same process (GPU vs GPU). Registered in
`Experiment/ENC-SCREEN-02/README.md` (pass: >= +1.0 on the ext g cells vs T3, no ext cell <= -1.0).

**Inputs:** `flag2027-lab-data` (`-p`), `flag2027-eda-bundle` (`-p`), `flag2027-feats-screen-p` (FLAG_47: dino2, ecapa2),
`flag2027-feats-screen2-p` (FLAG_47b: rn293, ecapa1024). **GPU T4**, Internet OFF. **Time:** 5 arms x 5 folds, about 1-1.5 h.

**Output:** `/kaggle/working/lab_out/encscreen02_fold*.parquet`; scored locally (`python Experiment/ENC-SCREEN-02/run.py report`).
"""
    env_c = ENV.replace('EXP = os.environ.get("LAB_EXP", "VBSEL-01")', 'EXP = os.environ.get("LAB_EXP", "ENC-SCREEN-02")')
    assert env_c != ENV
    c = cells(intro_c, env_c); c[3] = ("code", setup_c)
    B12.nb(c, "FLAG_48c_encscreen02.ipynb")
    print(f"FLAG_48c: embedded {len(md5_c)} code files ({len(blob_c) / 1e3:.0f} kB base64)")
