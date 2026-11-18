"""Generate the END-TO-END experiment notebooks (docs/PLAN_E2E_2026-10-10.md): one Kaggle run (GPU T4 x2) goes from the lab
features to the registered gate verdict AND a dev submission.zip, with nothing left to do on the local machine.

    FLAG_51_t3recipe.ipynb   T3RECIPE-01 (Track 3's projector recipe as a new Bangla stream)
    (FLAG_52 / FLAG_53 add their extraction cells in front of the same machinery: build_e2e_short.py, build_e2e_scaleid.py)

Machinery (same code as FLAG_48, plus):
  * the code is EMBEDDED (base64 zip + md5): the experiment's run.py, Experiment/E2E/e2e_lib.py, the SAFE-04 lab, STACK-02's
    purge / bootstrap; the data come from `flag2027-lab-data` (+ `flag2027-lab-vb` when the experiment uses VoxBlink),
    `flag2027-eda-bundle` and `flag2027-e2e-dev` (components of the dev files + SAFE-04 reference tables + fusers + ref zips)
  * JOBS = the SAFE-04 folds + the final head, one subprocess each, up to one per GPU at a time (CUDA_VISIBLE_DEVICES)
  * time guard: no job starts after E2E_BUDGET_H hours (default 10.5); finished jobs are recorded in e2e_out/progress.json
  * resume: attach this notebook's previous output; finished jobs are copied back and skipped
  * then: gate report (in the notebook), CA-FUSE refit, submission.zip (+ a copy at /kaggle/working/submission.zip)
Local smoke test (CPU, tiny):
    FLAG_LOCAL=1 FLAG_IN=<dir with lab/, eda_bundle/, e2edev/ (+ labvb/)> FLAG_OUT=<dir> LAB_FOLDS=1,2 TR_EPOCHS=1 TR_SEEDS=1 \
        L0_STEPS=10 python run_notebook_local.py FLAG_51_t3recipe.ipynb
"""
import base64, hashlib, io, json, zipfile
from pathlib import Path
import build_eval_notebooks as B12
import build_lab_notebook as L

R = Path(__file__).resolve().parents[1]
CODE_COMMON = L.CODE + ["Experiment/SAFE-04/stream_gate.py", "Experiment/SAFE-04/lab_fuse.py", "Experiment/STACK-02/run.py", "Experiment/STACK-02/pools.py", "Experiment/STACK-02/eligible.json",
                        "Experiment/STACK-02/fold_persons.json", "Experiment/E2E/e2e_lib.py"]
EXPS = {
    "T3RECIPE-01": dict(nb="FLAG_51_t3recipe.ipynb", code=["Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py"],
                        tag="t3recipe01", vb=True),
}
OPTIONAL_DIRS = {"Experiment/SCALE-VB/feats_vb": ["SCALE-VB", "VBSEL-01", "T3RECIPE-01", "T3REPRO-01", "T3RECIPE-02", "DSEL-02"], "kaggle/output/feats_e6144": ["ENBASE-01"]}
ENTRY = L.ENTRY.replace('"""Kaggle entry of the SAFE-04 lab: python lab_entry.py <experiment dir, e.g. BIGB-01> [args of its run.py]',
                        '"""Kaggle entry of the E2E lab: python lab_entry.py <experiment dir> [folds | report | final | zip]')
assert ENTRY != L.ENTRY


def code_blob(files):
    b = io.BytesIO(); md5 = {}
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            data = (R / f).read_bytes(); z.writestr(f, data); md5[f] = hashlib.md5(data).hexdigest()
        z.writestr("lab_entry.py", ENTRY); md5["lab_entry.py"] = hashlib.md5(ENTRY.encode()).hexdigest()
    return base64.b64encode(b.getvalue()).decode(), md5


ENV = r'''
import os, sys, json, time, shutil, subprocess, hashlib, base64, io, zipfile, threading
from pathlib import Path
T_START = time.time()
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORK = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
EXP = os.environ.get("LAB_EXP", "__EXP__")
TAG = "__TAG__"
EXP_CODE = __EXP_CODE__                                            # the experiment's own code: a resume needs the same files
FINAL_AFTER_REPORT = __FAR__                                      # True: the final head depends on the report (a chosen variant)
FOLDS = [int(x) for x in os.environ.get("LAB_FOLDS", "1,2,3,4,5").split(",") if x]
SPLIT = os.environ.get("E2E_SPLIT", "dev")                         # dev: flag2027-e2e-dev; eval: the components of a FLAG_50 run
BUDGET_H = float(os.environ.get("E2E_BUDGET_H", "10.5"))            # no new job starts after this many hours (Kaggle stops at 12)


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T_START) / 3600:4.1f}h]", *a, flush=True)


mans = [(m.parent, json.loads(m.read_text())) for m in sorted(IN.rglob("lab_manifest.json"))]
main = [(d, m) for d, m in mans if m.get("kind", "data") == "data"]
assert len(main) == 1, f"attach exactly one flag2027-lab-data (found {len(main)})"
DATA, MAN = main[0]
ROOTS = [d for d, _ in mans]
eda = sorted(IN.rglob("eda_bundle_manifest.json"))
assert len(eda) == 1, f"attach exactly one flag2027-eda-bundle (found {len(eda)})"
created = json.loads(eda[0].read_text())["created"]
assert created == MAN["eda_bundle_created"], f"eda bundle {created} != the lab's {MAN['eda_bundle_created']}"
e2e = [(m.parent, json.loads(m.read_text())) for m in sorted(IN.rglob("e2e_manifest.json"))]
e2e = [(d, m) for d, m in e2e if m.get("split") == SPLIT and (d / f"components_{SPLIT}.npz").exists()]
assert len(e2e) >= 1, f"attach flag2027-e2e-dev (or, for E2E_SPLIT=eval, the output of FLAG_50 on the Evaluation data)"
E2E_DIR, E2E_MAN = e2e[0]
SP = WORK / "sp"; (SP / "out/safe04").mkdir(parents=True, exist_ok=True)
EXTRA_LINKS = {}                                                    # lab path -> folder made by this notebook (extraction cells)
OUTD = WORK / "e2e_out"; OUTD.mkdir(parents=True, exist_ok=True)
LAB = WORK / "lab"
import torch
NGPU = torch.cuda.device_count()
GPU = [torch.cuda.get_device_name(i) for i in range(NGPU)] or ["none (CPU: slow)"]
log(f"experiment {EXP} ({TAG}) | folds {FOLDS} | split {SPLIT} ({E2E_DIR}, built {E2E_MAN['created']}) | lab data built {MAN['created']} | "
    f"eda {created} | GPUs {GPU} | CPUs {os.cpu_count()} | torch {torch.__version__} | budget {BUDGET_H} h")
'''

SETUP = r'''
# embedded code (= the local files, md5 below) -> /kaggle/working/lab; data folders linked next to it; SAFE-04 reference tables
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
for d, src in EXTRA_LINKS.items():                                   # features extracted above (FLAG_52 / 53)
    link(Path(src), LAB / d)
SAFE_SRC = E2E_DIR if (E2E_DIR / "safe04").is_dir() else next(d for d, m in [(m.parent, None) for m in IN.rglob("e2e_manifest.json")]
                                                              if (d / "safe04").is_dir())
for f in sorted((SAFE_SRC / "safe04").glob("*")):                     # reference tables of the gate (streams, rel, v11's T3, ...)
    shutil.copy(f, SP / "out/safe04" / f.name)
assert (LAB / "Experiment" / EXP / "run.py").exists(), f"{EXP} is not embedded in this notebook"
for f, h in CODE_MD5.items():
    print(f"  {h}  {f}")
ENVV = dict(os.environ, SAFE04_SCRATCH=str(SP), FLAG_LOCAL="1", FLAG_IN=str(eda[0].parent.parent), FLAG_OUT=str(SP / "out"),
            VOX_LID=str(LAB / "Experiment/T3ENC-01/feats_vox2lid/lid_summary.json"), E2E_DIR=str(E2E_DIR), E2E_SPLIT=SPLIT,
            E2E_OUT=str(OUTD), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
log(f"lab ready at {LAB}: {len(CODE_MD5)} code files; data: {[d for d in DATA_DIRS + list(OPTIONAL_DIRS) if (LAB / d).exists()]}; "
    f"{len(list((SP / 'out/safe04').glob('*')))} SAFE-04 reference tables")
'''

# Smoke only (16/10): FL3Z_STANDIN="<src>:<dst>" links an attached adapter's arrays (face_<src>_*.npy, ...) under the name of an
# adapter whose real run has not finished (e.g. clipB3:clipB3t), first in FL3_DIRS, so the whole notebook is wired before the real
# arrays exist. Must sit before SETUP (ENVV copies os.environ). Never set in a real run.
STANDIN = r'''
ST = os.environ.get("FL3Z_STANDIN", "")
if ST:
    src, dst = ST.split(":")
    hit = sorted({f.parent for f in IN.rglob(f"face_{src}_train.npy")})
    assert hit, f"stand-in source {src} not attached"
    sd = WORK / "standin" / dst
    sd.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in hit[0].glob(f"*_{src}*.npy"):
        if not f.name.startswith(f"face_{src}") and not f.name.endswith(f"_{src}.npy"):
            continue
        t = sd / f.name.replace(f"_{src}", f"_{dst}")
        if not t.exists():
            t.symlink_to(f)
        n += 1
    os.environ["FL3_DIRS"] = os.pathsep.join([str(sd)] + [p for p in os.environ.get("FL3_DIRS", "").split(os.pathsep) if p])
    log(f"SMOKE stand-in: {n} {src} arrays of {hit[0]} linked as {dst} in {sd}")
'''

RESUME = r'''
# resume: a previous version's e2e_out/ (same experiment) -> finished jobs and their files are copied back and skipped
PROG = OUTD / "progress.json"
prog = json.loads(PROG.read_text()) if PROG.exists() else {}
if not prog.get("done"):
    prevs = [] if LOCAL else [(q, json.loads(q.read_text())) for q in IN.rglob("e2e_out/progress.json")]
    prevs = [(q, p) for q, p in prevs if p.get("exp") == EXP and p.get("done")]
    # an output made by DIFFERENT code of this experiment (e.g. FLAG_51 before / after the SUB21a amendment of 11/10) is not
    # resumed: its finished folds would be mixed with folds of the new recipe
    changed = lambda p: [f for f in EXP_CODE if p.get("code_md5", {}).get(f) != CODE_MD5.get(f)]
    if os.environ.get("E2E_ALLOW_CODE_CHANGE") != "1":
        for q, p in prevs:
            if changed(p):
                log(f"!! NOT resuming from {q.parent}: made by different {EXP} code {changed(p)}; its folds would mix two "
                    "versions. This run starts fresh (E2E_ALLOW_CODE_CHANGE=1 overrides).")
        prevs = [(q, p) for q, p in prevs if not changed(p)]
    prevs.sort(key=lambda t: t[1].get("split") != SPLIT)           # the same split first
    for prev, p in prevs[:1]:
        same = p.get("split") == SPLIT
        for f in prev.parent.glob("*"):
            # another split (dev run -> Evaluation run): only the fold tables and the trained final models / heads move over;
            # the final job then reloads them and scores the new test files
            keep = same or f.name.endswith((".parquet", ".pt", "_selection.json"))
            if f.is_file() and f.name != "progress.json" and keep:
                shutil.copy(f, OUTD / f.name)
        for f in prev.parent.glob(f"{TAG}_fold*.parquet"):
            shutil.copy(f, SP / "out/safe04" / f.name)
        p = dict(p, split=SPLIT, done=[j for j in p["done"] if same or j.startswith("fold")])
        prog = p; log("resume from", prev.parent, f"(split {'same' if same else 'other'})", "jobs done", p["done"])
prog = prog if prog.get("exp") == EXP else {"exp": EXP, "split": SPLIT, "done": [], "seconds": {}, "files": {}}
prog["code_md5"] = CODE_MD5
PROG.write_text(json.dumps(prog, indent=1))
'''

JOBS = r'''
# JOBS: every SAFE-04 fold + the final head (started second, so it overlaps the folds); at most one subprocess per GPU;
# a job starts only if the budget allows it (elapsed < E2E_BUDGET_H and elapsed + 1.2 x the longest finished job < 11.5 h)
JOBLIST = [f"fold{s}" for s in FOLDS[:1]] + ([] if FINAL_AFTER_REPORT else ["final"]) + [f"fold{s}" for s in FOLDS[1:]]
todo = [j for j in JOBLIST if j not in prog["done"]]
slots = list(range(max(NGPU, 1)))
threads_per = str(max(1, (os.cpu_count() or 4) // len(slots)))
lock = threading.Lock()


def pump(job):
    for line in job["p"].stdout:
        job["tail"].append(line); del job["tail"][:-60]
        if "Warning" not in line and "warnings.warn" not in line:
            with lock:
                print(f"[{job['name']}] {line}", end="", flush=True)


def finish(job):
    name, t0 = job["name"], job["t0"]
    if job["p"].returncode != 0:
        print("".join(job["tail"])); raise RuntimeError(f"{EXP} job {name} failed (exit {job['p'].returncode})")
    files = sorted(SP.glob(f"out/safe04/{TAG}_{name}.parquet")) if name.startswith("fold") else []
    for f in files:
        shutil.copy(f, OUTD / f.name); prog["files"][f.name] = hashlib.md5(f.read_bytes()).hexdigest()
    prog["done"].append(name); prog["seconds"][name] = round(time.time() - t0)
    PROG.write_text(json.dumps(prog, indent=1))
    log(f"job {name} done [{time.time() - t0:.0f}s]")


def budget_ok():
    h = (time.time() - T_START) / 3600
    longest = max([v for v in prog["seconds"].values()] or [0]) / 3600
    return h < BUDGET_H and h + 1.2 * longest < 11.5


running = []
while todo or running:
    busy = {j["gpu"] for j in running}
    for g in [g for g in slots if g not in busy]:
        if not todo:
            break
        if not budget_ok():
            log(f"time budget: not starting {todo}"); todo = []; break
        name = todo.pop(0)
        env = dict(ENVV, SAFE04_THREADS=threads_per, CUDA_VISIBLE_DEVICES=str(g) if NGPU else "")
        args = [sys.executable, "-u", str(LAB / "lab_entry.py"), EXP] + (["final"] if name == "final" else [])
        if name.startswith("fold"):
            env["FOLDS"] = name[4:]
        p = subprocess.Popen(args, cwd=str(WORK), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        job = dict(name=name, p=p, t0=time.time(), gpu=g, tail=[])
        job["th"] = threading.Thread(target=pump, args=(job,), daemon=True); job["th"].start()
        running.append(job)
        log(f"job {name} started on {'GPU ' + str(g) if NGPU else 'CPU'}")
    for job in list(running):
        if job["p"].poll() is not None:
            job["th"].join(); running.remove(job); finish(job)
    time.sleep(2)
missing = [j for j in JOBLIST if j not in prog["done"]]
if missing:
    log(f"!! time budget reached: jobs left {missing}. Save Version, attach this output as input, run again (finished jobs are skipped).")
else:
    log("all jobs done:", prog["seconds"])
'''

REPORT = r'''
# registered gate, computed here (Experiment/E2E/e2e_lib.gate through the experiment's report()); needs every fold
GATE_TXT = OUTD / f"{TAG}_report.txt"
if all(f"fold{s}" in prog["done"] for s in (1, 2, 3, 4, 5)):
    r = subprocess.run([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP, "report"], cwd=str(WORK), env=ENVV,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("\n".join(l for l in r.stdout.splitlines() if "Warning" not in l)); assert r.returncode == 0, r.stderr[-3000:]
    shutil.copy(SP / "out/safe04" / f"{TAG}_report.txt", GATE_TXT)
else:
    print(f"gate not computed: folds done {[j for j in prog['done'] if j.startswith('fold')]} (all 5 are needed)")
'''

FINAL_LATE = r'''
# the final head of an experiment whose final variant is chosen by the report (FINAL_AFTER_REPORT); GPU 0
if FINAL_AFTER_REPORT and "final" not in prog["done"] and GATE_TXT.exists():
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP, "final"], cwd=str(WORK),
                         env=dict(ENVV, SAFE04_THREADS=str(os.cpu_count() or 4), CUDA_VISIBLE_DEVICES="0" if NGPU else ""),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-60:]
        if "Warning" not in line and "warnings.warn" not in line:
            print("[final]", line, end="", flush=True)
    if p.wait() != 0:
        print("".join(tail)); raise RuntimeError(f"{EXP} final failed")
    prog["done"].append("final"); prog["seconds"]["final"] = round(time.time() - t0); PROG.write_text(json.dumps(prog, indent=1))
    log(f"final done [{time.time() - t0:.0f}s]")
elif FINAL_AFTER_REPORT:
    print("final:", "done before" if "final" in prog["done"] else "waits for the report (all 5 folds)")
'''

ZIP = r'''
# final fuser (CA-FUSE refit on every SAFE-04 row with the new stream) -> submission.zip; English files = v11
if "final" in prog["done"] and all(f"fold{s}" in prog["done"] for s in (1, 2, 3, 4, 5)):
    r = subprocess.run([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP, "zip"], cwd=str(WORK), env=ENVV,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("\n".join(l for l in r.stdout.splitlines() if "Warning" not in l)); assert r.returncode == 0, r.stderr[-3000:]
    ZP = OUTD / f"submission_{TAG}_{SPLIT}.zip"
    shutil.copy(ZP, WORK / "submission.zip")
    log("submission:", ZP, "sha256", hashlib.sha256(ZP.read_bytes()).hexdigest()[:16])
else:
    print("zip not built: it needs the final head and the 5 folds (the fuser is refitted on the SAFE-04 rows with the new stream)")
'''

SUMMARY = r'''
prog.update(gpu=GPU, torch=torch.__version__, lab_data=MAN["created"], e2e=E2E_MAN["created"], finished=time.strftime("%Y-%m-%d %H:%M"))
PROG.write_text(json.dumps(prog, indent=1))
print(json.dumps({k: v for k, v in prog.items() if k != "code_md5"}, indent=1))
if (OUTD / f"{TAG}_report.txt").exists():
    print("\n".join(l for l in (OUTD / f"{TAG}_report.txt").read_text(encoding="utf-8").splitlines() if "DECISION" in l))
print("NEXT: download /kaggle/working/e2e_out (report, fold parquets, final models, fuser, zip). Submit submission.zip to CodaBench "
      "ONLY if the DECISION line says PASSES, and read it with the rule registered in the experiment README.")
'''


def build(exp):
    cfg = EXPS[exp]
    files = CODE_COMMON + [f for f in cfg["code"] if f not in CODE_COMMON]
    blob, md5 = code_blob(files)
    setup = SETUP.replace("__CODE_B64__", blob).replace("__CODE_MD5__", json.dumps(md5, indent=1)).replace(
        "__DATA_DIRS__", json.dumps(L.DATA_DIRS)).replace("__OPTIONAL_DIRS__", json.dumps(OPTIONAL_DIRS))
    no_final = cfg.get("final") is False                            # gate only: the final head is a later, separate run
    env = ENV.replace("__EXP__", exp).replace("__TAG__", cfg["tag"]).replace("__FAR__", str(bool(cfg.get("final_after_report") or no_final))).replace(
        "__EXP_CODE__", json.dumps(cfg["code"])).replace(
        'os.environ.get("LAB_FOLDS", "1,2,3,4,5")', f'os.environ.get("LAB_FOLDS", "{cfg.get("folds", "1,2,3,4,5")}")')
    cells = [("markdown", cfg["intro"]), ("code", env)] + cfg.get("pre", []) + [
        ("markdown", "## 1. Code (embedded) + data links + SAFE-04 reference tables"), ("code", setup),
        ("markdown", "## 2. Resume (optional)"), ("code", RESUME),
        ("markdown", "## 3. Jobs: SAFE-04 folds + final head, one per GPU at a time"), ("code", JOBS),
        ("markdown", "## 4. Registered gate (in the notebook)"), ("code", REPORT),
        *([("markdown", "## 4b. Final head of the chosen variant"), ("code", FINAL_LATE)] if cfg.get("final_after_report") else []),
        ("markdown", "## 5. Final fuser + submission.zip"), ("code", ZIP),
        ("markdown", "## 6. Summary"), ("code", SUMMARY)]
    B12.nb(cells, cfg["nb"])
    print(f"{cfg['nb']}: embedded {len(md5)} code files ({len(blob) / 1e3:.0f} kB base64)")


EXPS["T3RECIPE-01"]["intro"] = """
# FLAG 2027 — NB-51: T3RECIPE-01, Track 3's projector recipe as a new Bangla stream (end to end: gate + submission.zip)

**Why.** A teammate's system (Track 3, SUB21a) reaches CodaBench 20.18 (Bangla 14.65 / 22.75) against v11's 17.21 / 26.57. It differs from our
T3 head in three ways we never tested: (a) linear projectors with high dropout and AAM + cross-modal SupCon + prototype
alignment, gender-balanced batches; (b) VoxBlink as an auxiliary source with its OWN AAM head (VBSEL-01 only closed VoxBlink as
InfoNCE rows); (c) an ensemble of encoder pairs x seeds. Registered in `Experiment/T3RECIPE-01/README.md` before any result.

| arm | main identities | auxiliary sources (own AAM head each) | encoder pairs x seeds |
|---|---|---|---|
| **R** (decision) | v4 train + MAV-Celeb pool | VoxCeleb2 (LID-filtered) + VoxBlink-clean | 7 x 3 |
| RnoVB | same | VoxCeleb2 | 7 x 1 (compared with R's seed-0 models, `R_s0`) |
| RCC | same | VoxCeleb2 + VoxBlink | 1 (all faces x all voices) x 3 |
| Rmain | same | none | 7 x 1 (vs `R_s0`) |
| R19 | as R, SUB19B hyper-parameters (dropout .85 / .55, 64 aux ids) | VoxCeleb2 + VoxBlink-clean | 7 x 1 (vs `R_s0`) |

Hyper-parameters (amended 11/10, before any result): every arm but R19 uses Track 3's **SUB21a** (dropout face .70 / voice .40,
32 identities per auxiliary step; their CB 20.18).

**Gate (default, CLAUDE.md 4.3), computed in this notebook:** CA-FUSE(v11 streams + `trR`) vs v11 on SAFE-04 (5 folds, purge):
g cells >= +1.0 with >= 4/5 folds better in >= 2 g cells, held-out Urdu / Hindi >= -0.2; CI, no-v4, cells worse >= 0.5 reported.

**Then:** the decision arm is trained on every train person, its stream is computed for the 4 dev files (features from
`flag2027-e2e-dev`), the CA-FUSE fuser is refitted with it on every SAFE-04 row, and `submission.zip` is written (Bangla
re-scored, English files = v11 byte for byte). **Submit only if the DECISION line says PASSES.**

**Inputs (Add Input):** `flag2027-lab-data`, `flag2027-lab-vb` (**private**), `flag2027-eda-bundle`, `flag2027-e2e-dev`
(**private**, kaggle/flag2027-e2e-dev.zip). Optional: this notebook's previous output (resume).
**Settings:** **GPU T4 x2**, Internet OFF is fine. **Time:** about 4-5 h (2 folds at a time; the final head runs alongside); each fold trains 38 projectors of about 12k steps.
If a session stops (or the 10.5 h budget is reached): Save Version, attach the output, run again; finished jobs are skipped.
**Output:** `/kaggle/working/e2e_out/` (gate report, fold parquets, final models, fuser, zip) + `/kaggle/working/submission.zip`.
"""


def dev_only(src, what):
    """An extraction cell runs only on the dev split: an Evaluation run (E2E_SPLIT=eval, the dev output attached) reloads the
    exported final head and scores the new test files, so it needs no training features."""
    import textwrap
    head = ('if SPLIT != "dev":\n'
            f'    log("split {{}}: {what} skipped (the dev run\'s final head is reloaded)".format(SPLIT))\n'
            'else:\n')
    return head + textwrap.indent(src, "    ")


# ============================================================================ FLAG_53: extraction cells in front of the lab machinery
import build_eval_notebooks as _B12
import build_t3enc_notebook as _B40
PRE_ENV = r'''
import types
CACHE = Path(os.environ.get("FLAG_CACHE", "/tmp/flag_cache")); CACHE.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
'''
_pip = _B40.PIP.replace("!pip -q install ftfy regex kaldiio silero-vad 2>&1 | tail -1",
                        '!pip -q install ftfy regex kaldiio silero-vad av "speechbrain>=1.1.1" 2>&1 | tail -1')
assert _pip != _B40.PIP
EXPS["SCALE-ID-01"] = dict(
    nb="FLAG_53_scaleid.ipynb", code=["Experiment/SCALE-ID-01/run.py", "Experiment/E2E/v11_head.py"], tag="scaleid01", vb=False,
    pre=[("markdown", "## 0. Extraction: more VoxCeleb2 data from the official split archives (resumable per chunk)"),
         ("code", PRE_ENV), ("code", _pip), ("code", _B12.AUDIO12), ("code", _B40.CELL),
         ("code", dev_only((R / "kaggle/vendor_scaleid/vox2b_cell.py").read_text(encoding="utf-8"), "VoxCeleb2 extraction"))],
    intro="""
# FLAG 2027 — NB-53: SCALE-ID-01, more VoxCeleb2 data for v11's T3 head (end to end: extraction + gate + submission.zip)

**Why.** Identity count is our one proven lever (T3ENC-01, BIGB-01). FLAG_37 lost about 1,117 VoxCeleb2 dev speakers because
`vox2_mp4_1.zip` on the mirror is a TRUNCATED upload (27.0 GB vs ~50 GB). The same mirror holds the OFFICIAL split archives
(`vox2_dev_mp4_parta[a-i]`, `vox2_dev_aac_parta[a-h]`): intact zip64 files whose central directory sits in the last part
(checked 10/10). Every face track / voice clip is read by HTTP range, nothing is downloaded whole.
Registered in `Experiment/SCALE-ID-01/README.md` before any result.

**Extraction (section 0):** `new` = the missing dev speakers + the 120 VoxCeleb2 test speakers (6 faces / 8 voices each,
FLAG_37's rule); `more` = 6 more faces + 8 more voices for each of FLAG_37's 4,877 speakers (FLAG_43's design). Encoders:
FaRL16, CLIP-L, SimAM100 p / ft (T3 cell) + VoxLingua107 LID; a speaker with any clip of top-1 Bengali or p(bn) >= 0.2 loses
every clip (old rows included).

| arm | training rows | role |
|---|---|---|
| B | v11's head rows (v4 + MAV-Celeb + 4,852 VoxCeleb2) | same-device control |
| X / M | B + `new` / B + `more`, same epochs as B | diagnostics |
| **XM** | B + `new` + `more`, same epochs as B | **decision** (replaces v11's head) |

**Gate (default), computed here:** CA-FUSE(S6 + XM deep / cca + t3_relg) vs v11; sensitivity vs this run's B.
**Then:** final XM head (exported), dev streams, fuser refit, `submission.zip` (English = v11). **Submit only if PASSES.**

**Inputs (Add Input):** `flag2027-lab-data`, `flag2027-eda-bundle`, `flag2027-e2e-dev`. Optional: this notebook's previous
output (resume: extraction chunks, finished jobs). **Settings:** **GPU T4 x2**, **Internet ON** (Hugging Face, ModelScope,
GitHub). **Time:** extraction about 1.5-2.5 h (about 50k face tracks + 60k voice clips by range reads), then about 1 h of jobs.
**Output:** `/kaggle/working/feats_vox2b/` (keep it: a later run reuses it), `/kaggle/working/e2e_out/`, `/kaggle/working/submission.zip`.
""")


# ============================================================================ FLAG_52: short-crop extraction (FLAG_12 encoders + SimAM) in front
import build_vox2_notebook as _B37
PRE_ENV12 = r'''
import re, types, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
WORKDIR = WORK
CACHE = Path(os.environ.get("FLAG_CACHE", "/tmp/flag_cache"))
ECAPA_DIR = Path(os.environ.get("FLAG_ECAPA", str(CACHE / "ecapa_sb")))
OUT = WORK / "flag12_scan"                                  # FLAG_12's resume folder (nothing is extracted into it here)
EXT_SOURCES = ["v1_complete", "v2_complete"]                # MAV-Celeb rows (raw zips + flag2027-ext-meta)
SPLIT_FILTER = ["train"]                                    # only the organiser train list is needed (v4 voices)
ZIP_PREFIXES = ("train_set",)
ONLY_UNDER = None
for d_ in (CACHE, OUT):
    d_.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
'''
_seed12 = 'if re.search(r"feats[-_]eval", p.as_posix()) and out not in p.parents:'
DATA12_NOLAB = _B12.DATA12.replace(_seed12, 'if re.search(r"feats[-_]eval", p.as_posix()) and out not in p.parents and "/lab/kaggle/output/" not in p.as_posix():')
assert DATA12_NOLAB != _B12.DATA12
_pip12 = _B12.PIP12.replace("if not LOCAL:\n", "if not LOCAL:\n" + '''    !pip -q install kaldiio silero-vad 2>&1 | tail -1
    !pip -q install --no-deps "git+https://github.com/wenet-e2e/wespeaker.git@9fecd6cb4f47475d01761d87c826298dff4ef18c" 2>&1 | tail -1
''', 1)
assert _pip12 != _B12.PIP12
_pip12 = _pip12.replace("import numpy as np, pandas as pd", '''if LOCAL:                                         # local smoke test: wespeaker + its deps in an isolated folder
    for p_ in reversed([p for p in os.environ.get("FLAG_T3_PATH", "").split(";") if p]):
        sys.path.insert(0, p_)
import numpy as np, pandas as pd''', 1)
_short = (R / "kaggle/vendor_short/short_cell.py").read_text(encoding="utf-8").replace(
    "__TRIAL_JSON__", repr((R / "Experiment/SHORT-02/trial_voices.json").read_text(encoding="utf-8")))
EXPS["SHORT-02"] = dict(
    nb="FLAG_52_short.ipynb", code=["Experiment/SHORT-02/run.py", "Experiment/E2E/v11_head.py"], tag="short02", vb=False,
    pre=[("markdown", "## 0. Extraction: short crops of the validation and training voices (FLAG_12 encoders + SimAM; resumable)"),
         ("code", PRE_ENV12), ("code", _pip12), ("code", DATA12_NOLAB), ("code", _B12.AUDIO12), ("code", _B12.ENC12),
         ("code", "_speechbrain_import = speechbrain_import\n" + _B37.ORG_VOICE),
         ("code", "_build12, _release12 = build, release\n"), ("code", _B40.CELL),
         ("code", "release = _release12                     # FLAG_12's release (the T3 cell defines its own)\n"),
         ("code", dev_only(_short, "short-crop extraction"))],
    intro="""
# FLAG 2027 — NB-52: SHORT-02, v11 on SHORT voices + a T3 head trained with short crops (end to end: gate + submission.zip)

**Why.** 37 % of the dev Bangla voices (26 % English) last under 4 s; our validation has none, so the loss of v11 on short
voices was never measured (SHORT-01 only cut ImageBind). Registered in `Experiment/SHORT-02/README.md` before any result.

**Section 0 (extraction):** every voice of a SAFE-04 trial (2,887 v4 + 10,418 MAV-Celeb rows) cropped to 2 s and 4 s through
every voice encoder of v11's Bangla streams (organiser ECAPA = BTC, ECAPA-192, ImageBind, w2v2 age / gender, SimAM100 p / ft);
every training voice cropped once to U(2, 6) s through SimAM. Fixed seeds; crops of the signal the existing embeddings used.

**Jobs:** per fold, every v11 component is retrained / recomputed here and every trial is scored with full / 4 s / 2 s
voices. Arm **C** (decision) = v11's T3 head trained with the short-crop copies of its MAV-Celeb rows added.

**Report (in the notebook):** degradation table of v11 (fuser fitted on full-length rows, applied to each condition), stream
by stream; gate of C (short-voice g gain >= +1.0 in both lengths, full-length not worse than -0.2, held-out languages
>= -0.2); diagnostic D (fuser with log duration). Then the final C head, dev streams, fuser refit and `submission.zip`
(C replaces v11's head; English = v11). **Submit only if the DECISION line says PASSES.**

**Inputs (Add Input):** raw v4 (`train_set`), MAV-Celeb v1 / v2 (raw zips) + `flag2027-ext-meta`, `flag2027-lab-data`,
`flag2027-eda-bundle`, `flag2027-e2e-dev`. Optional: this notebook's previous output (resume).
**Settings:** **GPU T4 x2**, **Internet ON** (ImageBind, w2v2, ECAPA x2, SimAM x2). **Time:** extraction about 1-1.5 h,
jobs about 1 h, report about 10 min.
**Output:** `/kaggle/working/feats_short/` (keep it), `/kaggle/working/e2e_out/`, `/kaggle/working/submission.zip`.
""")


EXPS["HEADSWEEP-01"] = dict(
    nb="FLAG_54_headsweep.ipynb", code=["Experiment/HEADSWEEP-01/run.py", "Experiment/E2E/v11_head.py"], tag="headsweep01", vb=False,
    final_after_report=True,
    intro="""
# FLAG 2027 — NB-54: HEADSWEEP-01, a broad sweep of v11's T3 head recipe with a NESTED choice (gate + submission.zip)

**Why.** v11's T3 head still uses the EXP-007 recipe (hid 512, emb 128, dropout 0.3, face PCA 256, tau 0.07, ~1 200 steps)
tuned when we had 70 identities; it now trains on ~5 000. The one knob with evidence was never tried on it: training
longer (T3ENC-01: 3 600 steps +2.21 vs 1 200 steps +1.59). Registered in `Experiment/HEADSWEEP-01/README.md`.

| group | variants (one knob each vs v11's recipe `S0`) |
|---|---|
| length | L2 / L3 / L5 = 2 400 / 3 600 / 6 000 steps |
| capacity | E256, H1024 (hid 1 024 + emb 256), D1 / D5 (dropout 0.1 / 0.5), WD3, NOPCA (1 280-d face), PCA512 |
| loss | T05 / T10 (tau), TL (learned tau), M10 (cosine margin), HN (hard-negative weighting), IM (+ intra-modal InfoNCE) |
| batch / nets | K4 (P 256 x K 4), N6 (6 nets) |
| combination | BIG = L3 + H1024 + D1 + TL |

**Decision (no winner's curse):** `SEL` = for every outer fold, the variant that wins on the OTHER folds (inner fusers
purged of the persons of both folds) is scored on that fold; `SEL` vs v11 by the default gate. Every variant is reported
too (diagnostic). The final head uses the variant the same criterion picks on all 5 folds; it replaces v11's head.
**Submit only if the DECISION line (SEL) says PASSES.**

**Inputs (Add Input):** `flag2027-lab-data`, `flag2027-eda-bundle`, `flag2027-e2e-dev`. Optional: this notebook's previous
output (resume per fold AND per variant inside a fold). **Settings:** **GPU T4 x2**, Internet OFF is fine.
**Time:** about 32 head-trainings worth per fold (~70-80 min on a T4), 2 folds at a time: about 4 h, + report ~20 min + final.
""")


EXPS["DSEL-01"] = dict(
    nb="FLAG_55_datasel.ipynb", code=["Experiment/DSEL-01/run.py", "Experiment/DSEL-01/vox2_groups.json", "Experiment/E2E/v11_head.py"],
    tag="dsel01", vb=False, final_after_report=True,
    intro="""
# FLAG 2027 — NB-55: DSEL-01, which PARTS of the training data help v11's T3 head? (nested choice, gate + submission.zip)

**Why.** Not every part of the data has to help. v11's T3 head trains on ~5 000 identities: v4 train, MAV-Celeb v1 / v2
and 4 852 VoxCeleb2 speakers (LID-filtered), drawn uniformly by person (in-domain persons = 3.5 % of a batch). Never measured on
this head: clip label noise (about 1 % of external voice clips look like another person), the source mixture, and the value of
each VoxCeleb2 language part against a random part of the same size. Registered in `Experiment/DSEL-01/README.md`.

| group | variants (data / sampling only; v11's recipe `S0`) |
|---|---|
| A. label noise | NV1 / NV5 (voice LOO < q1 / q5), NF5 (face LOO < q5), NVF5, NSPK (79 noisy VoxCeleb2 speakers), TCH5 (cross-fitted teacher, 5 % rows) |
| B. mixture | WIN2 / WIN4 (v4 + MAV persons x2 / x4 in the PK draw), WPOP4 (+ VoxCeleb2 South-Asian, x4) |
| C. parts out | dV4, dMAV, dSA, dOTH, dEU (1 541 speakers), dEN (2 774); controls rEU / rEN (as many random speakers) |

**Decision (no winner's curse):** `SEL` = for every outer fold, the variant that wins on the OTHER folds (inner fusers purged
of the persons of both folds) is scored on that fold; `SEL` vs v11 by the default gate. The report adds the registered
**part-value table** (each part out vs S0; EU / EN vs their random controls). The final head uses the variant the same
criterion picks on all 5 folds (controls excluded). **Submit only if the DECISION line (SEL) says PASSES.**

**Inputs (Add Input):** `flag2027-lab-data`, `flag2027-eda-bundle`, `flag2027-e2e-dev`. Optional: this notebook's previous
output (resume per fold AND per variant inside a fold). **Settings:** **GPU T4 x2**, Internet OFF is fine.
**Time:** about 18 head-trainings worth per fold (~45 min on a T4), 2 folds at a time: about 2.5 h, + report ~30 min + final.
""")


EXPS["T3REPRO-01"] = dict(
    nb="FLAG_56_t3repro.ipynb", code=["Experiment/T3REPRO-01/run.py", "Experiment/T3REPRO-01/xteam_trials.csv",
                                      "Experiment/T3RECIPE-01/t3r.py"],
    tag="t3repro01", vb=True, folds="",
    intro="""
# FLAG 2027 — NB-56: T3REPRO-01, Track 3's SUB21a rebuilt on our data, scoring the XTEAM-01 package + the dev files

**Why.** The team must compare / fuse Track 3's system (CB 20.18) with v11 on a common labelled validation, and Track 3 cannot
score the package themselves, so we rebuild SUB21a from their report: main = the 70 FLAG train persons only (every MAV-Celeb
person of the XTEAM-01 package is new), auxiliary sources VoxCeleb2 (LID-filtered) + VoxBlink-clean with one AAM head each,
SUB21a hyper-parameters, 5 branches x 5 seeds (ECAPA-1024 / ResNet293 are not available: cat4 -> cat2). Registered in
`Experiment/T3REPRO-01/README.md` (and an amendment of XTEAM-01) before any result.

**This notebook** runs ONE job (`final`): the 25 projectors (split over the two GPUs), then the per-branch cosines of the
24 214 XTEAM-01 trials (no labels inside) and of the 4 dev files (features from `flag2027-e2e-dev`), with a level-0 self-test.
The score formula, the XTEAM-01 report (A: alone vs v11, B: nested fusion) and the zips are computed locally from `e2e_out/`.

**Inputs (Add Input):** `flag2027-lab-data`, `flag2027-lab-vb`, `flag2027-eda-bundle`, `flag2027-e2e-dev`. **GPU T4 x2**,
Internet OFF. **Time:** about 25 projectors x ~8 min / 2 GPUs = about 2 h.
""")


EXPS["T3RECIPE-02"] = dict(
    nb="FLAG_57_t3recipe02.ipynb", code=["Experiment/T3RECIPE-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py"],
    tag="t3recipe02", vb=True,
    intro="""
# FLAG 2027 — NB-57: T3RECIPE-02, one full face x voice pair (CC) with Track 3's SUB21a recipe (gate + submission.zip)

**Why.** In T3RECIPE-01 the decision arm (7 pairs, SUB19B) failed (g +0.46), but the single full pair RCC reached g +0.88 on 5
folds and +1.17 on folds 4-5 alone, which the pre-registered N3 rule (log 176) required before opening this experiment.
ONE arm, no selection: CC5 = [FaRL16 | CLIP-L] x [SimAM p | ft], SUB21a (dropout .70 / .40, 32 auxiliary ids per step),
5 seeds, VoxCeleb2 + VoxBlink with their own AAM heads; added to v11's Bangla CA-FUSE. Registered in
`Experiment/T3RECIPE-02/README.md`.

**Gate (default, CLAUDE.md 4.3) in this notebook; final head; CA-FUSE refit; zip (English = v11).** Submit only if PASSES.

**Inputs:** `flag2027-lab-data`, `flag2027-lab-vb`, `flag2027-eda-bundle`, `flag2027-e2e-dev`. **GPU T4 x2**, Internet OFF.
**Time:** 5 projectors per fold (~30 min on a T4), 2 folds at a time + the final head: about 2 h.
""")


EXPS["DSEL-02"] = dict(
    nb="FLAG_58_dsel02.ipynb", code=["Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py",
                                     "Experiment/DSEL-01/vox2_groups.json"],
    tag="dsel02", vb=True,
    intro="""
# FLAG 2027 — NB-58: DSEL-02, the 305 VoxCeleb2 South-Asian speakers moved into the MAIN identities of CC5 (gate + zip)

**Why.** DSEL-01: the VoxCeleb2 South-Asian speakers are the most valuable part per identity. In Track 3's recipe only the
main identities get gender-balanced batches and same-gender SupCon (Track 3: about -1.0 CB); these speakers sit in the
auxiliary VoxCeleb2 head. One arm SAM = CC5 (T3RECIPE-02: full pair, SUB21a, 5 seeds) with them in the main set.
Registered in `Experiment/DSEL-02/README.md`. Control = trCC5 of T3RECIPE-02 (attach its output).

**Inputs:** `flag2027-lab-data`, `flag2027-lab-vb`, `flag2027-eda-bundle`, `flag2027-e2e-dev` + the output of
`flag-2027-nb-57-t3recipe02` (kernel source). **GPU T4 x2**, Internet OFF. **Time:** about 2 h.
""")


EXPS["FACELORA-02"] = dict(
    nb="FLAG_59_facelora02.ipynb", code=["Experiment/FACELORA-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py"],
    tag="facelora02", vb=False, final=False,
    intro="""
# FLAG 2027 — NB-59: FACELORA-02, LoRA-adapted face encoders (FaRL16, CLIP-L) inside Track 3's recipe (gate only)

**Why.** Every gain so far came from heads on FROZEN encoders. FLAG_45 / FLAG_45c trained a LoRA adapter on each face encoder
against frozen SimAM voices of the Bengali-filtered VoxCeleb2 identities (no MAV-Celeb identity, so the adapters do not depend
on the SAFE-04 folds). This notebook asks whether the adapted embeddings LEARN better in the CC recipe (t3r.SUB21A, 3 seeds,
VoxCeleb2 as the only auxiliary source; VoxBlink faces have no adapted embedding). Registered in
`Experiment/FACELORA-02/README.md`.

| arm | faces | |
|---|---|---|
| B0 | farl16 + clipL (frozen) | control |
| FA | farlA + clipL | |
| CA | farl16 + clipA | |
| **FCA** | farlA + clipA | decision |

Same seeds, same rows, same dimension (1280) in every arm: only the face features differ.

**Rules (in the report):** (1) FCA - B0 (paired CA-FUSE, g cells) >= +0.3 with >= 4/5 folds -> the adapted encoders HELP;
(2) default gate of CA-FUSE(v11 + trFCA) vs v11; (3) FCA replaces CC5 only if (1), (2) and FCA - CC5 >= +0.3 with >= 4/5 folds.
No final head / zip in this run.

**Inputs (private copies, 12/10):** `flag2027-lab-data-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`, `flag2027-cc5-oof-p`
(trCC5 of T3RECIPE-02) and `flag2027-feats-facelora-p` (the adapted arrays `face_farlA_*`, `face_clipA_*` of FLAG_45 / FLAG_45c).
**GPU T4 x2**, Internet OFF. **Time:** 12 projectors per fold (~50-60 min on a T4), 2 folds at a time: about 3 h.
""")


EXPS["CCOPT-01"] = dict(
    nb="FLAG_60_ccopt01.ipynb", code=["Experiment/CCOPT-01/run.py", "Experiment/CCOPT-01/t3x.py", "Experiment/T3RECIPE-01/run.py",
                                      "Experiment/T3RECIPE-01/t3r.py"],
    tag="ccopt01", vb=True, final=False,
    intro="""
# FLAG 2027 — NB-60: CCOPT-01, optimisation changes of the CC recipe (R-Drop, cross-batch memory, block LayerNorm; gate only)

**Why.** CC5 (T3RECIPE-02: Track 3's projector recipe on the full face x voice pair) is the best CB system (21.31). Its recipe
was copied, never optimised for our rows; HEADSWEEP-01 closed dropout / size / epochs, but not these mechanisms. Five arms,
one seed each, the same pair, seeds, rows and sources (VoxCeleb2 + VoxBlink aux). Registered in `Experiment/CCOPT-01/README.md`.

| arm | change |
|---|---|
| B | control (t3r.SUB21A = CC5's seed 0) + gradient probe of the loss terms every 200 steps (no effect on training) |
| RD0 | main batch forwarded twice, mean of the two losses (control of RD) |
| RD | RD0 + 1.0 x symmetric KL of the in-batch matching distributions (R-Drop) |
| XB | cross-batch memory: 4096 rows per source, + 0.5 x SupCon against batch + memory after 1000 steps |
| BN | one LayerNorm per encoder block instead of one over the concatenation |

**Rules (in the report):** (1) X - B (paired CA-FUSE, g cells) >= +0.3 with >= 4/5 folds -> X HELPS (RD also needs RD - RD0);
(2) a helping arm replaces the CC recipe only if it passes the default gate vs v11 and X - CC5 >= +0.3 with >= 4/5 folds;
(3) gradient probe of B: conflict / imbalance between the main and auxiliary losses. No final head / zip in this run.

**Inputs:** `flag2027-lab-data`, `flag2027-lab-vb`, `flag2027-eda-bundle`, `flag2027-e2e-dev` + the output of
`flag-2027-nb-57-t3recipe02` (trCC5, kernel source). **GPU T4 x2**, Internet OFF. **Time:** 5 projectors per fold
(~45 min on a T4), 2 folds at a time: about 2.5 h.
""")
OPTIONAL_DIRS["Experiment/SCALE-VB/feats_vb"].append("CCOPT-01")


EXPS["DSEL-03"] = dict(
    nb="FLAG_61_dsel03.ipynb", code=["Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py",
                                     "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="dsel03", vb=True,
    intro="""
# FLAG 2027 — NB-61: DSEL-03, the VoxBlink South-Asian speakers moved into the MAIN identities of SAM (gate + zip)

**Why.** DSEL-02: SAM (CC5 + the 305 VoxCeleb2 South-Asian speakers in the main identities) PASSES the default gate (g +1.06).
VoxBlink-clean holds 1,118 South-Asian speakers by its OWN metadata (loc / lan; 447 female), today only in the VoxBlink auxiliary
AAM head. One arm SAM2 = SAM + those speakers in the main identities (gender-balanced batches, same-gender SupCon), same
hyper-parameters (t3r.SUB21A, 5 seeds). Registered in `Experiment/DSEL-03/README.md`.

**Report:** gate vs v11; paired SAM2 - CC5 (rule: replace CC5 only if the gate passes and >= +0.3 g with >= 4/5 folds) and
SAM2 - SAM; g-cell gain split by the face's sex.

**Inputs (private copies):** `flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`,
`flag2027-cc5-oof-p` (trCC5), `flag2027-dsel02-oof-p` (trSAM). **GPU T4 x2**, Internet OFF. **Time:** about 2.5-3 h.
""")


EXPS["SCALE-CC-01"] = dict(
    nb="FLAG_62_scalecc01.ipynb", code=["Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json",
                                        "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                        "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="scalecc01", vb=True, final_after_report=True,
    intro="""
# FLAG 2027 — NB-62: SCALE-CC-01, ~21k identities in Track 3's recipe + hyper-parameters re-tuned for the larger data

**Why.** More identities help in the CC recipe (VoxBlink with its own AAM head; DSEL-02: South-Asian speakers in the main
identities PASS the gate), not in the P512 head (SCALE-VB, SCALE-ID-01). SUB21a was tuned for less data: 32 auxiliary identities
per step (a VoxBlink identity is seen ~26 times per run), a main batch of 32 identities. Registered in
`Experiment/SCALE-CC-01/README.md`.

| part | identities |
|---|---|
| main (gender-balanced batches, same-gender SupCon) | v4 + MAV + every South-Asian VoxCeleb2 speaker (FLAG_37 + FLAG_53) + VoxBlink South-Asian speakers: ~1,630 |
| VoxCeleb2 aux (own AAM head) | the other VoxCeleb2 speakers, FLAG_37 + FLAG_53 rows: ~5,700 |
| VoxBlink aux (own AAM head) | the other VoxBlink-clean speakers: ~13,900 |

Variants (1 seed each, the same seed): V0 SUB21a; V1 aux_ids 64; V2 epochs 80; V3 main batch 64 ids (epochs 80, same steps);
V4 dropout .55 / .30. **Nested selection SEL** (best inner g gain), gate of SEL vs v11; replaces CC5 only if SEL - CC5 >= +0.3
(g, >= 4/5 folds). Final head = the variant with the best oof g gain, 5 seeds (after the report).

**Inputs (private copies):** `flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`,
`flag2027-cc5-oof-p`, `flag2027-dsel02-oof-p`, `flag2027-feats-vox2b-p` (FLAG_53 VoxCeleb2 arrays). **GPU T4 x2**, Internet OFF.
**Time:** about 4 h for the folds + 1-1.5 h for the final head.
""")


EXPS["SCALE-CC-02"] = dict(
    nb="FLAG_64_scalecc02.ipynb", code=["Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json",
                                        "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                        "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="scalecc02", vb=True, final_after_report=True,
    intro="""
# FLAG 2027 — NB-64: SCALE-CC-02, ~21k identities in Track 3's recipe with VoxBlink in the AUXILIARY head only + re-tuned parameters

**Why.** DSEL-03: moving 1,105 VoxBlink South-Asian speakers into the MAIN identities hurt the g cells (SAM2 - SAM -0.68): they
made 70 % of the main identities and diluted the MAV ones (34 % -> 10 %). SCALE-CC-02 = SCALE-CC-01 with that one change: main =
v4 + MAV + every South-Asian VoxCeleb2 speaker (FLAG_37 + FLAG_53, ~525 ids, MAV ~30 %); VoxCeleb2 aux = the other VoxCeleb2
speakers (FLAG_37 + FLAG_53, ~5,700); VoxBlink aux = all of VoxBlink-clean (~14,970). Registered in `Experiment/SCALE-CC-02/README.md`.

Variants (1 seed each, same seed): V0 SUB21a; V1 aux_ids 64; V2 epochs 80; V3 main batch 64 ids (epochs 80); V4 dropout .55 / .30;
**V5 = V1 + V3 + V4** (amendment 12/10 after SCALE-CC-01, before any SCALE-CC-02 result). Nested selection SEL; gate of SEL vs v11;
replaces CC5 only if SEL - CC5 >= +0.3 (g, >= 4/5 folds). Final head = the variant with the best oof g gain, 5 seeds (after the report).

**Inputs (private copies):** `flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`,
`flag2027-cc5-oof-p`, `flag2027-dsel02-oof-p`, `flag2027-feats-vox2b-p`. **GPU T4 x2**, Internet OFF. **Time:** about 4.5 h + 1-1.5 h.
""")


EXPS["DCUR-01"] = dict(
    nb="FLAG_66_dcur01.ipynb", code=["Experiment/DCUR-01/run.py", "Experiment/DCUR-01/keep_flags.npz", "Experiment/SCALE-CC-02/run.py",
                                     "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json",
                                     "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py",
                                     "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="dcur01", vb=True,
    intro="""
# FLAG 2027 — NB-66: DCUR-01, quality-filtered external samples (curated pool) in Track 3's recipe on ~21k identities

**Why.** The user (12/10): select the dataset first - keep the highest-quality samples, save the pool, grow it along the identity
axis. Quality metrics per external sample (face / voice consistency with the identity's other samples, clip duration, VoxBlink
face box and crop) and levels anchored on VoxCeleb2's 5th / 10th percentiles are in `Experiment/DCUR-01/curate.py`; the keep
flags are embedded (`keep_flags.npz`). Registered in `Experiment/DCUR-01/README.md`.

| arm | external samples (data D4 of SCALE-CC-02: main v4 + MAV + VoxCeleb2 SA; aux VoxCeleb2 + VoxBlink) |
|---|---|
| U | every sample (level U reproduces the existing row builders exactly) |
| **C5** | samples passing the 5th-percentile level (decision) |
| C5K | C5 + the best face / voice of an identity left with none (no identity lost) |
| C10 | stricter, 10th-percentile level |

SUB21a, 2 seeds per arm (same seeds). **Rules:** (1) C5 - U >= +0.3 (paired CA-FUSE, g cells, >= 4/5 folds) -> the curated pool
becomes the default training pool; (2) default gate of C5 vs v11; (3) replaces CC5 only if C5 - CC5 >= +0.3 (g, >= 4/5 folds).

**Inputs (private copies):** `flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`,
`flag2027-cc5-oof-p`, `flag2027-dsel02-oof-p`, `flag2027-feats-vox2b-p`. **GPU T4 x2**, Internet OFF. **Time:** about 4 h.
""")


EXPS["DCUR-02"] = dict(
    nb="FLAG_68_dcur02.ipynb", code=["Experiment/DCUR-02/run.py", "Experiment/DCUR-01/run.py", "Experiment/DCUR-01/keep_flags.npz",
                                     "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json",
                                     "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                     "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="dcur02", vb=True,
    intro="""
# FLAG 2027 — NB-68: DCUR-02, VoxBlink faces in the FLAG framing (FLAG_67) inside Track 3's recipe on ~21k identities

**Why.** YuNet on FLAG v4 TRAIN faces: face box 0.526 x 0.698, centre (0.501, 0.378); our VoxCeleb2 frames match, our VoxBlink
crops do not (centre y 0.500). FLAG_67 re-cut the 68,041 VoxBlink faces in the FLAG framing (frozen FaRL16 / CLIP-L).
Registered in `Experiment/DCUR-02/README.md`.

| arm | VoxBlink faces | VoxBlink South-Asian speakers |
|---|---|---|
| V-OLD | FLAG_46 loose crop | auxiliary head (D4) |
| **V-FLAG** | FLAG framing | auxiliary head (D4) |
| V-FLAG-SA | FLAG framing | MAIN identities (D3) |

SUB21a, 2 seeds per arm (same seeds; only the VoxBlink face columns / the VoxBlink SA placement differ). **Rules:** (1) V-FLAG -
V-OLD >= +0.3 (paired CA-FUSE, g cells, >= 4/5 folds) -> the FLAG framing replaces the old crops everywhere; (2) default gate of
V-FLAG vs v11; (3) replaces CC5 only if V-FLAG - CC5 >= +0.3 (g, >= 4/5 folds).

**Inputs (private copies):** `flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`,
`flag2027-cc5-oof-p`, `flag2027-dsel02-oof-p`, `flag2027-feats-vox2b-p`, `flag2027-vbflag-feats-p` (FLAG_67 arrays).
**GPU T4 x2**, Internet OFF. **Time:** about 3.5 h.
""")


EXPS["DVAL-01"] = dict(
    nb="FLAG_70_dval01.ipynb", code=["Experiment/DVAL-01/run.py", "Experiment/DCUR-01/run.py", "Experiment/DCUR-01/keep_flags.npz",
                                     "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json",
                                     "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                     "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="dval01", vb=True, final=False,
    intro="""
# FLAG 2027 — NB-70: DVAL-01, value of groups of auxiliary identities (drop 20 % by an attribute vs drop 20 % at random)

**Why.** The user's plan (12/10): keep identities with reliable labels, many independent observations and useful information, not
only the easy ones; remove noise and wrong labels, not valuable difficulty; value identities by GROUPS. Dropping a group changes
both its attribute and the identity count, so every group is compared with a random group of the same size and sources.
Registered in `Experiment/DVAL-01/README.md`.

| arm | auxiliary identities dropped (20 % of each source: VoxCeleb2 aux, VoxBlink) |
|---|---|
| B | none |
| **R20** | random (seed 0): control of every group |
| Q20 | lowest identity reliability (face / voice consistency with the identity's other samples) |
| H20 | hardest (closest to the 10 nearest same-gender identities, face and voice) |
| E20 | easiest |
| D20 | fewest videos |

CC recipe (SUB21a) on D4 at the sample level of DCUR-01's outcome; main identities unchanged; 2 seeds per arm (same seeds).
**Rules:** value of g = (drop g) - (drop R20), paired CA-FUSE, g cells: >= +0.3 with >= 4/5 folds = HARMFUL; <= -0.3 with >= 4/5
folds = VALUABLE; else no difference. Gate only (no final head here).

**Inputs (private copies):** `flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`,
`flag2027-cc5-oof-p`, `flag2027-dsel02-oof-p`, `flag2027-feats-vox2b-p`. **GPU T4 x2**, Internet OFF. **Time:** about 5-6 h.
""")


EXPS["CONTENT-01"] = dict(
    nb="FLAG_71_content01.ipynb", code=["Experiment/CONTENT-01/run.py", "Experiment/CONTENT-01/analysis.py", "Experiment/DCUR-01/run.py",
                                        "Experiment/DCUR-01/keep_flags.npz", "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py",
                                        "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json",
                                        "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py",
                                        "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="content01", vb=True, final=False,
    intro="""
# FLAG 2027 — NB-71: CONTENT-01, how much the face-voice score of a FIXED face moves when the utterance changes

**Why.** The user's idea (13/10): every speaker says different sentences; would the same sentence for everybody (TTS) make the
voice side better? Before generating anything, measure directly how sensitive the face-voice score is to the utterance of the
same person (the user's review: voice-voice EER alone cannot decide this). Registered in `Experiment/CONTENT-01/README.md`.

| step | what |
|---|---|
| scorer | the CC recipe (SUB21a) on D4 at level U (= DCUR-01 U), 2 seeds per SAFE-04 fold (purged) |
| grid | per held-out person: up to 3 face rows x up to 8 utterances (4 videos x 2; both languages of a MAV person); wrong persons: 5 same-gender (g) and 5 other-gender (ng) |
| M5 (rule 1) | EER of single utterances vs the mean of 4 utterances of the person (same face): the CEILING of any utterance-invariant scoring; < 1.0 closes the content-control group |
| M4 (rule 2) | within-face positive-score variance: language, log duration, same video as the face, between videos (channel / session), within one video (content proxy) |

No new data, no TTS, no dev / test data. **Inputs (private copies):** `flag2027-lab-data-p`, `flag2027-lab-vb-p`,
`flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`, `flag2027-cc5-oof-p`, `flag2027-dsel02-oof-p`, `flag2027-feats-vox2b-p`.
**GPU T4 x2**, Internet OFF. **Time:** about 1-1.5 h. **Output:** `e2e_out/content01_grid_fold*.parquet`, `content01_report.json`.
""")


EXPS["AUXP-01"] = dict(
    nb="FLAG_73_auxp01.ipynb", code=["Experiment/AUXP-01/run.py", "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py",
                                     "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py",
                                     "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py",
                                     "Experiment/DSEL-01/vox2_groups.json"],
    tag="auxp01", vb=True, final_after_report=True,   # 14/10: rule 3 passed -> final head run (resume: attach the gate run)
    intro="""
# FLAG 2027 — NB-73: AUXP-01, AVSpeech + LRS3 pseudo-identities as auxiliary heads in Track 3's recipe, re-tuned for the larger data

**Why.** COLLECT-01 (AVSpeech, ~17-19k pseudo-identities) and COLLECT-02 (LRS3-TED, ~4-5k talks) bring the training pool to ~42k
identities (Track 3's size). Weak labels (one clip per identity) -> AUXILIARY heads only, never main identities (DSEL-03). The user
(13/10): re-tune when the data change, and compare tuned vs tuned. Registered in `Experiment/AUXP-01/README.md`.

| arm | aux sources | extra tuning (all = SUB21a + dropout .55 / .30 = SCALE-CC-02 V4) |
|---|---|---|
| **B** | VoxCeleb2, VoxBlink | none: the tuned control on the old data |
| V0 | + AVSpeech, LRS3 | none |
| V1 | + AVSpeech, LRS3 | aux_ids 64 |
| V2 | + AVSpeech, LRS3 | epochs 80 |
| V3 | + AVSpeech, LRS3 | aux_weight 0.5 |
| V4 | + AVSpeech, LRS3 | aux_ids 64, ids_per_batch 64, epochs 80 |

1 seed per arm (same seed); SEL = nested choice among V0-V4. **Rules:** (1) SEL - B >= +0.3 mean and better at >= 4/5 folds ->
the new data HELP; (2) default gate of SEL vs v11; (3) replaces CC5 only if SEL - CC5 >= +0.3 (g, >= 4/5 folds) -> final head in a
later run (attach this output).

**Inputs:** as FLAG_64 (`flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`,
`flag2027-cc5-oof-p`, `flag2027-dsel02-oof-p`, `flag2027-feats-vox2b-p`) + notebook outputs `feats_avs/` (FLAG_69) and
`feats_lrs3/` (FLAG_72). **GPU T4 x2**, Internet OFF. **Time:** about 7-9 h (resumable per fold).
""")

EXPS["AUXP-02"] = dict(
    nb="FLAG_75_auxp02.ipynb", code=["Experiment/AUXP-02/run.py", "Experiment/AUXP-01/run.py", "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py",
                                     "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py",
                                     "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py",
                                     "Experiment/DSEL-01/vox2_groups.json"],
    tag="auxp02", vb=True, final=False,
    intro="""
# FLAG 2027 — NB-75: AUXP-02, South-Asian AVSpeech pseudo-identities (multi-video groups) as their own auxiliary head, tuned vs tuned

**Why.** AUXP-01: generic AVSpeech + LRS3 made no difference over the tuned control; the gain over CC5 came from the tuned recipe on
D4 (CB 21.17, within noise of CC5 / SAM). South-Asian identities are the most valuable per head (DSEL-01). COLLECT-03 / 03b: 3,656
South-Asian AVSpeech pseudo-identities, 333 with 2-12 videos (cross-video rows). Registered in `Experiment/AUXP-02/README.md`.

| arm | aux sources (main = D4 in every arm) | tuning (all SUB21a + dropout .55 / .30) |
|---|---|---|
| C1 / C2 / C4 | vox, vb, avs, lrs3 | aux_ids 64 / epochs 80 / aux_ids 64 + batch 64 + epochs 80 (= AUXP-01 V1 / V2 / V4) |
| S1 / S2 / S4 | + sa | as C1 / C2 / C4 |
| S4w2 / S4w05 | + sa weighted 2.0 / 0.5 | as C4 |

SEL = nested choice among S*, SELc among C*. **Rules:** (1) SEL - SELc >= +0.3 mean and better at >= 4/5 folds -> the South-Asian
data HELP; (2) default gate of SEL vs v11; (3) candidate = (1) HELPS and (2) passes -> final head later, zip, one CB run.

**One fold per notebook** (`LAB_FOLDS`); the report runs locally once all five fold tables are in. **Inputs:** as FLAG_73 +
`flag2027-feats-avssamv-p`. **GPU T4 x2**, Internet OFF. **Time:** about 2.5-3 h per fold.
""")

EXPS["ADAPT-01"] = dict(
    nb="FLAG_76_adapt01.ipynb", code=["Experiment/ADAPT-01/run.py", "Experiment/AUXP-01/run.py", "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py",
                                      "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py",
                                      "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py",
                                      "Experiment/DSEL-01/vox2_groups.json"],
    tag="adapt01", vb=True, final=False,
    intro="""
# FLAG 2027 — NB-76: ADAPT-01, stage 2 = in-domain adaptation of the CC head (FAME 2024 winner's last stage)

**Why.** ENS-CC-01 (14/10): CC heads trained on different data are near copies (rank correlation 0.87-0.99 on same-gender
negatives); mixing more data into one training run has saturated. FAME 2024's winner adapts on MAV only as the last stage (-8.8 EER
on English without it). Registered in `Experiment/ADAPT-01/README.md`.

Stage 1 of every arm = AUXP-01 V4 (D4 main; aux vox, vb, avs, lrs3; SUB21a + dropout .55 / .30, aux_ids 64, batch 64, 80 epochs),
trained ONCE per fold (cached in `t3r.train`). Stage 2 = the main-set loss only, on a subset of main identities:

| arm | stage-2 identities | epochs | lr x |
|---|---|---|---|
| A0 | none (stage 1 alone) | 0 | - |
| F5 / F15 / F5h / F15l | v4 + MAV | 5 / 15 / 5 / 15 | 0.3 / 0.3 / 1.0 / 0.1 |
| M5 | all D4 | 5 | 0.3 |
| V10 | v4 only | 10 | 0.3 |

SEL = nested choice among the stage-2 arms. **Rules:** (1) SEL - A0 >= +0.3 mean, better at >= 4/5 folds, and v4 g alone >= 0 ->
stage 2 HELPS; (2) default gate of SEL vs v11; (3) candidate = (1) and (2) -> final head later, zip, one CB run.

**One fold per notebook** (`LAB_FOLDS`); the report runs locally. **Inputs:** as FLAG_73 (AUXP-01). **GPU T4 x2**, Internet OFF.
**Time:** about 1-1.5 h per fold.
""")

EXPS["ENC-CC-01"] = dict(
    nb="FLAG_77_enccc01.ipynb", code=["Experiment/ENC-CC-01/run.py", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                      "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="enccc01", vb=True, final=False,
    intro="""
# FLAG 2027 — NB-77: ENC-CC-01, Track 3's `cat4` voice (SimAM p | ft | ResNet293-LM | ECAPA1024-LM) in the CC recipe

**Why.** ENS-CC-01 (14/10): more data into the CC head has saturated; new information is needed. Track 3's strongest branch uses
the 4-encoder voice concatenation; ENC-SCREEN-02 rejected it in the old head A, not in the CC recipe. Registered in
`Experiment/ENC-CC-01/README.md`. Main set v4 + MAV only (the auxiliary sources have no ResNet293 / ECAPA features yet), CC recipe
(SUB21a + dropout .55 / .30), 3 seeds: C2 (current voice) vs **C4** (cat4), diagnostics C3r / C3e.
**Rule:** C4 - C2 >= +0.5 on g cells, better at >= 4/5 folds -> extract ResNet293 + ECAPA1024 for every auxiliary source.

**Inputs:** `flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`, `flag2027-feats-screen2-p`.
**GPU T4 x2**, Internet OFF. **Time:** about 30-60 min (all five folds and the report in this notebook).
""")

EXPS["ENC-CC-02"] = dict(
    nb="FLAG_78_enccc02.ipynb", code=["Experiment/ENC-CC-02/run.py", "Experiment/ENC-CC-01/run.py", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                      "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="enccc02", vb=True, final=False,
    intro="""
# FLAG 2027 — NB-78: ENC-CC-02, DINOv2 / SigLIP2 faces and ECAPA2 voices in the CC recipe

**Why.** The CC head is what transferred to CB English (EN-CC-01, -4.65); more data into it has saturated (ENS-CC-01), so new
information must come from other encoders. DINOv2, SigLIP2 and ECAPA2 were only screened in the old head A (DINOv2 + ECAPA2
+0.99 vs a +1.0 bar). Registered in `Experiment/ENC-CC-02/README.md`. Main set v4 + MAV only (auxiliary sources have no such
features yet), CC recipe (SUB21a + dropout .55 / .30), 3 seeds: C2 control; FD (+DINOv2), FS (+SigLIP2), VE (+ECAPA2), FDVE.
**Rule:** an arm >= +0.5 over C2 on g cells, better at >= 4/5 folds -> extract that encoder at scale.

**Inputs:** `flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-eda-bundle-p`, `flag2027-e2e-dev-p`, `flag2027-feats-screen-p`,
`flag2027-feats-screen2-p`. **GPU T4 x2**, Internet OFF. **Time:** about 1 h per 2-3 folds (`LAB_FOLDS`).
""")

EXPS["UTTDIFF-01"] = dict(
    nb="FLAG_80_uttdiff01.ipynb", code=["Experiment/UTTDIFF-01/run.py", "Experiment/AUXP-01/run.py", "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py",
                                        "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py",
                                        "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py",
                                        "Experiment/DSEL-01/vox2_groups.json"],
    tag="uttdiff01", vb=True, final=False,
    intro="""
# FLAG 2027 — NB-80: UTTDIFF-01, auxiliary utterances filtered by face-voice difficulty; whole-clip voices for pseudo identities

**Why.** DQUAL-02 (14/10): hard utterances (low face-voice margin) are spread out and not explained by length or phonetics; the
pseudo identities use half clips (2.7 s) although whole clips (5.4 s) exist. Registered in `Experiment/UTTDIFF-01/README.md`.
Control C = AUXP-01 V4, trained first in each fold; its head scores every auxiliary row (margin vs 50 same-gender faces).

| arm | auxiliary data | parameters |
|---|---|---|
| C / Cd | as AUXP-01 | V4 / V4 + dropout .65 / .35 |
| H10 / H10d | hardest 10 % rows per source out | as C / Cd |
| HP / HPd | persons with > 50 % hard rows out | as C / Cd |
| W / Wd | AVSpeech / LRS3 with whole-clip voices | as C / Cd |

SEL_x = nested choice within each pair. **Rules:** SEL_x - SEL_C >= +0.3 (g cells, >= 4/5 folds) and the default gate -> final head,
zip, one CB run. **One fold per notebook** (`LAB_FOLDS`); report locally. **Inputs:** as FLAG_73. **GPU T4 x2**, Internet OFF.
**Time:** about 3 h per fold.
""")

EXPS["DATAX-01"] = dict(
    nb="FLAG_81_datax01.ipynb", code=["Experiment/DATAX-01/run.py", "Experiment/AUXP-01/run.py", "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py",
                                      "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py",
                                      "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py",
                                      "Experiment/DSEL-01/vox2_groups.json"],
    tag="datax01", vb=True, final=False, folds="1,2",
    intro="""
# FLAG 2027 — NB-81: DATAX-01, data processing of the CC head's training rows

**Why.** The user (14/10): training depends most on the data, so data processing comes first. Reading the code: the MAV rows of the CC
head's main set pair each voice clip with a face of the SAME video, while test positives (and the en_ext / urdu / hindi validation
cells) are cross-video; VoxCeleb2 / VoxBlink rows are already cross-video. Track 3 keeps only the 70 FLAG persons in the main set
(dropping MAV was better for them); our main batches hold only ~12 % v4 identities. DQUAL-01: 2-4 % of auxiliary rows have a face
gender different from the voice gender (VoxBlink 9.4 %). Registered in `Experiment/DATAX-01/README.md`.

| arm | main set | auxiliary heads | recipe |
|---|---|---|---|
| A0 / A0d7 | D4 (v4 + MAV same-video + VoxCeleb2 SA) | vox, vb, avs, lrs3 | AUXP-01 V4 / dropout .70 / .40 |
| XV | D4 with MAV cross-video rows | as A0 | V4 |
| M1 / M1d7 / M1L | v4 only | mavx, vsa, vox, vb, avs, lrs3 | V4 / .70 / .40 / epochs x4 |
| M2 / M2d7 | v4 + MAV cross-video | vsa, vox, vb, avs, lrs3 | V4 / .70 / .40 |
| UP | D4, 50 % v4 identities per main batch | as A0 | V4 |
| GM | D4 | aux rows with mismatched predicted face / voice gender dropped | V4 |

**Rules:** per data change, paired CA-FUSE gain on the g cells: >= +0.3 at >= 4/5 folds HELPS (tuned vs tuned for the main-set arms:
nested choices SEL_A / SEL_M1 / SEL_M2); main-set arms also judged on the in-domain v4 cells (>= +0.3). Candidates -> final head,
zip, one CB run. **Two folds per notebook** (`LAB_FOLDS`, one per GPU); the report runs locally once all five fold tables are in.
**Inputs:** as FLAG_73. **GPU T4 x2**, Internet OFF. **Time:** about 3-3.5 h.
""")

EXPS["DATAX-02"] = dict(
    nb="FLAG_83_datax02.ipynb", code=["Experiment/DATAX-02/run.py", "Experiment/DATAX-01/run.py", "Experiment/AUXP-01/run.py", "Experiment/SCALE-CC-02/run.py",
                                      "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json",
                                      "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py",
                                      "Experiment/DSEL-01/vox2_groups.json"],
    tag="datax02", vb=True, final=False, folds="1,2",
    intro="""
# FLAG 2027 — NB-83: DATAX-02, sampling and cleaning of the CC head's training rows

**Why.** The user (14/10): data processing first. Second batch next to DATAX-01 (pairing / main-set composition). Bangla is an
unheard language and MAV persons speak two languages; auxiliary batches are male-heavy; South-Asian speakers are the most valuable
per identity but dilute the main set; the main set keeps at most 150 MAV clips per person and was never cleaned.
Registered in `Experiment/DATAX-02/README.md`.

| arm | change against A0 (= AUXP-01 V4) |
|---|---|
| LX | main batches: the second row of an identity in ANOTHER language than the first (MAV persons) |
| XVLX | LX on the main set with MAV cross-video rows |
| AGB | every auxiliary batch half male / half female |
| SAUP | VoxBlink head: South-Asian speakers drawn 4x as often |
| MAVALL | MAV rows without the 150-clip cap |
| MAVCL | MAV voices in the lowest 5 % of within-person consistency dropped |

**Rules:** per arm vs A0, paired CA-FUSE gain on the g cells >= +0.3 at >= 4/5 folds HELPS; LX / XVLX also on urdu g + urdu ng +
hindi ng (>= +0.3, >= 4/5 folds). Candidates are combined with DATAX-01's in a later registered run. **Two folds per notebook**
(`LAB_FOLDS`); report locally. **Inputs:** as FLAG_73. **GPU T4 x2**, Internet OFF. **Time:** about 3 h.
""")

EXPS["LOLO-01"] = dict(
    nb="FLAG_85_lolo01.ipynb", code=["Experiment/LOLO-01/run.py", "Experiment/LOLO-01/hu_vox2_flag42.npy", "Experiment/LOLO-01/hu_vb_lid.npy",
                                     "Experiment/DATAX-02/run.py", "Experiment/DATAX-01/run.py", "Experiment/AUXP-01/run.py",
                                     "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json",
                                     "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                     "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="lolo01", vb=True, final=False, folds="1,2",
    intro="""
# FLAG 2027 — NB-85: LOLO-01, an "unheard language" validation for the CC head

**Why.** Our Bangla cells trail both other team systems by 1.4-2.3 EER while our English is the best of the team, and dev English /
Bangla are the same persons: the weakness is the UNHEARD language. SAFE-04's Urdu / Hindi cells are heard languages for the CC head
(other MAV persons' Urdu / Hindi clips are in its main set; Hindi speakers are in the auxiliary sources); its "hold out" regimes only
hold the language out of the fuser. CB 14/10 showed validation can be wrong by 3 EER on the test domain.
Registered in `Experiment/LOLO-01/README.md`.

World U removes EVERY Hindi / Urdu voice clip from the CC head's training data (MAV by `lang`; VoxCeleb2 / VoxBlink by VoxLingua107
LID: top-1 hi / ur or p(hi) + p(ur) >= 0.3; VoxBlink speakers with metadata lan hindi / urdu). Every arm: aux vox, vb, lrs3 (no
AVSpeech: no full LID stored). Recipe AUXP-01 V4, 1 seed.

| arm | world | main set | pair | recipe |
|---|---|---|---|---|
| H0 | heard | D4 | CC | V4 |
| U0 | unheard | D4 (MAV English clips only) | CC | V4 |
| U0d | unheard | as U0 | CC | dropout .70 / .40 |
| U0p | unheard | as U0 | all faces x SimAM pretrained only | V4 |
| U0m1 | unheard | v4 only; MAV + VoxCeleb2 SA as own aux heads | CC | V4 |

**Measure:** CA-FUSE(stream x conditions) fitted without Urdu / Hindi rows, mean EER of urdu g + urdu ng + hindi ng (held-out persons).
**Rules:** unheard penalty U0 - H0 reported; a design >= +0.3 better than U0 at >= 4/5 folds is a candidate for Bangla.
**Two folds per notebook**; report locally. **Inputs:** as FLAG_73. **GPU T4 x2**, Internet OFF. **Time:** about 2 h.
""")

EXPS["FACELORA-03"] = dict(
    nb="FLAG_86_facelora03.ipynb", code=["Experiment/FACELORA-03/run.py", "Experiment/AUXP-01/run.py", "Experiment/SCALE-CC-02/run.py",
                                         "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/SCALE-CC-01/vox2_gender.json",
                                         "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py", "Experiment/T3RECIPE-01/run.py",
                                         "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="facelora03", vb=True, final=False, folds="1,2",
    intro="""
# FLAG 2027 — NB-86: FACELORA-03 step 2, the CC head with the LoRA-adapted CLIP-L (adapters A3 / B3 of FLAG_65)

**Why.** FACELORA-02: a CLIP-L adapter made the CC stream 0.6-1.3 EER better on its own, but the gain did not survive v11's fusion.
FACELORA-03 trains the adapter on ~21k identities (FLAG_65) and scores it inside the CURRENT CC recipe (AUXP-01 V4 on D4; aux VoxCeleb2 +
VoxBlink, the sources with frames). Registered in `Experiment/FACELORA-03/README.md` (amended 14/10 before any result).

| arm | face columns (every row: main + auxiliary + test) |
|---|---|
| B0 | FaRL16 + frozen CLIP-L (control) |
| A3 | FaRL16 + CLIP-L of adapter A3 (InfoNCE) |
| B3 | FaRL16 + CLIP-L of adapter B3 (+ gender-balanced batches, same-gender InfoNCE)  <- decision |

**Rules:** (1) standalone B3 - B0 on en_ext g + urdu g + v4 g; (2) default gate of CA-FUSE(v11 + trB3); (3) gate passes and CA-FUSE
B3 - B0 >= +0.3 at >= 4/5 folds -> final head + zip + one CB run. **Inputs:** as FLAG_73 + the outputs of FLAG_65 a / b (adapted arrays;
attach them, or a private dataset made from them). **GPU T4 x2**, Internet OFF. **Two folds per notebook.**
""")

EXPS["LANGADV-01"] = dict(
    nb="FLAG_88_langadv01.ipynb", code=["Experiment/LANGADV-01/run.py", "Experiment/LANGADV-01/lang_vox2_flag42.npy", "Experiment/LANGADV-01/lang_vb.npy",
                                        "Experiment/LOLO-01/run.py", "Experiment/LOLO-01/hu_vox2_flag42.npy", "Experiment/LOLO-01/hu_vb_lid.npy",
                                        "Experiment/DATAX-02/run.py", "Experiment/DATAX-01/run.py", "Experiment/AUXP-01/run.py",
                                        "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json",
                                        "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                        "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="langadv01", vb=True, final=False, folds="1,2",
    intro="""
# FLAG 2027 — NB-88: LANGADV-01, a language adversary on the CC head's voice embeddings (reopens LANG-02 with the new data)

**Why.** LANG-01 / 02 (27-29/09) were tested on the old bridge head with ~150 MAV persons and 3 languages, measured on HEARD
languages. Now: the CC head with ~45k identities, VoxLingua107 labels for ~154k auxiliary clips (36 languages with >= 300 clips),
and LOLO-01's unheard-language protocol (all Hindi / Urdu voices removed from training). Registered in
`Experiment/LANGADV-01/README.md` (docs/REOPEN_2026-10-15.md, R2).

| arm | world | adversary (GRL) |
|---|---|---|
| H0 / HG | heard | none / lambda .3 |
| U0 / UG1 / UG3 | unheard (no Hindi / Urdu) | none / .1 / .3 |

**Rule:** UG1 or UG3 better than U0 by >= +0.3 on urdu g + urdu ng + hindi ng (fuser without Urdu / Hindi rows) at >= 4/5 folds,
English cells not worse than 0.2 -> candidate (full recipe + gate + one CB run, registered separately). **Two folds per notebook**;
report locally. **Inputs:** as FLAG_73. **GPU T4 x2**, Internet OFF. **Time:** about 1 h.
""")

EXPS["ENC-SCALE-02"] = dict(
    nb="FLAG_91_encscale02.ipynb", code=["Experiment/ENC-SCALE-02/run.py", "Experiment/FACELORA-03/run.py", "Experiment/AUXP-01/run.py",
                                         "Experiment/SCALE-CC-02/run.py", "Experiment/SCALE-CC-01/run.py", "Experiment/SCALE-CC-01/vox2b_groups.json",
                                         "Experiment/SCALE-CC-01/vox2_gender.json", "Experiment/DSEL-03/run.py", "Experiment/DSEL-02/run.py",
                                         "Experiment/T3RECIPE-01/run.py", "Experiment/T3RECIPE-01/t3r.py", "Experiment/DSEL-01/vox2_groups.json"],
    tag="encscale02", vb=True, final=False, folds="1,2",
    intro="""
# FLAG 2027 — NB-91: ENC-SCALE-02, DINOv2-L added to the CC head's face input, screened WITH auxiliary heads (reopens ENC-CC-02)

**Why.** ENC-CC-02 (221) screened DINOv2 in a CC head trained on v4 + MAV only (no auxiliary heads, weak screen: +0.01). The frames
dataset (FLAG_87) made DINOv2 extractable for every VoxCeleb2 / VoxBlink face (FLAG_90), so the screen can now use the real recipe.
Registered in `Experiment/ENC-SCALE-02/README.md` (docs/REOPEN_2026-10-15.md, R5).

| arm | face input | recipe |
|---|---|---|
| FD | FaRL16 + CLIP-L + DINOv2-L (2,304) | AUXP-01 V4, aux VoxCeleb2 + VoxBlink, 3 seeds |
| FDd | as FD | face dropout .70 |
| (B0) | FaRL16 + CLIP-L | FACELORA-03's B0 tables (same rows / seeds) |

**Rule:** SEL_FD - B0 >= +0.5 on the g cells at >= 4/5 folds -> candidate. **Inputs:** as FLAG_73 + `flag2027-feats-screen-p`
(MAV DINOv2) + the FLAG_90 arrays dataset. **GPU T4 x2**, Internet OFF. **Two folds per notebook.**
""")

if __name__ == "__main__":
    import sys
    for exp in (sys.argv[1:] or list(EXPS)):
        build(exp)
