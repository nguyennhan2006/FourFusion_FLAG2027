"""Generate the two END-TO-END Kaggle notebooks: raw organiser data (train + dev, or the Evaluation data) -> submission zip.

    FLAG_16_e2e_indep.ipynb    level 0: every pair scored on its own (SET-17 level-0, CB 26.35; or INDEP-01, CB 26.70)
    FLAG_17_e2e_cluster.ipynb  level 2: per-file statistics + test-time clustering (SET-17, CB 18.59; or SET-13, CB 21.01)

Both notebooks are the same chain, only the pipeline --mode differs:
    FLAG_12 feature cells (verbatim, from build_eval_notebooks)  ->  FLAG_14 AudioCLIP cells (verbatim, SET-17 only)
    ->  kaggle/flag_lib.py + kaggle/flag_v2.py + BEST/v6_eval/pipeline.py written to the working dir from their verbatim
        source (the three library cells)  ->  pipeline.main()  ->  submission zips (+ rank-corr with the reference zips)
Inputs on Kaggle: the raw v4 dataset (train_set + dev_set), `flag2027-members` (kaggle/flag2027-members.zip: the 45 trained
members + reference zips), optional `flag2027-feats-eval` / `flag2027-feats-av` or the other notebook's output (features
already extracted are skipped), optional `flag2027-feats-v2` (only needed to TRAIN members: r2mix needs the voice crops).
    python build_e2e_notebooks.py
    FLAG_LOCAL=1 FLAG_IN=<dir: raw data + feats_eval + feats_av + members> FLAG_OUT=<dir> E2E_N_MODELS=5 \
        python run_notebook_local.py FLAG_16_e2e_indep.ipynb --jupyter
"""
from pathlib import Path
import build_eval_notebooks as B12
import build_av_notebook as B14

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
nb = B12.nb


def intro(mode):
    lvl0 = mode == "indep"
    return f"""
# FLAG 2027 — NB-{16 if lvl0 else 17}: end to end, raw data -> submission ({'level 0: every pair on its own' if lvl0 else 'level 2: per-file statistics + test-time clustering'})

One notebook from the organiser's raw media (train + dev today, the Evaluation data on 11/11) to the zip we submit.

| `SYSTEM` (edit in the first cell) | this notebook produces | CodaBench dev |
|---|---|---|
| `set17` (default) | {'SET-17 scored at level 0: INDEP recipe of every member, ImageBind + age (+ 0.25 AudioCLIP) on every member of the English cells, constants from 20 000 train pairs' if lvl0 else 'SET-17: file-centred z-scores, ImageBind + age (+ 0.25 AudioCLIP) on every member of the English cells, ArcFace / ECAPA clusters, block mean b = 0.99, rank fusion'} | **{'26.35' if lvl0 else '18.59'}** |
| `set13` | {'INDEP-01: SET-13 scored at level 0' if lvl0 else 'SET-13: as above without --ib-all / AudioCLIP'} | **{'26.70' if lvl0 else '21.01'}** |

{'A pair (face, voice) gets a score that depends only on that image, that audio and constants measured on the train set: no statistic of the test file, no clusters, no trial list. The notebook re-scores half of a file and checks that no score moved (invariance self-test).' if lvl0 else 'Every trial is scored from the full face x voice matrix of its test file: per-file centring and z-scores (level 1), then faces and voices are clustered inside the file without labels and the matrix is averaged per (face cluster, voice cluster) block (level 2). The trial list is never read. `decisions.csv` reports the cluster diagnostics of every file; the pipeline also writes the level-0 and no-cluster zips of the same members for comparison.'}

**Steps (each cell says which notebook it comes from; the code is verbatim):**
1. **FLAG_12 cells**: ArcFace, ECAPA-192 / 6144, ViT-age, wav2vec2 age-gender, ImageBind vision / audio for every organiser list found (train + every test file). Arrays that already exist in an attached earlier output are copied and skipped.
2. **FLAG_14 cells** (`set17` only): AudioCLIP face / audio embeddings.
3. **Library cells**: the verbatim source of `kaggle/flag_lib.py`, `kaggle/flag_v2.py`, `BEST/v6_eval/pipeline.py` (written to the working dir, then imported).
4. **Pipeline**: `pipeline.main()` with `--mode {mode}`{' --ib-all English --aclip 0.25' if True else ''} (for `set17`). Members (s007 / s010B / r2mix, 15 seeds each) are loaded from the dataset `flag2027-members`; without it they are trained here (needs `flag2027-feats-v2` for the r2mix voice crops).
5. **Result**: `/kaggle/working/submission/` with `submission.zip` (= the mode of this notebook), `submission_set13.zip`, `submission_noclus.zip`, `submission_indep.zip`, `decisions.csv`, `invariance.json`; rank correlation with the reference zips of the members dataset (dev only: must be 1.0000).

**Inputs (Add Input):** (1) the raw organiser data (`train_set` + `dev_set` zips or folders; on 11/11 also the Evaluation data, then set `ONLY_UNDER`); (2) **`flag2027-members`**; (3) optional: `flag2027-feats-eval`, `flag2027-feats-av` or the output of the other e2e notebook (skips 1-2 h of extraction); (4) optional: `flag2027-feats-v2` (only to train members).
**Settings:** GPU T4, **Internet ON** (checkpoints ~7 GB + AudioCLIP). **Time on a T4:** extraction about 1 h 15 min (+ about 20 min AudioCLIP), pipeline about 30-60 min on Kaggle's CPU with 15 members. With features attached: only the pipeline.
**Submit:** `submission.zip`. Scores are written as -score (lower = same), the polarity CodaBench measured (docs/DATA.md). On new data, if the first result is > 50, flip the sign.
"""


ENV = r'''
import os, sys, io, re, time, json, types, shutil, zipfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)
LOCAL = os.environ.get("FLAG_LOCAL") == "1"          # CPU smoke test on a local copy of the data
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORKDIR = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
CACHE = Path(os.environ.get("FLAG_CACHE", "/tmp/flag_cache"))           # checkpoints stay out of the output
ECAPA_DIR = Path(os.environ.get("FLAG_ECAPA", str(CACHE / "ecapa_sb")))
REPOS = Path(os.environ.get("FLAG_REPOS", "/tmp/repos"))
OUT = WORKDIR / "feats_eval"                                            # FLAG_12 arrays
OUT_AV = WORKDIR / "feats_av"                                           # FLAG_14 arrays (AudioCLIP)
ORG = WORKDIR / "orgdata"                                               # organiser csv + lists, one folder for the pipeline
SUB = WORKDIR / "submission"
for d in (WORKDIR, CACHE, OUT, OUT_AV, ORG, SUB, REPOS):
    d.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)

# ------------------------------------------------------------------ what to run (edit here)
SYSTEM = os.environ.get("E2E_SYSTEM", "set17")        # set17: SET-17 (--ib-all English --aclip 0.25) | set13: SET-13 / INDEP-01
MODE = "__MODE__"                                     # this notebook's pipeline mode: indep (level 0) | auto (clusters)
N_MODELS = int(os.environ.get("E2E_N_MODELS", "15"))  # members per component; 5 = exactly the submitted members
ENCODERS = ["arcface", "ecapa", "vitage", "w2v2ag", "ibv", "iba"]   # FLAG_12; ibv right before iba: one ImageBind load
ENCODERS_AV = ["aclip"] if SYSTEM == "set17" else []                # FLAG_14
EXT_SOURCES, EXT_ENCODERS = [], []                                  # no MAV-Celeb here
SPLIT_FILTER = None            # e.g. ["no_gender/English"]; None = every split found (train + every test list)
ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")             # organiser zips unpacked when not extracted
ONLY_UNDER = None              # e.g. "eval": keep only lists whose path contains this (dev and eval both attached)
assert MODE in ("indep", "auto") and SYSTEM in ("set17", "set13")
print("LOCAL" if LOCAL else "KAGGLE", "| system", SYSTEM, "| mode", MODE, "| n_models", N_MODELS, "| input", IN, "| output", WORKDIR)
'''

PIP = r'''
# speechbrain -U: Kaggle's preinstalled one breaks on recent torchaudio. insightface / ImageBind --no-deps: their
# declared deps would replace onnxruntime-gpu by the CPU build and pin an old torch. ImageBind's data loaders need
# pytorchvideo, so the two loaders used here are reimplemented below and its module is stubbed. AudioCLIP is a git clone.
if not LOCAL:
    !pip -q install -U "speechbrain>=1.1.1" onnxruntime-gpu onnx iopath ftfy omegaconf fvcore einops 2>&1 | tail -2
    !pip -q install --no-deps "insightface==2.0" "git+https://github.com/facebookresearch/ImageBind" 2>&1 | tail -2
    if ENCODERS_AV:
        !git clone -q --depth 1 https://github.com/AndreyGuzhov/AudioCLIP.git /tmp/repos/AudioCLIP
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEV,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

ORGDATA = r'''
# ============================================================================ the organiser csv + lists in ONE folder (what flag_lib reads)
# train: train/{train_English_faces.csv, train_English_voices.csv, train_English.txt, meta_file_train_set.csv}
# test files: dev/<folder>_<name>_{faces.csv, voices.csv, test.txt}   (flag_lib layout A; layout B of the zips also works)
def _pick(name, must=None):
    hits = [p for p in SCAN["files"].get(name, [])] + [p for p in scan_raw["files"].get(name, [])]
    hits = [p for p in hits if must is None or must(p)]
    assert hits, f"{name} not found in the inputs"
    return sorted(hits, key=lambda p: len(p.as_posix()))[0]


scan_raw = scan_inputs(WORKDIR / "raw") if (WORKDIR / "raw").exists() else {"files": {}, "roots": []}
(ORG / "train").mkdir(exist_ok=True); (ORG / "dev").mkdir(exist_ok=True)
for n in ("train_English_faces.csv", "train_English_voices.csv", "train_English.txt", "meta_file_train_set.csv"):
    shutil.copy(_pick(n), ORG / "train" / n)
for k, v in S.items():
    if k == "train":
        continue
    folder, name = k.split("/")
    feats = v["txt"].parent / "features"
    for stem in ("faces.csv", "voices.csv"):
        src = feats / f"{name}_test_{stem}"
        if not src.exists():
            src = _pick(f"{name}_test_{stem}", must=lambda p, f=folder: f"/{f}/" in p.as_posix())
        shutil.copy(src, ORG / "dev" / f"{folder}_{name}_{stem}")
    shutil.copy(v["txt"], ORG / "dev" / f"{folder}_{name}_test.txt")
log("organiser data for the pipeline:", sorted(p.name for p in (ORG / "dev").glob("*")), "+ train")
'''

AV = r'''
# ============================================================================ FLAG_14: AudioCLIP arrays (SET-17 only)
# resume: aclip arrays of an earlier run (a folder named like flag2027-feats-av / feats_av) are copied
for n, ps in SCAN["files"].items():
    if n.endswith(".npy") and "_aclip_" in n and not (OUT_AV / n).exists():
        for p in ps:
            if re.search(r"feats[-_]av", p.as_posix()) and OUT_AV not in p.parents:
                shutil.copy(p, OUT_AV / n); break
_OUT12, OUT = OUT, OUT_AV                              # the FLAG_14 run cell writes to OUT
ENCODERS_12, ENCODERS = ENCODERS, ENCODERS_AV
'''

AV_END = r'''
OUT, ENCODERS = _OUT12, ENCODERS_12
'''


def lib_cell(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    q = chr(39) * 3
    assert q not in src, rel
    name = Path(rel).name
    var = "SRC_" + name.split(".")[0].upper()
    nl = chr(10)
    return (nl + "# ============================================================================ " + rel
            + " (verbatim), written to the working dir and imported" + nl
            + var + " = r" + q + src + q + nl
            + "Path(" + repr(name) + ").write_text(" + var + ", encoding=" + repr("utf-8") + ")" + nl
            + "print(" + repr(name + ":") + ", len(" + var + ".splitlines()), " + repr("lines") + ")" + nl)


RUN = r'''
# ============================================================================ the pipeline (BEST/v6_eval/pipeline.py, unchanged)
sys.path.insert(0, str(WORKDIR))
for m in ("flag_lib", "flag_v2", "pipeline"):
    sys.modules.pop(m, None)
os.environ["FLAG_DATA"] = str(ORG)
import pipeline

feats = [str(OUT)] + ([str(OUT_AV)] if ENCODERS_AV else [])
crops = next((p.parent for p in IN.rglob("voice_ecapa192crop_train.npy")), None)      # only needed to TRAIN r2mix
if crops is not None:
    feats.append(str(crops))
members = next((p.parent for p in IN.rglob("s007_seed1.pt")), None)
if members is None:
    members = WORKDIR / "members"
    log("no flag2027-members dataset attached: the members will be TRAINED here (r2mix needs flag2027-feats-v2 for the crops)")
refs = []
for name in ("indep", "set13"):
    p = next((p for p in IN.rglob(f"reference/{SYSTEM}/ref_{name}.zip")), None)
    if p is not None:
        refs.append(f"{name}={p}")
argv = ["pipeline.py", "--data", str(ORG), "--feats", *feats, "--out", str(SUB), "--mode", MODE, "--n-models", str(N_MODELS),
        "--members-from", str(members), "--threads", str(os.cpu_count() or 4)]
if SYSTEM == "set17":
    argv += ["--ib-all", "English", "--aclip", "0.25"]
if refs:
    argv += ["--reference", *refs]
print(" ".join(argv))
sys.argv = argv
t0 = time.time()
pipeline.main()
log(f"pipeline done in {(time.time() - t0) / 60:.1f} min")
'''

DONE = r'''
# ============================================================================ result
D = pd.read_csv(SUB / "decisions.csv"); print(D.to_string(index=False))
print(json.dumps(json.loads((SUB / "invariance.json").read_text()), indent=1))
final = WORKDIR / f"submission_{SYSTEM}_{MODE}.zip"
shutil.copy(SUB / "submission.zip", final)
with zipfile.ZipFile(final) as z:
    for n in z.namelist():
        d = pd.read_csv(z.open(n), sep=" ", header=None)
        print(f"{n:45s} {len(d):5d} rows, score mean {d[1].mean():+.3f}")
shutil.rmtree(WORKDIR / "raw", ignore_errors=True)
print(f"\nSUBMIT: {final}   (= {SUB / 'submission.zip'}; the other zips in {SUB} are the same members scored the other ways)")
'''


def build(mode, path):
    env = ENV.replace("__MODE__", mode)
    cells = [("markdown", intro(mode)), ("code", env), ("code", PIP),
             ("markdown", "## 1. FLAG_12: find the data (row order = the organiser .txt), resume from an earlier output"), ("code", B12.DATA12),
             ("markdown", "## 2. Organiser csv + lists for the pipeline"), ("code", ORGDATA),
             ("markdown", "## 3. FLAG_12: audio loading, batching, helpers"), ("code", B12.AUDIO12),
             ("markdown", "## 4. FLAG_12: encoders"), ("code", B12.ENC12),
             ("markdown", "## 5. FLAG_12: extract every array the pipeline reads (skipped where it exists)"), ("code", B12.RUN12),
             ("markdown", "## 6. FLAG_12: check against attached reference arrays (optional)"), ("code", B12.CHECK12),
             ("markdown", "## 7. FLAG_14: AudioCLIP (SET-17 only; the cell is a no-op for `set13`)"),
             ("code", AV),
             ("code", B14.AUDIO[len(B12.AUDIO12):]),                       # load_audio_sr only (AUDIO12 already ran)
             ("code", B14.ENC), ("code", B14.RUN), ("code", AV_END),
             ("markdown", "## 8. Libraries, verbatim: kaggle/flag_lib.py, kaggle/flag_v2.py, BEST/v6_eval/pipeline.py"),
             ("code", lib_cell("kaggle/flag_lib.py")), ("code", lib_cell("kaggle/flag_v2.py")), ("code", lib_cell("BEST/v6_eval/pipeline.py")),
             ("markdown", f"## 9. Run the pipeline: `--mode {mode}`"), ("code", RUN),
             ("markdown", "## 10. Result"), ("code", DONE)]
    nb(cells, path)


if __name__ == "__main__":
    build("indep", "FLAG_16_e2e_indep.ipynb")
    build("auto", "FLAG_17_e2e_cluster.ipynb")
