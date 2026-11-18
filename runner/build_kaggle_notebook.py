#!/usr/bin/env python3
"""Build the Kaggle orchestration notebooks of the 13.54 chain, and the code bundle they load.

    python runner/build_kaggle_notebook.py                       # -> notebooks/FourFusion_1354_{infer,train,assemble}.ipynb
    python runner/build_kaggle_notebook.py --bundle <dir>         # -> <dir>/fourfusion_code/ (+ ff_code_manifest.json): upload as a
                                                                  #    private Kaggle dataset and attach it to the notebook

Design (DESIGN 2.4, 3; derived from release/flag2027_e2e/build_notebook.py): the notebook holds no stage code. It finds the
repository (the attached code bundle, or a git clone at a pinned commit), then calls runner/ffrun.py, which runs every stage of the
plan in its own process with a private /kaggle/input and /kaggle/working. So the notebook stays small (Kaggle API limit about
1 MB) and does not change when a stage changes. Every code cell starts with print("[cell] ...", flush=True).

On Kaggle: attach the code bundle and the datasets the plan needs (the original intermediate datasets are found by slug, any
owner), choose FF_MODE in the first cell, run. A session is at most 12 h: stages that do not fit are 'deferred'; attach this
notebook's output (FF_WORK = /kaggle/working/ff_work) to the next session and run again (FF_ONLY splits the work over accounts)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

MODES = {
    "infer": dict(use_saved="all", gpu="GPU (T4) needed: e65xbs / e65xcs / e47rs / e50cc / e163i / t192_fvbx / e163i_fvb run on GPU",
                  what="mode (b): inference from the organisers' Evaluation media + the released checkpoints (about 12 GPU-hours, two Kaggle sessions or more)"),
    "train": dict(use_saved="collected", gpu="GPU (T4, some stages two GPUs) needed",
                  what="mode (a): rebuild everything from the raw and collected data (about 150-170 GPU-hours; split over many sessions with FF_ONLY)"),
    "assemble": dict(use_saved="all", gpu="CPU only",
                     what="mode (c): byte-exact assembly of the submitted zip from the saved per-pair scores (under one minute)"),
}
BUNDLE_DIRS = ["runner", "configs", "stages", "tools", "assembly", "docker", "docs"]
BUNDLE_FILES = ["README.md", "DATA.md", "CHECKPOINTS.md", "LICENSES.md"]
NO_BUNDLE = re.compile(r"(^|/)(_staging_backup|_PRIVATE[^/]*|__pycache__|\.git|ff_work)(/|$)")
DATA_EXT = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".wav", ".mp4", ".zip", ".pkl"}


def nb_json(cells, title):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}, "title": title}, "nbformat": 4, "nbformat_minor": 5}


def md(text, i):
    return {"cell_type": "markdown", "id": f"md{i:02d}", "metadata": {}, "source": text.strip("\n")}


def code(text, i):
    assert text.lstrip().startswith('print("[cell] '), "every new code cell must start with print(\"[cell] ...\", flush=True)"
    return {"cell_type": "code", "id": f"cd{i:02d}", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")}


def build(mode):
    m = MODES[mode]
    cells, k = [], 0

    def add(c):
        nonlocal k
        cells.append(c)
        k += 1

    add(md(f"""
# FourFusion · FLAG 2027 · reproduction of the Evaluation submission 13.54 · FF_MODE={mode}

Team **FourFusion** (CodaBench `minhnhien2811`): Nguyen Chon Nhan, Dang Minh Nhien, Vu Nguyen Anh Dung, Nguyen Huu Bao Loc;
advisor Nguyen Duy Tam Anh; University of Information Technology, VNU-HCM.

Target: the displayed Evaluation submission (CodaBench submission 1054704, EER 13.54 = 10.20 · 11.97 · 12.98 · 18.99), file
`xjf_r10_trsig.zip` (SHA-256 `b4354eaf988b2b4ea8c4e1c3fa44d7bcf631ea5cb3cd635ab2965021fe82e507`). This notebook runs
**{m['what']}**. Accelerator: {m['gpu']}.

| file of the zip | value (stored as −score: lower = same person) |
|---|---|
| no_gender / Bangla | −(logit F1s′ + logit RFX) |
| gender / Bangla | −(logit F1s′ + logit RFX′), RFX′ = RFX with the FVBIND B2 voice block |
| both English files | (FACTOR_ENFUSE English value) / σ[protocol] − 1.0 × z of the DP (Different Profile) branch; σ = 5.0983 (no_gender), 3.8376 (gender) |

How it runs: every original Kaggle run is a **stage** of the repository (`stages/<group>/<id>/stage.json` + `stage.ipy`, original
code cells verbatim with the listed changes). `runner/ffrun.py` plans the stages needed for the target, runs each in its own
process with a private `/kaggle/input` (the inputs the original run saw) and `/kaggle/working`, and keeps what a stage leaves as
an **artifact** under `FF_WORK/art/<original Kaggle id>`. An artifact comes from: this run → a saved copy (attached datasets,
matched by slug; `FF_DATA` folders) → the stage that makes it.

Attach: (1) the code bundle (`python runner/build_kaggle_notebook.py --bundle <dir>`, uploaded as a private dataset) unless
`FF_REPO_GIT` is set; (2) the data the plan lists as missing (cell 03 prints them): original Kaggle datasets are found by slug,
repository ids (`raw/...`, `scores/...`, `pretrained/...`) and the flat `05_scores/` files only under an `FF_DATA` root: with the
default `FF_DATA=auto` every attached folder that holds `00_raw/`, `01_collected/`, ..., `06_pretrained/` (the Drive layout) is used
(cell 02 prints what was found). **Mode (c) needs the folder holding `05_scores/` attached.** (3) For a continued session, the output
of the previous session of this notebook. Secrets: add `HF_TOKEN` as a Kaggle secret if a stage needs it (read, never printed).

**Literal `/kaggle/input` stages.** Some stages run code decoded from embedded payloads that scans the literal `/kaggle/input`
(the runner marks them `[literal /kaggle/input]` in the plan: e50cc, e112f0sx, e112f1sx, the F family, the RFX projector runs,
t189_b2 / t192_fvbx / t163p_fvb). In a Kaggle session that code sees every attachment of the session and none of the artifacts made
earlier in the same session. With `FF_LITERAL_STRICT=1` (default here) the runner therefore refuses such a stage unless every input
is attached at `/kaggle/input` and nothing else is (except this code bundle). Run each of them alone: a new session, `FF_ONLY=<stage>`,
exactly its inputs attached as Kaggle datasets / kernel outputs (plus the previous session's output for inputs made there). Mode (b)
on Kaggle therefore needs separate sessions for e50cc, e112f0sx, e112f1sx and t192_fvbx; on Linux or in a container the runner points
`/kaggle/input` at each stage's own inputs and one session runs the whole plan (`runner/README.md`).
""", k))
    add(code(f'''
print("[cell] 00 config", flush=True)
print("[boot] FourFusion 13.54 orchestrator, FF_MODE={mode}", flush=True)
import os
os.environ.setdefault("FF_MODE", "{mode}")              # infer | train | assemble
os.environ.setdefault("FF_TARGET", "final_verify")      # stage id(s) or artifact id(s)
os.environ.setdefault("FF_USE_SAVED", "{m['use_saved']}")        # all | collected | none
os.environ.setdefault("FF_ONLY", "")                    # e.g. "t153c" to run one stage in this session
os.environ.setdefault("FF_FORCE", "")
os.environ.setdefault("FF_SMOKE", "0")                  # "1" = smoke parameters of every stage (CPU allowed)
os.environ.setdefault("FF_PIP", "original")             # original | pinned | skip
os.environ.setdefault("FF_TIME_BUDGET_H", "11.3")
os.environ.setdefault("FF_WORK", "/kaggle/working/ff_work")
os.environ.setdefault("FF_DATA", "auto")                # auto = every attached folder holding 00_raw/ ... 06_pretrained/ (Drive layout); or explicit roots (os.pathsep)
os.environ.setdefault("FF_LITERAL_STRICT", "1")         # stages whose embedded code scans the literal /kaggle/input refuse extra attachments
FF_REPO_GIT = ""                                        # e.g. "https://github.com/nguyennhan2006/FourFusion_FLAG2027.git" (used only without a code bundle)
FF_REPO_REV = ""                                        # the commit to check out (pin it)
print({{k: os.environ[k] for k in ("FF_MODE", "FF_TARGET", "FF_USE_SAVED", "FF_ONLY", "FF_FORCE", "FF_SMOKE", "FF_PIP", "FF_TIME_BUDGET_H", "FF_WORK", "FF_DATA", "FF_LITERAL_STRICT")}}, flush=True)
''', k))
    add(code('''
print("[cell] 01 repository: attached code bundle or pinned git clone", flush=True)
import hashlib, json, subprocess, sys
from pathlib import Path
REPO = None
for depth in range(1, 7):
    hits = sorted(Path("/kaggle/input").glob("/".join(["*"] * depth) + "/runner/ffcore.py"))
    hits = [h.parents[1] for h in hits if (h.parents[1] / "configs" / "pipeline_1354.json").exists()]
    if hits:
        REPO = hits[0]
        break
if REPO is None and FF_REPO_GIT:
    REPO = Path("/kaggle/working/ff_repo")
    if not (REPO / "runner" / "ffcore.py").exists():
        subprocess.run(["git", "clone", "-q", FF_REPO_GIT, str(REPO)], check=True)
        if FF_REPO_REV:
            subprocess.run(["git", "-C", str(REPO), "checkout", "-q", FF_REPO_REV], check=True)
assert REPO is not None, "attach the code bundle (runner/build_kaggle_notebook.py --bundle) or set FF_REPO_GIT / FF_REPO_REV"
man = REPO / "ff_code_manifest.json"
if man.exists():
    M = json.loads(man.read_text(encoding="utf-8"))
    bad = [f for f, h in M["files"].items() if hashlib.sha256((REPO / f).read_bytes()).hexdigest() != h]
    assert not bad, f"code bundle altered: {bad[:5]}"
    print("code bundle", REPO, "|", len(M["files"]), "files, sha256 ok | built", M.get("built"), flush=True)
else:
    print("repository", REPO, flush=True)
os.environ["FF_REPO"] = str(REPO)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
''', k))
    add(code('''
print("[cell] 02 machine and secrets", flush=True)
import shutil
sys.path.insert(0, str(REPO / "runner"))
import ffcore
got = ffcore.load_kaggle_secrets()
print("python", sys.version.split()[0], "| GPU", ffcore.has_gpu(), "| secrets loaded:", got or "none", flush=True)
roots = ffcore.Config().data_roots                      # FF_DATA=auto: attached Drive-layout folders under /kaggle/input
print("FF_DATA roots:", [str(r) for r in roots] or "none found (mode (c) needs the 05_scores files attached; see the note above)", flush=True)
for r in roots:
    print("   ", r, "->", [d for d in ffcore.DATA_LAYOUT_DIRS if (r / d).is_dir()], flush=True)
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "-L"])
''', k))
    add(code('''
print("[cell] 03 plan", flush=True)
def ffrun(*args):
    p = subprocess.Popen([sys.executable, "-u", str(REPO / "runner" / "ffrun.py"), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=os.environ.copy())
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()
ffrun("plan")
''', k))
    add(code('''
print("[cell] 04 run the plan", flush=True)
rc = ffrun("run")
print("ffrun run exit code", rc, flush=True)
''', k))
    add(code('''
print("[cell] 05 status, final zip, verification", flush=True)
ffrun("status")
work = Path(os.environ["FF_WORK"])
zips = sorted(work.glob("art/notebooks/ff/final_blend/**/xjf_r10_trsig.zip"))
if zips:
    shutil.copy2(zips[0], "/kaggle/working/xjf_r10_trsig.zip")
    ffrun("verify", "zip", "/kaggle/working/xjf_r10_trsig.zip", "--expected", "xjf_r10_trsig")
else:
    print("final zip not produced in this session (see the status above: deferred / blocked stages)", flush=True)
assert rc in (0, 3), f"the run stopped (exit code {rc}); see the log above and FF_WORK/logs/"
''', k))
    return nb_json(cells, f"FourFusion 13.54 ({mode})")


def bundle(dest):
    """Copy the code of the repository (no data, no private folders) + ff_code_manifest.json (sha256 per file)."""
    out = Path(dest) / "fourfusion_code"
    shutil.rmtree(out, ignore_errors=True)
    files = {}
    for d in BUNDLE_DIRS:
        for p in sorted((REPO / d).rglob("*")) if (REPO / d).is_dir() else []:
            rel = p.relative_to(REPO).as_posix()
            if not p.is_file() or NO_BUNDLE.search(rel) or p.suffix.lower() in DATA_EXT or p.stat().st_size > 20 * 2**20:
                continue
            files[rel] = p
    for f in BUNDLE_FILES:
        if (REPO / f).is_file():
            files[f] = REPO / f
    import secret_scan_shim                                           # noqa: E402  (tools/secret_scan.py)
    S = secret_scan_shim.Scanner()
    man = {}
    for rel, p in files.items():
        t = out / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, t)
        man[rel] = hashlib.sha256(t.read_bytes()).hexdigest()
    S.scan_path(out)
    high = [f for f in S.findings if f["severity"] == "HIGH"]
    if high:
        shutil.rmtree(out, ignore_errors=True)
        sys.exit("secret scan found HIGH findings, bundle not written:\n" + "\n".join(f"{f['kind']} {f['where']}" for f in high[:20]))
    import time
    (out / "ff_code_manifest.json").write_text(json.dumps({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "files": man}, indent=1), encoding="utf-8")
    print(f"code bundle -> {out} ({len(man)} files, {sum(p.stat().st_size for p in files.values()) / 1e6:.1f} MB); "
          f"secret scan: {len(S.findings)} non-HIGH finding(s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "notebooks"))
    ap.add_argument("--modes", default="infer,train,assemble")
    ap.add_argument("--bundle", help="write the code bundle into this folder instead of the notebooks")
    a = ap.parse_args(argv)
    if a.bundle:
        sys.path.insert(0, str(REPO / "tools"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("secret_scan_shim", REPO / "tools" / "secret_scan.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["secret_scan_shim"] = mod
        bundle(a.bundle)
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode in a.modes.split(","):
        nb = build(mode)
        p = out / f"FourFusion_1354_{mode}.ipynb"
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        assert p.stat().st_size < 1_000_000
        print(f"wrote {p} ({p.stat().st_size / 1e3:.1f} kB, {len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
