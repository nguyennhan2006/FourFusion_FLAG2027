#!/usr/bin/env python3
"""FourFusion FLAG 2027: run the Evaluation path of U' = F1s' + RFS in Docker, one container per Kaggle notebook.

The host needs Python 3.8+ (standard library only) and Docker (GPU stages: NVIDIA Container Toolkit). Paths below are host
paths. Every stage mounts its inputs read-only at /kaggle/input/<slug> (the Kaggle layout the notebooks search with rglob)
and writes /kaggle/working = <out>/<split>/<stage>/. See docker/README.md.

    python docker/ff.py plan    [--split dev|eval]
    python docker/ff.py fetch   --data D [--stages ...] [--account A]       # team only: Kaggle datasets -> D/<owner>__<slug>
    python docker/ff.py kaggle-output STAGE KERNEL --out O [--split S]       # team only: a finished Kaggle run as a stage output
    python docker/ff.py pull-exact                                           # team only: the exact notebooks of the R11 runs
    python docker/ff.py fetch-models --models M                              # optional: ImageBind-huge -> M (SHA-256 checked)
    python docker/ff.py run     STAGE...|all --data D --out O [--split dev|eval] [--evalds DIR] [--only-under eval]
                                [--var 'NAMES={...}'] [--cache C] [--models M] [--exact] [--offline] [--no-gpu] [--image IMG] [--dry]
    python docker/ff.py make-u  --out O [--split S] [--counts 'name=n,...'] [--ref U_DEV_ZIP]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

DOCKER = Path(__file__).resolve().parent
PKG = DOCKER.parent                      # the release package root (kaggle/, Experiment/, docker/)
SPEC = json.loads((DOCKER / "stages.json").read_text(encoding="utf-8"))
STAGES, ORDER = SPEC["stages"], SPEC["order"]
EXACT = DOCKER / "exact"                 # pull-exact writes here; not part of the public package (see README)
NOT_SHIPPED = {"kaggle/FLAG_112_f0x.ipynb": "python kaggle/build_flag112.py f0x Experiment/CLEANTUNE-01/ct_decision.json",
               "kaggle/FLAG_112_f1x.ipynb": "python kaggle/build_flag112.py f1x Experiment/CLEANTUNE-01/ct_decision.json"}

# ImageBind-huge (12/11/2026): Meta's URL (dl.fbaipublicfiles.com) stopped answering. Every notebook is rewritten at run time to a
# Hugging Face copy pinned to a commit whose Git-LFS object has the SHA-256 of Meta's file (the notebook 163i checks it again).
# fetch-models downloads the same file once; run --models mounts it so no stage downloads ImageBind.
IB_META = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
IB_URL = ("https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/"
          "imagebind/imagebind_huge.pth")
IB_SHA = "d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157"
IB_FILE = "imagebind_huge.pth"


def die(msg):
    sys.exit(f"ff.py: {msg}")


def dsdir(data, ref):
    return Path(data) / ref.replace("/", "__")


def stage_out(out, split, sid):
    return Path(out) / split / sid


def done_flag(out, split, sid):
    return Path(out) / split / f"{sid}.DONE"


def pick(names):
    if names == ["all"]:
        return list(ORDER)
    bad = [n for n in names if n not in STAGES]
    if bad:
        die(f"unknown stage(s) {bad}; stages: {ORDER}")
    return names


# ---------------------------------------------------------------- notebook preparation (what kaggle/sat_push.sh + kapi did)
def env_of(spec, split, extra):
    env = {k: v.format(split=split) for k, v in spec.get("env", {}).items()}
    if split == "dev":
        env.update(spec.get("env_dev", {}))
    env.update(extra)
    return env


def prepare(sid, split, a):
    spec = STAGES[sid]
    if a.exact:
        src = EXACT / f"{sid}.ipynb"
        if not src.exists():
            die(f"{src} missing: run `python docker/ff.py pull-exact` first (team only), or drop --exact")
    else:
        src = PKG / spec["notebook"]
        if not src.exists():
            hint = NOT_SHIPPED.get(spec["notebook"], "see README section 'Notebooks not shipped'")
            die(f"{spec['notebook']} missing. Rebuild it from the repository root: {hint}")
    nb = json.loads(src.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    txt = ["".join(c["source"]) for c in code]
    if not a.exact:                                       # the sed edits of sat_push.sh (an exact notebook already has them)
        for old, new in spec.get("edits", []):
            n = sum(t.count(old) for t in txt)
            if n == 0:
                die(f"{sid}: edit {old!r} -> {new!r} matched nothing in {src.name}")
            txt = [t.replace(old, new) for t in txt]
    variables = dict(kv.split("=", 1) for kv in a.var)
    if a.models:
        variables.setdefault("IB_CKPT", repr(f"/kaggle/models/{IB_FILE}"))   # 163i: `IB_CKPT = None` = download
    if a.only_under and spec.get("test_data"):
        variables["ONLY_UNDER"] = repr(a.only_under)
    for name, value in variables.items():                  # `NAME = None` -> `NAME = value` (sat_push mk(), sổ tay 11/11)
        pat = re.compile(rf"(?m)^(\s*){re.escape(name)} = None(?=\s|;|$)")   # `;`: 163i's one-line `IB_CKPT = None; R221_CKPT = None`
        hits = sum(len(pat.findall(t)) for t in txt)
        if hits:
            txt = [pat.sub(lambda m: f"{m.group(1)}{name} = {value}", t) for t in txt]
            print(f"  {sid}: {name} = {value}")
        elif name != "ONLY_UNDER":
            print(f"  {sid}: no `{name} = None` line (not used by this notebook)")
    n_ib = sum(t.count(IB_META) for t in txt)              # exact notebooks still hold Meta's (dead) URL; same file, see IB_URL
    if n_ib:
        txt = [t.replace(IB_META, IB_URL) for t in txt]
        print(f"  {sid}: ImageBind URL -> Hugging Face copy ({n_ib}x)")
    for c, t in zip(code, txt):
        c["source"] = t.splitlines(keepends=True)
    env = env_of(spec, split, dict(kv.split("=", 1) for kv in a.env))
    params = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": [f'print("[cell] 00 run parameters (docker/ff.py, split={split})", flush=True)\n', "import os\n",
                         f"os.environ.update({json.dumps(env, sort_keys=True)})\n"]}
    if a.models:                                          # pre-downloaded ImageBind (--models): FLAG_CACHE holders find it there
        params["source"] += [f"_m = '/kaggle/models/{IB_FILE}'\n", "if os.path.exists(_m):\n",
                             "    _c = os.environ.get('FLAG_CACHE', '/tmp/flag_cache'); os.makedirs(_c, exist_ok=True)\n",
                             f"    _t = os.path.join(_c, '{IB_FILE}')\n",
                             "    if not os.path.exists(_t): os.symlink(_m, _t)\n",
                             "    print('ImageBind from --models:', _m, '->', _t, flush=True)\n"]
    if any("id" in c for c in nb["cells"]):
        params["id"] = "ff-params"
    first = "".join(nb["cells"][0]["source"]) if nb["cells"] else ""
    if "(kapi --env)" in first or "(kapi --smoke)" in first:
        nb["cells"][0] = params                           # replaces the Kaggle run's parameter cell
    else:
        nb["cells"].insert(0, params)
    dst = Path(a.out) / "_notebooks" / f"{sid}.{split}.ipynb"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return dst, env


def bind(src, dst, ro=True):
    return ["--mount", f"type=bind,source={Path(src).resolve()},target={dst}" + (",readonly" if ro else "")]


def run_stage(sid, split, a):
    spec = STAGES[sid]
    print(f"== {sid} ({spec['step']}): {spec['what']}")
    nbp, env = prepare(sid, split, a)
    out = stage_out(a.out, split, sid)
    img = a.image or SPEC["images"][spec["image"]]
    cmd = ["docker", "run", "--rm", "--name", f"ff-{split}-{sid}", "--shm-size", a.shm]
    if spec["image"] == "gpu" and not a.no_gpu:
        cmd += ["--gpus", "all"]
    if a.offline or not spec.get("internet", True):       # CPU stages ran without Internet on Kaggle as well
        cmd += ["--network", "none", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"]
    missing = []
    cmd += ["-e", "PYDEVD_DISABLE_FILE_VALIDATION=1"]
    mounts = bind(nbp, f"/kaggle/src/{sid}.ipynb") + bind(out, "/kaggle/out", ro=False)   # run_nb.py copies /kaggle/working here
    for ref in spec["datasets"]:
        d = dsdir(a.data, ref)
        missing += [] if d.is_dir() else [str(d)]
        mounts += bind(d, f"/kaggle/input/{ref.split('/')[1]}")
    if split == "eval" and spec.get("test_data"):
        if not a.evalds:
            die(f"{sid}: --evalds <folder with the organisers' Evaluation data> is required for split eval")
        mounts += bind(a.evalds, f"/kaggle/input/{Path(a.evalds).resolve().name}")
    for up in spec["after"]:
        if not done_flag(a.out, split, up).exists():
            missing.append(f"{done_flag(a.out, split, up)} (run stage {up} first, or `ff.py kaggle-output {up} ...`)")
        mounts += bind(stage_out(a.out, split, up), f"/kaggle/input/{STAGES[up]['slug'].format(split=split)}")
    if a.cache:
        Path(a.cache).mkdir(parents=True, exist_ok=True)
        mounts += bind(a.cache, "/root/.cache", ro=False)
    if a.models:
        f = Path(a.models) / IB_FILE
        if not f.is_file():
            missing.append(f"{f} (python docker/ff.py fetch-models --models {a.models})")
        mounts += bind(a.models, "/kaggle/models")
    cmd += mounts + [img, "python", "/opt/fourfusion/docker/run_nb.py", f"/kaggle/src/{sid}.ipynb", sid]
    print("  env:", env)
    if a.dry:
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if missing:
            print("  missing:", *missing, sep="\n    ")
        return True
    if missing:
        die(f"{sid}: missing inputs:\n  " + "\n  ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    done_flag(a.out, split, sid).unlink(missing_ok=True)
    log = Path(a.out) / split / f"{sid}.log"
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line); f.write(line); f.flush()
        rc = p.wait()
    if rc != 0:
        die(f"{sid} failed (exit {rc}); log {log}")
    done_flag(a.out, split, sid).write_text(f"image {img}\nnotebook {nbp}\n", encoding="utf-8")
    if spec.get("result"):
        hits = list(out.rglob(spec["result"].format(split=split)))
        print(f"  result: {hits[0] if hits else 'NOT FOUND: ' + spec['result']}")
    return True


# ---------------------------------------------------------------- team-only helpers (Kaggle API through kaggle/kapi.py)
def kapi():
    sys.path.insert(0, str(PKG / "kaggle"))
    import kapi as k                                       # credentials: an env file outside the package (kapi.py docstring)
    return k


def cmd_fetch(a):
    k = kapi(); accs = k.accounts()
    refs = sorted({r for s in pick(a.stages) for r in STAGES[s]["datasets"]})
    for ref in refs:
        d = dsdir(a.data, ref)
        if (d / ".complete").exists():
            print("have", ref); continue
        acc = a.account or (ref.split("/")[0] if ref.split("/")[0] in accs else None)
        if not acc:
            die(f"{ref}: no account of the env file owns it; pass --account <reader>")
        d.mkdir(parents=True, exist_ok=True)
        print(f"get {ref} as {acc} -> {d}", flush=True)
        rc, msg = k.cli(acc, ["datasets", "download", ref, "-p", str(d), "--unzip", "-q"], accs, timeout=8 * 3600)
        if rc != 0 and "404" in msg:                       # some datasets with a nested tree have no archive: get each file
            rc, msg = fetch_files(k, accs, acc, ref, d)
        if rc != 0:
            die(f"{ref}: {msg.strip()[-500:]}")
        (d / ".complete").write_text("ok\n")


def fetch_files(k, accs, acc, ref, d):
    names, token = [], None
    while True:
        rc, out = k.cli(acc, ["datasets", "files", ref, "--csv", "--page-size", "200"] + (["--page-token", token] if token else []), accs)
        if rc != 0:
            return rc, out
        m = re.search(r"Next Page Token = (\S+)", out)
        names += [l.split(",")[0] for l in out.splitlines()
                  if "," in l and not l.startswith(("name,", "Warning", "Next Page")) and not l.split(",")[0].endswith("/")]
        if not m:
            break
        token = m.group(1)
    print(f"  {ref}: no archive, {len(names)} files one by one", flush=True)
    for n in names:
        sub = d / Path(n).parent; sub.mkdir(parents=True, exist_ok=True)
        rc, out = k.cli(acc, ["datasets", "download", ref, "-f", n, "-p", str(sub), "--unzip", "-q"], accs, timeout=3600)
        if rc != 0:
            return rc, f"{n}: {out}"
        z = sub / (Path(n).name + ".zip")                  # a single file may come zipped
        if z.exists() and not (sub / Path(n).name).exists():
            import zipfile
            zipfile.ZipFile(z).extractall(sub); z.unlink()
    return 0, "ok"


def cmd_kaggle_output(a):
    k = kapi()
    out = stage_out(a.out, a.split, a.stage); out.mkdir(parents=True, exist_ok=True)
    rc, msg = k.cli(a.kernel.split("/")[0], ["kernels", "output", a.kernel, "-p", str(out), "-o"], timeout=4 * 3600)
    if rc != 0:
        die(msg.strip()[-500:])
    done_flag(a.out, a.split, a.stage).write_text(f"from Kaggle {a.kernel}\n", encoding="utf-8")
    print(f"{a.kernel} -> {out}")


def cmd_pull_exact(a):
    import tempfile
    k = kapi(); EXACT.mkdir(exist_ok=True)
    for sid in ORDER:
        ref = STAGES[sid]["pulled_from"]; d = Path(tempfile.mkdtemp())
        rc, msg = k.cli(ref.split("/")[0], ["kernels", "pull", ref, "-p", str(d)])
        nbs = list(d.glob("*.ipynb"))
        if rc != 0 or not nbs:
            die(f"{ref}: {msg.strip()[-300:]}")
        shutil.copy2(nbs[0], EXACT / f"{sid}.ipynb"); print(f"{ref} -> {EXACT / (sid + '.ipynb')}")


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_fetch_models(a):
    """ImageBind-huge (4.8 GB) -> <models>/imagebind_huge.pth from IB_URL (or --url), kept only if its SHA-256 is Meta's."""
    import urllib.request
    d = Path(a.models); d.mkdir(parents=True, exist_ok=True); t = d / IB_FILE
    if t.is_file() and sha256(t) == IB_SHA:
        print(f"{t}: present, SHA-256 ok"); return
    part = t.with_suffix(".part"); url = a.url or IB_URL
    print(f"download {url}\n      -> {t}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        n, total = 0, int(r.headers.get("Content-Length") or 0)
        for b in iter(lambda: r.read(1 << 22), b""):
            f.write(b); n += len(b)
            if n % (1 << 30) < (1 << 22):
                print(f"  {n / 2**30:.1f} / {total / 2**30:.1f} GB", flush=True)
    got = sha256(part)
    if got != IB_SHA:
        part.unlink(missing_ok=True); die(f"SHA-256 {got} != {IB_SHA}: file discarded")
    part.replace(t); print(f"{t}: SHA-256 ok ({IB_SHA[:16]}...)")


# ---------------------------------------------------------------- U = F1s + RFS (EVALPATH-U checks)
def cmd_make_u(a):
    sp = a.split
    f1 = list(stage_out(a.out, sp, "112f1sx").rglob(STAGES["112f1sx"]["result"].format(split=sp)))
    rf = list(stage_out(a.out, sp, "163i").rglob(STAGES["163i"]["result"]))
    if not f1 or not rf:
        die(f"need the F1s zip ({len(f1)} found) and the RFS zip ({len(rf)} found) under {Path(a.out) / sp}")
    root = Path(a.out).resolve(); rel = lambda p: "/out/" + p.resolve().relative_to(root).as_posix()
    (root / sp / "U").mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm"] + bind(root, "/out", ro=False)
    if a.ref:
        cmd += bind(a.ref, "/ref/" + Path(a.ref).name)
    cmd += [a.image or SPEC["images"]["cpu"], "bash", "/opt/fourfusion/Experiment/EVALPATH-U/make_eval_zip.sh",
            rel(f1[0]), rel(rf[0]), f"/out/{sp}/U/submission_U_{sp}.zip", sp]
    cmd += (["--counts", a.counts] if a.counts else []) + (["--ref", "/ref/" + Path(a.ref).name] if a.ref else [])
    print(" ".join(cmd)); sys.exit(subprocess.call(cmd))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--split", default="dev", choices=["dev", "eval"])
    p = sub.add_parser("fetch"); p.add_argument("--data", required=True); p.add_argument("--stages", nargs="+", default=["all"])
    p.add_argument("--account")
    p = sub.add_parser("kaggle-output"); p.add_argument("stage", choices=ORDER); p.add_argument("kernel")
    p.add_argument("--out", required=True); p.add_argument("--split", default="dev", choices=["dev", "eval"])
    sub.add_parser("pull-exact")
    p = sub.add_parser("fetch-models"); p.add_argument("--models", required=True); p.add_argument("--url", help="another copy (SHA-256 still checked)")
    p = sub.add_parser("run"); p.add_argument("stages", nargs="+")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--split", default="dev", choices=["dev", "eval"]); p.add_argument("--evalds")
    p.add_argument("--only-under"); p.add_argument("--var", action="append", default=[], help="NAME=python-literal for a `NAME = None` line")
    p.add_argument("--env", action="append", default=[], help="K=V added to the stage environment")
    p.add_argument("--cache", help="host folder mounted at /root/.cache (HF / torch hub weights)")
    p.add_argument("--exact", action="store_true", help="use docker/exact/<stage>.ipynb (the notebook source of the R11 run)")
    p.add_argument("--models", help="host folder with imagebind_huge.pth (fetch-models), mounted read-only at /kaggle/models")
    p.add_argument("--offline", action="store_true"); p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--image"); p.add_argument("--shm", default="16g"); p.add_argument("--dry", action="store_true")
    p = sub.add_parser("make-u"); p.add_argument("--out", required=True); p.add_argument("--split", default="dev", choices=["dev", "eval"])
    p.add_argument("--counts"); p.add_argument("--ref"); p.add_argument("--image")
    a = ap.parse_args()
    if a.cmd == "plan":
        for sid in ORDER:
            s = STAGES[sid]
            print(f"{s['step']:>4} {sid:8} {s['image']:3} after={','.join(s['after']) or '-':28} {s['what']}")
        print(f"   U  make-u   cpu  F1s ({STAGES['112f1sx']['result'].format(split=a.split)}) + RFS ({STAGES['163i']['result']})")
    elif a.cmd == "fetch":
        cmd_fetch(a)
    elif a.cmd == "kaggle-output":
        cmd_kaggle_output(a)
    elif a.cmd == "pull-exact":
        cmd_pull_exact(a)
    elif a.cmd == "fetch-models":
        cmd_fetch_models(a)
    elif a.cmd == "run":
        for sid in pick(a.stages):
            run_stage(sid, a.split, a)
    elif a.cmd == "make-u":
        cmd_make_u(a)


if __name__ == "__main__":
    main()
