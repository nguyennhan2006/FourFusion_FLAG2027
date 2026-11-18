r"""Lean reproduction package of the final Evaluation system FACTOR_ENFUSE (c5rl) -> FF_PKG_DEST (default E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl).

    python release/factor_enfuse/make_package.py all                 # resumable: finished items are kept in <DEST>.state/
    python release/factor_enfuse/make_package.py submission | notebooks | datasets | pretrained | rebuild | manifest

Only what this system uses: the Kaggle notebooks that ran (source + log), the outputs of the Evaluation-path runs, the Kaggle datasets
those runs attached (trained weights / inputs), the frozen pretrained models, and one CPU script that rebuilds the submitted zip.
TEAM-ONLY: organisers' data and VoxBlink-derived files inside (licences forbid redistribution)."""
import base64, gzip, hashlib, io, json, os, re, shutil, sys, time, urllib.request, zipfile
from pathlib import Path

R = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(R / "kaggle"))
import kapi                                                            # noqa: E402  (tokens stay inside kapi.cli subprocesses)

DEST = Path(os.environ.get("FF_PKG_DEST", "E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl"))
STATE = DEST.parent / f".{DEST.name}.state"
TH = "thanhhuyen11/flag-2027-nb"
TRAIN = [   # run order; source + log only (their products are datasets under datasets/; outputs read by a later run: TRAIN_OUT)
    "duongphihoang/flag-2027-nb-150-lidwhisper", "phantrnanna/flag-2027-nb-152-mmslid",                       # Bengali LID filter
    "phanvnhtrng/flag-2027-nb-153b-b3s", "quynhnhu82/flag-2027-nb-153c-c5s",                                  # face adapters B3s / C5s
    "dngxunqun/flag-2027-nb-164f-cleancomp01-final",                                                           # S6 + T3 v11 (strict filter)
    "locnguyentruongan/flag-2027-nb-112s-f0s", "locnguyentruongan/flag-2027-nb-112s-f1s",                     # F1s heads (B0s, CA)
    "tranthuytien2511/flag-2027-nb-160a-whv3", "thanhhuyen11/flag-2027-nb-160b-whv3", "phuonggmaii/flag-2027-nb-160c-whv3",
    "nguynchnngn1/flag-2027-nb-162a-rfs", "phanththui/flag-2027-nb-162b-rfs", "phanvnnnnn6789/flag-2027-nb-162c-rfs",   # RFS features
    "ngnnguynchn/flag-2027-nb-163-rfs-b", "onnhqunhh/flag-2027-nb-163-rfs-l",                                  # RFS bundle, LoRA C1
    "phantrnhuyntrn/flag-2027-nb-183-factor-p1", "phantrnhuyntrn/flag-2027-nb-183-factor-p2",                 # FACTOR-01 lab
    "phantrnhuyntrn/flag-2027-nb-184-factor-report",                                                           # FACTOR-01 gate
    "ngnnguynchn/flag-2027-nb-163-rfsfxc5rl-p"]                                                                # final RFX projector
EVAL = [f"{TH}-65xbs-b3sx-eval", f"{TH}-65xcs-c5sx-eval", f"{TH}-47rs-screen2-eval", f"{TH}-50cc-e2e-eval",
        f"{TH}-112f0sx-eval", f"{TH}-112f1sx-eval", f"{TH}-163i-rfsfxc5rl-eval"]                              # run order; + outputs
TRAIN_OUT = {"locnguyentruongan/flag-2027-nb-112s-f0s", "phantrnhuyntrn/flag-2027-nb-183-factor-p1",   # kernel outputs that a later
             "phantrnhuyntrn/flag-2027-nb-183-factor-p2", "phantrnhuyntrn/flag-2027-nb-184-factor-report"}  # run reads (+ the gate report)
# kaggle CLI runs through cmd.exe (kapi.cli shell=True): no | ^ < > & in a pattern. 163i caches models / code in its output.
OUT_SKIP = r"\A(?!.*imagebind)(?!.*\.pyc\Z)(?!.*/code/)"
LOG_ONLY = r"\A__log_only__\Z"
SUB = R / "Experiment/POSTEVAL-01/zips/posteval_FACTOR_ENFUSE_c5rl.zip"


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def acct(ref):
    o = ref.split("/")[0]
    return o if o in kapi.accounts() else "ngnnguynchn"


def is_done(key):
    return (STATE / key.replace("/", "__")).exists()


def set_done(key, info=""):
    STATE.mkdir(parents=True, exist_ok=True); (STATE / key.replace("/", "__")).write_text(info or "ok", encoding="utf-8")


def slug(k):
    return k.split("/")[1]


def runs():
    p = DEST / "notebooks" / "RUNS.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------------------------------------------------------------- submitted zip
def step_submission():
    S = DEST / "submission"; S.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUB, S / SUB.name)
    sha = hashlib.sha256(SUB.read_bytes()).hexdigest()
    frozen = json.loads((R / "Experiment/POSTEVAL-01/manifest.json").read_text(encoding="utf-8"))["zips"][SUB.name]["sha256"]
    assert sha == frozen, "zip differs from the one frozen in POSTEVAL-01/manifest.json"
    (S / "SHA256.txt").write_text(f"{sha}  {SUB.name}\n", encoding="utf-8")
    log("submission", sha[:16])


# ---------------------------------------------------------------- notebooks (source + log) and Evaluation-path outputs
def step_notebooks():
    rec = runs()
    for grp, ks in (("1_train", TRAIN), ("2_eval", EVAL)):
        for j, k in enumerate(ks, 1):
            if is_done("nb_" + k):
                continue
            d = DEST / "notebooks" / grp; d.mkdir(parents=True, exist_ok=True)
            tmp = STATE / "tmp" / slug(k); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
            rc, t = kapi.cli(acct(k), ["kernels", "pull", k, "-p", str(tmp), "-m"], timeout=600)
            o = DEST / "outputs" / slug(k) if grp == "2_eval" else tmp / "out"
            o.mkdir(parents=True, exist_ok=True)
            pat = OUT_SKIP if grp == "2_eval" else LOG_ONLY
            rc2, t2 = kapi.cli(acct(k), ["kernels", "output", k, "-p", str(o), "-o", "--file-pattern", pat], timeout=4 * 3600)
            nb, lg = list(tmp.glob("*.ipynb")), list(o.glob("*.log"))
            if rc or rc2 or not nb or not lg:
                log("!! notebook", k, "FAILED", (t + t2)[-300:]); continue
            pre = f"{j:02d}_{slug(k)}"
            shutil.move(str(nb[0]), d / f"{pre}.ipynb"); shutil.move(str(lg[0]), d / f"{pre}.log")
            m = json.loads((tmp / "kernel-metadata.json").read_text(encoding="utf-8"))
            st = kapi.cli(acct(k), ["kernels", "status", k], timeout=180)[1].strip().splitlines()
            rec[k] = dict(file=f"notebooks/{grp}/{pre}.ipynb", gpu=m.get("enable_gpu"), internet=m.get("enable_internet"),
                          datasets=m.get("dataset_sources", []), kernel_inputs=m.get("kernel_sources", []),
                          outputs=f"outputs/{slug(k)}/" if grp == "2_eval" else None, status=st[-1] if st else "?")
            (DEST / "notebooks" / "RUNS.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            shutil.rmtree(tmp, ignore_errors=True); set_done("nb_" + k)
            log("notebook", k, "ok", f"{sum(1 for p in o.rglob('*') if p.is_file())} output files" if grp == "2_eval" else "")
    for k in sorted(TRAIN_OUT):                                         # outputs that a later training run attached as input
        if is_done("out_" + k):
            continue
        o = DEST / "outputs" / slug(k); o.mkdir(parents=True, exist_ok=True)
        rc, t = kapi.cli(acct(k), ["kernels", "output", k, "-p", str(o), "-o", "--file-pattern", OUT_SKIP], timeout=4 * 3600)
        for lg in o.glob("*.log"):
            lg.unlink()                                                 # the log is already next to the notebook
        if rc:
            log("!! outputs", k, "FAILED", t[-300:]); continue
        rec = runs(); rec[k]["outputs"] = f"outputs/{slug(k)}/"
        (DEST / "notebooks" / "RUNS.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        set_done("out_" + k); log("outputs", k, f"{sum(1 for p in o.rglob('*') if p.is_file())} files")


# ---------------------------------------------------------------- every Kaggle dataset any of the 26 runs attached -> datasets/<owner>__<name>.zip
# Kept as the zip Kaggle serves: extracting ~10^5 small files onto the USB disk took ~3 h per GB; the README unzips them once.
def step_datasets():
    rec = runs()
    assert all(k in rec for k in TRAIN + EVAL), "run the notebooks step first (RUNS.json incomplete)"
    for ds in sorted({d for v in rec.values() for d in v["datasets"]}):
        if is_done("ds_" + ds):
            continue
        name = ds.replace("/", "__"); d = DEST / "datasets" / name; z = DEST / "datasets" / f"{name}.zip"
        if d.is_dir() and any(d.iterdir()) and not list(d.glob("*.zip")):     # already extracted by an earlier version
            set_done("ds_" + ds, "dir"); log("dataset", ds, "kept extracted folder"); continue
        tmp = STATE / "tmp" / name; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        rc, t = kapi.cli(acct(ds), ["datasets", "download", ds, "-p", str(tmp), "-o"], timeout=6 * 3600)
        got = list(tmp.glob("*.zip"))
        if rc or len(got) != 1:                                   # Kaggle answers 404 for the whole-dataset zip of a few datasets
            shutil.rmtree(tmp, ignore_errors=True)
            n = per_file(ds, d) if ds not in LOCAL else local_copy(ds, d)
            if n:
                set_done("ds_" + ds, "folder"); log("dataset", ds, f"{n} files (folder: whole zip refused)")
            else:
                log("!! dataset", ds, "FAILED", t[-300:])
            continue
        with zipfile.ZipFile(got[0]) as zz:
            n = len(zz.namelist())
        shutil.move(str(got[0]), str(z)); shutil.rmtree(d, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
        set_done("ds_" + ds); log("dataset", ds, f"{n} files, {z.stat().st_size / 1e9:.2f} GB zip")


# raw-v4-p was uploaded from the BTC zips in Input/; Kaggle shows each uploaded zip X.zip extracted as folder X/ (the README unzips them).
LOCAL = {"ngnnguynchn/flag2027-raw-v4-p": ["Input/train_set.zip", "Input/dev_set.zip", "Input/meta_file_train_set.csv"]}


def local_copy(ds, d):
    d.mkdir(parents=True, exist_ok=True)
    for f in LOCAL[ds]:
        shutil.copy2(REPO / f, d / Path(f).name)
    return len(LOCAL[ds])


def per_file(ds, d):
    """Download a dataset file by file (listing via `datasets files`, paginated), keeping relative paths."""
    names, tok = [], None
    while True:
        rc, t = kapi.cli(acct(ds), ["datasets", "files", ds, "--page-size", "200"] + (["--page-token", tok] if tok else []), timeout=600)
        if rc:
            return 0
        lines = t.splitlines(); tok = next((l.split("=", 1)[1].strip() for l in lines if l.startswith("Next Page Token")), None)
        i = next(k for k, l in enumerate(lines) if l.startswith("---")) + 1
        names += [l.split()[0] for l in lines[i:] if l.strip()]
        if not tok:
            break
    tmp = STATE / "tmp" / "perfile"
    for nm in names:
        if (d / nm).exists():
            continue
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        rc, t = kapi.cli(acct(ds), ["datasets", "download", ds, "-f", nm, "-p", str(tmp), "-o"], timeout=3600)
        got = [p for p in tmp.iterdir() if not p.name.endswith(".kaggle-partial")]
        if rc or len(got) != 1:
            log("!! file", ds, nm, t[-200:]); return 0
        (d / nm).parent.mkdir(parents=True, exist_ok=True)
        if got[0].name == Path(nm).name + ".zip" and not nm.endswith(".zip"):     # large single files arrive zipped
            with zipfile.ZipFile(got[0]) as zz:
                zz.extract(zz.namelist()[0], tmp / "x"); shutil.move(str(tmp / "x" / zz.namelist()[0]), str(d / nm))
        else:
            shutil.move(str(got[0]), str(d / nm))
    shutil.rmtree(tmp, ignore_errors=True)
    return len(names)


# ---------------------------------------------------------------- frozen pretrained models of the Evaluation path
URL_RE = re.compile(r"https?://[^\s'\"\)\]>,]+?\.(?:pt|pth|ckpt|bin|onnx)\b")
HF_RE = re.compile(r"['\"]((?:openai|facebook|Wespeaker|speechbrain|audeering|nateraw|microsoft|laion|MCplayer|google|timm|"
                   r"FacePerceiver|BAAI|pyannote|dima806|abhilash88|prithivMLmods)/[A-Za-z0-9_.\-]+)['\"]")


def texts_of(nb):
    src = "".join("".join(c["source"]) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"])
    out = [src]
    for b in re.findall(r"[A-Za-z0-9+/=]{200,}", src):              # code embedded as base64 zip / gzip payloads
        try:
            raw = base64.b64decode(b + "=" * (-len(b) % 4))
            if raw[:2] == b"PK":
                z = zipfile.ZipFile(io.BytesIO(raw))
                out += [z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith((".py", ".json", ".sh"))]
            elif raw[:2] == b"\x1f\x8b":
                out.append(gzip.decompress(raw).decode("utf-8", "replace"))
        except Exception:
            pass
    return out


def step_pretrained():
    P = DEST / "pretrained"; P.mkdir(parents=True, exist_ok=True)
    src = DEST / "outputs" / slug(EVAL[-1]) / "FLAG_RF_E2E" / "models"    # models 163i downloaded (SimAM, ResNet221, ECAPA, VGG-Face)
    if src.exists():
        for f in src.rglob("*"):
            if f.is_file():
                t = P / "rfs_models" / f.relative_to(src); t.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(f), t)
        shutil.rmtree(src, ignore_errors=True)
    FC = P / "cache" / "flag"; FC.mkdir(parents=True, exist_ok=True)   # FLAG_CACHE layout (flag_models._download: cache/<basename>)
    HUB = P / "cache" / "huggingface" / "hub"                           # HF cache layout: --cache pretrained/cache -> /root/.cache
    ib = Path("D:/ff_models/imagebind_huge.pth")                       # Meta's link died 12/11; same SHA-256 as release/docker/ff.py
    if ib.exists() and not (P / "imagebind_huge.pth").exists():
        shutil.copy2(ib, P / "imagebind_huge.pth")
    urls, hf = {}, {}
    for nb in (DEST / "notebooks" / "2_eval").glob("*.ipynb"):
        for t in texts_of(nb):
            for u in URL_RE.findall(t):
                urls.setdefault(u, set()).add(nb.stem)
            for h in HF_RE.findall(t):
                hf.setdefault(h, set()).add(nb.stem)
    have = {f.name for f in P.rglob("*") if f.is_file()}
    status = {}
    for u in sorted(urls):
        name = u.rsplit("/", 1)[1]
        if "imagebind" in u or name in have:
            status[u] = "present"; continue
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "python-urllib"}), timeout=120) as r, \
                    open(FC / (name + ".part"), "wb") as w:
                shutil.copyfileobj(r, w, 1 << 22)
            (FC / (name + ".part")).replace(FC / name); status[u] = "downloaded"
        except Exception as e:
            status[u] = f"FAILED: {e}"
        log("url", u, status[u])
    from huggingface_hub import snapshot_download
    for h in sorted(hf):
        if is_done("hf_" + h):
            status[h] = "downloaded"; continue
        try:
            d = Path(snapshot_download(h, cache_dir=str(HUB), allow_patterns=["*.json", "*.txt", "*.model", "*.yaml", "*.safetensors",
                                                                              "*.tiktoken"]))
            if not any(d.rglob("*.safetensors")):                       # repos without safetensors
                snapshot_download(h, cache_dir=str(HUB), allow_patterns=["pytorch_model.bin", "*.pt", "*.ckpt"])
            set_done("hf_" + h); status[h] = "downloaded"
        except Exception as e:
            status[h] = f"FAILED: {str(e)[:200]}"
        log("hf", h, status[h])
    (P / "SOURCES.json").write_text(json.dumps({k: dict(used_by=sorted((urls.get(k) or hf.get(k))), status=v) for k, v in status.items()},
                                               indent=1), encoding="utf-8")


# ---------------------------------------------------------------- rebuild: the six files the final CPU step reads + the script
def step_rebuild():
    B = DEST / "rebuild"; I = B / "in"; I.mkdir(parents=True, exist_ok=True)
    O = DEST / "outputs"
    pick = {"f1s_eval.zip": O / slug(EVAL[5]) / "submission_cleantune01_F1_CA_eval.zip", "rfx_eval.zip": O / slug(EVAL[6]) / "FLAG_RF_E2E/submission.zip",
            "components_eval.npz": O / slug(EVAL[3]) / "e2e_eval/components_eval.npz",
            "e2e_manifest.json": O / slug(EVAL[3]) / "e2e_eval/e2e_manifest.json",
            "cleantune01_fuser_bangla_CA.json": O / slug(EVAL[5]) / "e2e_out/cleantune01_fuser_bangla_CA.json",
            "ct01r2_final_eval.npz": O / slug(EVAL[5]) / "e2e_out/ct01r2_final_eval.npz"}
    for n, p in pick.items():
        if not p.exists():
            k = O / p.relative_to(O).parts[0]; hit = sorted(k.rglob(p.name)); assert len(hit) == 1, (p.name, hit); p = hit[0]   # only inside that run
        shutil.copy2(p, I / n)
    shutil.copy2(HERE / "rebuild_final.py", B / "rebuild_final.py")
    import subprocess
    r = subprocess.run([sys.executable, str(B / "rebuild_final.py")], capture_output=True, text=True, cwd=str(B))
    (B / "rebuild_report.txt").write_text(r.stdout + r.stderr, encoding="utf-8"); (B / "FACTOR_ENFUSE_c5rl_rebuilt.zip").unlink(missing_ok=True)
    log("rebuild:", (r.stdout.strip().splitlines() or ["?"])[-1])


def step_docker():
    """docker/: the repo's runner (release/docker) set up for this package: stages = the 7 notebooks of notebooks/2_eval (exact Kaggle
    sources), datasets from datasets/<owner>__<name>, 163i = the RFX inference run; for 163i the Evaluation data (--evalds) is mounted
    at the path its DATA_ROOT names (Kaggle's newer layout /kaggle/input/datasets/<owner>/<name>)."""
    D = DEST / "docker"; D.mkdir(parents=True, exist_ok=True)
    src = R / "release" / "docker"
    for f in ("Dockerfile", "run_nb.py"):
        shutil.copy2(src / f, D / f)
    ff = (src / "ff.py").read_text(encoding="utf-8")
    old = "        mounts += bind(a.evalds, f\"/kaggle/input/{Path(a.evalds).resolve().name}\")\n"
    assert ff.count(old) == 1, "ff.py changed: update step_docker"
    ff = ff.replace(old, "        tgt = f\"/kaggle/input/{Path(a.evalds).resolve().name}\"\n"
                         "        if sid == \"163i\":   # FACTOR_ENFUSE package: 163i's DATA_ROOT is the path of the Kaggle run\n"
                         "            tgt = \"/kaggle/input/datasets/ngnnguynchn/flag2027-evalphase-p\"\n"
                         "        mounts += bind(a.evalds, tgt)\n")
    (D / "ff.py").write_text(ff, encoding="utf-8")
    S = json.loads((src / "stages.json").read_text(encoding="utf-8"))
    nbs = {p.name[:2]: p for p in (DEST / "notebooks" / "2_eval").glob("*.ipynb")}
    for j, sid in enumerate(S["order"], 1):
        st = S["stages"][sid]; k = EVAL[j - 1]
        assert slug(k).split("-nb-")[1].startswith(sid), (sid, k)
        st["notebook"] = nbs[f"{j:02d}"].relative_to(DEST).as_posix(); st["edits"] = []; st.pop("pulled_from", None)
        st["datasets"] = [d for d in runs()[k]["datasets"] if d != "ngnnguynchn/flag2027-evalphase-p"]   # mounted by --evalds
    S["stages"]["163i"].update(what="RFX inference (re-extracts faces / voices, adds C5s + ResNet293 || ECAPA1024 blocks) -> RFX zip",
                               after=["47rs", "65xcs"], slug="flag-2027-nb-163i-rfsfxc5rl-{split}", step="RFX")
    S["about"] = "Evaluation path of FACTOR_ENFUSE (c5rl): the 7 Kaggle runs of 11-13/11/2026 (notebooks/2_eval). {split} = eval."
    (D / "stages.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    (DEST / ".dockerignore").write_text("*\n!docker/\n", encoding="utf-8")   # the image only needs docker/run_nb.py
    log("docker: stages", S["order"])


def step_readme():
    shutil.copy2(HERE / "README.md", DEST / "README.md")


def redact():
    """`kapi push --secret` writes the secret values into the first cell of the pushed notebook, and `kernels pull` brings them back."""
    vals = [v for v in kapi.secrets().values() if v]
    for f in [p for p in (DEST / "notebooks").rglob("*") if p.suffix in (".ipynb", ".log")]:
        t = t0 = f.read_text(encoding="utf-8", errors="replace")
        for v in vals:
            t = re.sub(r"os\.environ\[(['\"])(\w+)\1\] = (['\"])" + re.escape(v) + r"\3",
                       r"# os.environ['\2'] = '<removed from the package: set your own if needed>'", t).replace(v, "<removed>")
        if t != t0:
            f.write_text(t, encoding="utf-8"); log("redacted secret in", f.name)


def step_manifest():
    redact()
    files = sorted(p for p in DEST.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    with open(DEST / "MANIFEST.sha256", "w", encoding="utf-8") as w:
        for p in files:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
            w.write(f"{h.hexdigest()}  {p.relative_to(DEST).as_posix()}\n")
    log(f"MANIFEST.sha256: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB")


STEPS = dict(submission=step_submission, notebooks=step_notebooks, datasets=step_datasets, pretrained=step_pretrained,
             rebuild=step_rebuild, docker=step_docker, readme=step_readme, manifest=step_manifest)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    DEST.mkdir(parents=True, exist_ok=True)
    for s in (list(STEPS) if sys.argv[1:] == ["all"] else sys.argv[1:]):
        log("== step", s, "->", DEST); STEPS[s]()
    log("finished")
