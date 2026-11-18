"""Generate FLAG_50_e2e_v11.ipynb: raw organiser data -> system v11 (CodaBench dev 22.03) -> submission.zip, end to end on
Kaggle GPU T4 x2, resumable. The Evaluation pipeline (docs/PLAN_E2E_2026-10-10.md; TODO B3).

Chain (FLAG_30 = v9 extended):
  GPU 1, background   v11's T3 head (Experiment/E2E/v11_head.py, lab code embedded, data flag2027-lab-data + eda bundle):
                      trained once with BIGB-01's recipe and EXPORTED (v11_head/v11_t3head.pt); an attached earlier output's
                      export is reused, so the Evaluation run scores with the head of the dev run. When the notebook signals
                      that the T3 arrays of the test files exist, the job writes their T3 streams (v11_head/t3_streams.npz)
  GPU 0               FLAG_12 cells + FaRL / AdaFace (FLAG_30) + the 4 T3 encoders (vendor_t3enc/t3enc_cell.py, FLAG_40) on
                      train + every test list; w2v2 + T3 first (the head job needs them), then the rest; FLAG_14 AudioCLIP
  CPU                 Experiment/E2E/system_v11.py (verbatim): per test file the components (B3 + CA-FUSE conditions, CA-EN
                      streams + conditions, T3) and the v9 / v11 scores; level-0 self-test on half of every file
Outputs: submission.zip (= v11), submission/submission_v9.zip, e2e_<split>/ (components_<split>.npz + manifest + fusers +
SAFE-04 tables: the E2E_DIR of the experiment notebooks FLAG_51-53 for that split), checks.json.
On dev: every file is compared with the submitted v9 / v11 zips, the components with flag2027-e2e-dev, the fresh T3 arrays
with the lab's (train) and the e2e-dev's (dev) arrays.
    python build_e2e_v11.py
"""
import base64, hashlib, io, json, zipfile
from pathlib import Path
import build_best_notebook as B30
import build_eval_notebooks as B12
import build_e2e_notebooks as E
import build_av_notebook as B14
import build_t3enc_notebook as B40
import build_lab_notebook as L

R = Path(__file__).resolve().parents[1]
REFS = {"v9": R / "Experiment/CAFUSE-EN/submission_CA_EN_BN.zip", "v11": R / "Experiment/BIGB-01/submission_v11.zip",
        "v12": R / "Experiment/EN-T3-01/submission_v12.zip"}
HEAD_CODE = [f for f in L.CODE if not f.startswith(("Experiment/SCALE-VB", "Experiment/VBSEL-01", "Experiment/ENBASE-01",
                                                      "Experiment/BIGB-01"))] + ["Experiment/E2E/v11_head.py"]
HEAD_ENTRY = L.ENTRY.replace('run_py = ROOT / "Experiment" / sys.argv[1] / "run.py"', 'run_py = ROOT / "Experiment/E2E/v11_head.py"').replace(
    "sys.argv = [str(run_py)] + sys.argv[2:]", "sys.argv = [str(run_py)] + sys.argv[1:]").replace(
    '"""Kaggle entry of the SAFE-04 lab: python lab_entry.py <experiment dir, e.g. BIGB-01> [args of its run.py]',
    '"""Background job of FLAG_50: python head_entry.py both <out> <feats> (Experiment/E2E/v11_head.py)')
assert HEAD_ENTRY.count("v11_head.py") == 2

INTRO = """
# FLAG 2027 — NB-50: end to end, raw data -> system **v11** (CodaBench dev **22.03**) -> `submission.zip`

The Evaluation pipeline of the current best system (docs/PLAN_E2E_2026-10-10.md). Every pair is scored on its own from
train constants (level 0, organiser ruling of 30/09); the recipe per file language (organiser file names) is declared.

| File language | Recipe (v11) | dev EER (ng / g) |
|---|---|---|
| English | v9 = **CA-EN**: s007 + s010B deep / CCA, ImageBind, age, AudioCLIP + 6 pair conditions (`fuser_en.json`) | 16.87 / 27.49 |
| Bangla | **CA-FUSE + T3**: the 6 B3 streams + v11's T3 head (P512 InfoNCE on 70 v4 + 151 MAV-Celeb + 4 852 Bengali-filtered VoxCeleb2 identities; FaRL16 + CLIP-L / SimAM100 p + ft) + `t3_relg` + 7 conditions (`cafuse_t3_fuser_v11.json`) | 17.21 / 26.57 |

**How it runs (T4 x2):**
- **GPU 1, background job:** v11's T3 head is trained once (about 10 min) and **exported** (`v11_head/v11_t3head.pt`).
  An attached earlier output's export is reused, so the Evaluation run scores with the SAME head as the dev run.
  When the T3 arrays of the test files exist, the job writes their T3 streams.
- **GPU 0:** FLAG_12 encoders + FaRL / AdaFace + the 4 T3 encoders on train + every test list, then AudioCLIP.
- **CPU:** `system_v11.py` computes the components of every test file, the v9 / v11 scores, and the level-0 self-tests.

**Inputs (Add Input):**
- raw organiser data (`train_set` + `dev_set`; on 11/11 also the Evaluation data, then set `ONLY_UNDER`);
- `flag2027-members`, `flag2027-eda-bundle`, `flag2027-best-v7` (as FLAG_30);
- **`flag2027-lab-data`** (the head's training rows) and **`flag2027-e2e-dev`** (SAFE-04 tables + fusers for the `e2e_<split>/` folder; dev checks);
- optional: an earlier output of this notebook (arrays and the head export are reused).

**Settings:** **GPU T4 x2**, **Internet ON** (checkpoints: ImageBind, AudioCLIP, CLIP, FaRL ep16 / ep64, CLIP-L, AdaFace, w2v2, ECAPA, SimAM100 x 2).
**Time:** about 3-3.5 h on dev. Interrupted? Save Version, attach the output, run again: finished arrays and the head are reused.

**Output:**
- `submission.zip` (= v11; scores are written as -score, so lower = same);
- `submission/submission_v9.zip`, `submission/submission_v12.zip` (v12: English = CA_EN_T3, Bangla = v11; the other Evaluation candidate);
- `e2e_<split>/` (attach it to FLAG_51-53 with `E2E_SPLIT=<split>`);
- `submission/checks.json`.

**Pass marks on dev:** English files rank-corr >= 0.999 vs the submitted v11 (= v9) zip; Bangla rank-corr >= 0.97 vs v11 (the
head is retrained on a GPU; it is not bit-identical to the CPU head of the submitted zip). Then submit the dev zip ONCE to
CodaBench to confirm the pipeline (expected about 22.0 ± 0.3).
**Known limit:** file names / trial lists follow the dev layout (`flag_lib.NAMES`); check them against the Evaluation data.
"""

ENV = B30.ENV
_a = '["arcface", "ecapa", "vitage", "w2v2ag", "ibv", "iba", "farl", "adaface"]'
assert ENV.count(_a) == 1
ENV = ENV.replace(_a, '["w2v2ag", "farl16V", "clipLV", "sam100pA", "sam100fA", "arcface", "ecapa", "vitage", "ibv", "iba", "farl", "adaface"]')
ENV = ENV.rstrip() + r'''
T3_FIRST = ["w2v2ag", "farl16V", "clipLV", "sam100pA", "sam100fA"]   # what the background head job needs: extracted first
HEADOUT = WORKDIR / "v11_head"; HEADOUT.mkdir(exist_ok=True)
import torch as _t
NGPU = _t.cuda.device_count()
print("GPUs:", [_t.cuda.get_device_name(i) for i in range(NGPU)] or "none")
'''

PIP_T3 = r'''
# FaRL (ep64 for B3, ep16 for T3) on openai CLIP @ d50d76d; WeSpeaker @ 9fecd6c (SimAM100, as FLAG_40); AdaFace (yaml / omegaconf)
if not LOCAL:
    !pip -q install ftfy regex pyyaml omegaconf kaldiio silero-vad 2>&1 | tail -1
    # 29/10: one pip call per package; the openai CLIP build fails on Kaggle's current image (setup.py needs pkg_resources)
    # and, in a shared call, took wespeaker down with it (FLAG_103c smoke). CLIP then comes from a git clone (FLAG_103a smoke).
    !pip -q install --no-deps "git+https://github.com/wenet-e2e/wespeaker.git@9fecd6cb4f47475d01761d87c826298dff4ef18c" 2>&1 | tail -2
    !pip -q install --no-deps "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1" 2>&1 | tail -2
    try:
        import clip
    except ModuleNotFoundError:
        import subprocess
        _oc = Path("/tmp/openai_clip")
        if not (_oc / "clip").exists():
            subprocess.run(["git", "clone", "-q", "https://github.com/openai/CLIP.git", str(_oc)], check=True)
            subprocess.run(["git", "-C", str(_oc), "checkout", "-q", "d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"], check=True)
            _f = _oc / "clip/clip.py"
            _f.write_text(_f.read_text().replace("from pkg_resources import packaging", "import packaging, packaging.version"))
        sys.path.insert(0, str(_oc))
        import clip
    import importlib.util                       # not `import wespeaker`: its frontends need s3prl (SimAMVoice stubs them first)
    assert importlib.util.find_spec("wespeaker") is not None, "wespeaker did not install"
    print("openai CLIP:", clip.__file__, "| wespeaker:", importlib.util.find_spec("wespeaker").origin, flush=True)
else:                                           # local smoke test: wespeaker + its deps in an isolated folder
    for p_ in reversed([p for p in os.environ.get("FLAG_T3_PATH", "").split(";") if p]):
        sys.path.insert(0, p_)
'''

HEADJOB = r'''
# ============================================================================ v11's T3 head: background job (GPU 1 when there are two)
import base64, hashlib, subprocess
HEAD_B64 = "__HEAD_B64__"
HEAD_MD5 = __HEAD_MD5__
LAB = WORKDIR / "lab"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(HEAD_B64))) as z:
    for f in z.namelist():
        b = z.read(f); assert hashlib.md5(b).hexdigest() == HEAD_MD5[f], f
        (LAB / f).parent.mkdir(parents=True, exist_ok=True); (LAB / f).write_bytes(b)
labm = [m for m in IN.rglob("lab_manifest.json") if json.loads(m.read_text()).get("kind", "data") == "data"]
assert len(labm) == 1, "attach flag2027-lab-data (the T3 head's training rows)"
LABDATA = labm[0].parent
edam = sorted(IN.rglob("eda_bundle_manifest.json"))
assert len(edam) == 1, "attach flag2027-eda-bundle"
assert json.loads(edam[0].read_text())["created"] == json.loads(labm[0].read_text())["eda_bundle_created"], "eda bundle version"
for d in __DATA_DIRS__:
    tgt, src = LAB / d, LABDATA / d
    if not tgt.exists():
        tgt.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src, tgt, target_is_directory=True)
        except OSError:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(tgt), str(src)], check=True, capture_output=True)
for prev in IN.rglob("v11_t3head.pt"):                    # reuse the head of an earlier run (dev run -> Evaluation run)
    if not (HEADOUT / "v11_t3head.pt").exists():
        shutil.copy(prev, HEADOUT / "v11_t3head.pt"); log("v11 head reused from", prev)
(OUT / "t3_ready.json").unlink(missing_ok=True); (HEADOUT / "t3_streams.npz").unlink(missing_ok=True)
HEAD_LOG = open(HEADOUT / "job.log", "w")
henv = dict(os.environ, SAFE04_SCRATCH=str(WORKDIR / "lab_sp"), FLAG_LOCAL="1", FLAG_IN=str(edam[0].parent.parent),
            FLAG_OUT=str(WORKDIR / "lab_sp/out"), VOX_LID=str(LAB / "Experiment/T3ENC-01/feats_vox2lid/lid_summary.json"),
            SAFE04_THREADS=str(max(1, (os.cpu_count() or 4) // 2)), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
            CUDA_VISIBLE_DEVICES="1" if NGPU > 1 else ("0" if NGPU else ""))
(WORKDIR / "lab_sp/out/safe04").mkdir(parents=True, exist_ok=True)
HEAD_JOB = subprocess.Popen([sys.executable, "-u", str(LAB / "head_entry.py"), "both", str(HEADOUT), str(OUT)], cwd=str(WORKDIR),
                            env=henv, stdout=HEAD_LOG, stderr=subprocess.STDOUT)
log(f"v11 head job started (pid {HEAD_JOB.pid}, GPU {henv['CUDA_VISIBLE_DEVICES'] or 'none'}): log {HEADOUT / 'job.log'}")
'''

_seed = 'if re.search(r"feats[-_]eval", p.as_posix()) and out not in p.parents:'
assert B12.DATA12.count(_seed) == 1
DATA12 = B12.DATA12.replace(_seed, 'if re.search(r"feats[-_]eval", p.as_posix()) and out not in p.parents and "/lab/kaggle/output/" not in p.as_posix():'
                            '   # never the lab bundle (ECAPA-6144 placeholders)')

KEEP = r'''
_build_face2, _release12 = build, release                 # the T3 cell below defines its own build / release
'''

DISPATCH = r'''
_build_t3 = build


def build(name):
    return _build_t3(name) if name in NEW else _build_face2(name)


release = _release12                                      # FLAG_12's release keeps ImageBind between ibv and iba
print("T3 encoders registered:", list(NEW))
'''

_run = B30.RUN12
_cut = _run.index("CHECKS, FAILED = {}, {}")
_defs, _loop = _run[:_cut], _run[_cut + len("CHECKS, FAILED = {}, {}"):]
_m = 'MAIN = {"arcface": "", "ecapa": "192", "vitage": "", "w2v2ag": "", "ibv": "", "iba": "", "farl": "", "adaface": ""}'
assert _defs.count(_m) == 1
_defs = _defs.replace(_m, _m + "\nARRAYS.update(ARRAYS_NEW); MAIN.update(MAIN_NEW)                         # the 4 T3 encoders")
RUN_A = _defs + r'''CHECKS, FAILED = {}, {}
ENCODERS_ALL = list(ENCODERS)
ENCODERS = [e for e in ENCODERS_ALL if e in T3_FIRST]     # w2v2 + T3 first: the head job scores the test files meanwhile
''' + _loop + r'''
ENCODERS = ENCODERS_ALL
TEST_TAGS = [tag(k) for k in S if k != "train"]
missing = [f"{k}_{n}_{t}" for t in TEST_TAGS for k, n in (("face", "farl16"), ("face", "clipL"), ("voice", "sam100p"),
           ("voice", "sam100f"), ("voice", "w2v2aggen")) if not (OUT / f"{k}_{n}_{t}.npy").exists()]
assert not missing, f"T3 / w2v2 arrays missing: {missing}"
(OUT / "t3_ready.json").write_text(json.dumps({"splits": TEST_TAGS}))
log("T3 arrays of the test files ready -> the head job scores them:", TEST_TAGS)
'''
RUN_B = r'''
ENCODERS = [e for e in ENCODERS_ALL if e not in T3_FIRST]
''' + _loop + r'''
ENCODERS = ENCODERS_ALL
'''

T3CHECK = r'''
# ============================================================================ drift of the fresh T3 arrays (another Kaggle image = other library versions)
# train rows vs the arrays the head was trained with (flag2027-lab-data); dev rows vs flag2027-e2e-dev (when attached)
def cos_min_med(A, B):
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    c = np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
    return round(float(c.min()), 5), round(float(np.median(c)), 6)


T3DRIFT = {}
for kind, n in (("face", "farl16"), ("face", "clipL"), ("voice", "sam100p"), ("voice", "sam100f")):
    a, b = OUT / f"{kind}_{n}_train.npy", LABDATA / f"Experiment/T3ENC-01/feats_t3enc/{kind}_{n}_train.npy"
    if a.exists() and b.exists():
        T3DRIFT[f"train {kind}_{n}"] = cos_min_med(np.load(a), np.load(b))
E2EDEV = next((m.parent for m in IN.rglob("e2e_manifest.json") if json.loads(m.read_text()).get("kind") == "e2e-dev"), None)
if E2EDEV is not None and any(k.startswith("no_gender/") for k in S) and not ONLY_UNDER:   # dev only (11/11)
    man = json.loads((E2EDEV / "e2e_manifest.json").read_text()); zc = np.load(E2EDEV / "components_dev.npz")
    for f in man["files"]:
        for kind, n in (("face", "farl16"), ("face", "clipL"), ("voice", "sam100p"), ("voice", "sam100f")):
            p = OUT / f"{kind}_{n}_{f['key'].replace('/', '_')}.npy"
            if p.exists() and np.load(p, mmap_mode="r").shape == zc[f"{kind}_{n}_{f['i']}"].shape:
                T3DRIFT[f"{f['key']} {kind}_{n}"] = cos_min_med(np.load(p), zc[f"{kind}_{n}_{f['i']}"])
for k, v in T3DRIFT.items():
    print(f"  {k:40s} cos min {v[0]:.5f} median {v[1]:.6f}" + ("   <-- DRIFT" if v[1] < 0.999 else ""))
'''

SYSTEM = r'''
# ============================================================================ system v11: components, v9 / v11 scores, zips, e2e_<split>/
import importlib.util
for m in ("flag_lib", "flag_v2", "safe_lib", "system_v11", "v6p"):
    sys.modules.pop(m, None)
os.environ["FLAG_DATA"] = str(ORG)
spec = importlib.util.spec_from_file_location("system_v11", WORKDIR / "repo/Experiment/E2E/system_v11.py")
SY = importlib.util.module_from_spec(spec); spec.loader.exec_module(SY)


def input_dir(name):
    ps = sorted(SCAN["files"].get(name, []), key=lambda p: len(p.as_posix()))
    pref = os.environ.get("XALL_PREFER")                   # EVALPATH-U (opt-in): several datasets hold this file -> the one whose path has this text
    if pref and name == "s007_xall.pt":
        ps = [p for p in ps if pref in p.as_posix()] or ps
    return ps[0].parent if ps else None


v6_members, xall = input_dir("s007_seed1.pt"), input_dir("s007_xall.pt")
bundle, adaext = input_dir("eda_bundle_manifest.json"), input_dir("ext_v1_complete_face_adaface.npy")
assert v6_members is not None, "attach flag2027-members (v6 members: s007 / s010B seeds)"
assert xall is not None and adaext is not None, "attach flag2027-best-v7 (s007_xall.pt + ext AdaFace arrays)"
FUS = {k: json.loads((WORKDIR / "repo" / p).read_text()) for k, p in
       {"v9_bangla": "BEST/v7_safe/cafuse_fuser.json", "v9_english": "BEST/v7_safe/fuser_en.json",
        "v11_bangla": "Experiment/BIGB-01/cafuse_t3_fuser_v11.json", "v12_english": "Experiment/EN-T3-01/fuser_en_t3.json"}.items()}
t0 = time.time()
while HEAD_JOB.poll() is None:                             # the T3 head job (GPU 1) must have scored the test files
    time.sleep(10)
    if time.time() - t0 > 600:
        log("waiting for the head job ..."); t0 = time.time()
HEAD_LOG.flush()
print("".join((HEADOUT / "job.log").read_text(errors="replace").splitlines(True)[-12:]))
assert HEAD_JOB.returncode == 0 and (HEADOUT / "t3_streams.npz").exists(), "the v11 head job failed: see v11_head/job.log"
T3Z = np.load(HEADOUT / "t3_streams.npz")
sy = SY.System(ORG, [OUT, OUT_AV], [bundle, adaext], xall / "s007_xall.pt", v6_members, FUS["v9_bangla"], threads=os.cpu_count() or 4,
               n_models=N_MODELS)
is_dev = any(k.startswith("no_gender/") for k in S) and not ONLY_UNDER
SPL = "dev" if is_dev else "eval"
comps, checks = {}, {"t3_drift": T3DRIFT, "head": json.loads(str(T3Z["info"]))}
for i, (c, k) in enumerate(sy.cells):
    d = sy.file_components(c, k)
    d["t3"] = T3Z[tag(c)].astype(np.float64)
    d["v9"], d["v11"] = SY.scores(d, FUS)
    d["v12"] = SY.v12_scores(d, FUS)
    for kind, n in (("face", "farl16"), ("face", "clipL"), ("voice", "sam100p"), ("voice", "sam100f")):
        d[f"{kind}_{n}"] = np.load(OUT / f"{kind}_{n}_{tag(c)}.npy").astype(np.float32)
    d["selftest"] = sy.selftest(c, k)
    assert d["selftest"] < 1e-5 and np.isfinite(d["v11"]).all(), (c, d["selftest"])
    comps[i] = d
    checks[c] = dict(rows=len(d["pid"]), level0_half_file_max_abs_diff=d["selftest"])
    log(f"{c}: {len(d['pid'])} trials | level-0 self-test {d['selftest']:.1e}")
E2O = WORKDIR / f"e2e_{SPL}"; E2O.mkdir(exist_ok=True)
SY.save_components(E2O / f"components_{SPL}.npz", {i: {k: v for k, v in d.items() if k != "selftest"} for i, d in comps.items()})
(E2O / "fusers").mkdir(exist_ok=True)
for k, v in FUS.items():
    (E2O / "fusers" / f"{k}.json").write_text(json.dumps(v))
if E2EDEV is not None:                                   # the SAFE-04 tables of the experiment notebooks' gate
    shutil.copytree(E2EDEV / "safe04", E2O / "safe04", dirs_exist_ok=True)
files = [dict(i=i, key=d["key"], zip_name=d["zip_name"], protocol=d["protocol"], lang=d["lang"], is_en=d["is_en"], rows=len(d["pid"]))
         for i, d in comps.items()]
(E2O / "e2e_manifest.json").write_text(json.dumps(dict(created=time.strftime("%Y-%m-%d %H:%M"), kind=f"e2e-{SPL}", split=SPL, files=files,
    columns=dict(b3=SY.B3_COLS, co=SY.CO_COLS, en7=SY.EN7_COLS, enco=SY.ENCO_COLS, t3=SY.T3_COLS), checks=checks,
    source="FLAG_50_e2e_v11"), indent=1, default=str))
SY.write_zip(SUB / "submission.zip", comps, "v11"); SY.write_zip(SUB / "submission_v9.zip", comps, "v9")
SY.write_zip(SUB / "submission_v12.zip", comps, "v12")
shutil.copy(SUB / "submission.zip", WORKDIR / "submission.zip")
readz = lambda z, n: pd.read_csv(z.open(n), sep=" ", header=None, names=["pid", "s"], dtype={"pid": str})
for name, ref in (REF_ZIPS.items() if is_dev else []):    # dev: the submitted zips (v9 vs this run's v9, v11 vs its v11)
    with zipfile.ZipFile(SUB / {"v9": "submission_v9.zip", "v12": "submission_v12.zip"}.get(name, "submission.zip")) as a, zipfile.ZipFile(ref) as b:
        for i, d in comps.items():
            mine, other = readz(a, d["zip_name"]), readz(b, d["zip_name"])
            r = dict(same_pair_ids=bool((mine.pid.values == other.pid.values).all()), rank_corr=round(float(mine.s.corr(other.s, method="spearman")), 5),
                     max_abs_diff=float(np.abs(mine.s - other.s).max()))
            checks[f"vs submitted {name}: {d['key']}"] = r
            log(f"vs submitted {name} {d['key']}: {r}")
if E2EDEV is not None and is_dev:                        # dev: the components of the local build (flag2027-e2e-dev)
    zc = np.load(E2EDEV / "components_dev.npz")
    for i, d in comps.items():
        r = {k: round(float(np.abs(d[k] - zc[f"{k}_{i}"]).max()), 6) for k in ("b3", "co", "en7", "enco")}
        r["t3_corr"] = [round(float(np.corrcoef(d["t3"][:, j], zc[f"t3_{i}"][:, j])[0, 1]), 4) for j in range(3)]
        checks[f"vs e2e-dev components: {d['key']}"] = r
        log(f"vs e2e-dev components {d['key']}: {r}")
(SUB / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
(SUB / "broken.json").write_text(json.dumps(BROKEN, indent=1))
log("-> submission.zip (v11),", SUB / "submission_v9.zip", ",", E2O)
'''

DONE = r'''
with zipfile.ZipFile(WORKDIR / "submission.zip") as z:
    for n in z.namelist():
        d = pd.read_csv(z.open(n), sep=" ", header=None)
        print(f"{n:45s} {len(d):5d} rows, score mean {d[1].mean():+.3f}")
print("unreadable media:", {k: len(set(v)) for k, v in BROKEN.items()})
shutil.rmtree(WORKDIR / "raw", ignore_errors=True); shutil.rmtree(WORKDIR / "lab_sp", ignore_errors=True)
print("\nSUBMIT /kaggle/working/submission.zip (v11). Dev pass marks: English rank-corr >= 0.999 vs the submitted v11; Bangla >= 0.97 "
      "(GPU-retrained head). Keep this output: its v11_head/ and e2e_<split>/ are inputs of the Evaluation run and of FLAG_51-53.")
'''


def lib_file(rel):
    """= build_best_notebook.lib_file with a valid variable name for paths holding '-' (Experiment/BIGB-01/...)."""
    import re
    return B30.lib_file(rel).replace("SRC_" + rel.replace("/", "_").replace(".", "_").upper(),
                                     "SRC_" + re.sub(r"\W", "_", rel).upper())


def refs_cell():
    lines = ["\n# ============================================================================ the submitted zips (dev references)",
             "import base64, hashlib", "REF_ZIPS = {}"]
    for name, p in REFS.items():
        b = base64.b64encode(p.read_bytes()).decode()
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        lines.append(f"_b = (" + "".join(f'\n    "{b[i:i + 120]}"' for i in range(0, len(b), 120)) + ")")
        lines.append(f"REF_ZIPS[{name!r}] = WORKDIR / 'repo/reference/{p.name}'; REF_ZIPS[{name!r}].parent.mkdir(parents=True, exist_ok=True)")
        lines.append(f"REF_ZIPS[{name!r}].write_bytes(base64.b64decode(_b)); assert hashlib.md5(REF_ZIPS[{name!r}].read_bytes()).hexdigest() == {md5!r}")
    lines.append("print('references:', {k: v.name for k, v in REF_ZIPS.items()})")
    return "\n".join(lines) + "\n"


def head_blob():
    b = io.BytesIO(); md5 = {}
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in HEAD_CODE:
            data = (R / f).read_bytes(); z.writestr(f, data); md5[f] = hashlib.md5(data).hexdigest()
        z.writestr("head_entry.py", HEAD_ENTRY); md5["head_entry.py"] = hashlib.md5(HEAD_ENTRY.encode()).hexdigest()
    return base64.b64encode(b.getvalue()).decode(), md5


def build(path="FLAG_50_e2e_v11.ipynb"):
    blob, md5 = head_blob()
    hj = HEADJOB.replace("__HEAD_B64__", blob).replace("__HEAD_MD5__", json.dumps(md5, indent=1)).replace(
        "__DATA_DIRS__", json.dumps(L.DATA_DIRS))
    libs = ["kaggle/flag_lib.py", "kaggle/flag_v2.py", "BEST/v6_eval/pipeline.py", "BEST/v7_safe/safe_lib.py",
            "Experiment/E2E/system_v11.py", "BEST/v7_safe/cafuse_fuser.json", "BEST/v7_safe/fuser_en.json",
            "Experiment/BIGB-01/cafuse_t3_fuser_v11.json", "Experiment/EN-T3-01/fuser_en_t3.json"]
    cells = [("markdown", INTRO), ("code", ENV), ("code", E.PIP), ("code", PIP_T3),
             ("markdown", "## 0. v11's T3 head: background job on GPU 1 (trained once, exported, reused by the Evaluation run)"), ("code", hj),
             ("markdown", "## 1. FLAG_12: find the data (row order = the organiser .txt), resume from an earlier output"), ("code", DATA12),
             ("markdown", "## 2. Organiser csv + lists for the pipeline"), ("code", E.ORGDATA),
             ("markdown", "## 3. FLAG_12: audio loading, batching, helpers"), ("code", B12.AUDIO12),
             ("markdown", "## 4. Encoders: FLAG_12, FaRL + AdaFace (FLAG_30), the 4 T3 encoders (FLAG_40)"),
             ("code", B12.ENC12), ("code", B30.FACE2), ("code", KEEP), ("code", B40.CELL), ("code", DISPATCH),
             ("markdown", "## 5. Extract: w2v2 + T3 first (then the head job scores the test files), then the rest"), ("code", RUN_A), ("code", RUN_B),
             ("markdown", "## 6. Checks: attached reference arrays (FLAG_12) + T3 drift"), ("code", B12.CHECK12), ("code", T3CHECK),
             ("markdown", "## 7. FLAG_14: AudioCLIP (resampler pinned to polyphase, log 133)"),
             ("code", E.AV), ("code", B14.AUDIO[len(B12.AUDIO12):]), ("code", B14.ENC), ("code", B14.RUN), ("code", E.AV_END),
             ("markdown", "## 8. Libraries and fusers, verbatim, written to `repo/` with the repository layout"),
             *[("code", lib_file(f)) for f in libs], ("code", refs_cell()),
             ("markdown", "## 9. System v11: components, v9 / v11 scores, zips, `e2e_<split>/`"), ("code", SYSTEM),
             ("markdown", "## 10. Result"), ("code", DONE)]
    E.nb(cells, path)
    print(f"{path}: head job embeds {len(md5)} files ({len(blob) / 1e3:.0f} kB base64)")


if __name__ == "__main__":
    build()
