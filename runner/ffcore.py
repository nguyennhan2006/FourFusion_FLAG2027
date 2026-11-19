"""FourFusion FLAG 2027 - runner core for the 13.54 reproduction chain.

Derived from release/flag2027_e2e/orch.py (team repository; sha256 459eca0086ed5101...): same idea, extended with a data map.
  * Every original Kaggle run is one STAGE: stages/<group>/<id>/stage.json + stage.ipy (the original code cells, verbatim, with the
    patches listed in stage.json already applied, separated by CELL_SEP).
  * A stage runs in its own process (IPython InteractiveShell, so `!pip`, `%%writefile` keep working). It sees a private
    /kaggle/input (VIN: real directories + file links, Kaggle layout datasets/<owner>/<slug>, notebooks/<owner>/<slug>, or the
    'mount' the original run saw) and a private /kaggle/working (RUN). The two Kaggle path constants in the code are rewritten
    to those folders at run time; nothing else in the original cells changes, except the documented run-time rewrites below.
  * What a stage leaves in RUN becomes an ARTIFACT: FF_WORK/art/<kernel id>/ (+ _DONE.json) and its 'exports' (re-packed under
    the original Kaggle dataset ids). Artifact ids are the original Kaggle ids (configs/data_map.json).
  * Where an artifact comes from: made by this runner (FF_WORK/art) > saved copy under FF_DATA (configs/data_map.json; allowed or
    not by the artifact class, FF_MODE and FF_USE_SAVED: configs/modes.json) > Kaggle dataset / session output attached by hand
    (matched by slug, any owner) > run the stage that produces it.
  * Before every cell the runner prints `[cell] <stage> <k>/<n>: <first line of the cell>` (heartbeat rule) without touching
    the cell.
  * Run-time rewrites (logged with counts in the stage log and in _DONE.json): /kaggle/input and /kaggle/working -> VIN / RUN;
    configs/pipeline_1354.json 'runtime_rewrites' (dead ImageBind URL -> pinned HF copy with the same SHA-256); upload variables
    (os.environ['RFS_UPLOAD'] = '...' etc.) -> `pass  # disabled`; with FF_PIP=pinned the unpinned ImageBind git URL -> pinned
    commit. The 13.54 variant guard (configs/variant_1354.json) refuses forbidden artifacts / environment values. Every run-time
    change applied to all stages is listed in runner/RUNTIME_CHANGES.md.
  * Embedded code (base64 payloads decoded at run time, runner/payloads.py) cannot be rewritten: a module default
    os.environ.get("X", "/kaggle/input") gets X=<input view> exported; any other literal /kaggle/input makes the stage 'literal'
    (declared in stage.json runner_requirements, or detected automatically): Linux points /kaggle/input at the stage's input
    view, Kaggle needs a session holding exactly the stage's inputs, Windows refuses the stage.
  * lora_C1.pt guard: artifacts taken from a saved / attached copy, and every artifact in modes infer / assemble, must carry the
    ORIGINAL file (sha256 b97372e3...). In FF_MODE=train the LoRA is retrained by t163l_orig (not bit-identical): there the
    guard checks lineage instead (the root stage's dry check = 210 clips / 70 persons / all MAV-Celeb v1, the new hash recorded
    in _DONE.json, every downstream lora_C1.pt equal to a recorded hash). Smoke artifacts are only reported.

Configuration (environment variables, or keyword arguments of Config):
  FF_REPO   repository root (default: the parent of runner/)
  FF_DATA   data roots (= the Google Drive folder FourFusion_FLAG2027_data/), several separated by os.pathsep; 'auto' (default on
            Kaggle when unset) = every attached folder under FF_KIN that holds 00_raw/ ... 06_pretrained/ (Drive layout)
  FF_NO_DEV 1 = never mount the organisers' dev files (configs/data_map.json 'dev_files', e.g. dev_set.zip of raw-v4): the
            smoke-without-dev test of DESIGN G11
  FF_CACHE_LINK auto | symlink | copy: how the model cache (FF_MODELS) is seeded into FF_WORK/model_cache (auto = symlink on
            Kaggle, where the inputs are read-only, copy elsewhere so that a stage cannot write into FF_DATA; never hard links)
  FF_PIP_SCOPE all | v173 (FF_PIP=pinned only for the stages whose original run used the November image family, see
            configs/env_per_stage.json; other stages keep 'original')
  FF_LITERAL_STRICT 1 = on Kaggle, refuse a literal stage when the session holds attachments that are not its inputs
  FF_LITERAL_HONOUR_FALSE 0 = ignore an explicit runner_requirements.literal_kaggle_input=false (+ why) and fall back to detection
  FF_WORK   writable work folder (default ./ff_work; on Kaggle /kaggle/working/ff_work)
  FF_MODE   train | infer | assemble (default infer)
  FF_TARGET stage id(s) or artifact id(s), comma separated (default final_verify)
  FF_ONLY   run only these stages in this session (others are reported 'not_selected')
  FF_FORCE  stage ids / artifact ids to recompute even when a copy exists
  FF_SMOKE  1 = use each stage's smoke_env, GPU stages may run on CPU, estimated time x 0.1
  FF_USE_SAVED all | collected | none (default from configs/modes.json: train=collected, infer=all, assemble=all)
  FF_TIME_BUDGET_H stop accepting new stages when the session has used this many hours (default 11.3 on Kaggle, else 0 = none)
  FF_VARIANT 1354 (the only variant)
  FF_PIP    original | pinned | skip
  FF_MODELS folder with the pretrained models (layout of FF_DATA/06_pretrained), 'auto' (default: use pretrained/models when
            found) or 'off'
  FF_KIN    attached Kaggle inputs (default /kaggle/input)
  FF_TMP    scratch for VIN / RUN (default /kaggle/tmp or /tmp on Kaggle, FF_WORK/tmp elsewhere)
  FF_KEEP_VIN 1 = keep the VIN folder of finished stages (debug)
Secrets (HF_TOKEN) are read from the environment or the Kaggle secret store, passed only to stages that declare them in
compute.secrets or name them in their code, and never printed or written. Kaggle session tokens (KAGGLE_*TOKEN*, the secret
store and integration tokens) are never passed to a stage."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("FF_REPO") or HERE.parent).resolve()
CELL_SEP = "\n# <<<FF-CELL>>>\n"
SECRET_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
NEVER_PASS = ("KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR", "KAGGLE_USER_SECRETS_TOKEN", "KAGGLE_IAP_TOKEN",
              "KAGGLE_KERNEL_INTEGRATIONS", "KAGGLE_DATA_PROXY_TOKEN")
NEVER_PASS_RX = re.compile(r"^KAGGLE_.*(TOKEN|SECRET|KEY|PASSWORD|INTEGRATION)", re.I)   # any other Kaggle session credential
DATA_LAYOUT_DIRS = ("00_raw", "01_collected", "02_intermediate", "03_checkpoints", "04_run_outputs", "05_scores", "06_pretrained")
KINDS = ("raw", "collect", "data", "train", "infer", "assemble")
RUNNER_MARKERS = {"_DONE.json", "_FF_UNPACKED.json", "_FF_SEEDED"}


class StageBlocked(Exception):
    pass


class VariantError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------- helpers
def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jdump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def split_list(v):
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if str(x).strip()]
    return [x.strip() for x in re.split(r"[,\s]+", v or "") if x.strip()]


def split_paths(v):
    """os.pathsep-separated list; ';' also accepted on POSIX (':' is never a separator on Windows, drive letters)."""
    if not v:
        return []
    if isinstance(v, (list, tuple)):
        return [Path(x) for x in v if str(x).strip()]
    parts = re.split(r";" if os.name == "nt" else r"[:;]", str(v))
    return [Path(p.strip()) for p in parts if p.strip()]


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def on_kaggle():
    return Path("/kaggle/working").is_dir() and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")
                                                      or Path("/kaggle/input").is_dir())


def has_gpu():
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def safe_name(aid):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", aid)


def load_kaggle_secrets(names=SECRET_NAMES[:1]):
    """Kaggle secret store -> environment (never printed). No-op outside Kaggle or when already set."""
    got = []
    for k in names:
        if os.environ.get(k):
            continue
        try:
            from kaggle_secrets import UserSecretsClient  # noqa: E402  (only on Kaggle)
            v = UserSecretsClient().get_secret(k)
            if v:
                os.environ[k] = v
                got.append(k)
        except Exception:
            pass
    return got


def autodetect_data_roots(kin, depth=4):
    """Kaggle: attached folders that hold the Drive layout (00_raw/ ... 06_pretrained/), up to datasets/<owner>/<slug>/<sub>."""
    kin = Path(kin)
    if not kin.is_dir():
        return []
    found, frontier = [], [kin]
    for _ in range(depth):
        nxt = []
        for d in frontier:
            try:
                kids = sorted(p for p in d.iterdir() if p.is_dir())
            except OSError:
                continue
            for k in kids:
                if any((k / x).is_dir() for x in DATA_LAYOUT_DIRS):
                    found.append(k.resolve())
                else:
                    nxt.append(k)
        frontier = nxt
    return found


def mirror(src, dst, exclude=(), exclude_names=(), prefer=("symlink", "hardlink", "copy")):
    """Directory tree as real directories + file symlinks (os.walk / rglob never follow directory symlinks); falls back to hard
    links, then copies, where links are not allowed (Windows without developer mode, exFAT). `exclude` = top-level names left out
    (configs/data_map.json 'exclude', e.g. a zips/ folder kept beside extracted data); `exclude_names` = file names left out at
    any depth (FF_NO_DEV: the organisers' dev files). Returns (files, how).
    Caution: a link points at the source; a stage that opens an EXISTING mirrored file for writing writes into the source
    (FF_DATA). Kaggle inputs are read-only (the write fails); elsewhere pass prefer=("copy",) where that matters (model cache)."""
    src, dst = Path(src), Path(dst)
    n, how = 0, set()
    for root, dirs, files in os.walk(src, followlinks=True):
        r = Path(root)
        if r == src and exclude:
            dirs[:] = [x for x in dirs if x not in exclude]
            files = [x for x in files if x not in exclude]
        if exclude_names:
            files = [x for x in files if x not in exclude_names]
        d = dst / r.relative_to(src)
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            if r == src and f in RUNNER_MARKERS:                 # the runner's own markers are not part of an artifact
                continue
            t = d / f
            if os.path.lexists(t):
                continue
            how.add(link_or_copy(r / f, t, prefer=prefer))
            n += 1
    return n, ",".join(sorted(how)) or "-"


def link_or_copy(s, t, prefer=("symlink", "hardlink", "copy")):
    s, t = Path(s), Path(t)
    t.parent.mkdir(parents=True, exist_ok=True)
    for m in prefer:
        try:
            if m == "symlink":
                os.symlink(s.resolve(), t)
            elif m == "hardlink":
                os.link(s, t)
            else:
                shutil.copy2(s, t)
            return m
        except OSError:
            continue
    raise OSError(f"cannot link or copy {s} -> {t}")


def first_line(src, width=110):
    for line in src.splitlines():
        if line.strip():
            return line.strip()[:width]
    return ""


def cell_title(src):
    """As release/flag2027_e2e/build_notebook.cell_title: the [cell] / [boot] name of the first line, else the first line."""
    first = src.lstrip().splitlines()[0] if src.strip() else ""
    m = re.match(r'print\("\[(?:cell|boot)\] ?([^"]*)"', first)
    return (m.group(1) if m else first)[:90]


# ---------------------------------------------------------------------------------------------------------------- config
class Config:
    def __init__(self, **over):
        e = os.environ
        g = lambda k, d=None: over.get(k.lower()[3:]) if over.get(k.lower()[3:]) is not None else e.get(k, d)
        self.repo = Path(over.get("repo") or REPO).resolve()
        kag = on_kaggle()
        self.kaggle = kag
        fd = (g("FF_DATA", "") or "").strip()
        self.data_auto = fd.lower() == "auto" or (kag and not fd)
        self.data_roots = [p.resolve() for p in split_paths(fd)] if fd.lower() != "auto" else []
        if self.data_auto:                                  # Kaggle: attached Drive-layout folders (00_raw/ ... 06_pretrained/)
            self.data_roots += [p for p in autodetect_data_roots(Path(g("FF_KIN") or "/kaggle/input")) if p not in self.data_roots]
        self.no_dev = str(g("FF_NO_DEV") or "0").lower() in ("1", "true", "yes")
        self.cache_link = (g("FF_CACHE_LINK") or "auto").strip().lower()
        self.pip_scope = (g("FF_PIP_SCOPE") or "all").strip().lower()
        self.work = Path(g("FF_WORK") or ("/kaggle/working/ff_work" if kag else "./ff_work")).resolve()
        self.mode = (g("FF_MODE") or "infer").strip()
        self.target = split_list(g("FF_TARGET") or "")
        self.only = set(split_list(g("FF_ONLY") or ""))
        self.force = set(split_list(g("FF_FORCE") or ""))
        sm = over.get("smoke")
        self.smoke = bool(sm) if sm is not None else str(e.get("FF_SMOKE", "0")).lower() in ("1", "true", "yes")
        self.use_saved = (g("FF_USE_SAVED") or "").strip() or None
        tb = g("FF_TIME_BUDGET_H")
        self.time_budget_h = float(tb) if tb not in (None, "") else (11.3 if kag else 0.0)
        self.variant = (g("FF_VARIANT") or "1354").strip()
        self.pip = (g("FF_PIP") or "original").strip()
        self.models = (g("FF_MODELS") or "auto").strip()
        self.kin = Path(g("FF_KIN") or "/kaggle/input")
        tmp = g("FF_TMP")
        if not tmp:
            tmp = ("/kaggle/tmp" if Path("/kaggle/tmp").is_dir() else "/tmp") if kag else str(self.work / "tmp")
        self.tmp = Path(tmp).resolve()
        self.keep_vin = str(e.get("FF_KEEP_VIN", "0")) == "1"
        self.assume_saved = bool(over.get("assume_saved", False))
        if self.mode not in ("train", "infer", "assemble"):
            raise SystemExit(f"FF_MODE must be train | infer | assemble, not {self.mode!r}")
        if self.pip not in ("original", "pinned", "skip"):
            raise SystemExit(f"FF_PIP must be original | pinned | skip, not {self.pip!r}")
        if self.cache_link not in ("auto", "symlink", "copy"):
            raise SystemExit(f"FF_CACHE_LINK must be auto | symlink | copy, not {self.cache_link!r}")
        if self.pip_scope not in ("all", "v173"):
            raise SystemExit(f"FF_PIP_SCOPE must be all | v173, not {self.pip_scope!r}")
        if self.variant != "1354":
            raise SystemExit(f"FF_VARIANT={self.variant!r}: this repository only reproduces the 13.54 variant")

    def describe(self):
        return {"repo": str(self.repo), "data": [str(p) for p in self.data_roots], "work": str(self.work), "mode": self.mode,
                "target": self.target, "only": sorted(self.only), "force": sorted(self.force), "smoke": self.smoke,
                "use_saved": self.use_saved, "time_budget_h": self.time_budget_h, "variant": self.variant, "pip": self.pip,
                "models": self.models, "kin": str(self.kin), "tmp": str(self.tmp), "kaggle": self.kaggle, "data_auto": self.data_auto,
                "no_dev": self.no_dev, "cache_link": self.cache_link, "pip_scope": self.pip_scope,
                "literal_strict": os.environ.get("FF_LITERAL_STRICT", "0") == "1"}


# ---------------------------------------------------------------------------------------------------------------- stage
class Stage:
    """Normalised view of one stage.json (DESIGN 2.2), or of a skeleton entry of configs/pipeline_1354.json."""

    def __init__(self, d, sdir=None, skeleton=False):
        self.raw = d
        self.dir = Path(sdir) if sdir else None
        self.skeleton = skeleton
        self.id = d["id"]
        self.group = d.get("group", "")
        self.title = d.get("title", "")
        self.kind = d.get("kind", "data")
        self.role = d.get("role_in_1354", d.get("role", "score"))
        c = d.get("compute", {}) or {}
        dev = c.get("device") or d.get("device") or ("gpu" if d.get("gpu") else "cpu")
        self.gpu = str(dev).lower() == "gpu"
        self.gpus = c.get("gpus", d.get("gpus", 1 if self.gpu else 0))
        m = c.get("minutes_original", c.get("minutes", d.get("minutes_original", d.get("minutes"))))
        self.minutes = float(m) if isinstance(m, (int, float)) else 0.0
        self.internet = c.get("internet", d.get("internet"))
        self.secrets = list(c.get("secrets", d.get("secrets", [])) or [])
        self.inputs = []                                    # [(mount, artifact id)]
        self.input_exclude = {}                             # mount -> top-level names left out of that input (stage.json per-input 'exclude')
        for e in d.get("inputs", []) or []:
            if isinstance(e, str):
                self.inputs.append((e, e))
            elif isinstance(e, (list, tuple)):
                self.inputs.append((e[0], e[1]))
            else:
                aid = e["artifact"]
                self.inputs.append((e.get("mount") or aid, aid))
                if e.get("exclude"):                        # e.g. raw-v4 ["dev_set.zip", "dev_set"] (dev-exclusion, INTEGRATION_r1 M6)
                    self.input_exclude[e.get("mount") or aid] = [str(x) for x in e["exclude"]]
        o = d.get("outputs", {}) or {}
        self.kernel = o.get("kernel") or d.get("kernel") or f"notebooks/ff/{self.id}"
        ex = o.get("exports", d.get("exports", {})) or {}
        if isinstance(ex, list):                             # skeleton: list of artifact ids (content not specified)
            ex = {x: None for x in ex}
        self.exports = ex
        self.env = {k: str(v) for k, v in (d.get("env") or {}).items() if not k.startswith("_") and v is not None}
        self.smoke_env = {k: str(v) for k, v in (d.get("smoke_env") or {}).items() if not k.startswith("_") and v is not None}
        self.patches = d.get("patches") or []
        self.drop_cells = d.get("drop_cells") or []
        self.neutralize_upload = list(d.get("neutralize_upload") or [])
        self.pip_policy = d.get("pip_policy")
        self.variant = d.get("variant") or {}
        self.skip_with = list(d.get("skip_with") or [])
        self.verify = list(d.get("verify") or [])
        self.post_checks = list(d.get("post_checks") or [])
        self.scratch = list(d.get("scratch") or [])
        self.origin = d.get("origin") or {}
        code = d.get("code", "stage.ipy")
        self.code_name = code if isinstance(code, str) else ((code or {}).get("stage_ipy") or "stage.ipy")
        self.command = d.get("command")                     # optional: argv list run instead of stage.ipy (script-only stage)
        self.force_only = bool(d.get("force_only"))           # runs only when named in FF_FORCE
        self.saved_only = {e["artifact"] for e in d.get("inputs", []) or [] if isinstance(e, dict) and e.get("resolve") == "saved_only"}
        self.verifies = list((d.get("outputs") or {}).get("verifies") or [])
        rr = d.get("runner_requirements") or {}
        self.literal_kaggle_input = bool(rr.get("literal_kaggle_input"))
        # Explicit, reasoned 'false' (INTEGRATION_r1 M1): the stage owner checked that every literal /kaggle/input of the decoded
        # code is unreachable or redirected (e.g. the RFS_XDIRS path patch of rfs_train._xload); honoured only with a non-empty 'why'.
        self.literal_declared_false = ("literal_kaggle_input" in rr and rr.get("literal_kaggle_input") is False
                                       and bool(str(rr.get("why") or "").strip()))
        self.literal_why = str(rr.get("why") or "")
        self.kaggle_strict = bool(rr.get("kaggle_strict"))                # refuse extra Kaggle attachments for this stage
        v = self.variant if isinstance(self.variant, dict) else {}
        self.require_files = list(v.get("require_files") or [])
        fb = list(v.get("forbid_artifacts") or []) + list(d.get("forbidden_inputs") or [])
        self.forbid = [x if isinstance(x, str) else x.get("artifact") for x in fb if x]

    @property
    def outputs(self):
        return [self.kernel] + list(self.exports)

    @property
    def code_path(self):
        return (self.dir / self.code_name) if self.dir else None

    def runnable(self):
        if self.skeleton or self.dir is None:
            return False
        return (self.code_path is not None and self.code_path.exists()) or bool(self.command)

    def command_cell(self, ctx):
        """The one cell run for a script-only stage: stage.json "command" (argv list; placeholders {python} {stage_dir} {vin}
        {run} {repo} {data} {work})."""
        argv = self.command if isinstance(self.command, list) else str(self.command).split()
        argv = [str(x).format(**ctx) for x in argv]
        return (f'print("[cell] {self.id}: command", flush=True)\nimport subprocess\ncmd = {argv!r}\n'
                f'print("[ff] run:", " ".join(cmd), flush=True)\nrc = subprocess.run(cmd).returncode\n'
                f'assert rc == 0, f"command failed with exit code {{rc}}"\n')


# ---------------------------------------------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, cfg: Config | None = None, quiet=False):
        self.cfg = cfg or Config()
        r = self.cfg.repo
        self.pipeline = jload(r / "configs/pipeline_1354.json")
        self.dmap = jload(r / "configs/data_map.json")["artifacts"]
        self.modes = jload(r / "configs/modes.json")
        self.var = jload(r / "configs/variant_1354.json")
        try:
            self.envtab = jload(r / "configs/env_per_stage.json").get("stages", {})
        except Exception:
            self.envtab = {}
        self.mode = self.modes["modes"][self.cfg.mode]
        self.use_saved = self.cfg.use_saved or self.mode["use_saved_default"]
        if self.use_saved not in ("all", "collected", "none"):
            raise SystemExit(f"FF_USE_SAVED must be all | collected | none, not {self.use_saved!r}")
        self.alias = {a: k for k, v in self.dmap.items() for a in v.get("aliases", [])}
        self.warnings = []
        self.S, self.stage_files = {}, {}
        self._load_stages()
        self.order_ref = [s["id"] for s in self.pipeline["stages"]]
        self.producers = {}
        for st in self.S.values():
            for a in st.outputs:
                self.producers.setdefault(a, []).append(st.id)
        self.made_now = set()
        self._lit = {}                                       # stage id -> payloads.scan_stage_code result (cached)
        self._in_lora = {}                                   # stage id -> {sha256 of lora_C1.pt files in its inputs}
        self._env_defaults = []                              # env names exported for embedded module defaults (last stage)
        self._sw_logged = set()
        self.T0 = time.time()
        self.has_gpu = None
        self.state_file = self.cfg.work / "state.json"
        self.state = {"sessions": [], "stages": {}}
        if self.state_file.exists():
            try:
                self.state = jload(self.state_file)
            except Exception:
                pass
        self.quiet = quiet

    # ------------------------------------------------------------------ loading
    def _load_stages(self):
        roots = [self.cfg.repo / p for p in self.pipeline.get("stage_roots", ["stages"])]
        for root in roots:
            if not root.is_dir():
                continue
            for sj in sorted(root.glob("*/*/stage.json")):
                try:
                    d = jload(sj)
                except Exception as ex:
                    self.warnings.append(f"cannot read {sj}: {ex}")
                    continue
                if "id" not in d:
                    self.warnings.append(f"{sj}: no 'id'")
                    continue
                if d["id"] in self.S:
                    self.warnings.append(f"duplicate stage id {d['id']}: {sj} and {self.stage_files[d['id']]}")
                    continue
                if d["id"] != sj.parent.name:
                    self.warnings.append(f"{sj}: id {d['id']!r} differs from its folder name {sj.parent.name!r}")
                self.S[d["id"]] = Stage(d, sj.parent)
                self.stage_files[d["id"]] = sj
        for d in self.pipeline.get("stages", []):              # skeleton for stages without stage.json yet
            if d["id"] not in self.S:
                self.S[d["id"]] = Stage(d, None, skeleton=True)

    # ------------------------------------------------------------------ artifacts
    def entry(self, aid):
        aid = self.alias.get(aid, aid)
        return self.dmap.get(aid)

    def canonical(self, aid):
        return self.alias.get(aid, aid)

    def cls(self, aid):
        e = self.entry(aid)
        if e and e.get("class"):
            return e["class"]
        return "run_output" if aid.startswith("notebooks/") else "intermediate"

    def forbidden(self, aid):
        fa = self.var.get("forbidden_artifacts", {})
        return aid in fa or self.canonical(aid) in fa or self.cls(aid) == "forbidden"

    def producer_of(self, aid):
        cands = self.producers.get(aid) or self.producers.get(self.canonical(aid)) or []
        if not cands:
            for a in [k for k, v in self.alias.items() if v == self.canonical(aid)] + [self.canonical(aid)]:
                cands = self.producers.get(a) or []
                if cands:
                    break
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        pref = self.pipeline.get("producer_preference", {}).get(self.canonical(aid), {})
        p = pref.get(self.cfg.mode) or pref.get("default")
        if p in cands:
            return p
        real = [c for c in cands if not self.S[c].skeleton]
        return (real or cands)[0]

    def is_forced(self, aid):
        f = self.cfg.force
        return aid in f or self.canonical(aid) in f or (self.producer_of(aid) in f)

    def saved_allowed(self, aid):
        if self.forbidden(aid):
            return False
        if self.is_forced(aid):
            return False
        if self.producer_of(aid) is None:
            return True                                        # nothing could rebuild it: must come from outside
        if self.canonical(aid) in self.mode.get("never_saved", []):
            return False
        c = self.cls(aid)
        if c in self.modes.get("always_saved_classes", []):
            return True
        e = self.entry(aid) or {}
        if e.get("default_saved") and self.mode.get("respect_default_saved", {}).get(self.use_saved, True):
            return True
        return c in self.mode["saved_classes"][self.use_saved]

    def _done_ok(self, d):
        """An earlier runner artifact: usable unless it was made in SMOKE mode (and this run is not smoke) or for another variant."""
        try:
            info = jload(d / "_DONE.json")
        except Exception:
            return False
        if info.get("smoke") and not self.cfg.smoke:
            return False
        return info.get("variant", "1354") == self.cfg.variant

    def art_dir(self, aid):
        """Where the runner keeps an artifact it made: FF_WORK/art/<canonical id> (aliases of configs/data_map.json resolved)."""
        return self.cfg.work / "art" / self.canonical(aid)

    def _unpack(self, aid, zp):
        dst = self.cfg.work / "unpacked" / safe_name(self.canonical(aid))
        mark = dst / "_FF_UNPACKED.json"
        st = zp.stat()
        if mark.exists():
            try:
                m = jload(mark)
                if m.get("zip") == str(zp) and m.get("bytes") == st.st_size:
                    return dst
            except Exception:
                pass
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        self.log(f"  unpack {zp} -> {dst}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(dst)
        mark.write_text(json.dumps({"zip": str(zp), "bytes": st.st_size, "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
        return dst

    def saved_candidates(self, aid):
        """Paths under FF_DATA where a saved copy of aid may be, in search order (see configs/data_map.json _doc)."""
        e = self.entry(aid) or {}
        ids = [self.canonical(aid)] + [a for a, c in self.alias.items() if c == self.canonical(aid)]
        rels = []
        if e.get("path"):
            rels.append(e["path"])
        rels += e.get("alt_paths", [])
        out = []
        for root in self.cfg.data_roots:
            for rel in rels:
                out.append(root / rel)
            for i in ids:
                parts = i.split("/")
                out.append(root / i)                                   # Kaggle layout
                out.append(root / "art" / i)                           # an earlier FF_WORK
                if len(parts) == 3 and parts[0] == "datasets":         # team package layout
                    out += [root / "datasets" / f"{parts[1]}__{parts[2]}", root / "datasets" / f"{parts[1]}__{parts[2]}.zip"]
                if len(parts) == 3 and parts[0] == "notebooks":
                    out.append(root / "outputs" / parts[2])
        if e.get("e_package"):
            for root in self.cfg.data_roots:
                out.append(root / e["e_package"].rstrip("/"))
        return out

    def find_saved(self, aid):
        req = list((self.entry(aid) or {}).get("required_files") or [])     # names that must exist in a saved folder (top level)
        for p in self.saved_candidates(aid):
            if p.is_dir():
                if (p.parent.name == "art" or "/art/" in p.as_posix()) and (p / "_DONE.json").exists() and not self._done_ok(p):
                    continue
                try:
                    next(p.iterdir())
                except StopIteration:
                    continue                                         # empty folder = not there
                if req and not all((p / f).exists() for f in req):
                    self.log(f"  [saved] {aid}: {p} lacks {[f for f in req if not (p / f).exists()]} (data_map required_files): not used")
                    continue
                return p
            if p.is_file():
                epath = (self.entry(aid) or {}).get("path") or ""
                if p.suffix == ".zip" and not epath.endswith(".zip") and p.parent.name == "datasets":
                    return self._unpack(aid, p)                      # team package: datasets/<owner>__<slug>.zip
                return p                                             # a file artifact (e.g. evaluation_phase.zip)
        return None

    def find_attached(self, aid):
        """Kaggle: a dataset / kernel output attached by hand, matched by slug (any owner), or an output of an earlier session of
        the orchestration notebook (its FF_WORK attached as input)."""
        kin = self.cfg.kin
        if not kin.is_dir():
            return None
        for i in [self.canonical(aid)] + [a for a, c in self.alias.items() if c == self.canonical(aid)]:
            parts = i.split("/")
            for pat in (f"notebooks/*/*/ff_work/art/{i}", f"datasets/*/*/ff_work/art/{i}", f"*/ff_work/art/{i}", f"*/art/{i}"):
                for h in sorted(kin.glob(pat)):
                    if (h / "_DONE.json").exists() and self._done_ok(h):
                        return h
            if len(parts) == 3:
                kind, owner, slug = parts
                cands = [kin / kind / owner / slug] + sorted((kin / kind).glob(f"*/{slug}")) + [kin / slug]
                for c in cands:
                    if c.is_dir():
                        return c
        return None

    def skip_with_view(self, st):
        """stage.json skip_with (FF_DATA-relative folders / files) = a saved copy of the stage's kernel output. Work packages use it
        two ways, both supported: alternatives (a run-output folder OR a flat score file) and parts (two flat files that together are
        the output). Rule: the first existing folder entry is the saved copy; otherwise every existing file entry is linked by name
        into FF_WORK/saved_views/<kernel>. A folder that is the data_map path of one of the stage's exports, or of an output of
        another stage, is NOT a saved copy of this kernel (it holds different content) and is skipped with a note."""
        not_kernel = {}
        for a in st.exports:
            pa = (self.entry(a) or {}).get("path")
            if pa:
                not_kernel[pa.rstrip("/")] = f"export {a} of this stage"
        for other in self.S.values():
            if other.id == st.id:
                continue
            for a in other.outputs:
                pa = (self.entry(a) or {}).get("path")
                if pa:
                    not_kernel.setdefault(pa.rstrip("/"), f"output {a} of stage {other.id}")
        own = ((self.entry(st.kernel) or {}).get("path") or "").rstrip("/")
        for root in self.cfg.data_roots:
            ps = []
            for s in st.skip_with:
                s2 = s.rstrip("/")
                if s2 in not_kernel and s2 != own:
                    key = (st.id, s2)
                    if key not in self._sw_logged:
                        self._sw_logged.add(key)
                        self.log(f"  [saved view] {st.id}: skip_with {s2} is the {not_kernel[s2]}, not this kernel's output: not used as the kernel")
                    continue
                ps.append(root / s2)
            dirs = [p for p in ps if p.is_dir() and any(p.iterdir())]
            if dirs:
                return dirs[0]
            files = [p for p in ps if p.is_file()]
            if not files:
                continue
            missing = [p.name for p in ps if not p.exists() and p.suffix]
            if missing:
                self.log(f"  [saved view] {st.id}: skip_with files not found under {root}: {missing}")
            view = self.cfg.work / "saved_views" / safe_name(st.kernel)
            shutil.rmtree(view, ignore_errors=True)
            view.mkdir(parents=True)
            for p in files:
                link_or_copy(p, view / p.name)
            return view
        return None

    def locate(self, aid, saved_only=False):
        """-> (path, how) with how in made | saved | attached; (None, None) when not available under the current policy.
        saved_only (stage input with "resolve": "saved_only"): only a saved / attached copy counts, whatever the policy."""
        if saved_only:
            q = self.find_saved(aid) or self.find_attached(aid)
            return (q, "saved") if q is not None else (None, None)
        p = self.art_dir(self.canonical(aid))
        if (p / "_DONE.json").exists() and self._done_ok(p):
            if not self.is_forced(aid) or self.canonical(aid) in self.made_now:
                return p, "made"
        if self.forbidden(aid):
            return None, None
        if not self.saved_allowed(aid):
            return None, None
        q = self.find_saved(aid)
        if q is not None:
            return q, "saved"
        prod = self.producer_of(aid)
        if prod and self.S[prod].skip_with and self.canonical(aid) == self.canonical(self.S[prod].kernel):
            q = self.skip_with_view(self.S[prod])
            if q is not None:
                return q, "saved"
        q = self.find_attached(aid)
        if q is not None:
            return q, "attached"
        return None, None

    def soft_input(self, st, aid):
        """configs/modes.json soft_inputs_for_kinds: in that mode an input of such a stage whose producer cannot run in the mode is
        mounted when available and otherwise skipped (the assembly stages fall back to FF_DATA/05_scores on their own)."""
        if st.kind not in self.mode.get("soft_inputs_for_kinds", []):
            return False
        prod = self.producer_of(aid)
        return prod is not None and not self.mode_runs(self.S[prod])

    def verifiers_of(self, aid):
        return [s.id for s in self.S.values() if aid in s.verifies or self.canonical(aid) in s.verifies]

    def available(self, aid):
        if self.locate(aid)[0] is not None:
            return True
        if self.cfg.assume_saved and self.saved_allowed(aid):
            e = self.entry(aid)
            if e and e.get("path") and e.get("class") != "forbidden":
                return True
        return False

    def where_hint(self, aid):
        e = self.entry(aid)
        if e and e.get("path"):
            return f"FF_DATA/{e['path']}"
        return "FF_DATA/<add a configs/data_map.json entry>"

    # ------------------------------------------------------------------ plan
    def mode_runs(self, st):
        if st.id in self.cfg.force:
            return True
        if st.force_only:
            return False
        if st.kind == "collect" and self.modes.get("collect_kind_runs_only_when_forced", True):
            return False
        if st.id in self.mode.get("allow_train_stages", []):
            return True
        return st.kind in self.mode["run_kinds"]

    def resolve_targets(self, targets=None):
        t = list(targets or getattr(self, "_targets", None) or self.cfg.target or [self.pipeline.get("default_target", "final_verify")])
        return t

    def make_plan(self, targets=None):
        if targets:
            self._targets = list(targets)
        order, blocked, seen, visiting = [], [], set(), set()
        self.soft_missing = []

        def visit(aid, via):
            if aid in seen:
                return
            seen.add(aid)
            if self.forbidden(aid):
                blocked.append((aid, via, "FORBIDDEN for the 13.54 variant (configs/variant_1354.json)"))
                return
            if self.available(aid):
                for v in self.verifiers_of(aid):                   # check stages (outputs.verifies), e.g. raw_btc
                    vs = self.S[v]
                    if v not in order and v not in visiting and self.mode_runs(vs) and not self.available(vs.kernel):
                        visit_stage(v, vs.kernel, f"verifies {aid}")
                return
            prod = self.producer_of(aid)
            if prod is None:
                blocked.append((aid, via, f"no producing stage: put it at {self.where_hint(aid)}"))
                return
            visit_stage(prod, aid, via)

        def visit_stage(sid, aid, via):
            st = self.S[sid]
            if sid in order:
                return
            if sid in visiting:
                blocked.append((aid, via, f"CYCLE through stage {sid}"))
                return
            if not self.mode_runs(st):
                why = ("collect stages run only when forced (FF_FORCE)" if st.kind == "collect" else
                       "runs only when forced (stage.json force_only)" if st.force_only else
                       f"FF_MODE={self.cfg.mode} does not run '{st.kind}' stages")
                blocked.append((aid, via, f"{why} ({sid}): provide the saved copy at {self.where_hint(aid)}"))
                return
            visiting.add(sid)
            for _m, inp in st.inputs:
                if inp in st.saved_only:
                    if self.locate(inp, saved_only=True)[0] is None and not (self.cfg.assume_saved and (self.entry(inp) or {}).get("path")):
                        blocked.append((inp, sid, f"saved copy required (resolve: saved_only): put it at {self.where_hint(inp)}"))
                    continue
                if self.soft_input(st, inp) and not self.available(inp):
                    self.soft_missing.append((inp, sid))
                    continue
                visit(inp, sid)
            visiting.discard(sid)
            order.append(sid)

        for t in self.resolve_targets(targets):
            if t in self.S:
                st = self.S[t]
                if st.id in self.cfg.force or not all(self.available(o) for o in st.outputs):
                    visit_stage(t, st.kernel, "TARGET")
            else:
                visit(t, "TARGET")
        return order, blocked

    def dag_check(self, compare_skeleton=False):
        """Whole-graph checks over every stage (stage.json, skeleton for the others)."""
        rep = {"stages": len(self.S), "with_stage_json": sum(not s.skeleton for s in self.S.values()), "cycles": [], "duplicates": {},
               "unresolved_inputs": [], "forbidden_inputs": [], "unknown_artifacts": [], "warnings": list(self.warnings), "skeleton_diff": [],
               "nonraw_without_producer": [], "default_saved_frozen": [], "unconsumed_data_map_ids": [], "embedded_data": [],
               "variant_errors": []}
        start_classes = set(self.modes.get("start_classes", ["raw", "collected", "pretrained"]))
        consumed = {}
        for st in self.S.values():
            for _m, a in st.inputs:
                consumed.setdefault(self.canonical(a), []).append(st.id)
        for a, users in sorted(consumed.items()):
            e = self.entry(a) or {}
            c = self.cls(a)
            if c in start_classes or c == "forbidden":
                continue
            if self.producer_of(a) is None:                    # requirement 1: every intermediate needs a producing stage
                rep["nonraw_without_producer"].append({"artifact": a, "class": c, "consumers": sorted(set(users)),
                                                       "origin_run": e.get("origin_run"), "gap": e.get("requirement1_gap")})
            elif e.get("default_saved") or c in self.modes.get("always_saved_classes", []):
                rep["default_saved_frozen"].append({"artifact": a, "class": c, "producer": self.producer_of(a), "consumers": sorted(set(users)),
                                                    "why": "default_saved" if e.get("default_saved") else f"class {c} is always taken saved",
                                                    "gap": e.get("requirement1_gap")})
        for a, e in self.dmap.items():
            if e.get("class") == "embedded":
                rep["embedded_data"].append({"artifact": a, "sha256": e.get("sha256"), "carried_by": e.get("carried_by", []),
                                             "producer": e.get("producer"), "gap": e.get("requirement1_gap")})
            elif a not in consumed and e.get("class") != "forbidden" and not e.get("unused"):
                rep["unconsumed_data_map_ids"].append({"artifact": a, "class": e.get("class"), "producer": e.get("producer")})
        for st in self.S.values():                              # static variant check (no run needed): stage.json fields + stage code
            if st.skeleton:
                continue
            try:
                code = st.code_path.read_text(encoding="utf-8") if st.code_path and st.code_path.exists() else None
            except Exception:
                code = None
            errs, _w = self.variant_check(st, code)
            for e_ in errs:
                rep["variant_errors"].append((st.id, e_))
        pref = self.pipeline.get("producer_preference", {})
        for a, ps in self.producers.items():
            if len(ps) > 1 and self.canonical(a) not in pref:
                rep["duplicates"][a] = ps
        edges = {s: set() for s in self.S}
        for st in self.S.values():
            for _m, a in st.inputs:
                if self.forbidden(a):
                    rep["forbidden_inputs"].append((st.id, a))
                p = self.producer_of(a)
                if p is None:
                    if self.entry(a) is None:
                        rep["unresolved_inputs"].append((st.id, a))
                elif p != st.id:
                    edges[st.id].add(p)
                else:
                    rep["cycles"].append([st.id, "(consumes its own output)"])
                if self.entry(a) is None and p is not None and not a.startswith("notebooks/"):
                    rep["unknown_artifacts"].append((st.id, a))
        indeg = {s: 0 for s in self.S}
        users = {s: set() for s in self.S}
        for s, ps in edges.items():
            for p in ps:
                indeg[s] += 1
                users[p].add(s)
        q = [s for s, d in indeg.items() if d == 0]
        seen = 0
        while q:
            s = q.pop()
            seen += 1
            for u in users[s]:
                indeg[u] -= 1
                if indeg[u] == 0:
                    q.append(u)
        if seen != len(self.S):
            rep["cycles"].append(sorted(s for s, d in indeg.items() if d > 0))
        if compare_skeleton:
            sk = {d["id"]: Stage(d, None, True) for d in self.pipeline.get("stages", [])}
            for sid, st in self.S.items():
                if st.skeleton or sid not in sk:
                    if sid not in sk:
                        rep["skeleton_diff"].append((sid, "stage.json id not in the skeleton (configs/pipeline_1354.json)"))
                    continue
                a, b = sk[sid], st
                ia, ib = {x for _m, x in a.inputs}, {x for _m, x in b.inputs}
                ia, ib = {self.canonical(x) for x in ia}, {self.canonical(x) for x in ib}
                if ia != ib:
                    rep["skeleton_diff"].append((sid, f"inputs: only skeleton {sorted(ia - ib)} | only stage.json {sorted(ib - ia)}"))
                oa, ob = {self.canonical(x) for x in a.outputs}, {self.canonical(x) for x in b.outputs}
                if oa != ob:
                    rep["skeleton_diff"].append((sid, f"outputs: only skeleton {sorted(oa - ob)} | only stage.json {sorted(ob - oa)}"))
                if a.kind != b.kind:
                    rep["skeleton_diff"].append((sid, f"kind: skeleton {a.kind} | stage.json {b.kind}"))
        return rep

    # ------------------------------------------------------------------ variant guard
    def variant_check(self, st, code=None):
        errs, warns = [], []
        for _m, a in st.inputs:
            if self.forbidden(a):
                errs.append(f"input {a} is forbidden: {self.var['forbidden_artifacts'].get(a, 'class forbidden')}")
            if a in st.forbid or self.canonical(a) in st.forbid:
                errs.append(f"input {a} is listed as forbidden by the stage itself")
        fe = self.var.get("forbidden_env", {})
        for env in (st.env, st.smoke_env, {k: os.environ[k] for k in fe if k in os.environ}):
            for k, bad in fe.items():
                if k in env and str(env[k]) in bad:
                    errs.append(f"environment {k}={env[k]} is forbidden for 13.54")
        lc = (st.variant or {}).get("lora_c1") if isinstance(st.variant, dict) else None
        ok_lc, why_lc = self.lora_c1_value_ok(lc)
        if not ok_lc:
            errs.append(f"stage.json variant lora_c1={lc!r}: {why_lc}")
        if code is not None:
            for p in self.var.get("forbidden_code_patterns", []):
                if re.search(p["regex"], code):
                    errs.append(f"code matches forbidden pattern {p['regex']!r}: {p['why']}")
            code_nc = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))     # comments may name them
            for s in self.var.get("warn_code_strings", []):
                if s in code_nc:
                    warns.append(f"code mentions {s!r} outside comments (check it is not used)")
            exp = self.var.get("stage_env_expectations", {}).get(st.id, {})
            for k, v in exp.items():
                in_env = st.env.get(k) == v
                q = r"['\"]" + re.escape(v) + r"['\"]"
                forms = (r"os\.environ\[['\"]" + re.escape(k) + r"['\"]\]\s*=\s*" + q,                  # os.environ['K'] = 'V'
                         r"os\.environ\.setdefault\(\s*['\"]" + re.escape(k) + r"['\"]\s*,\s*" + q,      # os.environ.setdefault('K', 'V')
                         r"[(,]\s*" + re.escape(k) + r"\s*=\s*" + q,                                    # dict(..., K="V") / env kwargs
                         r"['\"]" + re.escape(k) + r"['\"]\s*:\s*" + q)                                 # {"K": "V"} subprocess env
                if not in_env and not any(re.search(rx, code) for rx in forms):
                    warns.append(f"expected flag {k}={v} not found in stage env or code")
            for ce in self.var.get("stage_code_expectations", {}).get(st.id, []):
                if isinstance(ce, dict) and ce.get("regex") and not re.search(ce["regex"], code, re.M):
                    warns.append(f"expected code {ce['regex']!r} not found ({ce.get('why', '')})")
        return errs, warns

    def lora_c1_value_ok(self, lc):
        """stage.json variant.lora_c1 is free text in several packages ('original, through rfx_eval.zip of e163i', 'not_used (...)').
        Accepted: absent, a value starting with 'original', or 'not used' / 'not_used' / 'none'. Refused: anything naming a fix
        variant (configs/variant_1354.json stage_variant_field.refuse_words) or any other value."""
        if lc is None:
            return True, ""
        sv = self.var.get("stage_variant_field", {})
        s = str(lc).strip().lower()
        words = re.findall(r"[a-z0-9_]+", s)
        bad = [w for w in sv.get("refuse_words", ["fix", "fixed", "c1fix", "lorafix", "lorafix1", "lf", "16.18", "c1fix-01"]) if w in words or
               (w not in ("fix", "lf") and w in s)]
        if bad:
            return False, f"names a fix variant ({bad}); 13.54 needs the ORIGINAL LoRA C1"
        acc = tuple(sv.get("accept_prefixes", ["original", "not used", "not_used", "none"]))
        if s.startswith(acc):
            return True, ""
        return False, f"unrecognised value (accepted prefixes {list(acc)})"

    def fc_hits(self, root, fc):
        """Files of one artifact (folder or single file) matched by a file_checks glob ('**/lora_C1.pt' -> name lora_C1.pt)."""
        root = Path(root)
        pat = fc["glob"].split("/")[-1]
        if root.is_file():
            return [root] if fnmatch.fnmatch(root.name, pat) else []
        if not root.is_dir():
            return []
        return sorted(p for p in root.rglob(pat) if p.is_file())

    def lineage_hashes(self, fc):
        """sha256 values that a file of check `fc` MADE in this FF_WORK may carry besides the fixed one: those recorded in _DONE.json by
        a lineage root stage (fc.made_in_session.lineage_roots) whose root check passed (non-smoke runs only)."""
        ok = set()
        spec = fc.get("made_in_session") or {}
        for sid in spec.get("lineage_roots", {}):
            st = self.S.get(sid)
            if st is None:
                continue
            for a in st.outputs:
                try:
                    info = jload(self.art_dir(a) / "_DONE.json")
                except Exception:
                    continue
                if info.get("smoke") and not self.cfg.smoke:
                    continue
                lin = (info.get("variant_lineage") or {}).get(fc.get("id", fc["glob"])) or {}
                if lin.get("root_check") == "passed":
                    ok |= set(lin.get("sha256", []))
        return ok

    def lineage_policy(self):
        """'fixed' = every matched file must equal the fixed sha256 (artifacts from saved copies; modes infer / assemble);
        'lineage' = FF_MODE=train, where the lineage root retrains the file; smoke runs only report."""
        return "lineage" if self.cfg.mode == "train" else "fixed"

    def file_checks(self, root, when, how="saved", st=None):
        """configs/variant_1354.json file_checks (lora_C1.pt = original) on one artifact -> (problems, notes, hashes found).
        how = made | saved | attached (inputs) or 'made' (outputs). Rules (DESIGN 2.7, 5.1):
          * saved / attached copies, and every artifact in FF_MODE infer / assemble: the fixed sha256 (b97372e3... = ORIGINAL C1);
          * FF_MODE=train, made in this FF_WORK: the fixed sha256 or a hash recorded by a lineage root whose root check passed
            (outputs of the root itself are checked in run_stage, after its dry check);
          * smoke: made files are reported only (smoke weights are placeholders; smoke artifacts are never reused by real runs)."""
        bad, notes, found = [], [], {}
        for fc in self.var.get("file_checks", []):
            if when not in fc.get("when", ["inputs", "outputs"]):
                continue
            key = fc.get("id", fc["glob"])
            for p in self.fc_hits(root, fc):
                h = sha256_file(p)
                found.setdefault(key, set()).add(h)
                if h == fc["sha256"]:
                    continue
                msg = f"{p}: sha256 {h[:16]}... != fixed {fc['sha256'][:16]}... ({fc['why']})"
                if how == "made" and self.cfg.smoke:
                    notes.append("smoke artifact, reported only: " + msg)
                    continue
                if how == "made" and self.lineage_policy() == "lineage":
                    if h in self.lineage_hashes(fc):
                        notes.append(f"{p}: sha256 {h[:16]}... = hash recorded by the lineage root (FF_MODE=train, retrained LoRA C1)")
                        continue
                    if when == "outputs" and st is not None and st.id in (fc.get("made_in_session") or {}).get("lineage_roots", {}):
                        notes.append(f"{p}: sha256 {h[:16]}... new file of the lineage root {st.id} (checked by its root check)")
                        continue
                    if when == "outputs" and st is not None and h in self._in_lora.get(st.id, {}).get(key, set()):
                        notes.append(f"{p}: sha256 {h[:16]}... copied from an input of {st.id} (propagation)")
                        continue
                    bad.append(msg + "; not a hash recorded by a lineage root in this FF_WORK and not copied from an input")
                    continue
                bad.append(msg)
        return bad, notes, found

    def lineage_root_check(self, st, fc):
        """Root check of a lineage root stage (configs/variant_1354.json file_checks[].made_in_session.lineage_roots[stage]):
        a JSON file of its output must contain the expected values (t163l_orig: ff_lora_drycheck.json = 210 clips / 70 persons /
        all MAV-Celeb v1), and the forbidden environment values must be absent. -> (passed, message)."""
        spec = (fc.get("made_in_session") or {}).get("lineage_roots", {}).get(st.id) or {}
        f = None
        for a in st.outputs:
            q = self.art_dir(a) / spec.get("file", "")
            if spec.get("file") and q.is_file():
                f = q
                break
        if f is None:
            return False, f"root check file {spec.get('file')} not found in the outputs of {st.id} (the dry check did not run?)"
        try:
            got = jload(f)
        except Exception as ex:
            return False, f"root check file {f}: {type(ex).__name__}: {ex}"
        exp = spec.get("expect", {})
        if not isinstance(got, dict) or not all(got.get(k) == v for k, v in exp.items()):      # exact per top-level key
            return False, f"root check {f.name}: values differ from {exp!r}"
        for k, badv in self.var.get("forbidden_env", {}).items():
            if str(st.env.get(k, os.environ.get(k, ""))) in badv:
                return False, f"root check: environment {k} is forbidden"
        return True, f"root check {f.name} passed ({json.dumps(spec.get('expect'))[:120]})"

    # ------------------------------------------------------------------ logging / state
    def log(self, *a):
        if not self.quiet:
            print(*a, flush=True)

    def save_state(self):
        jdump(self.state_file, self.state)

    def elapsed_min(self):
        return (time.time() - self.T0) / 60

    # ------------------------------------------------------------------ run one stage
    RUNNER = r'''import sys, json, time, os
SEP = %r
stage, log = sys.argv[2], sys.argv[3]
cells = [c for c in open(sys.argv[1], encoding="utf-8").read().split(SEP) if c.strip()]
n = len(cells)


def first(c):
    for l in c.splitlines():
        if l.strip():
            return l.strip()[:110]
    return ""


def plain(c):
    """Fallback without IPython: !cmd, %%%%writefile, %%pip only."""
    lines = c.split("\n")
    if lines and lines[0].startswith("%%%%writefile"):
        a = lines[0].split()
        mode, path = ("a" if "-a" in a else "w"), a[-1]
        body = "\n".join(lines[1:]) + "\n"
        return ("import os\nos.makedirs(os.path.dirname(%%r) or '.', exist_ok=True)\nopen(%%r, %%r, encoding='utf-8').write(%%r)\n"
                %% (path, path, mode, body))
    out = []
    for l in lines:
        s = l.lstrip(); ind = l[:len(l) - len(s)]
        if s.startswith("!"):
            out.append(ind + "__import__('subprocess').run(%%r, shell=True)" %% s[1:])
        elif s.startswith("%%pip "):
            out.append(ind + "__import__('subprocess').run([__import__('sys').executable, '-m', 'pip'] + %%r)" %% s[5:].split())
        elif s.startswith("%%"):
            raise SyntaxError("IPython magic needs IPython: " + s[:60])
        else:
            out.append(l)
    return "\n".join(out)


try:
    from IPython.core.interactiveshell import InteractiveShell
    sh = InteractiveShell.instance(colors="NoColor")
except Exception as e:
    sh = None
    print("[ff] IPython not available (%%s): cells run with plain exec" %% e, flush=True)
G = {"__name__": "__main__"}
for i, c in enumerate(cells):
    print("[cell] %%s %%d/%%d: %%s" %% (stage, i + 1, n, first(c)), flush=True)
    t, ok, err = time.time(), True, None
    if sh is not None:
        r = sh.run_cell(c, store_history=False)
        if not r.success:
            e = r.error_in_exec or r.error_before_exec
            ok, err = False, "%%s: %%s" %% (type(e).__name__, e)
    else:
        try:
            exec(compile(plain(c), "<%%s cell %%d>" %% (stage, i + 1), "exec"), G)
        except BaseException as e:
            ok, err = False, "%%s: %%s" %% (type(e).__name__, e)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": stage, "cell": i + 1, "of": n, "ok": ok, "seconds": round(time.time() - t, 1),
                            "first_line": first(c), "error": err}) + "\n")
    if not ok:
        print("[ff] cell %%d/%%d of stage %%s failed: %%s" %% (i + 1, n, stage, err), flush=True)
        sys.exit(1)
print("[ff] stage %%s: all %%d cells OK" %% (stage, n), flush=True)
''' % CELL_SEP

    def build_vin(self, st, vin):
        shutil.rmtree(vin, ignore_errors=True)
        vin.mkdir(parents=True)
        rows, seen_m = [], set()
        self._in_lora[st.id] = {}
        for mount, aid in st.inputs:
            if mount in seen_m:
                raise StageBlocked(f"{st.id}: two inputs mounted at {mount}")
            seen_m.add(mount)
            p, how = self.locate(aid, saved_only=aid in st.saved_only)
            if p is None and self.soft_input(st, aid):
                self.log(f"    {mount:70s} (soft input of an assembly stage, not available: not mounted)")
                rows.append({"mount": mount, "artifact": aid, "how": "absent (soft)", "path": None, "files": 0, "link": "-"})
                continue
            if p is None:
                raise StageBlocked(f"{st.id}: input {aid} not available")
            bad, fnotes, found = self.file_checks(p, "inputs", how=how, st=st)
            for k, hs in found.items():
                self._in_lora.setdefault(st.id, {}).setdefault(k, set()).update(hs)
            for n_ in fnotes:
                self.log(f"    [variant] {n_}")
            for rf in st.require_files:                      # stage.json variant.require_files
                if rf.get("artifact") in (aid, self.canonical(aid)) and rf.get("file") and rf.get("sha256"):
                    if rf.get("applies_to") and how not in rf["applies_to"]:      # e.g. ["saved", "attached"]: a pin of the saved copy
                        self.log(f"    [variant] {st.id}: require_files {rf['file']} applies to {rf['applies_to']}, input is '{how}': "
                                 "checked by the lineage guard instead" if how == "made" else f"    [variant] {st.id}: require_files {rf['file']} skipped ({how})")
                        continue
                    if how == "made" and self.lineage_policy() == "lineage" and any(
                            fnmatch.fnmatch(Path(rf["file"]).name, fc["glob"].split("/")[-1]) for fc in self.var.get("file_checks", [])):
                        continue                                       # retrained lora_C1.pt: the lineage guard above checked it
                    hits = [p] if p.is_file() else [q for q in p.rglob(Path(rf["file"]).name) if q.is_file()]
                    if not hits:
                        bad.append(f"required file {rf['file']} not found")
                    elif not any(sha256_file(q) == rf["sha256"] for q in hits):
                        bad.append(f"required file {rf['file']}: sha256 != {rf['sha256'][:16]}...")
            if bad:
                raise VariantError(f"{st.id}: input {aid}: " + "; ".join(bad))
            dev = list((self.entry(aid) or {}).get("dev_files") or []) if self.cfg.no_dev else []
            sx = list(getattr(st, "input_exclude", {}).get(mount, []))      # stage.json per-input exclude: honoured for every source
            if p.is_file():
                if p.name in dev or p.name in sx:
                    why = "FF_NO_DEV" if p.name in dev else "stage.json exclude"
                    self.log(f"    {mount:70s} {why}: {p.name} not mounted")
                    rows.append({"mount": mount, "artifact": aid, "how": how, "path": str(p), "files": 0,
                                 "link": "no-dev" if p.name in dev else "excluded"})
                    continue
                link_or_copy(p, vin / mount / p.name)
                n, lk = 1, "file"
            else:
                ex = list((self.entry(aid) or {}).get("exclude") or []) if how != "made" else []
                n, lk = mirror(p, vin / mount, exclude=tuple(ex + [x for x in sx if x not in ex]), exclude_names=dev)
            if dev:
                self.log(f"    {mount:70s} FF_NO_DEV: dev files {dev} left out")
            if sx:
                self.log(f"    {mount:70s} stage.json exclude: {sx} left out (top level of the input)")
            rows.append({"mount": mount, "artifact": aid, "how": how, "path": str(p), "files": n, "link": lk, **({"no_dev": dev} if dev else {}),
                         **({"excluded": sx} if sx else {})})
            self.log(f"    {mount:70s} {how:8s} {n:7d} files ({lk})  <- {p}")
        return rows

    def stage_code(self, st, vin, run_dir):
        if st.code_path is not None and st.code_path.exists():
            src = st.code_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        else:
            ctx = {"python": sys.executable, "stage_dir": str(st.dir), "vin": str(vin), "run": str(run_dir), "repo": str(self.cfg.repo),
                   "data": os.pathsep.join(str(p) for p in self.cfg.data_roots), "work": str(self.cfg.work)}
            src = st.command_cell(ctx)
        rew = []

        def count_sub(old, new, why):
            nonlocal src
            n = src.count(old)
            if n:
                src = src.replace(old, new)
                rew.append({"old": old[:100], "new": new[:100], "count": n, "why": why})

        errs, warns = self.variant_check(st, src)
        if errs:
            raise VariantError(f"{st.id}: " + "; ".join(errs))
        for w in warns:
            self.log(f"  [variant] {st.id}: WARNING {w}")
        for r in self.pipeline.get("runtime_rewrites", []):
            count_sub(r["old"], r["new"], r["why"])
        for var in sorted(set(self.pipeline.get("upload_vars", [])) | set(st.neutralize_upload)):
            rx = re.compile(r"^([ \t]*)(os\.environ\[(['\"])" + re.escape(var) + r"\3\]\s*=\s*(['\"])(?!\4)[^'\"\n]*\4[^\n]*)$", re.M)
            rx2 = re.compile(r"^([ \t]*)(os\.environ\.setdefault\(\s*(['\"])" + re.escape(var) + r"\3\s*,\s*(['\"])(?!\4)[^'\"\n]*\4\s*\)[^\n]*)$", re.M)
            for r_ in (rx, rx2):
                src, n = r_.subn(lambda m: f"{m.group(1)}pass  # [ff runner] upload disabled (output becomes an artifact): {m.group(2).strip()}", src)
                if n:
                    rew.append({"old": f"os.environ['{var}'] = <slug>", "new": "pass  # disabled", "count": n, "why": "no upload to Kaggle"})
        if self.pip_mode(st) == "pinned":
            for g in self.pipeline.get("pip", {}).get("git_pins", []):
                src, n = re.subn(g["regex"], g["new"], src)
                if n:
                    rew.append({"old": g["regex"], "new": g["new"], "count": n, "why": g["why"]})
        count_sub("/kaggle/input", vin.as_posix(), "stage's own input view")
        count_sub("/kaggle/working", run_dir.as_posix(), "stage's own working folder")
        p = run_dir / "_stage.ipy"
        p.write_text(src, encoding="utf-8", newline="\n")
        (run_dir / "_runner.py").write_text(self.RUNNER, encoding="utf-8", newline="\n")
        return p, rew

    def pip_mode(self, st):
        """FF_PIP wins when set to pinned / skip; with the default (original) a stage's own pip_policy applies. The pins of
        docker/constraints-kaggle-v173.txt describe the November image (configs/env_per_stage.json family 'v173'); with
        FF_PIP_SCOPE=v173 a stage whose original run used an older image keeps 'original'."""
        m = self.cfg.pip if self.cfg.pip != "original" else (st.pip_policy if st.pip_policy in ("original", "pinned", "skip") else "original")
        if m == "pinned" and self.cfg.pip_scope == "v173" and self.env_family(st) != "v173":
            return "original"
        return m

    def env_family(self, st):
        """configs/env_per_stage.json family of the stage's original run: v173 | older | local | unknown."""
        return (self.envtab.get(st.id) or {}).get("family", "unknown")

    def pip_warning(self, st, pip):
        if pip != "pinned":
            return None
        fam = self.env_family(st)
        if fam == "v173":
            return None
        e = self.envtab.get(st.id) or {}
        return (f"FF_PIP=pinned applies the November (v173) pins, but the original run of {st.id} used family '{fam}' "
                f"(python {e.get('python') or '?'}, torch {e.get('torch') or '?'}; configs/env_per_stage.json); "
                "set FF_PIP_SCOPE=v173 to keep 'original' for such stages")

    def pip_shim(self):
        d = self.cfg.work / "pipshim"
        files = {
            "pip/__init__.py": "__version__ = '0+ff.skip'\n",
            "pip/__main__.py": "import sys\nprint('[ff runner] pip skipped (FF_PIP=skip):', ' '.join(sys.argv[1:]), flush=True)\nsys.exit(0)\n",
            "pip/_internal/__init__.py": "",
            "pip/_internal/cli/__init__.py": "",
            "pip/_internal/cli/main.py": ("import sys\n\n\ndef main(args=None):\n    print('[ff runner] pip skipped (FF_PIP=skip):', "
                                          "' '.join(sys.argv[1:] if args is None else args), flush=True)\n    return 0\n"),
            "bin/pip": "#!/bin/sh\necho \"[ff runner] pip skipped (FF_PIP=skip): $*\"\nexit 0\n",
        }
        for rel, txt in files.items():
            f = d / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(txt, encoding="utf-8", newline="\n")
        try:
            os.chmod(d / "bin/pip", 0o755)
            shutil.copy2(d / "bin/pip", d / "bin/pip3")
        except OSError:
            pass
        return d

    def models_env(self):
        """Seed writable model caches from the pretrained folder (FF_MODELS, or the pretrained/models artifact)."""
        if self.cfg.models.lower() == "off":
            return {}
        m = Path(self.cfg.models) if self.cfg.models.lower() != "auto" else None
        if m is None:
            p, _how = self.locate("pretrained/models")
            m = p
        if m is None or not Path(m).is_dir():
            return {}
        m = Path(m)
        env, cache = {}, self.cfg.work / "model_cache"
        cl = self.cfg.cache_link if self.cfg.cache_link != "auto" else ("symlink" if self.cfg.kaggle else "copy")
        prefer = ("symlink", "copy") if cl == "symlink" else ("copy",)          # never hard links (a write would change FF_DATA)
        # tools/fetch_models.py writes cache/flag (FLAG_CACHE: WeSpeaker zips, rfs_models, ...) and cache/huggingface/hub
        # (HF_HUB_CACHE, refs/main pinned to the fetched commit). It writes no torch/ folder, so TORCH_HOME is not seeded.
        for src, dst, var in ((m / "cache/flag", cache / "flag", "FLAG_CACHE"), (m / "cache/huggingface/hub", cache / "hf_hub", "HF_HUB_CACHE")):
            if src.is_dir():
                if not (dst / "_FF_SEEDED").exists():
                    mirror(src, dst, prefer=prefer)
                    (dst / "_FF_SEEDED").write_text(f"{src} ({cl})")
                env[var] = str(dst)
        # Libraries with a fixed home cache and no environment variable: openai CLIP (~/.cache/clip, FaRL's CLIP base) and
        # insightface (~/.insightface/models/buffalo_l). Copied (not linked) into the home folder when missing there.
        if os.environ.get("FF_SEED_HOME_CACHES", "1") != "0":
            home = Path(os.path.expanduser("~"))
            for src, dst in ((m / "cache/clip", home / ".cache/clip"), (m / "insightface/models", home / ".insightface/models")):
                if src.is_dir():
                    for f in src.rglob("*"):
                        t = dst / f.relative_to(src)
                        if f.is_file() and not t.exists():
                            try:
                                t.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(f, t)
                            except OSError as ex:
                                self.log(f"  [models] cannot seed {t} ({ex})")
        if (m / "imagebind_huge.pth").is_file():
            env["FF_IMAGEBIND_CKPT"] = str(m / "imagebind_huge.pth")
        if os.environ.get("FF_HF_OFFLINE") == "1":
            env["HF_HUB_OFFLINE"] = "1"
        return env

    def literal_info(self, st):
        """runner/payloads.py scan of the stage's embedded code (cached): literal /kaggle/input uses and module env defaults."""
        if st.id not in self._lit:
            try:
                import payloads as PL                               # noqa: E402  (runner/payloads.py)
            except ImportError:
                sys.path.insert(0, str(HERE))
                import payloads as PL                               # noqa: E402
            txt = st.code_path.read_text(encoding="utf-8") if st.code_path and st.code_path.exists() else ""
            self._lit[st.id] = PL.scan_stage_code(txt)
        return self._lit[st.id]

    def is_literal(self, st):
        """-> (needs the literal /kaggle/input, 'declared' | 'detected' | None)."""
        if st.literal_kaggle_input:
            return True, "declared"
        if getattr(st, "literal_declared_false", False) and os.environ.get("FF_LITERAL_HONOUR_FALSE", "1") != "0":
            return False, "declared-false"                   # stage.json runner_requirements: false + why (detected uses reviewed)
        if os.environ.get("FF_LITERAL_DETECT", "1") != "0" and self.literal_info(st)["literal"]:
            return True, "detected"
        return False, None

    def stage_env(self, st, code, vin, run_dir):
        env = {k: v for k, v in os.environ.items() if k not in NEVER_PASS and not NEVER_PASS_RX.match(k)}
        for s in SECRET_NAMES:
            if s in env and s not in st.secrets and s not in code:
                env.pop(s)
        env.update(self.var.get("guard_env_defaults", {}))
        env.update(st.env)
        if self.cfg.smoke:
            env.update(st.smoke_env)
        for var in set(self.pipeline.get("upload_vars", [])) | set(st.neutralize_upload):
            env.pop(var, None)
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "FF_STAGE": st.id, "FF_STAGE_DIR": str(st.dir),
                    "FF_REPO": str(self.cfg.repo), "FF_DATA": os.pathsep.join(str(p) for p in self.cfg.data_roots),
                    "FF_WORK": str(self.cfg.work), "FF_MODE": self.cfg.mode, "FF_SMOKE": "1" if self.cfg.smoke else "0",
                    "FF_VIN": str(vin), "FF_RUN": str(run_dir), "FF_VARIANT": self.cfg.variant})
        env.setdefault("MPLBACKEND", "Agg")
        self._env_defaults = []
        for x in self.literal_info(st)["env_defaults"]:          # embedded module default os.environ.get(X, "/kaggle/input")
            if x not in env:
                env[x] = str(vin)
                self._env_defaults.append(x)
        pip = self.pip_mode(st)
        pw = self.pip_warning(st, pip)
        if pw:
            self.log(f"  [pip] WARNING {pw}")
        if pip == "pinned":
            c = self.cfg.repo / self.pipeline.get("pip", {}).get("constraints_file", "docker/constraints-kaggle-v173.txt")
            env["PIP_CONSTRAINT"] = str(c)
        elif pip == "skip":
            d = self.pip_shim()
            env["PYTHONPATH"] = str(d) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["PATH"] = str(d / "bin") + os.pathsep + env.get("PATH", "")
        env.update(self.models_env())
        fe = self.var.get("forbidden_env", {})
        for k, bad in fe.items():
            if env.get(k) in bad:
                raise VariantError(f"{st.id}: environment {k}={env[k]} is forbidden for 13.54")
        return env, pip

    def collect(self, st, run_dir, vin, meta):
        kdir = self.art_dir(st.kernel)
        shutil.rmtree(kdir, ignore_errors=True)
        kdir.mkdir(parents=True)
        skip = set(st.scratch) | {"_stage.ipy", "_runner.py"}
        dropped = 0
        for x in run_dir.iterdir():
            if x.name in skip:
                continue
            if x.is_symlink():
                dropped += 1
                continue
            shutil.move(str(x), str(kdir / x.name))
        vin_s = str(vin.resolve())
        for root, dirs, files in os.walk(kdir):
            for nm in dirs + files:
                q = Path(root) / nm
                if q.is_symlink():
                    try:
                        tgt = os.readlink(q)
                    except OSError:
                        tgt = ""
                    if tgt.startswith(vin_s) or not q.exists():
                        q.unlink()
                        dropped += 1
        done = {**meta, "dropped_links_into_input": dropped}
        jdump(kdir / "_DONE.json", done)
        made = [st.kernel]
        for aid, spec in st.exports.items():
            edir = self.art_dir(aid)
            shutil.rmtree(edir, ignore_errors=True)
            if spec is None:
                raise RuntimeError(f"{st.id}: export {aid} has no content spec (skeleton entry?)")
            if isinstance(spec, str):
                src = kdir / spec if spec not in (".", "./", "") else kdir
                if not src.is_dir():
                    raise RuntimeError(f"{st.id}: export {aid}: missing folder {spec}")
                for root, _d, files in os.walk(src):
                    for f in files:
                        s = Path(root) / f
                        if s.name == "_DONE.json" and Path(root) == kdir:
                            continue
                        link_or_copy(s, edir / s.relative_to(src), prefer=("hardlink", "copy"))
                edir.mkdir(parents=True, exist_ok=True)
            else:
                for dst, src in spec.items():
                    hits = sorted(kdir.glob(src)) if any(c in src for c in "*?[") else [kdir / src]
                    if not hits or not all(h.exists() for h in hits):
                        raise RuntimeError(f"{st.id}: export {aid}: missing {src}")
                    for h in hits:
                        t = edir / dst / h.name if dst.endswith("/") else edir / dst
                        if h.is_dir():
                            for root, _d, files in os.walk(h):
                                for f in files:
                                    s = Path(root) / f
                                    link_or_copy(s, t / s.relative_to(h), prefer=("hardlink", "copy"))
                        else:
                            link_or_copy(h, t, prefer=("hardlink", "copy"))
            jdump(edir / "_DONE.json", {**done, "export_of": st.kernel, "spec": spec})
            made.append(aid)
        return made

    @staticmethod
    def _json_get(obj, path):
        for k in [x for x in re.split(r"[./]", str(path)) if x]:
            obj = obj[int(k)] if isinstance(obj, list) else obj[k]
        return obj

    @staticmethod
    def _subset(expect, got):
        if isinstance(expect, dict):
            return isinstance(got, dict) and all(k in got and Runner._subset(v, got[k]) for k, v in expect.items())
        if isinstance(expect, float) or isinstance(got, float):
            try:
                return abs(float(expect) - float(got)) <= 1e-9 * max(1.0, abs(float(expect)))
            except (TypeError, ValueError):
                return False
        return expect == got

    def verify_stage(self, st):
        """stage.json 'verify' + 'post_checks' entries on the fresh output -> list of (status, message), status in
        OK | FAIL (fatal) | WARN (not fatal) | INFO. Fatal: post_checks, check 'exact', strict: true, and the zip / member hashes of
        'assemble' stages in FF_MODE=assemble (deterministic from the saved scores). Hashes of GPU / data stages, and of assembly
        stages fed by re-computed inputs (modes infer / train), are informational: a re-run is not byte-identical."""
        sys.path.insert(0, str(HERE))
        import verify as V                                           # noqa: E402  (runner/verify.py)
        res, exp_all = [], None
        kdirs = [self.art_dir(st.kernel)] + [self.art_dir(a) for a in st.exports]

        def find(rel):
            for d in kdirs:
                if (d / rel).exists():
                    return d / rel
            return None

        def put(ok, strict, msg):
            res.append(("OK" if ok else ("FAIL" if strict else "WARN"), msg))

        for c in st.post_checks:
            f = find(c.get("file", ""))
            if f is None:
                put(False, True, f"post_check: missing {c.get('file')}")
                continue
            try:
                got = self._json_get(jload(f), c["json_path"]) if c.get("json_path") else jload(f)
                put(self._subset(c.get("equals"), got), True, f"post_check {c['file']} {c.get('json_path', '')} == {c.get('equals')!r} (got {got!r})")
            except Exception as ex:
                put(False, True, f"post_check {c.get('file')}: {type(ex).__name__}: {ex}")
        for v in st.verify:
            if not isinstance(v, dict):
                res.append(("INFO", f"verify entry not a dict, skipped: {str(v)[:80]}"))
                continue
            rel = v.get("file")
            strict = bool(v.get("strict")) or v.get("check") == "exact"
            det = st.kind == "assemble" and self.cfg.mode == "assemble"      # byte-exact only when every input is the saved one
            try:
                self._verify_one(st, v, rel, find, put, res, strict, det, V)
            except Exception as ex:                                  # one malformed entry never stops the chain
                put(False, strict or det, f"verify {rel}: cannot evaluate this entry ({type(ex).__name__}: {ex})")
        return res

    HEX64 = re.compile(r"^[0-9a-f]{64}$")

    @staticmethod
    def _json_get_dotted(obj, path):
        """JSON lookup where a key may itself contain dots ('files.lora_C1.pt' -> obj['files']['lora_C1.pt'])."""
        if isinstance(obj, dict) and path in obj:
            return obj[path]
        if isinstance(obj, list):
            head, _, rest = path.partition(".")
            return Runner._json_get_dotted(obj[int(head)], rest) if rest else obj[int(head)]
        if not isinstance(obj, dict):
            raise KeyError(path)
        for i in [i for i, ch in enumerate(path) if ch == "."]:
            k = path[:i]
            if k in obj:
                try:
                    return Runner._json_get_dotted(obj[k], path[i + 1:])
                except (KeyError, IndexError, ValueError):
                    continue
        raise KeyError(path)

    def _lineage_ok_file(self, f):
        """A file named by a variant file_check (lora_C1.pt) whose hash is a lineage hash of this FF_WORK (FF_MODE=train)."""
        for fc in self.var.get("file_checks", []):
            if fnmatch.fnmatch(Path(f).name, fc["glob"].split("/")[-1]) and self.lineage_policy() == "lineage":
                h = sha256_file(f)
                key = fc.get("id", fc["glob"])
                if h in self.lineage_hashes(fc) or any(h in hs.get(key, set()) for hs in self._in_lora.values()):
                    return True, h
        return False, None

    def _verify_one(self, st, v, rel, find, put, res, strict, det, V):
        f = find(rel) if rel else None
        exp = v.get("expected")
        if f is None:
            if rel:
                put(False, strict or det, f"verify: missing {rel}")
            return
        lin_ok, lin_h = self._lineage_ok_file(f)
        if "bytes" in v:
            n = f.stat().st_size
            res.append(("OK" if n == v["bytes"] else "INFO", f"verify {rel}: {n} bytes (original {v['bytes']})"))
        want = v.get("sha256") if v.get("sha256") else (exp if isinstance(exp, str) and self.HEX64.match(exp) else None)
        if want and v.get("check") not in ("metadata", "info"):
            h = sha256_file(f)
            if h != want and lin_ok:
                res.append(("INFO", f"verify {rel}: sha256 {h[:16]}... != {want[:16]}... but = the lora_C1.pt lineage hash of this FF_MODE=train "
                                    "run (retrained LoRA C1; the variant guard checked its dry check / propagation instead)"))
            else:
                put(h == want, strict or det, f"verify {rel}: sha256 {h[:16]}... {'==' if h == want else '!='} {want[:16]}...")
        elif want or v.get("saved_sha256"):
            e = v.get("saved_sha256") or want
            h = sha256_file(f)
            res.append(("OK" if h == e else "INFO", f"verify {rel}: sha256 {h[:16]}... {'==' if h == e else '!='} saved {e[:16]}... "
                                                    "(informational: a GPU / training re-run is not byte-identical)"))
        if v.get("sha256_crlf"):                                     # hash of the CRLF copy embedded in later notebooks
            b = f.read_bytes()
            hs = {hashlib.sha256(b).hexdigest(), hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()}
            res.append(("OK" if v["sha256_crlf"] in hs else "INFO", f"verify {rel}: sha256 (LF or CRLF) {'matches' if v['sha256_crlf'] in hs else 'differs from'} "
                                                                     f"{v['sha256_crlf'][:16]}..."))
        mem = v.get("members_sha256")
        if not mem and isinstance(exp, dict) and (v.get("check") in (None, "members_sha256") and str(rel).endswith(".zip")):
            mem = exp                                                # {"member": sha256} written under 'expected'
        ref = v.get("members_sha256_from") or (exp if isinstance(exp, str) and ("#" in exp or exp.startswith("configs/")) else None)
        if ref and not mem:
            exp_all = jload(self.cfg.repo / "configs/expected_sha256.json")
            node = exp_all
            for k in ref.split("#", 1)[-1].split("/"):
                node = node[k]
            mem = node.get("members") if isinstance(node, dict) else None
        if mem:
            try:
                ok, rows = V.check_members(f, mem)
                bad = [r for r in rows if not r.startswith("OK")]
                put(ok, strict or det, f"verify {rel}: {len(mem)} member hashes " + ("all match" if ok else "; ".join(bad)))
            except Exception as ex:
                put(False, strict or det, f"verify {rel}: cannot read members ({ex})")
        if isinstance(exp, dict) and v.get("check") == "json":       # {"dotted.key": value} subset of a JSON file
            got = jload(f)
            bad = []
            for k, want_v in exp.items():
                try:
                    g = self._json_get_dotted(got, k)
                except (KeyError, IndexError, ValueError):
                    bad.append(f"{k}: missing")
                    continue
                if not self._subset(want_v, g):
                    bad.append(f"{k}: {g!r} != {want_v!r}")
            put(not bad, strict, f"verify {rel}: JSON {len(exp)} key(s) " + ("match" if not bad else "differ: " + "; ".join(bad)[:300]))
        efj = v.get("expect_file_json")
        if isinstance(efj, dict) and efj.get("file") and efj.get("key"):    # {"file", "key", "value"} on another output file
            f2 = find(efj["file"])
            if f2 is None:
                put(False, strict, f"verify: missing {efj['file']}")
            else:
                got_v = self._json_get_dotted(jload(f2), efj["key"])
                want_v = efj.get("value", efj.get("equals"))
                put(self._subset(want_v, got_v), strict, f"verify {efj['file']}: {efj['key']} == {want_v!r} (got {got_v!r})")
        if "json_field" in v and "equals" in v:
            got_v = self._json_get_dotted(jload(f), v["json_field"])
            put(self._subset(v["equals"], got_v), strict, f"verify {rel}: {v['json_field']} == {v['equals']!r} (got {got_v!r})")
        if "expect" in v or "values" in v or ("key" in v and "equals" in v):
            got = jload(f)
            if "key" in v:
                got_v = self._json_get(got, v["key"])
                put(self._subset(v["equals"], got_v), strict, f"verify {rel}: {v['key']} == {v['equals']!r} (got {got_v!r})")
            else:
                e = v.get("expect", v.get("values"))
                put(self._subset(e, got), strict, f"verify {rel}: JSON values {'match' if self._subset(e, got) else 'differ'}")
        if v.get("arrays_sha256"):
            try:
                import numpy as np                                   # noqa: E402
                z = np.load(f, allow_pickle=False)
                with zipfile.ZipFile(f) as zf:
                    member = {Path(n).stem: hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist()}
                bad = []
                for k, h in v["arrays_sha256"].items():
                    a = np.ascontiguousarray(z[k]) if k in z.files else None
                    hs = {member.get(k)} | ({hashlib.sha256(a.tobytes()).hexdigest()} if a is not None else set())
                    if h not in hs:
                        bad.append(k)
                put(not bad, strict, f"verify {rel}: {len(v['arrays_sha256'])} arrays " + ("match" if not bad else f"differ: {bad}"))
            except Exception as ex:
                put(False, strict, f"verify {rel}: arrays not checked ({type(ex).__name__}: {ex})")
        if v.get("spearman_ref"):
            refp = Path(v["spearman_ref"])
            if not refp.is_absolute():
                refp = next((r / refp for r in [self.cfg.repo] + self.cfg.data_roots if (r / refp).exists()), refp)
            if refp.exists():
                rows = V.compare_zips(f, refp)
                thr = float(v.get("min_spearman", 0.9999))
                put(all(r["spearman"] >= thr for r in rows), strict,
                    f"verify {rel} vs {refp.name}: Spearman " + ", ".join(f"{r['member']}: {r['spearman']:.6f} (max |d| {r['max_abs_diff']:.4g})" for r in rows))
            else:
                put(False, False, f"verify: reference {v['spearman_ref']} not found")

    def run_stage(self, st):
        if not st.runnable():
            raise StageBlocked(f"stage {st.id}: no runnable code in this repository yet "
                               f"({'skeleton only, no stage.json' if st.skeleton else 'stage.json has neither stage.ipy nor command'})")
        if self.has_gpu is None:
            self.has_gpu = has_gpu()
        if st.gpu and not self.has_gpu and not self.cfg.smoke:
            raise StageBlocked(f"stage {st.id} needs a GPU (run on a GPU machine, or FF_SMOKE=1 to try it on CPU)")
        vin, run_dir = self.cfg.tmp / "vin" / st.id, self.cfg.tmp / "run" / st.id
        logs = self.cfg.work / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log(f"\n[stage] {st.id}: RUN ({st.title}; ~{st.minutes * (0.1 if self.cfg.smoke else 1):.0f} min by the original run; "
                 f"session used {self.elapsed_min():.0f} min)")
        self.log("  inputs:")
        rows = self.build_vin(st, vin)
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        script, rew = self.stage_code(st, vin, run_dir)
        code = script.read_text(encoding="utf-8")
        for r in rew:
            self.log(f"  [rewrite] {r['count']} x {r['old'][:70]!r} -> {r['new'][:70]!r} ({r['why'][:80]})")
        env, pip = self.stage_env(st, code, vin, run_dir)
        cells_log = logs / f"{st.id}.cells.jsonl"
        if cells_log.exists():
            cells_log.unlink()
        t = time.time()
        undo_literal = self.literal_kaggle(st, vin, run_dir, rows)
        try:
            with open(logs / f"{st.id}.log", "w", encoding="utf-8") as lf:
                p = subprocess.Popen([sys.executable, "-u", str(run_dir / "_runner.py"), str(script), st.id, str(cells_log)], cwd=run_dir, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace", encoding="utf-8")
                tail = []
                for line in p.stdout:
                    lf.write(line)
                    lf.flush()
                    if not self.quiet:
                        print(f"  | {line}", end="", flush=True)
                    tail = (tail + [line])[-40:]
                code_rc = p.wait()
        finally:
            undo_literal()
        minutes = (time.time() - t) / 60
        rec = self.state["stages"].setdefault(st.id, {})
        if code_rc != 0:
            rec.update(status="failed", minutes=round(minutes, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"), tail="".join(tail[-15:]))
            self.save_state()
            raise RuntimeError(f"stage {st.id} failed (exit {code_rc}) after {minutes:.1f} min; log: {logs / (st.id + '.log')}")
        meta = {"stage": st.id, "minutes": round(minutes, 1), "mode": self.cfg.mode, "smoke": self.cfg.smoke, "variant": self.cfg.variant,
                "pip": pip, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "code_sha256": sha256_text(st.code_path.read_text(encoding="utf-8")) if st.code_path and st.code_path.exists() else None,
                "command": st.command, "rewrites": rew, "inputs": rows, "python": sys.version.split()[0], "repo_commit": self.repo_commit()}
        made = self.collect(st, run_dir, vin, meta)
        self.made_now.update(self.canonical(a) for a in made)
        problems, vnotes, lineage, root_status = [], [], {}, {}
        for a in made:
            b, n_, found = self.file_checks(self.art_dir(a), "outputs", how="made", st=st)
            problems += b
            vnotes += n_
            for k, hs in found.items():
                lineage.setdefault(a, {}).setdefault(k, set()).update(hs)
        for fc in self.var.get("file_checks", []):              # lineage root (t163l_orig in FF_MODE=train): its own root check
            key = fc.get("id", fc["glob"])
            if st.id not in (fc.get("made_in_session") or {}).get("lineage_roots", {}):
                continue
            new = {h for v in lineage.values() for h in v.get(key, set()) if h != fc["sha256"]}
            if new and not self.cfg.smoke and self.lineage_policy() == "lineage":
                ok, msg = self.lineage_root_check(st, fc)
                vnotes.append(msg)
                root_status[key] = "passed" if ok else "failed"
                if not ok:
                    problems.append(f"lineage root {st.id}: {msg}")
        for n_ in vnotes:
            self.log(f"  [variant] {n_}")
        for a in made:                                           # record the hashes (and the root check) in _DONE.json
            vl = {k: {"sha256": sorted(hs), "fixed_sha256_match": all(h == next((fc["sha256"] for fc in self.var.get("file_checks", [])
                                                                               if fc.get("id", fc["glob"]) == k), None) for h in hs),
                      "root_check": root_status.get(k, "n/a"), "stage": st.id, "policy": "smoke" if self.cfg.smoke else self.lineage_policy()}
                  for k, hs in lineage.get(a, {}).items()}
            if vl:
                dj = self.art_dir(a) / "_DONE.json"
                info = jload(dj)
                info["variant_lineage"] = vl
                jdump(dj, info)
        if self._env_defaults:
            meta_extra = {"env_defaults_exported": self._env_defaults}
            for a in made:
                dj = self.art_dir(a) / "_DONE.json"
                info = jload(dj)
                info.update(meta_extra)
                jdump(dj, info)
        if problems:
            rec.update(status="variant_violation", problems=problems)
            self.save_state()
            raise VariantError(f"stage {st.id} failed the 13.54 variant guard on its outputs: " + "; ".join(problems))
        ver = self.verify_stage(st) if (st.verify or st.post_checks) else []
        for s, msg in ver:
            self.log(f"  [verify] {s:4s} {msg}")
        if not self.cfg.keep_vin:
            shutil.rmtree(vin, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        fatal = [m for s, m in ver if s == "FAIL"]
        rec.update(status="verify_failed" if fatal else "done", minutes=round(minutes, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"),
                   verify=[[s, m] for s, m in ver])
        self.save_state()
        if fatal and not self.cfg.smoke:
            raise RuntimeError(f"stage {st.id}: {len(fatal)} fatal check(s) failed: " + " | ".join(fatal))
        self.log(f"[stage] {st.id}: DONE after {minutes:.1f} min -> {[str(self.art_dir(a)) for a in made]}")
        return "done"

    def literal_kaggle(self, st, vin, run_dir, rows=()):
        """stage.json runner_requirements.literal_kaggle_input: code decoded at run time (base64 zips written to disk and imported)
        reads the literal path /kaggle/input, which the runner's string rewrite cannot reach. On Linux the runner points /kaggle/input
        (and /kaggle/working) at the stage's own folders with symlinks for the duration of the stage (an empty directory there, e.g.
        in the Docker image, is replaced and restored); FF_LITERAL_KAGGLE=mounted = the caller already mounted exactly the stage's
        inputs at /kaggle/input (container). On Kaggle the notebook's /kaggle/input cannot change: the runner warns when it holds
        attachments that are not inputs of the stage (FF_LITERAL_STRICT=1 makes that an error). Returns a cleanup function."""
        lit_needed, lit_how = self.is_literal(st)
        if not lit_needed:
            if lit_how == "declared-false" and self.literal_info(st)["literal"]:
                u = [x for x in self.literal_info(st)["uses"] if x["kind"] == "literal"]
                self.log(f"  [literal /kaggle/input] {st.id}: {len(u)} literal use(s) in embedded code, declared not needed in stage.json "
                         f"(runner_requirements.literal_kaggle_input=false: {st.literal_why[:160]}); FF_LITERAL_HONOUR_FALSE=0 overrides")
            return lambda: None
        if lit_how == "detected":
            u = [x for x in self.literal_info(st)["uses"] if x["kind"] == "literal"]
            self.log(f"  [literal /kaggle/input] {st.id}: not declared in stage.json, detected in its embedded code "
                     f"({len(u)} use(s), e.g. {u[0]['member'].split('/')[-1]}:{u[0]['line']}); treated as literal_kaggle_input")
        kin, kwk = Path("/kaggle/input"), Path("/kaggle/working")
        if self.cfg.kaggle:
            # The decoded code scans the REAL /kaggle/input of this session. It cannot see artifacts made in this session
            # (FF_WORK/art) or saved views / unpacked zips under FF_WORK, and it sees every other attachment of the session.
            kin_r = kin.resolve()
            outside = []
            for r in rows:
                if r.get("how") == "absent (soft)":
                    continue
                pth = Path(r["path"]).resolve() if r.get("path") else None
                if r.get("how") == "made" or pth is None or not (pth == kin_r or kin_r in pth.parents):
                    outside.append(f"{r['artifact']} ({r.get('how')}: {r.get('path')})")
            if outside:
                raise StageBlocked(
                    f"{st.id}: its embedded code scans the literal /kaggle/input, but {len(outside)} input(s) are not attached there: "
                    f"{outside[:4]}. Run this stage alone in a NEW Kaggle session (FF_ONLY={st.id}) with exactly its inputs attached, "
                    "including the earlier session's output (its FF_WORK) for inputs made there; or run it on Linux / in a container "
                    "(runner/README.md 'Literal /kaggle/input stages')")
            repo_top = None
            try:
                rel = self.cfg.repo.relative_to(kin_r)
                repo_top = rel.parts[2] if rel.parts and rel.parts[0] in ("datasets", "notebooks", "models", "competitions") and len(rel.parts) > 2 \
                    else (rel.parts[0] if rel.parts else None)
            except ValueError:
                pass
            # An attachment counts as one of the stage's inputs only by its slug (Kaggle layout). A Drive-layout dataset that holds
            # these inputs among many other artifacts is 'extra': the scan would also see its other files.
            mine = {a.split("/")[-1] for _m, a in st.inputs} | {m.split("/")[-1] for m, _a in st.inputs}
            ffw_tops = set()                                   # an earlier session's FF_WORK attached as input holds made inputs
            for r in rows:
                if r.get("path") and ("/ff_work/art/" in Path(r["path"]).as_posix() or "/art/" in Path(r["path"]).as_posix()):
                    try:
                        rp = Path(r["path"]).resolve().relative_to(kin_r).parts
                        ffw_tops.add(rp[2] if rp[0] in ("datasets", "notebooks", "models", "competitions") and len(rp) > 2 else rp[0])
                    except (ValueError, IndexError):
                        pass
            if ffw_tops:
                self.log(f"  [literal /kaggle/input] WARNING {st.id}: earlier session output(s) {sorted(ffw_tops)} attached: the literal scan "
                         "also sees the other artifacts made in that session (acceptable when none holds a different copy of the files "
                         "this stage looks for)")
            mine |= ffw_tops
            tops = set()
            if kin.is_dir():
                for p in kin.iterdir():
                    tops |= {q.name for q in p.glob("*/*")} if p.name in ("datasets", "notebooks", "models", "competitions") else {p.name}
            extra = sorted(tops - mine - ({repo_top} if repo_top else set()))
            if repo_top:
                hold = [x for x in ("assembly/inputs", "05_scores") if any((self.cfg.repo / x).glob("*.np*")) or any((self.cfg.repo / x).glob("*.zip"))]
                msg = (f"{st.id}: the code attachment {repo_top!r} is visible to the literal scan"
                       + (f" and holds score files in {hold}: remove them from the bundle for this stage" if hold else " (no data files in it)"))
                if hold and os.environ.get("FF_LITERAL_STRICT") == "1":
                    raise StageBlocked(msg)
                self.log("  [literal /kaggle/input] " + ("WARNING " if hold else "") + msg)
            if extra:
                msg = (f"{st.id}: its embedded code scans the literal /kaggle/input, which also holds {len(extra)} attachment(s) that are not "
                       f"inputs of this stage ({extra[:6]}...); valid only if they hold no conflicting copy of the files it looks for "
                       "(see stage.json runner_requirements.why); safest: run this stage in a session with exactly its inputs attached "
                       "(the generated notebooks set FF_LITERAL_STRICT=1, which makes this an error)")
                if os.environ.get("FF_LITERAL_STRICT") == "1" or (st.kaggle_strict and os.environ.get("FF_LITERAL_KAGGLE") != "mounted"):
                    raise StageBlocked(msg + (" [stage.json runner_requirements.kaggle_strict]" if st.kaggle_strict else ""))
                self.log("  [literal /kaggle/input] WARNING " + msg)
            return lambda: None
        if os.name != "posix":
            raise StageBlocked(f"{st.id}: its embedded code reads the literal /kaggle/input ({lit_how}): run it on Linux, "
                               "in Docker, or in a Kaggle session with exactly its inputs attached")
        if os.environ.get("FF_LITERAL_KAGGLE") == "mounted":
            self.log("  [literal /kaggle/input] FF_LITERAL_KAGGLE=mounted: using the caller's /kaggle/input as is")
            return lambda: None
        undo = []
        for lit, target, required in ((kin, vin, True), (kwk, run_dir, False)):
            try:
                if lit.is_symlink():
                    lit.unlink()
                    undo.append((lit, None))
                elif lit.is_dir() and not any(lit.iterdir()):
                    lit.rmdir()
                    undo.append((lit, "dir"))
                elif lit.exists():
                    if required:
                        raise StageBlocked(f"{st.id}: {lit} exists and is not empty; mount exactly this stage's inputs there and set "
                                           "FF_LITERAL_KAGGLE=mounted, or remove it")
                    self.log(f"  [literal /kaggle] {lit} exists and is not empty: left as is")
                    continue
                else:
                    undo.append((lit, None))
                lit.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(target, lit)
            except StageBlocked:
                raise
            except OSError as ex:
                raise StageBlocked(f"{st.id}: cannot point {lit} at {target} ({ex}); create a writable /kaggle first "
                                   "(sudo mkdir -p /kaggle && sudo chown $USER /kaggle) or run in Docker")
        self.log(f"  [literal /kaggle] /kaggle/input -> {vin}, /kaggle/working -> {run_dir} for this stage")

        def cleanup():
            for lit, restore in undo:
                try:
                    if lit.is_symlink():
                        lit.unlink()
                    if restore == "dir":
                        lit.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
        return cleanup

    def repo_commit(self):
        try:
            r = subprocess.run(["git", "-C", str(self.cfg.repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    # ------------------------------------------------------------------ run a plan
    def run(self, targets=None, dry=False):
        self.state.setdefault("sessions", []).append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), **self.cfg.describe()})
        order, blocked = self.make_plan(targets)
        self.print_plan(order, blocked)
        if dry:
            return {"order": order, "blocked": blocked}
        status = {}
        for sid in order:
            st = self.S[sid]
            rec = self.state["stages"].setdefault(sid, {})
            if self.cfg.only and sid not in self.cfg.only:
                self.log(f"[stage] {sid}: outside FF_ONLY (another session runs it): skipped")
                status[sid] = rec["status"] = "not_selected"
                continue
            miss = [a for _m, a in st.inputs if (self.locate(a, saved_only=True)[0] is None if a in st.saved_only else
                                                 (not self.available(a) and not self.soft_input(st, a)))]
            if miss:
                self.log(f"[stage] {sid}: missing inputs {miss} -> blocked")
                status[sid] = rec["status"] = "blocked"
                rec["missing"] = miss
                continue
            est = st.minutes * (0.1 if self.cfg.smoke else 1.0)
            if self.cfg.time_budget_h > 0:
                left = self.cfg.time_budget_h * 60 - self.elapsed_min()
                if est > left:
                    self.log(f"[stage] {sid}: needs ~{est:.0f} min, {left:.0f} min left in this session -> deferred "
                             f"(attach this session's FF_WORK as input and run again)")
                    status[sid] = rec["status"] = "deferred"
                    continue
            status[sid] = self.run_stage(st)
        self.save_state()
        self.report(order)
        return status

    # ------------------------------------------------------------------ printing
    def print_plan(self, order, blocked):
        tot = sum(self.S[s].minutes for s in order)
        self.log(f"Target {self.resolve_targets()} | FF_MODE {self.cfg.mode} | FF_USE_SAVED {self.use_saved} | {len(order)} stage(s) to run, "
                 f"~{tot / 60:.1f} h by the original runs{' (SMOKE)' if self.cfg.smoke else ''}"
                 f"{' | planning with --assume-saved' if self.cfg.assume_saved else ''}:")
        for sid in order:
            st = self.S[sid]
            tag = "" if not self.cfg.only or sid in self.cfg.only else "   (outside FF_ONLY)"
            sk = "   [NO stage.json YET]" if st.skeleton else ("   [NO stage.ipy / command YET]" if not st.runnable() else "")
            lit, lh = (False, None) if st.skeleton else self.is_literal(st)
            lt = f"   [literal /kaggle/input: {lh}]" if lit else ""
            self.log(f"  {sid:20s} {st.kind:8s} {'GPU' if st.gpu else 'CPU'} ~{st.minutes:5.0f} min  {st.title[:70]}{tag}{sk}{lt}")
        if self.cfg.models.lower() != "off" and order:
            pm = Path(self.cfg.models) if self.cfg.models.lower() != "auto" else self.locate("pretrained/models")[0]
            self.log(f"\npretrained models (FF_MODELS={self.cfg.models}): " + (
                f"{pm} (FLAG_CACHE / HF_HUB_CACHE / CLIP / insightface caches seeded per stage)" if pm and Path(pm).is_dir() else
                "NOT FOUND: the stages download their models from the Internet at run time (unpinned where the original was); "
                "populate FF_DATA/06_pretrained with `python tools/fetch_models.py --dest <FF_DATA>/06_pretrained` (or FF_FORCE=models_fetch)"))
        soft = getattr(self, "soft_missing", [])
        if soft:
            self.log(f"\n{len(soft)} soft input(s) of assembly stages not available (not mounted; those stages read FF_DATA/05_scores):")
            for aid, via in soft:
                self.log(f"  {aid:70s} for {via}")
        if blocked:
            self.log(f"\n{len(blocked)} artifact(s) not available (the stage that needs one will stop as 'blocked'):")
            for aid, via, why in blocked:
                self.log(f"  {aid:70s} for {via:18s} {why}")

    def report(self, order=None):
        self.log(f"\nSession used {self.elapsed_min():.0f} min. Stage status:")
        for sid in (order or self.order_ref):
            if sid not in self.S:
                continue
            r = self.state["stages"].get(sid, {})
            outs = "".join("x" if self.locate(a)[0] is not None else "-" for a in self.S[sid].outputs)
            self.log(f"  {sid:20s} {self.S[sid].kind:8s} {r.get('status', '-'):17s} {str(r.get('minutes', '')):>7} min  outputs {outs}")

    def stages_markdown(self):
        rows = ["| # | stage | group | kind | role | device | min (orig.) | origin | run | stage.json |", "|---|---|---|---|---|---|---|---|---|---|"]
        ids = [s for s in self.order_ref if s in self.S] + sorted(s for s in self.S if s not in self.order_ref)
        for i, sid in enumerate(ids, 1):
            st = self.S[sid]
            o = st.origin
            rows.append(f"| {i} | `{sid}` | {st.group} | {st.kind} | {st.role} | {'GPU' if st.gpu else 'CPU'}{' x' + str(st.gpus) if st.gpus and st.gpus > 1 else ''} | "
                        f"{st.minutes:.0f} | {o.get('exactness', '-')} | {('`' + o['run'] + '`') if o.get('run') else '-'} | "
                        f"{'yes' if not st.skeleton else '**missing**'} |")
        return "\n".join(rows) + "\n"
