"""Generate the self-contained Evaluation-phase notebooks (docs/PLAN_EVAL.md).

    FLAG_12_features.ipynb  (E3)  every feature the Evaluation pipeline reads, for any organiser-format data,
                                  + ArcFace / ECAPA on the MAV-Celeb v1/v2 rows for STRESS-01
Every notebook carries its full code in its cells: no team .py is imported or embedded. Only pip packages.
Local smoke test: python build_eval_notebooks.py && python run_notebook_local.py FLAG_12_features.ipynb (FLAG_LOCAL=1 ...).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def nb(cells, path):
    d = {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                    **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
         "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11"}},
         "nbformat": 4, "nbformat_minor": 5}
    (HERE / path).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", path, len(cells), "cells")


# ============================================================================ FLAG_12_features
INTRO12 = """
# FLAG 2027 — NB-12: features for the Evaluation pipeline (E3 in `docs/PLAN_EVAL.md`)

One notebook produces **every array** that `BEST/v6_eval/pipeline.py` reads for a test file, with the same code as the
arrays behind SET-13 / INDEP-01. Run it on the dev set now (dry run, checked against the arrays we already have). Run it
on the Evaluation data on 11/11.

| array | model | dim | used by |
|---|---|---:|---|
| `face_arcface_<split>` (+ `_detok`) | insightface buffalo_l: SCRFD + 5-point crop + ArcFace R100 | 512 | clustering faces |
| `voice_ecapa192_<split>` | speechbrain spkrec-ecapa-voxceleb, whole file | 192 | clustering voices, member r2mix |
| `voice_ecapa6144_<split>` | same model, attentive-statistics pooling | 6144 | member s010B |
| `face_vitage` / `face_vitageprob` | nateraw/vit-age-classifier | 768 / 9 | age term |
| `voice_w2v2ag` / `_w2v2agage` / `_w2v2aggen` | audeering wav2vec2-large-robust-24-ft-age-gender | 1024 / 1 / 3 | age term |
| `face_ibv` / `voice_iba` | ImageBind-huge vision / audio (2 s clips covering the whole file) | 1024 | ImageBind term |

`<split>` = `<folder>_<list>`: `no_gender_English` etc. on dev. Rows follow the organiser's `.txt`, whose copy is saved
as `index_<split>.txt`. The organiser's VGGFace / ECAPA-192 come with their data and are not recomputed.

**MAV-Celeb v1 / v2 (optional, for STRESS-01).** ArcFace and ECAPA on the exact rows of `ext_<src>_{face,voice}_meta.csv`
(12 s centre crop, as their other features), saved as `ext_<src>_face_arcface.npy` etc. (fp16).

**Settings:** Accelerator **GPU T4 x2** (one GPU used), **Internet ON** (checkpoints ~7 GB, cached in `/tmp`).

**Inputs (Add Input):**
1. the organiser data with raw media: the v4 dataset (`train_set` + `dev_set`), and on 11/11 the Evaluation data.
   Any list `<folder>/<name>_test.txt` with its media beside it is found. Zips named `train_set*`, `dev_set*`,
   `eval*`, `test*` are unpacked if Kaggle did not extract them.
2. *optional*: MAV-Celeb v1 / v2 + `flag2027-ext-meta` (the four `ext_v{1,2}_complete_*_meta.csv`). Without them the
   v1/v2 part is skipped.
3. *optional, for the check*: `flag2027-feats-v2` and `flag2027-feats-models`. Every array that also exists there is
   compared row by row (cosine); the table is saved as `check_vs_reference.csv`.
4. *optional, resume*: this notebook's earlier output (`flag2027-feats-eval`). Its arrays are copied and finished work
   is skipped.

**Outputs:** `/kaggle/working/feats_eval/` -> dataset **`flag2027-feats-eval`**.
**Time on a T4:** v4 train + dev about 1 h 15 min (ImageBind audio and w2v2 are the slow ones); v1/v2 ArcFace + ECAPA
about 1 h (reading 40k voice files from the zips).
"""

ENV12 = r'''
import os, sys, io, re, time, json, types, shutil, zipfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)
LOCAL = os.environ.get("FLAG_LOCAL") == "1"          # CPU smoke test on a mini copy of the data
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORKDIR = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
CACHE = Path(os.environ.get("FLAG_CACHE", "/tmp/flag_cache"))           # checkpoints stay out of the output
ECAPA_DIR = Path(os.environ.get("FLAG_ECAPA", str(CACHE / "ecapa_sb")))
OUT = WORKDIR / "feats_eval"
for d in (WORKDIR, CACHE, OUT):
    d.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)

# ------------------------------------------------------------------ what to run (edit here)
ENCODERS = ["arcface", "ecapa", "vitage", "w2v2ag", "ibv", "iba"]   # ibv right before iba: one ImageBind load
EXT_SOURCES = ["v1_complete", "v2_complete"]                        # MAV-Celeb rows for STRESS-01; [] = skip
EXT_ENCODERS = ["arcface", "ecapa"]                                 # their ImageBind / age exist (FLAG_10)
SPLIT_FILTER = None            # e.g. ["no_gender/English"]; None = every split found (train + every test list)
ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")             # organiser zips unpacked when not extracted
ONLY_UNDER = None              # e.g. "eval": keep only lists whose path contains this (dev and eval both attached)
print("LOCAL" if LOCAL else "KAGGLE", "| input", IN, "| output", OUT, "| cache", CACHE)
'''

PIP12 = r'''
# speechbrain -U: Kaggle's preinstalled one breaks on recent torchaudio. insightface / ImageBind --no-deps: their
# declared deps would replace onnxruntime-gpu by the CPU build and pin an old torch. ImageBind's data loaders need
# pytorchvideo, so the two loaders used here are reimplemented below and its module is stubbed.
if not LOCAL:
    !pip -q install -U "speechbrain>=1.1.1" onnxruntime-gpu onnx iopath ftfy omegaconf fvcore 2>&1 | tail -2
    !pip -q install --no-deps "insightface==2.0" "git+https://github.com/facebookresearch/ImageBind" 2>&1 | tail -2
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEV,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

DATA12 = r'''
# ============================================================================ finding the data
def scan_inputs(inp: Path, media=("faces", "voices")):
    """One os.walk over the inputs that never enters a faces/ or voices/ folder (an extracted MAV-Celeb holds hundreds of
    thousands of files; every rglob over it costs minutes). The organiser csv live in features/{faces,voices}/: entered.
    -> {"files": name -> [paths], "roots": folders holding both faces/ and voices/}"""
    files, roots = {}, []
    for dp, dns, fns in os.walk(inp, followlinks=True):
        feat = Path(dp).name == "features"
        if not feat and all(m in dns for m in media):
            roots.append(Path(dp))
        dns[:] = sorted(d for d in dns if feat or d not in media)
        for f in fns:
            files.setdefault(f, []).append(Path(dp) / f)
    log(f"scanned {inp}: {sum(map(len, files.values()))} files outside media folders, {len(roots)} media roots")
    return dict(files=files, roots=roots)


def media_cols(txt: Path):
    """Column of the .wav and of the image in an organiser list (train: 2 / 3; test: 1 / 2), or None."""
    try:
        row = pd.read_csv(txt, sep=" ", header=None, nrows=1).iloc[0]
    except Exception:
        return None
    wav = [i for i, v in enumerate(row) if str(v).lower().endswith(".wav")]
    jpg = [i for i, v in enumerate(row) if str(v).lower().endswith((".jpg", ".jpeg", ".png"))]
    if not wav or not jpg:
        return None
    return wav[0], jpg[0]


def has_media(txt: Path, cols):
    row = pd.read_csv(txt, sep=" ", header=None, nrows=1).iloc[0]
    return (txt.parent / row[cols[0]]).exists() and (txt.parent / row[cols[1]]).exists()


def split_name(txt: Path):
    """train_English.txt -> 'train'; <folder>/English_test.txt -> '<folder>/English' (dev: 'no_gender/English')."""
    if txt.name == "train_English.txt":
        return "train"
    return f"{txt.parent.name}/{re.sub(r'_test$', '', txt.stem)}"


tag = lambda split: split.replace("/", "_")


def find_splits(scan):
    """Every organiser list whose media sit beside it. A features-only copy of a list (no media) is ignored.
    Two different lists with the same split name (dev and Evaluation both attached) stop the run: set ONLY_UNDER."""
    S = {}
    for name, paths in scan["files"].items():
        if not name.endswith(".txt") or not (name == "train_English.txt" or name.endswith("_test.txt")):
            continue
        for p in sorted(paths, key=lambda q: len(q.as_posix())):
            if ONLY_UNDER and ONLY_UNDER not in p.as_posix() and name != "train_English.txt":
                continue
            cols = media_cols(p)
            if cols is None or not has_media(p, cols):
                continue
            k = split_name(p)
            df = pd.read_csv(p, sep=" ", header=None)
            if k in S:
                same = len(df) == len(S[k]["wav"]) and df[cols[0]].tolist() == S[k]["wav"]
                assert same, f"two different lists named {k}: {S[k]['txt']} and {p} -> set ONLY_UNDER"
                continue
            S[k] = dict(txt=p, root=p.parent, wav=df[cols[0]].tolist(), jpg=df[cols[1]].tolist())
    return S


def unpack_zips(scan, work: Path):
    zips = [p for n, ps in scan["files"].items() if n.lower().endswith(".zip") and n.lower().startswith(ZIP_PREFIXES)
            for p in ps]
    for zp in zips:
        marker = work / f".{zp.stem}.done"
        if not marker.exists():
            log("unpacking", zp)
            with zipfile.ZipFile(zp) as z:
                z.extractall(work)
            marker.touch()
    return bool(zips)


SCAN = scan_inputs(IN)
S = find_splits(SCAN)
if "train" not in S or len(S) == 1:                  # something may still be zipped (Kaggle keeps some uploads as .zip)
    if unpack_zips(SCAN, WORKDIR / "raw"):
        for k, v in find_splits(scan_inputs(WORKDIR / "raw")).items():
            S.setdefault(k, v)
if SPLIT_FILTER:
    S = {k: v for k, v in S.items() if k in SPLIT_FILTER}
assert S, "no organiser list with media found: attach the raw v4 data (or the Evaluation data)"
for k, v in S.items():
    v["dur"] = np.array([sf.info(str(v["root"] / p)).duration for p in v["wav"]], dtype=np.float32)
    shutil.copy(v["txt"], OUT / f"index_{tag(k)}.txt")
    log(f"{k:22s} {len(v['wav']):5d} rows | voice median {np.median(v['dur']):.1f}s, max {v['dur'].max():.1f}s | {v['txt']}")
meta = SCAN["files"].get("meta_file_train_set.csv")
if meta:
    shutil.copy(meta[0], OUT / "meta_file_train_set.csv")


# ============================================================================ MAV-Celeb v1/v2 rows (optional)
def media_key(member):
    """'.../faces/<id>/<lang>/<video>/<file>' -> 'faces/<id>/<lang>/<video>/<file>' (drops any version folder)."""
    p = member.replace("\\", "/").split("/")
    for kind in ("faces", "voices"):
        if kind in p:
            return "/".join(p[p.index(kind):])
    return None


def ext_source(scan, name):
    """The attached zip for `name` (largest copy), else the extracted folder whose path names the source."""
    norm = lambda t: t.lower().replace("-", "_")
    zips = [p for n, ps in scan["files"].items() if n.lower().endswith(".zip") and norm(name) in norm(n) for p in ps]
    if zips:
        return max(zips, key=lambda p: p.stat().st_size)
    short = name.split("_")[0]
    for r in scan["roots"]:
        hint = norm(r.as_posix()) + "/"
        if f"/{short}/" in hint or name in hint:
            return r
    return None


class ExtRows:
    """The exact rows of ext_<name>_{face,voice}_meta.csv, read from the attached zip or folder (row order = csv)."""

    def __init__(self, src, name, meta_dir: Path, max_sec=12.0):
        fm = pd.read_csv(meta_dir / f"ext_{name}_face_meta.csv").path
        vm = pd.read_csv(meta_dir / f"ext_{name}_voice_meta.csv").path
        src = Path(src)
        if src.is_dir():
            from concurrent.futures import ThreadPoolExecutor
            self.face, self.voice = [src / media_key(p) for p in fm], [src / media_key(p) for p in vm]
            with ThreadPoolExecutor(32) as ex:
                ok = list(ex.map(lambda q: q.exists(), self.face + self.voice))
            if not all(ok):
                raise KeyError(f"{ok.count(False)} rows missing, e.g. {(self.face + self.voice)[ok.index(False)]}")
            self.read = lambda q: q.read_bytes()
        else:
            z = zipfile.ZipFile(src)
            by = {media_key(i.filename): i.filename for i in z.infolist() if not i.is_dir() and media_key(i.filename)}
            self.face, self.voice = [by[media_key(p)] for p in fm], [by[media_key(p)] for p in vm]
            self.read = z.read
        self.L = int(max_sec * 16000)          # 12 s centre crop, as the organiser-space features of these rows
        log(f"{name}: {len(self.face)} face rows, {len(self.voice)} voice rows in {src}")


EXT = {}
for name in EXT_SOURCES:
    metas = sorted(SCAN["files"].get(f"ext_{name}_voice_meta.csv", []), key=lambda p: -p.stat().st_size)
    src = ext_source(SCAN, name)
    if metas and src is not None:
        try:
            EXT[name] = ExtRows(src, name, metas[0].parent)
        except Exception as e:                    # optional: a bad zip or a missing row must not stop the run
            log(f"{name}: skipped ({type(e).__name__}: {e})")
    else:
        log(f"{name}: skipped (meta csv found: {bool(metas)}, source found: {src is not None})")


# ============================================================================ resume
def seed_from_previous(scan, out: Path):
    """Copy arrays of an earlier run of THIS notebook (a folder named like flag2027-feats-eval / feats_eval), so finished
    work is skipped. Arrays of other datasets (feats_v2, feats_models) are references for the check, never copied."""
    copied = []
    for n, ps in scan["files"].items():
        if not n.endswith(".npy") or (out / n).exists():
            continue
        for p in ps:
            if re.search(r"feats[-_]eval", p.as_posix()) and out not in p.parents:
                shutil.copy(p, out / n); copied.append(n); break
    log(f"resume: {len(copied)} arrays copied from an earlier run")
    return copied


SEEDED = seed_from_previous(SCAN, OUT)
print({k: len(v["wav"]) for k, v in S.items()}, "| external:", list(EXT))
'''

AUDIO12 = r'''
# ============================================================================ audio + batching
TARGET_SR = 16000
AUDIO_STATS = {"files": 0, "resampled": 0, "multichannel": 0}


def load_audio(src):
    """Mono float32 at 16 kHz as the organiser loads audio (librosa.load(sr=16000, mono=True): channel mean + soxr_hq).
    All v4 wavs are 16 kHz mono; MAV-Celeb v1 has 44.1 kHz files. `src` = path or file-like."""
    w, sr = sf.read(src if hasattr(src, "read") else str(src), dtype="float32", always_2d=True)
    AUDIO_STATS["files"] += 1
    if w.shape[1] > 1:
        AUDIO_STATS["multichannel"] += 1
        w = w.mean(1, keepdims=True)
    w = w[:, 0]
    if sr != TARGET_SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        AUDIO_STATS["resampled"] += 1
    return w


def centre_crop(w, L):
    if len(w) <= L:
        return w
    s = (len(w) - L) // 2
    return w[s:s + L]


def pack_batches(durs, max_padded_sec=240.0, max_bs=32):
    """Longest first; a batch costs len(batch) * longest item after padding."""
    order = np.argsort(-np.asarray(durs), kind="stable")
    batches, cur = [], []
    for i in order:
        longest = durs[cur[0]] if cur else durs[i]
        if cur and ((len(cur) + 1) * longest > max_padded_sec or len(cur) >= max_bs):
            batches.append(cur); cur = []
        cur.append(int(i))
    if cur:
        batches.append(cur)
    return batches


def run_batched(items, fn, durs, max_padded_sec=240.0, max_bs=32, cpu_fn=None, label=""):
    """fn(list of items) -> list of outputs, in item order. A CUDA OOM halves the batch; a single item goes to CPU."""
    out = [None] * len(items)
    t0, done_n = time.time(), 0

    def go(idx):
        try:
            res = fn([items[i] for i in idx])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                res = (cpu_fn or fn)([items[idx[0]]])
            else:
                h = len(idx) // 2
                go(idx[:h]); go(idx[h:])
                return
        for i, r in zip(idx, res):
            out[i] = r

    batches = pack_batches(durs, max_padded_sec, max_bs)
    for bi, b in enumerate(batches):
        go(b)
        done_n += len(b)
        if bi % 50 == 0 or done_n == len(items):
            el = time.time() - t0
            log(f"  {label} {done_n}/{len(items)}  {el:.0f}s  eta {el / done_n * (len(items) - done_n):.0f}s")
    return out


def cos_rows(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def amp(flag):
    return torch.autocast("cuda", dtype=torch.float16, enabled=bool(flag) and DEV.type == "cuda")


def read_rgb(src):
    from PIL import Image
    return Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src).convert("RGB")
'''

ENC12 = r'''
# ============================================================================ encoders
# Face encoders: embed(list of sources) -> {suffix: (N, d)}; "" is the main array. A source is a path or raw bytes.
# Voice encoders: embed(list of waveforms, durations) -> {suffix: (N, d)}.

class ArcFace:
    """insightface buffalo_l: SCRFD detection + 5-point norm_crop + ArcFace R100 (512). fp32, as FLAG_04."""
    kind = "face"

    def __init__(self):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        prov = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
        try:
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=prov)
        except TypeError:
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if DEV.type == "cuda" else -1, det_size=(224, 224), det_thresh=0.3)
        self.rec = self.app.models["recognition"]
        log("ArcFace recognition runs on", self.rec.session.get_providers())

    def align(self, img):
        """BGR image -> (112 x 112 crop on the ArcFace 5-point template, detected?). Keeps the big central face."""
        import cv2
        from insightface.utils import face_align
        faces = self.app.det_model.detect(img, max_num=0, metric="default")
        bboxes, kpss = faces if isinstance(faces, tuple) else (faces, None)
        if bboxes is not None and len(bboxes) and kpss is not None:
            h, w = img.shape[:2]
            ctr = np.array([w / 2, h / 2])
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            dist = np.linalg.norm((bboxes[:, :2] + bboxes[:, 2:4]) / 2 - ctr, axis=1)
            j = int(np.argmax(area - 2.0 * dist ** 2))
            return face_align.norm_crop(img, landmark=kpss[j], image_size=112), True
        h, w = img.shape[:2]; m = int(min(h, w) * 0.15)
        return cv2.resize(img[m:h - m, m:w - m], (112, 112)), False

    def embed(self, srcs, label=""):
        import cv2
        E, OK, t0 = [], [], time.time()
        for i, s in enumerate(srcs):
            img = cv2.imdecode(np.frombuffer(s, np.uint8), cv2.IMREAD_COLOR) if isinstance(s, (bytes, bytearray)) \
                else cv2.imread(str(s))
            assert img is not None, s
            a, ok = self.align(img)
            E.append(self.rec.get_feat(a).flatten().astype(np.float32)); OK.append(ok)
            if i % 2000 == 0:
                log(f"  arcface {label} {i}/{len(srcs)} {time.time() - t0:.0f}s")
        log(f"  arcface {label}: detection + alignment {100 * np.mean(OK):.1f}%")
        return {"": np.stack(E), "_detok": np.array(OK)}


def speechbrain_import():
    import torchaudio
    for n, f in [("list_audio_backends", lambda: ["soundfile"]), ("get_audio_backend", lambda: "soundfile"),
                 ("set_audio_backend", lambda *a, **k: None)]:
        if not hasattr(torchaudio, n):
            setattr(torchaudio, n, f)
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier


class Ecapa:
    """speechbrain/spkrec-ecapa-voxceleb -> 192 (ecapa192) + attentive-statistics pooling 6144 (ecapa6144). fp32."""
    kind = "voice"

    def __init__(self):
        EncoderClassifier = speechbrain_import()
        kw = {}
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        self.m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ECAPA_DIR),
                                                run_opts={"device": str(DEV)}, **kw)
        self.m.eval()
        self._pool = {}
        self.m.mods.embedding_model.asp.register_forward_hook(lambda mod, i, o: self._pool.__setitem__("x", o))

    @torch.no_grad()
    def batch(self, sigs, device=None):
        device = device or DEV
        lens = [len(s) for s in sigs]
        L = max(lens)
        x = torch.zeros(len(sigs), L)
        for i, s in enumerate(sigs):
            x[i, :len(s)] = torch.from_numpy(s)
        rel = torch.tensor([l / L for l in lens], dtype=torch.float32, device=device)
        mods = self.m.mods if device == DEV else self._cpu_mods()
        f = mods.mean_var_norm(mods.compute_features(x.to(device)), rel)
        e = mods.embedding_model(f, rel).squeeze(1)
        p = self._pool["x"].squeeze(-1)
        return list(zip(e.float().cpu().numpy(), p.float().cpu().numpy()))

    def _cpu_mods(self):
        if not hasattr(self, "_cpu"):
            import copy
            self._cpu = copy.deepcopy(self.m.mods).to("cpu")
            self._cpu.embedding_model.asp.register_forward_hook(lambda mod, i, o: self._pool.__setitem__("x", o))
        return self._cpu

    def embed(self, load, n, durs, label=""):
        res = run_batched(list(range(n)), lambda ii: self.batch([load(i) for i in ii]), durs,
                          cpu_fn=lambda ii: self.batch([load(i) for i in ii], device="cpu"), label=f"ecapa {label}")
        return {"192": np.stack([r[0] for r in res]).astype(np.float32),
                "6144": np.stack([r[1] for r in res]).astype(np.float32)}


class ViTAge:
    """nateraw/vit-age-classifier: post-LayerNorm CLS (768) + the 9 FairFace age-bin probabilities (fp16 autocast)."""
    kind = "face"
    ID = "nateraw/vit-age-classifier"

    def __init__(self):
        from transformers import ViTImageProcessor, ViTForImageClassification
        self.proc = ViTImageProcessor.from_pretrained(self.ID)
        self.m = ViTForImageClassification.from_pretrained(self.ID).to(DEV).eval()

    def prep(self, im):
        return self.proc(images=im, return_tensors="pt").pixel_values[0]

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        with amp(use_amp):
            h = self.m.vit(x.to(DEV)).last_hidden_state[:, 0]
            p = torch.softmax(self.m.classifier(h).float(), 1)
        return {"": h.float().cpu().numpy(), "prob": p.cpu().numpy()}


class AgeGenderHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))


class W2V2AgeGender:
    """audeering/wav2vec2-large-robust-24-ft-age-gender: mean hidden state (1024), age (x 100 = years), gender
    probabilities (female, male, child). 20 s windows, 10 s hop, length-weighted mean. fp16 autocast."""
    kind = "voice"
    ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"

    def __init__(self):
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = AgeGenderHead(config, 1)
                self.gender = AgeGenderHead(config, 3)
                self.post_init()        # not init_weights(): transformers >= 5 sets all_tied_weights_keys here

        # transformers 5.0 reads sys.modules[cls.__module__].__file__; a class defined in a notebook lives in __main__,
        # which has no __file__ under Jupyter (w2v2ag failed on Kaggle). Point it at the real wav2vec2 module.
        AgeGenderModel.__module__ = Wav2Vec2Model.__module__
        self.proc = Wav2Vec2Processor.from_pretrained(self.ID)
        self.m = AgeGenderModel.from_pretrained(self.ID).to(DEV).eval()
        self.W, self.H = 20 * 16000, 10 * 16000

    def windows(self, w):
        starts = [0] if len(w) <= self.W else list(range(0, len(w) - self.W + self.H, self.H))
        for i, s in enumerate(starts):
            seg = w[s:s + self.W]
            if i and len(seg) < 16000:
                break
            yield seg

    @torch.no_grad()
    def __call__(self, w, use_amp=True):
        acc, tot = None, 0
        for seg in self.windows(w):
            x = self.proc(seg, sampling_rate=16000, return_tensors="pt").input_values.to(DEV)
            with amp(use_amp):
                h = self.m.wav2vec2(x)[0].float().mean(1)[0]
            acc = h * len(seg) if acc is None else acc + h * len(seg)
            tot += len(seg)
        h = (acc / tot)[None]
        return {"": h[0].cpu().numpy(), "age": self.m.age(h)[0].cpu().numpy(),
                "gen": torch.softmax(self.m.gender(h), 1)[0].cpu().numpy()}


IB_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
_IB = {}


def imagebind():
    """ImageBind-huge, loaded once for vision and audio. imagebind/__init__ imports imagebind.data (needs
    pytorchvideo): stubbed, the two loaders used are reimplemented below."""
    if "m" not in _IB:
        sys.modules.setdefault("imagebind.data", types.ModuleType("imagebind.data"))
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        f = CACHE / "imagebind_huge.pth"
        if not f.exists():
            log("downloading", IB_URL)
            torch.hub.download_url_to_file(IB_URL, str(f) + ".part", progress=False)
            Path(str(f) + ".part").rename(f)
        m = imagebind_model.imagebind_huge(pretrained=False)
        m.load_state_dict(torch.load(f, map_location="cpu", mmap=True))
        _IB.update(m=m.to(DEV).eval(), MT=ModalityType)
    return _IB


class IBVision:
    kind = "face"

    def __init__(self):
        from torchvision import transforms as T
        self.ib = imagebind()
        self.tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def prep(self, im):
        return self.tf(im)

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        with amp(use_amp):
            e = self.ib["m"]({self.ib["MT"].VISION: x.to(DEV)})[self.ib["MT"].VISION]
        return {"": e.float().cpu().numpy()}


def ib_audio_clips(w, sr=16000, clip=2.0, mel=128, frames=204, mean=-4.268, std=9.138):
    """imagebind.data.load_and_transform_audio_data for one waveform, with n = max(3, ceil(duration / 2 s)) clips evenly
    spaced from start to end (covers the whole file; ImageBind's default n = 3 is identical up to 6 s)."""
    import torchaudio
    n = max(3, int(np.ceil(len(w) / sr / clip)))
    mx = max(len(w) / sr - clip, 0.0)
    out = []
    for k in range(n):
        s = mx * k / max(n - 1, 1)
        seg = torch.from_numpy(np.ascontiguousarray(w[int(s * sr):int((s + clip) * sr)], dtype=np.float32))[None]
        if seg.shape[1] < 400:
            seg = F.pad(seg, (0, 400 - seg.shape[1]))
        seg = seg - seg.mean()
        fb = torchaudio.compliance.kaldi.fbank(seg, htk_compat=True, sample_frequency=sr, use_energy=False,
                                               window_type="hanning", num_mel_bins=mel, dither=0.0,
                                               frame_length=25, frame_shift=10).T
        fb = F.pad(fb, (0, frames - fb.shape[1])) if fb.shape[1] < frames else fb[:, :frames]
        out.append((fb[None] - mean) / std)
    return torch.stack(out)


class IBAudio:
    kind = "voice"

    def __init__(self):
        self.ib = imagebind()

    @torch.no_grad()
    def __call__(self, w, use_amp=True):
        x = ib_audio_clips(w)[None].to(DEV)                       # the model averages the clips
        with amp(use_amp):
            e = self.ib["m"]({self.ib["MT"].AUDIO: x})[self.ib["MT"].AUDIO]
        return {"": e[0].float().cpu().numpy()}


def build(name):
    return {"arcface": ArcFace, "ecapa": Ecapa, "vitage": ViTAge, "w2v2ag": W2V2AgeGender,
            "ibv": IBVision, "iba": IBAudio}[name]()


def release(keep_ib=False):
    if not keep_ib:
        _IB.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
'''

RUN12 = r'''
# ============================================================================ running
def run_faces_tensor(enc, srcs, bs=64, label=""):
    out, t0 = {}, time.time()
    for s in range(0, len(srcs), bs):
        x = torch.stack([enc.prep(read_rgb(p() if callable(p) else p)) for p in srcs[s:s + bs]])
        for k, v in enc(x).items():
            out.setdefault(k, []).append(v)
        if (s // bs) % 40 == 0:
            log(f"  {label} {s + len(x)}/{len(srcs)}  {time.time() - t0:.0f}s")
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def run_voices_single(enc, n, load, label=""):
    """One file at a time: no padding enters a mean."""
    out, t0 = {}, time.time()
    for i in range(n):
        for k, v in enc(load(i)).items():
            out.setdefault(k, []).append(np.atleast_1d(v))
        if i % 1000 == 0 or i == n - 1:
            el = time.time() - t0
            log(f"  {label} {i + 1}/{n}  {el:.0f}s  eta {el / (i + 1) * (n - i - 1):.0f}s")
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


# output names: {kind}_{array}_{split}.npy for organiser splits, ext_{src}_{kind}_{array}.npy for v1/v2
ARRAYS = {"arcface": ("face", {"": "arcface", "_detok": "arcface_detok"}),
          "ecapa": ("voice", {"192": "ecapa192", "6144": "ecapa6144"}),
          "vitage": ("face", {"": "vitage", "prob": "vitageprob"}),
          "w2v2ag": ("voice", {"": "w2v2ag", "age": "w2v2agage", "gen": "w2v2aggen"}),
          "ibv": ("face", {"": "ibv"}), "iba": ("voice", {"": "iba"})}
MAIN = {"arcface": "", "ecapa": "192", "vitage": "", "w2v2ag": "", "ibv": "", "iba": ""}


def target(name, suffix, split=None, src=None):
    kind, names = ARRAYS[name]
    return OUT / (f"{kind}_{names[suffix]}_{tag(split)}.npy" if split else f"ext_{src}_{kind}_{names[suffix]}.npy")


def complete(name, split=None, src=None):
    return all(target(name, s, split, src).exists() for s in ARRAYS[name][1])


def save(name, R, split=None, src=None):
    for s, v in R.items():                         # main array last: its presence marks the job as done
        if s != MAIN[name]:
            np.save(target(name, s, split, src), v if split else v.astype(np.float16))
    np.save(target(name, MAIN[name], split, src), R[MAIN[name]] if split else R[MAIN[name]].astype(np.float16))


def extract(name, enc, split=None, src=None):
    kind = ARRAYS[name][0]
    if split:
        sp = S[split]
        faces = [sp["root"] / p for p in sp["jpg"]]
        load = lambda i: load_audio(sp["root"] / sp["wav"][i])
        n, durs, label = len(sp["wav"]), sp["dur"], split
    else:
        rows = EXT[src]
        faces = [rows.read(p) for p in rows.face] if name == "arcface" else [(lambda p=p: rows.read(p)) for p in rows.face]
        load = lambda i: centre_crop(load_audio(io.BytesIO(rows.read(rows.voice[i]))), rows.L)
        n, label = len(rows.voice), src
        durs = np.full(n, 12.0, np.float32)          # upper bound after the crop: batches stay within budget
    if kind == "face":
        R = enc.embed(faces, label=label) if name == "arcface" else run_faces_tensor(enc, faces, label=f"{name} {label}")
    elif name == "ecapa":
        R = enc.embed(load, n, durs, label=label)
    else:
        R = run_voices_single(enc, n, load, label=f"{name} {label}")
    save(name, R, split, src)
    log(f"saved {target(name, MAIN[name], split, src).name} {R[MAIN[name]].shape}")


def self_check(name, enc):
    """Batching / fp16 must not change the embeddings (min cosine over a few rows; must be > 0.995)."""
    k = next(k for k in S if k != "train") if len(S) > 1 else next(iter(S))
    sp = S[k]
    if name == "arcface":
        return {}
    if name == "ecapa":
        idx = list(np.argsort(-sp["dur"])[:3]) + list(range(min(5, len(sp["wav"]))))
        sigs = [load_audio(sp["root"] / sp["wav"][i]) for i in idx]
        b, s = enc.batch(sigs), [enc.batch([x])[0] for x in sigs]
        return {"batch192": float(cos_rows([r[0] for r in b], [r[0] for r in s]).min()),
                "batch6144": float(cos_rows([r[1] for r in b], [r[1] for r in s]).min())}
    if ARRAYS[name][0] == "face":
        x = torch.stack([enc.prep(read_rgb(sp["root"] / p)) for p in sp["jpg"][:6]])
        full = enc(x, use_amp=False)[""]
        single = np.concatenate([enc(x[i:i + 1], use_amp=False)[""] for i in range(len(x))])
        return {"batch": float(cos_rows(full, single).min()), "fp16": float(cos_rows(full, enc(x)[""]).min())}
    waves = [load_audio(sp["root"] / p) for p in sp["wav"][:3]]
    return {"fp16": float(cos_rows([enc(w, use_amp=False)[""] for w in waves], [enc(w)[""] for w in waves]).min())}


CHECKS, FAILED = {}, {}
for name in ENCODERS:
    jobs = [("split", k) for k in S if not complete(name, split=k)]
    if name in EXT_ENCODERS:
        jobs += [("src", s) for s in EXT if not complete(name, src=s)]
    if not jobs:
        log(f"== {name}: every array already exists -> skipped")
        continue
    t0 = time.time()
    try:
        enc = build(name)
        CHECKS[name] = self_check(name, enc)
        log(name, "self-check:", CHECKS[name])
        assert not CHECKS[name] or min(CHECKS[name].values()) > 0.995, f"{name}: batching / fp16 changes the output"
        for what, key in jobs:
            try:
                extract(name, enc, **({"split": key} if what == "split" else {"src": key}))
            except Exception as e:                  # one bad source must not cost the other splits
                FAILED[f"{name}:{key}"] = f"{type(e).__name__}: {e}"
                log(f"!! {name} on {key} FAILED: {FAILED[f'{name}:{key}']}")
    except Exception as e:
        FAILED[name] = f"{type(e).__name__}: {e}"
        log(f"!! {name} FAILED -> skipped: {FAILED[name]}")
    enc = None
    release(keep_ib=(name == "ibv"))
    log(f"== {name} done in {time.time() - t0:.0f}s")
print("failed:", FAILED or "none")
'''

CHECK12 = r'''
# ============================================================================ check against the arrays already in use
# Every array written here that also exists in an attached reference dataset (flag2027-feats-v2: arcface, ecapa;
# flag2027-feats-models: ibv, iba, vitage*, w2v2ag*) is compared row by row. Expected: cosine ~1.000 for fp32 models
# (ArcFace, ECAPA); >= 0.999 for the fp16 ones. A lower value means a different library version or preprocessing.
rows = []
for f in sorted(OUT.glob("*.npy")):
    refs = [p for p in SCAN["files"].get(f.name, []) if not re.search(r"feats[-_]eval", p.as_posix())]
    if not refs:
        continue
    a, b = np.load(f).astype(np.float64), np.load(refs[0]).astype(np.float64)
    if a.shape != b.shape:
        rows.append(dict(array=f.name, reference=str(refs[0].parent), note=f"shape {a.shape} vs {b.shape}")); continue
    if a.ndim == 1 or a.shape[1] < 2:
        d = np.abs(a - b).reshape(len(a), -1).max(1)
        rows.append(dict(array=f.name, reference=refs[0].parent.name, rows=len(a), max_abs_diff=float(d.max()),
                         mean_abs_diff=float(d.mean())))
    else:
        c = cos_rows(a, b)
        rows.append(dict(array=f.name, reference=refs[0].parent.name, rows=len(a), cos_min=round(float(c.min()), 5),
                         cos_mean=round(float(c.mean()), 6), rows_below_0999=int((c < 0.999).sum())))
CHECK = pd.DataFrame(rows)
if len(CHECK):
    CHECK.to_csv(OUT / "check_vs_reference.csv", index=False)
    print(CHECK.to_string(index=False))
else:
    print("no reference dataset attached: nothing to compare (optional)")
'''

DONE12 = r'''
man = {"created": time.strftime("%Y-%m-%d %H:%M"), "splits": {k: len(v["wav"]) for k, v in S.items()},
       "external": {k: [len(r.face), len(r.voice)] for k, r in EXT.items()}, "encoders": ENCODERS,
       "checks": CHECKS, "failed": FAILED, "audio": AUDIO_STATS, "resumed_arrays": len(SEEDED),
       "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
       "files": {f.name: list(np.load(f, mmap_mode="r").shape) for f in sorted(OUT.glob("*.npy"))}}
try:
    import speechbrain, insightface
    man["versions"].update(speechbrain=speechbrain.__version__, insightface=insightface.__version__)
except Exception:
    pass
(OUT / "feats_manifest.json").write_text(json.dumps(man, indent=1))
shutil.rmtree(WORKDIR / "raw", ignore_errors=True)
tot = sum(p.stat().st_size for p in OUT.glob("*")) / 1e9
print(f"{len(man['files'])} arrays, {tot:.2f} GB in {OUT} | failed: {FAILED or 'none'}")
print("NEXT: Save Version (Save & Run All) -> Output -> New Dataset -> name it flag2027-feats-eval")
'''

nb([("markdown", INTRO12), ("code", ENV12), ("code", PIP12),
    ("markdown", "## 1. Find the data: organiser lists with media (row order = the .txt), v1/v2 rows, resume"),
    ("code", DATA12),
    ("markdown", "## 2. Audio loading, batching, helpers"), ("code", AUDIO12),
    ("markdown", "## 3. Encoders (full code; one loaded at a time, ImageBind shared by vision and audio)"),
    ("code", ENC12),
    ("markdown", "## 4. Extract: each encoder over every split, then the v1/v2 rows"), ("code", RUN12),
    ("markdown", "## 5. Check against the arrays already in use (if their datasets are attached)"), ("code", CHECK12),
    ("markdown", "## 6. Manifest"), ("code", DONE12)], "FLAG_12_features.ipynb")
