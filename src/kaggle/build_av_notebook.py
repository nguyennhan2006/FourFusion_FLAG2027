"""Generate FLAG_14_av_zeroshot.ipynb: other pretrained image<->audio models, zero-shot, as possible partners of ImageBind
(docs/OPEN_DIRECTIONS.md, user request 29/09). Full code in the cells; the data-finding and audio cells are the ones of
FLAG_12 (already run on Kaggle). LanguageBind runs in a child process with transformers 4.57.3 (it breaks on 5.x).
    python build_av_notebook.py
    FLAG_LOCAL=1 FLAG_IN=<mini inputs> FLAG_OUT=<dir> FLAG_CACHE=<cache> python run_notebook_local.py FLAG_14_av_zeroshot.ipynb --jupyter
"""
import json
from pathlib import Path
import build_eval_notebooks as B12

HERE = Path(__file__).resolve().parent


def nb(cells, path):
    d = {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                    **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
         "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11"}},
         "nbformat": 4, "nbformat_minor": 5}
    (HERE / path).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", path, len(cells), "cells")


INTRO = """
# FLAG 2027 — NB-14: other image <-> audio models, zero-shot (partners for ImageBind)

ImageBind zero-shot is the largest gain we have (-5...-9 EER). A second model with a shared image-audio space, trained on
other data, may add to it even if it is weaker alone. Frozen models only, no training.

| encoder | face side | voice side | licence | source |
|---|---|---|---|---|
| `ibalign` | ImageBind-huge vision on the **ArcFace-aligned** face crop (112 -> 224) | (use the existing `voice_iba`) | CC-BY-NC 4.0 | facebookresearch/ImageBind |
| `w2c` + `clipb32` / `clipb16` | CLIP ViT-B/32 and ViT-B/16 image embedding | Wav2CLIP (audio distilled into CLIP image space) | MIT | descriptinc/lyrebird-wav2clip, openai CLIP (HF) |
| `aclip` | AudioCLIP image head (CLIP RN50, fine-tuned) | AudioCLIP audio head (ESResNeXt-fbsp, 44.1 kHz) | MIT | AndreyGuzhov/AudioCLIP |
| `lbind` | LanguageBind_Image (ViT-L/14) | LanguageBind_Audio_FT | MIT | PKU-YuanGroup/LanguageBind |

Not included: CAV-MAE (the Dropbox weights are gone), ONE-PEACE (4B parameters; later, only if these show a signal),
Speech2Face (no public weights).

**Rows:** the organiser splits found in the inputs (v4 train + the 4 dev files; rows = the organiser .txt) and a fixed
subset of MAV-Celeb v1 / v2: up to 16 voices per person and language, up to 20 faces per person (seed 0); the chosen
row numbers of `ext_<src>_{face,voice}_meta.csv` are saved as `ext_<src>_{face,voice}_rows.npy`. Voices of v1/v2: 12 s
centre crop, as their other features.

**Settings:** GPU T4, **Internet ON** (weights ~6 GB, cached in /tmp). **Inputs:** the raw v4 data; MAV-Celeb v1 / v2 +
`flag2027-ext-meta`; *optional* `flag2027-feats-models` (the probe then also reports ImageBind and correlations).
**Output:** `/kaggle/working/feats_av/` -> dataset `flag2027-feats-av`. **Time:** about 2 h on a T4.
**Resume:** attach the earlier output; finished arrays are skipped.
"""

ENV = r'''
import os, sys, io, re, time, json, types, shutil, zipfile, warnings, subprocess
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORKDIR = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
CACHE = Path(os.environ.get("FLAG_CACHE", "/tmp/flag_cache"))
REPOS = Path(os.environ.get("FLAG_REPOS", "/tmp/repos"))
OUT = WORKDIR / "feats_av"
for d in (WORKDIR, CACHE, OUT, REPOS):
    d.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)

ENCODERS = ["ibalign", "clipb32", "clipb16", "w2c", "aclip", "lbind"]
EXT_SOURCES = ["v1_complete", "v2_complete"]
EXT_VOICES_PER_LANG, EXT_FACES = 16, 20                 # per person; seed 0
SPLIT_FILTER = None
ZIP_PREFIXES = ("train_set", "dev_set", "eval", "test")
ONLY_UNDER = None
print("LOCAL" if LOCAL else "KAGGLE", "| input", IN, "| output", OUT)
'''

PIP = r'''
# ImageBind / insightface as in FLAG_12; Wav2CLIP --no-deps (pure python; librosa is preinstalled); AudioCLIP and
# LanguageBind are cloned (not pip packages). LanguageBind gets its own transformers 4.57.3 for its child process.
if not LOCAL:
    !pip -q install onnxruntime-gpu onnx iopath ftfy omegaconf fvcore einops 2>&1 | tail -2
    !pip -q install --no-deps "insightface==2.0" "git+https://github.com/facebookresearch/ImageBind" wav2clip 2>&1 | tail -2
    !pip -q install --target /tmp/tf457 "transformers==4.57.3" "tokenizers>=0.22,<0.23" "huggingface_hub<1.0" "safetensors" 2>&1 | tail -2
    !git clone -q --depth 1 https://github.com/AndreyGuzhov/AudioCLIP.git /tmp/repos/AudioCLIP
    !git clone -q --depth 1 https://github.com/PKU-YuanGroup/LanguageBind.git /tmp/repos/LanguageBind
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEV)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''


def data_cell():
    t = B12.DATA12
    for s in ['re.search(r"feats[-_]eval", p.as_posix())', "of THIS notebook (a folder named like flag2027-feats-eval / feats_eval)",
              "            self.read = z.read\n        self.L = int(max_sec * 16000)",
              "            self.read = lambda q: q.read_bytes()\n        else:"]:
        assert t.count(s) == 1, s
    t = t.replace('re.search(r"feats[-_]eval", p.as_posix())', 're.search(r"feats[-_]av", p.as_posix())')
    t = t.replace("of THIS notebook (a folder named like flag2027-feats-eval / feats_eval)",
                  "of THIS notebook (a folder named like flag2027-feats-av / feats_av)")
    t = t.replace("""            self.read = z.read
        self.L = int(max_sec * 16000)""", """            self.read = z.read
        self.zip = str(src) if not src.is_dir() else None
        self.meta_dir = meta_dir
        self.L = int(max_sec * 16000)""")
    t = t.replace("""            self.read = lambda q: q.read_bytes()
        else:""", """            self.read = lambda q: q.read_bytes()
            self.zip = None
        else:""")
    t += r'''

# ---- the fixed v1 / v2 subset (the other features of these rows exist already, by row number)
for name, rows in EXT.items():
    fm = pd.read_csv(rows.meta_dir / f"ext_{name}_face_meta.csv"); vm = pd.read_csv(rows.meta_dir / f"ext_{name}_voice_meta.csv")
    rng = np.random.RandomState(0)
    vsel = np.sort(np.concatenate([rng.permutation(g.index.values)[:EXT_VOICES_PER_LANG] for _, g in vm.groupby(["spk", "lang"])]))
    fsel = np.sort(np.concatenate([rng.permutation(g.index.values)[:EXT_FACES] for _, g in fm.groupby("spk")]))
    rows.fsel, rows.vsel = fsel, vsel
    np.save(OUT / f"ext_{name}_face_rows.npy", fsel); np.save(OUT / f"ext_{name}_voice_rows.npy", vsel)
    log(f"{name}: subset {len(fsel)} faces / {len(vsel)} voices of {len(fm)} / {len(vm)}")
'''
    return t


AUDIO = B12.AUDIO12 + r'''

def load_audio_sr(src, sr_out):
    """Mono float32 at `sr_out` from the ORIGINAL file rate (no detour through 16 kHz).
    Resampler pinned to polyphase (scipy resample_poly): it reproduces the arrays of the Kaggle image of September 2026
    that BEST v9 was built on (AudioCLIP voices: cos 0.99998). The librosa default of the October image (soxr_hq) moves
    AudioCLIP voices to cos 0.973 (log 133)."""
    try:
        if hasattr(src, "seek"):
            src.seek(0)
        w, sr = sf.read(src if hasattr(src, "read") else str(src), dtype="float32", always_2d=True)
        if w.size == 0 or not np.isfinite(w).all():
            raise ValueError(f"{w.shape[0]} samples, finite {bool(np.isfinite(w).all())}")
    except Exception as e:                             # as load_audio: 1 s of silence, listed in BROKEN (when defined)
        if "BROKEN" in globals():
            BROKEN["audio"].append(f"{_src_name(src) if '_src_name' in globals() else src} | {sr_out} Hz, {type(e).__name__}: {e}")
        print("!! unreadable audio -> 1 s of silence:", getattr(src, "name", src), type(e).__name__, e)
        return np.zeros(sr_out, np.float32)
    w = w.mean(1)
    if sr != sr_out:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=sr_out, res_type="polyphase").astype(np.float32)
    if len(w) < sr_out // 10:                          # as load_audio: clips < 0.1 s are tiled to 0.1 s
        w = np.tile(w, int(np.ceil((sr_out // 10) / len(w))))[:sr_out // 10]
    return w
'''

ENC = r'''
# ============================================================================ encoders
# face encoder: .embed_faces(list of sources: path or bytes) -> (N, d); voice encoder: .embed_voice(src, crop_sec) -> (d,)
IMG_NORM_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def clip_tf(size=224):
    from torchvision import transforms as T
    return T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size), T.ToTensor(),
                      T.Normalize(*IMG_NORM_CLIP)])


def batched_faces(srcs, prep, fwd, bs=32, label=""):
    out, t0 = [], time.time()
    for s in range(0, len(srcs), bs):
        x = torch.stack([prep(p) for p in srcs[s:s + bs]]).to(DEV)
        with torch.no_grad(), amp(True):
            out.append(fwd(x).float().cpu().numpy())
        if (s // bs) % 50 == 0:
            log(f"  {label} {s + len(x)}/{len(srcs)} {time.time() - t0:.0f}s")
    return np.concatenate(out).astype(np.float32)


def crop_sec(w, sr, sec):
    L = int(sec * sr)
    if not sec or len(w) <= L:
        return w
    s = (len(w) - L) // 2
    return w[s:s + L]


class IBAligned:
    """ImageBind-huge vision on the ArcFace 5-point aligned crop (the SCRFD + norm_crop of FLAG_12), resized to 224."""
    kind = "face"

    def __init__(self):
        sys.modules.setdefault("imagebind.data", types.ModuleType("imagebind.data"))
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        from insightface.app import FaceAnalysis
        f = CACHE / "imagebind_huge.pth"
        if not f.exists():
            torch.hub.download_url_to_file("https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth", str(f) + ".part", progress=False)
            Path(str(f) + ".part").rename(f)
        m = imagebind_model.imagebind_huge(pretrained=False)
        m.load_state_dict(torch.load(f, map_location="cpu", mmap=True))
        self.m, self.MT = m.to(DEV).eval(), ModalityType
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        self.app.prepare(ctx_id=0 if DEV.type == "cuda" else -1, det_size=(224, 224), det_thresh=0.3)
        self.tf = clip_tf(224)

    def crop(self, s):
        import cv2
        from PIL import Image
        from insightface.utils import face_align
        img = cv2.imdecode(np.frombuffer(s, np.uint8), cv2.IMREAD_COLOR) if isinstance(s, (bytes, bytearray)) else cv2.imread(str(s))
        faces = self.app.det_model.detect(img, max_num=0, metric="default")
        bb, kp = faces if isinstance(faces, tuple) else (faces, None)
        if bb is not None and len(bb) and kp is not None:
            h, w = img.shape[:2]; ctr = np.array([w / 2, h / 2])
            j = int(np.argmax((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1]) - 2.0 * np.linalg.norm((bb[:, :2] + bb[:, 2:4]) / 2 - ctr, axis=1) ** 2))
            a = face_align.norm_crop(img, landmark=kp[j], image_size=112)
        else:
            h, w = img.shape[:2]; mg = int(min(h, w) * 0.15); a = cv2.resize(img[mg:h - mg, mg:w - mg], (112, 112))
        return self.tf(Image.fromarray(a[:, :, ::-1]))

    def embed_faces(self, srcs, label=""):
        return {"": batched_faces(srcs, self.crop, lambda x: self.m({self.MT.VISION: x})[self.MT.VISION], label=f"ibalign {label}")}


class CLIPImage:
    """openai CLIP image embedding (HF weights), the space Wav2CLIP audio was distilled into."""
    kind = "face"

    def __init__(self, rid):
        from transformers import CLIPModel
        self.m = CLIPModel.from_pretrained(rid).to(DEV).eval(); self.tf = clip_tf(224)

    def embed_faces(self, srcs, label=""):
        def fwd(x):
            o = self.m.get_image_features(pixel_values=x)
            return o.pooler_output if hasattr(o, "pooler_output") else o          # transformers 5 returns an object
        return {"": batched_faces(srcs, lambda s: self.tf(read_rgb(s)), fwd, label=f"clip {label}")}


class Wav2CLIP:
    kind = "voice"

    def __init__(self):
        import wav2clip
        self.m = wav2clip.get_model(device=str(DEV))
        self.m.eval()

    @torch.no_grad()
    def embed_voice(self, src, crop):
        w = crop_sec(load_audio_sr(src, 16000), 16000, crop)
        return {"": self.m(torch.from_numpy(w[None]).to(DEV)).float().cpu().numpy()[0]}


class AudioCLIPEnc:
    """AudioCLIP-Full-Training: audio 44.1 kHz in the int16 range (as its training datasets), image = CLIP RN50 head."""
    kind = "both"

    def __init__(self):
        st = types.ModuleType("ignite_trainer")                       # only its base classes are needed by the model code
        st.AbstractNet = type("AbstractNet", (torch.nn.Module,), {}); st.AbstractTransform = type("AbstractTransform", (object,), {})
        sys.modules["ignite_trainer"] = st
        # The repo keeps its BPE vocabulary in git-lfs. Without git-lfs in the image (Kaggle, Python 3.13 image of Oct 2026)
        # `git clone` leaves a text pointer ("version https://git-lfs...") and AudioCLIP fails with BadGzipFile. Fetch the real
        # file (identical copy in openai/CLIP, md5 933b7abbbbde62c36f02f0e6ccde464f) when the clone has only the pointer.
        bpe = REPOS / "AudioCLIP" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe.exists() or bpe.read_bytes()[:2] != b"\x1f\x8b":
            import hashlib
            for url in ("https://media.githubusercontent.com/media/AndreyGuzhov/AudioCLIP/master/assets/bpe_simple_vocab_16e6.txt.gz",
                        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"):
                try:
                    torch.hub.download_url_to_file(url, str(bpe), progress=False)
                    if hashlib.md5(bpe.read_bytes()).hexdigest() == "933b7abbbbde62c36f02f0e6ccde464f":
                        print("AudioCLIP BPE vocabulary fetched from", url); break
                except Exception as e:
                    print("AudioCLIP BPE vocabulary: download failed from", url, type(e).__name__, e)
            assert hashlib.md5(bpe.read_bytes()).hexdigest() == "933b7abbbbde62c36f02f0e6ccde464f", "AudioCLIP BPE vocabulary missing"
        sys.path.insert(0, str(REPOS / "AudioCLIP"))
        from model import AudioCLIP
        f = CACHE / "AudioCLIP-Full-Training.pt"
        if not f.exists():
            torch.hub.download_url_to_file("https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt", str(f), progress=False)
        self.m = AudioCLIP(pretrained=str(f)).to(DEV).eval(); self.tf = clip_tf(224)

    def embed_faces(self, srcs, label=""):
        def fwd(x):
            ((_, f, _), _), _ = self.m(image=x.float())
            return f
        return {"": batched_faces(srcs, lambda s: self.tf(read_rgb(s)), fwd, label=f"aclip {label}")}

    @torch.no_grad()
    def embed_voice(self, src, crop):
        w = crop_sec(load_audio_sr(src, 44100), 44100, crop) * 32768.0
        ((a, _, _), _), _ = self.m(audio=torch.from_numpy(w[None, None]).to(DEV))
        return {"": a.float().cpu().numpy()[0]}


def build(name):
    if name == "ibalign":
        return IBAligned()
    if name in ("clipb32", "clipb16"):
        return CLIPImage({"clipb32": "openai/clip-vit-base-patch32", "clipb16": "openai/clip-vit-base-patch16"}[name])
    return {"w2c": Wav2CLIP, "aclip": AudioCLIPEnc}[name]()


def release():
    import gc
    gc.collect(); torch.cuda.empty_cache()
'''

LBIND = r'''
# ============================================================================ LanguageBind, in a child process (transformers 4.57.3)
LB_SCRIPT = r"""
import sys, io, json, types, zipfile, time
import numpy as np, torch, torch.nn.functional as F, soundfile as sf
job = json.load(open(sys.argv[1]))
import torchaudio
for n, f in [("set_audio_backend", lambda *a, **k: None), ("get_audio_backend", lambda: "soundfile")]:
    if not hasattr(torchaudio, n):
        setattr(torchaudio, n, f)
import transformers.models.clip.modeling_clip as mc
if not hasattr(mc, "_expand_mask"):                      # removed after transformers 4.3x; LanguageBind imports it
    def _expand_mask(mask, dtype, tgt_len=None):
        b, s = mask.size(); t = tgt_len if tgt_len is not None else s
        inv = 1.0 - mask[:, None, None, :].expand(b, 1, t, s).to(dtype)
        return inv.masked_fill(inv.to(torch.bool), torch.finfo(dtype).min)
    mc._expand_mask = _expand_mask
def emb_forward(self, pixel_values, interpolate_pos_encoding=False):   # 4.30 behaviour: no square-size check (audio 112 x 1036)
    b = pixel_values.shape[0]
    x = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
    x = torch.cat([self.class_embedding.expand(b, 1, -1), x], dim=1)
    return x + self.position_embedding(self.position_ids)
mc.CLIPVisionEmbeddings.forward = emb_forward
pkg = types.ModuleType("languagebind"); pkg.__path__ = [job["repo"] + "/languagebind"]; sys.modules["languagebind"] = pkg
from languagebind.audio.modeling_audio import LanguageBindAudio
from languagebind.audio.processing_audio import get_audio_transform
from languagebind.image.modeling_image import LanguageBindImage
dev = "cuda" if torch.cuda.is_available() else "cpu"
def load(cls, rid):
    m = cls.from_pretrained(rid).to(dev).eval()
    for mod in m.modules():
        c = getattr(mod, "config", None)
        if c is not None:
            try:
                c._attn_implementation = "eager"
            except Exception:
                pass
        if isinstance(getattr(mod, "position_ids", None), torch.Tensor):
            mod.position_ids.copy_(torch.arange(mod.position_ids.shape[-1]).expand_as(mod.position_ids))
    return m
zips = {}
def read(item):
    if item[0] == "path":
        return open(item[1], "rb").read()
    if item[1] not in zips:
        zips[item[1]] = zipfile.ZipFile(item[1])
    return zips[item[1]].read(item[2])
g = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o
for task in job["tasks"]:
    t0 = time.time()
    if task["kind"] == "face":
        from PIL import Image
        from torchvision import transforms as T
        m = load(LanguageBindImage, "LanguageBind/LanguageBind_Image")
        tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        out = []
        for s in range(0, len(task["items"]), 32):
            x = torch.stack([tf(Image.open(io.BytesIO(read(it))).convert("RGB")) for it in task["items"][s:s + 32]]).to(dev)
            with torch.no_grad():
                out.append(g(m.get_image_features(pixel_values=x)).float().cpu().numpy())
    else:
        m = load(LanguageBindAudio, "LanguageBind/LanguageBind_Audio_FT")
        tf = get_audio_transform(m.config)
        out = []
        for it in task["items"]:
            w, sr = sf.read(io.BytesIO(read(it)), dtype="float32", always_2d=True)
            w = w.mean(1)
            if task["crop"] and len(w) > task["crop"] * sr:
                L = int(task["crop"] * sr); s0 = (len(w) - L) // 2; w = w[s0:s0 + L]
            x = tf((torch.from_numpy(np.ascontiguousarray(w))[None], sr))
            with torch.no_grad():
                out.append(g(m.get_image_features(pixel_values=x[None].to(dev))).float().cpu().numpy())
    np.save(task["out"], np.concatenate(out).astype(np.float32 if task["fp32"] else np.float16))
    print(task["out"], len(task["items"]), f"{time.time() - t0:.0f}s", flush=True)
    del m; torch.cuda.empty_cache()
"""


def lbind_items(kind, split=None, src=None):
    if split:
        sp = S[split]
        return [["path", str(sp["root"] / p)] for p in (sp["jpg"] if kind == "face" else sp["wav"])]
    rows = EXT[src]
    sel = rows.fsel if kind == "face" else rows.vsel
    lst = rows.face if kind == "face" else rows.voice
    return [["zip", rows.zip, lst[i]] if rows.zip else ["path", str(lst[i])] for i in sel]


def run_lbind():
    tasks = []
    for kind in ("face", "voice"):
        for k in S:
            f = OUT / f"{kind}_lbind_{tag(k)}.npy"
            if not f.exists():
                tasks.append(dict(kind=kind, items=lbind_items(kind, split=k), out=str(f), crop=0, fp32=True))
        for s in EXT:
            f = OUT / f"ext_{s}_{kind}_lbind.npy"
            if not f.exists():
                tasks.append(dict(kind=kind, items=lbind_items(kind, src=s), out=str(f), crop=12, fp32=False))
    if not tasks:
        log("== lbind: every array already exists -> skipped"); return
    tasks.sort(key=lambda t: t["kind"])                        # one model load per kind
    (WORKDIR / "lb_job.json").write_text(json.dumps(dict(repo=str(REPOS / "LanguageBind"), tasks=tasks)))
    (WORKDIR / "lb_run.py").write_text(LB_SCRIPT)
    env = dict(os.environ)
    if not LOCAL:
        env["PYTHONPATH"] = "/tmp/tf457" + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, str(WORKDIR / "lb_run.py"), str(WORKDIR / "lb_job.json")], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-3000:] if r.returncode else "")
    if r.returncode:
        raise RuntimeError(f"LanguageBind child process failed ({r.returncode})")
'''

RUN = r'''
# ============================================================================ run: every encoder over the organiser splits, then the v1/v2 subset
def face_srcs(split=None, src=None):
    if split:
        return [S[split]["root"] / p for p in S[split]["jpg"]]
    rows = EXT[src]
    return [rows.read(rows.face[i]) for i in rows.fsel]


def voice_list(split=None, src=None):
    if split:
        return [S[split]["root"] / p for p in S[split]["wav"]], 0
    rows = EXT[src]
    return [io.BytesIO(rows.read(rows.voice[i])) for i in rows.vsel], 12


def out_name(enc, kind, split=None, src=None):
    return OUT / (f"{kind}_{enc}_{tag(split)}.npy" if split else f"ext_{src}_{kind}_{enc}.npy")


FAILED = {}
for name in ENCODERS:
    t0 = time.time()
    if name == "lbind":
        try:
            run_lbind()
        except Exception as e:
            FAILED[name] = f"{type(e).__name__}: {e}"; log("!! lbind FAILED:", FAILED[name])
        continue
    kinds = {"ibalign": ["face"], "clipb32": ["face"], "clipb16": ["face"], "w2c": ["voice"], "aclip": ["face", "voice"]}[name]
    jobs = [(k, sp, None) for k in kinds for sp in S] + [(k, None, s) for k in kinds for s in EXT]
    jobs = [j for j in jobs if not out_name(name, *j).exists()]
    if not jobs:
        log(f"== {name}: every array already exists -> skipped"); continue
    try:
        enc = build(name)
        for kind, split, src in jobs:
            try:
                if kind == "face":
                    X = enc.embed_faces(face_srcs(split, src), label=split or src)[""]
                else:
                    lst, crop = voice_list(split, src)
                    X, tv = [], time.time()
                    for i, v in enumerate(lst):
                        X.append(enc.embed_voice(v, crop)[""])
                        if i % 2000 == 0:
                            log(f"  {name} {split or src} {i}/{len(lst)} {time.time() - tv:.0f}s")
                    X = np.stack(X).astype(np.float32)
                np.save(out_name(name, kind, split, src), X if split else X.astype(np.float16))
                log(f"saved {out_name(name, kind, split, src).name} {X.shape}")
            except Exception as e:
                FAILED[f"{name}:{kind}:{split or src}"] = f"{type(e).__name__}: {e}"; log("!! FAILED", name, kind, split or src, e)
    except Exception as e:
        FAILED[name] = f"{type(e).__name__}: {e}"; log(f"!! {name} FAILED -> skipped: {FAILED[name]}")
    enc = None; release()
    log(f"== {name} done in {time.time() - t0:.0f}s")
print("failed:", FAILED or "none")
'''

PROBE = r'''
# ============================================================================ zero-shot probe (true labels, no training)
# v4 train: positives = two different rows of the same person (v4 has no video id: may include same-session pairs);
# v1/v2 subset: positives = face and voice of the same person from DIFFERENT videos (no session shortcut).
# Score = cosine after removing each side's mean over the probed rows (as the ImageBind term of SET-13).
from sklearn.metrics import roc_curve
def eer(s, y):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * (fpr[i] + fnr[i]) / 2
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
PAIRS = {"ibalign": ("ibalign", "iba"), "clipb32+w2c": ("clipb32", "w2c"), "clipb16+w2c": ("clipb16", "w2c"),
         "aclip": ("aclip", "aclip"), "lbind": ("lbind", "lbind")}
FM_DIR = next((p.parent for p in IN.rglob("face_ibv_train.npy")), None)    # flag2027-feats-models, optional


def get(kind, enc, split=None, src=None):
    f = out_name(enc, kind, split, src)
    if f.exists():
        return np.load(f).astype(np.float32)
    if enc in ("ibv", "iba") and FM_DIR is not None:                      # ImageBind from FLAG_10 (full ext rows)
        f = FM_DIR / (f"{kind}_{enc}_{tag(split)}.npy" if split else f"ext_{src}_{kind}_{enc}.npy")
        if not f.exists():
            return None
        a = np.load(f).astype(np.float32)
        return a if split else a[EXT[src].fsel if kind == "face" else EXT[src].vsel]
    return None


rows = []
rng = np.random.RandomState(0)
if "train" in S:
    spk = pd.read_csv(S["train"]["txt"], sep=" ", header=None)[4].astype(str).values
    gmap = {}
    meta = SCAN["files"].get("meta_file_train_set.csv")
    if meta:
        gmap = dict(pd.read_csv(meta[0], header=None).values)
    n = len(spk)
    by = {s: np.where(spk == s)[0] for s in np.unique(spk)}
    pos = [(i, rng.choice(by[spk[i]][by[spk[i]] != i])) for i in rng.randint(n, size=3000)]
    for sg in (False, True):
        neg = []
        while len(neg) < 3000:
            i, j = rng.randint(n, size=2)
            if spk[i] != spk[j] and (not sg or gmap.get(spk[i]) == gmap.get(spk[j])):
                neg.append((i, j))
        pr = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        for name, (fe, ve) in {**PAIRS, "ImageBind (FLAG_10)": ("ibv", "iba")}.items():
            Fx, Vx = get("face", fe, split="train"), get("voice", ve, split="train")
            if Fx is None or Vx is None:
                continue
            Fx, Vx = l2n(Fx - Fx.mean(0)), l2n(Vx - Vx.mean(0))
            rows.append(dict(data="v4 train", protocol="g" if sg else "ng", model=name,
                             eer=round(eer(np.sum(Fx[pr[:, 0]] * Vx[pr[:, 1]], 1), y), 2)))
for s in EXT:
    rws = EXT[s]
    fm = pd.read_csv(rws.meta_dir / f"ext_{s}_face_meta.csv").iloc[rws.fsel].reset_index(drop=True)
    vm = pd.read_csv(rws.meta_dir / f"ext_{s}_voice_meta.csv").iloc[rws.vsel].reset_index(drop=True)
    pos, neg = [], []
    for vi, r in vm.iterrows():
        cand = fm.index[(fm.spk == r.spk) & (fm.video != r.video)]
        if len(cand):
            pos.append((rng.choice(cand), vi))
        other = fm.index[fm.spk != r.spk]
        neg.append((rng.choice(other), vi))
    pr = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    for name, (fe, ve) in {**PAIRS, "ImageBind (FLAG_10)": ("ibv", "iba")}.items():
        Fx, Vx = get("face", fe, src=s), get("voice", ve, src=s)
        if Fx is None or Vx is None:
            continue
        Fx, Vx = l2n(Fx - Fx.mean(0)), l2n(Vx - Vx.mean(0))
        for lang in vm.lang.unique():
            m = np.r_[(vm.lang.values[[p[1] for p in pos]] == lang), (vm.lang.values[[p[1] for p in neg]] == lang)]
            rows.append(dict(data=f"{s} (cross-video)", protocol=f"ng {lang}", model=name,
                             eer=round(eer(np.sum(Fx[pr[m, 0]] * Vx[pr[m, 1]], 1), y[m]), 2)))
PR = pd.DataFrame(rows); PR.to_csv(OUT / "probe_zeroshot.csv", index=False)
print(PR.pivot_table(index="model", columns=["data", "protocol"], values="eer").to_string())
# reading guide: ImageBind zero-shot is ~25-27 (ng) on v1 English / Urdu cross-video (IB-01). Near 50 = no signal.
# A model worth fusing is one clearly below 45 whose errors are not those of ImageBind (checked locally afterwards).
'''

DONE = r'''
man = {"created": time.strftime("%Y-%m-%d %H:%M"), "splits": {k: len(v["wav"]) for k, v in S.items()},
       "external": {k: [len(r.fsel), len(r.vsel)] for k, r in EXT.items()}, "encoders": ENCODERS, "failed": FAILED,
       "files": {f.name: list(np.load(f, mmap_mode="r").shape) for f in sorted(OUT.glob("*.npy"))}}
(OUT / "feats_manifest.json").write_text(json.dumps(man, indent=1))
shutil.rmtree(WORKDIR / "raw", ignore_errors=True)
print(f"{len(man['files'])} arrays in {OUT} | failed: {FAILED or 'none'}")
print("NEXT: Save Version -> Output -> New Dataset -> flag2027-feats-av, then hand feats_av back")
'''

if __name__ == "__main__":
    nb([("markdown", INTRO), ("code", ENV), ("code", PIP),
        ("markdown", "## 1. Data: organiser splits (row order = the .txt) and a fixed v1/v2 subset"), ("code", data_cell()),
        ("markdown", "## 2. Audio loading and helpers (as FLAG_12)"), ("code", AUDIO),
        ("markdown", "## 3. Encoders"), ("code", ENC),
        ("markdown", "## 4. LanguageBind (child process, transformers 4.57.3)"), ("code", LBIND),
        ("markdown", "## 5. Extract"), ("code", RUN),
        ("markdown", "## 6. Zero-shot probe"), ("code", PROBE),
        ("markdown", "## 7. Manifest"), ("code", DONE)], "FLAG_14_av_zeroshot.ipynb")
