"""Generate FLAG_162{a,b,c}_rfs_<x>.ipynb (RFS-01 buoc 1, Experiment/RFS-01/README.md, nhat ky 360): the feature blocks of the RF-L2-092 recipe
(Experiment/RFS-01/ref/flag_final_core_092.py: E2EExtractor / L2Extractor / ClipFaceExtractor / VB2Extractor / AgeExtractor, embedded verbatim, sha256 checked)
for every row the CC / CA heads of our clean people store use.
  face : rfclip (CLIP ViT-L/14 image_embeds 768) and rfibf (ImageBind-huge vision 1024, fp16) ; rfagef (ViT age: expected age + 9 probabilities = 10)
  voice: rfsam (SimAM-ResNet100 VoxBlink2 ft 256, first 60 s), rfr221 (ResNet221-LM 256), rfxv (SpeechBrain ECAPA yangwang825 192),
         rfibv (ImageBind audio 1024 fp16) ; rfagev (audeering age-gender: age, p(f), p(m), p(child) = 4)
  a : MAV-Celeb rows through the FLAG_12 loop (v4 train, v1 / v2, the four dev lists); ages only on the splits (v4 train + dev), not v1 / v2
  b, c : the audio cache (VoxCeleb2 FLAG_37 + FLAG_53, VoxBlink FLAG_46: voices of the centre 3 s clips) and the frames (faces), two interleaved halves
Not here: whv3 (FLAG_160) and the LoRA block (needs the LoRA of step 2).
    python kaggle/build_flag162.py     -> kaggle/FLAG_162a_rfs_mav.ipynb, FLAG_162b_rfs_auxb.ipynb, FLAG_162c_rfs_auxc.ipynb
Smoke on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6;RFS_AUX_MAX=24
"""
import base64, hashlib, zlib
from pathlib import Path
import build_eval_notebooks as B12
import build_rep_notebooks as BR

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "Experiment/RFS-01/ref/flag_final_core_092.py"
SMOKE = '(LOCAL or os.environ.get("FLAG_SMOKE") == "1")'
MODES = {"a": ("FLAG_162a_rfs_mav.ipynb", "mav"), "b": ("FLAG_162b_rfs_auxb.ipynb", "auxb"), "c": ("FLAG_162c_rfs_auxc.ipynb", "auxc")}

INTRO = """
# FLAG 2027 — NB-162__X__: RFS-01 step 1, the RF-L2-092 feature blocks for our people store (lane __MODE__)

**Why.** The user decided (3/11, nhat ky 360): the final system = F1s + **RFS**, where RFS is the RF-092 recipe (CB 18.61) **re-learned on our filtered people
store**. We take only RF's code / technique, never its learned artifacts. This notebook only **extracts** the blocks RF's projector and mixer need, with
**RF's own extractors** (`flag_final_core_092.py`, embedded below, sha256 asserted). No Bengali filtering at extraction (step 2 filters with `lid_strict.json`).

| block | model | notes |
|---|---|---|
| `rfclip` 768 | `openai/clip-vit-large-patch14` @32bd6428, `image_embeds` | fp32 |
| `rfibf` 1024 | ImageBind-huge vision, fp16 | Resize 224 bicubic + CenterCrop + CLIP mean/std |
| `rfsam` 256 | SimAM-ResNet100 VoxBlink2 ft (WeSpeaker) | first 60 s |
| `rfr221` 256 | WeSpeaker ResNet221-LM | whole clip |
| `rfxv` 192 | SpeechBrain `yangwang825/ecapa-tdnn-vox2` (BTC space) | whole clip |
| `rfibv` 1024 | ImageBind-huge audio, fp16 | 2 s clips covering the clip (>= 3) |
| `rfagef` 10 / `rfagev` 4 | `nateraw/vit-age-classifier`, `audeering/wav2vec2-large-robust-24-ft-age-gender` | only v4 train + dev lists |

Lane a = MAV-Celeb rows (FLAG_12 loop). Lanes b / c = two interleaved halves of the aux store (faces: odd / even frame rows; voices: the audio cache chunks).
Deliberate differences from RF: aux voices are the cached centre 3 s clips (not whole clips), aux faces the cached 224 PNG frames; MAV audio is decoded by the
FLAG_12 loader (librosa soxr resample; only MAV v1 is not 16 kHz) and v1 / v2 voices are the 12 s centre crop as every array of this store.
**Private** (VoxBlink rows). GPU T4 x 2, Internet ON.
"""

PIP_EXTRA = r'''
# RF's own install recipe (ImageBind imports) on top of the framework's PIP12 cell
if not LOCAL:
    import subprocess, importlib.util
    for mod in ("wcwidth", "regex", "portalocker", "einops", "timm", "hyperpyyaml"):
        if importlib.util.find_spec(mod) is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", mod], check=True)
    import importlib.metadata as md_
    if tuple(int(x) for x in md_.version("ruamel.yaml").split(".")[:2]) >= (0, 19):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ruamel.yaml<0.19"], check=True)
'''

DEFS = r'''# ============================================================================ RFS-01 step 1: RF's extractors (core embedded verbatim) and checkpoints
import base64, zlib, hashlib, importlib.util, urllib.request
CORE_B64 = "__CORE_B64__"
CORE_SHA = "__CORE_SHA__"
RFS_DIR = WORKDIR / "rfs_code"; RFS_DIR.mkdir(exist_ok=True)
_src = zlib.decompress(base64.b64decode(CORE_B64))
assert hashlib.sha256(_src).hexdigest() == CORE_SHA, "embedded RF core differs from the registered file"
(RFS_DIR / "flag_final_core_092.py").write_bytes(_src)
log("RF core sha256", CORE_SHA[:12])


def load_core(dev, tag):
    """One module object per device (the core reads its global DEVICE at call time): a lane may run faces on cuda:0 and voices on cuda:1."""
    spec = importlib.util.spec_from_file_location(f"flag_final_core_092_{tag}", RFS_DIR / "flag_final_core_092.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.DEVICE = torch.device(dev)
    return m


URL = {"imagebind": "https://huggingface.co/PKU-Alignment/AnyRewardModel/resolve/40c7e2ade237848a194b2b77517da024c4ec8532/imagebind/imagebind_huge.pth",
       "r221": "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet221-LM/resolve/main/voxceleb_resnet221_LM.pt",
       "sam": "https://huggingface.co/MCplayer/voxblink2_samresnet100_ft/resolve/main/models/voxblink2_samresnet100_ft/avg_model.pt"}
SHA = {"imagebind": "d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157",
       "r221": "47d76239f1e865b273e31e30f691959061ff6565ed6c4e7be9639a65d9662eb5",
       "sam": "5aeee438ca23c0ca6e341bab6c6bf7f465497e1dc323bb1bc1074d6a0c778b11"}
CK = {}
for _n, _u in URL.items():
    _t = CACHE / "rfs_models" / f"{_n}.pt"; _t.parent.mkdir(exist_ok=True)
    for _k in range(4):
        if _t.exists():
            break
        try:
            log("download", _n); urllib.request.urlretrieve(_u, str(_t) + ".part"); Path(str(_t) + ".part").replace(_t)
        except OSError as e:
            log("  download error", _n, type(e).__name__, "- retry"); Path(str(_t) + ".part").unlink(missing_ok=True); time.sleep(30 * (_k + 1))
    _h = hashlib.sha256()
    with open(_t, "rb") as f_:
        for b_ in iter(lambda: f_.read(1 << 24), b""):
            _h.update(b_)
    assert _h.hexdigest() == SHA[_n], f"{_n}: sha256 differs from the registered checkpoint"
    CK[_n] = _t
log("checkpoints", {k: str(v.stat().st_size // 2 ** 20) + " MB" for k, v in CK.items()})
from PIL import Image
import torchvision.transforms as TV


def tile1s(w):
    """load_wave's rule: a clip shorter than 1 s is tiled to 1 s (uses only this clip)."""
    return w if len(w) >= 16000 else np.tile(w, int(np.ceil(16000 / len(w))))[:16000]


class RFFace:
    """CLIP-L image_embeds (fp32) and ImageBind vision (fp16 on GPU) of one PIL image each, exactly ClipFaceExtractor.faces / L1Extractor.faces."""

    def __init__(self, c, dev):
        self.c, self.dev = c, torch.device(dev); self.half = self.dev.type == "cuda"
        self.clip = c.ClipFaceExtractor()
        self.ib, self.MT = c.load_imagebind(CK["imagebind"])
        if not self.half:
            self.ib = self.ib.float()
        self.vis = TV.Compose([TV.Resize(224, interpolation=TV.InterpolationMode.BICUBIC), TV.CenterCrop(224), TV.ToTensor(),
                               TV.Normalize(c.IB_VIS_MEAN, c.IB_VIS_STD)])

    def prep(self, im):
        im = im.convert("RGB")
        return torch.stack([self.clip.p(images=im, return_tensors="pt")["pixel_values"][0], self.vis(im)])

    @torch.inference_mode()
    def run(self, x):
        cl = self.clip.m(pixel_values=x[:, 0].to(self.dev)).image_embeds.float().cpu().numpy()
        v = x[:, 1].to(self.dev)
        ibf = self.ib({self.MT.VISION: v.half() if self.half else v})[self.MT.VISION].float().cpu().numpy()
        return {"": ibf, "clip": cl}


class RFVoice:
    """ImageBind audio, ResNet221-LM, SimAM-ResNet100 (60 s), SpeechBrain ECAPA of one 16 kHz waveform (L2Extractor.voice_features, VB2Extractor)."""

    def __init__(self, c, dev):
        self.c, self.dev = c, torch.device(dev); self.half = self.dev.type == "cuda"
        self.ib, self.MT = c.load_imagebind(CK["imagebind"])
        if not self.half:
            self.ib = self.ib.float()
        r = c.ENCODERS["resnet221_lm"]["make"]()
        sd = torch.load(CK["r221"], map_location="cpu"); sd = sd.get("model", sd)
        res = r.load_state_dict(sd, strict=False)
        if res.missing_keys or set(res.unexpected_keys) - {"projection.weight"}:
            raise ValueError(f"ResNet221 checkpoint mismatch: {res}")
        self.r221 = r.to(self.dev).eval()
        self.sam = c.VB2Extractor(CK["sam"]).m
        EncoderClassifier = speechbrain_import()
        from speechbrain.utils.fetching import LocalStrategy
        self.btc = EncoderClassifier.from_hparams(source="yangwang825/ecapa-tdnn-vox2", savedir=str(CACHE / "sb_btc"),
                                                  run_opts={"device": str(self.dev)}, local_strategy=LocalStrategy.COPY)

    @torch.inference_mode()
    def feats(self, w):
        c = self.c; w = tile1s(np.ascontiguousarray(w, dtype=np.float32))
        a = c.ib_audio_clips(w)[None].to(self.dev)
        ibv = self.ib({self.MT.AUDIO: a.half() if self.half else a})[self.MT.AUDIO].float().cpu().numpy()[0]
        o = self.r221(c.kaldi_fbank(w)[None].to(self.dev))
        r2 = (o[-1] if isinstance(o, tuple) else o).squeeze(0).float().cpu().numpy()
        sam = self.sam(c.kaldi_fbank(w[:16000 * 60])[None].to(self.dev)).squeeze(0).float().cpu().numpy()
        xv = self.btc.encode_batch(torch.tensor(w)[None]).squeeze().float().cpu().numpy()
        return {"": ibv, "r221": r2, "sam": sam, "xv": xv}


class RFAge:
    """AgeExtractor's two models on one image / one waveform (face: [expected age, 9 probabilities]; voice: [age, p(f), p(m), p(child)])."""

    def __init__(self, c, dev):
        self.c, self.dev = c, torch.device(dev); self.ag = c.AgeExtractor()

    def prep(self, im):
        return self.ag.fp(im.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    @torch.inference_mode()
    def face(self, x):
        pr = torch.softmax(self.ag.fm(x.to(self.dev)).logits, -1).float().cpu().numpy()
        return {"": np.concatenate([(pr @ self.c.AGE_MID)[:, None], pr], 1)}

    @torch.inference_mode()
    def voice(self, w):
        w = tile1s(np.ascontiguousarray(w, dtype=np.float32))
        x = self.ag.vp(w, sampling_rate=16000, return_tensors="pt")["input_values"].to(self.dev)
        a, g = self.ag.vm(x)
        return {"": np.r_[100 * a.float().cpu().numpy()[0], g.float().cpu().numpy()[0]].astype(np.float32)}
'''

ENC_A = r'''# ============================================================================ the framework encoders (lane a: v4 train, v1 / v2, dev lists)
_C0, _AGE = {}, {}


def core0():
    if "c" not in _C0:
        _C0["c"] = load_core(str(DEV), "a")
    return _C0["c"]


class RFFaceEnc:
    kind = "face"

    def __init__(self):
        self.m = RFFace(core0(), DEV); self.prep = self.m.prep

    def __call__(self, x, use_amp=True):
        return self.m.run(x)


class RFVoiceEnc:
    kind = "voice"

    def __init__(self):
        self.m = RFVoice(core0(), DEV)

    def __call__(self, w, use_amp=True):
        return self.m.feats(w)


def age_models():
    if "m" not in _AGE:
        _AGE["m"] = RFAge(core0(), DEV)
    return _AGE["m"]


class RFAgeFEnc:
    kind = "face"

    def __init__(self):
        self.m = age_models(); self.prep = self.m.prep

    def __call__(self, x, use_amp=True):
        return self.m.face(x)


class RFAgeVEnc:
    kind = "voice"

    def __init__(self):
        self.m = age_models()

    def __call__(self, w, use_amp=True):
        return self.m.voice(w)


NEW = {"rfface": RFFaceEnc, "rfvoice": RFVoiceEnc, "rfagef": RFAgeFEnc, "rfagev": RFAgeVEnc}
ARRAYS_NEW = {"rfface": ("face", {"": "rfibf", "clip": "rfclip"}), "rfvoice": ("voice", {"": "rfibv", "r221": "rfr221", "sam": "rfsam", "xv": "rfxv"}),
              "rfagef": ("face", {"": "rfagef"}), "rfagev": ("voice", {"": "rfagev"})}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {}
ENCODERS = ["rfface", "rfagef", "rfvoice", "rfagev"]            # faces first (cheap); ages after their modality
EXT_ENCODERS = ["rfface", "rfvoice"]                            # ages only on v4 train + dev lists (RF: age stream of the no_gender files)


def build(name):
    return NEW[name]()


def release(keep_ib=False):
    import gc
    _AGE.clear(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


log("RFS-01 encoders ready:", ENCODERS, "| on v1 / v2:", EXT_ENCODERS)
'''

ENC_AUX = r'''# ============================================================================ aux lanes: the MAV loop of this framework does nothing
ENCODERS = EXT_ENCODERS = []
NEW, ARRAYS_NEW, MAIN_NEW, RESHAPE = {}, {}, {}, {}


def build(name):
    raise RuntimeError(name)


def release(keep_ib=False):
    import gc
    gc.collect()
'''

AUX = r'''print("[cell] rfs aux", flush=True)
# VoxCeleb2 (vox = FLAG_37 rows, vox2b = FLAG_53 rows) and VoxBlink (vb = FLAG_46): voices from the audio cache, faces from the cached frames,
# each source in its own voice_meta.csv / face_meta.csv row order; rows of the other lane (and rows without audio / frame) stay NaN.
import threading
import concurrent.futures as cf
SMOKE = os.environ.get("FLAG_SMOKE") == "1"
LANE = "__MODE__"
PAR = 1 if LANE == "auxc" else 0
BUDGET = float(os.environ.get("RFS_BUDGET_H", "8.0")) * 3600
LIM = int(os.environ.get("RFS_AUX_MAX", "0")) or (24 if SMOKE else 0)
T_START = time.time()


def _inroots():
    base = Path("/kaggle/input")
    out = []
    for p in (base.iterdir() if base.exists() else []):
        if p.name in ("datasets", "notebooks", "kernels", "models"):
            out += [q for o in p.iterdir() if o.is_dir() for q in o.iterdir() if q.is_dir()]
        elif p.is_dir():
            out.append(p)
    return [p for p in out if not any(k in p.name for k in ("mavceleb", "raw-v4", "ext-meta"))]


_roots = _inroots()
_meta = {}
for r in _roots:
    for kind in ("voice", "face"):
        for p in r.rglob(f"{kind}_meta.csv"):
            for s, par in (("vox", "feats_vox2"), ("vox2b", "feats_vox2b"), ("vb", "feats_vb")):
                if p.parent.name == par:
                    _meta.setdefault((kind, s), p)
assert len(_meta) == 6, f"face / voice meta found for {sorted(_meta)}: attach lab-data / feats-vox2b / lab-vb"
_tar = sorted(q for r in _roots for q in r.rglob("tar0000.npz") if q.parent.name == "vox")
assert _tar, "attach duongphihoang/flag2027-audio-cache-p (vox/tar*.npz, vb/*.npz)"
ACROOT = _tar[0].parent.parent
CH = [(p, "vox") for p in sorted((ACROOT / "vox").glob("tar*.npz"))] + [(p, "vb") for p in sorted((ACROOT / "vb").glob("*.npz"))]
CH = CH[PAR::2]                                                 # b = even chunks, c = odd chunks (as FLAG_160)
dt = {"spk": str, "video": str, "utt": str}
VM = {s: pd.read_csv(_meta[("voice", s)], dtype=dt) for s in ("vox", "vox2b", "vb")}
FM = {s: pd.read_csv(_meta[("face", s)], dtype=dt) for s in ("vox", "vox2b", "vb")}
ROW = {s: {k: r for r, k in enumerate(zip(d.spk, d.video, d.utt))} for s, d in VM.items()}
log("audio cache", ACROOT, len(CH), "chunks of this lane | voice rows", {s: len(d) for s, d in VM.items()}, "| face rows", {s: len(d) for s, d in FM.items()})

# -- frames (FACELORA-03's layout): FLAG_37 rows in vox_frames_png.zip ("<row>.png"), FLAG_53 / VoxBlink in the FLAG_63 chunk zips ("spk/video/utt.png")
FILES = SCAN["files"]
_z37 = FILES.get("vox_frames_png.zip") or FILES.get("vox_frames_png.zipdata")
_zch = FILES.get("frames_manifest.json", [])
assert _z37 and _zch, "attach onnhqunhh/flag2027-lora-frames-p (vox_frames_png.zipdata + frames/)"
Z37, FR63 = sorted(_z37, key=lambda p: len(p.as_posix()))[0], _zch[0].parent
MEMBER = {}
with zipfile.ZipFile(Z37) as z:
    for n in z.namelist():
        MEMBER["v37/" + n] = Z37
for part in ("vox2b", "vb"):
    for zf in sorted([*(FR63 / part).glob("*.zip"), *(FR63 / part).glob("*.zipdata")]):
        with zipfile.ZipFile(zf) as z:
            for n in z.namelist():
                MEMBER[f"{part}/" + n] = zf
log(f"frames indexed: {sum(k.startswith('v37/') for k in MEMBER)} FLAG_37, {sum(k.startswith('vox2b/') for k in MEMBER)} FLAG_53, {sum(k.startswith('vb/') for k in MEMBER)} VoxBlink")
_ZT = threading.local()


def frame_bytes(member):
    zs = getattr(_ZT, "z", None)
    if zs is None:
        zs = _ZT.z = {}
    zp = MEMBER[member]
    if zp not in zs:
        zs[zp] = zipfile.ZipFile(zp)
    return zs[zp].read(member.split("/", 1)[1])


def member_of(s, i, r):
    return f"v37/{i}.png" if s == "vox" else f"{s}/{r.spk}/{r.video}/{r.utt}.png"


ND = torch.cuda.device_count()
D_F, D_V = ("cuda:0", "cuda:1" if ND > 1 else "cuda:0") if ND else ("cpu", "cpu")
CF, CV = load_core(D_F, "f"), load_core(D_V, "v")
FACE, VOICE = RFFace(CF, D_F), RFVoice(CV, D_V)
log("models ready: faces on", D_F, "| voices on", D_V)
A = {}                                                          # (kind, src, block) -> (N, d) float16
for s in ("vox", "vox2b", "vb"):
    for b, d in (("clip", 768), ("ibf", 1024)):
        A[("face", s, b)] = np.full((len(FM[s]), d), np.nan, np.float16)
    for b, d in (("sam", 256), ("r221", 256), ("xv", 192), ("ibv", 1024)):
        A[("voice", s, b)] = np.full((len(VM[s]), d), np.nan, np.float16)
DONE = {"face": 0, "voice": 0}
STOP = {"budget": False}


def face_job():
    log("[aux-face] start")
    n_all = sum(len(d) for d in FM.values()); t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for s in ("vox", "vox2b", "vb"):
            rows = [(i, r) for i, r in enumerate(FM[s].itertuples()) if i % 2 == PAR and member_of(s, i, r) in MEMBER]
            if LIM:
                rows = rows[:LIM]
            for b0 in range(0, len(rows), 64):
                if time.time() - T_START > BUDGET:
                    STOP["budget"] = True; return
                chunk = rows[b0:b0 + 64]
                xs = list(ex.map(lambda ir: FACE.prep(Image.open(io.BytesIO(frame_bytes(member_of(s, *ir)))).convert("RGB")), chunk))
                R = FACE.run(torch.stack(xs))
                idx = [i for i, _ in chunk]
                A[("face", s, "clip")][idx] = R["clip"]; A[("face", s, "ibf")][idx] = R[""]
                DONE["face"] += len(idx)
                if LIM or (b0 // 64) % 20 == 0:
                    log(f"[aux-face] {s} {b0 + len(idx)}/{len(rows)} (all {DONE['face']}) {DONE['face'] / max(time.time() - t0, 1):.1f} img/s")


def clips(Z, a, b):
    W = Z["wav"][a:b]
    ln = Z["n"][a:b].astype(int) if "n" in Z.files else [W.shape[1]] * len(W)
    return [W[i][:max(int(ln[i]), 16000 // 2)].astype(np.float32) / 32768.0 for i in range(len(W))]


def voice_job():
    log("[aux-voice] start")
    t0 = time.time()
    for ci, (p, kind) in enumerate(CH):
        Z = np.load(p)
        keys = [tuple(map(str, k)) for k in zip(Z["spk"], Z["video"], Z["utt"])]
        for c0 in range(0, len(keys), 64):
            if time.time() - T_START > BUDGET:
                STOP["budget"] = True; return
            for j, w in enumerate(clips(Z, c0, c0 + 64)):
                F_ = VOICE.feats(w)
                for s in (("vb",) if kind == "vb" else ("vox", "vox2b")):
                    r = ROW[s].get(keys[c0 + j])
                    if r is not None:
                        for b, k in (("ibv", ""), ("sam", "sam"), ("r221", "r221"), ("xv", "xv")):
                            A[("voice", s, b)][r] = F_[k]
                DONE["voice"] += 1
                if LIM:
                    log(f"[aux-voice] clip {DONE['voice']}")
                if LIM and DONE["voice"] >= LIM:
                    return
        log(f"[aux-voice] chunk {ci + 1}/{len(CH)}: {DONE['voice']} clips ({DONE['voice'] / max(time.time() - t0, 1):.1f} clips/s, {time.time() - t0:.0f}s)")


ERR = []
def guard(fn):
    def w():
        try:
            fn()
        except BaseException as e:
            import traceback; ERR.append(f"{fn.__name__}: {e!r}"); traceback.print_exc()
    return w


if D_F == D_V == "cpu":                                         # smoke on CPU: one job at a time (two jobs would fight for the same cores)
    guard(face_job)(); guard(voice_job)()
else:
    th = [threading.Thread(target=guard(face_job)), threading.Thread(target=guard(voice_job))]
    [t.start() for t in th]; [t.join() for t in th]
assert not ERR, ERR
for s in ("vox", "vox2b", "vb"):
    (OUT / s).mkdir(exist_ok=True)
    shutil.copy(_meta[("voice", s)], OUT / s / "voice_meta.csv"); shutil.copy(_meta[("face", s)], OUT / s / "face_meta.csv")
    for (kind, ss, b), X in A.items():
        if ss == s:
            np.save(OUT / s / f"{kind}_rf{b}.npy", X)
AUX_COVER = {f"{s}/{kind}_rf{b}": int(np.isfinite(X[:, 0].astype(np.float32)).sum()) for (kind, s, b), X in A.items()}
log(f"[aux] faces {DONE['face']}, voices {DONE['voice']} [{time.time() - T_START:.0f}s]; budget stop: {STOP['budget']}; rows per array {AUX_COVER}")
CHECKS["rfs01_aux"] = dict(cover=AUX_COVER, chunks=len(CH), lane=LANE, faces=DONE["face"], voices=DONE["voice"], budget_stop=STOP["budget"])
'''

CHECK = r'''print("[cell] rfs check vs RF embeddings", flush=True)
# registered check: the RF-092 reproduction run (onnhqunhh/flag-rf-092-tailap-p, FLAG_RF_E2E/embeddings.npz, copied to the private dataset ngnnguynchn/flag2027-rf092-dev-emb-p: sorted unique dev files) vs the arrays above on
# the dev lists. Path-independent: every dev row is matched to its best row in the RF array on rfibf (faces) / rfibv (voices); then EVERY block (and the age) of the
# matched pair must agree (cos >= 0.999). Skipped without the reference attached.
_ref = SCAN["files"].get("embeddings.npz", [])
RFCHK = {}
if not _ref or LANE_MODE != "mav":
    log("RF embeddings.npz not attached / aux lane: check skipped", bool(_ref))
else:
    Rz = np.load(_ref[0])
    def _cos(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    def _best(M, Rm):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12); Rn = Rm / (np.linalg.norm(Rm, axis=1, keepdims=True) + 1e-12)
        sim = Mn @ Rn.T; j = sim.argmax(1)
        return j, sim[np.arange(len(M)), j]
    for kind, key, blocks in (("face", "ibf", [("rfclip", "clip"), ("rfibf", "ibf")]), ("voice", "ibv", [("rfsam", "sam"), ("rfr221", "r221"), ("rfxv", "xv"), ("rfibv", "ibv")])):
        Ms = {b: [] for b, _ in blocks}; Age = []; Rj = []
        for k in S:
            if k == "train":
                continue
            f = {b: OUT / f"{kind}_{b}_{tag(k)}.npy" for b, _ in blocks}
            if not all(p.exists() for p in f.values()):
                continue
            for b, _ in blocks:
                Ms[b].append(np.load(f[b]).astype(np.float32))
            fa = OUT / f"{kind}_rfage{kind[0]}_{tag(k)}.npy"
            Age.append(np.load(fa)[:, 0] if fa.exists() else np.full(len(Ms[blocks[0][0]][-1]), np.nan, np.float32))
        if not Ms[blocks[0][0]]:
            continue
        Ms = {b: np.concatenate(v) for b, v in Ms.items()}; Age = np.concatenate(Age)
        j, c_main = _best(Ms["rf" + key], Rz[f"{kind}_{key}"].astype(np.float32))
        res = {"rows": len(j), "match_cos_min": round(float(c_main.min()), 5)}
        for b, k_ in blocks:
            c = _cos(Ms[b], Rz[f"{kind}_{k_}"].astype(np.float32)[j]); res[b] = dict(cos_min=round(float(c.min()), 5), cos_mean=round(float(c.mean()), 6), below_0999=int((c < 0.999).sum()))
        if np.isfinite(Age).any():
            d = np.abs(Age - Rz[f"{kind}_age"][j]); res["age_max_abs_diff"] = round(float(np.nanmax(d)), 4)
        RFCHK[kind] = res
        log("RF check", kind, res)
    CHECKS["rfs01_vs_rf"] = RFCHK
    bad = {k: v for k, r in RFCHK.items() for k, v in r.items() if isinstance(v, dict) and v["cos_min"] < (0.99 if SMOKE_RUN else 0.999)}
    assert not bad, f"definition differs from RF: {bad}"
'''

UP = r'''print("[cell] upload", flush=True)
UPL = os.environ.get("RFS_UPLOAD", "")
(OUT / "rfs01_meta.json").write_text(json.dumps(dict(exp="RFS-01 step 1", lane="__MODE__", core_sha256="__CORE_SHA__", dtype="float16 (aux, v1 / v2) / float32 (v4 train + dev lists)",
        blocks="face: rfclip 768, rfibf 1024, rfagef 10 (splits only); voice: rfsam 256, rfr221 256, rfxv 192, rfibv 1024, rfagev 4 (splits only)",
        rows="a: split / ext arrays in the FLAG_12 order (feats_manifest.json: splits, external); b, c: <vox|vox2b|vb>/{face,voice}_rf<block>.npy in that source's face_meta.csv / voice_meta.csv order (copied beside), NaN = not in this lane (faces: i % 2 == 0 in b, 1 in c; voices: even / odd chunks of the cache list)",
        voice_clip="aux: cached centre 3 s clip; MAV: FLAG_12 loader (whole file, v1 / v2 12 s centre crop)", faces="aux: cached 224 PNG frames",
        unreadable_file="a: framework stand-in (1 s silence / black image), listed in feats_manifest.json broken"), indent=1))
if UPL and os.environ.get("FLAG_SMOKE") != "1":
    try:
        import kagglehub
        kagglehub.dataset_upload(UPL, str(OUT), version_notes="RFS-01 step 1 lane __MODE__")
        print("uploaded ->", UPL, flush=True)
    except Exception as e:
        print(f"upload to {UPL} failed ({e!r}); the arrays are in the notebook output", flush=True)
print("ALL CELLS OK", flush=True)
'''


def cells(x):
    nb, mode = MODES[x]
    core = CORE.read_bytes()
    b64 = base64.b64encode(zlib.compress(core, 9)).decode()
    sha = hashlib.sha256(core).hexdigest()
    defs = DEFS.replace("__CORE_B64__", b64).replace("__CORE_SHA__", sha)
    enc = ENC_A if mode == "mav" else ENC_AUX
    chk = CHECK.replace("LANE_MODE", repr(mode)).replace("SMOKE_RUN", '(os.environ.get("FLAG_SMOKE") == "1")')
    out = BR.nb_cells(INTRO.replace("__X__", x).replace("__MODE__", mode), BR.env(f"feats_rfs_{x}", ["rfface"], "RFS-01 step 1 (replaced below)"),
                      B12.PIP12 + PIP_EXTRA, f"feats_rfs_{x}",
                      [("code", B12.ENC12), ("code", defs), ("code", enc)], chk, f"flag2027-feats-rfs-{x}-p")
    res, n_env, n_data = [], 0, 0
    for kind, src in out:
        if kind == "code" and "if LOCAL and os.environ.get(" in src:
            src = src.replace("if LOCAL and os.environ.get(", f"if {SMOKE} and os.environ.get("); n_env += 1
        if kind == "code" and "if LOCAL and SMOKE_ROWS:" in src:
            src = src.replace("if LOCAL and SMOKE_ROWS:", f"if {SMOKE} and SMOKE_ROWS:"); n_data += 1
        res.append((kind, src))
    assert n_env == 1 and n_data == 1, (n_env, n_data)
    if mode != "mav":
        res += [("markdown", "## RFS-01: aux rows (audio cache + frames)"), ("code", AUX.replace("__MODE__", mode))]
    res += [("code", UP.replace("__MODE__", mode).replace("__CORE_SHA__", sha))]
    k = 0
    for i, (kind, src) in enumerate(res):                      # luat notebook Kaggle: dong dau moi cell code la print("[cell] ...")
        if kind == "code":
            k += 1
            assert not src.lstrip().startswith("%%"), src[:40]
            if not src.lstrip().startswith('print("[cell]'):
                res[i] = (kind, f'print("[cell] {x}{k}", flush=True)\n' + src.lstrip("\n"))
    return res


if __name__ == "__main__":
    for x in MODES:
        B12.nb(cells(x), MODES[x][0])
        print(MODES[x][0], "built")
