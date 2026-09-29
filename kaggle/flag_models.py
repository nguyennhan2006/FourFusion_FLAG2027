"""FLAG 2027 — MODEL-01: extra frozen encoders as bridge streams (docs/PLAN_MODELS.md §3 S1) + ImageBind probe (§4 X3).

The bridge (VGGFace fc7 <-> ECAPA-BTC) is the bottleneck, and recognition-grade encoders (ArcFace, ReDimNet) made it
worse. The candidates here are chosen for *soft attributes* (age, region, build) and are added next to VGG / ECAPA,
never swapped in:

    face  vitage   nateraw/vit-age-classifier (FairFace)      768 CLS  + vitageprob (9 age bins)
          farl     FaRL ViT-B/16, LAION-Face 20M, ep64        512 CLIP image embedding
          siglip2  google/siglip2-base-patch16-224            768 pooled image feature
          adaface  CVLFace AdaFace IR-101, WebFace12M         512, on the same SCRFD 5-point crop as ArcFace (control)
          ibv      ImageBind-huge vision                      1024, joint image-audio space
    voice w2v2ag   audeering wav2vec2-large-robust age-gender 1024 mean hidden + w2v2agage (1) + w2v2aggen (female, male, child)
          iba      ImageBind-huge audio                       1024, joint space; ImageBind's 2 s clips, covering the whole file

Outputs follow flag_extract: face_<enc>_<split>.npy / voice_<enc>_<split>.npy in organiser row order (flag_v2.FeatureStore
reads them by name). MAV-Celeb v1/v2 rows follow the existing ext_<src>_{face,voice}_meta.csv, so a new stream lines up
row for row with the VGG / BTC features already extracted: ext_<src>_face_<enc>.npy (fp16).
Licences to declare: ImageBind CC-BY-NC 4.0, audeering CC-BY-NC-SA 4.0.
"""
from __future__ import annotations
import io, os, sys, time, types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import flag_extract as X
from flag_extract import DEV, log, tag

FACE_ENCODERS = ["vitage", "farl", "siglip2", "adaface", "ibv"]
VOICE_ENCODERS = ["w2v2ag", "iba"]


def _amp(flag):
    return torch.autocast("cuda", dtype=torch.float16, enabled=bool(flag) and DEV.type == "cuda")


def read_rgb(src):
    """path or raw bytes -> PIL RGB (the organiser's VGGFace also reads the jpg as-is, no detection)."""
    from PIL import Image
    return Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src).convert("RGB")


# ============================================================================ face encoders
# interface: prep(PIL) -> CHW tensor ; __call__(NCHW tensor, amp) -> {suffix: (N, d) array}, "" = main embedding

class ViTAge:
    ID = "nateraw/vit-age-classifier"

    def __init__(self, cache):
        from transformers import ViTImageProcessor, ViTForImageClassification
        self.proc = ViTImageProcessor.from_pretrained(self.ID)
        self.m = ViTForImageClassification.from_pretrained(self.ID).to(DEV).eval()

    def prep(self, im):
        return self.proc(images=im, return_tensors="pt").pixel_values[0]

    @torch.no_grad()
    def __call__(self, x, amp=True):
        with _amp(amp):
            h = self.m.vit(x.to(DEV)).last_hidden_state[:, 0]          # post-LayerNorm CLS, what the classifier reads
            p = torch.softmax(self.m.classifier(h).float(), 1)
        return {"": h.float().cpu().numpy(), "prob": p.cpu().numpy()}


class FaRL:
    URL = ("https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/"
           "FaRL-Base-Patch16-LAIONFace20M-ep64.pth")

    def __init__(self, cache):
        import clip
        m, self.pre = clip.load("ViT-B/16", device="cpu", download_root=str(cache))
        ck = _download(self.URL, cache)
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        vis = [k for k in m.state_dict() if k.startswith("visual.")]
        hit = [k for k in vis if k in sd and sd[k].shape == m.state_dict()[k].shape]
        assert len(hit) == len(vis), f"FaRL checkpoint covers {len(hit)}/{len(vis)} visual tensors"
        m.load_state_dict(sd, strict=False)
        self.m = m.visual.float().to(DEV).eval()

    def prep(self, im):
        return self.pre(im)

    @torch.no_grad()
    def __call__(self, x, amp=True):
        with _amp(amp):
            return {"": self.m(x.to(DEV)).float().cpu().numpy()}


class SigLIP2:
    ID = "google/siglip2-base-patch16-224"

    def __init__(self, cache):
        from transformers import AutoModel, AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained(self.ID)
        self.m = AutoModel.from_pretrained(self.ID).to(DEV).eval()

    def prep(self, im):
        o = self.proc(images=im, return_tensors="pt")
        assert set(o.keys()) == {"pixel_values"}, f"variable-resolution processor output {list(o.keys())}"
        return o.pixel_values[0]

    @torch.no_grad()
    def __call__(self, x, amp=True):
        with _amp(amp):
            o = self.m.get_image_features(pixel_values=x.to(DEV))
        # transformers 4.x returns the pooled tensor; 5.x returns the vision output, whose pooler_output is that tensor
        e = o if torch.is_tensor(o) else o.pooler_output
        return {"": e.float().cpu().numpy()}


class AdaFace:
    """CVLFace release of AdaFace IR-101 (WebFace12M). Input: RGB 112x112 on the ArcFace template, scaled to [-1, 1].
    Built with the release's own factory, exactly what its wrapper.py does, but without transformers: the wrapper is a
    PreTrainedModel that never calls post_init(), which transformers >= 5 requires (all_tied_weights_keys)."""
    ID = "minchul/cvlface_adaface_ir101_webface12m"

    def __init__(self, cache):
        import yaml
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
        path = snapshot_download(self.ID)
        cwd = os.getcwd()
        os.chdir(path); sys.path.insert(0, path)            # the release imports `models` and reads configs relative to it
        try:
            from models import get_model
            m = get_model(OmegaConf.create(yaml.safe_load(open("pretrained_model/model.yaml"))))
            m.load_state_dict_from_path("pretrained_model/model.pt")
        finally:
            os.chdir(cwd); sys.path.remove(path)
        self.m = m.to(DEV).eval()
        self.det = _detector()

    def prep(self, im):
        bgr = np.asarray(im)[..., ::-1].copy()
        a, _, _ = X.align_face(self.det, bgr)
        rgb = torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return (rgb - 0.5) / 0.5

    @torch.no_grad()
    def __call__(self, x, amp=False):                                 # small net: keep fp32
        out = self.m(x.to(DEV))
        out = out[0] if isinstance(out, (tuple, list)) else out
        return {"": out.float().cpu().numpy()}


def _detector():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=0 if DEV.type == "cuda" else -1, det_size=(224, 224), det_thresh=0.3)   # as flag_extract.ArcFace
    return app.det_model


# ============================================================================ ImageBind (shared by ibv / iba)
class ImageBind:
    URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
    VIS_MEAN, VIS_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, cache):
        # imagebind/__init__ imports imagebind.data, which needs pytorchvideo; we reimplement the two loaders we use
        sys.modules.setdefault("imagebind.data", types.ModuleType("imagebind.data"))
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        self.MT = ModalityType
        m = imagebind_model.imagebind_huge(pretrained=False)
        m.load_state_dict(torch.load(_download(self.URL, cache), map_location="cpu", mmap=True))
        self.m = m.to(DEV).eval()
        from torchvision import transforms as T
        self.tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                             T.ToTensor(), T.Normalize(self.VIS_MEAN, self.VIS_STD)])

    def vision(self):
        return _IBVision(self)

    def audio(self):
        return _IBAudio(self)


class _IBVision:
    def __init__(self, ib):
        self.ib = ib

    def prep(self, im):
        return self.ib.tf(im)

    @torch.no_grad()
    def __call__(self, x, amp=True):
        with _amp(amp):
            e = self.ib.m({self.ib.MT.VISION: x.to(DEV)})[self.ib.MT.VISION]
        return {"": e.float().cpu().numpy()}


def ib_audio_clips(w, sr=16000, clip=2.0, n=None, mel=128, frames=204, mean=-4.268, std=9.138):
    """imagebind.data.load_and_transform_audio_data for one waveform: n clips of `clip` s, evenly spaced from start to
    end (pytorchvideo ConstantClipsPerVideoSampler), kaldi fbank, padded / cut to `frames`, normalised.
    ImageBind's default is n = 3 (6 s heard). n=None covers the whole file instead: max(3, ceil(duration / clip)),
    identical to the default up to 6 s. With 3 clips, two 12 s crops of one utterance gave embeddings at cos 0.79."""
    import torchaudio
    if n is None:
        n = max(3, int(np.ceil(len(w) / sr / clip)))
    mx = max(len(w) / sr - clip, 0.0)
    out = []
    for k in range(n):
        s = mx * k / max(n - 1, 1)
        seg = torch.from_numpy(np.ascontiguousarray(w[int(s * sr):int((s + clip) * sr)], dtype=np.float32))[None]
        if seg.shape[1] < 400:                                         # shorter than one 25 ms window
            seg = F.pad(seg, (0, 400 - seg.shape[1]))
        seg = seg - seg.mean()
        fb = torchaudio.compliance.kaldi.fbank(seg, htk_compat=True, sample_frequency=sr, use_energy=False,
                                               window_type="hanning", num_mel_bins=mel, dither=0.0,
                                               frame_length=25, frame_shift=10).T
        fb = F.pad(fb, (0, frames - fb.shape[1])) if fb.shape[1] < frames else fb[:, :frames]
        out.append((fb[None] - mean) / std)
    return torch.stack(out)                                            # (n, 1, mel, frames)


class _IBAudio:
    def __init__(self, ib):
        self.ib = ib

    @torch.no_grad()
    def __call__(self, w, amp=True):
        x = ib_audio_clips(w)[None].to(DEV)                            # (1, clips, 1, mel, T): the model averages clips
        with _amp(amp):
            e = self.ib.m({self.ib.MT.AUDIO: x})[self.ib.MT.AUDIO]
        return {"": e[0].float().cpu().numpy()}


# ============================================================================ voice encoders
# interface: __call__(16 kHz mono float32, amp) -> {suffix: (d,) array}; one file at a time, so no padding enters a mean

def _windows(w, W, H):
    """flag_extract.extract_ssl's windowing: fixed windows, the last one kept only if >= 1 s."""
    starts = [0] if len(w) <= W else list(range(0, len(w) - W + H, H))
    for i, s in enumerate(starts):
        seg = w[s:s + W]
        if i and len(seg) < 16000:
            break
        yield seg


class W2V2AgeGender:
    def __init__(self, cache, window_sec=20.0, hop_sec=10.0):
        import agegender as AG
        self.m, self.proc = AG.build_voice_age_gender(DEV)
        self.W, self.H = int(window_sec * 16000), int(hop_sec * 16000)

    @torch.no_grad()
    def __call__(self, w, amp=True):
        acc, tot = None, 0
        for seg in _windows(w, self.W, self.H):
            x = self.proc(seg, sampling_rate=16000, return_tensors="pt").input_values.to(DEV)
            with _amp(amp):
                h = self.m.wav2vec2(x)[0].float().mean(1)[0]
            acc = h * len(seg) if acc is None else acc + h * len(seg)
            tot += len(seg)
        h = (acc / tot)[None]
        return {"": h[0].cpu().numpy(), "age": self.m.age(h)[0].cpu().numpy(),
                "gen": torch.softmax(self.m.gender(h), 1)[0].cpu().numpy()}


# ============================================================================ the organiser's own encoders, other layers (S3 / S4)
class BTCFace:
    """The organiser's VGG-Face (flag_extract.VGGFace; fc7 reproduces their csv at cos 1.0000), with the layers
    docs/OPEN_DIRECTIONS.md S3 / S4 need. fp32 throughout, so fc7 can be checked against the organiser csv.
        ""     fc7 post-ReLU 4096 (the organiser feature itself)
        fc6    fc6 post-ReLU 4096
        pool5  pool5 averaged over its 7 x 7 grid, 512
        flip   fc7 of the horizontally mirrored image (test-time augmentation)"""

    def __init__(self, cache):
        self.v = X.VGGFace(cache / "_vggface")
        self._h = {}
        self.v.m.relu6.register_forward_hook(lambda mod, i, o: self._h.__setitem__("fc6", o.detach().clone()))
        self.v.m.pool5.register_forward_hook(lambda mod, i, o: self._h.__setitem__("pool5", o.mean((2, 3))))

    def prep(self, im):
        return torch.from_numpy(self.v.prep(im))

    @torch.no_grad()
    def __call__(self, x, amp=False):
        x = x.to(DEV)
        self.v.m(x)
        out = {"": self.v._o["x"].float().cpu().numpy(), "fc6": self._h["fc6"].float().cpu().numpy(),
               "pool5": self._h["pool5"].float().cpu().numpy()}
        self.v.m(x.flip(-1))
        out["flip"] = self.v._o["x"].float().cpu().numpy()
        return out


class BTCVoice:
    """The organiser's ECAPA (yangwang825/ecapa-tdnn-vox2; the 192 output reproduces their csv at cos 1.0000), one
    file at a time (no padding).
        ""    192 output (the organiser feature itself)
        asp   attentive-statistics pooling output, 3072: the layer before the 192 projection. EXP-010B's 6144 came
              from a different model (speechbrain spkrec-ecapa-voxceleb)
        tta   mean of the 192 output over the whole file and two 80 % crops (start, end); = "" for files < 2 s"""

    def __init__(self, cache):
        self.v = X.OrganiserVoice(cache / "_ecapa_btc")
        self._h = {}
        self.v.m.mods.embedding_model.asp.register_forward_hook(lambda mod, i, o: self._h.__setitem__("asp", o))

    @torch.no_grad()
    def __call__(self, w, amp=False):
        e = self.v([w])[0]
        asp = self._h["asp"][0, :, 0].float().cpu().numpy()
        views = [e]
        if len(w) >= 32000:
            L = int(0.8 * len(w))
            views += [self.v([w[:L]])[0], self.v([w[-L:]])[0]]
        return {"": e, "asp": asp, "tta": np.mean(views, 0)}


def check_organiser(out: Path, scan):
    """cos(our re-extracted fc7 / 192, the organiser's train csv): must be ~1.0000, else the layers beside them are
    from a different network state and S3 / S4 would compare the wrong thing."""
    res = {}
    for kind, csv, name in [("face", "train_English_faces.csv", "btcface"), ("voice", "train_English_voices.csv", "btcvoice")]:
        p, f = scan["files"].get(csv), out / f"{kind}_{name}_train.npy"
        if not p or not f.exists():
            res[kind] = "no csv or no array"; continue
        given = pd.read_csv(p[0], header=None).iloc[:, :-1].values.astype(np.float32)
        ours = np.load(f)
        if len(given) != len(ours):
            res[kind] = f"rows {len(ours)} vs csv {len(given)}"; continue
        c = _cos(given, ours)
        res[kind] = dict(mean=round(float(c.mean()), 5), min=round(float(c.min()), 5))
    return res


# ============================================================================ building + running
_SHARED = {}


def build(name, cache):
    """One encoder by name. ImageBind is loaded once and shared by ibv / iba (4.5 GB) until release()."""
    if name in ("ibv", "iba"):
        if "ib" not in _SHARED:
            _SHARED["ib"] = ImageBind(cache)
        return _SHARED["ib"].vision() if name == "ibv" else _SHARED["ib"].audio()
    return {"vitage": ViTAge, "farl": FaRL, "siglip2": SigLIP2, "adaface": AdaFace,
            "w2v2ag": W2V2AgeGender, "btcface": BTCFace, "btcvoice": BTCVoice}[name](cache)


def release():
    _SHARED.clear()
    torch.cuda.empty_cache()


def _download(url, cache):
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / url.rsplit("/", 1)[1]
    if not f.exists():
        log("downloading", url)
        torch.hub.download_url_to_file(url, str(f) + ".part", progress=False)
        Path(str(f) + ".part").rename(f)
    return f


def run_faces(enc, srcs, bs=64, label=""):
    """srcs: paths or zero-arg callables returning bytes. -> {suffix: (N, d) float32}."""
    out, t0 = {}, time.time()
    for s in range(0, len(srcs), bs):
        x = torch.stack([enc.prep(read_rgb(p() if callable(p) else p)) for p in srcs[s:s + bs]])
        for k, v in enc(x).items():
            out.setdefault(k, []).append(v)
        if (s // bs) % 40 == 0:
            log(f"  {label} {s + len(x)}/{len(srcs)}  {time.time() - t0:.0f}s")
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def run_voices(enc, n, load, label=""):
    """load(i) -> 16 kHz mono waveform. -> {suffix: (N, d) float32}."""
    out, t0 = {}, time.time()
    for i in range(n):
        for k, v in enc(load(i)).items():
            out.setdefault(k, []).append(np.atleast_1d(v))
        if i % 1000 == 0 or i == n - 1:
            el = time.time() - t0
            log(f"  {label} {i + 1}/{n}  {el:.0f}s  eta {el / (i + 1) * (n - i - 1):.0f}s")
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


def _cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def check_face(enc, srcs):
    """min cosine of (batched vs one-by-one) and (fp16 vs fp32) on a few images. Both must be ~1."""
    x = torch.stack([enc.prep(read_rgb(p)) for p in srcs])
    full = enc(x, amp=False)[""]
    single = np.concatenate([enc(x[i:i + 1], amp=False)[""] for i in range(len(x))])
    half = enc(x, amp=True)[""]
    return dict(batch=float(_cos(full, single).min()), fp16=float(_cos(full, half).min()))


def check_voice(enc, waves):
    a = np.stack([enc(w, amp=False)[""] for w in waves])
    b = np.stack([enc(w, amp=True)[""] for w in waves])
    return dict(fp16=float(_cos(a, b).min()))


def seed_from_previous(scan, out: Path, encoders):
    """Resume across Kaggle versions: copy into `out` every array of these encoders found in the inputs (an earlier
    run's output attached as a dataset). Only our own file patterns are copied, so e.g. feats_v2's ArcFace arrays
    are not. An encoder whose arrays are all present is then skipped entirely (see done)."""
    import re, shutil
    pat = re.compile(rf"^((face|voice)_({'|'.join(encoders)})[a-z]*_.+|ext_.+_(face|voice)_({'|'.join(encoders)})[a-z]*)\.npy$")
    copied = []
    for n, ps in scan["files"].items():
        if pat.match(n) and not (out / n).exists():
            shutil.copy(ps[0], out / n); copied.append(n)
    log(f"seeded {len(copied)} arrays from earlier outputs" + (f": {sorted(copied)[:6]}..." if copied else ""))
    return copied


def done(name, kind, S, ext, out: Path):
    """All main arrays of this encoder exist (v4 splits + every external source), so it need not even be loaded."""
    return (all((out / f"{kind}_{name}_{tag(k)}.npy").exists() for k in S)
            and all((out / f"ext_{src}_{kind}_{name}.npy").exists() for src in ext))


def extract_v4(name, enc, S, out: Path, kind):
    """All five v4 splits for one encoder; an existing main array is skipped (resumable)."""
    for split, sp in S.items():
        f = out / f"{kind}_{name}_{tag(split)}.npy"
        if f.exists():
            log("skip (exists)", f.name); continue
        if kind == "face":
            R = run_faces(enc, [sp["root"] / p for p in sp["jpg"]], label=f"{name} {split}")
        else:
            R = run_voices(enc, len(sp["wav"]), lambda i: X.load_audio(sp["root"] / sp["wav"][i]), label=f"{name} {split}")
        for k, v in R.items():                                        # side outputs first: the main file marks "done"
            if k:
                np.save(out / f"{kind}_{name}{k}_{tag(split)}.npy", v)
        np.save(f, R[""])
        log(f"{f.name} {R[''].shape}")


def scan_inputs(inp: Path, media=("faces", "voices")):
    """One walk over the attached inputs that never enters a faces/ or voices/ folder. An extracted MAV-Celeb v1/v2
    on a Kaggle mount holds hundreds of thousands of files, so every rglob over /kaggle/input costs minutes (the first
    version of this notebook ran about eight of them before extracting anything).
    -> {"files": name -> [paths], "roots": folders holding both faces/ and voices/}"""
    files, roots = {}, []
    for dp, dns, fns in os.walk(inp, followlinks=True):
        feat = Path(dp).name == "features"              # the organiser's csv live in features/{faces,voices}/: enter
        if not feat and all(m in dns for m in media):
            roots.append(Path(dp))
        dns[:] = sorted(d for d in dns if feat or d not in media)
        for f in fns:
            files.setdefault(f, []).append(Path(dp) / f)
    log(f"scanned {inp}: {sum(map(len, files.values()))} files outside media folders, {len(roots)} media roots")
    return dict(files=files, roots=roots)


def _has_media(txt: Path, wav_col, jpg_col):
    row = pd.read_csv(txt, sep=" ", header=None, nrows=1).iloc[0]
    return (txt.parent / row[wav_col]).exists() and (txt.parent / row[jpg_col]).exists()


def v4_index(scan, work: Path):
    """flag_extract.index_splits on the raw v4 data, each split searched only inside the folder of its own txt (never
    /kaggle/input or a common parent: train and dev may sit in different datasets, and an rglob there walks v1/v2).
    A features-only copy of the txt files (no media beside it) is ignored. Zipped v4 data is extracted to `work`;
    only the v4 zips ever are."""
    F = scan["files"]
    train = [p for p in F.get("train_English.txt", []) if _has_media(p, 2, 3)]
    dev = {f"{prot}/{lang}": [p for p in F.get(f"{lang}_test.txt", []) if f"/{prot}/" in p.as_posix() and _has_media(p, 1, 2)]
           for prot in ("no_gender", "gender") for lang in ("English", "Bangla")}
    if train and all(dev.values()):
        roots = {"train": train[0].parent, **{k: v[0].parent for k, v in dev.items()}}
        return X.index_splits(train[0].parent, roots)
    import zipfile
    zips = [p for n, ps in F.items() if n.lower().startswith(("train_set", "dev_set")) and n.lower().endswith(".zip")
            for p in ps]
    assert zips, "no v4 media under the inputs: attach the raw dataset (train_set.zip + dev_set.zip)"
    work.mkdir(parents=True, exist_ok=True)
    for zp in zips:
        marker = work / f".{zp.stem}.done"
        if not marker.exists():
            log("extracting", zp.name)
            with zipfile.ZipFile(zp) as z:
                z.extractall(work)
            marker.touch()
    return X.index_splits(work)


def ext_source(scan, name):
    """flag_extract.find_attached_source on the scan: the attached zip for `name` (largest), else the extracted
    folder whose path names the source (.../v1/ for v1_complete; Kaggle slugs use hyphens)."""
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


# ---------------------------------------------------------------------------- MAV-Celeb v1/v2 (validation rows)
def _media_key(member):
    """'.../faces/<id>/<lang>/<video>/<file>' -> 'faces/<id>/<lang>/<video>/<file>' (drops any version folder)."""
    p = member.replace("\\", "/").split("/")
    for kind in ("faces", "voices"):
        if kind in p:
            return "/".join(p[p.index(kind):])
    return None


class ExtRows:
    """The exact rows of an existing ext_<name>_{face,voice}_meta.csv, read from the attached zip or folder.
    A folder (Kaggle auto-extracts uploaded zips) is never listed: each row is opened by its path under the root,
    after one threaded existence check, so a missing file fails here and not halfway through an encoder."""

    def __init__(self, src, name, meta_dir: Path, max_sec=12.0):
        fm = pd.read_csv(meta_dir / f"ext_{name}_face_meta.csv").path
        vm = pd.read_csv(meta_dir / f"ext_{name}_voice_meta.csv").path
        src = Path(src)
        if src.is_dir():
            from concurrent.futures import ThreadPoolExecutor
            self.face, self.voice = [src / _media_key(p) for p in fm], [src / _media_key(p) for p in vm]
            with ThreadPoolExecutor(32) as ex:
                ok = list(ex.map(lambda q: q.exists(), self.face + self.voice))
            if not all(ok):
                raise KeyError(f"{ok.count(False)} rows missing, e.g. {(self.face + self.voice)[ok.index(False)]}")
            self.read = lambda q: q.read_bytes()
        else:
            import zipfile
            z = zipfile.ZipFile(src)
            by = {}
            for i in z.infolist():
                k = None if i.is_dir() else _media_key(i.filename)
                if k:
                    by[k] = i.filename
            self.face, self.voice = [by[_media_key(p)] for p in fm], [by[_media_key(p)] for p in vm]
            self.read = z.read
        self.L = int(max_sec * 16000)          # centre crop, as extract_external(max_sec=12) did for the BTC features
        log(f"{name}: {len(self.face)} face rows, {len(self.voice)} voice rows resolved in {src}")

    def wave(self, i):
        w = X.load_audio(io.BytesIO(self.read(self.voice[i])))
        if len(w) > self.L:
            s = (len(w) - self.L) // 2
            w = w[s:s + self.L]
        return w


def extract_ext(name, enc, rows: ExtRows, src_name, out: Path, kind):
    f = out / f"ext_{src_name}_{kind}_{name}.npy"
    if f.exists():
        log("skip (exists)", f.name); return
    if kind == "face":
        R = run_faces(enc, [(lambda m=m: rows.read(m)) for m in rows.face], label=f"{name} {src_name}")
    else:
        R = run_voices(enc, len(rows.voice), rows.wave, label=f"{name} {src_name}")
    for k, v in R.items():
        if k:
            np.save(out / f"ext_{src_name}_{kind}_{name}{k}.npy", v.astype(np.float16))
    np.save(f, R[""].astype(np.float16))
    log(f"{f.name} {R[''].shape}")


# ============================================================================ probe: sanity + ImageBind zero-shot
def probe(out: Path, spk, gmap, seeds=(1, 2, 3), n=3000):
    """Train split, true labels, no training.
    * identity EER of every new stream on its own (centred cosine; catches broken preprocessing: AdaFace must land
      near ArcFace's ~2 %, the attribute streams far above it);
    * ImageBind zero-shot face<->voice EER (raw and train-centred cosine): the X3 go / no-go (near 50 % -> stop);
    * attribute sanity: w2v2 gender vs our meta labels, mean predicted ages."""
    from flag_lib import build_trials, eer_from_scores, l2n
    import agegender as AG
    rows = []
    trials = {(s, sg): build_trials(spk, gmap, seed=s, n_pos=n, n_neg=n, same_gender=sg)
              for s in seeds for sg in (False, True)}

    def eer(Sf, Sv):
        r = {}
        for sg in (False, True):
            r["g" if sg else "ng"] = round(float(np.mean([
                eer_from_scores(np.sum(Sf[trials[s, sg][0]] * Sv[trials[s, sg][1]], 1), trials[s, sg][2])
                for s in seeds])), 2)
        return r

    for kind in ("face", "voice"):
        for f in sorted(out.glob(f"{kind}_*_train.npy")):
            A = np.load(f).astype(np.float32)
            if A.ndim != 2 or A.shape[1] < 8:
                continue                                                # side outputs (age, gender probs)
            Z = l2n((A - A.mean(0)) / (A.std(0) + 1e-6))
            rows.append(dict(test="identity (same modality)", stream=f.stem.rsplit("_", 1)[0], dim=A.shape[1], **eer(Z, Z)))
    if (out / "face_ibv_train.npy").exists() and (out / "voice_iba_train.npy").exists():
        Fv, Av = np.load(out / "face_ibv_train.npy"), np.load(out / "voice_iba_train.npy")
        rows.append(dict(test="ImageBind zero-shot face<->voice", stream="raw cosine", dim=Fv.shape[1], **eer(l2n(Fv), l2n(Av))))
        rows.append(dict(test="ImageBind zero-shot face<->voice", stream="train-centred", dim=Fv.shape[1],
                         **eer(l2n(Fv - Fv.mean(0)), l2n(Av - Av.mean(0)))))
    P = pd.DataFrame(rows)
    sanity = {}
    if (out / "voice_w2v2aggen_train.npy").exists():
        sanity["w2v2 gender agreement with meta"] = round(AG.gender_agreement(
            np.load(out / "voice_w2v2aggen_train.npy"), [gmap[s] for s in spk]), 3)
        sanity["w2v2 mean age (years)"] = round(float(np.load(out / "voice_w2v2agage_train.npy").mean() * 100), 1)
    if (out / "face_vitageprob_train.npy").exists():
        mids = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75])            # centres of the 9 FairFace age bins
        sanity["vit-age mean age (years)"] = round(float((np.load(out / "face_vitageprob_train.npy") @ mids).mean()), 1)
    return P, sanity
