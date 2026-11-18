"""Generate FLAG_27_embed_extract.ipynb: EMBED-SCOUT extraction (plan 1/10, after ERR-03 / COMBO-01).
New pretrained voice / face embeddings for the exact rows of the eda bundle (v4 train in organiser order, MAV-Celeb v1/v2
in ext_*_meta order) and for the dev lists, so FLAG_28_l0_scout can swap them into the level-0 harness.

The notebook reuses the FLAG_12 cells verbatim (data discovery, audio loading, batching, ImageBind, the extraction loop,
resume) and adds one cell with the new encoders. Full code in the cells: the notebook imports no team .py.
    python build_scout_notebooks.py
"""
from pathlib import Path
import build_eval_notebooks as B12

HERE = Path(__file__).resolve().parent

INTRO = """
# FLAG 2027 — NB-27: EMBED-SCOUT extraction (new pretrained voice / face embeddings)

ERR-03 showed that ImageBind carries the system and that the MLP bridge (VGG face <-> organiser ECAPA voice) is the weak, overfitting part. This notebook extracts **other pretrained embeddings**, so that FLAG_28 can test them in the bridge under the same level-0 protocol. Every model is used frozen; nothing is trained or fine-tuned.

| array | model | pretraining | rule check (30/09 ruling) |
|---|---|---|---|
| `voice_rdn6vox` | ReDimNet2-B6, `lm`, VoxCeleb2 (PalabraAI, MIT) | VoxCeleb2 | allowed (VoxCeleb) |
| `voice_wsr221` | WeSpeaker ResNet221-LM (`english`) | VoxCeleb2 | allowed |
| `voice_wssim34` | WeSpeaker SimAM-ResNet34, VoxBlink2 -> VoxCeleb2 (`vblinkf`) | VoxBlink2 + VoxCeleb2 | allowed (named by the organisers) |
| `voice_wavlmsv` | microsoft/wavlm-base-plus-sv (x-vector head) | SSL: Libri-Light + GigaSpeech + VoxPopuli English; SV: VoxCeleb1 | English-only SSL data: no Bengali; declare |
| `voice_unisatsv` | microsoft/unispeech-sat-base-plus-sv | same data as WavLM Base+; SV: VoxCeleb1 | as above |
| `voice_sbresnet` | speechbrain/spkrec-resnet-voxceleb | VoxCeleb1 + 2 | allowed |
| `face_facenet` | InceptionResnetV1, VGGFace2 (facenet-pytorch) | VGGFace2 | face encoders are free |
| `face_ibvflip` | ImageBind-huge vision, mean of the image and its mirror | (ImageBind) | per-sample augmentation of ONE image; uses no other test sample |

**Not included** (see Experiment/EXPERIMENT_LOG.md, entry 100):
- W2V-BERT 2.0 SV: its SSL pretraining covers 143 languages, Bengali included;
- ReDimNet2 `vb2+vox2+cnc2`: also trained on CN-Celeb, which is not in the organisers' list;
- CAM++ (WeSpeaker: Chinese data; 3D-Speaker: needs ModelScope);
- CAV-MAE: the Dropbox checkpoints returned HTML in FLAG_14, and the code needs timm 0.4.5;
- WAVE-7B: compute.

**Rows:** identical to the arrays we already use.
- v4 `train` and the dev lists: the organiser's `.txt` order.
- MAV-Celeb v1/v2: the rows of `ext_<src>_{face,voice}_meta.csv`, voices cut to a 12 s centre crop, like the ECAPA arrays of the eda bundle.

Voices are embedded **one file at a time**: none of these models takes a length mask, so padding would change the embedding.

**Inputs (Add Input):** the same as for FLAG_12:
1. the raw v4 dataset (`train_set` + `dev_set`);
2. MAV-Celeb v1 and v2 + `flag2027-ext-meta`;
3. optional: this notebook's earlier output, to resume.

**Settings:** GPU T4, **Internet ON** (HuggingFace, torch.hub, ModelScope, GitHub releases).

**Time:** about 40-60 min per voice model (about 51 000 files: 40 000 of them v1/v2 voices read from the zips), a few minutes per face model. **The whole list does not fit in one session.** The list `ENCODERS` (first cell) is in priority order and the run is resumable: run, Save Version, attach the output, run again until every array exists.

**Output:** `/kaggle/working/feats_scout/` -> dataset **`flag2027-feats-scout`** (the input of FLAG_28).
"""

ENV = (B12.ENV12
       .replace('OUT = WORKDIR / "feats_eval"', 'OUT = WORKDIR / "feats_scout"')
       .replace('ENCODERS = ["arcface", "ecapa", "vitage", "w2v2ag", "ibv", "iba"]   # ibv right before iba: one ImageBind load',
                'ENCODERS = ["rdn6vox", "wsr221", "wssim34", "wavlmsv",            # batch 1 (priority order)\n'
                '            "facenet", "ibvflip", "unisatsv", "sbresnet"]         # batch 2')
       .replace('EXT_ENCODERS = ["arcface", "ecapa"]                                 # their ImageBind / age exist (FLAG_10)',
                'EXT_ENCODERS = list(ENCODERS)                                       # every model also on the v1/v2 rows'))
assert 'feats_scout' in ENV and '"rdn6vox"' in ENV and 'EXT_ENCODERS = list(ENCODERS)' in ENV

PIP = (B12.PIP12.replace(
    '''    !pip -q install --no-deps "insightface==2.0" "git+https://github.com/facebookresearch/ImageBind" 2>&1 | tail -2''',
    '''    !pip -q install --no-deps "insightface==2.0" "git+https://github.com/facebookresearch/ImageBind" 2>&1 | tail -2
    # WeSpeaker and facenet-pytorch: --no-deps, their pins would replace Kaggle's torch; the few runtime deps by hand
    !pip -q install --no-deps "git+https://github.com/wenet-e2e/wespeaker.git" "facenet-pytorch==2.6.0" 2>&1 | tail -2
    !pip -q install kaldiio silero-vad 2>&1 | tail -2'''))
assert "wespeaker" in PIP

DATA = B12.DATA12.replace('re.search(r"feats[-_]eval", p.as_posix())', 're.search(r"feats[-_]scout", p.as_posix())')
assert 'feats[-_]scout' in DATA

ENC_NEW = r'''
# ============================================================================ EMBED-SCOUT encoders
# Voice: __call__(waveform 16 kHz mono float32, use_amp) -> {"": (d,)}, one file at a time, fp32.
# Face: prep(PIL image) -> tensor; __call__(batch tensor, use_amp) -> {"": (N, d)}.

class OneFileVoice:
    kind = "voice"

    def __call__(self, w, use_amp=False):
        return {"": self.emb(np.ascontiguousarray(w, dtype=np.float32))}


class ReDimNet2(OneFileVoice):
    """PalabraAI/redimnet2 (MIT): raw waveform in, no length mask (padding would change the output)."""
    def __init__(self, dataset="vox2"):
        self.m = torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name="b6", train_type="lm", dataset=dataset,
                                pretrained=True, trust_repo=True).to(DEV).eval()

    @torch.no_grad()
    def emb(self, w):
        e = self.m(torch.from_numpy(w)[None].to(DEV))
        e = e[0] if isinstance(e, (tuple, list)) else e
        return e.reshape(-1).float().cpu().numpy()


class WeSpeakerVoice(OneFileVoice):
    """wespeaker.load_model(tag): fbank + CMN inside; checkpoints from ModelScope. WeSpeaker reads wav files with
    torchaudio.load(normalize=False), i.e. int16-scaled samples: our [-1, 1] waveform is scaled by 2^15 to match."""
    def __init__(self, tag):
        # wespeaker.frontend imports s3prl / openai-whisper / peft for its SSL frontends at import time; the fbank models
        # used here need none of them, so those modules are stubbed (found by the local test, log 100)
        # (diarization helpers too: umap / hdbscan / onnxruntime, unused for embeddings)
        for name, attr in (("wespeaker.frontend.s3prl", "S3prlFrontend"), ("wespeaker.frontend.whisper_encoder", "whisper_encoder"),
                           ("wespeaker.frontend.w2vbert", "W2VBertFrontend"), ("wespeaker.diar.umap_clusterer", "cluster"),
                           ("wespeaker.diar.extract_emb", "subsegment")):
            if name not in sys.modules:
                stub = types.ModuleType(name); setattr(stub, attr, None); sys.modules[name] = stub
        import wespeaker
        try:
            self.m = wespeaker.load_model(tag)
        except FileNotFoundError:           # vblinkf: its zip carries __MACOSX entries and wespeaker's unzip misses the files
            d = Path.home() / ".wespeaker" / tag
            for zf_path in d.glob("*.zip"):
                with zipfile.ZipFile(zf_path) as zf:
                    for mem in zf.namelist():
                        if not mem.startswith("__MACOSX") and not mem.endswith("/"):
                            (d / Path(mem).name).write_bytes(zf.read(mem))
            self.m = wespeaker.load_model(str(d))
        self.m.set_device("cuda:0" if DEV.type == "cuda" else "cpu")

    @torch.no_grad()
    def emb(self, w):
        e = self.m.extract_embedding_from_pcm(torch.from_numpy(w * 32768.0)[None], 16000)
        return e.reshape(-1).float().cpu().numpy()


class HFXVector(OneFileVoice):
    """transformers AutoModelForAudioXVector (WavLM / UniSpeech-SAT SV checkpoints): the x-vector embedding."""
    def __init__(self, repo):
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
        self.fe = AutoFeatureExtractor.from_pretrained(repo)
        self.m = AutoModelForAudioXVector.from_pretrained(repo).to(DEV).eval()

    @torch.no_grad()
    def emb(self, w):
        x = self.fe(w, sampling_rate=16000, return_tensors="pt")
        return self.m(**{k: v.to(DEV) for k, v in x.items()}).embeddings[0].float().cpu().numpy()


class SBResNet(OneFileVoice):
    """speechbrain/spkrec-resnet-voxceleb (VoxCeleb1 + 2)."""
    def __init__(self):
        EncoderClassifier = speechbrain_import()
        kw = {}
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        self.m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-resnet-voxceleb", savedir=str(CACHE / "sb_resnet"),
                                                run_opts={"device": str(DEV)}, **kw)
        self.m.eval()

    @torch.no_grad()
    def emb(self, w):
        return self.m.encode_batch(torch.from_numpy(w)[None].to(DEV)).reshape(-1).float().cpu().numpy()


class FaceNet:
    """facenet-pytorch InceptionResnetV1 pretrained on VGGFace2: 160 x 160, (x - 127.5) / 128, 512-d."""
    kind = "face"

    def __init__(self):
        from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
        self.m = InceptionResnetV1(pretrained="vggface2").to(DEV).eval()

    def prep(self, im):
        from PIL import Image
        x = torch.from_numpy(np.asarray(im.resize((160, 160), Image.BILINEAR), dtype=np.float32)).permute(2, 0, 1)
        return (x - 127.5) / 128.0

    @torch.no_grad()
    def __call__(self, x, use_amp=False):
        return {"": self.m(x.to(DEV)).float().cpu().numpy()}


class IBVisionFlip(IBVision):
    """ImageBind vision, mean of the embedding of the image and of its mirror image (one image only)."""
    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        a = IBVision.__call__(self, x, use_amp)[""]
        b = IBVision.__call__(self, torch.flip(x, dims=[3]), use_amp)[""]
        return {"": (a + b) / 2}


NEW = {"rdn6vox": lambda: ReDimNet2("vox2"), "wsr221": lambda: WeSpeakerVoice("english"),
       "wssim34": lambda: WeSpeakerVoice("vblinkf"), "wavlmsv": lambda: HFXVector("microsoft/wavlm-base-plus-sv"),
       "unisatsv": lambda: HFXVector("microsoft/unispeech-sat-base-plus-sv"), "sbresnet": SBResNet,
       "facenet": FaceNet, "ibvflip": IBVisionFlip}
_build12 = build


def build(name):
    return NEW[name]() if name in NEW else _build12(name)


ARRAYS_NEW = {k: ("face" if k in ("facenet", "ibvflip") else "voice", {"": k}) for k in NEW}
print("new encoders:", list(NEW))
'''

RUN = B12.RUN12.replace(
    '''MAIN = {"arcface": "", "ecapa": "192", "vitage": "", "w2v2ag": "", "ibv": "", "iba": ""}''',
    '''MAIN = {"arcface": "", "ecapa": "192", "vitage": "", "w2v2ag": "", "ibv": "", "iba": ""}
ARRAYS.update(ARRAYS_NEW); MAIN.update({k: "" for k in ARRAYS_NEW})          # EMBED-SCOUT arrays''')
assert "ARRAYS.update(ARRAYS_NEW)" in RUN

DONE = (B12.DONE12.replace("Save Version (Save & Run All) -> Output -> New Dataset -> name it flag2027-feats-eval",
                           "Save Version (Save & Run All) -> Output -> New Dataset -> name it flag2027-feats-scout "
                           "(re-run with this output attached until every encoder is done)"))
assert "flag2027-feats-scout" in DONE

if __name__ == "__main__":
    B12.nb([("markdown", INTRO), ("code", ENV), ("code", PIP),
            ("markdown", "## 1. Find the data (FLAG_12 cell): organiser lists with media, v1/v2 rows, resume"), ("code", DATA),
            ("markdown", "## 2. Audio loading, batching, helpers (FLAG_12 cell)"), ("code", B12.AUDIO12),
            ("markdown", "## 3. FLAG_12 encoders (ImageBind is reused by `ibvflip`)"), ("code", B12.ENC12),
            ("markdown", "## 4. EMBED-SCOUT encoders"), ("code", ENC_NEW),
            ("markdown", "## 5. Extract every encoder over every split, then the v1/v2 rows (FLAG_12 loop, resumable)"), ("code", RUN),
            ("markdown", "## 6. Manifest"), ("code", DONE)], "FLAG_27_embed_extract.ipynb")
