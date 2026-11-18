# ============================================================================ T3ENC-01 encoders: the encoder family of Track 3's VoxBlink system, frozen
# Track 3 (a teammate's VoxBlink pipeline, CB Bangla ng ~16 / g ~25) used these four encoders; here they run on OUR rows
# (MAV-Celeb v4 train, v1/v2, dev lists; and VoxCeleb2 in FLAG_41), so we can test locally whether the encoders and a
# cross-modal head carry the gain WITHOUT VoxBlink data (docs/PLAN_VOXBLINK_2026-10-07.md, V1). Every sample on its own.
#   farl16V   FaRL ViT-B/16 LAION-Face 20M **ep16** (github.com/FacePerceiver/FaRL release, 652 MB) on openai CLIP ViT-B/16
#             (CLIP preprocessing: 224 bicubic, centre crop); 512-d image embedding (our existing `farl` is the ep64 copy)
#   clipLV    openai/clip-vit-large-patch14 @ 32bd642 (transformers): get_image_features, 768-d
#   sam100pA  WeSpeaker voxblink2_samresnet100 (VoxBlink2 pretrained), official ModelScope zip (wenet/wespeaker_pretrained_models)
#   sam100fA  WeSpeaker voxblink2_samresnet100_ft (VoxBlink2 pretrained -> VoxCeleb2 large-margin fine-tune)
#             both: wespeaker fbank + CMN, int16-scaled waveform (as our wssim34 / wsr221 cells), 256-d
import zipfile as _zf

FARL16_URL = "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
FARL16_SIZE = 652165542
CLIPL_ID, CLIPL_REV = "openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"
WESPK_TREE = "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"
SAM_ZIPS = {"sam100pA": ("voxblink2_samresnet100.zip", 186890339), "sam100fA": ("voxblink2_samresnet100_ft.zip", 186890839)}
T3_STATS = {}


def fetch(url, dest, size=None):
    dest = Path(dest)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    torch.hub.download_url_to_file(url, str(dest) + ".part", progress=False)
    Path(str(dest) + ".part").replace(dest)
    assert size is None or dest.stat().st_size == size, f"{dest.name}: {dest.stat().st_size} bytes, expected {size}"
    return dest


class FaRL16Face:
    kind = "face"

    def __init__(self):
        import clip
        m, self.pre = clip.load("ViT-B/16", device="cpu", download_root=str(CACHE / "clip"))
        sd = torch.load(fetch(FARL16_URL, CACHE / "FaRL-ep16.pth", FARL16_SIZE), map_location="cpu", weights_only=False)["state_dict"]
        vis = [k for k in m.state_dict() if k.startswith("visual.")]
        hit = [k for k in vis if k in sd and sd[k].shape == m.state_dict()[k].shape]
        assert len(hit) == len(vis), f"FaRL ep16 covers {len(hit)}/{len(vis)} visual tensors"
        m.load_state_dict(sd, strict=False)
        self.m = m.visual.float().to(DEV).eval()

    def prep(self, im):
        return self.pre(im)

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        return {"": self.m(x.to(DEV)).float().cpu().numpy()}


class CLIPLFace:
    kind = "face"

    def __init__(self):
        from transformers import CLIPModel, CLIPImageProcessor
        self.proc = CLIPImageProcessor.from_pretrained(CLIPL_ID, revision=CLIPL_REV)
        self.m = CLIPModel.from_pretrained(CLIPL_ID, revision=CLIPL_REV, torch_dtype=torch.float32).to(DEV).eval()

    def prep(self, im):
        return self.proc(images=im.convert("RGB"), return_tensors="pt").pixel_values[0]

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        f = self.m.get_image_features(pixel_values=x.to(DEV))
        f = f.pooler_output if hasattr(f, "pooler_output") else f          # transformers 5 may return an output object
        return {"": f.float().cpu().numpy()}


def wespeaker_dir(zip_name, size):
    """The official zip (ModelScope, as wespeaker's own hub does), unpacked without the __MACOSX entries."""
    d = CACHE / "wespeaker" / zip_name[:-4]
    if (d / "avg_model.pt").exists() and (d / "config.yaml").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    import requests
    tree = requests.get(WESPK_TREE, timeout=120).json()["Data"]
    url = [it["Url"] for it in tree if it.get("Key") == zip_name][0]
    z = fetch(url, d / zip_name, size)
    with _zf.ZipFile(z) as zf:
        for mem in zf.namelist():
            if not mem.startswith("__MACOSX") and not mem.endswith("/"):
                (d / Path(mem).name).write_bytes(zf.read(mem))
    # some releases (voxceleb_resnet293_LM) name the pair <model>.pt / <model>.yaml instead of avg_model.pt / config.yaml
    pts, ymls = sorted(d.glob("*.pt")), sorted(d.glob("*.yaml"))
    if not (d / "avg_model.pt").exists() and len(pts) == 1:
        pts[0].rename(d / "avg_model.pt")
    if not (d / "config.yaml").exists() and len(ymls) == 1:
        ymls[0].rename(d / "config.yaml")
    assert (d / "avg_model.pt").exists() and (d / "config.yaml").exists(), f"{zip_name}: unexpected content {zf.namelist()[:6]}"
    return d


class SimAMVoice:
    kind = "voice"

    def __init__(self, name):
        for mod, attr in (("wespeaker.frontend.s3prl", "S3prlFrontend"), ("wespeaker.frontend.whisper_encoder", "whisper_encoder"),
                          ("wespeaker.frontend.w2vbert", "W2VBertFrontend"), ("wespeaker.diar.umap_clusterer", "cluster"),
                          ("wespeaker.diar.extract_emb", "subsegment")):
            if mod not in sys.modules:                     # SSL frontends / diarisation helpers: not used by fbank models
                stub = types.ModuleType(mod); setattr(stub, attr, None); sys.modules[mod] = stub
        import wespeaker
        self.m = wespeaker.load_model(str(wespeaker_dir(*SAM_ZIPS[name])))
        self.m.set_device("cuda:0" if DEV.type == "cuda" else "cpu")
        T3_STATS[name] = {"params_M": round(sum(p.numel() for p in self.m.model.parameters()) / 1e6, 1)}

    @torch.no_grad()
    def __call__(self, w, use_amp=True):
        pcm = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32) * 32768.0)[None]
        feats = self.m.compute_features(pcm, sample_rate=16000, cmn=True)
        out = self.m.model(feats.to(self.m.device))
        out = out[-1] if isinstance(out, tuple) else out
        return {"": out[0].reshape(-1).float().cpu().numpy()}


NEW = {"farl16V": FaRL16Face, "clipLV": CLIPLFace, "sam100pA": lambda: SimAMVoice("sam100pA"),
       "sam100fA": lambda: SimAMVoice("sam100fA")}


def build(name):
    return NEW[name]()


def release(keep_ib=False):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


ARRAYS_NEW = {"farl16V": ("face", {"": "farl16"}), "clipLV": ("face", {"": "clipL"}),
              "sam100pA": ("voice", {"": "sam100p"}), "sam100fA": ("voice", {"": "sam100f"})}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {}
