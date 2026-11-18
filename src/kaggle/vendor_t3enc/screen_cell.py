# ============================================================================ ENC-SCREEN-01 candidate encoders (frozen), screened locally with the A-arm head
# Every sample on its own. Rule status: face encoders: any; voice encoders: pretrained on VoxCeleb (allowed by the organisers),
# no training by us. w2v-BERT 2.0 based models were left out (their self-supervised pretraining very likely contains Bengali).
#   rn293A   WeSpeaker voxceleb_resnet293_LM (VoxCeleb2, large-margin fine-tuned), official ModelScope zip; fbank + CMN, 256-d
#   ecapa2A  ECAPA2 (Jenthe/ECAPA2 @ 207cb6d, VoxCeleb2, CC BY-NC 4.0), TorchScript, 16 kHz waveform, 192-d
#   sig2V    google/siglip2-so400m-patch14-384 @ e8e4872 (Apache-2.0), image features 1152-d
#   dino2V   facebook/dinov2-large @ 47b73ee (Apache-2.0), CLS token (pooler output) 1024-d
SAM_ZIPS.update({"rn293A": ("voxceleb_resnet293_LM.zip", 124116126)})
ECAPA2_ID, ECAPA2_REV = "Jenthe/ECAPA2", "207cb6d137c671a12ba820ebec3b719549b06c0f"
SIG2_ID, SIG2_REV = "google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
DINO2_ID, DINO2_REV = "facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"


class ECAPA2Voice:
    kind = "voice"

    def __init__(self):
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(ECAPA2_ID, "ecapa2.pt", revision=ECAPA2_REV)
        self.m = torch.jit.load(f, map_location=DEV).eval()

    @torch.no_grad()
    def __call__(self, w, use_amp=True):
        x = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))[None].to(DEV)
        return {"": self.m(x).float().reshape(-1).cpu().numpy()}


class SigLIP2Face:
    kind = "face"

    def __init__(self):
        from transformers import AutoModel, AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained(SIG2_ID, revision=SIG2_REV, use_fast=False)   # pinned: same pixels in every notebook
        self.m = AutoModel.from_pretrained(SIG2_ID, revision=SIG2_REV, torch_dtype=torch.float32).to(DEV).eval()

    def prep(self, im):
        return self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        with torch.autocast("cuda", enabled=use_amp and DEV.type == "cuda"):
            f = self.m.get_image_features(pixel_values=x.to(DEV))
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        return {"": f.float().cpu().numpy()}


class DINOv2Face:
    kind = "face"

    def __init__(self):
        from transformers import AutoModel, AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained(DINO2_ID, revision=DINO2_REV, use_fast=False)
        self.m = AutoModel.from_pretrained(DINO2_ID, revision=DINO2_REV, torch_dtype=torch.float32).to(DEV).eval()

    def prep(self, im):
        return self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        with torch.autocast("cuda", enabled=use_amp and DEV.type == "cuda"):
            o = self.m(pixel_values=x.to(DEV))
        return {"": o.pooler_output.float().cpu().numpy()}


NEW = {"sig2V": SigLIP2Face, "dino2V": DINOv2Face, "rn293A": lambda: SimAMVoice("rn293A"), "ecapa2A": ECAPA2Voice}
ARRAYS_NEW = {"sig2V": ("face", {"": "sig2"}), "dino2V": ("face", {"": "dino2"}), "rn293A": ("voice", {"": "rn293"}),
              "ecapa2A": ("voice", {"": "ecapa2"})}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {}
