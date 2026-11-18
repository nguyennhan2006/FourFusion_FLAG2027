"""Generate the FACELORA-02 adapter notebooks (registered in Experiment/FACELORA-02/README.md; the adapter recipe of
FACELORA-01): a LoRA adapter on a face encoder, trained with multi-positive symmetric InfoNCE against FROZEN SimAM-ResNet100
voice embeddings of the Bengali-filtered VoxCeleb2 identities of FLAG_37 / FLAG_41 (faces and voices of different videos; no
MAV-Celeb identity in training), then the adapted face embedding of every MAV-Celeb row (v4 train, dev lists, v1/v2) and of the
VoxCeleb2 frames.
    FLAG_45_facelora.ipynb          FaRL ep16 (ViT-B/16)   -> farlA (512)   feats_facelora/
    FLAG_45c_facelora_clipl.ipynb   CLIP ViT-L/14 (HF)     -> clipA (768)   feats_facelora_clipl/
    python build_facelora_notebook.py [farl16 | clipL]
Smoke test on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6;VOX_SMOKE_IDS=6;LORA_STEPS=3;LORA_P=4;LORA_K=2
"""
import sys
from pathlib import Path
import build_eval_notebooks as B12
import build_rep_notebooks as BR

HERE = Path(__file__).resolve().parent
CELL = (HERE / "vendor_t3enc/t3enc_cell.py").read_text(encoding="utf-8")
SMOKE = '(LOCAL or os.environ.get("FLAG_SMOKE") == "1")'

VARIANTS = {
    "farl16": dict(nb="FLAG_45_facelora.ipynb", out="feats_facelora", arr="farlA", ds="flag2027-feats-facelora", num="45",
                   model="FaRL ep16 (openai CLIP ViT-B/16 + FaRL weights, as `farl16` of FLAG_40)", blocks="6-11 of 12",
                   lora="`c_fc` / `c_proj`", dim=512, train="~25 min", mav="~30 min"),
    "clipL": dict(nb="FLAG_45c_facelora_clipl.ipynb", out="feats_facelora_clipl", arr="clipA", ds="flag2027-feats-facelora-clipl",
                  num="45c", model="CLIP ViT-L/14 (`openai/clip-vit-large-patch14` @ 32bd642, transformers, as `clipL` of FLAG_40)",
                  blocks="18-23 of 24", lora="`fc1` / `fc2`", dim=768, train="~50 min", mav="~45 min"),
}

INTRO = """
# FLAG 2027 — NB-{num}: FACELORA-02, a LoRA adapter on the face encoder {enc} trained against frozen SimAM voices

**Why.** Every gain so far came from heads on FROZEN encoders. The voice side may not be fine-tuned on Bengali speech, but
the **face encoder may**. This notebook pulls the face embedding of {model} towards the SimAM voice space with a LoRA adapter
(rank 16 on the MLPs {lora} of blocks {blocks}), trained on Bengali-filtered VoxCeleb2 only, then embeds every MAV-Celeb row
and the VoxCeleb2 frames with it (`{arr}`, {dim}-d). Whether it helps is decided by the FACELORA-02 lab notebook (FLAG_59:
Track 3 recipe, arms B0 / FA / CA / FCA, same seeds), registered in `Experiment/FACELORA-02/README.md`.

**Registered (no selection by result):** P = 32 identities x K = 4 (face, voice) pairs, face from a video different from
the voice's, random horizontal flip; symmetric multi-positive InfoNCE, tau 0.05; AdamW (LoRA 1e-4, voice projection 1e-3,
wd 1e-2), 100 warm-up steps, cosine, **1,500 steps**, fp16, seed 0; gradient checkpointing on the adapted blocks (memory
only). 300 VoxCeleb2 identities are held out only to print a sanity EER (steps 0 / 500 / 1000 / 1500).
**Wiring check:** at step 0 (B = 0) the adapter IS the frozen encoder, so 64 VoxCeleb2 frames must reproduce FLAG_41's
frozen arrays (mean cosine >= 0.99; frames cached as lossless PNG), else the notebook stops before training.

| output | content |
|---|---|
| `face_{arr}_*.npy`, `ext_*_face_{arr}.npy` | adapted embedding of every MAV-Celeb row (FLAG_40 row order) |
| `vox_face_{arr}.npy` (+ `face_meta.csv`) | the same for the 29,251 VoxCeleb2 frames, FLAG_37 / FLAG_41 row order (NaN = frame not fetched) |
| `facelora_state.pt`, `train_log.json`, `vproj_W.npy`, `vproj_b.npy` | adapter, loss / sanity-EER curve, voice projection |
| `vox_frames_png.zip` | cached frames, lossless PNG (resume; the other FACELORA notebook can reuse them) |

**Rules:** face encoder fine-tuning is allowed; the voice encoders stay frozen; training data = VoxCeleb2 with the
Bengali-flagged speakers of FLAG_42 removed (**required**: the notebook stops without `lid_summary.json`); no MAV-Celeb
identity is used for training or compared with VoxCeleb2. Declare in the system description.

**Inputs (Add Input):** raw v4 (`train_set` + `dev_set`), MAV-Celeb v1/v2 + `flag2027-ext-meta` (as FLAG_40);
**`flag2027-lab-data`** (FLAG_41's VoxCeleb2 arrays, FLAG_37's key lists, FLAG_42's `lid_summary.json`; or the outputs of
FLAG_41 / FLAG_42 themselves); optional: this notebook's previous output (resume). **Settings:** GPU T4, **Internet ON**.

**Time:** about 3-4.5 h: VoxCeleb2 frames ~1.5 h (29k HTTP range reads; skipped when a `vox_frames_png.zip` is attached), training
{train}, MAV-Celeb faces {mav}, VoxCeleb2 frames ~10-20 min. Resumable (frames, adapter, finished arrays).

**Output:** `/kaggle/working/{out}/`.
"""

PIP = r'''# openai CLIP (commit d50d76d, FaRL's code base) --no-deps; av decodes the face tracks.
if not LOCAL:
    !pip -q install ftfy regex av 2>&1 | tail -1
    !pip -q install --no-deps "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1" 2>&1 | tail -1
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEV,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

TRAIN = r'''
# ============================================================================ FACELORA-02 adapter training (registered settings; no selection by result)
import math, tarfile, requests, struct, zlib, threading
import concurrent.futures as cf
from huggingface_hub import HfFileSystem
from torch.utils.checkpoint import checkpoint

SMOKE = (LOCAL or os.environ.get("FLAG_SMOKE") == "1")
CFG = dict(enc=LORA_ENC, rank=16, alpha=32, blocks={"farl16": list(range(6, 12)), "clipL": list(range(18, 24))}[LORA_ENC],
           P=int(os.environ.get("LORA_P", "32")), K=int(os.environ.get("LORA_K", "4")), steps=int(os.environ.get("LORA_STEPS", "1500")),
           tau=0.05, lr_lora=1e-4, lr_proj=1e-3, wd=1e-2, warmup=100, held_out=300, seed=0, flip=0.5, grad_ckpt=True)
SMOKE_IDS = int(os.environ.get("VOX_SMOKE_IDS", "0")) if SMOKE else 0
REF_FACE = {"farl16": "face_farl16.npy", "clipL": "face_clipL.npy"}   # FLAG_41 frozen arrays (wiring check)
ZIPS = [2, 3, 4, 5, 6]
ZIP_PATH = "datasets/ProgramComputer/voxceleb/vox2/vox2_mp4_{}.zip"
ZIP_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_{}.zip"
FS = HfFileSystem()
FRAMES = OUT / "vox_frames_png.zip"                  # lossless: CLIP-L moves to cos ~0.93 under JPEG q92
FILES = SCAN["files"]                                                           # the data cell's one walk over the inputs
for n in ("vox_frames_png.zip", "facelora_state.pt", "train_log.json"):          # resume: cached frames, adapter
    prev = [p for p in FILES.get(n, []) if p.parent.name == OUT.name]
    if prev and not (OUT / n).exists():
        shutil.copy(prev[0], OUT / n); log("resume:", n, "from", prev[0].parent)
if not FRAMES.exists() and not SMOKE:                                            # frames cached by the other FACELORA notebook
    fz = sorted(FILES.get("vox_frames_png.zip", []), key=lambda f: -f.stat().st_size)
    if fz:
        shutil.copy(fz[0], FRAMES); log("frames reused from", fz[0])

# ---- the VoxCeleb2 samples of FLAG_37 / 41: key lists + frozen SimAM voice embeddings + the frozen face arrays (wiring check)
# FLAG_41's arrays (its output, or the copy in flag2027-lab-data: lab/Experiment/T3ENC-01/feats_vox2t3) and FLAG_37's key
# lists (beside them in FLAG_41's output, or lab/Experiment/VOX2-01/feats_vox2; the same files)
vt3 = [p for p in FILES.get("voice_sam100p.npy", []) if "feats_facelora" not in p.as_posix()]
assert vt3, "attach flag2027-lab-data (or the output of FLAG_41): VoxCeleb2 T3 arrays + key lists"
V41 = vt3[0].parent
fmeta = [p for p in FILES.get("face_meta.csv", []) if (p.parent / "voice_meta.csv").exists() and "feats_facelora" not in p.as_posix()]
META = V41 if (V41 / "face_meta.csv").exists() else sorted(fmeta, key=lambda p: "feats_vox2" not in p.as_posix())[0].parent
for f in V41.glob("vox2_mp4_*_listing.csv"):                                     # FLAG_41's zip listings (if attached): no HF listing
    if not (OUT / f.name).exists():
        shutil.copy(f, OUT / f.name)
FK = pd.read_csv(META / "face_meta.csv", dtype={"spk": str, "video": str, "utt": str})
VK = pd.read_csv(META / "voice_meta.csv", dtype={"spk": str, "video": str, "utt": str})
log(f"VoxCeleb2 arrays from {V41}, key lists from {META}")
FK["row0"] = np.arange(len(FK))
l2n_ = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
VE = np.concatenate([l2n_(np.load(V41 / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
assert len(VE) == len(VK) and np.isfinite(VE).all() and len(np.load(V41 / REF_FACE[LORA_ENC], mmap_mode="r")) == len(FK)
lid = [p for p in FILES.get("lid_summary.json", []) if "vox2" in p.as_posix().lower() or "nb-42" in p.as_posix()] or FILES.get("lid_summary.json", [])
assert lid, "attach flag2027-lab-data (feats_vox2lid) or the output of FLAG_42: lid_summary.json, the Bengali filter is REQUIRED"
DROP = set(json.loads(lid[0].read_text())["speakers_dropped_ids"])
FILTERED = True
log(f"VoxCeleb2: {len(FK)} faces, {len(VK)} voices, {FK.spk.nunique()} ids | Bengali filter (FLAG_42): {len(DROP)} speakers dropped")
if SMOKE_IDS:
    keep = sorted((set(FK.spk) & set(VK.spk)) - DROP)[:SMOKE_IDS]
    fk_m, vk_m = FK.spk.isin(keep).values, VK.spk.isin(keep).values
    FK, VK, VE = FK[fk_m].reset_index(drop=True), VK[vk_m].reset_index(drop=True), VE[vk_m]
    CFG["held_out"] = 1
    log(f"smoke test: {len(keep)} VoxCeleb2 identities, {len(FK)} faces, {len(VK)} voices")


# ---- frames: the middle frame of every FLAG_37 face track, cached as lossless PNG in vox_frames_png.zip (resumable per chunk);
#      FLAG_41 embedded the decoded frame itself, so the step-0 wiring check compares the same pixels
def get_retry(url, tries=4, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=300, **kw); r.raise_for_status(); return r
        except Exception as e:
            log(f"  retry {k + 1}/{tries}: {type(e).__name__}: {e}"); time.sleep(10 * (k + 1))
    raise RuntimeError(f"cannot fetch {url}")


def zip_listing(k):
    f = OUT / f"vox2_mp4_{k}_listing.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"video": str, "utt": str})
    with FS.open(ZIP_PATH.format(k), "rb", block_size=2 ** 19) as fh:
        rows = [(i.filename, i.header_offset, i.compress_size, i.compress_type)
                for i in zipfile.ZipFile(fh).infolist() if i.filename.endswith(".mp4")]
    L_ = pd.DataFrame(rows, columns=["name", "off", "csize", "ctype"])
    sp = L_.name.str.split("/")
    L_["spk"], L_["video"], L_["utt"], L_["zip"] = sp.str[-3], sp.str[-2], sp.str[-1].str[:-4], k
    L_.to_csv(f, index=False)
    return L_


_TL, URLS = threading.local(), {}


def resolve(k):
    with requests.get(ZIP_URL.format(k), headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=120, stream=True) as r:
        return r.url


def middle_frame_png(mp4):
    import av
    with av.open(io.BytesIO(mp4)) as c:
        st = c.streams.video[0]; n = st.frames or 50
        last = None
        for k, fr in enumerate(c.decode(st)):
            last = fr
            if k >= n // 2:
                break
        if last is None:
            return None
        buf = io.BytesIO(); last.to_image().convert("RGB").save(buf, "PNG", compress_level=3); return buf.getvalue()


def fetch_jpg(job):
    k, off, csize, ctype = job
    if not hasattr(_TL, "s"):
        _TL.s = requests.Session()
    for attempt in range(4):
        try:
            r = _TL.s.get(URLS[k], headers={"Range": f"bytes={off}-{off + 30 + 1024 + csize}"}, timeout=120)
            if r.status_code in (401, 403):
                URLS[k] = resolve(k); continue
            b = r.content
            n, e = struct.unpack("<HH", b[26:30])
            d = b[30 + n + e:30 + n + e + csize]
            return middle_frame_png(zlib.decompress(d, -15) if ctype == 8 else d)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


have = set()
if FRAMES.exists():
    with zipfile.ZipFile(FRAMES) as z:
        have = {int(n[:-4]) for n in z.namelist() if n.endswith(".png")}
todo = [int(r) for r in FK.row0 if int(r) not in have]                          # frame names = FLAG_37 row numbers
if todo:
    LST = pd.concat([zip_listing(k) for k in ZIPS], ignore_index=True)
    for k in ZIPS:
        URLS[k] = resolve(k)
    SEL = FK.set_index("row0").loc[todo, ["spk", "video", "utt"]].reset_index().merge(
        LST[["spk", "video", "utt", "zip", "off", "csize", "ctype"]], how="left", on=["spk", "video", "utt"])
    assert SEL.zip.notna().all(), f"{SEL.zip.isna().sum()} face keys not in the zip listings"
    CH = 2400
    log(f"frames: {len(SEL)} to fetch ({len(have)} cached), {len(SEL) // CH + 1} chunks")
    for c0 in range(0, len(SEL), CH):
        t0 = time.time(); S_ = SEL.iloc[c0:c0 + CH]
        with cf.ThreadPoolExecutor(16) as ex:
            jpgs = list(ex.map(fetch_jpg, zip(S_.zip.astype(int), S_.off.astype(int), S_.csize.astype(int), S_.ctype.astype(int))))
        with zipfile.ZipFile(FRAMES, "a", zipfile.ZIP_STORED) as z:
            for row, b in zip(S_.row0, jpgs):
                if b is not None:
                    z.writestr(f"{row}.png", b)
        n_ok = sum(b is not None for b in jpgs)
        log(f"  chunk {c0 // CH}: {n_ok}/{len(S_)} frames [{time.time() - t0:.0f}s]")
wanted = set(int(r) for r in FK.row0)
with zipfile.ZipFile(FRAMES) as z:
    JPG0 = {int(n[:-4]): z.read(n) for n in z.namelist() if n.endswith(".png") and int(n[:-4]) in wanted}
JPG = {i: JPG0[int(r)] for i, r in enumerate(FK.row0) if int(r) in JPG0}          # by row of FK (smoke: re-indexed)
log(f"frames in memory: {len(JPG)}/{len(FK)} ({sum(len(b) for b in JPG.values()) / 1e6:.0f} MB)")

# ---- the face encoder + LoRA, voice projection
if LORA_ENC == "farl16":
    import clip as _clip
    _base, PRE = _clip.load("ViT-B/16", device="cpu", download_root=str(CACHE / "clip"))
    _sd = torch.load(fetch(FARL16_URL, CACHE / "FaRL-ep16.pth", FARL16_SIZE), map_location="cpu", weights_only=False)["state_dict"]
    _vis = [k for k in _base.state_dict() if k.startswith("visual.")]
    assert all(k in _sd and _sd[k].shape == _base.state_dict()[k].shape for k in _vis)
    _base.load_state_dict(_sd, strict=False)
    VIS = _base.visual.float()
    LAYERS, MLP_NAMES = VIS.transformer.resblocks, ("c_fc", "c_proj")
    ENC_FWD = lambda x: VIS(x)
    DIM = int(VIS.proj.shape[1])
else:
    from transformers import CLIPModel, CLIPImageProcessor
    _proc = CLIPImageProcessor.from_pretrained(CLIPL_ID, revision=CLIPL_REV)
    _cm = CLIPModel.from_pretrained(CLIPL_ID, revision=CLIPL_REV, torch_dtype=torch.float32)
    VIS = nn.ModuleDict(dict(vm=_cm.vision_model, proj=_cm.visual_projection)).float()
    del _cm
    PRE = lambda im: _proc(images=im.convert("RGB"), return_tensors="pt").pixel_values[0]
    LAYERS, MLP_NAMES = VIS["vm"].encoder.layers, ("fc1", "fc2")
    ENC_FWD = lambda x: VIS["proj"](VIS["vm"](pixel_values=x).pooler_output)     # = get_image_features
    DIM = int(VIS["proj"].out_features)
assert DIM == {"farl16": 512, "clipL": 768}[LORA_ENC]
REF = REF_FACE[LORA_ENC]
for p in VIS.parameters():
    p.requires_grad = False


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(x, self.B @ self.A) * self.scale


class Ckpt(nn.Module):
    """Gradient checkpointing of one adapted block during training (memory only; the same function)."""
    def __init__(self, blk):
        super().__init__(); self.blk = blk

    def forward(self, *a, **k):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.blk, *a, use_reentrant=False, **k)
        return self.blk(*a, **k)


def add_lora(cfg):
    names = []
    for i in cfg["blocks"]:
        mlp = LAYERS[i].mlp
        for n in MLP_NAMES:
            setattr(mlp, n, LoRALinear(getattr(mlp, n), cfg["rank"], cfg["alpha"])); names.append(f"layer{i}.mlp.{n}")
    if cfg["grad_ckpt"]:
        for i in cfg["blocks"]:
            LAYERS[i] = Ckpt(LAYERS[i])
    return names


LORA_NAMES = add_lora(CFG)
VIS = VIS.to(DEV)
VPROJ = nn.Linear(VE.shape[1], DIM).to(DEV)
with torch.no_grad():
    VPROJ.bias.zero_()
lora_params = [p for n, p in VIS.named_parameters() if p.requires_grad]
log(f"{LORA_ENC} LoRA: {len(LORA_NAMES)} layers, {sum(p.numel() for p in lora_params) / 1e6:.2f} M trainable "
    f"(+ vproj {sum(p.numel() for p in VPROJ.parameters()) / 1e6:.2f} M), output {DIM}")

# ---- identities, held-out, batch sampler
from PIL import Image
rng = np.random.RandomState(CFG["seed"]); torch.manual_seed(CFG["seed"])
fv = FK.groupby("spk").indices; vv = VK.groupby("spk").indices
ids = sorted(s for s in set(fv) & set(vv) if s not in DROP and any(i in JPG for i in fv[s]))
held = sorted(rng.choice(ids, min(CFG["held_out"], max(1, len(ids) // 10)), replace=False)) if len(ids) > 2 else []
train_ids = [s for s in ids if s not in set(held)]
fvid, vvid = FK.video.values, VK.video.values
GEN = dict(zip(FK.spk, FK.gender)) if "gender" in FK else {}
log(f"identities: {len(ids)} usable ({len(DROP)} dropped) -> train {len(train_ids)}, held out {len(held)}")


def prep_jpg(i, flip):
    im = Image.open(io.BytesIO(JPG[i])).convert("RGB")
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return PRE(im)


def sample_batch():
    P, K = CFG["P"], CFG["K"]
    spk = rng.choice(train_ids, min(P, len(train_ids)), replace=False)
    fi, vi, lab = [], [], []
    for j, s in enumerate(spk):
        faces = [i for i in fv[s] if i in JPG]
        for _ in range(K):
            v = rng.choice(vv[s])
            cand = [i for i in faces if fvid[i] != vvid[v]] or faces
            fi.append(int(rng.choice(cand))); vi.append(int(v)); lab.append(j)
    flips = rng.rand(len(fi)) < CFG["flip"]
    with cf.ThreadPoolExecutor(8) as ex:
        X = torch.stack(list(ex.map(prep_jpg, fi, flips)))
    return X, torch.from_numpy(VE[vi]), torch.tensor(lab)


def multi_pos_infonce(S, lab):
    """S: logits (n x n), lab: identity of every row; positives = same identity (K per row); symmetric over rows and columns."""
    same = (lab[:, None] == lab[None, :]).float()
    def one(L):
        logp = L - torch.logsumexp(L, 1, keepdim=True)
        return -((logp * same).sum(1) / same.sum(1)).mean()
    return 0.5 * (one(S) + one(S.t()))


@torch.no_grad()
def embed_faces(rows, bs=128):
    VIS.eval(); out = []
    for s in range(0, len(rows), bs):
        X = torch.stack([prep_jpg(i, False) for i in rows[s:s + bs]]).to(DEV)
        with torch.autocast("cuda", enabled=DEV.type == "cuda"):
            out.append(F.normalize(ENC_FWD(X).float(), dim=1).cpu())
    return torch.cat(out).numpy()


def sanity_eer():
    """Held-out identities: face of one video vs voice of ANOTHER video; negatives = other held-out identities (all / same gender)."""
    from sklearn.metrics import roc_curve
    if not held:
        return {}
    rows = [i for s in held for i in fv[s] if i in JPG]
    Fh = dict(zip(rows, embed_faces(rows)))
    VPROJ.eval()
    with torch.no_grad():
        Vh = F.normalize(VPROJ(torch.from_numpy(VE).to(DEV)).float(), dim=1).cpu().numpy()
    r2 = np.random.RandomState(1); pos, neg, negg = [], [], []
    for s in held:
        fr = [i for i in fv[s] if i in JPG]
        for v in vv[s][:4]:
            cand = [i for i in fr if fvid[i] != vvid[v]]
            if cand:
                pos.append(float(Fh[r2.choice(cand)] @ Vh[v]))
            o = held[r2.randint(len(held))]
            if o != s:
                fo = [i for i in fv[o] if i in JPG]
                sc = float(Fh[r2.choice(fo)] @ Vh[v]); neg.append(sc)
                if GEN.get(o) == GEN.get(s):
                    negg.append(sc)
    def eer(p, n):
        if len(p) < 2 or len(n) < 2:
            return None
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]; s_ = np.r_[p, n]
        fpr, tpr, _ = roc_curve(y, s_); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return round(100 * float(fpr[i] + fnr[i]) / 2, 2)
    return {"ng": eer(pos, neg), "g": eer(pos, negg) if len(negg) > 20 else None, "pos": len(pos), "neg": len(neg)}


# ---- train (or load)
STATE = OUT / "facelora_state.pt"
if STATE.exists():
    st = torch.load(STATE, map_location="cpu", weights_only=False)
    assert st["cfg"]["enc"] == LORA_ENC, st["cfg"]
    VIS.load_state_dict({**VIS.state_dict(), **{k: v for k, v in st["vis"].items()}}); VPROJ.load_state_dict(st["vproj"])
    log("adapter loaded from the previous output:", st["cfg"])
    WIRING = st.get("wiring")
else:
    # wiring check: at step 0 (B = 0) the adapter is the frozen encoder -> FLAG_41's frozen arrays on the same frames
    chk = [i for i in range(len(FK)) if i in JPG][:64]
    ref = l2n_(np.load(V41 / REF, mmap_mode="r")[FK.row0.values[chk]].astype(np.float32))
    cos = (embed_faces(chk) * ref).sum(1)
    WIRING = dict(rows=len(chk), mean_cos=round(float(cos.mean()), 5), min_cos=round(float(cos.min()), 5))
    log(f"wiring check vs FLAG_41 {REF} at step 0: {WIRING}")
    assert WIRING["mean_cos"] >= 0.99, f"the adapter at step 0 does not reproduce the frozen {LORA_ENC} arrays: {WIRING}"
    opt = torch.optim.AdamW([{"params": lora_params, "lr": CFG["lr_lora"]}, {"params": VPROJ.parameters(), "lr": CFG["lr_proj"]}], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CFG["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, CFG["steps"])))))
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    LOG = {"cfg": CFG, "filtered": FILTERED, "wiring": WIRING, "loss": [], "sanity": {}}
    LOG["sanity"]["0"] = sanity_eer(); log("sanity EER before training:", LOG["sanity"]["0"])
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        VIS.train(); VPROJ.train()
        X, V_, lab = sample_batch()
        X, V_, lab = X.to(DEV), V_.to(DEV), lab.to(DEV)
        with torch.autocast("cuda", enabled=DEV.type == "cuda"):
            f = F.normalize(ENC_FWD(X).float(), dim=1); v = F.normalize(VPROJ(V_).float(), dim=1)
            loss = multi_pos_infonce(f @ v.t() / CFG["tau"], lab)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        LOG["loss"].append(float(loss))
        if step % 50 == 0 or step == CFG["steps"]:
            log(f"step {step}/{CFG['steps']} loss {np.mean(LOG['loss'][-50:]):.4f} [{time.time() - t0:.0f}s]"
                + (f" | GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if DEV.type == "cuda" else ""))
        if step in (500, 1000, CFG["steps"]):
            LOG["sanity"][str(step)] = sanity_eer(); log(f"sanity EER at step {step}:", LOG["sanity"][str(step)])
    torch.save({"vis": {k: v.cpu() for k, v in VIS.state_dict().items() if k.endswith((".A", ".B"))}, "vproj": VPROJ.state_dict(),
                "cfg": CFG, "lora_names": LORA_NAMES, "filtered": FILTERED, "dropped": sorted(DROP), "wiring": WIRING}, STATE)
    (OUT / "train_log.json").write_text(json.dumps(LOG, indent=1))
    log("adapter saved ->", STATE.name)
VIS.eval(); VPROJ.eval()
np.save(OUT / "vproj_W.npy", VPROJ.weight.detach().cpu().numpy()); np.save(OUT / "vproj_b.npy", VPROJ.bias.detach().cpu().numpy())
'''

ENC = r'''
# ============================================================================ the adapted encoder, in the FLAG_12 encoder interface
class Adapted:
    kind = "face"

    def __init__(self):
        VIS.eval()

    def prep(self, im):
        return PRE(im)

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        with torch.autocast("cuda", enabled=use_amp and DEV.type == "cuda"):
            return {"": ENC_FWD(x.to(DEV)).float().cpu().numpy()}


NEW = {ARR + "V": Adapted}


def build(name):
    return NEW[name]()


def release(keep_ib=False):
    import gc
    gc.collect()


ARRAYS_NEW = {ARR + "V": ("face", {"": ARR})}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {}
'''

VOX = r'''
# ============================================================================ the adapted embedding of the VoxCeleb2 frames (FLAG_37 / 41 row order) + manifest extras
rows = [i for i in range(len(FK)) if i in JPG]
A = np.full((len(FK), DIM), np.nan, np.float16)
if rows:
    A[rows] = embed_faces(rows).astype(np.float16)
np.save(OUT / f"vox_face_{ARR}.npy", A); FK.drop(columns="row0").to_csv(OUT / "face_meta.csv", index=False)
TL = json.loads((OUT / "train_log.json").read_text()) if (OUT / "train_log.json").exists() else None
CHECKS["facelora"] = {"enc": LORA_ENC, "array": ARR, "filtered": FILTERED, "dropped_speakers": len(DROP), "train_ids": len(train_ids),
                      "held_out": len(held), "vox_frames_embedded": len(rows), "vox_rows": len(FK), "cfg": CFG, "wiring": WIRING,
                      "train_log": dict(TL, loss=TL["loss"][::50]) if TL else None}
log(f"VoxCeleb2 frames embedded: {len(rows)}/{len(FK)} | sanity EER:", TL["sanity"] if TL else "n/a")
'''


def build(enc):
    v = VARIANTS[enc]
    env = BR.env(v["out"], [v["arr"] + "V"], f"the adapted {enc} face encoder (trained below)")
    env = env.replace("if LOCAL and os.environ.get(", f"if {SMOKE} and os.environ.get(")
    assert env.count(SMOKE) == 2, "smoke switches of the env cell"
    env += f'LORA_ENC, ARR = "{enc}", "{v["arr"]}"   # FACELORA-02 variant of this notebook\n'
    data = BR.data(v["out"])
    assert data.count("if LOCAL and SMOKE_ROWS:") == 1
    data = data.replace("if LOCAL and SMOKE_ROWS:", f"if {SMOKE} and SMOKE_ROWS:")
    intro = INTRO.format(enc=enc, **v)
    cells = [("markdown", intro), ("code", env), ("code", PIP),
             ("markdown", "## 1. Find the data (FLAG_12 cell): organiser lists with media, v1/v2 rows, resume"), ("code", data),
             ("markdown", "## 2. Audio loading, batching, helpers (FLAG_12 cell)"), ("code", B12.AUDIO12),
             ("markdown", "## 3. Track 3 encoder helpers (FaRL ep16 download, CLIP-L id; same cell as FLAG_40)"), ("code", CELL),
             ("markdown", "## 4. FACELORA-02: frames, wiring check, LoRA, training (registered settings)"), ("code", TRAIN),
             ("markdown", "## 5. The adapted encoder"), ("code", ENC),
             ("markdown", f"## 6. `{v['arr']}` of every MAV-Celeb row (FLAG_12 loop, resumable)"), ("code", BR.run(())),
             ("markdown", f"## 7. `{v['arr']}` of the VoxCeleb2 frames"), ("code", VOX),
             ("markdown", "## 8. Manifest"), ("code", BR.done(v["ds"]))]
    B12.nb(cells, v["nb"])
    print(v["nb"], "built")


if __name__ == "__main__":
    for e in (sys.argv[1:] or list(VARIANTS)):
        build(e)
