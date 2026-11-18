"""Generate FLAG_37_vox2_feats.ipynb (VOX2-01): features of ~4,900 VoxCeleb2 identities (voices and face tracks of
VoxCeleb2 dev) in OUR feature spaces, so the local experiments can train the bridge / adapters
on thousands of identities (docs/PLAN_EMBED_2026-10-05.md, VOX2-01 a-d).

Sources (Hugging Face, streamed; only one tar / a few hundred face tracks are in memory at a time):
  voices  gaunernst/voxceleb2-dev-wds   779 WebDataset tars (~105 MB), original AAC, 1,092,009 clips, 5,994 speakers
  faces   ProgramComputer/voxceleb vox2/vox2_mp4_{2..6}.zip   VoxCeleb2 face tracks (224 x 224, the VoxCeleb framing, close
          to MAV-Celeb's); single tracks read by HTTP range requests; 4,877 speakers (vox2_mp4_1.zip does not open)
  map     ProgramComputer/voxceleb vox2/vox2_meta.csv   gender
VGGFace2 crops (chronopt-research/cropped-vggface2-224) were tried first and dropped: they are tight face crops, while
MAV-Celeb faces are loose head-and-shoulders frames.
Encoders = exactly those of our v4 / v1 / v2 features (full code in the cells):
  faces   ImageBind vision (ibv), VGGFace fc7 (vgg; the organiser's 4096-d face feature), ViT age (vitageprob), on the
          middle frame of a face track
  voices  ImageBind audio (iba), organiser ECAPA yangwang825/ecapa-tdnn-vox2 (btc), speechbrain ECAPA (ecapa192),
          audeering wav2vec2 age / gender (w2v2agage, w2v2aggen)
    python build_vox2_notebook.py
    FLAG_LOCAL=1 FLAG_OUT=<dir> FLAG_CACHE=<cache with imagebind_huge.pth> VOX_SMOKE_IDS=4 VOX_NF=2 VOX_NV=2 \
        python run_notebook_local.py FLAG_37_vox2_feats.ipynb --jupyter
"""
from pathlib import Path
import build_eval_notebooks as B12

HERE = Path(__file__).resolve().parent
FX = (HERE / "flag_extract.py").read_text(encoding="utf-8")
ORG_VOICE = FX[FX.index("class OrganiserVoice:"):FX.index("class VGGFace:")]
VGG_FACE = FX[FX.index("class VGGFace:"):FX.index("def _parse(member):")]

INTRO = """
# FLAG 2027 — NB-37: VOX2-01, features of ~4,900 VoxCeleb2 identities in our feature spaces

**Why.** Every face-voice model we trained learned from at most 221 identities. The FAME 2026 winners trained on about
6,000 (VoxCeleb2, or VoxBlink), and the best rule-abiding team on our board beats us exactly where identity beyond
gender and age matters (the same-gender cells). This notebook computes, for ~4,900 VoxCeleb2 identities, the same
features our system uses, so the local experiments (bridge on 221 -> 1k -> 6k identities, ImageBind adapters with
nested selection, `relg` with thousands of anchors, hard same-gender age-matched negatives) can run on CPU.

| side | source | per identity | features |
|---|---|---|---|
| voice | VoxCeleb2 dev, WebDataset copy `gaunernst/voxceleb2-dev-wds` (original AAC; CC BY-SA 4.0 card, VoxCeleb terms apply) | up to `NV` = 8 clips, at most 2 per YouTube video, 12 s centre crop | `iba`, `btc`, `ecapa192`, `w2v2agage`, `w2v2aggen` |
| face | VoxCeleb2 dev face tracks, `ProgramComputer/voxceleb` `vox2_mp4_{2..6}.zip` (single tracks read by HTTP range requests) | the middle frame of the first track of up to `NF` = 6 different videos | `ibv`, `vgg`, `vitageprob` |
| map | `vox2_meta.csv` | gender | — |

Faces and voices keep their YouTube video id, so the experiments can pair a face and a voice of DIFFERENT videos (no
session shortcut), as in the test. VoxCeleb face tracks are framed like MAV-Celeb (loose head crops); VGGFace2 crops
were dropped because they are tight. Coverage: 4,877 of the 5,994 speakers (`vox2_mp4_1.zip` does not open remotely).

**Rules (read before using the output).**
- Encoders are frozen; nothing is derived from the challenge test data.
- **No linking of challenge people:** we do NOT compare names or biometric features of MAV-Celeb people with these
  corpora (FLAG website: "don't link them to other data sources"). Whether external identities that may overlap must
  be removed, and how, is asked of the organisers (docs/email_organizers.md, follow-up questions 5-6).
- **Bengali:** VoxCeleb2 may contain some Bengali speech. Encoders pretrained on VoxCeleb2 are explicitly allowed;
  training a small head on their frozen embeddings of these clips is question 6 to the organisers. Every clip keeps its
  key (speaker / video / utterance), so a language filter can be applied later without re-extraction.
- Declare VoxCeleb2 (audio and face tracks) and the Hugging Face copies in the system description.
- Caveat for the experiments: both ECAPA models were trained on VoxCeleb2, so these speakers are their TRAINING
  speakers (cleaner voice embeddings than on new people).

**Settings:** GPU T4, **Internet ON**. No input dataset is needed; attach ONLY this notebook's previous output to resume
(finished face chunks / tars are skipped).

**Time:** about 3-4 h: faces ~1 h (~29k face tracks, ~6 GB of range reads with 16 threads, middle frame decoded);
voices ~2-3 h (~150-250 tars streamed, ~39k clips decoded and embedded by four models). Resumable per chunk / tar.

**Output:** `/kaggle/working/feats_vox2/` -> dataset **`flag2027-feats-vox2`** (about 1 GB, float16).
"""

ENV = B12.ENV12.replace('OUT = WORKDIR / "feats_eval"', 'OUT = WORKDIR / "feats_vox2"')
assert ENV != B12.ENV12

PIP = r'''
# speechbrain -U: Kaggle's preinstalled one breaks on recent torchaudio. ImageBind --no-deps (its pins would replace
# Kaggle's torch); its data loaders are reimplemented in the encoder cell. av = PyAV, decodes the AAC (.m4a) clips.
if not LOCAL:
    !pip -q install -U "speechbrain>=1.1.1" iopath ftfy omegaconf fvcore av 2>&1 | tail -2
    !pip -q install --no-deps "git+https://github.com/facebookresearch/ImageBind" 2>&1 | tail -2
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEV,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

ORG = r'''
# ============================================================================ organiser-space encoders (= kaggle/flag_extract.py)
# The organiser's features, reproduced exactly on v4 train (cos 1.0000): voice = speechbrain yangwang825/ecapa-tdnn-vox2
# ("btc"), face = VGGFace (Parkhi 2015) fc7 after ReLU, 224 x 224, BGR minus the VGGFace means ("vgg").
_speechbrain_import = speechbrain_import

''' + ORG_VOICE + VGG_FACE

CONFIG = r'''
# ============================================================================ VOX2-01 settings and the identity list
import tarfile, requests, struct, zlib, threading
import concurrent.futures as cf
from huggingface_hub import HfFileSystem

NF = int(os.environ.get("VOX_NF", "6"))                 # face frames per identity, each from a DIFFERENT video
NV = int(os.environ.get("VOX_NV", "8"))                 # voice clips per identity
PER_VIDEO = 2                                            # at most 2 voice clips per YouTube video: more videos per person
MAX_TARS = int(os.environ.get("VOX_MAX_TARS", "260"))   # of 779 (~105 MB each), streamed, never stored
CLIP_SEC = 12.0                                          # centre crop, as the v1/v2 rows
THREADS = 16                                             # parallel HTTP range reads of face tracks
SEED = 0
SMOKE_IDS = int(os.environ.get("VOX_SMOKE_IDS", "0")) if LOCAL else 0
ZIPS = [2, 3, 4, 5, 6]                                   # vox2_mp4_1.zip does not open remotely (BadZipFile): ~1.1k speakers lost
ZIP_PATH = "datasets/ProgramComputer/voxceleb/vox2/vox2_mp4_{}.zip"
ZIP_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_{}.zip"
VOICE_URL = "https://huggingface.co/datasets/gaunernst/voxceleb2-dev-wds/resolve/main/voxceleb2-dev-{:04d}.tar"
META_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_meta.csv"
FS = HfFileSystem()
FACE_PARTS, VOICE_PARTS = OUT / "parts_face", OUT / "parts_voice"
for d in (FACE_PARTS, VOICE_PARTS):
    d.mkdir(parents=True, exist_ok=True)
for prev in IN.rglob("feats_vox2") if IN.exists() and not LOCAL else []:          # resume: earlier output attached
    for sub in ("parts_face", "parts_voice"):
        for f in (prev / sub).glob("*.npz"):
            if not (OUT / sub / f.name).exists():
                shutil.copy(f, OUT / sub / f.name)
    for f in prev.glob("vox2_mp4_*_listing.csv"):
        if not (OUT / f.name).exists():
            shutil.copy(f, OUT / f.name)


def get_retry(url, tries=4, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=300, **kw); r.raise_for_status(); return r
        except Exception as e:
            log(f"  retry {k + 1}/{tries} {url.rsplit('/', 1)[-1]}: {type(e).__name__}: {e}"); time.sleep(10 * (k + 1))
    raise RuntimeError(f"cannot fetch {url}")


def zip_listing(k):
    """Central directory of vox2_mp4_<k>.zip (remote, ~8 s): one row per face track (dev/mp4/<spk>/<video>/<utt>.mp4)."""
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


meta_f = CACHE / "vox2_meta.csv"
if not meta_f.exists():
    meta_f.write_bytes(get_retry(META_URL).content)
M = pd.read_csv(meta_f, dtype=str)
M.columns = [c.strip() for c in M.columns]
M = M.apply(lambda c: c.str.strip())
M = M[M.Set == "dev"].reset_index(drop=True)            # VoxCeleb2 dev = the speakers of the WebDataset voice copy
LST = pd.concat([zip_listing(k) for k in ZIPS], ignore_index=True)
M = M[M["VoxCeleb2 ID"].isin(set(LST.spk))].reset_index(drop=True)
if SMOKE_IDS:                                            # local smoke test: speakers of the first clips of tar 0
    seen = []
    with get_retry(VOICE_URL.format(0), stream=True) as r:
        r.raw.decode_content = True
        for m in tarfile.open(fileobj=r.raw, mode="r|"):
            s_ = m.name.split("/")[0]
            if s_ in set(M["VoxCeleb2 ID"]) and s_ not in seen:
                seen.append(s_)
            if len(seen) >= SMOKE_IDS:
                break
    M = M[M["VoxCeleb2 ID"].isin(seen)].reset_index(drop=True)
GENDER = dict(zip(M["VoxCeleb2 ID"], M["Gender"]))
if LOCAL and os.environ.get("VOX_SMOKE_STUB_IB"):        # local smoke test on a small-RAM PC only: ImageBind (the
    class IBVision:                                       # unchanged FLAG_12 cell, run on Kaggle in FLAG_12 / 33 / 35)
        prep = lambda self, im: torch.zeros(3, 4, 4)      # is replaced by random vectors; never used on Kaggle
        __call__ = lambda self, x, use_amp=True: {"": np.random.rand(len(x), 1024).astype(np.float32)}

    class IBAudio:
        __call__ = lambda self, w, use_amp=True: {"": np.random.rand(1024).astype(np.float32)}
    log("smoke test: ImageBind stubbed")
log(f"identities (VoxCeleb2 dev with face tracks in zips {ZIPS}): {len(M)} ({(M.Gender == 'm').sum()} m / "
    f"{(M.Gender == 'f').sum()} f) | NF {NF}, NV {NV}, <= {PER_VIDEO} voice clips per video, <= {MAX_TARS} tars")
'''

FACES = r'''
# ============================================================================ faces: one middle frame per face track, NF different videos
_TL, URLS = threading.local(), {}


def resolve(k):
    """Final (signed CDN) URL of a zip, for HTTP range requests; re-resolved when it expires."""
    with requests.get(ZIP_URL.format(k), headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=120, stream=True) as r:
        return r.url


def middle_frame(mp4):
    import av
    with av.open(io.BytesIO(mp4)) as c:
        st = c.streams.video[0]; n = st.frames or 50
        last = None
        for k, fr in enumerate(c.decode(st)):
            last = fr
            if k >= n // 2:
                break
        return last.to_image().convert("RGB") if last is not None else None


def fetch_frame(job):
    """(zip, local-header offset, compressed size, method) -> PIL middle frame, or None after 4 failed tries."""
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
            return middle_frame(zlib.decompress(d, -15) if ctype == 8 else d)
        except Exception:
            time.sleep(2 * (attempt + 1))
    BROKEN["image"].append(f"vox2_mp4_{k} @ {off}")
    return None


def pick(n, k):
    """k indices evenly spread over n items (all if n <= k)."""
    return list(range(n)) if n <= k else [int(round(x)) for x in np.linspace(0, n - 1, k)]


for k in ZIPS:
    URLS[k] = resolve(k)
L2 = LST[LST.spk.isin(set(M["VoxCeleb2 ID"]))].sort_values(["spk", "video", "utt"])
sel = []                                                 # per speaker: the first track of NF videos spread over its videos
for spk, g in L2.groupby("spk", sort=True):
    first = g.drop_duplicates("video")
    sel += [first.iloc[i] for i in pick(len(first), NF)]
SEL = pd.DataFrame(sel).reset_index(drop=True)
spks = sorted(SEL.spk.unique()); CH = 400                # chunks of 400 speakers, saved one by one (resume)
log(f"faces: {len(SEL)} tracks of {len(spks)} speakers, {len(spks) // CH + 1} chunks, {THREADS} threads")
FACE_ENC = None
for c0 in range(0, len(spks), CH):
    part = FACE_PARTS / f"faces_{c0 // CH:03d}.npz"
    if part.exists():
        continue
    if FACE_ENC is None:
        FACE_ENC = {"ibv": IBVision(), "vgg": VGGFace(CACHE / "vggface"), "vitage": ViTAge()}
    t0 = time.time(); nb0 = len(BROKEN["image"])
    S_ = SEL[SEL.spk.isin(set(spks[c0:c0 + CH]))].reset_index(drop=True)
    with cf.ThreadPoolExecutor(THREADS) as ex:
        imgs = list(ex.map(fetch_frame, zip(S_.zip, S_.off, S_.csize, S_.ctype)))
    ok = [i for i, im in enumerate(imgs) if im is not None]
    S_, imgs = S_.iloc[ok].reset_index(drop=True), [imgs[i] for i in ok]
    R = {"ibv": [], "vgg": [], "vitageprob": []}
    for s in range(0, len(imgs), 64):
        b = imgs[s:s + 64]
        R["ibv"].append(FACE_ENC["ibv"](torch.stack([FACE_ENC["ibv"].prep(im) for im in b]))[""])
        R["vgg"].append(FACE_ENC["vgg"]([FACE_ENC["vgg"].prep(im) for im in b]))
        R["vitageprob"].append(FACE_ENC["vitage"](torch.stack([FACE_ENC["vitage"].prep(im) for im in b]))["prob"])
    np.savez(part, spk=S_.spk.to_numpy(dtype=str), video=S_.video.astype(str).to_numpy(dtype=str), utt=S_.utt.astype(str).to_numpy(dtype=str),
             **{n: np.concatenate(v).astype(np.float16) for n, v in R.items() if v})
    log(f"  chunk {c0 // CH}: {len(imgs)} faces of {S_.spk.nunique()} speakers, {len(BROKEN['image']) - nb0} failed "
        f"[{time.time() - t0:.0f}s]")
FACE_ENC = None
release(keep_ib=True)                                     # ImageBind stays loaded for the audio branch
'''

VOICES = r'''
# ============================================================================ voices: stream the tars, keep up to NV clips per speaker
def decode_m4a(b):
    """AAC bytes -> mono float32 at 16 kHz (PyAV); VoxCeleb2 is already 16 kHz, so no rate change in practice."""
    import av
    out = []
    with av.open(io.BytesIO(b)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        for fr in c.decode(c.streams.audio[0]):
            out += [o.to_ndarray().reshape(-1) for o in res.resample(fr)]
        out += [o.to_ndarray().reshape(-1) for o in res.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(TARGET_SR // 10, np.float32)


need = set(M["VoxCeleb2 ID"])
cnt, pervid = {s: 0 for s in need}, {}
for p in sorted(VOICE_PARTS.glob("tar*.npz")):            # resume: counts of the tars already done
    Zp = np.load(p)
    for s, v in zip(Zp["spk"], Zp["video"]):
        cnt[s] = cnt.get(s, 0) + 1; pervid[(s, v)] = pervid.get((s, v), 0) + 1
order = [0] if SMOKE_IDS else list(np.random.RandomState(SEED).permutation(779))
VOICE_ENC = None
L = int(CLIP_SEC * TARGET_SR)
for t in order[:MAX_TARS]:
    part = VOICE_PARTS / f"tar{t:04d}.npz"
    if part.exists():
        continue
    if all(cnt[s] >= NV for s in need):
        log("every speaker has NV clips"); break
    t0 = time.time(); c2, pv2 = dict(cnt), dict(pervid); waves, keys = [], []
    try:
        with get_retry(VOICE_URL.format(t), stream=True) as r:
            r.raw.decode_content = True
            tf = tarfile.open(fileobj=r.raw, mode="r|")
            for m in tf:
                if not m.name.endswith(".m4a"):
                    continue
                spk, vid, utt = m.name[:-4].split("/")[-3:]
                if spk not in need or c2[spk] >= NV or pv2.get((spk, vid), 0) >= PER_VIDEO:
                    continue
                try:
                    w = decode_m4a(tf.extractfile(m).read())
                except Exception as e:
                    BROKEN["audio"].append(f"{m.name} | {type(e).__name__}: {e}"); continue
                if len(w) < int(MIN_SEC * TARGET_SR):
                    continue
                waves.append(centre_crop(w, L)); keys.append((spk, vid, utt, len(w) / TARGET_SR))
                c2[spk] += 1; pv2[(spk, vid)] = pv2.get((spk, vid), 0) + 1
    except Exception as e:                                # a broken stream: skip this tar, counts untouched
        log(f"!! tar {t}: {type(e).__name__}: {e} -> skipped"); continue
    if not waves:
        np.savez(part, spk=np.array([], str), video=np.array([], str)); continue
    if VOICE_ENC is None:
        VOICE_ENC = {"iba": IBAudio(), "btc": OrganiserVoice(CACHE / "ecapa_btc"), "ecapa": Ecapa(), "w2v2": W2V2AgeGender()}
    R = {"iba": np.stack([VOICE_ENC["iba"](w)[""] for w in waves])}
    od = np.argsort([len(w) for w in waves])              # length-sorted batches, as the organiser's script
    btc = [None] * len(waves)
    for s in range(0, len(od), 32):
        for i, e in zip(od[s:s + 32], VOICE_ENC["btc"]([waves[i] for i in od[s:s + 32]])):
            btc[i] = e
    R["btc"] = np.stack(btc)
    R["ecapa192"] = VOICE_ENC["ecapa"].embed(lambda i: waves[i], len(waves), np.array([len(w) / TARGET_SR for w in waves]),
                                             label=f"tar{t}")["192"]
    ag = [VOICE_ENC["w2v2"](w) for w in waves]
    R["w2v2agage"] = np.stack([a["age"] for a in ag]); R["w2v2aggen"] = np.stack([a["gen"] for a in ag])
    k = np.array(keys, dtype=object)
    np.savez(part, spk=k[:, 0].astype(str), video=k[:, 1].astype(str), utt=k[:, 2].astype(str), dur=k[:, 3].astype(np.float32),
             **{n: v.astype(np.float16) for n, v in R.items()})
    cnt, pervid = c2, pv2
    full = sum(cnt[s] >= NV for s in need)
    log(f"tar {t:04d}: {len(waves)} clips [{time.time() - t0:.0f}s] | speakers with NV clips {full}/{len(need)}, "
        f"with >= 1: {sum(cnt[s] > 0 for s in need)}")
VOICE_ENC = None
release()
'''

ASSEMBLE = r'''
# ============================================================================ assemble, zero-shot probe, manifest
def gather(parts, keys):
    Zs = [np.load(p) for p in sorted(parts) if len(np.load(p)[keys[0]])]
    return {k: np.concatenate([z[k] for z in Zs]) for k in Zs[0].files} if Zs else {}


Fc, Vc = gather(FACE_PARTS.glob("*.npz"), ["spk"]), gather(VOICE_PARTS.glob("tar*.npz"), ["spk"])
for n in ("ibv", "vgg", "vitageprob"):
    np.save(OUT / f"face_{n}.npy", Fc[n])
for n in ("iba", "btc", "ecapa192", "w2v2agage", "w2v2aggen"):
    np.save(OUT / f"voice_{n}.npy", Vc[n])
FM_ = pd.DataFrame({"spk": Fc["spk"], "video": Fc["video"], "utt": Fc["utt"]})
VM_ = pd.DataFrame({"spk": Vc["spk"], "video": Vc["video"], "utt": Vc["utt"], "dur": Vc["dur"]})
FM_["gender"], VM_["gender"] = FM_.spk.map(GENDER), VM_.spk.map(GENDER)
FM_.to_csv(OUT / "face_meta.csv", index=False); VM_.to_csv(OUT / "voice_meta.csv", index=False)
both = sorted(set(FM_.spk) & set(VM_.spk))
log(f"faces {len(FM_)} ({FM_.spk.nunique()} ids) | voices {len(VM_)} ({VM_.spk.nunique()} ids, "
    f"{VM_.groupby('spk').video.nunique().mean():.1f} videos / id) | ids with both: {len(both)}")

# zero-shot probe: ImageBind face <-> voice (positives = same identity, face and voice of DIFFERENT videos), negatives
# of the same gender;
# organiser ECAPA voice <-> voice (these are its TRAINING speakers: expect a very low EER)
from sklearn.metrics import roc_curve
def eer(s, y):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * (fpr[i] + fnr[i]) / 2



l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
rng = np.random.RandomState(0); PROBE = {}
if len(both) > 3:
    fi_of = FM_.groupby("spk").indices; vi_of = VM_.groupby("spk").indices
    Fi, Vi = l2n(Fc["ibv"].astype(np.float32)), l2n(Vc["iba"].astype(np.float32))
    Fi, Vi = l2n(Fi - Fi.mean(0)), l2n(Vi - Vi.mean(0))
    pos, neg = [], []
    for _ in range(min(20000, 50 * len(both))):
        s = both[rng.randint(len(both))]; o = both[rng.randint(len(both))]
        vi = rng.choice(vi_of[s]); cand = [f for f in fi_of[s] if FM_.video.values[f] != VM_.video.values[vi]]
        if cand:
            pos.append((rng.choice(cand), vi))
        if o != s and GENDER[o] == GENDER[s]:
            neg.append((rng.choice(fi_of[o]), rng.choice(vi_of[s])))
    if pos and neg:
        pr = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        PROBE["imagebind_face_voice_same_gender"] = round(float(eer(np.sum(Fi[pr[:, 0]] * Vi[pr[:, 1]], 1), y)), 2)
    B_ = l2n(Vc["btc"].astype(np.float32)); vv_p, vv_n = [], []
    for s in both:
        if len(vi_of[s]) > 1:
            a, b = rng.choice(vi_of[s], 2, replace=False); vv_p.append(float(B_[a] @ B_[b]))
        o = both[rng.randint(len(both))]
        if o != s:
            vv_n.append(float(B_[rng.choice(vi_of[s])] @ B_[rng.choice(vi_of[o])]))
    if vv_p and vv_n:
        PROBE["btc_voice_voice"] = round(float(eer(np.r_[vv_p, vv_n], np.r_[np.ones(len(vv_p)), np.zeros(len(vv_n))])), 2)
print("zero-shot probe (EER %):", PROBE)

man = {"created": time.strftime("%Y-%m-%d %H:%M"), "identities": len(M), "faces": len(FM_), "voices": len(VM_),
       "ids_with_both": len(both), "settings": dict(NF=NF, NV=NV, PER_VIDEO=PER_VIDEO, MAX_TARS=MAX_TARS, CLIP_SEC=CLIP_SEC),
       "sources": {"voices": "huggingface.co/datasets/gaunernst/voxceleb2-dev-wds (VoxCeleb2 dev, audio)",
                   "faces": "huggingface.co/datasets/ProgramComputer/voxceleb vox2_mp4_{2..6}.zip (VoxCeleb2 dev face tracks)",
                   "map": META_URL},
       "probe": PROBE, "broken": BROKEN, "audio": AUDIO_STATS,
       "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
       "rules": "no MAV-Celeb identity was compared with these corpora; clips keep their keys for a later language filter"}
(OUT / "feats_manifest.json").write_text(json.dumps(man, indent=1, default=str))
print(json.dumps({k: v for k, v in man.items() if k not in ("broken",)}, indent=1, default=str))
print("NEXT: Save Version -> Output -> New Dataset -> name it flag2027-feats-vox2 (private). If the session stopped "
      "early, attach the output and run again: finished files / tars are skipped.")
'''

if __name__ == "__main__":
    B12.nb([("markdown", INTRO), ("code", ENV), ("code", PIP),
            ("markdown", "## 1. Audio / image helpers (FLAG_12 cell)"), ("code", B12.AUDIO12),
            ("markdown", "## 2. Encoders: FLAG_12 cell (ImageBind, ViT age, w2v2 age/gender, speechbrain ECAPA) + the organiser's ECAPA and VGGFace"),
            ("code", B12.ENC12), ("code", ORG),
            ("markdown", "## 3. Identity list: VoxCeleb2 dev speakers with face tracks (zips 2-6)"), ("code", CONFIG),
            ("markdown", "## 4. Faces: middle frame of VoxCeleb2 face tracks of different videos, read remotely; resumable per chunk"), ("code", FACES),
            ("markdown", "## 5. Voices (VoxCeleb2): stream the tars; resumable per tar"), ("code", VOICES),
            ("markdown", "## 6. Assemble, zero-shot probe, manifest"), ("code", ASSEMBLE)], "FLAG_37_vox2_feats.ipynb")
