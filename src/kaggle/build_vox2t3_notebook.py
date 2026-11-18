"""Generate FLAG_41_vox2_t3enc.ipynb (T3ENC-01, V1 of docs/PLAN_VOXBLINK_2026-10-07.md): the four Track 3 encoders
(FaRL ep16, CLIP ViT-L/14, SimAM-ResNet100 pretrained / fine-tuned) on EXACTLY the VoxCeleb2 samples of FLAG_37: the key
lists face_meta.csv / voice_meta.csv of `flag2027-feats-vox2` fix every sample, the arrays come out row-aligned with
FLAG_37's (so the old and new features of one sample sit on the same row).
    python build_vox2t3_notebook.py
    FLAG_LOCAL=1 FLAG_OUT=<dir> FLAG_CACHE=<cache> FLAG_IN=<dir with feats_vox2/> VOX_SMOKE_IDS=3 FLAG_T3_PATH=<isolated dir> \
        python run_notebook_local.py FLAG_41_vox2_t3enc.ipynb --jupyter
"""
from pathlib import Path
import build_eval_notebooks as B12
import build_t3enc_notebook as T3

HERE = Path(__file__).resolve().parent
CELL = (HERE / "vendor_t3enc/t3enc_cell.py").read_text(encoding="utf-8")

INTRO = """
# FLAG 2027 — NB-41: T3ENC-01, the Track 3 encoders on the VoxCeleb2 samples of FLAG_37

**Why.** Twin of FLAG_40 (same four encoders on our MAV-Celeb rows). With these, the local experiment can train a
face <-> voice head on MAV-Celeb + ~4,900 VoxCeleb2 identities in the encoder space of Track 3's VoxBlink system, and
measure whether the encoders and a head, without VoxBlink data, already reach its level (plan V1).

**Samples:** exactly those of FLAG_37, read from its key lists (`face_meta.csv`, `voice_meta.csv` in
`flag2027-feats-vox2`): face = middle frame of the same VoxCeleb2 face track, voice = the same clip, 12 s centre crop.
The output rows follow the key lists, so they align with FLAG_37's arrays.

| array | model | dim |
|---|---|---|
| `face_farl16` | FaRL ViT-B/16 LAION-Face 20M ep16 | 512 |
| `face_clipL` | `openai/clip-vit-large-patch14` @ 32bd642, image features | 768 |
| `voice_sam100p` | WeSpeaker `voxblink2_samresnet100` (VoxBlink2 pretrained) | 256 |
| `voice_sam100f` | WeSpeaker `voxblink2_samresnet100_ft` (VoxBlink2 -> **VoxCeleb2** fine-tune: these speakers are its training speakers) | 256 |

**Rules:** as FLAG_37: frozen encoders; no MAV-Celeb identity is compared with VoxCeleb2; Bengali: VoxCeleb2 may contain some
(organiser question 6); every clip keeps its key for a later language filter. Declare VoxCeleb2 in the system description.

**Inputs (Add Input):** **`flag2027-vox2-keys`** (the two key lists, 2.3 MB, upload of `kaggle_upload/flag2027-vox2-keys/`) OR the FLAG_37 output `flag2027-feats-vox2`: only `face_meta.csv` / `voice_meta.csv` are read. Optional: this
notebook's previous output, to resume. **Settings:** GPU T4, **Internet ON**.

**Time:** about 3-4 h: faces ~1 h (29k face tracks by HTTP range reads, two models); voices ~2-3 h (the same tars as FLAG_37,
streamed until every key is found, two models). Resumable per chunk / tar.

**Output:** `/kaggle/working/feats_vox2t3/` -> dataset **`flag2027-feats-vox2t3`** (about 0.2 GB, float16).
"""

ENV = B12.ENV12.replace('OUT = WORKDIR / "feats_eval"', 'OUT = WORKDIR / "feats_vox2t3"')
assert ENV != B12.ENV12
PIP = T3.PIP.replace('!pip -q install ftfy regex kaldiio silero-vad', '!pip -q install ftfy regex kaldiio silero-vad av')
assert PIP != T3.PIP

CONFIG = r'''
# ============================================================================ the FLAG_37 key lists, zip listings, helpers
import tarfile, requests, struct, zlib, threading
import concurrent.futures as cf
from huggingface_hub import HfFileSystem

MAX_TARS = 779
CLIP_SEC = 12.0                                          # centre crop, as FLAG_37
THREADS = 16
SEED = 0                                                 # FLAG_37's tar order: RandomState(0).permutation(779)
SMOKE_IDS = int(os.environ.get("VOX_SMOKE_IDS", "0")) if LOCAL else 0
ZIPS = [2, 3, 4, 5, 6]
ZIP_PATH = "datasets/ProgramComputer/voxceleb/vox2/vox2_mp4_{}.zip"
ZIP_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_{}.zip"
VOICE_URL = "https://huggingface.co/datasets/gaunernst/voxceleb2-dev-wds/resolve/main/voxceleb2-dev-{:04d}.tar"
FS = HfFileSystem()
FACE_PARTS, VOICE_PARTS = OUT / "parts_face", OUT / "parts_voice"
for d in (FACE_PARTS, VOICE_PARTS):
    d.mkdir(parents=True, exist_ok=True)
for prev in (IN.rglob("feats_vox2t3") if IN.exists() and not LOCAL else []):     # resume: earlier output attached
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
    """Central directory of vox2_mp4_<k>.zip (= FLAG_37): one row per face track (dev/mp4/<spk>/<video>/<utt>.mp4)."""
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


fk = [p for p in IN.rglob("face_meta.csv") if "feats_vox2t3" not in str(p)] if IN.exists() else []
vk = [p for p in IN.rglob("voice_meta.csv") if "feats_vox2t3" not in str(p)] if IN.exists() else []
assert fk and vk, "attach flag2027-vox2-keys (or the FLAG_37 output flag2027-feats-vox2): its face_meta.csv / voice_meta.csv fix the samples"
FK = pd.read_csv(fk[0], dtype={"spk": str, "video": str, "utt": str})
VK = pd.read_csv(vk[0], dtype={"spk": str, "video": str, "utt": str})
if SMOKE_IDS:                                            # local smoke test: speakers whose kept clips sit in the first tar
    keyset, keep = set(zip(VK.spk, VK.video, VK.utt)), []
    with get_retry(VOICE_URL.format(int(np.random.RandomState(SEED).permutation(779)[0])), stream=True) as r:
        r.raw.decode_content = True
        for m in tarfile.open(fileobj=r.raw, mode="r|"):
            k_ = tuple(m.name[:-4].split("/")[-3:]) if m.name.endswith(".m4a") else None
            if k_ in keyset and k_[0] not in keep:
                keep.append(k_[0])
            if len(keep) >= SMOKE_IDS:
                break
    FK, VK = FK[FK.spk.isin(keep)].reset_index(drop=True), VK[VK.spk.isin(keep)].reset_index(drop=True)
    MAX_TARS = 1
log(f"keys: {len(FK)} faces / {FK.spk.nunique()} ids, {len(VK)} voices / {VK.spk.nunique()} ids (from {fk[0].parent})")


'''

FACES = r'''
# ============================================================================ faces: the same middle frame of the same face track as FLAG_37
_TL, URLS = threading.local(), {}


def resolve(k):
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


LST = pd.concat([zip_listing(k) for k in ZIPS], ignore_index=True)
for k in ZIPS:
    URLS[k] = resolve(k)
SEL = FK[["spk", "video", "utt"]].merge(LST[["spk", "video", "utt", "zip", "off", "csize", "ctype"]], how="left",
                                         on=["spk", "video", "utt"])
assert SEL.zip.notna().all(), f"{SEL.zip.isna().sum()} face keys not found in the zip listings"
SEL["row"] = np.arange(len(SEL)); CH = 2400
log(f"faces: {len(SEL)} tracks, {len(SEL) // CH + 1} chunks, {THREADS} threads")
FACE_ENC = None
for c0 in range(0, len(SEL), CH):
    part = FACE_PARTS / f"faces_{c0 // CH:03d}.npz"
    if part.exists():
        continue
    if FACE_ENC is None:
        FACE_ENC = {"farl16": build("farl16V"), "clipL": build("clipLV")}
    t0 = time.time(); nb0 = len(BROKEN["image"])
    S_ = SEL.iloc[c0:c0 + CH]
    with cf.ThreadPoolExecutor(THREADS) as ex:
        imgs = list(ex.map(fetch_frame, zip(S_.zip.astype(int), S_.off.astype(int), S_.csize.astype(int), S_.ctype.astype(int))))
    ok = [i for i, im in enumerate(imgs) if im is not None]
    rows_ok, imgs = S_.row.values[ok], [imgs[i] for i in ok]
    R = {n: [] for n in FACE_ENC}
    for s in range(0, len(imgs), 64):
        b = imgs[s:s + 64]
        for n, enc in FACE_ENC.items():
            R[n].append(enc(torch.stack([enc.prep(im) for im in b]))[""])
    np.savez(part, row=rows_ok, **{n: np.concatenate(v).astype(np.float16) for n, v in R.items() if v})
    log(f"  chunk {c0 // CH}: {len(imgs)} faces, {len(BROKEN['image']) - nb0} failed [{time.time() - t0:.0f}s]")
FACE_ENC = None
release()
'''

VOICES = r'''
# ============================================================================ voices: the same clips as FLAG_37 (12 s centre crop), streamed
def decode_m4a(b):
    import av
    out = []
    with av.open(io.BytesIO(b)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        for fr in c.decode(c.streams.audio[0]):
            out += [o.to_ndarray().reshape(-1) for o in res.resample(fr)]
        out += [o.to_ndarray().reshape(-1) for o in res.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(TARGET_SR // 10, np.float32)


row_of = {(s, v, u): i for i, (s, v, u) in enumerate(zip(VK.spk, VK.video, VK.utt))}
done = set()
for p in sorted(VOICE_PARTS.glob("tar*.npz")):            # resume
    done |= set(np.load(p)["row"].tolist())
order = list(np.random.RandomState(SEED).permutation(779))
VOICE_ENC = None
L = int(CLIP_SEC * TARGET_SR)
for t in order[:MAX_TARS]:
    if len(done) >= len(row_of):
        log("every voice key found"); break
    part = VOICE_PARTS / f"tar{t:04d}.npz"
    if part.exists():
        continue
    t0 = time.time(); waves, rows = [], []
    try:
        with get_retry(VOICE_URL.format(t), stream=True) as r:
            r.raw.decode_content = True
            tf = tarfile.open(fileobj=r.raw, mode="r|")
            for m in tf:
                if not m.name.endswith(".m4a"):
                    continue
                key = tuple(m.name[:-4].split("/")[-3:])
                i = row_of.get(key)
                if i is None or i in done:
                    continue
                try:
                    w = decode_m4a(tf.extractfile(m).read())
                except Exception as e:
                    BROKEN["audio"].append(f"{m.name} | {type(e).__name__}: {e}"); continue
                waves.append(centre_crop(w, L)); rows.append(i)
    except Exception as e:
        log(f"!! tar {t}: {type(e).__name__}: {e} -> skipped"); continue
    if not waves:
        np.savez(part, row=np.array([], int)); continue
    if VOICE_ENC is None:
        VOICE_ENC = {"sam100p": build("sam100pA"), "sam100f": build("sam100fA")}
    R = {n: np.stack([enc(w)[""] for w in waves]) for n, enc in VOICE_ENC.items()}
    np.savez(part, row=np.array(rows), **{n: v.astype(np.float16) for n, v in R.items()})
    done |= set(rows)
    log(f"tar {t:04d}: {len(waves)} clips [{time.time() - t0:.0f}s] | keys found {len(done)}/{len(row_of)}")
VOICE_ENC = None
release()
'''

ASSEMBLE = r'''
# ============================================================================ assemble in key-list order (= FLAG_37 rows), probe, manifest
def assemble(parts, n_rows, names):
    out = {n: None for n in names}
    for p in sorted(parts):
        Z = np.load(p)
        if not len(Z["row"]):
            continue
        for n in names:
            if out[n] is None:
                out[n] = np.full((n_rows, Z[n].shape[1]), np.nan, np.float16)
            out[n][Z["row"]] = Z[n]
    return out


Fo = assemble(FACE_PARTS.glob("*.npz"), len(FK), ["farl16", "clipL"])
Vo = assemble(VOICE_PARTS.glob("tar*.npz"), len(VK), ["sam100p", "sam100f"])
cov = {}
for n, A in {**{f"face_{k}": v for k, v in Fo.items()}, **{f"voice_{k}": v for k, v in Vo.items()}}.items():
    if A is not None:
        np.save(OUT / f"{n}.npy", A); cov[n] = round(float(np.isfinite(A[:, 0].astype(np.float32)).mean()), 4)
FK.to_csv(OUT / "face_meta.csv", index=False); VK.to_csv(OUT / "voice_meta.csv", index=False)
log(f"coverage (share of key-list rows filled): {cov}")

from sklearn.metrics import roc_curve
def eer(s, y):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * (fpr[i] + fnr[i]) / 2


l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
PROBE, rng = {}, np.random.RandomState(0)
for side, A, M_ in (("face_farl16", Fo["farl16"], FK), ("face_clipL", Fo["clipL"], FK), ("voice_sam100p", Vo["sam100p"], VK),
                    ("voice_sam100f", Vo["sam100f"], VK)):
    if A is None:
        continue
    ok = np.isfinite(A[:, 0].astype(np.float32)); X = l2n(np.nan_to_num(A.astype(np.float32)))
    by = {s: np.where((M_.spk.values == s) & ok)[0] for s in M_.spk.unique()}
    sp = [s for s, ix in by.items() if len(ix) > 1]
    if len(sp) < 3:
        continue
    pos = [tuple(rng.choice(by[s], 2, replace=False)) for s in rng.choice(sp, 3000)]
    neg = [(rng.choice(by[a]), rng.choice(by[b])) for a, b in rng.choice(sp, (3000, 2)) if a != b]
    pr = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    PROBE[side] = round(float(eer(np.sum(X[pr[:, 0]] * X[pr[:, 1]], 1), y)), 2)
print("within-modality identity probe (EER %; sam100f: VoxCeleb2 = its fine-tune speakers, expect very low):", PROBE)
man = {"created": time.strftime("%Y-%m-%d %H:%M"), "faces": len(FK), "voices": len(VK), "coverage": cov, "probe": PROBE,
       "encoders": {"farl16": FARL16_URL, "clipL": f"{CLIPL_ID}@{CLIPL_REV}", "sam100p": SAM_ZIPS["sam100pA"][0],
                    "sam100f": SAM_ZIPS["sam100fA"][0]},
       "samples": "the key lists of FLAG_37 (flag2027-feats-vox2); rows aligned with its arrays",
       "broken": BROKEN, "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
       "rules": "no MAV-Celeb identity was compared with these corpora; clips keep their keys for a later language filter"}
(OUT / "feats_manifest.json").write_text(json.dumps(man, indent=1, default=str))
print(json.dumps({k: v for k, v in man.items() if k != "broken"}, indent=1, default=str))
print("NEXT: Save Version -> Output -> New Dataset -> name it flag2027-feats-vox2t3 (private). If the session stopped "
      "early, attach the output and run again: finished chunks / tars are skipped.")
'''

if __name__ == "__main__":
    B12.nb([("markdown", INTRO), ("code", ENV), ("code", PIP),
            ("markdown", "## 1. Audio / image helpers (FLAG_12 cell)"), ("code", B12.AUDIO12),
            ("markdown", "## 2. The four Track 3 encoders (same cell as FLAG_40)"), ("code", CELL),
            ("markdown", "## 3. FLAG_37 key lists, zip listings"), ("code", CONFIG),
            ("markdown", "## 4. Faces (same face tracks as FLAG_37); resumable per chunk"), ("code", FACES),
            ("markdown", "## 5. Voices (same clips as FLAG_37); resumable per tar"), ("code", VOICES),
            ("markdown", "## 6. Assemble in FLAG_37 row order, probe, manifest"), ("code", ASSEMBLE)], "FLAG_41_vox2_t3enc.ipynb")
