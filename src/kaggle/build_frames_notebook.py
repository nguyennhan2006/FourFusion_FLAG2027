"""Generate FLAG_63_frames.ipynb: a LOSSLESS (PNG) face-frame cache for face-encoder fine-tuning on many identities
(FACELORA-03; user 12/10: identity diversity + encoder fine-tuning). No model, CPU only (Kaggle CPU session: no GPU hours).
  vb/<part>_<shard>.zip   the 68,041 VoxBlink-clean face crops of FLAG_46 (feats_vb/face_meta.csv rows): middle frame of the
                          utterance, loose crop 1.3 x the official face box, 224 x 224 (FLAG_46's vb_media.loose_crop, same frame)
  vox2b/faces_NNN.zip     the 36,426 VoxCeleb2 face tracks of FLAG_53 (new + more speakers): middle frame of the 224 x 224 track
Member names: <spk>/<video>/<utt>.png. The FLAG_37 frames already exist (FLAG_45 output vox_frames_png.zip, by row number).
Inputs: flag2027-frames-index-p (FLAG_46 selection / timestamps / shard listing / face_meta; FLAG_53 mp4 listings / face_meta).
    python build_frames_notebook.py
Smoke test on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;FRAMES_SMOKE_N=30
"""
from pathlib import Path
import build_eval_notebooks as B12

HERE = Path(__file__).resolve().parent
MEDIA = (HERE / "vendor_vb/vb_media.py").read_text(encoding="utf-8")

INTRO = """
# FLAG 2027 — NB-63: lossless face-frame cache for face-encoder fine-tuning on ~21k identities (FACELORA-03)

**Why.** The user's direction (12/10): identity diversity and fine-tuning the encoder. FACELORA-02 adapted FaRL / CLIP-L on the
4,852 VoxCeleb2 identities of FLAG_37 only. To fine-tune on ~21k identities we need the frames of VoxBlink-clean (FLAG_46) and of
the VoxCeleb2 speakers FLAG_53 added. CLIP-L moves to cosine 0.93 under JPEG (FACELORA-02), so frames are kept **lossless (PNG)**
and re-cut **exactly** like the frozen features (same frame, same crop), so a step-0 wiring check can compare them.

| output | rows | frame |
|---|---|---|
| `frames/vb/<part>_<shard>.zip` | 68,041 VoxBlink-clean faces (FLAG_46's rows) | middle frame of the utterance, loose crop 1.3 x, 224 |
| `frames/vox2b/faces_NNN.zip` | 36,426 VoxCeleb2 faces (FLAG_53's rows) | middle frame of the 224 x 224 track |

**Rules:** VoxBlink is CC BY-NC-SA (no redistribution): keep this notebook and its output **private**. Bengali / Bangladesh
VoxBlink speakers were never selected (FLAG_46); the LID speaker filter is applied again where the frames are used.

**Inputs:** `flag2027-frames-index-p`. **Settings:** CPU, **Internet ON**. **Time:** about 3-5 h (resumable per shard / chunk:
attach this notebook's output and run again). **Output:** `/kaggle/working/frames/` (about 8-10 GB).
"""

ENV = r'''
import os, sys, io, re, json, time, shutil, zipfile, threading, struct, zlib, hashlib
from pathlib import Path
import numpy as np, pandas as pd
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORK = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
OUT = WORK / "frames"
for d in (OUT, OUT / "vb", OUT / "vox2b"):
    d.mkdir(parents=True, exist_ok=True)
SMOKE = LOCAL or os.environ.get("FLAG_SMOKE") == "1"
SMOKE_N = int(os.environ.get("FRAMES_SMOKE_N", "30"))
TIME_LIMIT_H = float(os.environ.get("FRAMES_TIME_LIMIT_H", "11"))
THREADS = int(os.environ.get("FRAMES_THREADS", "12"))
T_START = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T_START) / 3600:4.1f}h]", *a, flush=True)


def time_left():
    return time.time() - T_START < TIME_LIMIT_H * 3600


log("frames cache | smoke", SMOKE, "| threads", THREADS, "| time limit", TIME_LIMIT_H, "h")
'''

PIP = r'''
if not LOCAL:
    !pip -q install av 2>&1 | tail -1
import requests, av
log("av", av.__version__)
# resume: chunk zips of an earlier run (this notebook's output attached)
n_res = 0
for prev in ([] if LOCAL else [p for p in IN.rglob("frames") if p.is_dir() and (p / "vb").exists()]):
    for f in list((prev / "vb").glob("*.zip")) + list((prev / "vox2b").glob("*.zip")):
        tgt = OUT / f.parent.name / f.name
        if not tgt.exists():
            shutil.copy(f, tgt); n_res += 1
log("resume: chunk zips copied from an earlier run:", n_res)
IDX_VB = next(p.parent for p in IN.rglob("shard_listing.csv"))
IDX_V2 = next(p.parent for p in IN.rglob("listing_dev_mp4.parquet"))
log("index:", IDX_VB, "|", IDX_V2)
'''

VB = r'''
# ============================================================================ VoxBlink: FLAG_46's face rows, same frame, same crop, PNG
import concurrent.futures as cf
SHARD = "https://huggingface.co/datasets/adollahamini1998/VoxBlink-Video-clean-8-{}/resolve/main/{}"
CROP_SCALE = 1.3
dt = {"spk": str, "video": str, "utt": str}
FMV = pd.read_csv(IDX_VB / "face_meta.csv", dtype=dt)
SEL = pd.read_csv(IDX_VB / "selection.csv", dtype=dt)
TSV = pd.read_csv(IDX_VB / "timestamps.csv", dtype=dt)
LSV = pd.read_csv(IDX_VB / "shard_listing.csv")
LSV["spk"], LSV["video"] = LSV.member.str.split("/").str[0], LSV.member.str.split("/").str[1].str[:-4]
JOB = SEL.merge(TSV, on=["spk", "video", "utt"]).merge(LSV[["part", "shard", "spk", "video", "off", "size"]],
                                                       on=["part", "shard", "spk", "video"])
want = set(zip(FMV.spk, FMV.video, FMV.utt))
JOB = JOB[[k in want for k in zip(JOB.spk, JOB.video, JOB.utt)]].drop_duplicates(["spk", "video", "utt"]).reset_index(drop=True)
log(f"VoxBlink: {len(FMV)} face rows wanted, {len(JOB)} with selection + timestamp + member offset, {JOB.spk.nunique()} speakers, "
    f"{JOB.groupby(['part', 'shard']).ngroups} shards")
_TL = threading.local()


def cdn_url(part, shard):
    with requests.get(SHARD.format(part, shard), headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=120, stream=True) as r:
        return r.url


def decode_frames(mp4, frames_wanted):
    """= vb_media.decode_video without the audio: the RGB frames at the wanted indices (decoded in order)."""
    want, got = sorted(set(int(f) for f in frames_wanted)), {}
    with av.open(io.BytesIO(mp4)) as c:
        vs = c.streams.video[0] if c.streams.video else None
        if vs is not None:
            vs.thread_type = "AUTO"
            last = max(want) if want else -1
            for k, fr in enumerate(c.decode(vs)):
                if k in want:
                    got[k] = fr.to_ndarray(format="rgb24")
                if k >= last:
                    break
    return got


def vb_job(job):
    (part, shard, spk, video), g = job
    if not hasattr(_TL, "s"):
        _TL.s, _TL.urls = requests.Session(), {}
    for attempt in range(4):
        try:
            url = _TL.urls.get((part, shard)) or _TL.urls.setdefault((part, shard), cdn_url(part, shard))
            off, size = int(g.off.iloc[0]), int(g["size"].iloc[0])
            b = _TL.s.get(url, headers={"Range": f"bytes={off}-{off + size - 1}"}, timeout=300).content
            if len(b) != size:
                _TL.urls.pop((part, shard), None); raise IOError(f"short read {len(b)} / {size}")
            frames = decode_frames(b, g.mid.tolist())
            out = []
            for r in g.itertuples():
                if r.mid in frames:
                    im, _ = loose_crop(frames[r.mid], (r.x, r.y, r.w, r.h), CROP_SCALE)
                    buf = io.BytesIO(); im.save(buf, "PNG", compress_level=3)
                    out.append((f"{spk}/{video}/{r.utt}.png", buf.getvalue()))
            return out, None
        except Exception as e:
            err = f"{spk}/{video}: {type(e).__name__}: {e}"; time.sleep(3 * (attempt + 1))
    return [], err


order = JOB.groupby(["part", "shard"]).size().sort_index()
if SMOKE:
    order = order.iloc[:1]
STATS_VB = dict(shards=0, videos=0, failed=0, faces=0)
for (part, shard), n in order.items():
    zf = OUT / "vb" / f"{part}_{shard}.zip"
    if zf.exists():
        continue
    if not time_left():
        log(f"time limit reached: stopping before VoxBlink shard {shard} (attach the output and run again)"); break
    t0 = time.time()
    G = JOB[(JOB.part == part) & (JOB.shard == shard)]
    jobs = list(G.groupby(["part", "shard", "spk", "video"]))
    if SMOKE:
        jobs = jobs[:SMOKE_N]
    got, errs = [], []
    with cf.ThreadPoolExecutor(THREADS) as ex:
        for res, err in ex.map(vb_job, jobs):
            got += res
            if err:
                errs.append(err)
    tmp = zf.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for name, b in got:
            z.writestr(name, b)
    tmp.replace(zf)
    STATS_VB["shards"] += 1; STATS_VB["videos"] += len(jobs); STATS_VB["failed"] += len(errs); STATS_VB["faces"] += len(got)
    log(f"VoxBlink {shard}: {len(jobs)} videos ({len(errs)} failed), {len(got)}/{len(G)} faces [{time.time() - t0:.0f}s]"
        + (f" | e.g. {errs[0][:120]}" if errs else ""))
log("VoxBlink done this session:", STATS_VB)
'''

VOX2B = r'''
# ============================================================================ VoxCeleb2 (FLAG_53 rows): middle frame of the track, PNG
HF = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/"
SOURCES = {"dev_mp4": [f"vox2_dev_mp4_parta{c}" for c in "abcdefghi"], "test_mp4": ["vox2_test_mp4.zip"]}


class Remote:
    """= FLAG_53 Remote: a zip made of byte-chunk parts, read by absolute offset with HTTP range requests."""
    def __init__(self, names):
        self.names, self.urls, self.sizes = names, {}, []
        for n in names:
            with requests.get(HF + n, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=120, stream=True) as r:
                r.raise_for_status()
                self.urls[n] = r.url
                self.sizes.append(int(r.headers["Content-Range"].split("/")[-1]))
        self.starts = np.r_[0, np.cumsum(self.sizes)]
        self.tl = threading.local()

    def _get(self, j, a, b):
        n = self.names[j]
        if not hasattr(self.tl, "s"):
            self.tl.s = requests.Session()
        for attempt in range(5):
            try:
                r = self.tl.s.get(self.urls[n], headers={"Range": f"bytes={a}-{b}"}, timeout=300)
                if r.status_code in (401, 403):
                    with requests.get(HF + n, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=120, stream=True) as q:
                        self.urls[n] = q.url
                    continue
                r.raise_for_status()
                assert len(r.content) == b - a + 1, (len(r.content), b - a + 1)
                return r.content
            except Exception:
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"range {a}-{b} of {n} failed")

    def read(self, off, n):
        out, end = [], off + n
        while off < end:
            j = int(np.searchsorted(self.starts, off, side="right") - 1)
            a = off - self.starts[j]; b = min(end, self.starts[j + 1]) - self.starts[j] - 1
            out.append(self._get(j, int(a), int(b))); off = int(self.starts[j] + b + 1)
        return b"".join(out)

    def member(self, off, csize, method):
        h = self.read(off, 30)
        assert h[:4] == b"PK\x03\x04", "bad local header"
        n, e = struct.unpack("<HH", h[26:30])
        d = self.read(off + 30 + n + e, csize)
        return zlib.decompress(d, -15) if method == 8 else d


def middle_frame(mp4):
    """= FLAG_53 / FLAG_37 middle_frame."""
    with av.open(io.BytesIO(mp4)) as c:
        st = c.streams.video[0]; n = st.frames or 50
        last = None
        for k, fr in enumerate(c.decode(st)):
            last = fr
            if k >= n // 2:
                break
        return last.to_image().convert("RGB") if last is not None else None


FM2 = pd.read_csv(IDX_V2 / "face_meta.csv", dtype=str)
L2 = pd.concat([pd.read_parquet(IDX_V2 / f"listing_{k}.parquet").assign(src=k) for k in SOURCES], ignore_index=True)
J2 = FM2.merge(L2[["spk", "video", "utt", "src", "off", "csize", "method"]], on=["spk", "video", "utt"], how="left")
assert J2.src.notna().all(), f"{J2.src.isna().sum()} FLAG_53 face rows not in the listings"
if SMOKE:
    J2 = J2.iloc[:SMOKE_N]
REM = {k: Remote(v) for k, v in SOURCES.items() if k in set(J2.src)}
log(f"VoxCeleb2 (FLAG_53): {len(J2)} face tracks ({FM2.kind.value_counts().to_dict()}), {J2.spk.nunique()} speakers")


def v2_job(r):
    try:
        im = middle_frame(REM[r.src].member(int(r.off), int(r.csize), int(r.method)))
        if im is None:
            return None
        buf = io.BytesIO(); im.save(buf, "PNG", compress_level=3)
        return f"{r.spk}/{r.video}/{r.utt}.png", buf.getvalue()
    except Exception:
        return None


CH = 1200
STATS_V2 = dict(chunks=0, faces=0, failed=0)
for c0 in range(0, len(J2), CH):
    zf = OUT / "vox2b" / f"faces_{c0 // CH:03d}.zip"
    if zf.exists():
        continue
    if not time_left():
        log(f"time limit reached: stopping before VoxCeleb2 chunk {c0 // CH} (attach the output and run again)"); break
    t0 = time.time(); S_ = J2.iloc[c0:c0 + CH]
    with cf.ThreadPoolExecutor(THREADS) as ex:
        res = list(ex.map(v2_job, S_.itertuples(index=False)))
    ok = [x for x in res if x is not None]
    tmp = zf.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for name, b in ok:
            z.writestr(name, b)
    tmp.replace(zf)
    STATS_V2["chunks"] += 1; STATS_V2["faces"] += len(ok); STATS_V2["failed"] += len(S_) - len(ok)
    log(f"VoxCeleb2 chunk {c0 // CH}: {len(ok)}/{len(S_)} faces [{time.time() - t0:.0f}s]")
log("VoxCeleb2 done this session:", STATS_V2)
'''

DONE = r'''
# ============================================================================ manifest + a check that every member decodes to 224 x 224 RGB
from PIL import Image
man = dict(created=time.strftime("%Y-%m-%d %H:%M"), smoke=SMOKE, parts={})
for part in ("vb", "vox2b"):
    zs = sorted((OUT / part).glob("*.zip"))
    names = []
    for f in zs:
        with zipfile.ZipFile(f) as z:
            names += z.namelist()
    man["parts"][part] = dict(zips=len(zs), faces=len(names), bytes=int(sum(f.stat().st_size for f in zs)))
    if zs:
        with zipfile.ZipFile(zs[0]) as z:
            for n in z.namelist()[:5]:
                im = Image.open(io.BytesIO(z.read(n)))
                assert im.size == (224, 224) and im.mode == "RGB", (n, im.size, im.mode)
man["wanted"] = dict(vb=int(len(FMV)), vox2b=int(len(FM2)))
man["complete"] = dict(vb=man["parts"]["vb"]["faces"] >= 0.98 * len(FMV), vox2b=man["parts"]["vox2b"]["faces"] >= 0.98 * len(FM2))
(OUT / "frames_manifest.json").write_text(json.dumps(man, indent=1))
log(json.dumps(man))
print("NEXT:", "all frames cached" if all(man["complete"].values()) else "not complete: Save Version, attach this output, run again",
      "| then FACELORA-03 attaches this notebook's output (same account)")
'''

if __name__ == "__main__":
    cells = [("markdown", INTRO), ("code", ENV), ("code", PIP), ("markdown", "## Helpers of FLAG_46 (vb_media.py)"), ("code", MEDIA),
             ("markdown", "## 1. VoxBlink face crops (FLAG_46's rows)"), ("code", VB),
             ("markdown", "## 2. VoxCeleb2 faces of FLAG_53"), ("code", VOX2B), ("markdown", "## 3. Manifest"), ("code", DONE)]
    B12.nb(cells, "FLAG_63_frames.ipynb")
