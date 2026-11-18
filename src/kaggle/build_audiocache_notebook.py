"""Generate FLAG_63v_audio_cache.ipynb: VOICELORA-01 step 0 (Experiment/VOICELORA-01/README.md, registered 15/10): an audio cache of
the ~21k identities the face adapters were trained on, so that a voice adapter can be trained on raw audio. CPU only, Internet ON.
  vb/<part>_<shard>.npz     VoxBlink-clean voice rows of FLAG_46 (feats_vb/voice_meta.csv), re-fetched like FLAG_63 (same shard index):
                            utterance = audio[start/FPS*SR : (end+1)/FPS*SR] of the video (FLAG_46's cut)
  vox/tar<NNNN>.npz         VoxCeleb2 voice rows of FLAG_37 (feats_vox2) and FLAG_53 (feats_vox2b), streamed from the same WebDataset
                            tars (gaunernst/voxceleb2-dev-wds, original AAC)
Each row: the centre CROP_SEC (3 s) of the clip, 16 kHz mono int16 (padded with zeros; `n` = valid samples). Speakers dropped by the
registered LID rule of each source (lid_summary.json) are never fetched. A sample of WIRE_N clips per source keeps the centre 12 s
(the T3 cut) for the wiring check: frozen SimAM (sam100p / sam100f) on the re-fetched audio must reproduce the stored arrays.
    python build_audiocache_notebook.py
Smoke test on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;AUDIO_SMOKE_N=20;AUDIO_SMOKE_TARS=2
"""
from pathlib import Path
import build_eval_notebooks as B12

HERE = Path(__file__).resolve().parent
T3CELL = (HERE / "vendor_t3enc/t3enc_cell.py").read_text(encoding="utf-8")

INTRO = """
# FLAG 2027 — NB-63v: audio cache of the ~21k face-adapter identities (VOICELORA-01 step 0)

**Why.** FACELORA-03 B3 (a CLIP-L adapter trained to match frozen SimAM voices) gave the CC head +0.5 (CB 19.77). VOICELORA-01 asks
the mirror question: adapt the voice encoder to match the B3 faces. Raw audio of the VoxCeleb2 / VoxBlink identities is not in any
dataset (the earlier extractions streamed it), so step 0 caches it once. Registered in `Experiment/VOICELORA-01/README.md`.

| output | rows | audio |
|---|---|---|
| `audio_cache/vb/<part>_<shard>.npz` | VoxBlink-clean voice rows of FLAG_46 (LID-dropped speakers excluded) | FLAG_46's utterance cut, centre 3 s |
| `audio_cache/vox/tar<NNNN>.npz` | VoxCeleb2 voice rows of FLAG_37 + FLAG_53 (LID-dropped speakers excluded) | original AAC, centre 3 s |
| `audio_cache/wire_<source>.npz` | `WIRE_N` clips per source, centre 12 s | for the wiring check |

**Wiring check:** frozen SimAM sam100p / sam100f on the re-fetched centre 12 s must reproduce the stored `voice_sam100p` /
`voice_sam100f` of each source (mean cosine >= 0.99), else the cache does not hold the clips the embeddings came from.

**Rules:** VoxBlink is CC BY-NC-SA (no redistribution): this notebook, its output and the dataset stay **private**. No Bengali voice:
the speakers dropped by each source's registered LID rule are not fetched.

**Inputs:** `flag2027-frames-index-p` (VoxBlink shard index / timestamps), `flag2027-lab-vb-p` (feats_vb: voice rows + stored arrays),
`flag2027-lab-data-p` (feats_vox2 rows, feats_vox2t3 arrays, feats_vox2lid), `flag2027-feats-vox2b-p` (FLAG_53 rows + arrays).
**Settings:** CPU, **Internet ON**, secret `HF_TOKEN` (kapi `--secret HF_TOKEN`). **Time:** about 5-8 h (resumable per shard / tar:
attach this notebook's output and run again). **Output:** `/kaggle/working/audio_cache/` (about 13-15 GB), uploaded to the private
dataset `AUDIO_UPLOAD` when set.
"""

ENV = r'''
import os, sys, io, re, json, time, shutil, tarfile, threading, types
from pathlib import Path
import numpy as np, pandas as pd
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORK = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
OUT = WORK / "audio_cache"
for d in (OUT, OUT / "vb", OUT / "vox"):
    d.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("FLAG_CACHE", "/tmp/flag_cache")); CACHE.mkdir(parents=True, exist_ok=True)
SMOKE = LOCAL or os.environ.get("FLAG_SMOKE") == "1"
SMOKE_N = int(os.environ.get("AUDIO_SMOKE_N", "20"))                 # smoke: videos of the first VoxBlink shard
SMOKE_TARS = int(os.environ.get("AUDIO_SMOKE_TARS", "2"))            # smoke: VoxCeleb2 tars streamed
TIME_LIMIT_H = float(os.environ.get("AUDIO_TIME_LIMIT_H", "10.5"))
THREADS = int(os.environ.get("AUDIO_THREADS", "12"))
TAR_THREADS = int(os.environ.get("AUDIO_TAR_THREADS", "4"))
SR, FPS, CROP_SEC, WIRE_SEC = 16000, 25, 3.0, 12.0                  # FLAG_46's FPS; T3 cut = centre 12 s
WIRE_N = int(os.environ.get("AUDIO_WIRE_N", "48"))
L_CROP, L_WIRE = int(CROP_SEC * SR), int(WIRE_SEC * SR)
T_START = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T_START) / 3600:4.1f}h]", *a, flush=True)


def time_left():
    return time.time() - T_START < TIME_LIMIT_H * 3600


def centre_crop(w, L):                                               # = FLAG_12's centre_crop
    if len(w) <= L:
        return w
    s = (len(w) - L) // 2
    return w[s:s + L]


def to_i16(w, L):
    """centre L samples as int16 (zero padded) + the valid length."""
    c = centre_crop(w, L)
    out = np.zeros(L, np.int16); out[:len(c)] = np.clip(np.round(c * 32767.0), -32768, 32767).astype(np.int16)
    return out, len(c)


log("audio cache | smoke", SMOKE, "| threads", THREADS, TAR_THREADS, "| time limit", TIME_LIMIT_H, "h | crop", CROP_SEC, "s")
'''

PIP = r'''
# PyAV decodes AAC / mp4 audio; WeSpeaker (commit 9fecd6c, as the T3 notebooks) only for the wiring check (--no-deps: its pins
# would replace Kaggle's torch)
if not LOCAL:
    !pip -q install av kaldiio silero-vad 2>&1 | tail -1
    !pip -q install --no-deps "git+https://github.com/wenet-e2e/wespeaker.git@9fecd6cb4f47475d01761d87c826298dff4ef18c" 2>&1 | tail -1
import requests, av, torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HDR = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
log("av", av.__version__, "| torch", torch.__version__, "| device", DEV, "| HF token", bool(HDR))
# resume: chunk files of an earlier run (this notebook's output attached)
n_res = 0
for prev in ([] if LOCAL else [p for p in IN.rglob("audio_cache") if p.is_dir() and (p / "vb").exists()]):
    for f in list((prev / "vb").glob("*.npz")) + list((prev / "vox").glob("*.npz")) + list(prev.glob("wire_*.npz")):
        tgt = OUT / f.relative_to(prev)
        if not tgt.exists():
            tgt.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, tgt); n_res += 1
log("resume: chunk files copied from an earlier run:", n_res)
'''

INDEX = r'''
# ============================================================================ the voice rows of each source, minus the LID-dropped speakers
def one_dir(fname, parent):
    hits = sorted({p.parent for p in IN.rglob(fname) if p.parent.name == parent})
    assert hits, f"{fname} under a folder named {parent} not found: attach its dataset"
    return hits[0]


dt = {"spk": str, "video": str, "utt": str}
D_VB, D_V37, D_V2B = one_dir("voice_meta.csv", "feats_vb"), one_dir("voice_meta.csv", "feats_vox2"), one_dir("voice_meta.csv", "feats_vox2b")
D_V37T3, D_LID37 = one_dir("voice_sam100p.npy", "feats_vox2t3"), one_dir("lid_summary.json", "feats_vox2lid")
drop = lambda f: set(json.loads(Path(f).read_text())["speakers_dropped_ids"])
META = {"vb": pd.read_csv(D_VB / "voice_meta.csv", dtype=dt), "vox": pd.read_csv(D_V37 / "voice_meta.csv", dtype=dt),
        "vox2b": pd.read_csv(D_V2B / "voice_meta.csv", dtype=dt)}
DROP = {"vb": drop(D_VB / "lid_summary.json"), "vox": drop(D_LID37 / "lid_summary.json"), "vox2b": drop(D_V2B / "lid_summary.json")}
ARRS = {"vb": D_VB, "vox": D_V37T3, "vox2b": D_V2B}                  # stored voice_sam100p / voice_sam100f, rows = META rows
ROW = {s: {k: i for i, k in enumerate(zip(m.spk, m.video, m.utt))} for s, m in META.items()}
WANT = {s: {k for k in ROW[s] if k[0] not in DROP[s]} for s in META}
for s in META:
    log(f"{s}: {len(META[s])} voice rows, {len(DROP[s])} LID-dropped speakers -> {len(WANT[s])} rows, "
        f"{len({k[0] for k in WANT[s]})} speakers | arrays {ARRS[s]}")
IDX_VB = next(p.parent for p in IN.rglob("shard_listing.csv"))
SEL = pd.read_csv(IDX_VB / "selection.csv", dtype=dt)
TSV = pd.read_csv(IDX_VB / "timestamps.csv", dtype=dt)
LSV = pd.read_csv(IDX_VB / "shard_listing.csv")
LSV["spk"], LSV["video"] = LSV.member.str.split("/").str[0], LSV.member.str.split("/").str[1].str[:-4]
JOB = SEL.merge(TSV, on=["spk", "video", "utt"]).merge(LSV[["part", "shard", "spk", "video", "off", "size"]], on=["part", "shard", "spk", "video"])
JOB = JOB[[k in WANT["vb"] for k in zip(JOB.spk, JOB.video, JOB.utt)]].drop_duplicates(["spk", "video", "utt"]).reset_index(drop=True)
log(f"VoxBlink: {len(JOB)}/{len(WANT['vb'])} wanted rows with selection + timestamp + member offset, {JOB.groupby(['part', 'shard']).ngroups} shards")
'''

VB = r'''
# ============================================================================ VoxBlink: FLAG_46's utterance cut of the re-fetched video, centre 3 s
import concurrent.futures as cf
SHARD = "https://huggingface.co/datasets/adollahamini1998/VoxBlink-Video-clean-8-{}/resolve/main/{}"
_TL = threading.local()


def cdn_url(part, shard):
    with requests.get(SHARD.format(part, shard), headers={"Range": "bytes=0-0", **HDR}, allow_redirects=True, timeout=120, stream=True) as r:
        return r.url


def decode_audio(b):
    """The whole audio track as mono float32 at SR (= vb_media.decode_video's audio branch)."""
    out = []
    with av.open(io.BytesIO(b)) as c:
        if not c.streams.audio:
            return np.zeros(0, np.float32)
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for fr in c.decode(c.streams.audio[0]):
            out += [o.to_ndarray().reshape(-1) for o in res.resample(fr)]
        out += [o.to_ndarray().reshape(-1) for o in res.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


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
            audio = decode_audio(b)
            out = []
            for r in g.itertuples():
                wav = audio[int(r.start / FPS * SR):int((r.end + 1) / FPS * SR)]
                if len(wav) >= SR // 2:                                # FLAG_46 kept utterances of >= 0.5 s
                    out.append(((spk, video, r.utt), wav))
            return out, None
        except Exception as e:
            err = f"{spk}/{video}: {type(e).__name__}: {e}"; time.sleep(3 * (attempt + 1))
    return [], err


def save_chunk(path, items):
    """items = [((spk, video, utt), float wav)] -> npz with the centre CROP_SEC as int16."""
    if not items:
        np.savez(path, spk=np.array([], str), video=np.array([], str), utt=np.array([], str), dur=np.zeros(0, np.float32),
                 n=np.zeros(0, np.int32), wav=np.zeros((0, L_CROP), np.int16)); return
    W, N = zip(*[to_i16(w, L_CROP) for _, w in items])
    k = np.array([x for x, _ in items], str)
    np.savez(path, spk=k[:, 0], video=k[:, 1], utt=k[:, 2], dur=np.array([len(w) / SR for _, w in items], np.float32),
             n=np.array(N, np.int32), wav=np.stack(W))


WIRE = {s: [] for s in META}                                         # (key, centre 12 s) for the wiring check, first WIRE_N per source
order = JOB.groupby(["part", "shard"]).size().sort_index()
if SMOKE:
    order = order.iloc[:1]
STATS_VB = dict(shards=0, videos=0, failed=0, rows=0)
for (part, shard), n in order.items():
    zf = OUT / "vb" / f"{part}_{shard}.npz"
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
    for k, w in got:
        if len(WIRE["vb"]) < WIRE_N:
            WIRE["vb"].append((k, centre_crop(w, L_WIRE)))
    tmp = zf.with_suffix(".part.npz"); save_chunk(tmp, got); tmp.replace(zf)
    STATS_VB["shards"] += 1; STATS_VB["videos"] += len(jobs); STATS_VB["failed"] += len(errs); STATS_VB["rows"] += len(got)
    log(f"VoxBlink {shard}: {len(jobs)} videos ({len(errs)} failed), {len(got)}/{len(G)} rows [{time.time() - t0:.0f}s]"
        + (f" | e.g. {errs[0][:120]}" if errs else ""))
log("VoxBlink done this session:", STATS_VB)
'''

VOX = r'''
# ============================================================================ VoxCeleb2 (FLAG_37 + FLAG_53 rows): the same WebDataset tars, centre 3 s
import concurrent.futures as cf
VOICE_URL = "https://huggingface.co/datasets/gaunernst/voxceleb2-dev-wds/resolve/main/voxceleb2-dev-{:04d}.tar"
SRC_OF = {k: "vox" for k in WANT["vox"]}
SRC_OF.update({k: "vox2b" for k in WANT["vox2b"] if k not in SRC_OF})   # a clip in both lists is cached once (as vox)


def decode_m4a(b):
    out = []
    with av.open(io.BytesIO(b)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for fr in c.decode(c.streams.audio[0]):
            out += [o.to_ndarray().reshape(-1) for o in res.resample(fr)]
        out += [o.to_ndarray().reshape(-1) for o in res.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def tar_job(t):
    """Stream tar t once (decoding inside the stream, as FLAG_37); keep the wanted clips. -> (t, [(key, src, wav)], error)"""
    err = None
    for attempt in range(4):
        got = []
        try:
            with requests.get(VOICE_URL.format(t), headers=HDR, timeout=300, stream=True) as r:
                r.raise_for_status(); r.raw.decode_content = True
                tf = tarfile.open(fileobj=r.raw, mode="r|")
                for m in tf:
                    if not m.name.endswith(".m4a"):
                        continue
                    k = tuple(m.name[:-4].split("/")[-3:])
                    if k in SRC_OF:
                        got.append((k, SRC_OF[k], decode_m4a(tf.extractfile(m).read())))
            return t, got, None
        except Exception as e:
            err = f"tar {t}: {type(e).__name__}: {e}"; time.sleep(10 * (attempt + 1))
    return t, [], err


found = set()
for f in sorted((OUT / "vox").glob("tar*.npz")):                    # resume: clips of the tars already done
    Zp = np.load(f)
    found |= set(zip(Zp["spk"], Zp["video"], Zp["utt"]))
tars = [t for t in range(779) if not (OUT / "vox" / f"tar{t:04d}.npz").exists()]
if SMOKE:
    tars = tars[:SMOKE_TARS]
STATS_VOX = dict(tars=0, failed=0, rows=0)
t0 = time.time()
with cf.ThreadPoolExecutor(TAR_THREADS) as ex:
    it = iter(tars)
    pending = {}
    for t in it:
        pending[ex.submit(tar_job, t)] = t
        if len(pending) >= TAR_THREADS:
            break
    while pending:
        done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
        for fut in done:
            pending.pop(fut)
            t, got, err = fut.result()
            if err:
                STATS_VOX["failed"] += 1; log("!!", err, "-> skipped (run again to retry)"); continue
            for k, src, w in got:
                if len(WIRE[src]) < WIRE_N and len(w) >= SR // 2:
                    WIRE[src].append((k, centre_crop(w, L_WIRE)))
            items = [(k, w) for k, src, w in got]
            path = OUT / "vox" / f"tar{t:04d}.npz"; tmp = path.with_suffix(".part.npz")
            save_chunk(tmp, items); tmp.replace(path)
            found |= {k for k, _ in items}
            STATS_VOX["tars"] += 1; STATS_VOX["rows"] += len(items)
            if STATS_VOX["tars"] % 20 == 0 or SMOKE:
                log(f"VoxCeleb2: {STATS_VOX['tars']} tars this session, {len(found)}/{len(SRC_OF)} wanted clips cached [{time.time() - t0:.0f}s]")
        if len(found) >= len(SRC_OF):
            log("every wanted VoxCeleb2 clip is cached"); break
        if not time_left():
            log("time limit reached: no new tar (attach the output and run again)"); continue
        for t in it:
            pending[ex.submit(tar_job, t)] = t
            break
log("VoxCeleb2 done this session:", STATS_VOX, "| cached", len(found), "/", len(SRC_OF))
'''

WIRECELL = r'''
# ============================================================================ wiring check: frozen SimAM on the re-fetched audio vs the stored arrays
for s in WIRE:                                                       # keep the samples (a resumed run reuses them)
    f = OUT / f"wire_{s}.npz"
    if WIRE[s] and not f.exists():
        k = np.array([x for x, _ in WIRE[s]], str)
        W, N = zip(*[to_i16(w, L_WIRE) for _, w in WIRE[s]])
        np.savez(f, spk=k[:, 0], video=k[:, 1], utt=k[:, 2], n=np.array(N, np.int32), wav=np.stack(W))
exec(compile(T3CELL_SRC, "t3enc_cell", "exec"))                       # SimAMVoice, wespeaker_dir, fetch (T3 notebooks' code)
WIRING = {}
encs = {n: SimAMVoice(n) for n in ("sam100pA", "sam100fA")}
for s in ("vb", "vox", "vox2b"):
    f = OUT / f"wire_{s}.npz"
    if not f.exists():
        WIRING[s] = "no sample"; continue
    Z = np.load(f)
    rows = [ROW[s][k] for k in zip(Z["spk"], Z["video"], Z["utt"])]
    res = {}
    for n, arr in (("sam100pA", "voice_sam100p.npy"), ("sam100fA", "voice_sam100f.npy")):
        ref = np.load(ARRS[s] / arr, mmap_mode="r")[rows].astype(np.float32)
        new = np.stack([encs[n](Z["wav"][i, :Z["n"][i]].astype(np.float32) / 32767.0)[""] for i in range(len(rows))])
        cos = (new * ref).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(ref, axis=1) + 1e-8)
        res[n] = dict(rows=len(rows), mean_cos=round(float(cos.mean()), 5), min_cos=round(float(cos.min()), 5))
    WIRING[s] = res
    log(f"wiring {s}: {res}")
ok = all(all(r["mean_cos"] >= 0.99 for r in v.values()) for v in WIRING.values() if isinstance(v, dict))
log("WIRING", "OK" if ok else "!! FAILED: the cache does not hold the clips the stored embeddings came from", json.dumps(WIRING))
'''

DONE = r'''
# ============================================================================ manifest + (optional) upload to a private dataset of this account
cnt = {}
for sub in ("vb", "vox"):
    n = 0
    for f in (OUT / sub).glob("*.npz"):
        n += int(len(np.load(f)["spk"]))
    cnt[sub] = n
size = sum(f.stat().st_size for f in OUT.rglob("*.npz")) / 2 ** 30
man = dict(created=time.strftime("%Y-%m-%d %H:%M"), crop_sec=CROP_SEC, sr=SR, rows=cnt, wanted={s: len(WANT[s]) for s in WANT},
           size_gb=round(size, 2), wiring=WIRING, lid_dropped={s: len(DROP[s]) for s in DROP}, smoke=SMOKE,
           complete=bool(cnt["vb"] >= 0.97 * len(JOB) and cnt["vox"] >= 0.97 * len(SRC_OF)))
(OUT / "audio_manifest.json").write_text(json.dumps(man, indent=1, default=str))
log("manifest:", json.dumps(man, default=str))
UP = os.environ.get("AUDIO_UPLOAD", "")
if UP and not SMOKE and man["complete"] and ok:
    try:
        import kagglehub
        kagglehub.dataset_upload(UP, str(OUT), version_notes="VOICELORA-01 step 0: audio cache (centre 3 s, 16 kHz int16)")
        log("uploaded", OUT, "->", UP)
    except Exception as e:
        log(f"upload to {UP} failed ({e!r}); the cache is in the notebook output")
else:
    log(f"no upload (AUDIO_UPLOAD={UP!r}, smoke={SMOKE}, complete={man['complete']}, wiring ok={ok}): attach the output and run again if incomplete")
'''


def build():
    t3src = "T3CELL_SRC = " + repr(T3CELL) + "\n"
    cells = [("markdown", INTRO), ("code", ENV), ("code", PIP),
             ("markdown", "## 1. Voice rows of each source (LID-dropped speakers excluded)"), ("code", INDEX),
             ("markdown", "## 2. VoxBlink: re-fetch the videos (FLAG_63's shard index), FLAG_46's utterance cut"), ("code", VB),
             ("markdown", "## 3. VoxCeleb2: stream the WebDataset tars"), ("code", VOX),
             ("markdown", "## 4. Wiring check (frozen SimAM vs the stored arrays)"), ("code", t3src + WIRECELL),
             ("markdown", "## 5. Manifest + upload"), ("code", DONE)]
    B12.nb(cells, "FLAG_63v_audio_cache.ipynb")
    print("FLAG_63v_audio_cache.ipynb built")


if __name__ == "__main__":
    build()
