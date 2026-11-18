"""Generate FLAG_46_voxblink.ipynb (V2-a of docs/PLAN_KAGGLE_2026-10-08.md §6): VoxBlink-clean identities in the Track 3
encoder space (FaRL ep16, CLIP ViT-L/14, SimAM-ResNet100 pretrained / fine-tuned) + spoken-language ID of every clip, with
Bengali speakers removed (metadata before extraction, LID filter locally). Rows: per speaker up to V_MAX videos x U_MAX
utterances; face = middle frame of the utterance, loose crop 1.3 x the official face box (MAV-Celeb framing), 224; voice =
the utterance (official frame range at 25 fps), 12 s centre crop. Resumable per shard.
    python build_vb_notebook.py
    FLAG_LOCAL=1 FLAG_OUT=<dir> FLAG_CACHE=<cache> VB_SMOKE=1 FLAG_T3_PATH=<isolated site dir> \
        python run_notebook_local.py FLAG_46_voxblink.ipynb --jupyter
"""
from pathlib import Path
import build_eval_notebooks as B12
import build_t3enc_notebook as T3

HERE = Path(__file__).resolve().parent
CELL = (HERE / "vendor_t3enc/t3enc_cell.py").read_text(encoding="utf-8")
MEDIA = (HERE / "vendor_vb/vb_media.py").read_text(encoding="utf-8")

INTRO = """
# FLAG 2027 — NB-46: VoxBlink-clean identities (Bengali removed) in the Track 3 encoder space

**Why.** The lever that separates the FAME 2026 top-2 from the rest, and our only gate pass (T3ENC-01), is the number of
training identities. A teammate's system trained on ~15,000 VoxBlink identities reaches CB Bangla 14 / 24; ours (4,877
VoxCeleb2) 17.64 / 27.25. This notebook adds the VoxBlink-clean identities to our training pool, in exactly the encoder
space of T3ENC-01, so the local experiment SCALE-VB can train the head on MAV-Celeb + VoxCeleb2 + VoxBlink.

| step | what |
|---|---|
| speakers | VoxBlink **clean** subset present in the mirror (~15,250); **excluded before anything is downloaded**: `spk2lan = bengali` or `spk2loc = Bangladesh` |
| rows | per speaker up to **3 videos x 2 utterances** (the 2 longest clean utterances >= 2 s of each video; videos in a fixed pseudo-random order) |
| face | middle frame of the utterance, **loose crop 1.3 x** the official face box (MAV-Celeb framing, checked by eye), 224 x 224 -> `farl16`, `clipL` |
| voice | the utterance (official frame range, 25 fps), 12 s centre crop -> `sam100p`, `sam100f`, **language ID** (VoxLingua107, 107 probabilities) |

**Bengali:** speakers labelled Bengali / Bangladesh are never downloaded; every extracted clip is language-identified and the
registered speaker-level rule (drop every clip of a speaker with any clip whose top language is Bengali or p(Bengali) >= 0.2) is
applied locally before any training. LID only filters training data, never in the scoring system.

**Source and licence:** media = the unofficial Hugging Face crawl `adollahamini1998/VoxBlink-Video-clean-8-{0..7}` (YouTube
shorts); annotations = the official VoxBlink release (copy in `adollahamini1998/ScriptsForVoxBlinkResources`). VoxBlink is CC
BY-NC-SA 4.0 and its authors forbid redistribution: research use inside the challenge only, **keep the output dataset private**,
declare the source and the Bengali filter in the system description. No MAV-Celeb identity is compared with VoxBlink.

**Settings:** GPU T4, **Internet ON**. No input needed (attach this notebook's previous output to resume).
**Time:** about 5-7 h for all speakers (shard listing ~30 min, timestamps ~15 min, ~45k shorts fetched and decoded, four
encoders + LID). A session stops starting new shards after `TIME_LIMIT_H`; Save Version, attach the output, run again.

**Output:** `/kaggle/working/feats_vb/` -> dataset **`flag2027-feats-vb`** (private, about 0.6 GB float16).
"""

ENV = B12.ENV12.replace('OUT = WORKDIR / "feats_eval"', 'OUT = WORKDIR / "feats_vb"')
assert ENV != B12.ENV12
PIP = T3.PIP.replace('!pip -q install ftfy regex kaldiio silero-vad', '!pip -q install -U "speechbrain>=1.1.1" av ftfy regex kaldiio silero-vad')
assert PIP != T3.PIP

CONFIG = r'''
# ============================================================================ settings, VoxBlink annotations, speaker / utterance selection
import hashlib, requests, threading, tarfile
import concurrent.futures as cf
V_MAX, U_MAX, MIN_DUR, MIN_DUR2 = 3, 2, 2.0, 1.0     # videos per speaker, utterances per video, preferred / fallback min duration (s)
CROP_SCALE, FPS, CLIP_SEC = 1.3, 25, 12.0
TIME_LIMIT_H = float(os.environ.get("VB_TIME_LIMIT_H", "10.5"))
LIST_THREADS, FETCH_THREADS = 24, 12
SMOKE = os.environ.get("VB_SMOKE") == "1" and LOCAL
RES = "https://huggingface.co/datasets/adollahamini1998/ScriptsForVoxBlinkResources/resolve/main/"
PROG = "https://huggingface.co/datasets/adollahamini1998/VoxBlinkProgressVideos/resolve/main/progress/download_progress_clean_8_{}.json"
TREE = "https://huggingface.co/api/datasets/adollahamini1998/VoxBlink-Video-clean-8-{}/tree/main"
SHARD = "https://huggingface.co/datasets/adollahamini1998/VoxBlink-Video-clean-8-{}/resolve/main/{}"
T_START = time.time()
PARTS_DIR = OUT / "parts"; PARTS_DIR.mkdir(parents=True, exist_ok=True)
for prev in (IN.rglob("feats_vb") if IN.exists() and not LOCAL else []):          # resume: earlier output attached
    for f in list((prev / "parts").glob("*.npz")) + [prev / n for n in ("shard_listing.csv", "timestamps.csv", "selection.csv")]:
        dst = (PARTS_DIR / f.name) if f.parent.name == "parts" else (OUT / f.name)
        if f.exists() and not dst.exists():
            shutil.copy(f, dst)


def get_retry(url, tries=5, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=300, **kw); r.raise_for_status(); return r
        except Exception as e:
            log(f"  retry {k + 1}/{tries} {url.rsplit('/', 1)[-1][:60]}: {type(e).__name__}: {e}"); time.sleep(5 * (k + 1))
    raise RuntimeError(f"cannot fetch {url}")


def cached(name, url):
    f = CACHE / name
    if not f.exists():
        f.write_bytes(get_retry(url).content)
    return f


meta = {}
for k in ("spk2lan", "spk2loc", "spk2gender"):
    rows = [l.split() for l in cached(k, RES + f"meta/{k}").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta[k] = {r[0]: " ".join(r[1:2]) if k != "spk2loc" else " ".join(r[1:]) for r in rows}
dur = {}
for l in cached("utt2dur", RES + "meta/utt2dur").read_text().splitlines():
    p = l.split()
    if len(p) == 2:
        dur[p[0]] = float(p[1])
clean = set(cached("utt_clean.txt", RES + "data/utt_clean.txt").read_text().split())
split_utt = lambda u: (u[:7], u[8:-6], u[-5:])                  # idXXXXX - <youtube id, may contain '-'> - NNNNN
log(f"annotations: {len(meta['spk2lan'])} speakers with a language, {len(meta['spk2loc'])} with a location, "
    f"{len(dur)} utterances with a duration, {len(clean)} clean utterances")

# ---- mirror: per part, the uploaded videos and their shard. A shard's name carries its creation time (unix s); each distinct
# uploaded_at of the progress file belongs to the newest shard created before it (gap 1-77 s on every part, checked 09/10).
# The mirror keeps changing: part 7 has one shard no upload points to (an orphan, skipped), so counts need not match.
import bisect
from datetime import datetime
vid_shard, SHARDS, ORPHANS = {}, {}, []
for part in (range(1) if SMOKE else range(8)):
    prog = json.loads(cached(f"prog_{part}.json", PROG.format(part)).read_text())["completed"]
    tree = get_retry(TREE.format(part)).json()
    shards = sorted((int(x["path"].split("_")[-1][:-4]), x["path"], x.get("size", 0)) for x in tree if x["path"].endswith(".tar"))
    ups = sorted({v["uploaded_at"] for s in prog.values() for v in s["videos"].values() if v.get("status") == "uploaded"})
    ts = [t for t, _, _ in shards]
    up2shard, used = {}, set()
    for u in ups:
        i = bisect.bisect_right(ts, datetime.fromisoformat(u).timestamp()) - 1
        assert i >= 0, (part, u, "upload older than every shard")
        gap = datetime.fromisoformat(u).timestamp() - ts[i]
        assert gap <= 900, (part, u, shards[i][1], f"gap {gap:.0f} s")          # an upload follows its shard by seconds
        up2shard[u] = (part, shards[i][1]); used.add(i)
    assert len(used) == len(ups), (part, "two uploads mapped to one shard", len(used), len(ups))
    ORPHANS += [(part, shards[i][1]) for i in range(len(shards)) if i not in used]
    for i, (_, name, size) in enumerate(shards):
        if i in used:
            SHARDS[(part, name)] = size
    for spk, s in prog.items():
        for vid, v in s["videos"].items():
            if v.get("status") == "uploaded":
                vid_shard[(spk, vid)] = up2shard[v["uploaded_at"]]
spk_all = sorted({s for s, _ in vid_shard})
bad = {s for s in spk_all if meta["spk2lan"].get(s, "").lower() == "bengali" or "bangladesh" in meta["spk2loc"].get(s, "").lower()}
log(f"mirror: {len(vid_shard)} uploaded videos of {len(spk_all)} speakers in {len(SHARDS)} shards | excluded before download "
    f"(spk2lan bengali / spk2loc Bangladesh): {len(bad)} speakers | orphan shards skipped (no upload points to them): {ORPHANS}")
if SMOKE:                                                        # local smoke test: the speakers of the smallest shard of part 0
    small = min((k for k in SHARDS if k[0] == 0), key=lambda k: SHARDS[k])
    keep = sorted({s for (s, v), sh in vid_shard.items() if sh == small})[:3]
    vid_shard = {k: v for k, v in vid_shard.items() if k[0] in keep and v == small}
    log(f"smoke test: shard {small[1]} ({SHARDS[small] / 1e6:.0f} MB), speakers {keep}")

# ---- selection (deterministic): per speaker V_MAX videos in md5 order, per video the U_MAX longest clean utterances
SEL_F = OUT / "selection.csv"
if SEL_F.exists() and not SMOKE:
    SEL = pd.read_csv(SEL_F, dtype={"spk": str, "video": str, "utt": str})
else:
    by_vid = {}
    for u in clean:
        spk, vid, n = split_utt(u)
        if (spk, vid) in vid_shard and spk not in bad:
            by_vid.setdefault((spk, vid), []).append((dur.get(u, 0.0), n))
    rows, vids_of = [], {}
    for (s, v) in by_vid:
        vids_of.setdefault(s, []).append(v)
    for spk in sorted(vids_of):
        vids = sorted(vids_of[spk], key=lambda v: hashlib.md5(f"{spk}/{v}".encode()).hexdigest())
        taken = 0
        for vid in vids:
            cand = sorted(by_vid[(spk, vid)], reverse=True)
            pick = [c for c in cand if c[0] >= MIN_DUR][:U_MAX] or [c for c in cand if c[0] >= MIN_DUR2][:1]
            if not pick:
                continue
            part, shard = vid_shard[(spk, vid)]
            rows += [dict(spk=spk, video=vid, utt=n, dur=d, part=part, shard=shard) for d, n in pick]
            taken += 1
            if taken >= V_MAX:
                break
    SEL = pd.DataFrame(rows)
    if not SMOKE:
        SEL.to_csv(SEL_F, index=False)
log(f"selection: {len(SEL)} utterances, {SEL.groupby(['spk', 'video']).ngroups} videos, {SEL.spk.nunique()} speakers, "
    f"{SEL.shard.nunique()} shards | median duration {SEL.dur.median():.1f} s")

# ---- timestamps of the selected utterances (one pass over timestamp.tar.gz: frame range + face box at the middle frame)
TS_F = OUT / "timestamps.csv"
if TS_F.exists() and not SMOKE:
    TS = pd.read_csv(TS_F, dtype={"spk": str, "video": str, "utt": str})
else:
    want = {(r.spk, r.video, r.utt) for r in SEL.itertuples()}
    got, t0 = [], time.time()
    with get_retry(RES + "timestamp.tar.gz", stream=True) as r:
        tf = tarfile.open(fileobj=r.raw, mode="r|gz")
        for m in tf:
            if not m.isfile():
                continue
            p = m.name.split("/")
            key = (p[-3], p[-2], p[-1][:-4]) if len(p) >= 3 else None
            if key in want:
                fr, bx = parse_timestamp(tf.extractfile(m).read().decode("utf8", "replace"))
                if len(fr):
                    j = len(fr) // 2
                    got.append(dict(spk=key[0], video=key[1], utt=key[2], start=int(fr[0]), end=int(fr[-1]), mid=int(fr[j]),
                                    x=int(bx[j, 0]), y=int(bx[j, 1]), w=int(bx[j, 2]), h=int(bx[j, 3])))
                if len(got) >= len(want):
                    break
    TS = pd.DataFrame(got)
    if not SMOKE:
        TS.to_csv(TS_F, index=False)
    log(f"timestamps: {len(TS)}/{len(want)} selected utterances found [{time.time() - t0:.0f}s]")
JOB = SEL.merge(TS, on=["spk", "video", "utt"], how="inner")
log(f"jobs: {len(JOB)} utterances with a timestamp, {JOB.spk.nunique()} speakers")
'''

LISTING = r'''
# ============================================================================ member offsets of the needed shards (header range reads, cached)
LIST_F = OUT / "shard_listing.csv"
LST = pd.read_csv(LIST_F) if LIST_F.exists() else pd.DataFrame(columns=["part", "shard", "member", "off", "size"])
done_shards = set(zip(LST.part, LST.shard))
need = sorted(set(zip(JOB.part, JOB.shard)) - done_shards)
_TL = threading.local()


def cdn_url(part, shard):
    with requests.get(SHARD.format(part, shard), headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=120, stream=True) as r:
        return r.url


def list_one(ps):
    part, shard = ps
    if not hasattr(_TL, "s"):
        _TL.s = requests.Session()
    for k in range(4):
        try:
            mem = tar_list_remote(_TL.s, cdn_url(part, shard))
            return [dict(part=part, shard=shard, member=n, off=o, size=z) for n, o, z in mem]
        except Exception as e:
            log(f"  listing retry {k + 1} {shard}: {type(e).__name__}: {e}"); time.sleep(5 * (k + 1))
    return []


if need:
    t0 = time.time(); rows = []
    with cf.ThreadPoolExecutor(LIST_THREADS) as ex:
        for i, res in enumerate(ex.map(list_one, need)):
            rows += res
            if (i + 1) % 50 == 0 or i + 1 == len(need):
                log(f"  listed {i + 1}/{len(need)} shards, {len(rows)} members [{time.time() - t0:.0f}s]")
                if not SMOKE:
                    pd.concat([LST, pd.DataFrame(rows)], ignore_index=True).to_csv(LIST_F, index=False)
    LST = pd.concat([LST, pd.DataFrame(rows)], ignore_index=True)
    if not SMOKE:
        LST.to_csv(LIST_F, index=False)
LST["spk"], LST["video"] = LST.member.str.split("/").str[0], LST.member.str.split("/").str[1].str[:-4]
JOB = JOB.merge(LST[["spk", "video", "off", "size"]], on=["spk", "video"], how="inner")
log(f"jobs with a member offset: {len(JOB)} utterances, {JOB.groupby(['spk', 'video']).ngroups} videos")
'''

EXTRACT = r'''
# ============================================================================ per shard: fetch the selected shorts, cut, crop, encode (resumable)
from speechbrain.inference.classifiers import EncoderClassifier
try:
    from speechbrain.utils.fetching import LocalStrategy
    _ls = {"local_strategy": LocalStrategy.COPY}
except ImportError:
    _ls = {}
LIDM = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=str(CACHE / "lid_voxlingua107"),
                                      run_opts={"device": str(DEV)}, **_ls)
LIDM.eval()
LABELS = [LIDM.hparams.label_encoder.ind2lab[i] for i in range(len(LIDM.hparams.label_encoder.ind2lab))]
BN = [i for i, l in enumerate(LABELS) if l.split(":")[0].strip() == "bn"]
assert len(LABELS) == 107 and len(BN) == 1


@torch.no_grad()
def lid_probs(w):
    out_prob = LIDM.classify_batch(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))[None].to(DEV))[0]
    return torch.softmax(out_prob.float().reshape(1, -1), 1)[0].cpu().numpy()


ENC = None
ENC_LIST = ["farl16V", "clipLV", "sam100pA", "sam100fA"]
if LOCAL and os.environ.get("VB_ENCODERS"):                 # local smoke test on a small-RAM PC only: a subset of the encoders
    ENC_LIST = os.environ["VB_ENCODERS"].split(",")
L_CLIP = int(CLIP_SEC * TARGET_SR)
STATS = {"videos": 0, "videos_failed": 0, "faces": 0, "faces_missing_frame": 0, "voices": 0, "voices_short": 0}


def fetch_decode(job):
    """One short: range-fetch the member, decode the wanted middle frames + the audio, cut / crop every selected utterance."""
    (part, shard, spk, video), g = job
    if not hasattr(_TL, "s"):
        _TL.s = requests.Session()
    try:
        url = _TL.urls.get((part, shard)) if hasattr(_TL, "urls") else None
        if url is None:
            _TL.urls = getattr(_TL, "urls", {}); url = _TL.urls[(part, shard)] = cdn_url(part, shard)
        off, size = int(g.off.iloc[0]), int(g["size"].iloc[0])
        b = _TL.s.get(url, headers={"Range": f"bytes={off}-{off + size - 1}"}, timeout=300).content
        if len(b) != size:
            _TL.urls.pop((part, shard), None); raise IOError(f"short read {len(b)} / {size}")
        frames, audio = decode_video(b, g.mid.tolist(), TARGET_SR)
    except Exception as e:
        return None, f"{spk}/{video}: {type(e).__name__}: {e}"
    out = []
    for r in g.itertuples():
        face = inside = None
        if r.mid in frames:
            face, inside = loose_crop(frames[r.mid], (r.x, r.y, r.w, r.h), CROP_SCALE)
        a0, a1 = int(r.start / FPS * TARGET_SR), int((r.end + 1) / FPS * TARGET_SR)
        wav = audio[a0:a1]
        out.append(dict(spk=spk, video=video, utt=r.utt, face=face, inside=inside,
                        wav=centre_crop(wav, L_CLIP) if len(wav) >= TARGET_SR // 2 else None, dur=len(wav) / TARGET_SR))
    return out, None


order = JOB.groupby(["part", "shard"]).size().sort_index()
log(f"shards to process: {len(order)} ({sum(1 for k in order.index if (PARTS_DIR / ('%d_%s.npz' % k)).exists())} already done)")
for (part, shard), n in order.items():
    pf = PARTS_DIR / f"{part}_{shard}.npz"
    if pf.exists():
        continue
    if time.time() - T_START > TIME_LIMIT_H * 3600:
        log(f"time limit {TIME_LIMIT_H} h reached: stopping before {shard} (attach the output and run again)"); break
    if ENC is None:
        ENC = {k: build(k) for k in ENC_LIST}
    t0 = time.time()
    G = JOB[(JOB.part == part) & (JOB.shard == shard)]
    jobs = list(G.groupby(["part", "shard", "spk", "video"]))
    rows, errs = [], []
    with cf.ThreadPoolExecutor(FETCH_THREADS) as ex:
        for res, err in ex.map(fetch_decode, jobs):
            STATS["videos"] += 1
            if err:
                errs.append(err); STATS["videos_failed"] += 1; BROKEN["audio"].append(err)
            else:
                rows += res
    F_ = [r for r in rows if r["face"] is not None]; V_ = [r for r in rows if r["wav"] is not None]
    STATS["faces"] += len(F_); STATS["faces_missing_frame"] += len(rows) - len(F_)
    STATS["voices"] += len(V_); STATS["voices_short"] += len(rows) - len(V_)
    R = {}
    for n_, k in (("farl16", "farl16V"), ("clipL", "clipLV")):
        if k in ENC:
            R[n_] = np.concatenate([ENC[k](torch.stack([ENC[k].prep(r["face"]) for r in F_[s:s + 64]]))[""] for s in range(0, len(F_), 64)]) if F_ else np.zeros((0, 1))
    for n_, k in (("sam100p", "sam100pA"), ("sam100f", "sam100fA")):
        if k in ENC:
            R[n_] = np.stack([ENC[k](r["wav"])[""] for r in V_]) if V_ else np.zeros((0, 1))
    R["lid"] = np.stack([lid_probs(r["wav"]) for r in V_]) if V_ else np.zeros((0, 107))
    np.savez(pf, f_spk=np.array([r["spk"] for r in F_], str), f_video=np.array([r["video"] for r in F_], str),
             f_utt=np.array([r["utt"] for r in F_], str), f_inside=np.array([r["inside"] for r in F_], np.float32),
             v_spk=np.array([r["spk"] for r in V_], str), v_video=np.array([r["video"] for r in V_], str),
             v_utt=np.array([r["utt"] for r in V_], str), v_dur=np.array([r["dur"] for r in V_], np.float32),
             **{k: v.astype(np.float16) for k, v in R.items()})
    log(f"{shard}: {len(jobs)} videos ({len(errs)} failed), {len(F_)} faces, {len(V_)} voices [{time.time() - t0:.0f}s] | "
        f"total {time.time() - T_START:.0f}s")
ENC = None
release()
'''

ASSEMBLE = r'''
# ============================================================================ assemble, LID report (registered rule), crop stats, probe, manifest
Zs = [np.load(p) for p in sorted(PARTS_DIR.glob("*.npz"))]
cat = lambda k: np.concatenate([z[k] for z in Zs if len(z[k])]) if any(len(z[k]) for z in Zs) else np.zeros(0)
FM_ = pd.DataFrame({"spk": cat("f_spk"), "video": cat("f_video"), "utt": cat("f_utt"), "inside": cat("f_inside")})
VM_ = pd.DataFrame({"spk": cat("v_spk"), "video": cat("v_video"), "utt": cat("v_utt"), "dur": cat("v_dur")})
for n_ in ("farl16", "clipL"):
    if len(FM_) and n_ in Zs[0].files:
        np.save(OUT / f"face_{n_}.npy", cat(n_))
for n_ in ("sam100p", "sam100f"):
    if len(VM_) and n_ in Zs[0].files:
        np.save(OUT / f"voice_{n_}.npy", cat(n_))
summary = {}
if len(VM_):
    P = cat("lid").astype(np.float32); np.save(OUT / "voice_lid_probs.npy", P.astype(np.float16))
    top, pbn = P.argmax(1), P[:, BN[0]]
    flag = (top == BN[0]) | (pbn >= 0.2)
    drop = sorted(set(VM_.spk.values[flag]))
    summary = {"clips": int(len(VM_)), "clips_top1_bengali": int((top == BN[0]).sum()), "clips_pbn_ge_0.2": int((pbn >= 0.2).sum()),
               "speakers": int(VM_.spk.nunique()), "speakers_flagged": len(drop), "clips_of_flagged": int(VM_.spk.isin(drop).sum()),
               "top_languages": pd.Series([LABELS[i].split(":")[0] for i in top]).value_counts().head(15).to_dict()}
    (OUT / "lid_summary.json").write_text(json.dumps({**summary, "speakers_dropped_ids": drop,
                                                      "rule": "drop every clip of a speaker with any clip whose top language is Bengali or p(Bengali) >= 0.2"}, indent=1))
for M_ in (FM_, VM_):
    M_["lan"] = M_.spk.map(meta["spk2lan"]); M_["loc"] = M_.spk.map(meta["spk2loc"]); M_["gender"] = M_.spk.map(meta["spk2gender"])
FM_.to_csv(OUT / "face_meta.csv", index=False); VM_.to_csv(OUT / "voice_meta.csv", index=False)
(OUT / "lid_labels.json").write_text(json.dumps(LABELS, indent=0))

from sklearn.metrics import roc_curve
def eer(s, y):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return round(100 * float(fpr[i] + fnr[i]) / 2, 2)


PROBE, rng = {}, np.random.RandomState(0)
for side, M_, stem in (("face_clipL", FM_, "face_clipL"), ("face_farl16", FM_, "face_farl16"), ("voice_sam100f", VM_, "voice_sam100f")):
    f = OUT / f"{stem}.npy"
    if not f.exists() or M_.spk.nunique() < 3:
        continue
    X = np.load(f).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    by = {s: np.where(M_.spk.values == s)[0] for s in M_.spk.unique()}
    sp = [s for s, ix in by.items() if len(set(M_.video.values[ix])) > 1]
    if len(sp) < 3:
        continue
    pos, neg = [], []
    for s in rng.choice(sp, min(3000, 10 * len(sp))):
        ix = by[s]; a = rng.choice(ix); other = [j for j in ix if M_.video.values[j] != M_.video.values[a]]
        if other:
            pos.append((a, rng.choice(other)))
        o = sp[rng.randint(len(sp))]
        if o != s:
            neg.append((a, rng.choice(by[o])))
    pr = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    PROBE[side + " (cross-video)"] = eer(np.sum(X[pr[:, 0]] * X[pr[:, 1]], 1), y)
man = {"created": time.strftime("%Y-%m-%d %H:%M"), "speakers_faces": int(FM_.spk.nunique()), "speakers_voices": int(VM_.spk.nunique()),
       "faces": len(FM_), "voices": len(VM_), "excluded_before_download": len(bad), "stats": STATS, "lid": summary,
       "crop": {"scale": CROP_SCALE, "mean_inside": float(FM_.inside.mean()) if len(FM_) else None,
                "share_inside_lt_0.6": float((FM_.inside < 0.6).mean()) if len(FM_) else None},
       "settings": dict(V_MAX=V_MAX, U_MAX=U_MAX, MIN_DUR=MIN_DUR, MIN_DUR2=MIN_DUR2, CLIP_SEC=CLIP_SEC), "probe": PROBE,
       "shards_done": len(Zs), "shards_needed": int(JOB.groupby(["part", "shard"]).ngroups),
       "encoders": {"farl16": FARL16_URL, "clipL": f"{CLIPL_ID}@{CLIPL_REV}", "sam100p": SAM_ZIPS["sam100pA"][0],
                    "sam100f": SAM_ZIPS["sam100fA"][0], "lid": "speechbrain/lang-id-voxlingua107-ecapa"},
       "source": "adollahamini1998/VoxBlink-Video-clean-8-{0..7} (unofficial crawl) + official VoxBlink annotations; CC BY-NC-SA 4.0, no redistribution",
       "broken": BROKEN, "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
(OUT / "feats_manifest.json").write_text(json.dumps(man, indent=1, default=str))
print(json.dumps({k: v for k, v in man.items() if k != "broken"}, indent=1, default=str))
print("NEXT: Save Version -> Output -> New Dataset -> name it flag2027-feats-vb (PRIVATE). If shards_done < shards_needed, "
      "attach the output and run again: finished shards are skipped.")
'''

if __name__ == "__main__":
    B12.nb([("markdown", INTRO), ("code", ENV), ("code", PIP),
            ("markdown", "## 1. Audio / image helpers (FLAG_12 cell)"), ("code", B12.AUDIO12),
            ("markdown", "## 2. The four Track 3 encoders (same cell as FLAG_40 / 41 / 43)"), ("code", CELL),
            ("markdown", "## 3. VoxBlink media helpers (tar index by range reads, timestamps, loose crop, decoding)"), ("code", MEDIA),
            ("markdown", "## 4. Annotations, mirror map, selection, timestamps"), ("code", CONFIG),
            ("markdown", "## 5. Member offsets of the needed shards (cached)"), ("code", LISTING),
            ("markdown", "## 6. Fetch, cut, crop, encode, language-ID; resumable per shard"), ("code", EXTRACT),
            ("markdown", "## 7. Assemble, LID report, crop stats, probe, manifest"), ("code", ASSEMBLE)], "FLAG_46_voxblink.ipynb")
