# ============================================================================ SCALE-ID-01: the VoxCeleb2 speakers our T3 head never saw
# FLAG_37 read faces from ProgramComputer/voxceleb vox2_mp4_{2..6}.zip: vox2_mp4_1.zip is a TRUNCATED upload (27.0 GB where the
# others hold ~50 GB; 244 GB in all vs 267 GB for the official archive), so its central directory is missing (BadZipFile) and
# ~1,117 dev speakers were lost. The same mirror holds the OFFICIAL split archives (vox2_dev_mp4_parta[a-i], 267 GB, and
# vox2_dev_aac_parta[a-h], 77 GB): byte chunks of one zip64 file whose central directory sits in the last part (checked
# 10/10: 1,243,744 / 1,243,573 entries). Every member is read by HTTP range on the right part(s); nothing is downloaded whole.
# Two additions (column `kind`):
#   new   VoxCeleb2 dev speakers absent from FLAG_37's key list + the 120 VoxCeleb2 test speakers, sampled with FLAG_37's rule:
#         NF = 6 face tracks (first track of 6 videos spread over the speaker's videos, middle frame of the 224 x 224 track),
#         NV = 8 voice clips (<= 2 per video, 12 s centre crop)
#   more  FLAG_37's speakers: 6 MORE_IDS face tracks (first tracks of videos FLAG_37 did not use, then second tracks of used ones)
#         and 8 MORE_IDS voice clips (not FLAG_37's; <= 2 per video counting FLAG_37's clips) = FLAG_43's design
# Encoders = the T3 cell (FaRL16,
# CLIP-L, SimAM100 p / ft) + VoxLingua107 LID (FLAG_42) with the registered Bengali rule (drop every clip of a speaker with any
# clip of top-1 Bengali or p(bn) >= 0.2). Resumable per chunk (attach the previous output).
import requests, struct, zlib, threading
import concurrent.futures as cf

HF = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/"
SOURCES = {"dev_mp4": [f"vox2_dev_mp4_parta{c}" for c in "abcdefghi"], "dev_aac": [f"vox2_dev_aac_parta{c}" for c in "abcdefgh"],
           "test_mp4": ["vox2_test_mp4.zip"], "test_aac": ["vox2_test_aac.zip"]}
NF, NV, PER_VIDEO, CLIP_SEC, THREADS = 6, 8, 2, 12.0, 16
VB = WORK / "feats_vox2b"
FP_, VP_ = VB / "parts_face", VB / "parts_voice"
for d in (VB, FP_, VP_):
    d.mkdir(parents=True, exist_ok=True)
SMOKE_IDS = int(os.environ.get("VOX_SMOKE_IDS", "0")) if (LOCAL or os.environ.get("FLAG_SMOKE") == "1") else 0   # FLAG_SMOKE: Kaggle CPU smoke (kapi --smoke)
for prev in ([] if LOCAL else IN.rglob("feats_vox2b")):          # resume: chunks and listings of an earlier run
    if VB in prev.parents or prev == VB:
        continue
    for f in list(prev.glob("parts_*/*.npz")) + list(prev.glob("listing_*.parquet")):
        tgt = VB / f.relative_to(prev)
        if not tgt.exists():
            tgt.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, tgt)


class Remote:
    """A remote zip made of byte-chunk parts (or one part), read by absolute offset with HTTP range requests."""
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
        """n bytes at absolute offset off (may span parts)."""
        out, end = [], off + n
        while off < end:
            j = int(np.searchsorted(self.starts, off, side="right") - 1)
            a = off - self.starts[j]; b = min(end, self.starts[j + 1]) - self.starts[j] - 1
            out.append(self._get(j, int(a), int(b))); off = int(self.starts[j] + b + 1)
        return b"".join(out)

    def listing(self):
        """Central directory -> DataFrame(name, off, csize, method) of every file member."""
        total = int(self.starts[-1])
        tail = self.read(total - 65536, 65536)
        i = tail.rfind(b"PK\x06\x06")
        if i >= 0:
            cs, co = struct.unpack("<QQ", tail[i + 40:i + 56])
        else:
            i = tail.rfind(b"PK\x05\x06"); cs, co = struct.unpack("<II", tail[i + 12:i + 20])
        cd = self.read(co, cs)
        rows, p = [], 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            method, = struct.unpack_from("<H", cd, p + 10)
            csz, usz = struct.unpack_from("<II", cd, p + 20)
            n, m, k = struct.unpack_from("<HHH", cd, p + 28)
            off, = struct.unpack_from("<I", cd, p + 42)
            name = cd[p + 46:p + 46 + n].decode("utf-8", "replace")
            ex = cd[p + 46 + n:p + 46 + n + m]
            if 0xFFFFFFFF in (csz, usz, off):                          # zip64 extra field: usize, csize, offset (only those set)
                q = 0
                while q + 4 <= len(ex):
                    hid, hsz = struct.unpack_from("<HH", ex, q)
                    if hid == 1:
                        vals, r_ = [], q + 4
                        for v in (usz, csz, off):
                            if v == 0xFFFFFFFF:
                                vals.append(struct.unpack_from("<Q", ex, r_)[0]); r_ += 8
                            else:
                                vals.append(v)
                        usz, csz, off = vals
                        break
                    q += 4 + hsz
            if not name.endswith("/"):
                rows.append((name, off, csz, method))
            p += 46 + n + m + k
        return pd.DataFrame(rows, columns=["name", "off", "csize", "method"])

    def member(self, off, csize, method):
        h = self.read(off, 30)
        assert h[:4] == b"PK\x03\x04", "bad local header"
        n, e = struct.unpack("<HH", h[26:30])
        d = self.read(off + 30 + n + e, csize)
        return zlib.decompress(d, -15) if method == 8 else d


REM, LST = {}, {}
for key, names in SOURCES.items():
    REM[key] = Remote(names)
    f = VB / f"listing_{key}.parquet"
    if f.exists():
        L_ = pd.read_parquet(f)
    else:
        t0 = time.time(); L_ = REM[key].listing()
        ext = ".mp4" if key.endswith("mp4") else ".m4a"
        L_ = L_[L_.name.str.endswith(ext)].reset_index(drop=True)
        sp = L_.name.str.split("/")
        L_["spk"], L_["video"], L_["utt"] = sp.str[-3], sp.str[-2], sp.str[-1].str[:-4]
        L_.to_parquet(f)
        log(f"{key}: central directory {len(L_)} members [{time.time() - t0:.0f}s]")
    L_["src"] = key
    LST[key] = L_

# ---------------------------------------------------------------- identities and samples
meta = pd.read_csv(io.BytesIO(requests.get(HF + "vox2_meta.csv", timeout=120).content), dtype=str)
meta.columns = [c.strip().lstrip("﻿") for c in meta.columns]
meta = meta.apply(lambda c: c.str.strip())
kroot = next(r / "Experiment/VOX2-01/feats_vox2" for r in ROOTS if (r / "Experiment/VOX2-01/feats_vox2/face_meta.csv").exists())
OF = pd.read_csv(kroot / "face_meta.csv", dtype=str); OV = pd.read_csv(kroot / "voice_meta.csv", dtype={"spk": str, "video": str, "utt": str})
OLD = sorted(set(OF.spk))
FL = pd.concat([LST["dev_mp4"], LST["test_mp4"]], ignore_index=True).sort_values(["spk", "video", "utt"])
AL = pd.concat([LST["dev_aac"], LST["test_aac"]], ignore_index=True).sort_values(["spk", "video", "utt"])
NEW_IDS = sorted((set(meta[meta.Set == "dev"]["VoxCeleb2 ID"]) - set(OLD)) | set(meta[meta.Set == "test"]["VoxCeleb2 ID"]))
NEW_IDS = sorted(set(NEW_IDS) & set(FL.spk) & set(AL.spk))
MORE_IDS = sorted(set(OLD) & set(FL.spk) & set(AL.spk))
if SMOKE_IDS:
    NEW_IDS, MORE_IDS = NEW_IDS[:SMOKE_IDS], MORE_IDS[:SMOKE_IDS]
log(f"identities: new {len(NEW_IDS)} (VoxCeleb2 dev outside FLAG_37's {len(OLD)} + test), more samples for {len(MORE_IDS)}")


def pick(n, k):
    return list(range(n)) if n <= k else [int(round(x)) for x in np.linspace(0, n - 1, k)]


used_fv = set(zip(OF.spk, OF.video)); used_f = set(zip(OF.spk, OF.video, OF.utt)); used_v = set(zip(OV.spk, OV.video, OV.utt))
vcount = OV.groupby(["spk", "video"]).size().to_dict()
FSEL, VSEL = [], []
fl_new, fl_more = FL[FL.spk.isin(set(NEW_IDS))], FL[FL.spk.isin(set(MORE_IDS))]
for spk, g in fl_new.groupby("spk", sort=True):                   # new: FLAG_37's rule
    first = g.drop_duplicates("video")
    FSEL += [dict(first.iloc[i], kind="new") for i in pick(len(first), NF)]
for spk, g in fl_more.groupby("spk", sort=True):                  # more: unused videos first, then second tracks of used ones
    g = g.reset_index(drop=True)
    firstmask = ~g.duplicated("video").values
    first = g[firstmask]
    unused = first[[(spk, v) not in used_fv for v in first.video]]
    cand = [dict(unused.iloc[i], kind="more") for i in pick(len(unused), NF)]
    if len(cand) < NF:
        rest = g[~firstmask]
        rest = rest[[(spk, v, u) not in used_f for v, u in zip(rest.video, rest.utt)]].drop_duplicates("video")
        cand += [dict(r, kind="more") for _, r in rest.head(NF - len(cand)).iterrows()]
    FSEL += cand
FSEL = pd.DataFrame(FSEL).reset_index(drop=True)
for kind, ids in (("new", NEW_IDS), ("more", MORE_IDS)):
    for spk, g in AL[AL.spk.isin(set(ids))].groupby("spk", sort=True):
        byv = []
        for vid, v in g.groupby("video", sort=True):
            rows = [r for r in v.itertuples(index=False) if (spk, vid, r.utt) not in used_v]
            room = PER_VIDEO - vcount.get((spk, vid), 0)        # <= 2 clips per video, FLAG_37's clips included
            byv.append(rows[:max(0, room)])
        cand = [r for k in range(PER_VIDEO) for v in byv if k < len(v) for r in [v[k]]]
        VSEL += [dict(r._asdict(), kind=kind) for r in cand[:NV + 4]]  # spares for unreadable / empty clips
VSEL = pd.DataFrame(VSEL).reset_index(drop=True)
log(f"faces: {len(FSEL)} tracks ({FSEL.kind.value_counts().to_dict()}); voice candidates: {len(VSEL)} ({VSEL.kind.value_counts().to_dict()})")


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


def decode_m4a(b):
    import av
    out = []
    with av.open(io.BytesIO(b)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        for fr in c.decode(c.streams.audio[0]):
            out += [o.to_ndarray().reshape(-1) for o in res.resample(fr)]
        out += [o.to_ndarray().reshape(-1) for o in res.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(TARGET_SR // 10, np.float32)


def face_job(r):
    try:
        return middle_frame(REM[r.src].member(int(r.off), int(r.csize), int(r.method)))
    except Exception as e:
        BROKEN["image"].append(f"{r.name} | {type(e).__name__}: {e}"); return None


def voice_job(r):
    try:
        return decode_m4a(REM[r.src].member(int(r.off), int(r.csize), int(r.method)))
    except Exception as e:
        BROKEN["audio"].append(f"{r.name} | {type(e).__name__}: {e}"); return None


# ---------------------------------------------------------------- faces (chunks of 1 200 tracks)
FENC = None
CH = 1200
for c0 in range(0, len(FSEL), CH):
    part = FP_ / f"faces_{c0 // CH:03d}.npz"
    if part.exists():
        continue
    t0 = time.time(); S_ = FSEL.iloc[c0:c0 + CH]
    with cf.ThreadPoolExecutor(THREADS) as ex:
        imgs = list(ex.map(face_job, S_.itertuples(index=False)))
    ok = [i for i, im in enumerate(imgs) if im is not None]
    if FENC is None:
        FENC = {"farl16": build("farl16V"), "clipL": build("clipLV")}
    R_ = {n: [] for n in FENC}
    for s in range(0, len(ok), 64):
        b = [imgs[i] for i in ok[s:s + 64]]
        for n, enc in FENC.items():
            R_[n].append(enc(torch.stack([enc.prep(im) for im in b]))[""])
    np.savez(part, spk=S_.spk.to_numpy(dtype=str)[ok], video=S_.video.to_numpy(dtype=str)[ok], utt=S_.utt.to_numpy(dtype=str)[ok],
             kind=S_.kind.to_numpy(dtype=str)[ok], **{n: np.concatenate(v).astype(np.float16) for n, v in R_.items() if v})
    log(f"  faces chunk {c0 // CH}: {len(ok)}/{len(S_)} [{time.time() - t0:.0f}s]")
FENC = None
release()

# ---------------------------------------------------------------- voices (chunks of 120 speakers): SimAM p / ft + LID
from speechbrain.inference.classifiers import EncoderClassifier
try:
    from speechbrain.utils.fetching import LocalStrategy
    _ls = {"local_strategy": LocalStrategy.COPY}
except ImportError:
    _ls = {}
VENC, LIDM = None, None
spks = sorted(VSEL.spk.unique()); CHS = 120
for c0 in range(0, len(spks), CHS):
    part = VP_ / f"voices_{c0 // CHS:03d}.npz"
    if part.exists():
        continue
    t0 = time.time()
    S_ = VSEL[VSEL.spk.isin(set(spks[c0:c0 + CHS]))].reset_index(drop=True)
    with cf.ThreadPoolExecutor(THREADS) as ex:
        waves = list(ex.map(voice_job, S_.itertuples(index=False)))
    keep, cnt = [], {}
    for i, w in enumerate(waves):
        s = S_.spk[i]
        if w is None or len(w) < int(MIN_SEC * TARGET_SR) or cnt.get(s, 0) >= NV:
            continue
        keep.append(i); cnt[s] = cnt.get(s, 0) + 1
    if VENC is None:
        VENC = {"sam100p": build("sam100pA"), "sam100f": build("sam100fA")}
        LIDM = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=str(CACHE / "lid_voxlingua107"),
                                              run_opts={"device": str(DEV)}, **_ls)
        LIDM.eval()
    L = int(CLIP_SEC * TARGET_SR)
    crops = [centre_crop(waves[i], L) for i in keep]
    R_ = {n: np.stack([enc(w)[""] for w in crops]) for n, enc in VENC.items()}
    with torch.no_grad():
        R_["lid"] = np.stack([torch.softmax(LIDM.classify_batch(torch.from_numpy(np.ascontiguousarray(w))[None].to(DEV))[0].float().reshape(1, -1), 1)[0].cpu().numpy()
                              for w in crops])
    np.savez(part, spk=S_.spk.to_numpy(dtype=str)[keep], video=S_.video.to_numpy(dtype=str)[keep], utt=S_.utt.to_numpy(dtype=str)[keep],
             kind=S_.kind.to_numpy(dtype=str)[keep], dur=np.array([len(waves[i]) / TARGET_SR for i in keep], np.float32), **{n: v.astype(np.float16) for n, v in R_.items()})
    log(f"  voices chunk {c0 // CHS}: {len(keep)} clips of {len(set(S_.spk.values[keep]))} speakers [{time.time() - t0:.0f}s]")
VENC = None
release()

# ---------------------------------------------------------------- assemble, LID rule, manifest
def cat(parts, keys):
    Z = [np.load(p, allow_pickle=True) for p in sorted(parts)]            # this notebook's own part files
    Z = [z for z in Z if len(z["spk"])]
    return {k: (lambda a: a.astype(str) if a.dtype == object else a)(np.concatenate([z[k] for z in Z])) for k in keys}


Fz = cat(FP_.glob("faces_*.npz"), ["spk", "video", "utt", "kind", "farl16", "clipL"])
Vz = cat(VP_.glob("voices_*.npz"), ["spk", "video", "utt", "kind", "dur", "sam100p", "sam100f", "lid"])
pd.DataFrame({k: Fz[k] for k in ("spk", "video", "utt", "kind")}).to_csv(VB / "face_meta.csv", index=False)
pd.DataFrame({k: Vz[k] for k in ("spk", "video", "utt", "kind", "dur")}).to_csv(VB / "voice_meta.csv", index=False)
for k in ("farl16", "clipL"):
    np.save(VB / f"face_{k}.npy", Fz[k])
for k in ("sam100p", "sam100f"):
    np.save(VB / f"voice_{k}.npy", Vz[k])
np.save(VB / "voice_lid_probs.npy", Vz["lid"])
labels = [LIDM.hparams.label_encoder.ind2lab[i] for i in range(len(LIDM.hparams.label_encoder.ind2lab))] if LIDM is not None else None
if labels is None:                                              # every voice chunk came from an earlier run: labels from the hub
    LIDM = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=str(CACHE / "lid_voxlingua107"),
                                          run_opts={"device": "cpu"}, **_ls)
    labels = [LIDM.hparams.label_encoder.ind2lab[i] for i in range(len(LIDM.hparams.label_encoder.ind2lab))]
BN = [i for i, l in enumerate(labels) if l.split(":")[0].strip() == "bn"]
assert len(labels) == 107 and len(BN) == 1
P_ = Vz["lid"].astype(np.float32); top = P_.argmax(1); pbn = P_[:, BN[0]]
flag = (top == BN[0]) | (pbn >= 0.2)
drop = sorted(set(Vz["spk"][flag]))
lid = dict(clips=int(len(P_)), clips_flagged=int(flag.sum()), speakers=int(len(set(Vz["spk"]))), speakers_dropped=len(drop),
           clips_dropped=int(np.isin(Vz["spk"], drop).sum()), speakers_dropped_ids=drop,
           top_languages=pd.Series([labels[i].split(":")[0] for i in top]).value_counts().head(15).to_dict())
(VB / "lid_summary.json").write_text(json.dumps(lid, indent=1))
VMAN = dict(created=time.strftime("%Y-%m-%d %H:%M"), new_identities=len(NEW_IDS), more_identities=len(MORE_IDS), faces=int(len(Fz["spk"])), voices=int(len(Vz["spk"])),
            face_speakers=int(len(set(Fz["spk"]))), voice_speakers=int(len(set(Vz["spk"]))), lid={k: v for k, v in lid.items() if k != "speakers_dropped_ids"},
            sources=SOURCES, sizes={k: [int(x) for x in REM[k].sizes] for k in REM}, broken={k: len(v) for k, v in BROKEN.items()},
            by_kind={"faces": pd.Series(Fz["kind"]).value_counts().to_dict(), "voices": pd.Series(Vz["kind"]).value_counts().to_dict()},
            rule="new = VoxCeleb2 dev outside FLAG_37's list + test (NF 6 / NV 8, <= 2 per video); more = 6 faces + 8 voices more for "
                 "FLAG_37's speakers (unused videos first; <= 2 clips per video counting FLAG_37's); LID rule of FLAG_42 over these clips")
(VB / "feats_manifest.json").write_text(json.dumps(VMAN, indent=1))
print(json.dumps(VMAN, indent=1))
EXTRA_LINKS["Experiment/SCALE-ID-01/feats_vox2b"] = VB
