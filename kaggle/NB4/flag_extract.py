"""FLAG 2027 v2 — re-extract features from the official raw media (Kaggle NB-1, FLAG_04_extract).

Design goals (docs/PLAN_KAGGLE.md §3):
  * never OOM: audio is sorted by duration and packed into batches by *padded seconds*; an OOM halves
    the batch and retries, a single file that still fails runs on CPU (exact, just slower);
  * never silently lose quality: ECAPA / ArcFace stay fp32; batched vs one-by-one embeddings are
    compared; faces are detected + 5-point aligned (resize-only is the fallback and is counted);
  * resumable: one .npy per (encoder, split); an existing file is skipped, so re-running resumes.

Row order of every output == row order of the organisers' *.txt, i.e. of their feature CSVs.
"""
from __future__ import annotations
import sys, time, json, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf

SPLITS = ["train", "no_gender/English", "no_gender/Bangla", "gender/English", "gender/Bangla"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tag(split):
    return split.replace("/", "_")


# ----------------------------------------------------------------------------- locating the data
def media_root(inp: Path, work: Path) -> Path:
    """Kaggle may keep the uploaded zips or auto-extract them; support both."""
    if next(inp.rglob("*.wav"), None) is not None:
        return inp
    zips = sorted(inp.rglob("*.zip"))
    assert zips, f"no .wav and no .zip under {inp}: attach the RAW dataset (train_set.zip + dev_set.zip)"
    work.mkdir(parents=True, exist_ok=True)
    for zp in zips:
        marker = work / f".{zp.stem}.done"
        if marker.exists():
            continue
        log("extracting", zp.name)
        with zipfile.ZipFile(zp) as z:
            z.extractall(work)
        marker.touch()
    return work


def resolve_split(root: Path, txt_name, wav_col, jpg_col, must_contain=None):
    """Pick the copy of `txt_name` whose media actually sit next to it (a features-only dataset may
    carry a txt of the same name without media)."""
    cands = [p for p in root.rglob(txt_name) if must_contain is None or must_contain in p.as_posix()]
    assert cands, f"no {txt_name} under {root} (filter={must_contain})"
    for p in sorted(cands, key=lambda q: len(q.as_posix())):
        df = pd.read_csv(p, sep=" ", header=None)
        if (p.parent / df[wav_col].iloc[0]).exists() and (p.parent / df[jpg_col].iloc[0]).exists():
            return dict(txt=p, root=p.parent, wav=df[wav_col].tolist(), jpg=df[jpg_col].tolist())
    raise FileNotFoundError(f"{txt_name}: {len(cands)} copies, none has its media beside it")


def index_splits(root: Path):
    S = {"train": resolve_split(root, "train_English.txt", 2, 3)}
    for prot in ["no_gender", "gender"]:
        for lang in ["English", "Bangla"]:
            S[f"{prot}/{lang}"] = resolve_split(root, f"{lang}_test.txt", 1, 2, must_contain=f"/{prot}/")
    for k, v in S.items():
        v["dur"] = np.array([sf.info(str(v["root"] / p)).duration for p in v["wav"]], dtype=np.float32)
        log(f"{k:20s} {len(v['wav']):5d} rows | dur median {np.median(v['dur']):.1f}s max {v['dur'].max():.1f}s")
    return S


def copy_metadata(inp: Path, S, out: Path):
    """Ship the txt index + gender meta with the features so NB-2 needs no raw media."""
    meta = sorted(inp.rglob("meta_file_train_set.csv"))
    if meta:
        shutil.copy(meta[0], out / "meta_file_train_set.csv")
    for k, v in S.items():
        shutil.copy(v["txt"], out / f"index_{tag(k)}.txt")


# ----------------------------------------------------------------------------- batching
def pack_batches(durs, max_padded_sec=240.0, max_bs=32):
    """Longest first; a batch costs len(batch) * longest_item seconds after padding."""
    order = np.argsort(-np.asarray(durs), kind="stable")
    batches, cur = [], []
    for i in order:
        longest = durs[cur[0]] if cur else durs[i]
        if cur and ((len(cur) + 1) * longest > max_padded_sec or len(cur) >= max_bs):
            batches.append(cur); cur = []
        cur.append(int(i))
    if cur:
        batches.append(cur)
    return batches


def run_batched(items, fn, durs, max_padded_sec, max_bs, cpu_fn=None, label=""):
    """fn(list_of_items) -> list of per-item outputs. Halves the batch on CUDA OOM."""
    out = [None] * len(items)
    batches = pack_batches(durs, max_padded_sec, max_bs)
    t0, done, n_oom = time.time(), 0, 0

    def go(idx):
        nonlocal n_oom
        try:
            res = fn([items[i] for i in idx])
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            torch.cuda.empty_cache()
            if len(idx) == 1:
                log(f"  OOM on a single item ({durs[idx[0]]:.1f}s) -> CPU")
                res = (cpu_fn or fn)([items[idx[0]]])
            else:
                h = len(idx) // 2
                go(idx[:h]); go(idx[h:])
                return
        for i, r in zip(idx, res):
            out[i] = r

    for bi, b in enumerate(batches):
        go(b)
        done += len(b)
        if bi % 50 == 0 or done == len(items):
            el = time.time() - t0
            log(f"  {label} {done}/{len(items)}  {el:.0f}s  eta {el / done * (len(items) - done):.0f}s  oom={n_oom}")
    return out


# ----------------------------------------------------------------------------- ECAPA
def _speechbrain_import():
    import torchaudio
    for m in [m for m in list(sys.modules) if m == "speechbrain" or m.startswith("speechbrain.")]:
        del sys.modules[m]
    for name, f in [("list_audio_backends", lambda: ["soundfile"]), ("get_audio_backend", lambda: "soundfile"),
                    ("set_audio_backend", lambda *a, **k: None)]:
        if not hasattr(torchaudio, name):
            setattr(torchaudio, name, f)
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier


class Ecapa:
    """speechbrain/spkrec-ecapa-voxceleb. Returns (192-d output, 6144-d attentive-stat-pooling output)."""

    def __init__(self, savedir):
        EncoderClassifier = _speechbrain_import()
        kw = {}
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        self.m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(savedir),
                                                run_opts={"device": str(DEV)}, **kw)
        self.m.eval()
        self.emb = self.m.mods.embedding_model
        self._pool = {}
        self.emb.asp.register_forward_hook(lambda mod, i, o: self._pool.__setitem__("x", o))

    @torch.no_grad()
    def __call__(self, sigs, device=None):
        device = device or DEV
        lens = [len(s) for s in sigs]
        L = max(lens)
        x = torch.zeros(len(sigs), L)
        for i, s in enumerate(sigs):
            x[i, :len(s)] = torch.from_numpy(s)
        rel = torch.tensor([l / L for l in lens], dtype=torch.float32, device=device)
        mods = self.m.mods if device == DEV else self._cpu_mods()
        f = mods.compute_features(x.to(device))
        f = mods.mean_var_norm(f, rel)
        e = mods.embedding_model(f, rel).squeeze(1)
        p = self._pool["x"].squeeze(-1)
        return list(zip(e.float().cpu().numpy(), p.float().cpu().numpy()))

    def _cpu_mods(self):
        if not hasattr(self, "_cpu"):
            import copy
            self._cpu = copy.deepcopy(self.m.mods).to("cpu")
            self._cpu.embedding_model.asp.register_forward_hook(lambda mod, i, o: self._pool.__setitem__("x", o))
        return self._cpu


def read_wav(path):
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == 16000, (path, sr)
    return w[:, 0]


def crop_plan(durs, target_durs, k, seed=0, min_sec=2.0):
    """k crops per utterance, length drawn from the dev-Bangla duration distribution.
    Returns (start_sec, len_sec) arrays of shape (k, n); an utterance shorter than the draw is kept whole."""
    rng = np.random.RandomState(seed)
    n = len(durs)
    L = rng.choice(np.asarray(target_durs), size=(k, n))
    L = np.clip(L, min_sec, None)
    L = np.minimum(L, durs[None, :])
    S = rng.uniform(0, 1, size=(k, n)) * (durs[None, :] - L)
    return S.astype(np.float32), L.astype(np.float32)


def extract_ecapa(S, out: Path, n_crops=3, max_padded_sec=240.0, max_bs=32):
    ecapa = Ecapa(out.parent / "_ecapa_ckpt")
    for split, sp in S.items():
        f192, f6144 = out / f"voice_ecapa192_{tag(split)}.npy", out / f"voice_ecapa6144_{tag(split)}.npy"
        if f192.exists() and f6144.exists():
            log("skip (exists)", split); continue
        paths = [sp["root"] / p for p in sp["wav"]]
        res = run_batched(paths, lambda ps: ecapa([read_wav(p) for p in ps]), sp["dur"], max_padded_sec, max_bs,
                          cpu_fn=lambda ps: ecapa([read_wav(p) for p in ps], device="cpu"), label=f"ecapa {split}")
        np.save(f192, np.stack([r[0] for r in res]).astype(np.float32))
        np.save(f6144, np.stack([r[1] for r in res]).astype(np.float32))
        log(split, "saved", np.load(f6144, mmap_mode="r").shape)

    # duration-matched crops of the TRAIN utterances (plan §0.4)
    fc192, fc6144 = out / "voice_ecapa192crop_train.npy", out / "voice_ecapa6144crop_train.npy"
    if n_crops and not (fc192.exists() and fc6144.exists()):
        tgt = np.concatenate([S["no_gender/Bangla"]["dur"], S["gender/Bangla"]["dur"]])
        tr = S["train"]
        st, ln = crop_plan(tr["dur"], tgt, n_crops)
        np.save(out / "crop_plan_train.npy", np.stack([st, ln]))
        paths = [tr["root"] / p for p in tr["wav"]]
        c192, c6144 = [], []
        for k in range(n_crops):
            items = list(range(len(paths)))

            def load(i, k=k):
                w = read_wav(paths[i]); a = int(st[k, i] * 16000); return w[a:a + int(ln[k, i] * 16000)]
            res = run_batched(items, lambda ii: ecapa([load(i) for i in ii]), ln[k], max_padded_sec * 2, max_bs * 2,
                              cpu_fn=lambda ii: ecapa([load(i) for i in ii], device="cpu"), label=f"crop{k}")
            c192.append(np.stack([r[0] for r in res])); c6144.append(np.stack([r[1] for r in res]))
        np.save(fc192, np.stack(c192).astype(np.float32)); np.save(fc6144, np.stack(c6144).astype(np.float32))
        log("crops saved", np.load(fc6144, mmap_mode="r").shape,
            f"median crop {np.median(ln):.1f}s vs train median {np.median(tr['dur']):.1f}s")
    return ecapa


def check_batch_consistency(ecapa, S, n=24, seed=0):
    """Padding must not change embeddings: compare batched vs one-at-a-time on the longest+random files."""
    sp = S["no_gender/English"]
    rng = np.random.RandomState(seed)
    idx = list(np.argsort(-sp["dur"])[:4]) + list(rng.choice(len(sp["wav"]), n - 4, replace=False))
    sigs = [read_wav(sp["root"] / sp["wav"][i]) for i in idx]
    batched = []
    for b in pack_batches(sp["dur"][idx], 240.0, 32):
        batched += list(zip(b, ecapa([sigs[i] for i in b])))
    batched = [r for _, r in sorted(batched, key=lambda t: t[0])]
    single = [ecapa([s])[0] for s in sigs]
    c = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    c192 = [c(b[0], s[0]) for b, s in zip(batched, single)]
    c6144 = [c(b[1], s[1]) for b, s in zip(batched, single)]
    log(f"batched vs single cosine: 192-d min {min(c192):.5f} | 6144-d min {min(c6144):.5f}")
    return dict(min_cos_192=min(c192), min_cos_6144=min(c6144))


# ----------------------------------------------------------------------------- ArcFace
class ArcFace:
    """insightface buffalo_l: SCRFD detection + 5-point norm_crop + ArcFace R100 (512-d)."""

    def __init__(self):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        avail = ort.get_available_providers()
        prov = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in avail]
        try:
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=prov)
        except TypeError:
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if DEV.type == "cuda" else -1, det_size=(224, 224), det_thresh=0.3)
        self.rec = self.app.models["recognition"]
        used = self.rec.session.get_providers()
        log("onnxruntime providers available", avail, "| recognition runs on", used)

    def embed(self, path):
        import cv2
        from insightface.utils import face_align
        img = cv2.imread(str(path))
        assert img is not None, path
        faces = self.app.det_model.detect(img, max_num=0, metric="default")
        bboxes, kpss = faces if isinstance(faces, tuple) else (faces, None)
        if bboxes is not None and len(bboxes) and kpss is not None:
            h, w = img.shape[:2]
            ctr = np.array([w / 2, h / 2])
            # prefer the big, central face: the crops are centred on the speaker
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            dist = np.linalg.norm((bboxes[:, :2] + bboxes[:, 2:4]) / 2 - ctr, axis=1)
            j = int(np.argmax(area - 2.0 * dist ** 2))
            aimg = face_align.norm_crop(img, landmark=kpss[j], image_size=112)
            ok, score = True, float(bboxes[j, 4])
        else:
            h, w = img.shape[:2]; m = int(min(h, w) * 0.15)
            aimg = cv2.resize(img[m:h - m, m:w - m], (112, 112))
            ok, score = False, 0.0
        return self.rec.get_feat(aimg).flatten().astype(np.float32), ok, score


def extract_arcface(S, out: Path):
    af = ArcFace()
    stats = {}
    for split, sp in S.items():
        f = out / f"face_arcface_{tag(split)}.npy"
        if f.exists():
            log("skip (exists)", split); continue
        t0 = time.time(); E, OK = [], []
        for i, p in enumerate(sp["jpg"]):
            e, ok, _ = af.embed(sp["root"] / p)
            E.append(e); OK.append(ok)
            if i % 1000 == 0:
                log(f"  arcface {split} {i}/{len(sp['jpg'])} {time.time() - t0:.0f}s")
        np.save(f, np.stack(E)); np.save(out / f"face_arcface_detok_{tag(split)}.npy", np.array(OK))
        stats[split] = float(np.mean(OK))
        log(f"{split}: detection+alignment rate {100 * np.mean(OK):.1f}%")
    return stats


# ----------------------------------------------------------------------------- self-supervised speech (H2b)
def extract_ssl(S, out: Path, model_id, short, window_sec=20.0, hop_sec=10.0, fp16=True):
    """Per-layer time-mean of hidden states -> (N, n_layers+1, H) float16. One file at a time with
    fixed windows: no padding, and attention cost is bounded by the window, not by the 81 s file."""
    from transformers import AutoModel, AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEV).eval()
    use_amp = fp16 and DEV.type == "cuda"
    W, H = int(window_sec * 16000), int(hop_sec * 16000)

    @torch.no_grad()
    def embed(w, amp=use_amp):
        starts = [0] if len(w) <= W else list(range(0, len(w) - W + H, H))
        acc, tot = None, 0.0
        for s in starts:
            seg = w[s:s + W]
            if len(seg) < 16000 and tot > 0:
                break
            x = fe(seg, sampling_rate=16000, return_tensors="pt").input_values.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                hs = model(x, output_hidden_states=True).hidden_states
            m = torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy()
            acc = m * len(seg) if acc is None else acc + m * len(seg)
            tot += len(seg)
        return acc / tot

    # fp16 sanity check against fp32 on a few files
    sp = S["no_gender/Bangla"]
    chk = [embed(read_wav(sp["root"] / sp["wav"][i]), amp=False) for i in range(4)]
    chk16 = [embed(read_wav(sp["root"] / sp["wav"][i])) for i in range(4)]
    cmin = min(float(np.dot(a[l], b[l]) / np.linalg.norm(a[l]) / np.linalg.norm(b[l]))
               for a, b in zip(chk, chk16) for l in range(a.shape[0]))
    log(f"{short}: fp16 vs fp32 per-layer cosine min {cmin:.5f}")
    for split, spx in S.items():
        f = out / f"voice_{short}_{tag(split)}.npy"
        if f.exists():
            log("skip (exists)", split); continue
        t0 = time.time(); E = []
        for i, p in enumerate(spx["wav"]):
            E.append(embed(read_wav(spx["root"] / p)).astype(np.float16))
            if i % 1000 == 0:
                log(f"  {short} {split} {i}/{len(spx['wav'])} {time.time() - t0:.0f}s")
        np.save(f, np.stack(E))
        log(split, "saved", np.load(f, mmap_mode="r").shape)
    model.to("cpu"); torch.cuda.empty_cache()
    return cmin


def write_manifest(out: Path, extra: dict):
    files = {p.name: list(np.load(p, mmap_mode="r").shape) for p in sorted(out.glob("*.npy"))}
    man = dict(created=time.strftime("%Y-%m-%d %H:%M"), files=files, **extra)
    (out / "feats_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log("manifest:", len(files), "arrays")
    return man


# ----------------------------------------------------------------------------- generic hub voice encoder (round 3)
def load_redimnet2(model_name, dataset, train_type="lm"):
    """ReDimNet2 (PalabraAI, MIT). Input: raw 16 kHz waveform (B, samples). Frozen, eval mode."""
    m = torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name=model_name, train_type=train_type,
                       dataset=dataset, pretrained=True, trust_repo=True)
    return m.to(DEV).eval()


class EncoderTooSlow(RuntimeError):
    pass


def extract_hub_voice(S, out: Path, short, model, crop_plan_file=None, max_minutes=None, probe_files=50):
    """One file at a time: the model takes no length mask, so batching padded audio would change the
    embedding (ReDimNet2: cos 0.63 with 3 s of zero padding). bs=1 keeps it exact.
    Writes voice_<short>_<split>.npy and, with the NB-1 crop plan, voice_<short>crop_train.npy -- the
    SAME crops as r2_mix, so the encoder is the only variable.

    max_minutes: speed guard. After `probe_files` files the audio-seconds throughput is measured and the
    whole job (full splits + crops) projected. Over budget -> crops are dropped; still over budget ->
    EncoderTooSlow is raised so the caller can move on (the vb2+vox2+cnc2 checkpoint adds GroupNorm
    layers and ran ~20x slower than vox2 on CPU)."""
    @torch.no_grad()
    def emb(w):
        x = torch.from_numpy(np.ascontiguousarray(w)).float()[None].to(DEV)
        try:
            e = model(x)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); log("  OOM on one file -> CPU")
            e = model.to("cpu")(x.cpu()); model.to(DEV)
        e = e[0] if isinstance(e, (tuple, list)) else e
        return e.reshape(-1).float().cpu().numpy()

    plan = np.load(crop_plan_file) if crop_plan_file is not None else None
    full_sec = float(sum(sp["dur"].sum() for sp in S.values()))
    crop_sec = float(plan[1].sum()) if plan is not None else 0.0
    do_crops, probed, t_all, sec_all = plan is not None, max_minutes is None, 0.0, 0.0

    for split, sp in S.items():
        f = out / f"voice_{short}_{tag(split)}.npy"
        if f.exists():
            log("skip (exists)", f.name); continue
        t0 = time.time(); E = []
        for i, p in enumerate(sp["wav"]):
            t1 = time.time()
            E.append(emb(read_wav(sp["root"] / p)))
            t_all += time.time() - t1; sec_all += float(sp["dur"][i])
            if not probed and len(E) >= probe_files:
                probed = True
                rate = t_all / max(sec_all, 1e-6)                   # compute seconds per audio second
                eta_full, eta_crop = rate * full_sec / 60, rate * crop_sec / 60
                log(f"  {short} speed probe: {rate * 5:.3f}s per 5s of audio -> ETA full {eta_full:.0f} min, crops {eta_crop:.0f} min (budget {max_minutes})")
                if eta_full > max_minutes:
                    raise EncoderTooSlow(f"{short}: projected {eta_full:.0f} min > budget {max_minutes} min")
                if do_crops and eta_full + eta_crop > max_minutes:
                    do_crops = False; log(f"  {short}: crops dropped to stay within budget")
            if i % 2000 == 0:
                log(f"  {short} {split} {i}/{len(sp['wav'])} {time.time() - t0:.0f}s")
        np.save(f, np.stack(E).astype(np.float32)); log(split, "saved", np.load(f, mmap_mode="r").shape)
    fc = out / f"voice_{short}crop_train.npy"
    if do_crops and not fc.exists():
        st, ln = plan
        tr = S["train"]; assert st.shape[1] == len(tr["wav"]), "crop plan does not match the train index"
        C = []
        for k in range(st.shape[0]):
            t0 = time.time(); E = []
            for i, p in enumerate(tr["wav"]):
                w = read_wav(tr["root"] / p); a = int(st[k, i] * 16000)
                E.append(emb(w[a:a + int(ln[k, i] * 16000)]))
            C.append(np.stack(E)); log(f"  {short} crop{k} done {time.time() - t0:.0f}s")
        np.save(fc, np.stack(C).astype(np.float32)); log("crops saved", np.load(fc, mmap_mode="r").shape)
