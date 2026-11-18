"""Generate the REPRESENTATION notebooks of the Kaggle plan of 03/10 (docs/KAGGLE_PLAN_2026-10-03.md).

After CAFUSE-02 / CFE-01 the open question is no longer the fusion but whether a NEW source of identity information exists.
Each notebook extracts one family of frozen representations for the exact rows of the eda bundle (v4 train and the dev
lists in organiser order, MAV-Celeb v1/v2 in ext_*_meta order), so they can be scored on the local SAFE-04 harness
(level 0, person folds, held-out Urdu / Hindi):

  FLAG_33_peav.ipynb    PEAV-01      Meta PE-AV (audio-video contrastive on 100M+ videos): zero-shot face <-> voice
  FLAG_34_layers.ipynb  LAYER-ID-01  every hidden layer of WavLM Base+ / Large / HuBERT Large (mean-pooled) + the two
                                     WeSpeaker models that failed in FLAG_27 (device bug, fixed here)
  FLAG_35_tokens.ipynb  LOCAL-ID-01  ImageBind face patch tokens (CLS + 4 x 4 grid, in the joint space) and one ImageBind
                                     audio embedding per 2 s clip, for late-interaction scoring

Same cells as FLAG_27 (FLAG_12 data discovery, audio loading, resumable extraction loop), one encoder cell, a zero-shot
probe on TRUE labels (v4 train, MAV-Celeb cross-video; never the dev lists) and the manifest. Full code in the cells.
Wide outputs are flattened to one row per sample and saved as float16; `feats_manifest.json` -> "reshape" gives (L, D).
    python build_rep_notebooks.py
    FLAG_LOCAL=1 FLAG_IN=<mini> FLAG_OUT=<dir> FLAG_CACHE=<cache> FLAG_SMOKE_ROWS=12 \
        python run_notebook_local.py FLAG_33_peav.ipynb --jupyter
"""
import build_eval_notebooks as B12
import build_scout_notebooks as B27


# ============================================================================ shared pieces
def env(out, encoders, note):
    e = B12.ENV12
    for a, b in [('OUT = WORKDIR / "feats_eval"', f'OUT = WORKDIR / "{out}"'),
                 ('ENCODERS = ["arcface", "ecapa", "vitage", "w2v2ag", "ibv", "iba"]   # ibv right before iba: one ImageBind load',
                  f'ENCODERS = {encoders!r}   # {note}'),
                 ('EXT_ENCODERS = ["arcface", "ecapa"]                                 # their ImageBind / age exist (FLAG_10)',
                  'EXT_ENCODERS = list(ENCODERS)                                       # every model also on the v1/v2 rows')]:
        assert e.count(a) == 1, a
        e = e.replace(a, b)
    return e.rstrip() + '''
SMOKE_ROWS = int(os.environ.get("FLAG_SMOKE_ROWS", "0"))   # local smoke test only: rows kept per split / source
if LOCAL and os.environ.get("FLAG_ENCODERS"):              # local smoke test only: a subset of the encoders
    ENCODERS = EXT_ENCODERS = os.environ["FLAG_ENCODERS"].split(",")
if LOCAL and os.environ.get("FLAG_SPLITS"):                # local smoke test only: a subset of the organiser lists
    SPLIT_FILTER = os.environ["FLAG_SPLITS"].split(",")
'''


def data(out):
    a = 're.search(r"feats[-_]eval", p.as_posix())'
    assert B12.DATA12.count(a) == 1
    d = B12.DATA12.replace(a, f're.search(r"{out.replace("_", "[-_]")}", p.as_posix())')
    return d.rstrip() + r'''
if LOCAL and SMOKE_ROWS:                     # local smoke test: the first rows of every split and source
    for v in S.values():
        v.update(wav=v["wav"][:SMOKE_ROWS], jpg=v["jpg"][:SMOKE_ROWS], dur=v["dur"][:SMOKE_ROWS])
    for r in EXT.values():
        r.face, r.voice = r.face[:SMOKE_ROWS], r.voice[:SMOKE_ROWS]
    log(f"smoke test: every split and source cut to {SMOKE_ROWS} rows")
'''


def run(keep_ib=()):
    r = B12.RUN12
    for a, b in [
        ('''MAIN = {"arcface": "", "ecapa": "192", "vitage": "", "w2v2ag": "", "ibv": "", "iba": ""}''',
         '''MAIN = {"arcface": "", "ecapa": "192", "vitage": "", "w2v2ag": "", "ibv": "", "iba": ""}
ARRAYS.update(ARRAYS_NEW); MAIN.update(MAIN_NEW)                         # this notebook's arrays


def _dt(v, split):
    """float16 for the v1/v2 rows (as FLAG_12) and for every wide array (layers / tokens); float32 otherwise."""
    return v if (split and v.ndim == 2 and v.shape[1] <= 4096) or v.ndim == 1 else v.astype(np.float16)'''),
        ("            np.save(target(name, s, split, src), v if split else v.astype(np.float16))",
         "            np.save(target(name, s, split, src), _dt(v, split))"),
        ("    np.save(target(name, MAIN[name], split, src), R[MAIN[name]] if split else R[MAIN[name]].astype(np.float16))",
         "    np.save(target(name, MAIN[name], split, src), _dt(R[MAIN[name]], split))"),
        ('    release(keep_ib=(name == "ibv"))', f'    release(keep_ib=(name in {("ibv",) + tuple(keep_ib)!r}))')]:
        assert r.count(a) == 1, a
        r = r.replace(a, b)
    return r


def done(dataset):
    d = B12.DONE12
    for a, b in [('(OUT / "feats_manifest.json").write_text(json.dumps(man, indent=1))',
                  'man["reshape"] = RESHAPE                 # flattened wide arrays: row -> (L, D)\n'
                  '(OUT / "feats_manifest.json").write_text(json.dumps(man, indent=1))'),
                 ("name it flag2027-feats-eval", f"name it {dataset} (re-run with this output attached until every encoder is done)")]:
        assert d.count(a) == 1, a
        d = d.replace(a, b)
    return d


PROBE_LIB = r'''
# ============================================================================ zero-shot probe on TRUE labels (no training)
# v4 train: positives = two rows of the same person (v4 has no video id, so some may share a session);
# MAV-Celeb v1/v2: positives = face and voice of the same person from DIFFERENT videos. The dev lists are never used.
from sklearn.metrics import roc_curve


def eer(s, y):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * (fpr[i] + fnr[i]) / 2


l2n = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def load_arr(stem, kind, split=None, src=None):
    f = OUT / (f"{kind}_{stem}_{tag(split)}.npy" if split else f"ext_{src}_{kind}_{stem}.npy")
    return np.load(f).astype(np.float32) if f.exists() else None


def probe_pairs(n=3000, seed=0):
    """-> list of (data, protocol, face rows, voice rows, labels, split, src)."""
    rng, out = np.random.RandomState(seed), []
    if "train" in S:
        nr = len(S["train"]["wav"])
        spk = pd.read_csv(S["train"]["txt"], sep=" ", header=None)[4].astype(str).values[:nr]
        meta = SCAN["files"].get("meta_file_train_set.csv")
        gmap = dict(pd.read_csv(meta[0], header=None).values) if meta else {}
        by = {s: np.where(spk == s)[0] for s in np.unique(spk)}
        ok = [i for i in range(nr) if len(by[spk[i]]) > 1]
        if ok:
            pos = [(i, rng.choice(by[spk[i]][by[spk[i]] != i])) for i in rng.choice(ok, n)]
            for sg in (False, True):
                neg, tries = [], 0
                while len(neg) < n and tries < 100 * n:
                    i, j = rng.randint(nr, size=2); tries += 1
                    if spk[i] != spk[j] and (not sg or (gmap.get(spk[i]) == gmap.get(spk[j]) and spk[i] in gmap)):
                        neg.append((i, j))
                pr = np.array(pos + neg)
                out.append(("v4 train", "g" if sg else "ng", pr[:, 0], pr[:, 1], np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                            "train", None))
    for s, rws in EXT.items():
        md = SCAN["files"][f"ext_{s}_voice_meta.csv"][0].parent
        fm = pd.read_csv(md / f"ext_{s}_face_meta.csv").iloc[:len(rws.face)].reset_index(drop=True)
        vm = pd.read_csv(md / f"ext_{s}_voice_meta.csv").iloc[:len(rws.voice)].reset_index(drop=True)
        vidx = rng.choice(len(vm), min(n, len(vm)), replace=False)
        pos, neg = [], []
        for vi in vidx:
            r = vm.iloc[vi]
            cand = np.where((fm.spk.values == r.spk) & (fm.video.values != r.video))[0]
            if len(cand):
                pos.append((rng.choice(cand), vi))
            other = np.where(fm.spk.values != r.spk)[0]
            if len(other):
                neg.append((rng.choice(other), vi))
        if not pos or not neg:
            continue
        pr = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        for lang in sorted(vm.lang.unique()):
            m = vm.lang.values[pr[:, 1]] == lang
            if m.sum() and len(np.unique(y[m])) == 2:
                out.append((f"{s} cross-video", f"ng {lang}", pr[m, 0], pr[m, 1], y[m], None, s))
    return out


def run_probe(scorers, name="probe_zeroshot.csv"):
    """scorers: {label: fn(split, src, face rows, voice rows) -> scores or None}."""
    rows = []
    for data_, prot, fi, vi, y, split, src in probe_pairs():
        for lab, fn in scorers.items():
            try:
                s = fn(split, src, fi, vi)
            except Exception as e:
                log(f"probe {lab} on {data_}: {type(e).__name__}: {e}"); s = None
            if s is not None:
                rows.append(dict(data=data_, protocol=prot, model=lab, n=len(y), eer=round(eer(s, y), 2)))
    PR = pd.DataFrame(rows)
    if len(PR):
        PR.to_csv(OUT / name, index=False)
        print(PR.pivot_table(index="model", columns=["data", "protocol"], values="eer").round(2).to_string())
    else:
        print("probe: nothing to score (arrays missing, or no same-person pairs among these rows)")
    return PR
'''


# ============================================================================ FLAG_33: PE-AV
PEAV_INTRO = """
# FLAG 2027 — NB-33: PEAV-01, Meta PE-AV zero-shot face <-> voice

**Why.** ImageBind zero-shot carries our system (ERR-03), and every attempt to add identity information on top of it
(new encoders, CFE, more people, higher-capacity fusion) has stalled. The remaining gap to the leaders is in the
gender-constrained cells (g/En 27.3 vs 22.0), where only identity cues beyond gender help.
**PE-AV** (Meta, Dec 2025, Apache-2.0) is an audio-video encoder trained contrastively on 100M+ videos. That is about
50 times the AudioSet data behind ImageBind, and talking-face video is common in it. Its audio head and video head share
one space (their dot product is the model's audio-video logit), so a face image (as a one-frame video) and a voice can be
compared zero-shot.

| array | content |
|---|---|
| `face_peav` / `voice_peav` | head embedding in the audio-video space (1024-d for every size); face = 1-frame video |
| `face_peavf4` | the same face as a 4-identical-frame video (closer to the training input) |
| `face_peavpool` / `voice_peavpool` | pooled encoder output before the head (for learned bridges) |

**Rule check (30/09 ruling + answer of 03/10):** frozen, no training. The video tower is a face encoder (any is
allowed). The audio tower is pretrained on large public video data that may contain incidental Bengali speech, the same
status as ImageBind / AudioCLIP, which the organisers allowed. Declare it in the system description.

**Rows:** the arrays we already use (v4 train and the dev lists in the organiser's order; MAV-Celeb v1/v2 in
`ext_*_meta.csv` order, voices as a 12 s centre crop). The probe at the end uses **true labels of v4 train and v1/v2
only**, never the dev lists.

**Inputs (Add Input):** the same as FLAG_27:
1. the raw v4 dataset (`train_set` + `dev_set`);
2. MAV-Celeb v1 and v2 + `flag2027-ext-meta`;
3. optional: this notebook's earlier output, to resume.

**Settings:** GPU **T4** (or P100), **Internet ON** (HuggingFace). The first cell upgrades transformers (PE-AV needs >= 5.1).

**Time:** about 8 h for pe-av-large on a T4 (measured 04/10: 28 000 faces 1.3 h; 51 000 voices in float32 about 0.45 s each, 6.5 h). Resumable: if the session stops, Save Version,
attach the output and run again.

**Output:** `/kaggle/working/feats_peav/` -> dataset **`flag2027-feats-peav`**.
"""

PEAV_PIP = r'''
# PE-AV is in transformers >= 5.1 and Kaggle may ship 4.x: upgraded BEFORE the first import of transformers. pip may warn
# about other preinstalled packages pinned to transformers 4; they are not used here.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before torch: fewer fragmentation OOMs
if not LOCAL:
    !pip -q install -U "transformers>=5.17,<6" "huggingface_hub>=1.0,<2" "tokenizers>=0.22" "safetensors>=0.8" hf_xet 2>&1 | tail -2
''' + B12.PIP12[B12.PIP12.index("import numpy as np"):]
assert PEAV_PIP.count("import numpy as np") == 1 and "insightface" not in PEAV_PIP

PEAV_ENC = r'''
# ============================================================================ PE-AV encoders
# Built from the model's own blocks (video_encoder + video_head, audio_encoder + audio_head), exactly as
# PeAudioVideoModel.forward computes logits_audio_video; peav_joint_check() below verifies it on real rows.
# Memory and precision (Kaggle T4 run of 03/10): the whole model in float32 (8.9 GB) left no room for the activations, and
# float16 autocast changed the AUDIO embedding beyond the 0.995 self-check (faces: 0.999998, fine). Hence:
#   - the model is loaded on the CPU, where the joint check runs; then only the four blocks used here move to the GPU
#     (the text tower and the audio-video fusion stay on the CPU);
#   - faces: float16 autocast, PEAV_FACE_CHUNK images per forward, the chunk halved on a CUDA OOM;
#   - voices: float32; a file that does not fit is embedded as the length-weighted mean over 30 s windows (counted).
PEAV_REPO = os.environ.get("FLAG_PEAV_REPO", "facebook/pe-av-large")    # smoke test: facebook/pe-av-small
PEAV_SR = 48000
PEAV_FACE_CHUNK = 8                  # images per forward (each seen as a 1-frame and a 4-frame video)
PEAV_VOICE_WIN = 30.0                # seconds, only for a file that does not fit on the GPU
PEAV_STATS = {"face_chunk": PEAV_FACE_CHUNK, "voice_windowed": 0}
_PE = {}


def peav():
    """The model in float32 on the CPU."""
    if "m" not in _PE:
        from transformers import AutoProcessor, PeAudioVideoModel
        kw = dict(cache_dir=str(CACHE / "hf"))
        proc = AutoProcessor.from_pretrained(PEAV_REPO, **kw)
        m = PeAudioVideoModel.from_pretrained(PEAV_REPO, dtype=torch.float32, **kw).eval()
        _PE.update(m=m, fe=proc.feature_extractor, vp=proc.video_processor, on_dev=False)
        log(f"PE-AV {PEAV_REPO}: {sum(p.numel() for p in m.parameters()) / 1e9:.2f} B parameters (on the CPU)")
    return _PE


def peav_on_device():
    """Move the video encoder + head and the audio encoder + head to DEV (once)."""
    pe = peav()
    if not pe["on_dev"]:
        m = pe["m"]
        blocks = (m.video_model.video_encoder, m.video_model.video_head, m.audio_model.audio_encoder, m.audio_model.audio_head)
        for b in blocks:
            b.to(DEV)
        pe["on_dev"] = True
        n = sum(p.numel() for b in blocks for p in b.parameters()) / 1e9
        log(f"PE-AV video / audio towers on {DEV}: {n:.2f} B parameters"
            + (f", {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated" if DEV.type == "cuda" else ""))
    return pe


dev_of = lambda mod: next(mod.parameters()).device


def peav_video_input(im):
    """One PIL image -> (3, H, W) with the model's own video processor (336 px, mean / std 0.5)."""
    return peav()["vp"](videos=[np.array(im)[None]], return_tensors="pt")["pixel_values_videos"][0, 0]


def peav_audio_input(w):
    """16 kHz float waveform -> the feature extractor's input at 48 kHz (input_values, padding_mask)."""
    import torchaudio
    w48 = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), TARGET_SR, PEAV_SR)
    x = peav()["fe"](w48.numpy(), sampling_rate=PEAV_SR, return_tensors="pt")
    return x["input_values"], x.get("padding_mask")


def face_blocks(vm, x, frames, use_amp):
    """(B, 3, H, W) images as `frames`-frame videos -> (head embedding, pooled output), float32."""
    v = x.to(dev_of(vm.video_encoder))[:, None].expand(-1, frames, -1, -1, -1).contiguous()
    with amp(use_amp):
        pool = vm.video_encoder(pixel_values_videos=v, return_dict=True).pooler_output
        emb = vm.video_head(pool)
    return emb.float(), pool.float()


def voice_blocks(am, iv, pm):
    """Always float32. iv (B, 1, T), pm (B, T) or None -> (head embedding, pooled output)."""
    d = dev_of(am.audio_encoder)
    pool = am.audio_encoder(input_values=iv.to(d), padding_mask=None if pm is None else pm.to(d), return_dict=True).pooler_output
    return am.audio_head(pool).float(), pool.float()


class PEAVFace:
    kind = "face"
    FRAMES = 4                                     # second variant: the image repeated as a 4-frame video

    def __init__(self):
        self.pe = peav_on_device()

    def prep(self, im):
        return peav_video_input(im)

    def _chunk(self, x, use_amp):
        vm, out = self.pe["m"].video_model, {}
        for T_, key in ((1, ""), (self.FRAMES, "f4")):
            emb, pool = face_blocks(vm, x, T_, use_amp)
            out[key] = emb.cpu().numpy()
            if T_ == 1:
                out["pool"] = pool.cpu().numpy()
        return out

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        parts, i = [], 0
        while i < len(x):
            c, oom = PEAV_STATS["face_chunk"], False
            try:
                parts.append(self._chunk(x[i:i + c], use_amp)); i += c
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:                                    # outside the except block: the traceback no longer pins tensors
                torch.cuda.empty_cache()
                assert c > 1, "PE-AV: a single face does not fit on the GPU"
                PEAV_STATS["face_chunk"] = c // 2
                log(f"PE-AV faces: CUDA OOM, chunk {c} -> {c // 2}")
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class PEAVVoice:
    kind = "voice"

    def __init__(self):
        self.pe = peav_on_device()

    @torch.no_grad()
    def __call__(self, w, use_amp=False):              # float32 whatever use_amp says (see the note at the top)
        am, oom = self.pe["m"].audio_model, False
        try:
            emb, pool = voice_blocks(am, *peav_audio_input(w))
            return {"": emb[0].cpu().numpy(), "pool": pool[0].cpu().numpy()}
        except torch.cuda.OutOfMemoryError:
            oom = True
        assert oom
        torch.cuda.empty_cache()                       # a very long file: length-weighted mean over 30 s windows
        PEAV_STATS["voice_windowed"] += 1
        L = int(PEAV_VOICE_WIN * TARGET_SR)
        segs = [w[s:s + L] for s in range(0, len(w), L) if len(w[s:s + L]) >= TARGET_SR] or [w[:L]]
        res = [voice_blocks(am, *peav_audio_input(s_)) for s_ in segs]
        wt = np.array([len(s_) for s_ in segs], np.float64); wt /= wt.sum()
        return {"": sum(a * e[0].cpu().numpy() for a, (e, _) in zip(wt, res)),
                "pool": sum(a * p[0].cpu().numpy() for a, (_, p) in zip(wt, res))}


@torch.no_grad()
def peav_joint_check(n=2):
    """On the CPU, before the towers move: the block-by-block embeddings (the functions the extraction uses) must reproduce
    the model's own logits_audio_video on n real (face, voice) rows."""
    pe = peav(); m = pe["m"]
    assert not pe["on_dev"], "run before the towers move to the GPU"
    k = next(iter(S)); sp = S[k]
    faces = torch.stack([peav_video_input(read_rgb(sp["root"] / p)) for p in sp["jpg"][:n]])
    waves = [load_audio(sp["root"] / p)[:TARGET_SR * 6] for p in sp["wav"][:n]]
    L = min(len(w) for w in waves)
    ins = [peav_audio_input(w[:L]) for w in waves]
    iv = torch.cat([a for a, _ in ins]); pm = None if ins[0][1] is None else torch.cat([b for _, b in ins])
    o = m(pixel_values_videos=faces[:, None], input_values=iv, padding_mask=pm)
    F_ = face_blocks(m.video_model, faces, 1, False)[0].numpy()
    V_ = voice_blocks(m.audio_model, iv, pm)[0].numpy()
    ours = (V_ @ F_.T) * float(m.audio_video_logit_scale) + float(m.audio_video_logit_bias)
    d = float(np.abs(ours - o.logits_audio_video.float().numpy()).max())
    log(f"PE-AV joint check on {k} (CPU): max |logit difference| = {d:.2e} (logit scale {float(m.audio_video_logit_scale):.3f})")
    return d


NEW = {"peavV": PEAVFace, "peavA": PEAVVoice}
_build12 = build


def build(name):
    return NEW[name]() if name in NEW else _build12(name)


ARRAYS_NEW = {"peavV": ("face", {"": "peav", "f4": "peavf4", "pool": "peavpool"}),
              "peavA": ("voice", {"": "peav", "pool": "peavpool"})}
MAIN_NEW = {"peavV": "", "peavA": ""}
RESHAPE = {}
try:
    CHECKS_JOINT = peav_joint_check()
except Exception as e:                       # informative only: the extraction does not depend on it
    CHECKS_JOINT = f"{type(e).__name__}: {e}"
    log("PE-AV joint check failed:", CHECKS_JOINT)
'''

PEAV_PROBE = PROBE_LIB + r'''
FM = {p.parent for n in ("face_ibv_train.npy",) for p in SCAN["files"].get(n, [])}     # optional: flag2027-feats-eval
FM = next(iter(FM), None)


def ib_ref(split, src, fi, vi):
    if FM is None or split != "train":
        return None
    Fx, Vx = l2n(np.load(FM / "face_ibv_train.npy")[fi]), l2n(np.load(FM / "voice_iba_train.npy")[vi])
    return np.sum(Fx * Vx, 1)


def pe(face_stem, centred=False, raw=False):
    def f(split, src, fi, vi):
        Fa, Va = load_arr(face_stem, "face", split, src), load_arr("peav", "voice", split, src)
        if Fa is None or Va is None:
            return None
        if raw:
            return np.sum(Fa[fi] * Va[vi], 1)
        if centred:                                     # train-side means of the probed rows (train / v1-v2 data only)
            Fa, Va = Fa - Fa.mean(0), Va - Va.mean(0)
        return np.sum(l2n(Fa[fi]) * l2n(Va[vi]), 1)
    return f


PR = run_probe({"PE-AV cosine (1 frame)": pe("peav"), "PE-AV cosine (4 frames)": pe("peavf4"),
                "PE-AV raw logit (1 frame)": pe("peav", raw=True), "PE-AV centred cosine": pe("peav", centred=True),
                "ImageBind (reference, if attached)": ib_ref})
CHECKS["peav_joint_max_abs_logit_diff"] = CHECKS_JOINT
CHECKS["peav_stats"] = PEAV_STATS          # face chunk in use, voices embedded by windows
# Reading guide: ImageBind zero-shot is about 25-27 EER (ng) on v1/v2 cross-video and 33-38 in g. Near 50 = no signal.
# Worth scoring locally: clearly below ImageBind in ANY g cell, or below 45 with errors unlike ImageBind's (checked locally).
'''


# ============================================================================ FLAG_34: layers + WeSpeaker
LAYERS_INTRO = """
# FLAG 2027 — NB-34: LAYER-ID-01, every hidden layer of frozen SSL speech models (+ WeSpeaker, fixed)

**Why.** EMBED-SCOUT (FLAG_27/28) tested only FINAL speaker embeddings, and all of them tied inside the relational
bridge. Paralinguistic cues that could link a voice to a face, such as age, physique and voice quality, are known to
peak in the MIDDLE layers of SSL speech models, not in the last one. This notebook keeps every layer, mean-pooled over
frames, so LAYER-ID-01 can test each layer, and learned layer mixtures, as voice profiles locally.

| array | model | pretraining data | shape per row |
|---|---|---|---|
| `voice_wlmbp` | microsoft/wavlm-base-plus | Libri-Light 60k h + GigaSpeech 10k h + VoxPopuli 24k h, **English only** | 13 x 768 |
| `voice_wlmlg` | microsoft/wavlm-large | same 94k h English | 25 x 1024 |
| `voice_hubl` | facebook/hubert-large-ll60k | Libri-Light 60k h English | 25 x 1024 |
| `voice_wsr221` | WeSpeaker ResNet221-LM | VoxCeleb2 | 256 |
| `voice_wssim34` | WeSpeaker SimAM-ResNet34 (VoxBlink2 -> VoxCeleb2) | VoxBlink2 + VoxCeleb2 (named by the organisers) | 256 |

**Rule check:** all frozen. The SSL models saw English audio only (no Bengali). The WeSpeaker models are VoxCeleb / VoxBlink2
encoders, explicitly allowed on 30/09.

**WeSpeaker fix:** in FLAG_27 both models failed with "Input type (torch.FloatTensor) and weight type
(torch.cuda.FloatTensor)". WeSpeaker's `extract_embedding_from_pcm` leaves the fbank on the CPU. Here the fbank is moved to
the model's device.

**Rows, inputs, settings:** as FLAG_27 (raw v4 `train_set` + `dev_set`, MAV-Celeb v1/v2 + `flag2027-ext-meta`; optional:
the previous output to resume). GPU T4, **Internet ON**.

**Time:** about 1-2.5 h per model (about 51 000 files), 6-8 h in all. The list `ENCODERS` is in priority order and the run is
resumable: if the session stops, Save Version, attach the output, run again.

**Output:** `/kaggle/working/feats_layers2/` -> dataset **`flag2027-feats-layers2`** (about 4 GB, float16).
"""

LAYERS_ENC = r'''
# ============================================================================ LAYER-ID encoders
class HFLayers:
    """Every hidden state of a frozen SSL model (CNN output + each Transformer layer), mean-pooled over frames, one file at
    a time (no padding enters a mean), float32 compute. Row = L x D values, flattened."""
    kind = "voice"

    def __init__(self, repo):
        from transformers import AutoFeatureExtractor, AutoModel
        self.fe = AutoFeatureExtractor.from_pretrained(repo, cache_dir=str(CACHE / "hf"))
        self.m = AutoModel.from_pretrained(repo, cache_dir=str(CACHE / "hf")).to(DEV).eval()
        self.repo = repo

    @torch.no_grad()
    def __call__(self, w, use_amp=False):
        x = self.fe(np.ascontiguousarray(w, dtype=np.float32), sampling_rate=TARGET_SR, return_tensors="pt")
        hs = self.m(x["input_values"].to(DEV), output_hidden_states=True).hidden_states
        return {"": torch.stack([h[0].float().mean(0) for h in hs]).reshape(-1).cpu().numpy()}


class WeSpeakerFixed(WeSpeakerVoice):
    """FLAG_27's WeSpeaker loader; embedding = model(fbank ON THE MODEL'S DEVICE) (the library leaves it on the CPU)."""
    @torch.no_grad()
    def emb(self, w):
        pcm = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32) * 32768.0)[None]
        feats = self.m.compute_features(pcm, sample_rate=TARGET_SR, cmn=True)
        out = self.m.model(feats.to(self.m.device))
        out = out[-1] if isinstance(out, tuple) else out
        return out[0].reshape(-1).float().cpu().numpy()


LAYER_MODELS = {"wlmbp": ("microsoft/wavlm-base-plus", 13, 768), "wlmlg": ("microsoft/wavlm-large", 25, 1024),
                "hubl": ("facebook/hubert-large-ll60k", 25, 1024)}
NEW = {**{k: (lambda r=r: HFLayers(r)) for k, (r, _, _) in LAYER_MODELS.items()},
       "wsr221": lambda: WeSpeakerFixed("english"), "wssim34": lambda: WeSpeakerFixed("vblinkf")}


def build(name):                             # only this cell's encoders (the FLAG_27 cell above is kept for WeSpeakerVoice)
    return NEW[name]()


ARRAYS_NEW = {k: ("voice", {"": k}) for k in NEW}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {f"voice_{k}": [L, D] for k, (_, L, D) in LAYER_MODELS.items()}
'''

LAYERS_PROBE = PROBE_LIB + r'''
# Per-layer VOICE <-> VOICE identity probe on v4 train (no face involved): EER of cosine between two clips of the same
# person vs two different persons of the same gender. It shows where speaker identity sits in each model; the face <->
# voice test runs locally (LAYER-ID-01).
rows = []
if "train" in S:
    nr = len(S["train"]["wav"])
    spk = pd.read_csv(S["train"]["txt"], sep=" ", header=None)[4].astype(str).values[:nr]
    meta = SCAN["files"].get("meta_file_train_set.csv")
    gmap = dict(pd.read_csv(meta[0], header=None).values) if meta else {}
    rng = np.random.RandomState(0)
    by = {s: np.where(spk == s)[0] for s in np.unique(spk)}
    ok = [i for i in range(nr) if len(by[spk[i]]) > 1]
    pos = [(i, rng.choice(by[spk[i]][by[spk[i]] != i])) for i in rng.choice(ok, 3000)] if ok else []
    neg, tries = [], 0
    while len(neg) < 3000 and tries < 300000:
        i, j = rng.randint(nr, size=2); tries += 1
        if spk[i] != spk[j] and gmap.get(spk[i]) == gmap.get(spk[j]) and spk[i] in gmap:
            neg.append((i, j))
    if pos and neg:
        pr = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        for k in NEW:
            A = load_arr(k, "voice", split="train")
            if A is None:
                continue
            shp = RESHAPE.get(f"voice_{k}", [1, A.shape[1]])
            A = l2n(A.reshape(len(A), *shp))
            for l in range(shp[0]):
                rows.append(dict(model=k, layer=l, eer_same_gender=round(eer(np.sum(A[pr[:, 0], l] * A[pr[:, 1], l], -1), y), 2)))
LP = pd.DataFrame(rows)
if len(LP):
    LP.to_csv(OUT / "probe_layers_voice_identity.csv", index=False)
    print(LP.pivot_table(index="layer", columns="model", values="eer_same_gender").round(2).to_string())
'''


# ============================================================================ FLAG_35: tokens
TOKENS_INTRO = """
# FLAG 2027 — NB-35: LOCAL-ID-01, ImageBind face patch tokens and per-clip audio embeddings

**Why.** Every scorer we use compares one face vector with one voice vector. ANATOMY-01 found the identity information
beyond gender and age to be weak and spread over many directions, and global pooling may wash it out. LOCAL-ID scores
a pair by **late interaction** between local tokens, e.g. s(F, V) = mean_i max_j cos(f_i, v_j) as in ColBERT, first
zero-shot, then with small learned projections (locally, on train persons only).

| array | content | shape per row |
|---|---|---|
| `face_ibtok` | ImageBind vision: the head (LayerNorm + projection) applied to EVERY trunk token. Row 0 = CLS (= the usual embedding before normalisation); rows 1-16 = the 16 x 16 patch tokens averaged over a 4 x 4 grid | 17 x 1024 |
| `voice_ibclip` | ImageBind audio: one embedding per 2 s clip (n = max(3, ceil(duration / 2)) clips evenly spaced, at most 12), each clip seen as its own sample; zero rows after the n clips | 12 x 1024 |
| `voice_ibclipn` | n, the number of real clips | 1 |

Both live in ImageBind's joint space, so late interaction works without any training.

**Rule check:** ImageBind as already used (allowed, declared); per-sample tokens use nothing outside the sample.

**Rows, inputs, settings:** as FLAG_27 (raw v4 `train_set` + `dev_set`, MAV-Celeb v1/v2 + `flag2027-ext-meta`; optional:
the previous output to resume). GPU T4, **Internet ON** (ImageBind checkpoint).

**Time:** about 2-3 h. Resumable.

**Output:** `/kaggle/working/feats_tokens/` -> dataset **`flag2027-feats-tokens`** (about 2.3 GB, float16).
"""

TOKENS_ENC = r'''
# ============================================================================ LOCAL-ID encoders (ImageBind tokens)
GRID, KCLIP = 4, 12


class IBVisionTokens(IBVision):
    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        m, mt = self.ib["m"], self.ib["MT"].VISION
        head = m.modality_heads[mt]
        assert isinstance(head[0], nn.LayerNorm) and isinstance(head[2], nn.Linear), head
        with amp(use_amp):
            inp = m.modality_preprocessors[mt](**{mt: x.to(DEV)})
            tok = m.modality_trunks[mt](**inp["trunk"])
            z = head[2](head[0](tok)).float()                              # (B, 1 + 16 * 16, 1024)
        B, P, D = z.shape
        side = int(round((P - 1) ** 0.5)); s = side // GRID
        g = z[:, 1:].reshape(B, GRID, s, GRID, s, D).mean((2, 4)).reshape(B, GRID * GRID, D)
        return {"": torch.cat([z[:, :1], g], 1).reshape(B, -1).cpu().numpy()}


def ib_audio_clips_k(w, kmax=KCLIP, sr=16000, clip=2.0):
    """ib_audio_clips with at most kmax clips (same placement rule: evenly spaced from start to end)."""
    n = min(max(3, int(np.ceil(len(w) / sr / clip))), kmax)
    mx = max(len(w) / sr - clip, 0.0)
    out = []
    for k in range(n):
        s = mx * k / max(n - 1, 1)
        seg = w[int(s * sr):int((s + clip) * sr)]
        out.append(ib_audio_clips(np.ascontiguousarray(seg, dtype=np.float32))[0])   # one clip -> 1 fbank (3 copies)
    return torch.stack(out)


class IBAudioClips(IBAudio):
    @torch.no_grad()
    def __call__(self, w, use_amp=True):
        x = ib_audio_clips_k(w)[:, None].to(DEV)                          # (n, 1, 1, 128, 204): n samples of 1 clip
        with amp(use_amp):
            e = self.ib["m"]({self.ib["MT"].AUDIO: x})[self.ib["MT"].AUDIO].float().cpu().numpy()
        out = np.zeros((KCLIP, e.shape[1]), np.float32); out[:len(e)] = e
        return {"": out.reshape(-1), "n": np.array([len(e)], np.float32)}


NEW = {"ibtokV": IBVisionTokens, "ibtokA": IBAudioClips}
_build12 = build


def build(name):
    return NEW[name]() if name in NEW else _build12(name)


ARRAYS_NEW = {"ibtokV": ("face", {"": "ibtok"}), "ibtokA": ("voice", {"": "ibclip", "n": "ibclipn"})}
MAIN_NEW = {"ibtokV": "", "ibtokA": ""}
RESHAPE = {"face_ibtok": [1 + GRID * GRID, 1024], "voice_ibclip": [KCLIP, 1024]}


@torch.no_grad()
def tokens_check():
    """CLS row of face_ibtok must equal IBVision; the mean of the clip rows must equal IBAudio for files <= 24 s."""
    k = next(iter(S)); sp = S[k]
    tv, iv = IBVisionTokens(), IBVision()
    x = torch.stack([iv.prep(read_rgb(sp["root"] / p)) for p in sp["jpg"][:4]])
    a = l2n_t(tv(x, use_amp=False)[""].reshape(len(x), -1, 1024)[:, 0]); b = iv(x, use_amp=False)[""]
    cv = float(cos_rows(a, b).min())
    ta, ia = IBAudioClips(), IBAudio()
    cs = []
    for p in sp["wav"][:4]:
        w = load_audio(sp["root"] / p)[:TARGET_SR * 20]
        r = ta(w, use_amp=False); n = int(r["n"][0])
        cs.append(float(cos_rows(r[""].reshape(KCLIP, -1)[:n].mean(0, keepdims=True), ia(w, use_amp=False)[""][None])[0]))
    log(f"tokens check on {k}: CLS vs ImageBind vision min cos {cv:.5f}; clip mean vs ImageBind audio min cos {min(cs):.5f}")
    return {"cls_vs_ibv": cv, "clipmean_vs_iba": min(cs)}


l2n_t = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
try:
    CHECKS_TOK = tokens_check()
except Exception as e:
    CHECKS_TOK = f"{type(e).__name__}: {e}"
    log("tokens check failed:", CHECKS_TOK)
'''

TOKENS_PROBE = PROBE_LIB + r'''
def tok_arrays(split, src):
    Fa, Va = load_arr("ibtok", "face", split, src), load_arr("ibclip", "voice", split, src)
    Na = load_arr("ibclipn", "voice", split, src)
    if Fa is None or Va is None or Na is None:
        return None
    return l2n(Fa.reshape(len(Fa), -1, 1024)), l2n(Va.reshape(len(Va), KCLIP, 1024)), Na.reshape(-1).astype(int)


def li(mode):
    def f(split, src, fi, vi):
        T = tok_arrays(split, src)
        if T is None:
            return None
        Fa, Va, Na = T
        out = np.empty(len(fi))
        for t, (i, j) in enumerate(zip(fi, vi)):
            v = Va[j, :Na[j]]
            if mode == "global":                                        # CLS vs mean of clips (= the usual ib stream)
                out[t] = float(Fa[i, 0] @ l2n(v.mean(0)))
            else:
                C = (Fa[i, 1:] if mode.startswith("patch") else Fa[i, :1]) @ v.T     # (tokens, clips)
                out[t] = {"patch_maxmean": C.max(1).mean(), "patch_mean": C.mean(), "cls_max": C.max(),
                          "patch_softmax": (np.exp(10 * C) / np.exp(10 * C).sum(1, keepdims=True) * C).sum(1).mean()}[mode]
        return out
    return f


PR = run_probe({f"IB {m}": li(m) for m in ("global", "cls_max", "patch_maxmean", "patch_mean", "patch_softmax")})
CHECKS["tokens"] = CHECKS_TOK
# Reading guide: "IB global" reproduces the ImageBind stream. Late interaction is worth the local test (LOCAL-ID-01,
# with learned projections) only if a zero-shot variant is not clearly worse than global, or wins any g cell.
'''


def nb_cells(intro, envc, pip, out, enc_cells, probe, dataset, keep_ib=()):
    return [("markdown", intro), ("code", envc), ("code", pip),
            ("markdown", "## 1. Find the data (FLAG_12 cell): organiser lists with media, v1/v2 rows, resume"), ("code", data(out)),
            ("markdown", "## 2. Audio loading, batching, helpers (FLAG_12 cell)"), ("code", B12.AUDIO12),
            ("markdown", "## 3. Encoders"), *enc_cells,
            ("markdown", "## 4. Extract every encoder over every split, then the v1/v2 rows (FLAG_12 loop, resumable)"),
            ("code", run(keep_ib)),
            ("markdown", "## 5. Zero-shot probe on true labels (v4 train, MAV-Celeb cross-video; never the dev lists)"),
            ("code", probe),
            ("markdown", "## 6. Manifest"), ("code", done(dataset))]


if __name__ == "__main__":
    B12.nb(nb_cells(PEAV_INTRO, env("feats_peav", ["peavV", "peavA"], "faces first (cheap), then voices"), PEAV_PIP,
                    "feats_peav", [("code", B12.ENC12), ("code", PEAV_ENC)], PEAV_PROBE, "flag2027-feats-peav"),
           "FLAG_33_peav.ipynb")
    B12.nb(nb_cells(LAYERS_INTRO, env("feats_layers2", ["wlmbp", "wlmlg", "wsr221", "wssim34", "hubl"], "priority order"),
                    B27.PIP, "feats_layers2", [("code", B12.ENC12), ("code", B27.ENC_NEW), ("code", LAYERS_ENC)],
                    LAYERS_PROBE, "flag2027-feats-layers2"),
           "FLAG_34_layers.ipynb")
    B12.nb(nb_cells(TOKENS_INTRO, env("feats_tokens", ["ibtokV", "ibtokA"], "one ImageBind load for both"), B12.PIP12,
                    "feats_tokens", [("code", B12.ENC12), ("code", TOKENS_ENC)], TOKENS_PROBE, "flag2027-feats-tokens",
                    keep_ib=("ibtokV",)),
           "FLAG_35_tokens.ipynb")
