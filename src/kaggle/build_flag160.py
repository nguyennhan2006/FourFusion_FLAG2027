"""Generate FLAG_160{a,b,c}_whisper_<x>.ipynb (WHISPER-01, Experiment/WHISPER-01/README.md, nhat ky 354): the frozen Whisper-large-v3
ENCODER as one more voice block (the RF-092 recipe), for every voice row the CC / CA heads use.
  a : MAV-Celeb rows (v4 train, v1 / v2, the four dev lists) through the FLAG_12 loop (FLAG_103c's route)
  b, c : the audio cache (VoxCeleb2 FLAG_37 + FLAG_53, VoxBlink FLAG_46), two interleaved halves of the chunk list, 2 GPUs each
Array `whv3` (1280-d float16; `voice_whv3_<split>.npy`, `ext_<src>_voice_whv3.npy`, `<vox|vox2b|vb>/voice_whv3.npy`; rows not done in a lane = NaN).
    python build_flag160.py     -> kaggle/FLAG_160a_whisper_mav.ipynb, FLAG_160b_whisper_auxb.ipynb, FLAG_160c_whisper_auxc.ipynb
Smoke on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6;WH_AUX_MAX=64
"""
import build_eval_notebooks as B12
import build_rep_notebooks as BR

SMOKE = '(LOCAL or os.environ.get("FLAG_SMOKE") == "1")'
REPO, REV = "openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"
MODES = {"a": ("FLAG_160a_whisper_mav.ipynb", "mav"), "b": ("FLAG_160b_whisper_auxb.ipynb", "auxb"),
         "c": ("FLAG_160c_whisper_auxc.ipynb", "auxc")}

WH = r'''# ============================================================================ WHISPER-01: frozen Whisper-large-v3 encoder, mean of the first-30 s frames
import math, torch
WH_REPO, WH_REV = "__REPO__", "__REV__"
WH_MODE = "__MODE__"
if WH_MODE != "mav":                       # aux lanes: the MAV loop of this framework does nothing (no encoder)
    ENCODERS = EXT_ENCODERS = []


class WhisperV3:
    """Whisper-large-v3 encoder only, fp16 (fp32 on CPU). x = feature_extractor(w[:30 s]); h = encoder(x).last_hidden_state (1500 frames);
    n = max(1, min(1500, ceil(len(w) / 16000 / 30 * 1500))); vector = h[:n].mean(0) (1280-d). Padding frames beyond the clip are dropped."""
    kind = "voice"

    def __init__(self, dev=None):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.dev = torch.device(dev) if dev is not None else DEV
        self.dt = torch.float16 if self.dev.type == "cuda" else torch.float32
        self.fe = WhisperFeatureExtractor.from_pretrained(WH_REPO, revision=WH_REV, cache_dir=str(CACHE / "hf"))
        m = WhisperModel.from_pretrained(WH_REPO, revision=WH_REV, torch_dtype=self.dt, cache_dir=str(CACHE / "hf"))
        self.enc = m.encoder.to(self.dev).eval()
        del m

    @torch.no_grad()
    def embed(self, ws):
        ws = [np.ascontiguousarray(w[:16000 * 30], dtype=np.float32) for w in ws]
        try:                                                       # STFT on the GPU (the numpy one is the bottleneck)
            x = self.fe(ws, sampling_rate=16000, return_tensors="pt", device=str(self.dev)).input_features
        except TypeError:
            x = self.fe(ws, sampling_rate=16000, return_tensors="pt").input_features
        h = self.enc(x.to(self.dev, self.dt)).last_hidden_state
        out = []
        for i, w in enumerate(ws):
            n = max(1, min(h.shape[1], math.ceil(len(w) / 16000 / 30 * h.shape[1])))
            out.append(h[i, :n].float().mean(0).cpu().numpy())
        return np.stack(out)

    def __call__(self, w, use_amp=True):
        return {"": self.embed([w])[0]}


NEW = {"whv3": WhisperV3}
ARRAYS_NEW = {"whv3": ("voice", {"": "whv3"})}
MAIN_NEW = {"whv3": ""}
RESHAPE = {}


def build(name):
    return NEW[name]()


if WH_MODE == "mav":                       # wiring: batched = single, GPU STFT = CPU STFT (cos > 0.999)
    _m = WhisperV3(); _r = np.random.default_rng(0)
    _w = [(_r.standard_normal(16000 * k) * 0.05).astype(np.float32) for k in (3, 7)]
    _b = _m.embed(_w); _s = np.stack([_m.embed([w])[0] for w in _w])
    _c = float((np.sum(_b * _s, 1) / (np.linalg.norm(_b, axis=1) * np.linalg.norm(_s, axis=1))).min())
    print(f"WhisperV3 wiring: dim {_b.shape[1]}, min cos(batched, single) = {_c:.6f}", flush=True)
    assert _b.shape[1] == 1280 and _c > 0.999, _c
    del _m
    release()
print("WHISPER-01 cell ready, mode", WH_MODE, flush=True)
'''

AUX = r'''print("[cell] whisper aux: audio-cache clips", flush=True)
# VoxCeleb2 (vox = FLAG_37 + FLAG_53 more rows, vox2b = FLAG_53 new) and VoxBlink rows from the audio cache, in each source's voice_meta.csv row order
import pandas as pd, threading
import torch.nn.functional as F
SMOKE = os.environ.get("FLAG_SMOKE") == "1"


def _inroots():
    base = Path("/kaggle/input")
    out = []
    for p in (base.iterdir() if base.exists() else []):
        if p.name in ("datasets", "notebooks", "kernels", "models"):
            out += [q for o in p.iterdir() if o.is_dir() for q in o.iterdir() if q.is_dir()]
        elif p.is_dir():
            out.append(p)
    return [p for p in out if not any(k in p.name for k in ("mavceleb", "raw-v4", "ext-meta"))]


_roots = _inroots()
_vm = {}
for r in _roots:
    for p in r.rglob("voice_meta.csv"):
        for s, par in (("vox", "feats_vox2"), ("vox2b", "feats_vox2b"), ("vb", "feats_vb")):
            if p.parent.name == par:
                _vm.setdefault(s, p)
assert set(_vm) == {"vox", "vox2b", "vb"}, f"voice_meta.csv found for {sorted(_vm)}: attach lab-data / feats-vox2b / lab-vb"
_tar = sorted(q for r in _roots for q in r.rglob("tar0000.npz") if q.parent.name == "vox")
assert _tar, "attach duongphihoang/flag2027-audio-cache-p (vox/tar*.npz, vb/*.npz)"
ACROOT = _tar[0].parent.parent
CH = [(p, "vox") for p in sorted((ACROOT / "vox").glob("tar*.npz"))] + [(p, "vb") for p in sorted((ACROOT / "vb").glob("*.npz"))]
CH = CH[(1 if WH_MODE == "auxc" else 0)::2]                    # b = even chunks, c = odd chunks
dt = {"spk": str, "video": str, "utt": str}
VM = {s: pd.read_csv(p, dtype=dt) for s, p in _vm.items()}
ROW = {s: {k: r for r, k in enumerate(zip(d.spk, d.video, d.utt))} for s, d in VM.items()}
print("audio cache", ACROOT, len(CH), "chunks of this lane; voice rows", {s: len(d) for s, d in VM.items()}, flush=True)
DEVS = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
MS = [WhisperV3(d) for d in DEVS]
BS = 16 if DEVS[0] != "cpu" else 2


def wh_many(ws):
    """the clips split over every GPU, BS per forward"""
    res, parts = [None] * len(MS), np.array_split(np.arange(len(ws)), len(MS))
    def run(k):
        idx = parts[k]
        res[k] = np.concatenate([MS[k].embed([ws[j] for j in idx[b:b + BS]]) for b in range(0, len(idx), BS)]) if len(idx) else np.zeros((0, 1280), np.float32)
    th = [threading.Thread(target=run, args=(k,)) for k in range(len(MS))]
    [t.start() for t in th]; [t.join() for t in th]
    return np.concatenate(res)


def clips(Z, a, b):
    W = Z["wav"][a:b]
    ln = Z["n"][a:b].astype(int) if "n" in Z.files else [W.shape[1]] * len(W)
    return [W[i][:max(int(ln[i]), 16000 // 2)].astype(np.float32) / 32768.0 for i in range(len(W))]


_w0 = clips(np.load(CH[0][0]), 0, 2)
_g = wh_many(_w0)
_c = MS[0].embed([_w0[0]])[0]
AUX_WIRING = float(_g[0] @ _c / (np.linalg.norm(_g[0]) * np.linalg.norm(_c) + 1e-8))
print(f"aux path vs single clip: cos = {AUX_WIRING:.6f}", flush=True)
assert AUX_WIRING > 0.999, AUX_WIRING
lim = int(os.environ.get("WH_AUX_MAX", "0")) or (64 if SMOKE else 0)
A, n_done, t0 = {}, 0, time.time()
for ci, (p, kind) in enumerate(CH):
    Z = np.load(p)
    keys = [tuple(map(str, k)) for k in zip(Z["spk"], Z["video"], Z["utt"])]
    for c0 in range(0, len(keys), 64):
        E = wh_many(clips(Z, c0, c0 + 64))
        for s in (("vb",) if kind == "vb" else ("vox", "vox2b")):
            if s not in A:
                A[s] = np.full((len(VM[s]), 1280), np.nan, np.float16)
            for j, k in enumerate(keys[c0:c0 + 64]):
                r = ROW[s].get(k)
                if r is not None:
                    A[s][r] = E[j]
        n_done += len(E)
        if lim and n_done >= lim:
            break
    print(f"[aux] chunk {ci + 1}/{len(CH)}: {n_done} clips ({n_done / max(time.time() - t0, 1):.1f} clips/s, {time.time() - t0:.0f}s)", flush=True)
    if lim and n_done >= lim:
        break
for s, X in A.items():
    (OUT / s).mkdir(exist_ok=True); np.save(OUT / s / "voice_whv3.npy", X)
AUX_COVER = {f"{s}/whv3": int(np.isfinite(X[:, 0].astype(np.float32)).sum()) for s, X in A.items()}
print(f"[aux] {n_done} clips [{time.time() - t0:.0f}s]; rows per array {AUX_COVER}", flush=True)
CHECKS["whisper01_aux"] = dict(wiring=AUX_WIRING, cover=AUX_COVER, chunks=len(CH), mode=WH_MODE, clips=n_done)
'''

UP = r'''print("[cell] upload", flush=True)
UPL = os.environ.get("WH_UPLOAD", "")
(OUT / "whisper01_meta.json").write_text(json.dumps(dict(repo=WH_REPO, revision=WH_REV, mode=WH_MODE, dim=1280, dtype="float16",
        recipe="first 30 s, encoder last_hidden_state, mean of the first n frames, n=max(1,min(1500,ceil(len/16000/30*1500)))",
        rows="a: split / ext arrays in the FLAG_12 order; b, c: <vox|vox2b|vb>/voice_whv3.npy in voice_meta.csv order, NaN = not in this lane (b = even chunks, c = odd chunks of the cache list)",
        unreadable_file="a: framework stand-in (1 s of silence), listed in feats_manifest.json broken")))
if UPL and os.environ.get("FLAG_SMOKE") != "1":
    try:
        import kagglehub
        kagglehub.dataset_upload(UPL, str(OUT), version_notes=f"WHISPER-01 whv3 lane {WH_MODE}")
        print("uploaded ->", UPL, flush=True)
    except Exception as e:
        print(f"upload to {UPL} failed ({e!r}); the arrays are in the notebook output", flush=True)
print("ALL CELLS OK", flush=True)
'''

INTRO = """
# FLAG 2027 — NB-160__X__: WHISPER-01, frozen Whisper-large-v3 encoder as a voice block (lane __MODE__)

**Why.** RF-L2-045 / 082 (RF-092 review, docs/RF092_REVIEW_2026-11-03.md section 3 item 4): the biggest single encoder gain of a teammate's system
(CB -0.69; dropping it costs +0.51). We have only used Whisper for language ID. This notebook only **extracts** the feature; learning happens later
(Experiment/WHISPER-01/README.md). Frozen pretrained model: allowed by the organisers (17/10), declared. No Bengali filtering at extraction (done at learning time, `LID_STRICT=1`).

`openai/whisper-large-v3` @ `06f233fe...`, encoder only, fp16; first 30 s; vector = mean of the frames covering the clip (1280-d).
Lane a = MAV-Celeb rows (v4 train, v1/v2, dev lists); lanes b / c = two halves of the audio cache (VoxCeleb2 / VoxBlink, centre 3 s clips).
**Private** (VoxBlink rows). GPU T4 x 2, Internet ON.
"""


def cells(x):
    nb, mode = MODES[x]
    out = BR.nb_cells(INTRO.replace("__X__", x).replace("__MODE__", mode), BR.env(f"feats_whv3_{x}", ["whv3"], "Whisper-large-v3 encoder"),
                      B12.PIP12, f"feats_whv3_{x}", [("code", B12.ENC12), ("code", WH.replace("__REPO__", REPO).replace("__REV__", REV).replace("__MODE__", mode))],
                      BR.LAYERS_PROBE.replace("for k in NEW:", 'for k in ("whv3",):'), f"flag2027-feats-whv3-{x}")
    res, n_env, n_data = [], 0, 0
    for kind, src in out:
        if kind == "code" and "if LOCAL and os.environ.get(" in src:
            src = src.replace("if LOCAL and os.environ.get(", f"if {SMOKE} and os.environ.get("); n_env += 1
        if kind == "code" and "if LOCAL and SMOKE_ROWS:" in src:
            src = src.replace("if LOCAL and SMOKE_ROWS:", f"if {SMOKE} and SMOKE_ROWS:"); n_data += 1
        res.append((kind, src))
    assert n_env == 1 and n_data == 1, (n_env, n_data)
    if mode != "mav":
        res += [("markdown", "## WHISPER-01: audio-cache rows"), ("code", AUX)]
    res += [("code", UP)]
    k = 0
    for i, (kind, src) in enumerate(res):                      # luat notebook Kaggle: dong dau moi cell code la print("[cell] ...")
        if kind == "code":
            k += 1
            assert not src.lstrip().startswith("%%"), src[:40]
            if not src.lstrip().startswith('print("[cell]'):
                res[i] = (kind, f'print("[cell] {x}{k}", flush=True)\n' + src.lstrip("\n"))
    return res


if __name__ == "__main__":
    for x in MODES:
        B12.nb(cells(x), MODES[x][0])
        print(MODES[x][0], "built")
