"""Generate FLAG_103c_tap01_voice.ipynb (TAP-01 step 1, Experiment/TAP-01/README.md, registered 29/10 before any result): the frozen
sam100p (WeSpeaker SimAM-ResNet100, VoxBlink2-pretrained) on the centre 3 s of every voice row, three arrays per row:
  sam100p3s  the 256-d embedding (3 s cut; the stored sam100p used 12 s)
  samS4      the pooling output before the bottleneck (statistics of the last stage)
  samS3      mean | std over time of the stage-3 output (channels x frequency)
MAV-Celeb rows (v4 train, v1 / v2, the test lists) through the FLAG_12 loop (FLAG_67x's cut: centre 3 s, zero-padded at the end);
VoxCeleb2 / VoxBlink rows from the audio cache (already the centre 3 s), in each source's voice_meta.csv row order (NaN = no clip).
    python build_tap_voice_notebook.py
Smoke test on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6;TAP_AUX_MAX=128
"""
import build_eval_notebooks as B12
import build_rep_notebooks as BR
import build_t3enc_notebook as T3
import build_vlx_notebook as VLXB

SMOKE = '(LOCAL or os.environ.get("FLAG_SMOKE") == "1")'
OUT, DS, NB = "feats_tap01_voice", "flag2027-feats-tap01-voice-p", "FLAG_103c_tap01_voice.ipynb"

TAPV = r'''# ============================================================================ TAP-01: frozen sam100p, centre 3 s, three arrays per clip
import torch
''' + VLXB.VLX.split("class AdaptedVoice(SimAMVoice):")[0].split("import torch\n", 1)[1] + r'''

def _crop3(w):
    w = np.asarray(w, np.float32)
    L = 3 * 16000
    if len(w) > L:
        o = (len(w) - L) // 2
        w = w[o:o + L]
    elif len(w) < L:
        w = np.pad(w, (0, L - len(w)))
    return w


class TapVoice(SimAMVoice):
    """sam100p with forward hooks on the pooling layer (S4) and on layer3 (S3)."""

    def __init__(self):
        super().__init__("sam100pA")
        net = self.m.model
        kids = [n for n, _ in net.named_children()]
        print(f"TapVoice: {type(net).__name__}, children {kids}", flush=True)
        pn = next(n for n in ("pool", "pooling", "stats_pool") if hasattr(net, n))
        l3 = [n for n, _ in net.named_modules() if n.split(".")[-1] == "layer3"]   # SimAM_ResNet100_ASP: front.layer3
        assert len(l3) == 1, ("layer3 not unique", l3)
        self.bn = next((getattr(net, n) for n in ("bottleneck", "seg_1") if hasattr(net, n)), None)
        self.h, self.checked = {}, False
        getattr(net, pn).register_forward_hook(lambda mod, i, o: self.h.__setitem__("S4", o[-1] if isinstance(o, tuple) else o))
        net.get_submodule(l3[0]).register_forward_hook(lambda mod, i, o: self.h.__setitem__("S3", o))
        T3_STATS["samTapA"] = {"children": kids, "pool": f"{pn}: {type(getattr(net, pn)).__name__}", "layer3": l3[0],
                               "bottleneck": type(self.bn).__name__, "crop_s": 3}
        print(f"TapVoice: pooling {pn} = {type(getattr(net, pn)).__name__}, stage 3 = {l3[0]}, bottleneck {type(self.bn).__name__}", flush=True)

    def stats(self, n):
        x3 = self.h["S3"]
        x3 = x3.reshape(x3.shape[0], -1, x3.shape[-1]).float()                   # (B, C x F, T)
        return self.h["S4"].reshape(n, -1).float(), torch.cat([x3.mean(-1), x3.std(-1)], 1)

    @torch.no_grad()
    def __call__(self, w, use_amp=True):
        e = super().__call__(_crop3(w), use_amp)[""]
        s4, s3 = self.stats(1)
        if not self.checked and self.bn is not None:                               # the bottleneck of S4 must give the embedding
            r = self.bn(s4.to(next(self.bn.parameters()).device, next(self.bn.parameters()).dtype))[0].float().cpu().numpy()
            c = float(r @ e / (np.linalg.norm(r) * np.linalg.norm(e) + 1e-8))
            T3_STATS["samTapA"]["bottleneck_cos"] = round(c, 6)
            print(f"TapVoice wiring: cos(bottleneck(S4), embedding) = {c:.6f}", flush=True)
            assert c > 0.9999, c
            self.checked = True
        return {"": e, "S4": s4[0].cpu().numpy(), "S3": s3[0].cpu().numpy()}


NEW = {"samTapA": TapVoice}
ARRAYS_NEW = {"samTapA": ("voice", {"": "sam100p3s", "S4": "samS4", "S3": "samS3"})}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {}
print("TAP-01 voice cell ready: encoders", list(NEW), flush=True)
'''

AUX = r'''
# ============================================================================ TAP-01: VoxCeleb2 / VoxBlink rows from the audio cache (centre 3 s, as cached)
import pandas as pd
SMOKE = os.environ.get("FLAG_SMOKE") == "1"          # 29/10: the real run had no SMOKE name (smoke never reached it)
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
dt = {"spk": str, "video": str, "utt": str}
VM = {s: pd.read_csv(p, dtype=dt) for s, p in _vm.items()}
ROW = {s: {k: r for r, k in enumerate(zip(d.spk, d.video, d.utt))} for s, d in VM.items()}
print("audio cache", ACROOT, len(CH), "chunks; voice rows", {s: len(d) for s, d in VM.items()}, flush=True)
TV = TapVoice()
NET = TV.m.model
NAMES = {"": "sam100p3s", "S4": "samS4", "S3": "samS3"}


@torch.no_grad()
def tap_batch(ws):
    fs = [TV.m.compute_features(torch.from_numpy(np.asarray(w, np.float32))[None], sample_rate=16000, cmn=True)[0] for w in ws]
    T = max(f.shape[0] for f in fs)
    X = torch.stack([F.pad(f, (0, 0, 0, T - f.shape[0])) for f in fs]).to(TV.m.device)
    e = NET(X)
    e = (e[-1] if isinstance(e, tuple) else e).float()
    s4, s3 = TV.stats(len(ws))
    return {"sam100p3s": e.cpu().numpy(), "samS4": s4.cpu().numpy(), "samS3": s3.cpu().numpy()}


import torch.nn.functional as F
_w0 = np.load(CH[0][0])["wav"][0]
_a = tap_batch([_w0])["samS4"][0]; _b = TV(_w0.astype(np.float32) / 32768.0)["S4"]
AUX_WIRING = float(_a @ _b / (np.linalg.norm(_a) * np.linalg.norm(_b) + 1e-8))
print(f"aux path vs TapVoice on the same cached clip: cos(S4) = {AUX_WIRING:.6f}", flush=True)
assert AUX_WIRING > 0.999, AUX_WIRING
lim = int(os.environ.get("TAP_AUX_MAX", "0")) or (128 if SMOKE else 0)
A, n_done, t0 = {}, 0, time.time()
for p, kind in CH:
    Z = np.load(p)
    keys = [tuple(map(str, k)) for k in zip(Z["spk"], Z["video"], Z["utt"])]
    for c0 in range(0, len(keys), 64):
        E = tap_batch(list(Z["wav"][c0:c0 + 64]))
        for s in (("vb",) if kind == "vb" else ("vox", "vox2b")):
            for name, X in E.items():
                if (s, name) not in A:
                    A[s, name] = np.full((len(VM[s]), X.shape[1]), np.nan, np.float16)
            for j, k in enumerate(keys[c0:c0 + 64]):
                r = ROW[s].get(k)
                if r is not None:
                    for name, X in E.items():
                        A[s, name][r] = X[j]
        n_done += len(keys[c0:c0 + 64])
        if lim and n_done >= lim:
            break
    if n_done and n_done % 20000 < 64:
        print(f"[aux] {n_done} clips [{time.time() - t0:.0f}s]", flush=True)
    if lim and n_done >= lim:
        break
for (s, name), X in A.items():
    (OUT / s).mkdir(exist_ok=True); np.save(OUT / s / f"voice_{name}.npy", X)
AUX_COVER = {f"{s}/{name}": int(np.isfinite(X[:, 0].astype(np.float32)).sum()) for (s, name), X in A.items()}
print(f"[aux] {n_done} clips [{time.time() - t0:.0f}s]; rows per array {AUX_COVER}", flush=True)
CHECKS["tap01_aux"] = dict(wiring=AUX_WIRING, cover=AUX_COVER, seg1=T3_STATS.get("samTapA"))
UP = os.environ.get("TAP_UPLOAD", "")
if UP and not SMOKE:
    try:
        import kagglehub
        kagglehub.dataset_upload(UP, str(OUT), version_notes="TAP-01: sam100p 3 s taps (sam100p3s, samS4, samS3) of MAV + VoxCeleb2 / VoxBlink rows")
        print("uploaded ->", UP, flush=True)
    except Exception as e:
        print(f"upload to {UP} failed ({e!r}); the arrays are in the notebook output", flush=True)
'''

INTRO = """
# FLAG 2027 — NB-103c: TAP-01 step 1, pre-bottleneck taps of the frozen sam100p (centre 3 s) for every voice row

**Why.** The 256-d SimAM embedding is the AAM bottleneck of the pooled statistics: deep speaker nets keep less pitch / timbre there
(a voice-to-face cue). The FAME 2026 winner fed ECAPA's 6144-d pooled statistics (before the bottleneck) to a linear head.
Registered in `Experiment/TAP-01/README.md` (29/10, before any result).

| array | tap | cut |
|---|---|---|
| `sam100p3s` | the 256-d embedding | centre 3 s |
| `samS4` | pooling output before the bottleneck (checked: `seg_1(S4)` = the embedding) | centre 3 s |
| `samS3` | mean, std over time of the stage-3 output | centre 3 s |

One cut for every row (the audio cache holds the centre 3 s of the VoxCeleb2 / VoxBlink clips), a fixed rule chosen before any
result, not derived from any test file. **Rule check:** frozen pretrained encoder (declared); no Bengali speech trains anything; test
files only embedded, one at a time (level 0).

**Inputs:** `ngnnguynchn/flag2027-raw-v4-p`, `chnnhnnguyn/external-data-mavceleb`, `ngnnguynchn/flag2027-ext-meta-p`,
`ngnnguynchn/flag2027-lab-data-p`, `ngnnguynchn/flag2027-feats-vox2b-p`, `ngnnguynchn/flag2027-lab-vb-p`,
`duongphihoang/flag2027-audio-cache-p`. **GPU T4**, **Internet ON** (WeSpeaker zip), **private** (VoxBlink).
**Output:** `feats_tap01_voice/` (MAV arrays + `vox/`, `vox2b/`, `vb/`), uploaded to `TAP_UPLOAD` when set. **Time:** about 2 h.
"""


_CLIP = ' "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"'
assert T3.PIP.count(_CLIP) == 1
PIP = T3.PIP.replace(_CLIP, "")          # 29/10 smoke: the openai CLIP build fails on Kaggle's image and took wespeaker down with it


def cells():
    out = BR.nb_cells(INTRO, BR.env(OUT, ["samTapA"], "TAP-01 taps of the frozen sam100p"),
                      PIP, OUT, [("code", VLXB.T3CELL), ("code", TAPV)], VLXB.PROBE.replace("sam100fL", "sam100p3s"), DS)
    res, n_env, n_data = [], 0, 0
    for kind, src in out:
        if kind == "code" and "if LOCAL and os.environ.get(" in src:
            src = src.replace("if LOCAL and os.environ.get(", f"if {SMOKE} and os.environ.get("); n_env += 1
        if kind == "code" and "if LOCAL and SMOKE_ROWS:" in src:
            src = src.replace("if LOCAL and SMOKE_ROWS:", f"if {SMOKE} and SMOKE_ROWS:"); n_data += 1
        res.append((kind, src))
    assert n_env == 1 and n_data == 1, (n_env, n_data)
    return res + [("markdown", "## TAP-01: VoxCeleb2 / VoxBlink rows from the audio cache"), ("code", AUX)]


if __name__ == "__main__":
    B12.nb(cells(), NB)
    print(NB, "built")
