"""Generate FLAG_10_models.ipynb and FLAG_11_layers.ipynb. FLAG_10: MODEL-01 extraction (docs/PLAN_MODELS.md §3 S1 + §4 X3 probe).

Extra frozen encoders (flag_models.py) on v4 train + dev and on the MAV-Celeb v1/v2 validation rows, plus a probe
table (identity EER of each stream, ImageBind zero-shot face<->voice EER, attribute sanity). No training here: the
bridge experiments run on CPU from the saved arrays.
Local smoke test: FLAG_LOCAL=1 FLAG_IN=<mini data> FLAG_OUT=<dir> FLAG_CACHE=<model cache>.
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def embed_lib(*names):
    parts = ["# Writes the helper modules next to this notebook (no .py upload needed).", "import base64"]
    for n in names:
        b = base64.b64encode((HERE / n).read_text(encoding="utf-8").encode()).decode()
        b = "\n".join(b[i:i + 100] for i in range(0, len(b), 100))
        parts.append(f"open({n!r}, 'wb').write(base64.b64decode(''.join('''\n{b}\n'''.split())))")
    parts.append(f"print('written:', {list(names)!r})")
    return "\n".join(parts) + "\n"


def nb(cells, path):
    d = {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                    **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
         "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11"}},
         "nbformat": 4, "nbformat_minor": 5}
    (HERE / path).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", path, len(cells), "cells")


INTRO = """
# FLAG 2027 — NB-10: extra encoders for the bridge (MODEL-01) + ImageBind probe

Plan: `docs/PLAN_MODELS.md` §3 S1 and §4 X3. The bridge VGG <-> ECAPA-BTC is the bottleneck; recognition-grade encoders
(ArcFace, ReDimNet) made it worse. These encoders are chosen for **soft attributes** and are *added* as streams:

| stream | model | dim | role |
|---|---|---:|---|
| `face_vitage` (+ `face_vitageprob`) | nateraw/vit-age-classifier (FairFace) | 768 (+9) | attribute stream, as the FAME26 winner's ViT age-gender |
| `face_farl` | FaRL ViT-B/16, LAION-Face 20M | 512 | general face representation |
| `face_siglip2` | google/siglip2-base-patch16-224 | 768 | general image semantics |
| `face_adaface` | AdaFace IR-101 WebFace12M (CVLFace) | 512 | control: recognition-grade, same crop as ArcFace |
| `face_ibv` / `voice_iba` | ImageBind-huge vision / audio | 1024 | **probe only** (rules: pending organiser answer) |
| `voice_w2v2ag` (+ `age`, `gen`) | audeering wav2vec2-large-robust age-gender | 1024 (+1, +3) | voice attribute stream |

**Settings:** Accelerator **GPU T4 x2** (one GPU used), **Internet ON** (checkpoints: ~9 GB, cached in `/tmp`, not in the output).

**Inputs (Add Input):**
1. the raw v4 dataset (`Input/`: `train_set.zip` + `dev_set.zip` + `meta_file_train_set.csv`), as for FLAG_04;
2. *optional, for Urdu/Hindi validation*: the MAV-Celeb v1 / v2 datasets (`mavceleb_v1_complete*.zip`, `..._v2_...`)
   **and** a small dataset `flag2027-ext-meta`: upload `kaggle/flag2027-ext-meta.zip` (0.4 MB, the four
   `ext_v{1,2}_complete_{face,voice}_meta.csv` of `kaggle/output/feats_ext_full/`, a superset of `feats_ext`).
   Without them v1/v2 is skipped.

**Outputs:** `/kaggle/working/feats_models/` -> make it the dataset **`flag2027-feats-models`**. Arrays keep the organiser row
order (v4) or the existing ext meta row order (v1/v2, fp16), so they line up with the VGG / BTC features.
`probe.csv` + `probe.json`: identity EER per stream, ImageBind zero-shot EER, attribute sanity.

**Resume after a failure:** attach the earlier version's output dataset (`flag2027-feats-models`) as an extra input.
Its arrays are copied into the output and every encoder that is already complete is skipped, so only the missing ones run.

**Safety:** one encoder is loaded at a time (ImageBind shared by vision and audio); an encoder that fails is skipped and
reported, it never kills the run. Batched-vs-single and fp16-vs-fp32 cosines are checked before each encoder runs.
Voices run one file at a time (no padding inside a mean). **Resumable:** finished arrays are skipped on re-run.
Expected time on a T4: downloads ~5 min, v4 ~30 min, v1/v2 ~1-1.5 h (40k voice files, many at 44.1 kHz) -> about 2 h in all.
"""

ENV = '''
import os, sys
from pathlib import Path
LOCAL = os.environ.get("FLAG_LOCAL") == "1"          # CPU smoke test on a mini copy of the data
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORKDIR = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
CACHE = Path(os.environ.get("FLAG_CACHE", "/tmp/flag_models_cache"))   # checkpoints stay out of the output
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
print("LOCAL" if LOCAL else "KAGGLE", "| input", IN, "| working", WORKDIR, "| cache", CACHE)
'''

PIP = '''
# insightface --no-deps: its declared deps would pull CPU onnxruntime over onnxruntime-gpu (as FLAG_04).
# ImageBind / CLIP --no-deps: their pins (torch 1.13, pytorchvideo) would break Kaggle's torch; flag_models only needs
# the model code, and reimplements the two data loaders it uses.
if not LOCAL:
    !pip -q install onnxruntime-gpu onnx iopath ftfy omegaconf fvcore 2>&1 | tail -2
    !pip -q install --no-deps "insightface==2.0" "git+https://github.com/facebookresearch/ImageBind" "git+https://github.com/openai/CLIP.git" 2>&1 | tail -2
import numpy as np, pandas as pd, torch, json, time, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''

CFG = '''
import flag_extract as X, flag_models as M
OUT = WORKDIR / "@OUTDIR@"; OUT.mkdir(exist_ok=True)
@ENCODERS@
# ONE walk over /kaggle/input that skips faces/ and voices/ folders: an extracted v1/v2 holds hundreds of thousands
# of files, and every rglob over it costs minutes. Everything below looks things up in SCAN instead.
SCAN = M.scan_inputs(IN)
S = M.v4_index(SCAN, WORKDIR / "raw")
meta = SCAN["files"].get("meta_file_train_set.csv")
assert meta, "meta_file_train_set.csv (train gender labels) not found in the inputs: attach the raw v4 dataset"
X.copy_metadata(meta[0].parent, S, OUT)

# MAV-Celeb v1/v2 validation rows (optional): the archive/folder + the existing meta csv that fixes the row order.
# Several copies (capped feats_ext vs feats_ext_full) -> take the largest: it is the superset, joined on `path` later.
EXT = {}
for name in ["v1_complete", "v2_complete"]:
    metas = sorted(SCAN["files"].get(f"ext_{name}_voice_meta.csv", []), key=lambda p: -p.stat().st_size)
    src = M.ext_source(SCAN, name)
    if metas and src is not None:
        try:
            print(f"{name}: rows from {metas[0].parent}")
            EXT[name] = M.ExtRows(src, name, metas[0].parent)
        except Exception as e:                    # optional source: a bad zip / missing row must not stop the run
            print(f"{name}: skipped, cannot read its rows from {src} ({type(e).__name__}: {e})")
    else:
        print(f"{name}: skipped (meta csv found: {bool(metas)}, source found: {src is not None})")
print("v4 rows per split:", {k: len(v["wav"]) for k, v in S.items()}, "| external:", list(EXT))

# Resume: attach an earlier version's output (@DATASET@) and its arrays are copied here, so only the
# encoders still missing run (e.g. after an encoder failed). Nothing to copy on a first run.
SEEDED = M.seed_from_previous(SCAN, OUT, ORDER)
'''

RUN = '''
CHECKS, FAILED = {}, {}
face_probe = [S["train"]["root"] / p for p in S["train"]["jpg"][:6]]
voice_probe = [X.load_audio(S["no_gender/Bangla"]["root"] / p) for p in S["no_gender/Bangla"]["wav"][:4]]
for name in ORDER:
    kind = "face" if name in FACE else "voice"
    t0 = time.time()
    if M.done(name, kind, S, EXT, OUT):
        print(f"== {name}: every array already exists -> skipped (not even loaded)")
        continue
    try:
        enc = M.build(name, CACHE)
        CHECKS[name] = M.check_face(enc, face_probe) if kind == "face" else M.check_voice(enc, voice_probe)
        print(name, "checks:", CHECKS[name])
        assert min(CHECKS[name].values()) > 0.995, f"{name}: batching / fp16 changes the embedding {CHECKS[name]}"
        M.extract_v4(name, enc, S, OUT, kind)
        for src_name, rows in EXT.items():
            M.extract_ext(name, enc, rows, src_name, OUT, kind)
    except Exception as e:                        # an optional encoder must not kill the run
        FAILED[name] = f"{type(e).__name__}: {e}"
        print(f"!! {name} FAILED -> skipped: {FAILED[name]}")
    enc = None
    if name != "ibv":                              # keep ImageBind loaded for the audio pass right after
        M.release()
    print(f"== {name} done in {time.time() - t0:.0f}s")
print("failed:", FAILED or "none")
'''

PROBE = '''
tr = pd.read_csv(OUT / "index_train.txt", sep=" ", header=None)
spk = tr[4].values
gmap = dict(pd.read_csv(OUT / "meta_file_train_set.csv", header=None).values)
P, SANITY = M.probe(OUT, spk, gmap, seeds=(1,) if LOCAL else (1, 2, 3), n=200 if LOCAL else 3000)
P.to_csv(OUT / "probe.csv", index=False)
print(P.to_string(index=False))
print(json.dumps(SANITY, indent=1))
# Reading guide (train, true labels, no training):
#  identity rows  - adaface should be near ArcFace (~2 % ng); a stream near 50 % is broken, not "attribute-rich".
#  ImageBind rows - zero-shot face<->voice. PLAN_V4: near chance (>= ~45 %) -> stop the ImageBind line, no LoRA.
#                   For scale: the trained bridge (VGG + BTC, internal) is ~22 / ~30 (ng / g) at sample level.
#  sanity         - w2v2 gender agreement should be > 0.9 (the 'gender' cells cannot use it; age is what matters there).
'''

DONE = '''
import shutil
man = X.write_manifest(OUT, dict(checks=CHECKS, failed=FAILED, sanity=SANITY, external=list(EXT),
                                 encoders=ORDER, audio=X.AUDIO_STATS))
(OUT / "probe.json").write_text(json.dumps(dict(sanity=SANITY, checks=CHECKS, failed=FAILED), indent=1))
for p in [WORKDIR / "raw"]:
    shutil.rmtree(p, ignore_errors=True)
tot = sum(p.stat().st_size for p in OUT.glob("*")) / 1e9
print(f"\\n{len(man['files'])} arrays, {tot:.2f} GB in {OUT}")
print("NEXT: Save Version (Save & Run All) -> open the version -> Output -> 'New Dataset' -> name it @DATASET@")
'''

def make(path, intro, pip, encoders, outdir, dataset, probe, probe_title):
    fill = lambda t: t.replace("@ENCODERS@", encoders.strip()).replace("@OUTDIR@", outdir).replace("@DATASET@", dataset)
    nb([("markdown", fill(intro)), ("code", ENV), ("code", embed_lib("flag_lib.py", "flag_extract.py", "agegender.py", "flag_models.py")),
        ("markdown", "## 1. Setup, v4 index (row order = organiser txt), v1/v2 rows"), ("code", pip), ("code", fill(CFG)),
        ("markdown", "## 2. Extract: one encoder at a time, v4 then v1/v2"), ("code", RUN),
        ("markdown", probe_title), ("code", probe),
        ("markdown", "## 4. Manifest + clean-up"), ("code", fill(DONE))], path)


ENC10 = """
FACE = list(M.FACE_ENCODERS)             # vitage farl siglip2 adaface ibv
VOICE = list(M.VOICE_ENCODERS)           # w2v2ag iba   (order: ImageBind vision then audio share one load)
ORDER = [e for e in FACE if e != "ibv"] + ["w2v2ag", "ibv", "iba"]
"""
make("FLAG_10_models.ipynb", INTRO, PIP, ENC10, "feats_models", "flag2027-feats-models", PROBE,
     "## 3. Probe: identity EER per stream, ImageBind zero-shot, attribute sanity")

# ============================================================================ FLAG_11: the organiser's encoders, other layers
INTRO11 = """
# FLAG 2027 — NB-11: the organiser's own encoders, other layers + test-time augmentation (S3 / S4)

Plan: `docs/OPEN_DIRECTIONS.md` S3 and S4. VGG fc7 + ECAPA-BTC 192 are our best bridge pair; both are the LAST layers of
their networks, pushed toward recognition. The earlier layers may keep more of the soft attributes a voice predicts.
Both encoders are the exact ones the organisers used (`flag_extract.VGGFace`, `flag_extract.OrganiserVoice`, cos 1.0000
with their csv), so every array here lives beside the features the bridge already uses.

| stream | layer | dim | direction |
|---|---|---:|---|
| `face_btcface` | fc7 post-ReLU = the organiser feature (checked against their csv) | 4096 | sanity |
| `face_btcfacefc6` | fc6 post-ReLU | 4096 | S3 |
| `face_btcfacepool5` | pool5 averaged over its 7 x 7 grid | 512 | S3 |
| `face_btcfaceflip` | fc7 of the horizontally mirrored image | 4096 | S4 (average with fc7) |
| `voice_btcvoice` | 192 output = the organiser feature (checked against their csv) | 192 | sanity |
| `voice_btcvoiceasp` | attentive-statistics pooling, the layer before the 192 projection | 3072 | S3 |
| `voice_btcvoicetta` | mean of the 192 output over the whole file + two 80 % crops | 192 | S4 |

**Settings:** Accelerator **GPU T4 x2** (one GPU used), **Internet ON** (VGG-Face 580 MB from robots.ox.ac.uk, the
organiser's ECAPA from Hugging Face; cached in `/tmp`, not in the output). fp32 throughout.

**Inputs:** exactly as FLAG_10: the raw v4 dataset; optional MAV-Celeb v1 / v2 + `flag2027-ext-meta` for Urdu / Hindi rows.
The raw v4 dataset also carries the organiser's `train_English_{faces,voices}.csv`, used for the reproduction check.

**Resume after a failure:** attach this notebook's earlier output (`@DATASET@`); finished encoders are skipped.

**Outputs:** `/kaggle/working/@OUTDIR@/` -> dataset **`@DATASET@`** (~1 GB: v4 fp32, v1/v2 fp16).
Expected time on a T4: downloads ~3 min, VGG-Face ~10 min, ECAPA with its TTA crops ~30 min.
"""

PIP11 = '''
# -U matters: Kaggle preinstalls an old speechbrain (< 1.1.1 breaks on torchaudio 2.9+), as FLAG_04 / FLAG_08.
if not LOCAL:
    !pip -q install -U "speechbrain>=1.1.1" soundfile 2>&1 | tail -2
import numpy as np, pandas as pd, torch, json, time, speechbrain
print("torch", torch.__version__, "| speechbrain", speechbrain.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''

ENC11 = """
FACE, VOICE = ["btcface"], ["btcvoice"]  # the organiser's VGG-Face / ECAPA, with their other layers and TTA
ORDER = FACE + VOICE
"""

PROBE11 = '''
ORG = M.check_organiser(OUT, SCAN)
print("re-extracted fc7 / 192 vs the organiser csv (must be ~1.0000):", ORG)
tr = pd.read_csv(OUT / "index_train.txt", sep=" ", header=None)
spk = tr[4].values
gmap = dict(pd.read_csv(OUT / "meta_file_train_set.csv", header=None).values)
P, SANITY = M.probe(OUT, spk, gmap, seeds=(1,) if LOCAL else (1, 2, 3), n=200 if LOCAL else 3000)
SANITY["organiser_csv_cos"] = ORG
P.to_csv(OUT / "probe.csv", index=False)
print(P.to_string(index=False))
# Identity EER per layer (same modality, train): earlier layers should be worse at identity than fc7 / 192.
# That is expected and is not the test; the bridge harness (Experiment/MODEL-01) decides whether they help face <-> voice.
'''

make("FLAG_11_layers.ipynb", INTRO11, PIP11, ENC11, "feats_layers", "flag2027-feats-layers", PROBE11,
     "## 3. Checks: reproduction of the organiser csv, identity EER per layer")
