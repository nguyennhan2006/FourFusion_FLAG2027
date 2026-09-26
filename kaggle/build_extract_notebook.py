"""Generate FLAG_03_extract.ipynb — re-extract features from the official raw media.

The decisive experiment (RESEARCH_DIRECTIONS.md §5.1): change ONLY the voice feature and see
whether the 192-d bottleneck hypothesis holds. The notebook also re-extracts ECAPA's own 192-d
output, which lets us check whether the organisers' CSV is that same encoder.
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = (HERE / "flag_lib.py").read_text(encoding="utf-8")
_B64 = base64.b64encode(LIB.encode("utf-8")).decode("ascii")
_B64_WRAPPED = "\n".join(_B64[i:i + 100] for i in range(0, len(_B64), 100))

WRITE_LIB = (
    "# Writes flag_lib.py (the evaluation code from the sweep notebook) next to this notebook.\n"
    "import base64\n"
    "_SRC = '''\n" + _B64_WRAPPED + "\n'''\n"
    "open('flag_lib.py', 'wb').write(base64.b64decode(''.join(_SRC.split())))\n"
    "print('flag_lib.py written')\n"
)

INTRO = """# FLAG 2027 — re-extract features from raw audio / images

**Why this notebook exists.** Our current system (30.90, top 4) uses the 192-d voice embeddings the
organisers ship. The FAME 2026 winner used the *same* ECAPA-TDNN but took the **6144-d layer before
the output**, plus a richer face encoder. Every internal symptom says we are at the ceiling of the
provided features: a 28-config architecture sweep spread only 0.5 EER, augmentation did not help,
and a closed-form linear CCA already reached 35.79.

**Rules.** The evaluation plan ships the data *"Alongside the audios (.wav) and images (.jpg)"* and
*"provided alongside pre-extracted features"*, and states *"A pretrained encoder for faces or voices
is allowed."* Re-extracting with a frozen pretrained encoder is therefore on solid ground. (Training
on an **external** face-voice corpus such as VoxCeleb2 is a separate question — ask the organisers.)

**Discipline.** This notebook changes exactly one thing at a time:

| run | face | voice | verdict it gives |
|---|---|---|---|
| A (control) | provided 4096-d | provided 192-d | our current 30.90 system |
| B | provided 4096-d | **ECAPA 6144-d** | is the 192-d voice the bottleneck? |
| C | provided 4096-d | ECAPA 192-d (ours) | does our extraction reproduce the organisers' CSV? |
| D | **ArcFace 512-d** | provided 192-d | is VGGFace fc7 the bottleneck? |
| E | ArcFace 512-d | ECAPA 6144-d | both |

**Settings:** GPU T4 x2, **Internet ON** (needed to download the pretrained encoders).

**Input — one dataset is enough:** upload `Input/` (or the two zips). Kaggle may auto-extract them;
the notebook handles either form. The organisers' feature CSVs ship inside `train_set/features/` and
`dev_set/*/features/`, so the control run A and the encoder cross-check are covered by the same upload.
"""

SETUP = '''# -U matters: Kaggle preinstalls an older speechbrain, and a plain `pip install speechbrain`
# reports "already satisfied" and leaves it. Versions before 1.1.1 call torchaudio.list_audio_backends()
# at import time, which torchaudio 2.9+ removed.
!pip -q install -U "speechbrain>=1.1.1" soundfile 2>&1 | tail -2
import os, sys, time, zipfile, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torchaudio, soundfile as sf

# torchaudio >= 2.9 removed the legacy backend API that speechbrain probes on import.
# Restore just enough of it; we read audio with soundfile anyway.
for _name, _fn in [("list_audio_backends", lambda: ["soundfile"]),
                   ("get_audio_backend", lambda: "soundfile"),
                   ("set_audio_backend", lambda *a, **k: None)]:
    if not hasattr(torchaudio, _name):
        setattr(torchaudio, _name, _fn)

# Drop any half-imported speechbrain left by a previous failed attempt in this kernel, otherwise the
# retry raises "partially initialized module 'speechbrain' ... circular import".
for _m in [m for m in list(sys.modules) if m == "speechbrain" or m.startswith("speechbrain.")]:
    del sys.modules[_m]

import speechbrain as _sb
print("speechbrain", _sb.__version__, "(need >= 1.1.1 for torchaudio 2.9+)")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__, "| device", DEV,
      "| GPUs", torch.cuda.device_count())

IN = Path("/kaggle/input")
# Kaggle may keep the uploaded archives OR auto-extract them; handle both.
wavs = list(IN.rglob("*.wav"))
if wavs:
    WORK = IN
    print(f"raw media already extracted by Kaggle: {len(wavs)} wav found, reading in place")
else:
    zips = sorted(IN.rglob("*.zip"))
    assert zips, ("No .wav and no .zip under /kaggle/input - attach the raw media "
                  "(Input/train_set.zip + Input/dev_set.zip, or their extracted form)")
    WORK = Path("/kaggle/working/raw"); WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            z.extractall(WORK)
        print("extracted", zp.name, round(time.time() - t0), "s", flush=True)
    wavs = list(WORK.rglob("*.wav"))

jpgs = list(WORK.rglob("*.jpg"))
print(f"wav: {len(wavs)} | jpg: {len(jpgs)}")
assert len(wavs) > 10000, f"expected ~11349 wav files, found {len(wavs)}"

CSVS = sorted(IN.rglob("train_English_voices.csv"))
assert CSVS, ("The organisers' feature CSVs are needed for run A (control) and the encoder check. "
              "They ship inside train_set/features/, so attaching the raw dataset is usually enough.")
print("organiser CSV:", CSVS[0])
'''

INDEX = '''# Row order must match the organisers' .txt files exactly - their CSVs are aligned to them,
# so our new features stay directly comparable.
#
# Careful: the same .txt name can appear in several attached datasets (e.g. a features-only upload
# that has no media). Pick the copy whose media files actually resolve next to it.
def resolve_split(txt_name, wav_col, jpg_col, must_contain=None):
    cands = [p for p in WORK.rglob(txt_name) if must_contain is None or must_contain in p.as_posix()]
    assert cands, f"no {txt_name} found (filter={must_contain})"
    tried = []
    for p in sorted(cands, key=lambda q: len(q.as_posix())):
        df = pd.read_csv(p, sep=" ", header=None)
        wav, jpg = df[wav_col].tolist(), df[jpg_col].tolist()
        if (p.parent / wav[0]).exists() and (p.parent / jpg[0]).exists():
            return dict(root=p.parent, wav=wav, jpg=jpg)
        tried.append(str(p.parent))
    raise FileNotFoundError(
        f"{txt_name}: found {len(cands)} copies but none has its media beside it. "
        f"Checked these folders: {tried}. "
        "Attach the RAW dataset (the one containing the .wav and .jpg files).")

SPLITS = {"train": resolve_split("train_English.txt", 2, 3)}
for prot in ["no_gender", "gender"]:
    for lang in ["English", "Bangla"]:
        SPLITS[f"{prot}/{lang}"] = resolve_split(f"{lang}_test.txt", 1, 2, must_contain=f"/{prot}/")

for k, v in SPLITS.items():
    root = v["root"]
    shown = root.relative_to(WORK) if WORK in root.parents or WORK == root else root
    print(f"{k:22s} {len(v['wav']):5d} rows | root={shown}")
print("total files to encode:", sum(len(v["wav"]) for v in SPLITS.values()))
'''

VOICE = '''# This cell is self-contained on purpose: if an earlier attempt failed part-way through importing
# speechbrain, the half-built module stays in sys.modules and every retry then dies with
# "partially initialized module 'speechbrain' ... circular import". Clear it, re-apply the torchaudio
# shim, and only then import.
import sys, torchaudio
for _m in [m for m in list(sys.modules) if m == "speechbrain" or m.startswith("speechbrain.")]:
    del sys.modules[_m]
for _name, _fn in [("list_audio_backends", lambda: ["soundfile"]),
                   ("get_audio_backend", lambda: "soundfile"),
                   ("set_audio_backend", lambda *a, **k: None)]:
    if not hasattr(torchaudio, _name):
        setattr(torchaudio, _name, _fn)

from speechbrain.inference.speaker import EncoderClassifier

_kw = {}
try:   # newer speechbrain copies instead of symlinking, which is safer inside /kaggle/working
    from speechbrain.utils.fetching import LocalStrategy
    _kw["local_strategy"] = LocalStrategy.COPY
except Exception:
    pass

ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                       savedir="/kaggle/working/ecapa",
                                       run_opts={"device": str(DEV)}, **_kw)
emb_model = ecapa.mods.embedding_model.eval()

# ECAPA: ... -> attentive statistics pooling (6144-d) -> BN -> conv1d (192-d).
# Hook the pooling output to get the richer representation the FAME 2026 winner used.
_pool = {}
emb_model.asp.register_forward_hook(lambda m, i, o: _pool.__setitem__("x", o.detach()))

@torch.no_grad()
def voice_embed(paths, root, bs=16):
    out192, out6144 = [], []
    for s in range(0, len(paths), bs):
        batch = paths[s:s + bs]
        sigs, lens = [], []
        for p in batch:
            w, sr = sf.read(str(Path(root) / p), dtype="float32", always_2d=True)
            assert sr == 16000, sr
            w = torch.from_numpy(w[:, 0])
            sigs.append(w); lens.append(len(w))
        L = max(lens)
        x = torch.zeros(len(sigs), L)
        for i, w in enumerate(sigs):
            x[i, :len(w)] = w
        rel = torch.tensor([l / L for l in lens], dtype=torch.float32)
        feats = ecapa.mods.compute_features(x.to(DEV))
        feats = ecapa.mods.mean_var_norm(feats, rel.to(DEV))
        e = emb_model(feats, rel.to(DEV))              # (B, 1, 192)
        out192.append(e.squeeze(1).cpu().numpy())
        out6144.append(_pool["x"].squeeze(-1).cpu().numpy())
        if s % (bs * 50) == 0:
            print(f"   {s}/{len(paths)}", flush=True)
    return np.concatenate(out192), np.concatenate(out6144)

VOICE_FEATS = {}
for name, sp in SPLITS.items():
    t0 = time.time()
    a, b = voice_embed(sp["wav"], sp["root"])
    VOICE_FEATS[name] = dict(ecapa192=a, ecapa6144=b)
    np.save(f"voice_ecapa192_{name.replace('/', '_')}.npy", a)
    np.save(f"voice_ecapa6144_{name.replace('/', '_')}.npy", b)
    print(f"{name:22s} {a.shape} {b.shape}  ({round(time.time()-t0)}s)", flush=True)
'''

CHECK_CSV = '''# Free sanity check: is the organisers' 192-d CSV this same ECAPA?
# (Only possible where we also have their CSV — i.e. if the feature dataset is attached too.)
feat_csv = CSVS   # located in the setup cell
if feat_csv:
    given = pd.read_csv(feat_csv[0], header=None).iloc[:, :-1].values.astype(np.float32)
    ours = VOICE_FEATS["train"]["ecapa192"]
    gn = given / np.linalg.norm(given, axis=1, keepdims=True)
    on = ours / np.linalg.norm(ours, axis=1, keepdims=True)
    diag = np.sum(gn * on, 1)
    print(f"cosine(ours_192, organisers_192): mean {diag.mean():.4f}  median {np.median(diag):.4f}")
    print("norm ratio (ours/given):", round(float(np.linalg.norm(ours, axis=1).mean() /
                                                  np.linalg.norm(given, axis=1).mean()), 3))
    print("-> ~1.0 means the organisers used this exact encoder and 6144-d really is 'the layer before'.")
else:
    print("organiser feature CSVs not attached; skipping this check")
'''

FACE = '''!pip -q install insightface onnxruntime-gpu 2>&1 | tail -2
import insightface, cv2
from insightface.app import FaceAnalysis

# ArcFace R100 (buffalo_l). Faces are already cropped to 224x224, so we bypass detection and
# feed the whole crop to the recognition model.
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(224, 224))
rec = app.models["recognition"]

def face_embed(paths, root, bs=32):
    out = []
    for s in range(0, len(paths), bs):
        imgs = []
        for p in paths[s:s + bs]:
            im = cv2.imread(str(Path(root) / p))
            imgs.append(cv2.resize(im, (112, 112)))
        blob = cv2.dnn.blobFromImages(np.array(imgs), 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        out.append(rec.session.run(rec.output_names, {rec.input_name: blob})[0])
        if s % (bs * 50) == 0:
            print(f"   {s}/{len(paths)}", flush=True)
    return np.concatenate(out)

FACE_FEATS = {}
for name, sp in SPLITS.items():
    t0 = time.time()
    f = face_embed(sp["jpg"], sp["root"])
    FACE_FEATS[name] = f
    np.save(f"face_arcface_{name.replace('/', '_')}.npy", f)
    print(f"{name:22s} {f.shape}  ({round(time.time()-t0)}s)", flush=True)
'''

EVAL = '''# The decisive comparison. Same pipeline, same loss, same protocol — only the features change.
# Point flag_lib at WORK: if Kaggle did not auto-extract, the CSVs live under /kaggle/working/raw,
# not /kaggle/input.
os.environ["FLAG_DATA"] = str(WORK)
print("FLAG_DATA =", os.environ["FLAG_DATA"])
import flag_lib as L

Xf_old, Xv_old, spk, gmap, vgrp = L.load_train()      # organiser features, for the control run
ALL = np.arange(len(spk))
SEEDS = [1, 2, 3]
SPL = {s: (lambda tv: (np.where(tv[0])[0], np.where(tv[1])[0]))(L.speaker_split(spk, s)) for s in SEEDS}
TRIALS = {(s, sg): L.build_trials(spk[SPL[s][1]], gmap, seed=s, n_pos=3000, n_neg=3000, same_gender=sg)
          for s in SEEDS for sg in [False, True]}

def evaluate(Xf, Xv, tag, cfg=None, n_seeds=3, cca_weight=0.25):
    """Trains our current recipe on the given feature pair and returns (int_ng, int_g)."""
    cfg = cfg or dict(hid=512, emb=128, epochs=40, drop=0.3, dedup_clip=False)
    out = {"no_gender": [], "gender": []}
    for s in SEEDS:
        tri, vai = SPL[s]
        pr = L.Prep(Xf[tri], Xv[tri], key=f"{tag}_{s}")
        aF, aV = L.Aligner(pr.f(Xf[tri])), L.Aligner(pr.v(Xv[tri]))
        embs = []
        for i in range(n_seeds):
            net, prep, _ = L.train_model(Xf, Xv, spk, gmap, vgrp, tri, cfg, seed=s + 100 * i)
            embs.append(L.embed_one(net, prep, Xf[vai], Xv[vai], alF=aF, alV=aV))
        m = L.RidgeCCA(k=4, reg=1.0, pca_x=min(128, Xf.shape[1])).fit(Xf[tri], Xv[tri])
        cca = (m.transform_x(L.center_to(Xf[vai], Xf[tri].mean(0))),
               m.transform_y(L.center_to(Xv[vai], Xv[tri].mean(0))))
        for sg in [False, True]:
            fi, vj, lab = TRIALS[(s, sg)]
            sc = L.fuse_scores(embs, fi, vj, cca=cca, cca_weight=cca_weight)
            out["gender" if sg else "no_gender"].append(L.eer_from_scores(sc, lab))
    return float(np.mean(out["no_gender"])), float(np.mean(out["gender"]))

V6144 = VOICE_FEATS["train"]["ecapa6144"]
V192 = VOICE_FEATS["train"]["ecapa192"]
FARC = FACE_FEATS["train"]

runs = {
    "A control (given 4096 + given 192)": (Xf_old, Xv_old),
    "B given face + ECAPA 6144":          (Xf_old, V6144),
    "C given face + our ECAPA 192":       (Xf_old, V192),
    "D ArcFace 512 + given 192":          (FARC,   Xv_old),
    "E ArcFace 512 + ECAPA 6144":         (FARC,   V6144),
}
rows = []
for tag, (a, b) in runs.items():
    t0 = time.time()
    ng, g = evaluate(a, b, tag)
    rows.append(dict(run=tag, face_dim=a.shape[1], voice_dim=b.shape[1],
                     int_ng=round(ng, 2), int_g=round(g, 2), sec=round(time.time() - t0)))
    pd.DataFrame(rows).to_csv("feature_ablation.csv", index=False)
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print()
print(df.to_string(index=False))
print("\\nControl int_g was 30.42 locally -> CodaBench 30.90.")
print("If B beats A clearly, the 192-d voice really was the bottleneck.")
'''

PREFLIGHT = '# Preflight: catch a score-orientation mistake before spending GPU time.\n# NOTE what this can and cannot do. It verifies our INTERNAL convention (eer_from_scores expects\n# higher = same). It CANNOT tell us which direction the CodaBench scorer wants - that was settled\n# experimentally: submitting -d^2 scored 57.83 and +d^2 scored 42.17 with the same checkpoint,\n# so the live scorer is distance-oriented (LOWER = same) and write_submission negates accordingly.\nimport flag_lib as _L\n_lab = np.r_[np.ones(500), np.zeros(500)]\n_good = np.r_[np.random.RandomState(0).normal(3, 1, 500), np.random.RandomState(1).normal(-3, 1, 500)]\n_e_ok, _e_flip = _L.eer_from_scores(_good, _lab), _L.eer_from_scores(-_good, _lab)\nprint(f"internal convention: aligned {_e_ok:.2f}%  |  inverted {_e_flip:.2f}%")\nassert _e_ok < 5 and _e_flip > 95, "eer_from_scores is not oriented as higher = same"\nprint("submission polarity: write_submission() writes -score, i.e. LOWER = same, per the measured "\n      "scorer behaviour recorded in docs/DATA.md")\n'

SUBMIT = '''# Build a submission from whichever feature pair won, using the same recipe as EXP-007.
# Overall on CodaBench is the mean of the four cells, so rank by the mean of both internal protocols
# rather than int_g alone. (Calibration caveat: int_ng runs ~4 points optimistic vs CodaBench while
# int_g has tracked it closely, and neither sees Bangla - this stays a proxy.)
df["proxy_avg"] = (df.int_ng + df.int_g) / 2
print(df.sort_values("proxy_avg")[["run", "int_ng", "int_g", "proxy_avg"]].to_string(index=False))
BEST_RUN = df.sort_values("proxy_avg").iloc[0].run
print("best run:", BEST_RUN)
FEATS = {"A control (given 4096 + given 192)": ("old", "old"), "B given face + ECAPA 6144": ("old", "6144"),
         "C given face + our ECAPA 192": ("old", "192"), "D ArcFace 512 + given 192": ("arc", "old"),
         "E ArcFace 512 + ECAPA 6144": ("arc", "6144")}[BEST_RUN]

def feats_for(split):
    f = FACE_FEATS[split] if FEATS[0] == "arc" else None
    v = {"old": None, "6144": VOICE_FEATS[split]["ecapa6144"], "192": VOICE_FEATS[split]["ecapa192"]}[FEATS[1]]
    return f, v

Xf_tr, Xv_tr = feats_for("train")
Xf_tr = Xf_tr if Xf_tr is not None else Xf_old
Xv_tr = Xv_tr if Xv_tr is not None else Xv_old

cfg = dict(hid=512, emb=128, epochs=40, drop=0.3, dedup_clip=False)
prep_all = L.Prep(Xf_tr, Xv_tr, key="final")
aF, aV = L.Aligner(prep_all.f(Xf_tr)), L.Aligner(prep_all.v(Xv_tr))
nets = [L.train_model(Xf_tr, Xv_tr, spk, gmap, vgrp, ALL, cfg, seed=1 + 100 * i)[:2] for i in range(10)]
cca = L.RidgeCCA(k=4, reg=1.0, pca_x=min(128, Xf_tr.shape[1])).fit(Xf_tr, Xv_tr)

devs, scores = {}, {}
for key in L.NAMES:
    split = f"{key[0]}/{key[1]}"
    a_old, b_old, t = L.load_dev(*key)
    devs[key] = (a_old, b_old, t)
    f, v = feats_for(split)
    a = f if f is not None else a_old
    b = v if v is not None else b_old
    embs = [L.embed_one(net, prep, a, b, alF=aF, alV=aV) for net, prep in nets]
    cca_emb = (cca.transform_x(L.center_to(a, Xf_tr.mean(0))), cca.transform_y(L.center_to(b, Xv_tr.mean(0))))
    ii = np.arange(len(t))
    scores[key] = L.fuse_scores(embs, ii, ii, cca=cca_emb, cca_weight=0.25)

L.write_submission("submission_newfeat.zip", scores, devs)
for k, fn in L.NAMES.items():
    d = pd.read_csv(zipfile.ZipFile("submission_newfeat.zip").open(fn), sep=" ", header=None, names=["pid", "s"])
    print(fn, len(d), bool((d.pid.values == devs[k][2].pair_id.values).all()), int(d.s.isna().sum()))
'''

cells = [
    ("markdown", INTRO),
    ("code", WRITE_LIB),
    ("markdown", "## 1. Unpack the official raw media"), ("code", SETUP), ("code", INDEX),
    ("markdown", "## 2. Voice — ECAPA-TDNN, both the 192-d output and the 6144-d pooling layer"),
    ("code", VOICE), ("code", CHECK_CSV),
    ("markdown", "## 3. Face — ArcFace R100 (buffalo_l)"), ("code", FACE),
    ("markdown", "## 4. The decisive ablation — one variable at a time"), ("code", PREFLIGHT), ("code", EVAL),
    ("markdown", "## 5. Submission from the winning feature pair"), ("code", SUBMIT),
]

nb = {
    "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
               **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}, "accelerator": "GPU"},
    "nbformat": 4, "nbformat_minor": 5,
}
(HERE / "FLAG_03_extract.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote FLAG_03_extract.ipynb", len(cells), "cells")
