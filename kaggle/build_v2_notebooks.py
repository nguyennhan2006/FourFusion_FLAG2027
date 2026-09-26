"""Generate the v2 Kaggle notebooks (docs/PLAN_KAGGLE.md §4):

    FLAG_04_extract.ipynb   NB-1  raw media -> .npy features (GPU, Internet ON), output becomes a dataset
    FLAG_05_ablate.ipynb    NB-2  phases P0-P6 on those features, candidate submissions (GPU, Internet OFF)
    FLAG_06_submit.ipynb    NB-3  retrain pasted configs on 70 speakers, per-cell merge, zip (GPU, Internet OFF)

The libraries are embedded (base64) so no .py upload is needed. Every notebook also runs locally with
FLAG_LOCAL=1 + FLAG_IN / FLAG_OUT pointing at a mini copy of the data (smoke test, CPU).
"""
import base64, json
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


ENV = '''
import os, sys
from pathlib import Path
LOCAL = os.environ.get("FLAG_LOCAL") == "1"          # CPU smoke test on a mini copy of the data
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORKDIR = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
print("LOCAL" if LOCAL else "KAGGLE", "| input", IN, "| working", WORKDIR)
'''

# ============================================================================ NB-1
NB1_INTRO = """
# FLAG 2027 — NB-1: re-extract features from the raw media

Plan: `docs/PLAN_KAGGLE.md`. Runs **once**; its output folder becomes the Kaggle dataset `flag2027-feats-v2`
that NB-2 / NB-3 read (so no training notebook ever touches wav/jpg again).

**Settings:** Accelerator **GPU T4 x2** (one GPU is used), **Internet ON** (pretrained encoders),
**Persistence: Files** is not needed. Input: the raw dataset (`Input/` = `train_set.zip` + `dev_set.zip` +
`meta_file_train_set.csv`; Kaggle may auto-extract, both forms work).

| output | shape | what |
|---|---|---|
| `voice_ecapa192_<split>.npy` | N x 192 | ECAPA output — cross-check vs organiser CSV |
| `voice_ecapa6144_<split>.npy` | N x 6144 | ECAPA attentive-stat-pooling (FAME 2026 winner's layer) |
| `voice_ecapa{192,6144}crop_train.npy` | 3 x N x D | train utterances cropped to the dev-**Bangla** duration distribution |
| `face_arcface_<split>.npy` | N x 512 | ArcFace R100, SCRFD detection + 5-point alignment |
| `voice_<ssl>_<split>.npy` | N x (L+1) x H fp16 | per-layer mean of WavLM-Large / XLS-R-300m (multilingual) |

**OOM safety:** audio sorted by length and packed by *padded seconds* (the longest dev file is 81.6 s);
CUDA OOM halves the batch; a single file that still fails is run on CPU. SSL models use 20 s windows.
**Quality checks printed:** batched-vs-single cosine (must be > 0.999), face detection rate, fp16-vs-fp32 cosine.
**Resumable:** every array is written as soon as it is done and skipped on re-run.

Expected time on a T4: ECAPA + crops ~15 min, ArcFace ~10 min, each SSL model ~15-25 min.
"""

NB1_PIP = '''
# -U matters: Kaggle preinstalls an old speechbrain (<1.1.1 breaks on torchaudio 2.9+).
# insightface is pinned and installed --no-deps: its declared deps would pull CPU `onnxruntime` over
# `onnxruntime-gpu` and a second OpenCV; Kaggle already ships scipy / scikit-image / opencv.
if not LOCAL:
    !pip -q install -U "speechbrain>=1.1.1" onnxruntime-gpu onnx 2>&1 | tail -3
    !pip -q install --no-deps "insightface==2.0" 2>&1 | tail -3
import numpy as np, torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''

NB1_CFG = '''
import flag_extract as X
OUT = WORKDIR / "feats"; OUT.mkdir(exist_ok=True)
N_CROPS = 3                      # duration-matched crops per train utterance
# multilingual SSL encoders (plan H2b). Set to [] to skip. Local smoke test uses a small model.
SSL = [("microsoft/wavlm-large", "wavlm"), ("facebook/wav2vec2-xls-r-300m", "xlsr")]
if LOCAL:
    SSL = [("microsoft/wavlm-base-plus", "wavlmbp")]
MAX_PADDED_SEC = 240.0           # per ECAPA batch; OOM halves it automatically
root = X.media_root(IN, WORKDIR / "raw")
S = X.index_splits(root)
X.copy_metadata(IN, S, OUT)
print("total files per modality:", sum(len(v["wav"]) for v in S.values()))
'''

NB1_ECAPA = '''
ecapa = X.extract_ecapa(S, OUT, n_crops=N_CROPS, max_padded_sec=MAX_PADDED_SEC)
CONS = X.check_batch_consistency(ecapa, S, n=12 if LOCAL else 24)
# local CPU test measured 0.99969 / 0.99989 (fp32 reduction order differs between batch sizes)
assert min(CONS.values()) > 0.995, f"padding changes the embedding: {CONS} - do not use these features"
if min(CONS.values()) < 0.999:
    print("WARNING: batched vs single cosine below 0.999:", CONS)
del ecapa; torch.cuda.empty_cache()
'''

NB1_CHECK = '''
# Is the organisers' 192-d CSV this same ECAPA?  ~1.0 -> yes, and 6144-d really is "the layer before".
import pandas as pd
given = pd.read_csv(sorted(IN.rglob("train_English_voices.csv"))[0], header=None).iloc[:, :-1].values
ours = np.load(OUT / "voice_ecapa192_train.npy")
assert len(given) == len(ours)
unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
cos = np.sum(unit(given) * unit(ours), 1)
ENC_MATCH = dict(mean=float(cos.mean()), p5=float(np.percentile(cos, 5)))
print("cos(ours ECAPA-192, organiser CSV):", ENC_MATCH)
# Local mini test gave mean ~0.02 while both give the same speaker EER (5.5%): rows ARE aligned, the
# organisers simply used a different ECAPA checkpoint. NB-2 P0 re-checks alignment via unimodal EER.
if ENC_MATCH["mean"] < 0.9:
    print("-> different encoder from speechbrain/spkrec-ecapa-voxceleb (expected); alignment is checked in NB-2 P0")
'''

NB1_FACE = '''
DET = X.extract_arcface(S, OUT)
for k, v in DET.items():
    if v < 0.9:
        print(f"WARNING {k}: only {100*v:.1f}% faces detected+aligned; the rest used a resized centre crop")
'''

NB1_SSL = '''
SSL_COS = {}
for model_id, short in SSL:
    try:
        SSL_COS[short] = X.extract_ssl(S, OUT, model_id, short)
    except Exception as e:                       # an optional encoder must not kill the run
        print(f"SSL {model_id} failed: {type(e).__name__}: {e}")
'''

NB1_DONE = '''
import shutil
man = X.write_manifest(OUT, dict(consistency=CONS, encoder_match=ENC_MATCH, face_detect=DET, ssl_fp16_cos=SSL_COS,
                                 n_crops=N_CROPS))
# keep the output small: drop extracted raw media and model checkpoints from /kaggle/working
for p in [WORKDIR / "raw", WORKDIR / "_ecapa_ckpt", OUT.parent / "_ecapa_ckpt"]:
    shutil.rmtree(p, ignore_errors=True)
tot = sum(p.stat().st_size for p in OUT.glob("*")) / 1e9
print(f"\\n{len(man['files'])} arrays, {tot:.2f} GB in {OUT}")
print("NEXT: Save Version (Save & Run All) -> open the version -> Output -> 'New Dataset' -> name it flag2027-feats-v2")
'''

nb([("markdown", NB1_INTRO), ("code", ENV), ("code", embed_lib("flag_lib.py", "flag_extract.py")),
    ("markdown", "## 1. Setup and index (row order = organiser txt)"), ("code", NB1_PIP), ("code", NB1_CFG),
    ("markdown", "## 2. ECAPA-TDNN: 192-d + 6144-d, plus duration-matched train crops"), ("code", NB1_ECAPA),
    ("code", NB1_CHECK),
    ("markdown", "## 3. ArcFace R100 with detection + alignment"), ("code", NB1_FACE),
    ("markdown", "## 4. Multilingual self-supervised encoders (optional, H2b)"), ("code", NB1_SSL),
    ("markdown", "## 5. Manifest + clean-up"), ("code", NB1_DONE)], "FLAG_04_extract.ipynb")

# ============================================================================ NB-2
NB2_INTRO = """
# FLAG 2027 — NB-2: ablation phases P0-P6 on the re-extracted features

Plan: `docs/PLAN_KAGGLE.md` §5. **One variable per run.** Every phase carries its winner forward only if
it beats the incumbent by >= 0.3 proxy (noise floor across splits); otherwise the incumbent stays.

**Settings:** GPU T4 (x1 is enough), **Internet OFF**. Inputs: `flag2027-feats-v2` (NB-1 output) **and**
`flag2027-features` (organiser CSVs, for the control run).

* Selection metric: `proxy = (int_ng + int_g) / 2` over 3 speaker-disjoint splits x 3 models; P4 uses `int_g`.
* `int_g_short`: gender-protocol EER when val voices are cropped to Bangla-like length (only for encoders with crops).
* Internal validation is English-only. **P5 (Bangla) is decided on CodaBench**, never internally.
* Results go to `ablation.csv` after every run; re-running the notebook resumes (finished runs are cached).
"""

NB2_SETUP = '''
import numpy as np, pandas as pd, json, torch
os.environ["FLAG_DATA"] = str(IN)
import flag_v2 as V
V.polarity_preflight()
store = V.FeatureStore(IN)
T = V.Table(WORKDIR / "ablation.csv")
# budget: 3 splits x 3 models per run on Kaggle; 1 x 1 with few epochs for the local smoke test
KW = dict(seeds=(1,), n_models=1) if LOCAL else dict(seeds=(1, 2, 3), n_models=3)
EP = dict(epochs=2) if LOCAL else {}
HAS = {n: store.has("voice", n) for n in ["ecapa192", "ecapa6144"]}
HAS["arcface"] = store.has("face", "arcface")
SSL = [p.name.split("_")[1] for p in (store.fd.glob("voice_*_train.npy") if store.fd else [])
       if p.name.split("_")[1] not in ("ecapa192", "ecapa6144", "ecapa192crop", "ecapa6144crop")]
print("available:", HAS, "| ssl:", SSL, "| device", V.DEVICE)
'''

NB2_P0 = '''
# P0 - are the new features sane, and how good is each one on its own (no training)?
rows = []
if HAS["ecapa192"]:
    a, b = store.voice("given", "train"), store.voice("ecapa192", "train")
    c = np.sum(V.l2n(a) * V.l2n(b), 1)
    print(f"cos(own ECAPA-192, organiser 192): mean {c.mean():.4f}  p5 {np.percentile(c, 5):.4f}")
for kind, name in [("face", "vgg"), ("face", "arcface"), ("voice", "given"), ("voice", "ecapa192"), ("voice", "ecapa6144")]:
    if kind == "face" and name == "arcface" and not HAS["arcface"]: continue
    if kind == "voice" and name != "given" and not HAS[name]: continue
    X_ = store.face(name, "train") if kind == "face" else store.voice(name, "train")
    rows.append(dict(kind=kind, feat=name, dim=X_.shape[1], **V.unimodal_eer(X_, store.spk, store.gmap)))
    C = store.voice_crops(name) if kind == "voice" else None
    if C is not None:
        rows.append(dict(kind=kind, feat=name + " (Bangla-length crop)", dim=C.shape[2], **V.unimodal_eer(C[0], store.spk, store.gmap)))
# SSL: pick the best block of layers by unimodal speaker EER (cheap, no training)
BEST_SSL = {}
for s in SSL:
    nl = np.load(store.fd / f"voice_{s}_train.npy", mmap_mode="r").shape[1]
    wins = sorted({(a, min(a + w - 1, nl - 1)) for w in (3, 6) for a in range(1, nl, 3)})
    best = None
    for a, b in wins:
        r = V.unimodal_eer(store.voice(f"{s}_L{a}-{b}", "train"), store.spk, store.gmap)
        rows.append(dict(kind="voice", feat=f"{s}_L{a}-{b}", dim=None, **r))
        if best is None or r["g"] < best[1]: best = (f"{s}_L{a}-{b}", r["g"])
    BEST_SSL[s] = best[0]
P0 = pd.DataFrame(rows); P0.to_csv(WORKDIR / "p0_unimodal.csv", index=False)
print(P0.sort_values(["kind", "g"]).to_string(index=False))
print("best SSL layer blocks:", BEST_SSL)
'''

NB2_P1 = '''
# P1 - head/regularisation on the ORGANISER features (control = our current recipe)
CTRL = dict(dict(head="mlp", drop=0.3, emb=128, epochs=40), **EP)
T.run(store, "P1", "P1 control mlp drop.3 emb128", CTRL, **KW)
T.run(store, "P1", "P1 linear drop.3", dict(CTRL, head="linear"), **KW)
T.run(store, "P1", "P1 linear drop.9", dict(CTRL, head="linear", drop=0.9), **KW)
T.run(store, "P1", "P1 linear drop.9 emb192", dict(CTRL, head="linear", drop=0.9, emb=192), **KW)
T.run(store, "P1", "P1 mlp drop.9", dict(CTRL, drop=0.9), **KW)
CFG, INC = V.pick(T, "P1", "P1 control mlp drop.3 emb128")
'''

NB2_P2 = '''
# P2 - which feature is the bottleneck?  Same head/loss as the P1 winner, only the features change.
feats = [("vgg", "given", "A vgg + given192 (control)")]
if HAS["ecapa6144"]: feats.append(("vgg", "ecapa6144", "B vgg + ecapa6144"))
if HAS["ecapa192"]:  feats.append(("vgg", "ecapa192", "C vgg + own ecapa192"))
if HAS["arcface"]:   feats.append(("arcface", "given", "D arcface + given192"))
if HAS["arcface"] and HAS["ecapa6144"]: feats.append(("arcface", "ecapa6144", "E arcface + ecapa6144"))
for s, blk in BEST_SSL.items():
    feats.append(("vgg", blk, f"F vgg + {blk}"))
    if HAS["arcface"]: feats.append(("arcface", blk, f"G arcface + {blk}"))
names = {}
for f, v, nm in feats:
    name = f"P2 {nm}"
    T.run(store, "P2", name, dict(CFG, face=f, voice=v), **KW); names[nm] = name
# control row equals the P1 incumbent config on organiser features
CFG, INC = V.pick(T, "P2", names["A vgg + given192 (control)"])
print(T.df[T.df.phase == "P2"][["run", "int_ng", "int_g", "proxy", "sd_g"]].to_string(index=False))
'''

NB2_P3 = '''
# P3 - loss: add shared-centre AAM, then explicit alignment, then orthogonality (one at a time)
base_name = INC
T.run(store, "P3", "P3 = incumbent", CFG, **KW)
for m, s in [(0.2, 16.0), (0.2, 32.0), (0.3, 32.0)]:
    T.run(store, "P3", f"P3 +AAM m{m} s{int(s)}", dict(CFG, w_aam=1.0, margin=m, scale=s), **KW)
CFG_A, INC_A = V.pick(T, "P3", "P3 = incumbent")
T.run(store, "P3", "P3 winner +mse1", dict(CFG_A, w_mse=1.0), **KW)
T.run(store, "P3", "P3 winner +orth.5", dict(CFG_A, w_orth=0.5), **KW)
T.run(store, "P3", "P3 AAM only (no InfoNCE)", dict(CFG_A, w_nce=0.0, w_aam=1.0), **KW)
CFG_STD, INC_STD = V.pick(T, "P3", INC_A)
print("STANDARD model:", INC_STD)
'''

NB2_P4 = '''
# P4 - Gender model for the gender/* cells: judged on int_g only
T.run(store, "P4", "P4 = standard", CFG_STD, **KW)
T.run(store, "P4", "P4 same-gender negatives", dict(CFG_STD, same_gender_only=True), **KW)
CFG_SG, INC_SG = V.pick(T, "P4", "P4 = standard", key="int_g")
for lam in (0.05, 0.2):
    T.run(store, "P4", f"P4 winner +GRL gender {lam}", dict(CFG_SG, grl_gender=lam), **KW)
CFG_GEN, INC_GEN = V.pick(T, "P4", INC_SG, key="int_g")
print("GENDER model:", INC_GEN)
'''

NB2_P5 = '''
# P5 - Bangla. Internal val cannot see Bangla, so these become CodaBench submissions.
# (int_g_short is reported as a weak hint: English val voices cropped to Bangla-like length.)
has_crops = store.voice_crops(CFG_STD["voice"]) is not None
P5 = {"base": (CFG_STD, CFG_GEN)}
if has_crops:
    for view in ("mix", "crops"):
        T.run(store, "P5", f"P5 std voice_view={view}", dict(CFG_STD, voice_view=view), **KW)
    P5["crops_mix"] = (dict(CFG_STD, voice_view="mix"), dict(CFG_GEN, voice_view="mix"))
T.run(store, "P5", "P5 std +DANN language .1 (uses dev audio)", dict(CFG_STD, dann_lang=0.1), **KW)
P5["dann"] = (dict(CFG_STD, dann_lang=0.1), dict(CFG_GEN, dann_lang=0.1))
print(T.df[T.df.phase == "P5"][["run", "int_ng", "int_g", "proxy"] + (["int_g_short"] if "int_g_short" in T.df else [])].to_string(index=False))
'''

NB2_P6 = '''
# P6 - CCA fusion weight (internal), then write the candidate submissions
for w in (0.0, 0.15, 0.35):
    T.run(store, "P6", f"P6 std cca_w={w}", CFG_STD, cca_w=w, **KW)
d6 = T.df[T.df.run.str.startswith("P6") | (T.df.run == INC_STD)]
W_BEST = 0.25
best6 = T.df[T.df.phase == "P6"].sort_values("proxy").iloc[0]
if best6.proxy <= T.get(INC_STD)["proxy"] - 0.3:
    W_BEST = float(best6.run.split("=")[1])
print("CCA weight:", W_BEST)

N_FINAL = 2 if LOCAL else 10
SUBS = WORKDIR / "submissions"; SUBS.mkdir(exist_ok=True)
MANI = []
for tag_, (cs, cg) in P5.items():
    std = V.dev_scores(store, cs, n_models=N_FINAL, cca_w=W_BEST)
    gen = std if cg == cs else V.dev_scores(store, cg, n_models=N_FINAL, cca_w=W_BEST)
    p = SUBS / f"sub_{tag_}.zip"
    V.write_zip(store, p, V.merge_cells(std, gen)); V.check_zip(store, p)
    MANI.append(dict(zip=p.name, standard=json.dumps(cs), gender=json.dumps(cg), cca_w=W_BEST))
pd.DataFrame(MANI).to_csv(SUBS / "submissions.csv", index=False)
print(pd.DataFrame(MANI)[["zip", "cca_w"]])
'''

NB2_SUM = '''
print(T.df[["phase", "run", "int_ng", "int_g", "proxy", "sd_g", "sec"]].to_string(index=False))
print("""
Submit to CodaBench, in this order (one variable each):
  1. submissions/sub_base.zip       - new features + P1-P4 winners      (vs 30.90)
  2. submissions/sub_crops_mix.zip  - + duration-matched voice crops    (read the two Bangla cells)
  3. submissions/sub_dann.zip       - + DANN language on dev audio      (transductive: note it in the report)
Then per-cell merge the best cells with NB-3 (FLAG_06_submit).""")
'''

nb([("markdown", NB2_INTRO), ("code", ENV), ("code", embed_lib("flag_lib.py", "flag_v2.py")), ("code", NB2_SETUP),
    ("markdown", "## P0 — feature sanity + unimodal quality"), ("code", NB2_P0),
    ("markdown", "## P1 — head / regularisation (organiser features)"), ("code", NB2_P1),
    ("markdown", "## P2 — features (the decisive test)"), ("code", NB2_P2),
    ("markdown", "## P3 — loss"), ("code", NB2_P3),
    ("markdown", "## P4 — Gender model"), ("code", NB2_P4),
    ("markdown", "## P5 — Bangla (decided on CodaBench)"), ("code", NB2_P5),
    ("markdown", "## P6 — fusion weight + candidate submissions"), ("code", NB2_P6),
    ("markdown", "## Summary"), ("code", NB2_SUM)], "FLAG_05_ablate.ipynb")

# ============================================================================ NB-3
NB3_INTRO = """
# FLAG 2027 — NB-3: targeted submissions from configs (no ablation)

Each entry of `RUNS` is trained on all 70 speakers (ensemble of 10) and scored on all four dev cells
-> one zip per run. Per-cell merging of zips that CodaBench has scored is done locally with
`kaggle/merge_cells.py` (no retraining).

Round 2 (after NB-2): the voice feature decides the Bangla cells, and internal validation cannot see Bangla.
So each run changes ONE thing vs `r2_ecapa192` (own ECAPA-192, same recipe as EXP-007):
* `r2_ecapa192_mix` - + duration-matched voice crops (Bangla-length) as training views
* `r2_wavlm`        - WavLM-Large layers 4-9 instead of ECAPA (internal 26.61 vs control 26.16)

**Settings:** GPU, Internet OFF. Inputs: `flag2027-feats-v2` + `flag2027-features`. ~10 min per run on a T4.
"""

NB3_CFG = '''
import numpy as np, pandas as pd, json
os.environ["FLAG_DATA"] = str(IN)
import flag_v2 as V
V.polarity_preflight()
store = V.FeatureStore(IN)

# ---- edit here ---------------------------------------------------------------------------------
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)         # = EXP-007 recipe
RUNS = {
    "r2_ecapa192":     dict(REC, voice="ecapa192"),
    "r2_ecapa192_mix": dict(REC, voice="ecapa192", voice_view="mix"),
    "r2_wavlm":        dict(REC, voice="wavlm_L4-9"),
}
CCA_W = 0.25
N_MODELS = 2 if LOCAL else 10
if LOCAL:
    RUNS = {k: dict(v, epochs=2) for k, v in RUNS.items()}
    RUNS = {k: v for k, v in RUNS.items() if store.has("voice", v["voice"])}
# -------------------------------------------------------------------------------------------------
for k, v in RUNS.items():
    print(k, {x: v[x] for x in ("face", "voice", "voice_view", "w_aam", "same_gender_only")})
'''

NB3_RUN = '''
import time
SUBS = WORKDIR / "submissions"; SUBS.mkdir(exist_ok=True)
rows = []
for name, cfg in RUNS.items():
    out = SUBS / f"{name}.zip"
    if out.exists():
        print("skip (exists)", out.name); continue
    t0 = time.time()
    sc = V.dev_scores(store, cfg, n_models=N_MODELS, cca_w=CCA_W)
    np.savez(SUBS / f"{name}_scores.npz", **{"/".join(k): v for k, v in sc.items()})
    V.write_zip(store, out, sc); V.check_zip(store, out)
    rows.append(dict(zip=out.name, sec=round(time.time() - t0), cfg=json.dumps(cfg)))
    pd.DataFrame(rows).to_csv(SUBS / "runs.csv", mode="a", header=not (SUBS / "runs.csv").exists(), index=False)
    rows = []
    print(f"{name}: {round(time.time() - t0)}s")
print(sorted(p.name for p in SUBS.glob("*.zip")))
'''

nb([("markdown", NB3_INTRO), ("code", ENV), ("code", embed_lib("flag_lib.py", "flag_v2.py")),
    ("code", NB3_CFG), ("code", NB3_RUN)], "FLAG_06_submit.ipynb")


# ============================================================================ NB-4 (round 3): ReDimNet2
NB4_INTRO = """
# FLAG 2027 — NB-4: ReDimNet2 voice encoders (round 3)

One question, one variable: **does multilingual pretraining of the voice encoder help the Bangla cells?**
Two ReDimNet2-B6 checkpoints that differ ONLY in training data (MIT licence, github.com/PalabraAI/redimnet2):

| short | checkpoint | training data |
|---|---|---|
| `rdn6vox` | b6 / lm / `vox2` | VoxCeleb2 (mostly English) |
| `rdn6multi` | b6 / lm / `vb2+vox2+cnc2_v0` | VoxBlink2 + VoxCeleb2 + CN-Celeb2 (multilingual) |

Everything else is the EXP-007 recipe, and the crops reuse NB-1's `crop_plan_train.npy`, so `*_mix` runs are
directly comparable with `r2_ecapa192_mix`.

**Settings:** GPU T4, **Internet ON** (torch.hub download). **Inputs:** the raw dataset (wav), `flag2027-features`,
`flag2027-feats-v2` (crop plan). ~45-60 min total.

⚠️ ReDimNet2 has no length mask: zero padding changes its embedding (local test: cos 0.63 with 3 s of padding),
so extraction runs **one file at a time** (no batching). Slower, but exact.

Output: `feats_v3/` (new arrays + manifest), `submissions/r3_*.zip`, `*_scores.npz`, `*_matrix.npz`
(full face x voice score matrices for graph refinement), `runs.csv`, `p0_unimodal_v3.csv`.
"""

NB4_EXTRACT = '''
import numpy as np, pandas as pd, torch, time, json
import flag_extract as X
OUT3 = WORKDIR / "feats_v3"; OUT3.mkdir(exist_ok=True)
root = X.media_root(IN, WORKDIR / "raw")
S = X.index_splits(root)
plan = sorted(IN.rglob("crop_plan_train.npy"))
assert plan, "crop_plan_train.npy not found: attach flag2027-feats-v2"
# multilingual first: it is the question of this round, and it is the slow one (extra GroupNorm layers)
ENCS = [("b6", "vb2+vox2+cnc2_v0", "rdn6multi"), ("b6", "vox2", "rdn6vox")]
BUDGET_MIN = 0.5 if LOCAL else 150          # per encoder; the guard drops crops, then the encoder
DONE_ENC = []
for mn, ds, short in ENCS:
    t0 = time.time()
    model = X.load_redimnet2(mn, ds)
    try:
        X.extract_hub_voice(S, OUT3, short, model, crop_plan_file=plan[0], max_minutes=BUDGET_MIN,
                            probe_files=10 if LOCAL else 50)
        DONE_ENC.append(short)
    except X.EncoderTooSlow as e:
        print("SKIPPED:", e)
    del model; torch.cuda.empty_cache()
    print(f"{short}: {time.time() - t0:.0f}s")
print("extracted:", DONE_ENC)
X.write_manifest(OUT3, dict(encoders=ENCS, extracted=DONE_ENC, crop_plan=str(plan[0])))
'''

NB4_TRAIN = '''
os.environ["FLAG_DATA"] = str(IN)
os.environ["FLAG_FEATS_EXTRA"] = str(OUT3)
import flag_v2 as V
V.polarity_preflight()
store = V.FeatureStore(IN)
# unimodal sanity (speaker EER on train, no training): a broken extraction shows up here immediately
rows = []
for n in ["given", "ecapa192", "rdn6vox", "rdn6multi"]:
    if store.has("voice", n):
        rows.append(dict(feat=n, **V.unimodal_eer(store.voice(n, "train"), store.spk, store.gmap)))
        C = store.voice_crops(n)
        if C is not None:
            rows.append(dict(feat=n + " (Bangla-length crop)", **V.unimodal_eer(C[0], store.spk, store.gmap)))
P0 = pd.DataFrame(rows); P0.to_csv(WORKDIR / "p0_unimodal_v3.csv", index=False); print(P0.to_string(index=False))

REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)      # = EXP-007 recipe, same as r2_*
RUNS = {f"r3_{v}{'_mix' if m else ''}": dict(REC, voice=v, voice_view="mix" if m else "full")
        for v in ["rdn6multi", "rdn6vox"] if store.has("voice", v)
        for m in (False, True) if not m or store.voice_crops(v) is not None}
print("runs:", list(RUNS))
N_MODELS = 2 if LOCAL else 10
if LOCAL:
    RUNS = {k: dict(v, epochs=2) for k, v in RUNS.items()}
SUBS = WORKDIR / "submissions"; SUBS.mkdir(exist_ok=True)
for name, cfg in RUNS.items():
    out = SUBS / f"{name}.zip"
    if out.exists():
        print("skip (exists)", out.name); continue
    t0 = time.time()
    sc = V.dev_scores(store, cfg, n_models=N_MODELS, cca_w=0.25, matrix_path=SUBS / f"{name}_matrix.npz")
    np.savez(SUBS / f"{name}_scores.npz", **{"/".join(k): v for k, v in sc.items()})
    V.write_zip(store, out, sc); V.check_zip(store, out)
    pd.DataFrame([dict(zip=out.name, sec=round(time.time() - t0), cfg=json.dumps(cfg))]).to_csv(
        SUBS / "runs.csv", mode="a", header=not (SUBS / "runs.csv").exists(), index=False)
    print(f"{name}: {time.time() - t0:.0f}s")
print(sorted(p.name for p in SUBS.glob("*.zip")))
'''

NB4_DONE = '''
import shutil
shutil.rmtree(WORKDIR / "raw", ignore_errors=True)      # never ship extracted media in the output
tot = sum(p.stat().st_size for p in WORKDIR.rglob("*") if p.is_file()) / 1e9
print(f"output {tot:.2f} GB. Download: submissions/ (zips + npz) and feats_v3/ (for the next rounds).")
'''

nb([("markdown", NB4_INTRO), ("code", ENV), ("code", embed_lib("flag_lib.py", "flag_extract.py", "flag_v2.py")),
    ("markdown", "## 1. Extract ReDimNet2-B6 (vox2) and (vb2+vox2+cnc2), one file at a time"), ("code", NB4_EXTRACT),
    ("markdown", "## 2. Unimodal sanity + EXP-007 recipe on each encoder (full / Bangla-length crops)"), ("code", NB4_TRAIN),
    ("code", NB4_DONE)], "FLAG_07_redimnet.ipynb")
