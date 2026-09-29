"""Generate FLAG_08_external.ipynb (EXT-01): MAV-Celeb v1-v3 as extra TRAINING identities.

The organiser encoders were reproduced exactly (VGGFace fc7, yangwang825/ecapa-tdnn-vox2: cos 1.0000 with the
v4 CSVs), so external rows live in the same feature space as v4 train/dev and the EXP-007 recipe (organiser
features, the strongest English system) can simply see ~270 identities instead of 70.
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
# FLAG 2027 — NB-5: external identities (MAV-Celeb v1-v3) for the face-voice bridge

Every encoder swap failed (ArcFace, ReDimNet2): the bottleneck is the cross-modal bridge learned from only
70 speakers. External data is allowed if declared, so this notebook adds the organisers' own earlier releases:

| source | languages | ids | wav | size |
|---|---|---:|---:|---:|
| `v3_train` | English + German | 50 | 3.0k | 1.6 GB |
| `v1_complete` | English + Urdu | 70 (68 people: duplicate ids merged by name) | 19.6k | 11 GB |
| `v2_complete` | English + Hindi | 84 | 20.7k | 13 GB |

No Bangla anywhere (FAME rule: no pretraining on the unheard language).
Features are extracted with the organisers' exact encoders (verified cos 1.0000 on v4 train):
voice `yangwang825/ecapa-tdnn-vox2` (192-d), face VGGFace fc7 (4096-d). 8 frames per video.

**Settings:** GPU T4, **Internet ON** (HF + VGGFace weights; Drive only for sources not attached). If `mavceleb_v1_complete*.zip` / `mavceleb_v2_complete*.zip` are attached as a dataset they are used directly. **Inputs:** `flag2027-features`,
`flag2027-feats-v2`. Zips are downloaded one at a time to local disk and deleted after extraction.
Expected: download + extraction ~40-70 min (Drive speed dominates), training ~40 min.

Output: `feats_ext/` (features + meta, ~0.3 GB), `overlap.csv`, `ablation_ext.csv`, `submissions/ext_*.zip`
+ `*_scores.npz` + `*_matrix.npz`.
"""

ENV = '''
import os, sys
from pathlib import Path
LOCAL = os.environ.get("FLAG_LOCAL") == "1"
IN = Path(os.environ.get("FLAG_IN", "/kaggle/input"))
WORKDIR = Path(os.environ.get("FLAG_OUT", "/kaggle/working"))
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
print("LOCAL" if LOCAL else "KAGGLE", "| input", IN, "| working", WORKDIR)
'''

EXTRACT = '''
if not LOCAL:
    !pip -q install -U "speechbrain>=1.1.1" 2>&1 | tail -2
import shutil, time, numpy as np, pandas as pd, torch
import flag_extract as X
OUTX = WORKDIR / "feats_ext"; OUTX.mkdir(exist_ok=True)
TMP = Path(os.environ.get("FLAG_TMP", "/tmp/flag_ext")); TMP.mkdir(parents=True, exist_ok=True)
print("free disk at", TMP, round(shutil.disk_usage(TMP).free / 1e9, 1), "GB")
ve = X.OrganiserVoice(WORKDIR / "_ecapa_btc")
fe = X.VGGFace(WORKDIR / "_vggface")
# Google Drive "Quota exceeded": open the organiser's link in a browser, "Make a copy" into your own Drive,
# share it as "Anyone with the link", and paste the NEW file id here, e.g. {"v2_complete": "1AbC..."}.
# The mirror is tried first, the organiser's id second.
DRIVE_IDS = {}
SOURCES = {"mini_v3": None} if LOCAL else {k: ([DRIVE_IDS[k], fid] if k in DRIVE_IDS else fid, gb)
                                           for k, (fid, gb) in X.EXT_SOURCES.items()}
for name, spec in SOURCES.items():
    if (OUTX / f"ext_{name}_voice.npy").exists() and (OUTX / f"ext_{name}_face.npy").exists():
        print("skip (exists)", name); continue
    t0 = time.time()
    try:
        attached = None if LOCAL else X.find_attached_source(IN, name)
        if LOCAL:
            zp = Path(os.environ["FLAG_EXT_MINI"])
        elif attached is not None:                       # uploaded as a Kaggle dataset: no Drive download
            zp = attached
            print(f"{name}: using attached {zp}")
        else:
            fid, gb = spec
            zp = X.fetch_zip(fid, TMP / f"{name}.zip", min_free_gb=gb * 1.2 + 2)
        X.write_speaker_meta(zp, name, OUTX)
        X.extract_external(zp, name, OUTX, ve, fe, frames_per_video=8, max_wav_per_spk=150, max_sec=12)   # training uses <=150 rows/speaker
    except (AssertionError, RuntimeError, OSError) as e:        # Drive quota / disk: skip this source, keep going
        print(f"SKIPPED {name}: {type(e).__name__}: {e}")
        continue
    if not LOCAL and (TMP / f"{name}.zip").exists():
        (TMP / f"{name}.zip").unlink()                         # only after a successful extraction
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
del ve, fe; torch.cuda.empty_cache()
X.write_manifest(OUTX, dict(sources=list(SOURCES), encoders="organiser VGGFace fc7 + yangwang825/ecapa-tdnn-vox2"))
'''

OVERLAP = '''
# Identity overlap: an external speaker who is also a v4 train/dev person would leak. Face centroids (VGG fc7,
# standardised with v4-train stats, PCA-256) per identity; threshold calibrated on v4 train: two disjoint halves
# of the SAME v4 speaker vs DIFFERENT v4 speakers.
os.environ["FLAG_DATA"] = str(IN)
os.environ["FLAG_FEATS_EXTRA"] = str(OUTX)
import flag_v2 as V
store = V.FeatureStore(IN)
Xf = store.Xf; mf, sf = Xf.mean(0), Xf.std(0) + 1e-6
P = np.linalg.svd((Xf - mf) / sf, full_matrices=False)[2][:256].T
emb = lambda A: V.l2n(((A - mf) / sf) @ P)
rng = np.random.RandomState(0)
same, diff, cent = [], [], {}
for s_ in np.unique(store.spk):
    i = np.where(store.spk == s_)[0]
    if len(i) < 8: continue
    rng.shuffle(i); h = len(i) // 2
    a, b = V.l2n(emb(Xf[i[:h]]).mean(0, keepdims=True))[0], V.l2n(emb(Xf[i[h:]]).mean(0, keepdims=True))[0]
    same.append(a @ b); cent[s_] = V.l2n(emb(Xf[i]).mean(0, keepdims=True))[0]
C4 = np.stack(list(cent.values())); diff = (C4 @ C4.T)[np.triu_indices(len(C4), 1)]
# Threshold = midpoint between the SAME-person and DIFFERENT-person distributions. Cross-dataset centroids sit a
# little higher than within-v4 different pairs (shared crop style), so the different-person tail alone would flag
# 12/50 v3 Europeans as "Bangladeshi celebrities" (checked locally); a true overlap sits near the same-person mode.
THR = float((np.percentile(diff, 99.5) + np.percentile(same, 1)) / 2)
print(f"v4 train: same-person half-centroid cos p1 {np.percentile(same, 1):.3f} | different-person p99.5 {np.percentile(diff, 99.5):.3f} -> centroid threshold {THR:.3f}")
# dev check: mean cos of a centroid to its 20 nearest faces, calibrated with v4 train people vs OTHER people's faces
E4 = emb(Xf); ks = list(cent)
absent = [np.sort(E4[store.spk != s_] @ cent[s_])[::-1][:20].mean() for s_ in ks]
present = [np.sort(E4[store.spk == s_] @ cent[s_])[::-1][:20].mean() for s_ in ks]
THR_DEV = float((np.percentile(absent, 99) + np.percentile(present, 5)) / 2)
print(f"dev check: top-20 face cos ABSENT p99 {np.percentile(absent, 99):.3f} | PRESENT p5 {np.percentile(present, 5):.3f} -> threshold {THR_DEV:.3f}")
devF = np.concatenate([store.dev[k][0] for k in V.CELLS]); devE = emb(devF)
rows = []
for f in sorted(OUTX.glob("ext_*_face.npy")):
    src = f.name[4:-9]; fm = pd.read_csv(OUTX / f"ext_{src}_face_meta.csv"); A = np.load(f).astype(np.float32)
    E_ = emb(A)
    for s_, g in fm.groupby("spk"):
        c = V.l2n(E_[g.index].mean(0, keepdims=True))[0]
        best_tr = C4 @ c
        nn_dev = np.sort(devE @ c)[::-1][:20].mean()       # centroid vs its 20 nearest dev faces
        rows.append(dict(src=src, spk=s_, key=f"{src}:{s_}", max_cos_v4train=round(float(best_tr.max()), 3),
                         v4train_id=list(cent)[int(best_tr.argmax())], top20_cos_dev=round(float(nn_dev), 3)))
OV = pd.DataFrame(rows)
OV["overlap"] = (OV.max_cos_v4train > THR) | (OV.top20_cos_dev > THR_DEV)
OV.to_csv(WORKDIR / "overlap.csv", index=False)
EXCLUDE = tuple(OV[OV.overlap].key)
print(OV.sort_values("max_cos_v4train", ascending=False).head(12).to_string(index=False))
print(f"excluded {len(EXCLUDE)} of {len(OV)} external identities:", EXCLUDE)
'''

ABLATE = '''
# Internal validation = v4 speaker-disjoint splits (external rows are always TRAINING only). One variable at a time.
REC = dict(V.BASE, head="mlp", drop=0.3, emb=128, epochs=40)           # EXP-007 recipe
ALL = tuple(s for s in (["mini_v3"] if LOCAL else list(X.EXT_SOURCES)) if (OUTX / f"ext_{s}_voice.npy").exists())
print("external sources available:", ALL)
KW = dict(seeds=(1,), n_models=1) if LOCAL else dict(seeds=(1, 2, 3), n_models=3)
if LOCAL:
    REC["epochs"] = 2
T = V.Table(WORKDIR / "ablation_ext.csv")
T.run(store, "EXT", "A control (v4 only)", REC, **KW)
# Round 1 showed v3 (European, En/German) HURTS every cell: face-voice mappings do not transfer across populations.
# So the South-Asian sources are tested one at a time, then together, then with v3 for completeness.
SA = tuple(x for x in ("v1_complete", "v2_complete") if x in ALL)
VARIANTS = {}
for x in SA:
    VARIANTS[f"B + {x}"] = dict(REC, ext=(x,), ext_exclude=EXCLUDE)
if len(SA) == 2:
    VARIANTS["C + v1+v2, cap 150"] = dict(REC, ext=SA, ext_exclude=EXCLUDE)
    VARIANTS["D + v1+v2, cap 400"] = dict(REC, ext=SA, ext_cap=400, ext_exclude=EXCLUDE)
if len(ALL) > len(SA) and SA:
    VARIANTS["E + all incl. v3"] = dict(REC, ext=ALL, ext_exclude=EXCLUDE)
if LOCAL:
    VARIANTS = {"B + mini": dict(REC, ext=ALL, ext_exclude=EXCLUDE)}
for name, cfg in VARIANTS.items():
    T.run(store, "EXT", name, cfg, **KW)
print(T.df[["run", "int_ng", "int_g", "proxy", "sd_g", "sec"]].to_string(index=False))
'''

SUBMIT = '''
# Dev scores for the control and every external variant (the local selector decides; nothing is chosen here).
import json
SUBS = WORKDIR / "submissions"; SUBS.mkdir(exist_ok=True)
N_MODELS = 2 if LOCAL else 10
RUNS = {"ext_A_control": REC}
RUNS.update({"ext_" + n.split()[0] + "_" + n.split("+ ")[1].split(",")[0].replace("+", "").replace(" ", ""): c
             for n, c in VARIANTS.items()})
print("dev runs:", list(RUNS))
for name, cfg in RUNS.items():
    out = SUBS / f"{name}.zip"
    if out.exists():
        print("skip (exists)", out.name); continue
    t0 = time.time()
    sc = V.dev_scores(store, cfg, n_models=N_MODELS, cca_w=0.25, matrix_path=SUBS / f"{name}_matrix.npz")
    np.savez(SUBS / f"{name}_scores.npz", **{"/".join(k): v for k, v in sc.items()})
    V.write_zip(store, out, sc); V.check_zip(store, out)
    pd.DataFrame([dict(zip=out.name, sec=round(time.time() - t0), cfg=json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}))]).to_csv(
        SUBS / "runs.csv", mode="a", header=not (SUBS / "runs.csv").exists(), index=False)
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
for p in [WORKDIR / "_ecapa_btc", WORKDIR / "_vggface"]:
    shutil.rmtree(p, ignore_errors=True)
print("Download: submissions/, feats_ext/, overlap.csv, ablation_ext.csv")
'''

nb([("markdown", INTRO), ("code", ENV), ("code", embed_lib("flag_lib.py", "flag_extract.py", "flag_v2.py")),
    ("markdown", "## 1. Download + extract (organiser encoders), one zip at a time"), ("code", EXTRACT),
    ("markdown", "## 2. Identity overlap with v4 train/dev -> exclusion list"), ("code", OVERLAP),
    ("markdown", "## 3. Internal ablation (v4 speaker-disjoint), external rows in training only"), ("code", ABLATE),
    ("markdown", "## 4. Dev scores for local selection"), ("code", SUBMIT)], "FLAG_08_external.ipynb")
