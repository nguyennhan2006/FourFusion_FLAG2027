"""Generate FLAG_42_vox2_lid.ipynb (T3ENC-01, Bengali filter registered 08/10): spoken-language identification of EXACTLY
the 38,345 VoxCeleb2 voice clips the T3 head trains on (FLAG_37 / FLAG_41 key list, the same 12 s centre crop), with
speechbrain/lang-id-voxlingua107-ecapa. Output: the 107 language probabilities per clip; the filter itself is applied
locally (Experiment/T3ENC-01/README.md: drop every clip of a speaker with any clip whose top language is Bengali or
p(Bengali) >= 0.2). Used only to filter training data, never in the scoring system.
    python build_vox2lid_notebook.py
    FLAG_LOCAL=1 FLAG_OUT=<dir> FLAG_CACHE=<cache> FLAG_IN=<dir with face_meta.csv / voice_meta.csv> VOX_SMOKE_IDS=3 \
        python run_notebook_local.py FLAG_42_vox2_lid.ipynb --jupyter
"""
from pathlib import Path
import build_eval_notebooks as B12
import build_vox2t3_notebook as T41

INTRO = """
# FLAG 2027 — NB-42: language identification of the VoxCeleb2 clips used by T3ENC-01 (Bengali filter)

**Why.** T3ENC-01 passed the registered gate (CA-FUSE + the T3 head trained with 4,877 VoxCeleb2 identities: g +1.59, CI
[+0.33, +2.71]). The organisers' rule: Bengali speech must not be used to train, fine-tune or adapt an audio-side model.
VoxCeleb2 is not language-labelled, so before the head enters a submission, every clip it trains on is language-identified
and Bengali speakers are removed (rule registered before this run: drop **every clip of a speaker** with any clip whose top
language is Bengali or with p(Bengali) >= 0.2).

**What runs:** `speechbrain/lang-id-voxlingua107-ecapa` (ECAPA-TDNN, 107 languages incl. `bn: Bengali`) on exactly the
38,345 clips of the FLAG_37 key list, on the same 12 s centre crop the head was trained on. Used only to filter training
data; it is never part of the scoring system.

**Inputs (Add Input):** **`flag2027-vox2-keys`** (or the FLAG_37 output `flag2027-feats-vox2`): only `face_meta.csv` /
`voice_meta.csv` are read. Optional: this notebook's previous output, to resume. **Settings:** GPU T4, **Internet ON**.

**Time:** about 1.5-2.5 h (the same tars as FLAG_37 / FLAG_41 are streamed until every key is found; the LID model is small).
Resumable per tar.

**Output:** `/kaggle/working/feats_vox2lid/` -> dataset **`flag2027-vox2-lid`** (about 10 MB).
"""

ENV = B12.ENV12.replace('OUT = WORKDIR / "feats_eval"', 'OUT = WORKDIR / "feats_vox2lid"')
assert ENV != B12.ENV12

PIP = r'''# speechbrain -U: Kaggle's preinstalled one breaks on recent torchaudio; av = PyAV decodes the AAC (.m4a) clips.
if not LOCAL:
    !pip -q install -U "speechbrain>=1.1.1" av 2>&1 | tail -2
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEV,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

CONFIG = T41.CONFIG.replace("FACE_PARTS, VOICE_PARTS = OUT / \"parts_face\", OUT / \"parts_voice\"",
                            "FACE_PARTS, VOICE_PARTS = OUT / \"parts_face\", OUT / \"parts_lid\"").replace(
    'for prev in (IN.rglob("feats_vox2t3") if IN.exists() and not LOCAL else []):',
    'for prev in (IN.rglob("feats_vox2lid") if IN.exists() and not LOCAL else []):').replace(
    '    for sub in ("parts_face", "parts_voice"):', '    for sub in ("parts_lid",):').replace(
    '"feats_vox2t3" not in str(p)', '"feats_vox2lid" not in str(p)')
assert CONFIG.count("feats_vox2lid") == 3 and "parts_lid" in CONFIG

LID = r'''
# ============================================================================ language ID of the same clips (12 s centre crop), streamed
from speechbrain.inference.classifiers import EncoderClassifier
try:                                                   # copy, not symlink, the fetched files (Windows smoke test; same on Kaggle)
    from speechbrain.utils.fetching import LocalStrategy
    _ls = {"local_strategy": LocalStrategy.COPY}
except ImportError:
    _ls = {}
LIDM = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=str(CACHE / "lid_voxlingua107"),
                                      run_opts={"device": str(DEV)}, **_ls)
LIDM.eval()
LABELS = [LIDM.hparams.label_encoder.ind2lab[i] for i in range(len(LIDM.hparams.label_encoder.ind2lab))]
BN = [i for i, l in enumerate(LABELS) if l.split(":")[0].strip() == "bn"]
assert len(LABELS) == 107 and len(BN) == 1, (len(LABELS), BN)
log(f"LID: {len(LABELS)} languages, Bengali = {LABELS[BN[0]]!r} (index {BN[0]})")


def decode_m4a(b):
    import av
    out = []
    with av.open(io.BytesIO(b)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        for fr in c.decode(c.streams.audio[0]):
            out += [o.to_ndarray().reshape(-1) for o in res.resample(fr)]
        out += [o.to_ndarray().reshape(-1) for o in res.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(TARGET_SR // 10, np.float32)


@torch.no_grad()
def lid_probs(w):
    out_prob = LIDM.classify_batch(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))[None].to(DEV))[0]
    return torch.softmax(out_prob.float().reshape(1, -1), 1)[0].cpu().numpy()       # classifier ends in log-softmax: softmax = probabilities


row_of = {(s, v, u): i for i, (s, v, u) in enumerate(zip(VK.spk, VK.video, VK.utt))}
done = set()
for p in sorted(VOICE_PARTS.glob("tar*.npz")):            # resume
    done |= set(np.load(p)["row"].tolist())
order = list(np.random.RandomState(SEED).permutation(779))
L = int(CLIP_SEC * TARGET_SR)
for t in order[:MAX_TARS]:
    if len(done) >= len(row_of):
        log("every voice key found"); break
    part = VOICE_PARTS / f"tar{t:04d}.npz"
    if part.exists():
        continue
    t0 = time.time(); probs, rows = [], []
    try:
        with get_retry(VOICE_URL.format(t), stream=True) as r:
            r.raw.decode_content = True
            tf = tarfile.open(fileobj=r.raw, mode="r|")
            for m in tf:
                if not m.name.endswith(".m4a"):
                    continue
                i = row_of.get(tuple(m.name[:-4].split("/")[-3:]))
                if i is None or i in done:
                    continue
                try:
                    w = decode_m4a(tf.extractfile(m).read())
                except Exception as e:
                    BROKEN["audio"].append(f"{m.name} | {type(e).__name__}: {e}"); continue
                probs.append(lid_probs(centre_crop(w, L))); rows.append(i)
    except Exception as e:
        log(f"!! tar {t}: {type(e).__name__}: {e} -> skipped"); continue
    np.savez(part, row=np.array(rows, int), p=np.array(probs, np.float16).reshape(len(rows), len(LABELS)))
    done |= set(rows)
    log(f"tar {t:04d}: {len(rows)} clips [{time.time() - t0:.0f}s] | keys found {len(done)}/{len(row_of)}")
'''

ASSEMBLE = r'''
# ============================================================================ assemble in key-list order, apply the REGISTERED rule (report), manifest
P = np.full((len(VK), len(LABELS)), np.nan, np.float32)
for p in sorted(VOICE_PARTS.glob("tar*.npz")):
    Z = np.load(p)
    if len(Z["row"]):
        P[Z["row"]] = Z["p"].astype(np.float32)
ok = np.isfinite(P[:, 0])
np.save(OUT / "voice_lid_probs.npy", P.astype(np.float16))
VK.to_csv(OUT / "voice_meta.csv", index=False)
(OUT / "lid_labels.json").write_text(json.dumps(LABELS, indent=0))
top = np.where(ok, np.nanargmax(np.nan_to_num(P, nan=-1), 1), -1)
pbn = np.where(ok, P[:, BN[0]], np.nan)
flag = ok & ((top == BN[0]) | (pbn >= 0.2))
spk_drop = sorted(set(VK.spk.values[flag]))
dist = pd.Series([LABELS[i].split(":")[0] for i in top[ok]]).value_counts()
summary = {"clips": int(len(VK)), "clips_scored": int(ok.sum()), "clips_top1_bengali": int((top == BN[0]).sum()),
           "clips_pbn_ge_0.2": int(np.nansum(pbn >= 0.2)), "clips_flagged": int(flag.sum()),
           "speakers": int(VK.spk.nunique()), "speakers_dropped": len(spk_drop),
           "clips_dropped": int(VK.spk.isin(spk_drop).sum()), "top_languages": dist.head(15).to_dict()}
(OUT / "lid_summary.json").write_text(json.dumps({**summary, "speakers_dropped_ids": spk_drop}, indent=1))
print(json.dumps(summary, indent=1))
print(f"check: probabilities sum to 1 (max deviation {float(np.nanmax(np.abs(np.nansum(P, 1)[ok] - 1))) if ok.any() else float('nan'):.2e})")
man = {"created": time.strftime("%Y-%m-%d %H:%M"), "model": "speechbrain/lang-id-voxlingua107-ecapa", "summary": summary,
       "rule": "drop every clip of a speaker with any clip whose top language is Bengali or p(Bengali) >= 0.2 (registered 08/10)",
       "samples": "the FLAG_37 key list, 12 s centre crop (the audio the T3 head trains on)", "broken": BROKEN,
       "versions": {"torch": torch.__version__}}
(OUT / "feats_manifest.json").write_text(json.dumps(man, indent=1, default=str))
print("NEXT: Save Version -> Output -> New Dataset -> name it flag2027-vox2-lid (private).")
'''

if __name__ == "__main__":
    B12.nb([("markdown", INTRO), ("code", ENV), ("code", PIP),
            ("markdown", "## 1. Audio helpers (FLAG_12 cell)"), ("code", B12.AUDIO12),
            ("markdown", "## 2. FLAG_37 key lists (same cell as FLAG_41)"), ("code", CONFIG),
            ("markdown", "## 3. Language ID of every clip; resumable per tar"), ("code", LID),
            ("markdown", "## 4. Assemble, registered rule (report only), manifest"), ("code", ASSEMBLE)], "FLAG_42_vox2_lid.ipynb")
