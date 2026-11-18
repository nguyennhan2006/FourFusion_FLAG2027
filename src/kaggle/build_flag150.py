r"""FLAG_150 (LIDSTRICT-01 layer 3, 2/11): Whisper-large-v3 language ID on EVERY cached training clip (the centre 3 s of the
VoxCeleb2 / VoxBlink clips the CC heads and adapters learn from; duongphihoang/flag2027-audio-cache-p) and on every MAV-Celeb v1 / v2
voice file (full utterance, 30 s windows; + VoxLingua107, which FourFusion never ran on MAV). South-Asian risk speakers first.
Flag (Experiment/LIDSTRICT-01/README.md, docs/DATA_CLEAN_2026-11-02.md): Whisper top-1 bn / as or p(bn) >= 0.1; VoxLingua107
top-1 bn / as or p(bn) + p(as) >= 0.2; one flagged window drops the speaker. Output lid_whisper.json + per-clip parquet.

    python kaggle/build_flag150.py      -> kaggle/FLAG_150_lidwhisper.ipynb
GPU T4 x 2, Internet ON (HF models, pip speechbrain), private (VoxBlink). Smoke: --smoke 'WL_MODEL=openai/whisper-tiny;WL_MAX=24'.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Experiment"


def risk_lists():
    lab = [l.split(":")[0].strip() for l in json.load(open(EXP / "T3ENC-01/feats_vox2lid/lid_labels.json"))]
    ia = [lab.index(c) for c in "hi ur as bn ne pa ta te ml mr gu kn si sd".split()]
    bn, as_ = lab.index("bn"), lab.index("as")
    out = {}
    for src, dirs in [("vox2", ["T3ENC-01/feats_vox2lid", "SCALE-ID-01/feats_vox2b"]), ("vb", ["SCALE-VB/feats_vb"])]:
        P = np.concatenate([np.load(EXP / d / "voice_lid_probs.npy").astype(np.float32) for d in dirs])
        s = np.concatenate([pd.read_csv(EXP / d / "voice_meta.csv", dtype=str).spk.to_numpy() for d in dirs])
        r = np.isin(P.argmax(1), ia) | (P[:, bn] + P[:, as_] >= .01)
        out[src] = sorted(set(s[r]))
    return out


SETUP = r'''print("[cell] setup", flush=True)
print("[boot] FLAG_150 LIDSTRICT-01 layer 3 (Whisper-large-v3 + VoxLingua107 on MAV)", flush=True)
import os, sys, json, time, re, zipfile, io, threading, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch
OUT = Path("/kaggle/working/lidw"); OUT.mkdir(parents=True, exist_ok=True)
WL_MODEL = os.environ.get("WL_MODEL", "openai/whisper-large-v3")
WL_MAX = int(os.environ.get("WL_MAX", "0"))                       # smoke: clips per source and pass
SR = 16000
RISK = json.loads(r"""__RISK__""")
RISK = {k: set(v) for k, v in RISK.items()}
T0 = time.time()
def droots(key):
    """attached dataset folders whose name contains key (a full rglob of /kaggle/input walks MAV's face frames: 34 min in the smoke)"""
    I = Path("/kaggle/input")
    c = [p for p in I.iterdir() if p.is_dir()] + [q for q in I.glob("datasets/*/*") if q.is_dir()]
    return [p for p in c if key in p.name.lower()]
def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
log("risk speakers", {k: len(v) for k, v in RISK.items()}, "| model", WL_MODEL, "| GPUs", torch.cuda.device_count())
'''

WHISPER = r'''print("[cell] whisper model", flush=True)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.tokenization_whisper import LANGUAGES
DEVS = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
DT = torch.float16 if DEVS[0] != "cpu" else torch.float32
PROC = WhisperProcessor.from_pretrained(WL_MODEL)
TOK = PROC.tokenizer
_ids = TOK.convert_tokens_to_ids([f"<|{c}|>" for c in LANGUAGES])
CODES = [c for c, i in zip(LANGUAGES, _ids) if i is not None and i != TOK.unk_token_id]
LID_IDS = [i for i in _ids if i is not None and i != TOK.unk_token_id]
SOT = TOK.convert_tokens_to_ids("<|startoftranscript|>")
IBN, IAS = CODES.index("bn"), CODES.index("as")
MODELS = [WhisperForConditionalGeneration.from_pretrained(WL_MODEL, torch_dtype=DT, attn_implementation="sdpa").to(d).eval() for d in DEVS]
BS = 16 if DEVS[0] != "cpu" else 4
log(f"whisper {WL_MODEL}: {len(CODES)} languages on {DEVS}")


@torch.no_grad()
def _wlid(waves, k):
    m, d = MODELS[k], DEVS[k]
    f = PROC.feature_extractor(waves, sampling_rate=SR, return_tensors="pt").input_features.to(d, DT)
    e = m.model.encoder(f).last_hidden_state
    x = m.model.decoder(input_ids=torch.full((len(waves), 1), SOT, device=d), encoder_hidden_states=e).last_hidden_state
    return m.proj_out(x)[:, -1, LID_IDS].float().softmax(-1).cpu().numpy()


def wlid(waves):
    """language probabilities (len(waves) x len(CODES)), the work split over every GPU"""
    res = [None] * len(DEVS)
    parts = np.array_split(np.arange(len(waves)), len(DEVS))
    def run(k):
        idx = parts[k]
        res[k] = np.concatenate([_wlid([waves[j] for j in idx[b:b + BS]], k) for b in range(0, len(idx), BS)]) if len(idx) else np.zeros((0, len(CODES)), np.float32)
    th = [threading.Thread(target=run, args=(k,)) for k in range(len(DEVS))]
    [t.start() for t in th]; [t.join() for t in th]
    return np.concatenate(res)


def rows_of(P, base):
    top = P.argmax(1)
    return pd.DataFrame({**base, "w_top1": [CODES[i] for i in top], "w_ptop1": P[np.arange(len(P)), top], "w_pbn": P[:, IBN],
                         "w_pas": P[:, IAS], "w_phi": P[:, CODES.index("hi")], "w_pur": P[:, CODES.index("ur")]})
_t = np.random.default_rng(0).standard_normal(SR * 3).astype(np.float32) * 0.01
log("wiring check (noise):", rows_of(wlid([_t, _t]), {"k": [0, 1]}).to_dict("records")[0])
'''

CACHE = r'''print("[cell] audio cache: every cached VoxCeleb2 / VoxBlink training clip (risk speakers first)", flush=True)
_t = [p for r in droots("audio-cache") for p in r.rglob("tar0000.npz") if "/vox/" in p.as_posix()]
assert _t, "attach duongphihoang/flag2027-audio-cache-p"
ACROOT = _t[0].parent.parent
CH = [(p, "vox2") for p in sorted((ACROOT / "vox").glob("tar*.npz"))] + [(p, "vb") for p in sorted((ACROOT / "vb").glob("*.npz"))]
log("audio cache", ACROOT, len(CH), "chunks")
seen = {"vox2": set(), "vb": set()}
for ps, name in ((1, "risk"), (2, "rest")):
    done, cnt, t1 = 0, {"vox2": 0, "vb": 0}, time.time()
    for ci, (p, src) in enumerate(CH):
        Z = np.load(p)
        spk = Z["spk"].astype(str); n = Z["n"].astype(int)
        risk = np.isin(spk, list(RISK[src]))
        sel = np.where(risk if ps == 1 else ~risk)[0]
        if WL_MAX:                                                  # smoke: a few clips of each source and pass
            sel = sel[:max(0, WL_MAX - cnt[src])]
        if not len(sel):
            continue
        cnt[src] += len(sel)
        W = Z["wav"]
        waves = [W[i][:max(int(n[i]), SR // 2)].astype(np.float32) / 32768.0 for i in sel]
        P = wlid(waves)
        D = rows_of(P, {"src": src, "spk": spk[sel], "chunk": p.name, "idx": sel, "n": n[sel], "risk": ps == 1})
        D.to_parquet(OUT / f"cache_p{ps}_{ci:04d}.parquet")
        seen[src] |= set(spk[sel]); done += len(sel)
        log(f"pass {name} chunk {ci + 1}/{len(CH)} {src}: {len(sel)} clips (total {done}, {done / max(time.time() - t1, 1):.1f} clips/s)")
miss = {s: sorted(RISK[s] - seen[s]) for s in RISK}
log("risk speakers without a cached clip (not in the training pools, or blocked earlier):", {s: len(v) for s, v in miss.items()})
(OUT / "risk_not_in_cache.json").write_text(json.dumps(miss))
'''

MAV = r'''print("[cell] MAV-Celeb v1 / v2 voices: VoxLingua107 (whole file) + Whisper (30 s windows)", flush=True)
import soundfile as sf
def version_of(parts, i):
    for q in parts[:i][::-1]:
        m = re.search(r"(?:^|[^a-z0-9])v([12])(?:[^0-9]|$)", q.lower())
        if m:
            return "v" + m.group(1)
    return None
items = []                                                      # (version, id, lang, video, file, opener)
MROOTS = droots("mavceleb")
assert MROOTS, "attach chnnhnnguyn/external-data-mavceleb"
for r in MROOTS:
    for dp, dn, fn in os.walk(r):
        dn[:] = [d for d in dn if d != "faces"]                # skip the face frames (the slow part of the tree)
        for f in fn:
            if not f.lower().endswith(".wav"):
                continue
            parts = Path(dp, f).parts
            if "voices" in parts:
                i = parts.index("voices"); v = version_of(parts, i)
                if v and len(parts) > i + 4:
                    items.append((v, parts[i + 1], parts[i + 2], parts[i + 3], parts[i + 4], ("f", str(Path(dp, f)))))
if not items:                                                   # zips not auto-extracted
    for z in [z for r in MROOTS for z in r.rglob("*.zip")]:
        if not re.search(r"v[12]_complete", z.name.lower()):
            continue
        with zipfile.ZipFile(z) as zz:
            for nm in zz.namelist():
                parts = tuple(nm.split("/"))
                if nm.lower().endswith(".wav") and "voices" in parts:
                    i = parts.index("voices"); v = version_of(parts, i) or ("v1" if "v1_complete" in z.name.lower() else "v2")
                    if len(parts) > i + 4:
                        items.append((v, parts[i + 1], parts[i + 2], parts[i + 3], parts[i + 4], ("z", str(z), nm)))
items.sort(key=lambda r: r[:5])
if WL_MAX:
    items = items[:WL_MAX]
log("MAV voice files", len(items), pd.Series([r[0] for r in items]).value_counts().to_dict() if items else {})
assert items, "attach chnnhnnguyn/external-data-mavceleb (MAV-Celeb v1 / v2)"
_zc = {}
def read(op):
    if op[0] == "f":
        w, sr = sf.read(op[1], dtype="float32", always_2d=True)
    else:
        z = _zc.setdefault(op[1], zipfile.ZipFile(op[1]))
        w, sr = sf.read(io.BytesIO(z.read(op[2])), dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != SR:
        import torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
    return w
# VoxLingua107 (speechbrain), never run on MAV by FourFusion
VL = None
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speechbrain"], check=True)
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    VL = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="/kaggle/working/lid_vl107",
                                        run_opts={"device": DEVS[0]}, local_strategy=LocalStrategy.COPY)
    VLAB = [VL.hparams.label_encoder.ind2lab[i].split(":")[0].strip() for i in range(len(VL.hparams.label_encoder.ind2lab))]
    VBN, VAS = VLAB.index("bn"), VLAB.index("as")
    log("VoxLingua107 ready,", len(VLAB), "labels")
except Exception as e:
    log("!! VoxLingua107 unavailable:", repr(e)[:300])
recs, wins, keys = [], [], []
t1 = time.time()
for k, (v, sid, lang, vid, fn, op) in enumerate(items):
    try:
        w = read(op)
    except Exception as e:
        recs.append(dict(src="mav_" + v, spk=sid, lang=lang, video=vid, file=fn, err=repr(e)[:120])); continue
    r = dict(src="mav_" + v, spk=sid, lang=lang, video=vid, file=fn, dur=len(w) / SR)
    if VL is not None:
        with torch.no_grad():
            q = torch.softmax(VL.classify_batch(torch.from_numpy(w)[None].to(DEVS[0]))[0].float().reshape(1, -1), 1)[0].cpu().numpy()
        r.update(v_top1=VLAB[int(q.argmax())], v_pbn=float(q[VBN]), v_pas=float(q[VAS]))
    recs.append(r)
    for s in range(0, max(len(w), 1), 30 * SR):
        seg = w[s:s + 30 * SR]
        if len(seg) >= SR // 2 or s == 0:
            wins.append(seg); keys.append(len(recs) - 1)
    if len(wins) >= 256 or k == len(items) - 1:
        P = wlid(wins)
        for j, kk in enumerate(keys):
            o = recs[kk]; top = int(P[j].argmax())
            flag =CODES[top] in ("bn", "as") or P[j, IBN] >= 0.1
            o["w_flag"] = bool(o.get("w_flag", False) or flag)
            o["w_pbn"] = float(max(o.get("w_pbn", 0.0), P[j, IBN])); o["w_pas"] = float(max(o.get("w_pas", 0.0), P[j, IAS]))
            o.setdefault("w_top1s", []).append(CODES[top])
        wins, keys = [], []
        log(f"MAV {k + 1}/{len(items)} files ({(k + 1) / max(time.time() - t1, 1):.1f} files/s)")
M = pd.DataFrame(recs)
M["w_top1s"] = M["w_top1s"].map(lambda x: ",".join(x) if isinstance(x, list) else "")
M.to_parquet(OUT / "mav.parquet")
'''

SUMMARY = r'''print("[cell] summary: flagged speakers (one flagged window drops the speaker)", flush=True)
C = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("cache_p*.parquet"))]) if list(OUT.glob("cache_p*.parquet")) else pd.DataFrame()
flag = {}
if len(C):
    C["flag"] = C.w_top1.isin(["bn", "as"]) | (C.w_pbn >= 0.1)
    for s in ("vox2", "vb"):
        flag[s] = sorted(set(C.spk[(C.src == s) & C.flag]))
M = pd.read_parquet(OUT / "mav.parquet")
mf = M.get("w_flag", pd.Series(False, index=M.index)).fillna(False).astype(bool)
if "v_top1" in M:
    mf |= M.v_top1.isin(["bn", "as"]) | (M.v_pbn.fillna(0) + M.v_pas.fillna(0) >= 0.2)
if "err" in M:
    mf |= M.err.notna()                                           # no evidence = flagged (principle 5)
for v in ("mav_v1", "mav_v2"):
    flag[v] = sorted(set(M.spk[(M.src == v) & mf]))
lis = []                                                        # layer 4: borderline speakers to listen to
if len(C):
    g = C.assign(sc=C.w_pbn + C.w_pas).groupby(["src", "spk"]).sc.max()
    lis = [dict(src=a, spk=b, max_w_pbn_pas=round(float(x), 3)) for (a, b), x in g.items() if 0.05 <= x and b not in set(flag.get(a, []))]
J = dict(exp="LIDSTRICT-01", layer="3 (FLAG_150)", model=WL_MODEL, created=time.strftime("%Y-%m-%d %H:%M"),
         rule="Whisper top-1 bn / as or p(bn) >= 0.1 on any window; MAV also VoxLingua107 top-1 bn / as or p(bn) + p(as) >= 0.2; unreadable file = flagged",
         clips={"cache": int(len(C)), "mav": int(len(M))}, speakers_scanned={s: int(C.spk[C.src == s].nunique()) for s in ("vox2", "vb")} if len(C) else {},
         mav_speakers={v: int(M.spk[M.src == v].nunique()) for v in ("mav_v1", "mav_v2")},
         counts={k: len(v) for k, v in flag.items()}, flagged=flag, listen=lis[:500], voxlingua_on_mav="v_top1" in M)
(OUT / "lid_whisper.json").write_text(json.dumps(J, indent=1))
pd.DataFrame(lis).to_csv(OUT / "listen_candidates.csv", index=False)
log("FLAGGED speakers:", J["counts"], "| listen candidates", len(lis))
if len(C):
    log("top-1 languages (cache):", C.w_top1.value_counts().head(12).to_dict())
log("ALL CELLS OK")
'''


def main():
    risk = json.dumps(risk_lists())
    cells = [SETUP.replace("__RISK__", risk), WHISPER, CACHE, MAV, SUMMARY]
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "markdown", "metadata": {}, "source": __doc__}] +
                   [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in cells]}
    out = ROOT / "kaggle/FLAG_150_lidwhisper.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    for c in cells:
        compile(c, "cell", "exec")
    print(out, f"{out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
