"""Generate the FACELORA-03 adapter notebooks (registered in Experiment/FACELORA-03/README.md): a LoRA adapter on CLIP ViT-L/14
trained against FROZEN SimAM voices on ~21k identities (VoxCeleb2 of FLAG_37 + FLAG_53, VoxBlink-clean of FLAG_46), every batch
16 VoxCeleb2 + 16 VoxBlink identities, 4,600 steps; then the adapted CLIP-L of every MAV-Celeb row and of every VoxCeleb2 /
VoxBlink face row (row order of the stored arrays).
    FLAG_65_facelora3_a.ipynb   A3: multi-positive symmetric InfoNCE (FACELORA-02's loss)        -> clipA3 (768)
    FLAG_65_facelora3_b.ipynb   B3: + gender-balanced batches and 0.5 x same-gender InfoNCE      -> clipB3 (768)
    python build_facelora3_notebook.py [a | b]
Frames: FLAG_45 output (vox_frames_png.zip, FLAG_37 rows by row number) + FLAG_63 output (frames/vb, frames/vox2b by spk/video/utt):
both are private notebook outputs of ONE account, so this notebook runs on that account.
Smoke test on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6;SMOKE_IDS=4;LORA_STEPS=3;LORA_P=4;LORA_K=2
"""
import json, sys
from pathlib import Path
import build_eval_notebooks as B12
import build_rep_notebooks as BR

HERE = Path(__file__).resolve().parent
CELL = (HERE / "vendor_t3enc/t3enc_cell.py").read_text(encoding="utf-8")
SMOKE = '(LOCAL or os.environ.get("FLAG_SMOKE") == "1")'
VARIANTS = {"a": dict(arm="A3", arr="clipA3", sg=0.0, out="feats_facelora3_a"),
            "b": dict(arm="B3", arr="clipB3", sg=0.5, out="feats_facelora3_b"),
            # FACELORA-04 (15/10, Experiment/FACELORA-04/README.md): the B3 recipe on FaRL ep16 (FACELORA-02's FaRL LoRA settings)
            "fb": dict(arm="FB4", arr="farlB4", sg=0.5, out="feats_facelora4_fb", enc="farl16", nb="FLAG_65f_facelora4_fb.ipynb",
                       ds="flag2027-feats-facelora4-fb-p"),
            # FACELORA-05 (16/10, Experiment/FACELORA-05/README.md): the B3 recipe with a larger CLIP-L adapter (MLP + attention, blocks 12-23)
            "c5": dict(arm="C5", arr="clipC5", sg=0.5, out="feats_facelora5_c5", exp="FACELORA-05", nb="FLAG_65g_facelora5_c5.ipynb",
                       ds="flag2027-feats-facelora5-c5-p"),
            # B3-EVAL (16/10, BEST/v1977): the SAVED B3 adapter (no training) -> clipB3 of every MAV-Celeb row, test files included
            "b3x": dict(arm="B3", arr="clipB3", sg=0.5, out="feats_clipb3_x", exp="B3-EVAL", nb="FLAG_65x_clipb3_extract.ipynb",
                        ds="flag2027-feats-clipb3-test-p"),
            # TAP-01 (29/10, Experiment/TAP-01/README.md): the SAVED B3 adapter + frozen FaRL ep16, intermediate taps of every face row
            "tap": dict(arm="B3", arr="clipB3", sg=0.5, out="feats_tap01_face", exp="TAP-01", nb="FLAG_103a_tap01_face.ipynb",
                        ds="flag2027-feats-tap01-face-p"),
            # TTA-02 (16/10, Experiment/TTA-02/README.md): the SAVED B3 adapter, embedding = mean of the image and its horizontal flip
            "b3t": dict(arm="B3", arr="clipB3t", sg=0.5, out="feats_clipb3_t", exp="TTA-02", nb="FLAG_65t_clipb3_tta.ipynb",
                        ds="flag2027-feats-clipb3-tta-p"),
            # FACELORA-06 (16/10, Experiment/FACELORA-06/README.md): the B3 recipe, half of each source's identities per step South-Asian
            "sa": dict(arm="SA", arr="clipSA", sg=0.5, out="feats_facelora6_sa", exp="FACELORA-06", nb="FLAG_65s_facelora6_sa.ipynb",
                       ds="flag2027-feats-facelora6-sa-p"),
            # COTRAIN-01 (19/10, Experiment/COTRAIN-01/README.md): the C5 recipe towards [sam100p | sam100fM] (VOICELORA-02's adapted voice)
            "co": dict(arm="CO", arr="clipCO", sg=0.5, out="feats_cotrain1_co", exp="COTRAIN-01", nb="FLAG_65c_cotrain1.ipynb",
                       ds="flag2027-feats-cotrain1-co-p"),
            # JOINT-01 (19/10, Experiment/JOINT-01/README.md): the C5 face adapter + the unfrozen part of sam100f in one loss (T4 x 2)
            "jt": dict(arm="JT", arr="clipJT", sg=0.5, out="feats_joint1_jt", exp="JOINT-01", nb="FLAG_65j_joint1.ipynb",
                       ds="flag2027-feats-joint1-p"),
            # CLEANTUNE-01 (31/10, Experiment/CLEANTUNE-01/README.md): B3 / C5 retrained with FLAG_53 rows also filtered by FLAG_42's
            # Bengali list (12 of its speakers sat in FLAG_53 `more`); same array names, marker bn42_clean.json for the CC head
            "b3c": dict(arm="B3", arr="clipB3", sg=0.5, out="feats_cleantune_b3c", nb="FLAG_109b_b3c.ipynb",
                        ds="flag2027-feats-ct-b3c-p", bn42=True),
            "c5c": dict(arm="C5", arr="clipC5", sg=0.5, out="feats_cleantune_c5c", exp="FACELORA-05", nb="FLAG_109c_c5c.ipynb",
                        ds="flag2027-feats-ct-c5c-p", bn42=True),
            # ALF-01 (31/10, Experiment/ALF-01/README.md): TAP-01's taps through the SAVED B3c adapter (CLEANTUNE-01 step 0) instead of
            # B3 (FLAG_107 leak). No bn42 marker here: the CC head takes its clipB3 (and the marker) from the B3c dataset itself.
            "tapc": dict(arm="B3", arr="clipB3", sg=0.5, out="feats_tap01c_face", exp="TAP-01", nb="FLAG_103ac_alf01_tapc.ipynb",
                         ds="flag2027-feats-tap01c-face-p", saved="ngnnguynchn/flag2027-feats-ct-b3c-p"),
            # CLEANTUNE-01 Evaluation path (2/11, docs/PLAN_EVAL_F0F1_2026-11-02.md): the SAVED clean adapters B3c / C5c, no training,
            # -> clip<X> of the test lists attached (dev = R11 rehearsal, Evaluation on 11/11); every other file of the saved dataset
            # (train / v1 / v2 / VoxCeleb2 / VoxBlink arrays, bn42_clean.json) copied in, so this output replaces the dataset in FLAG_112x
            "b3cx": dict(arm="B3", arr="clipB3", sg=0.5, out="feats_clipb3c_x", exp="B3-EVAL", nb="FLAG_65xb_clipb3c_extract.ipynb",
                         ds="flag2027-feats-clipb3c-test-p", saved="ngnnguynchn/flag2027-feats-ct-b3c-p", copy_all=True),
            "c5cx": dict(arm="C5", arr="clipC5", sg=0.5, out="feats_clipc5c_x", exp="B3-EVAL", c5=True, nb="FLAG_65xc_clipc5c_extract.ipynb",
                         ds="flag2027-feats-clipc5c-test-p", saved="nammo0/flag2027-feats-ct-c5c-p", copy_all=True),
            # LIDSTRICT-01 (3/11, Experiment/LIDSTRICT-01/README.md): B3c / C5c with the strict list (lid_strict.json) on top of the bn42
            # filter; the marker carries lid_strict = [vox2, vb] counts, which FACELORA-03 adir() checks under LID_STRICT=1
            "b3s": dict(arm="B3", arr="clipB3", sg=0.5, out="feats_lidstrict_b3s", nb="FLAG_153b_b3s.ipynb",
                        ds="flag2027-feats-ls-b3s-p", bn42=True, strict=True),
            "c5s": dict(arm="C5", arr="clipC5", sg=0.5, out="feats_lidstrict_c5s", exp="FACELORA-05", nb="FLAG_153c_c5s.ipynb",
                        ds="flag2027-feats-ls-c5s-p", bn42=True, strict=True),
            # PEFACE-01 step 0 (9/11, Experiment/PEFACE-01/README.md): FROZEN PE-Core-L14-336 (no LoRA, no training) -> peL of every
            # face row; the LIDSTRICT-01 pools are still built so the marker lets the CA head take the arrays under LID_STRICT=1
            "pe0": dict(arm="PE0", arr="peL", sg=0.5, out="feats_peface01_pe0", exp="PEFACE-01", enc="pecore",
                        nb="FLAG_129a_peface01_pe0.ipynb", ds="flag2027-feats-peface01-pe0-p", bn42=True, strict=True)}

LS_ANCHOR = 'dropvb = set(json.loads((DVB / "lid_summary.json").read_text())["speakers_dropped_ids"])\n'
LS_ADD = r'''# LIDSTRICT-01 (3/11): + the strict list (VoxLingua107 bn + as rule, Whisper-large-v3 layer of FLAG_150, DQUAL-01 G3);
# attached dataset flag2027-lid-strict-p. Speaker level: voice AND face rows of these speakers leave both pools.
LS = json.loads(one("lid_strict.json").read_text())
assert LS.get("exp") == "LIDSTRICT-01" and all(len(LS[k]) == LS["counts"][k] for k in ("vox2", "vb")), "wrong lid_strict.json"
drop42 |= set(LS["vox2"]); drop53 |= set(LS["vox2"]); dropvb |= set(LS["vb"])
log("LIDSTRICT-01 strict list:", {k: LS["counts"][k] for k in ("vox2", "vb")})
'''
LS_MARK = ("flag42_in_vox2b=int(VK2B.spk.isin(drop42 - drop53).sum()))))",
           "flag42_in_vox2b=int(VK2B.spk.isin(drop42 - drop53).sum()), lid_strict=[len(LS[\"vox2\"]), len(LS[\"vb\"])])))")
INTRO_LS = """
# LIDSTRICT-01 (3/11): adapter {arm} with the strict Bengali / harmful-data list

`lid_strict.json` (dataset `flag2027-lid-strict-p`) is unioned into the FLAG_42 / FLAG_53 / VoxBlink drop lists before the pools are
built: VoxLingua107 bn + as rule, Whisper-large-v3 (FLAG_150), DQUAL-01 G3. The marker `bn42_clean.json` carries `lid_strict`.
"""

BN42_FILTER = (('r.spk not in drop53))', 'r.spk not in drop42 | drop53))'),
               ("list(~VK2B.spk.isin(drop53))", "list(~VK2B.spk.isin(drop42 | drop53))"))
BN42_MARK = r'''
# CLEANTUNE-01 (31/10): no Bengali-flagged speaker (FLAG_42, FLAG_53, VoxBlink lists) in either training pool; the marker lets the CC
# head refuse arrays of an adapter trained without this filter (BN42=1)
_bad = sorted(i for i in set(FT[FT.ok].id) | set(VT[VT.ok].id)
              if (i.startswith("vox2:") and i[5:] in drop42 | drop53) or (i.startswith("vb:") and i[3:] in dropvb))
assert not _bad, f"Bengali-flagged speakers in the training pools: {_bad[:5]}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "bn42_clean.json").write_text(json.dumps(dict(exp="CLEANTUNE-01", drop42=len(drop42), drop53=len(drop53), dropvb=len(dropvb),
    face_ids=int(FT[FT.ok].id.nunique()), voice_ids=int(VT[VT.ok].id.nunique()),
    flag42_in_vox2b=int(VK2B.spk.isin(drop42 - drop53).sum()))))
log("CLEANTUNE-01 Bengali filter: 0 flagged speakers in the pools;", (OUT / "bn42_clean.json").read_text())
'''
INTRO_CT = """
# FLAG 2027 — NB-109{v}: CLEANTUNE-01 step 0: adapter {arm} retrained on Bengali-clean identities

**Why (Experiment/CLEANTUNE-01/README.md, registered 31/10).** 12 VoxCeleb2 speakers of FLAG_42's Bengali list sit in FLAG_53 `more`
rows (71 face, 91 voice rows), which FLAG_65 / FLAG_65g filtered with FLAG_53's own list only. This notebook is the {arm} adapter of
{exp} with the FLAG_53 rows filtered by **FLAG_42 + FLAG_53** (rule: every clip of a speaker with any Bengali-flagged clip goes).
Everything else is unchanged; the arrays keep the name `{arr}` and the output carries `bn42_clean.json` (the CC head checks it).
"""

INTRO = """
# FLAG 2027 — NB-65{v}: FACELORA-03 adapter {arm}: CLIP-L fine-tuned (LoRA) on ~21k identities against frozen SimAM voices

**Why.** FACELORA-02: a LoRA adapter on CLIP-L trained on 4,552 VoxCeleb2 identities made the CC stream 0.6-1.3 EER better on its
own, but the gain did not survive the fusion with v11. The user's direction (12/10): more identities, fine-tune the encoder,
re-tune the parameters for the data size. Registered in `Experiment/FACELORA-03/README.md`.

| | |
|---|---|
| identities | VoxCeleb2 FLAG_37 + FLAG_53 (LID rules of FLAG_42 / FLAG_53) + VoxBlink-clean (LID + noise rule of `vb_rows`): ~21k; 300 VoxCeleb2 identities held out (sanity EER only) |
| batch | 32 identities = **16 VoxCeleb2 + 16 VoxBlink** x 4 (face, voice) pairs of different videos{bal} |
| loss | symmetric multi-positive InfoNCE, tau 0.05{loss} |
| optimiser | AdamW (LoRA 1e-4, voice projection 1e-3, wd 1e-2), 200 warm-up, cosine, **4,600 steps**, fp16, seed 0; checkpoint every 500 steps |
| encoder | CLIP ViT-L/14 @ 32bd642 + LoRA r16 / a32 on fc1 / fc2 of blocks 18-23 (= FACELORA-02), gradient checkpointing |

**Wiring check:** at step 0 the adapter IS the frozen CLIP-L: 64 frames of each source must reproduce the stored CLIP-L arrays
(cosine >= 0.99), else the notebook stops before training.

**Outputs (`/kaggle/working/{out}/`):** `face_{arr}_*.npy` / `ext_*_face_{arr}.npy` (MAV-Celeb rows, FLAG_40 order),
`vox_face_{arr}.npy` (FLAG_37 rows), `vox2b_face_{arr}.npy` (FLAG_53 rows), `vb_face_{arr}.npy` (FLAG_46 rows), adapter +
train log.

**Inputs:** `flag2027-raw-v4-p`, MAV-Celeb v1 / v2 (`chnnhnnguyn/external-data-mavceleb`), `flag2027-ext-meta-p`,
`flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-feats-vox2b-p`, and the outputs of FLAG_45 (FaRL run: `vox_frames_png.zip`)
and FLAG_63 (`frames/`) of the SAME account. **GPU T4**, **Internet ON**, **private** (VoxBlink). **Time:** about 5-6 h (resumable).
"""

PIP = r'''# openai CLIP (FaRL's code base, only for the T3 cell's imports) --no-deps; av is not needed (frames are cached)
if not LOCAL:
    !pip -q install ftfy regex 2>&1 | tail -1
    !pip -q install --no-deps "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1" 2>&1 | tail -1
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEV,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

TRAIN = r'''
# ============================================================================ FACELORA-03: data of the three sources (registered filters)
import math, threading, requests
import concurrent.futures as cf
from torch.utils.checkpoint import checkpoint
from PIL import Image

SMOKE = (LOCAL or os.environ.get("FLAG_SMOKE") == "1")
CFG = dict(arm=ARM, enc="clipL", rank=16, alpha=32, blocks=list(range(18, 24)), P=int(os.environ.get("LORA_P", "32")),
           K=int(os.environ.get("LORA_K", "4")), steps=int(os.environ.get("LORA_STEPS", "4600")), tau=0.05, sg=SG,
           lr_lora=1e-4, lr_proj=1e-3, wd=1e-2, warmup=200, held_out=300, seed=0, flip=0.5, ckpt_every=500)
SMOKE_IDS = int(os.environ.get("SMOKE_IDS", "0")) if SMOKE else 0
FILES = SCAN["files"]
l2n_ = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def one(name, where=None):
    c = [p for p in FILES.get(name, []) if where is None or where in p.as_posix()]
    assert c, f"{name} ({where}) not found in the inputs"
    return sorted(c, key=lambda p: len(p.as_posix()))[0]


def voices(d):
    return np.concatenate([l2n_(np.load(d / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)


dt = {"spk": str, "video": str, "utt": str}
# -- VoxCeleb2 of FLAG_37 (lab-data: arrays in T3ENC-01/feats_vox2t3, key lists in VOX2-01/feats_vox2)
D37 = one("voice_sam100p.npy", "feats_vox2t3").parent
M37 = one("face_meta.csv", "VOX2-01/feats_vox2").parent
FK37, VK37 = pd.read_csv(M37 / "face_meta.csv", dtype=dt), pd.read_csv(M37 / "voice_meta.csv", dtype=dt)
VE37 = voices(D37)
# -- VoxCeleb2 of FLAG_53 (flag2027-feats-vox2b-p)
D2B = one("voice_sam100f.npy", "feats_vox2b").parent
FK2B, VK2B = pd.read_csv(D2B / "face_meta.csv", dtype=dt), pd.read_csv(D2B / "voice_meta.csv", dtype=dt)
VE2B = voices(D2B)
# -- VoxBlink-clean of FLAG_46 (flag2027-lab-vb-p)
DVB = one("voice_sam100f.npy", "SCALE-VB/feats_vb").parent
FKVB, VKVB = pd.read_csv(DVB / "face_meta.csv", dtype=dt), pd.read_csv(DVB / "voice_meta.csv", dtype=dt)
VEVB = voices(DVB)
assert len(VE37) == len(VK37) and len(VE2B) == len(VK2B) and len(VEVB) == len(VKVB)
# -- Bengali LID rules (speaker level) of FLAG_42, FLAG_53 (also drops old rows of its flagged speakers), FLAG_46
drop42 = set(json.loads(one("lid_summary.json", "feats_vox2lid").read_text())["speakers_dropped_ids"])
drop53 = set(json.loads((D2B / "lid_summary.json").read_text())["speakers_dropped_ids"])
dropvb = set(json.loads((DVB / "lid_summary.json").read_text())["speakers_dropped_ids"])
# -- VoxBlink noise rule of vb_rows: clip LOO cosine of SimAM ft below the 1st percentile of the same statistic on the
#    Bengali-filtered VoxCeleb2 clips of FLAG_37
def loo_cos(V, spk):
    """= SCALE-VB loo_cos: per clip, cosine to the mean of the OTHER clips of the same speaker (NaN for a single clip)."""
    out = np.full(len(V), np.nan)
    for _, idx in pd.Series(np.arange(len(V))).groupby(np.asarray(spk)).indices.items():
        if len(idx) < 2:
            continue
        S = V[idx].sum(0)
        out[idx] = np.einsum("ij,ij->i", V[idx], l2n_((S[None] - V[idx]) / (len(idx) - 1)))
    return out


XV = l2n_(np.load(D37 / "voice_sam100f.npy").astype(np.float32))
keepx = ~VK37.spk.isin(drop42).values
thr = float(np.nanquantile(loo_cos(XV[keepx], VK37.spk.values[keepx]), 0.01))
cvb = loo_cos(l2n_(np.load(DVB / "voice_sam100f.npy").astype(np.float32)), VKVB.spk.values)
vb_voice_ok = ~VKVB.spk.isin(dropvb).values & ~(cvb < thr)
log(f"VoxBlink noise threshold {thr:.4f}: {int((~vb_voice_ok).sum())} voice clips dropped (LID + noise)")
# -- genders: FLAG_37 face_meta; VoxCeleb2 official meta (gender column only) for FLAG_53; VoxBlink face_meta
VOX2_GENDER_JSON = r"""__VOX2_GENDER__"""
GEN = {}
# FLAG_53 genders from Experiment/SCALE-CC-01/vox2_gender.json (embedded at build time; the Hugging Face vox2_meta.csv read
# failed in the smoke of 14/10 with KeyError 'VoxCeleb2 ID')
GEN.update({"vox2:" + s: g for s, g in json.loads(VOX2_GENDER_JSON).items()})
GEN.update({"vox2:" + s: g for s, g in zip(FK37.spk, FK37.gender)})
GEN.update({"vb:" + s: ("m" if str(g).lower().startswith("m") else "f") for s, g in zip(FKVB.spk, FKVB.gender)})

# -- frames: FLAG_37 rows in vox_frames_png.zip ("<row>.png"), FLAG_53 / VoxBlink in the FLAG_63 chunk zips ("spk/video/utt.png")
Z37 = one("vox_frames_png.zip") if FILES.get("vox_frames_png.zip") else one("vox_frames_png.zipdata")   # zipdata: the FLAG_87 bridge
ZCH = [p for p in FILES.get("frames_manifest.json", [])]
FR63 = ZCH[0].parent if ZCH else None
assert FR63 is not None, "attach the output of FLAG_63 (frames/)"
MEMBER = {}                                                     # member name -> zip path
with zipfile.ZipFile(Z37) as z:
    for n in z.namelist():
        MEMBER["v37/" + n] = Z37
for part in ("vox2b", "vb"):
    for zf in sorted([*(FR63 / part).glob("*.zip"), *(FR63 / part).glob("*.zipdata")]):
        with zipfile.ZipFile(zf) as z:
            for n in z.namelist():
                MEMBER[f"{part}/" + n] = zf
log(f"frames indexed: {sum(k.startswith('v37/') for k in MEMBER)} FLAG_37, {sum(k.startswith('vox2b/') for k in MEMBER)} FLAG_53, "
    f"{sum(k.startswith('vb/') for k in MEMBER)} VoxBlink")
_ZT = threading.local()


def frame_bytes(member):
    zs = getattr(_ZT, "z", None)
    if zs is None:
        zs = _ZT.z = {}
    zp = MEMBER[member]
    if zp not in zs:
        zs[zp] = zipfile.ZipFile(zp)
    return zs[zp].read(member.split("/", 1)[1])


# -- one table of faces and one of voices over the three sources; identity label: vox2:<spk> (FLAG_37 + FLAG_53) / vb:<spk>
faces, voices_ = [], []
for i, r in enumerate(FK37.itertuples()):
    faces.append(("v37", i, f"v37/{i}.png", "vox2:" + r.spk, r.video, r.spk not in drop42 | drop53))
for i, r in enumerate(FK2B.itertuples()):
    faces.append(("vox2b", i, f"vox2b/{r.spk}/{r.video}/{r.utt}.png", "vox2:" + r.spk, r.video, r.spk not in drop53))
for i, r in enumerate(FKVB.itertuples()):
    faces.append(("vb", i, f"vb/{r.spk}/{r.video}/{r.utt}.png", "vb:" + r.spk, r.video, r.spk not in dropvb))
FT = pd.DataFrame(faces, columns=["src", "row", "member", "id", "video", "ok"])
FT["ok"] &= FT.member.isin(set(MEMBER))
VE = np.concatenate([VE37, VE2B, VEVB])
VT = pd.DataFrame({"src": ["v37"] * len(VK37) + ["vox2b"] * len(VK2B) + ["vb"] * len(VKVB),
                   "id": list("vox2:" + VK37.spk) + list("vox2:" + VK2B.spk) + list("vb:" + VKVB.spk),
                   "video": list(VK37.video) + list(VK2B.video) + list(VKVB.video),
                   "ok": list(~VK37.spk.isin(drop42 | drop53)) + list(~VK2B.spk.isin(drop53)) + list(vb_voice_ok)})
fv = FT[FT.ok].groupby("id").indices; vv = VT[VT.ok].groupby("id").indices
fidx, vidx = FT[FT.ok].index.values, VT[VT.ok].index.values
FACE_OF = {s: fidx[ix] for s, ix in fv.items()}; VOICE_OF = {s: vidx[ix] for s, ix in vv.items()}
ids = sorted(s for s in set(FACE_OF) & set(VOICE_OF) if s in GEN)
rng = np.random.RandomState(CFG["seed"]); torch.manual_seed(CFG["seed"])
voxids = [s for s in ids if s.startswith("vox2:")]
held = sorted(rng.choice(voxids, min(CFG["held_out"], max(1, len(voxids) // 10)), replace=False)) if len(voxids) > 2 else []
POOL = {"vox": [s for s in voxids if s not in set(held)], "vb": [s for s in ids if s.startswith("vb:")]}
if SMOKE_IDS:
    POOL = {k: v[:SMOKE_IDS] for k, v in POOL.items()}; held = held[:2]
log(f"identities: VoxCeleb2 {len(POOL['vox'])} (+ {len(held)} held out), VoxBlink {len(POOL['vb'])} | faces {int(FT.ok.sum())}, "
    f"voices {int(VT.ok.sum())} | genders known {sum(s in GEN for s in ids)}/{len(ids)}")
assert POOL["vox"] and POOL["vb"]
'''

MODEL = r'''
# ============================================================================ CLIP-L + LoRA (FACELORA-02's), fast exact preprocessing, wiring check
from transformers import CLIPModel, CLIPImageProcessor
_proc = CLIPImageProcessor.from_pretrained(CLIPL_ID, revision=CLIPL_REV)
_cm = CLIPModel.from_pretrained(CLIPL_ID, revision=CLIPL_REV, torch_dtype=torch.float32)
VIS = nn.ModuleDict(dict(vm=_cm.vision_model, proj=_cm.visual_projection)).float()
del _cm
LAYERS, MLP_NAMES = VIS["vm"].encoder.layers, ("fc1", "fc2")
ENC_FWD = lambda x: VIS["proj"](VIS["vm"](pixel_values=x).pooler_output)          # = get_image_features
DIM = int(VIS["proj"].out_features)
for p in VIS.parameters():
    p.requires_grad = False
MEAN = torch.tensor(_proc.image_mean).view(3, 1, 1); STD = torch.tensor(_proc.image_std).view(3, 1, 1)


def PRE(im):
    return _proc(images=im.convert("RGB"), return_tensors="pt").pixel_values[0]


def PRE_FAST(im):
    """224 x 224 RGB frames: resize / centre crop are identities, so the processor reduces to rescale + normalise."""
    a = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (a - MEAN) / STD


def load_frame(i, flip=False):
    im = Image.open(io.BytesIO(frame_bytes(FT.member.values[i]))).convert("RGB")
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return PRE_FAST(im) if im.size == (224, 224) else PRE(im)


chk_rows = [int(i) for i in FT[FT.ok].groupby("src").head(16).index]
d = max(float((PRE_FAST(Image.open(io.BytesIO(frame_bytes(FT.member.values[i]))).convert("RGB"))
               - PRE(Image.open(io.BytesIO(frame_bytes(FT.member.values[i]))).convert("RGB"))).abs().max()) for i in chk_rows)
log(f"fast preprocessing vs CLIPImageProcessor on {len(chk_rows)} frames: max |diff| {d:.2e}")
assert d < 1e-4, "fast preprocessing differs from the processor"


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(x, self.B @ self.A) * self.scale


class Ckpt(nn.Module):
    def __init__(self, blk):
        super().__init__(); self.blk = blk

    def forward(self, *a, **k):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.blk, *a, use_reentrant=False, **k)
        return self.blk(*a, **k)


for i in CFG["blocks"]:
    mlp = LAYERS[i].mlp
    for n in MLP_NAMES:
        setattr(mlp, n, LoRALinear(getattr(mlp, n), CFG["rank"], CFG["alpha"]))
for i in CFG["blocks"]:
    LAYERS[i] = Ckpt(LAYERS[i])
VIS = VIS.to(DEV)
VPROJ = nn.Linear(VE.shape[1], DIM).to(DEV)
with torch.no_grad():
    VPROJ.bias.zero_()
lora_params = [p for n, p in VIS.named_parameters() if p.requires_grad]
log(f"CLIP-L LoRA: {sum(p.numel() for p in lora_params) / 1e6:.2f} M trainable (+ vproj {sum(p.numel() for p in VPROJ.parameters()) / 1e6:.2f} M)")


@torch.no_grad()
def embed_rows(rows, bs=128):
    VIS.eval(); out = []
    with cf.ThreadPoolExecutor(8) as ex:
        for s in range(0, len(rows), bs):
            X = torch.stack(list(ex.map(load_frame, rows[s:s + bs]))).to(DEV)
            with torch.autocast("cuda", enabled=DEV.type == "cuda"):
                out.append(F.normalize(ENC_FWD(X).float(), dim=1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, DIM), np.float32)


STATE, CKPT = OUT / "facelora_state.pt", OUT / "facelora_ckpt.pt"
for n in ("facelora_state.pt", "facelora_ckpt.pt", "train_log.json"):         # resume: an earlier run of this notebook
    prev = [p for p in FILES.get(n, []) if p.parent.name == OUT.name]
    if prev and not (OUT / n).exists():
        shutil.copy(prev[0], OUT / n); log("resume:", n)
REF = {"v37": D37 / "face_clipL.npy", "vox2b": D2B / "face_clipL.npy", "vb": DVB / "face_clipL.npy"}
WIRING = {}
if not STATE.exists() and not CKPT.exists():
    for src, f in REF.items():
        rows = [int(i) for i in FT[FT.ok & (FT.src == src)].index[:64]]
        ref = l2n_(np.load(f, mmap_mode="r")[FT.row.values[rows]].astype(np.float32))
        cos = (embed_rows(rows) * ref).sum(1)
        WIRING[src] = dict(rows=len(rows), mean_cos=round(float(cos.mean()), 5), min_cos=round(float(cos.min()), 5))
    log("wiring check at step 0 (adapter = frozen CLIP-L) vs the stored arrays:", WIRING)
    assert all(w["mean_cos"] >= 0.99 for w in WIRING.values()), f"the frames / encoder do not reproduce the stored CLIP-L: {WIRING}"
'''

LOOP = r'''
# ============================================================================ training (registered: 16 VoxCeleb2 + 16 VoxBlink ids per step)
fvid, vvid = FT.video.values, VT.video.values


def pick_ids(pool, n, balanced):
    if not balanced:
        return list(rng.choice(pool, min(n, len(pool)), replace=False))
    m = [s for s in pool if GEN[s] == "m"]; f = [s for s in pool if GEN[s] != "m"]
    h = n // 2
    return list(rng.choice(m, min(h, len(m)), replace=False)) + list(rng.choice(f, min(n - h, len(f)), replace=False))


_POOLG = {}


def sample_batch():
    P, K = CFG["P"], CFG["K"]
    spk = pick_ids(POOL["vox"], P // 2, CFG["sg"] > 0) + pick_ids(POOL["vb"], P - P // 2, CFG["sg"] > 0)
    fi, vi, lab, male = [], [], [], []
    for j, s in enumerate(spk):
        fr = FACE_OF[s]
        for _ in range(K):
            v = int(rng.choice(VOICE_OF[s]))
            cand = [i for i in fr if fvid[i] != vvid[v]] or list(fr)
            fi.append(int(rng.choice(cand))); vi.append(v); lab.append(j); male.append(GEN[s] == "m")
    flips = rng.rand(len(fi)) < CFG["flip"]
    with cf.ThreadPoolExecutor(8) as ex:
        X = torch.stack(list(ex.map(load_frame, fi, flips)))
    return X, torch.from_numpy(VE[vi]), torch.tensor(lab), torch.tensor(male)


def multi_pos_infonce(S, lab):
    same = (lab[:, None] == lab[None, :]).float()
    def one_(L):
        logp = L - torch.logsumexp(L, 1, keepdim=True)
        return -((logp * same).sum(1) / same.sum(1)).mean()
    return 0.5 * (one_(S) + one_(S.t()))


def sanity_eer():
    """Held-out VoxCeleb2 identities: face of one video vs voice of ANOTHER video; negatives = other held-out ids (all / same gender)."""
    from sklearn.metrics import roc_curve
    if len(held) < 2:
        return {}
    rows = [int(i) for s in held for i in FACE_OF[s][:6]]
    Fh = dict(zip(rows, embed_rows(rows)))
    VPROJ.eval()
    with torch.no_grad():
        vrows = [int(v) for s in held for v in VOICE_OF[s][:4]]
        Vh = dict(zip(vrows, F.normalize(VPROJ(torch.from_numpy(VE[vrows]).to(DEV)).float(), dim=1).cpu().numpy()))
    r2 = np.random.RandomState(1); pos, neg, negg = [], [], []
    for s in held:
        fr = [i for i in FACE_OF[s][:6]]
        for v in VOICE_OF[s][:4]:
            cand = [i for i in fr if fvid[i] != vvid[v]]
            if cand:
                pos.append(float(Fh[int(r2.choice(cand))] @ Vh[int(v)]))
            o = held[r2.randint(len(held))]
            if o != s:
                sc = float(Fh[int(r2.choice(FACE_OF[o][:6]))] @ Vh[int(v)]); neg.append(sc)
                if GEN[o] == GEN[s]:
                    negg.append(sc)
    def eer(p, n):
        if len(p) < 2 or len(n) < 2:
            return None
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]; s_ = np.r_[p, n]
        fpr, tpr, _ = roc_curve(y, s_); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return round(100 * float(fpr[i] + fnr[i]) / 2, 2)
    return {"ng": eer(pos, neg), "g": eer(pos, negg) if len(negg) > 20 else None, "pos": len(pos), "neg": len(neg)}


opt = torch.optim.AdamW([{"params": lora_params, "lr": CFG["lr_lora"]}, {"params": VPROJ.parameters(), "lr": CFG["lr_proj"]}], weight_decay=CFG["wd"])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CFG["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, CFG["steps"])))))
scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
LOG = {"cfg": CFG, "wiring": WIRING, "loss": [], "sanity": {}, "pools": {k: len(v) for k, v in POOL.items()}}
start = 1
if STATE.exists():
    st = torch.load(STATE, map_location="cpu", weights_only=False)
    VIS.load_state_dict({**VIS.state_dict(), **st["vis"]}); VPROJ.load_state_dict(st["vproj"])
    LOG = json.loads((OUT / "train_log.json").read_text()) if (OUT / "train_log.json").exists() else LOG
    start = CFG["steps"] + 1; log("adapter loaded (training done earlier):", st["cfg"]["arm"])
elif CKPT.exists():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    VIS.load_state_dict({**VIS.state_dict(), **ck["vis"]}); VPROJ.load_state_dict(ck["vproj"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    rng.set_state(ck["rng"]); torch.set_rng_state(ck["torch_rng"]); LOG = ck["log"]; start = ck["step"] + 1
    log(f"resume from checkpoint at step {ck['step']}")
if start == 1:
    LOG["sanity"]["0"] = sanity_eer(); log("sanity EER before training:", LOG["sanity"]["0"])
t0 = time.time()
BUDGET_H = float(os.environ.get("LORA_BUDGET_H", "0") or 0)          # 14/10: stop cleanly (checkpoint) after this many training hours
stopped = False
for step in range(start, CFG["steps"] + 1):
    VIS.train(); VPROJ.train()
    X, V_, lab, male = sample_batch()
    X, V_, lab, male = X.to(DEV), V_.to(DEV), lab.to(DEV), male.to(DEV)
    with torch.autocast("cuda", enabled=DEV.type == "cuda"):
        f = F.normalize(ENC_FWD(X).float(), dim=1); v = F.normalize(VPROJ(V_).float(), dim=1)
        SIM = f @ v.t() / CFG["tau"]                                  # not S: S is the data cell's dict of organiser lists
        loss = multi_pos_infonce(SIM, lab)
        if CFG["sg"] > 0:                                              # B3: same-gender negatives only (g cells)
            for sel in (male, ~male):
                if sel.sum() > 2:
                    loss = loss + CFG["sg"] * multi_pos_infonce(SIM[sel][:, sel], lab[sel])
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    LOG["loss"].append(float(loss))
    if step % 50 == 0 or step == CFG["steps"]:
        log(f"step {step}/{CFG['steps']} loss {np.mean(LOG['loss'][-50:]):.4f} [{time.time() - t0:.0f}s]"
            + (f" | GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if DEV.type == "cuda" else ""))
    if step in (1000, CFG["steps"] // 2, CFG["steps"]):
        LOG["sanity"][str(step)] = sanity_eer(); log(f"sanity EER at step {step}:", LOG["sanity"][str(step)])
    if step % CFG["ckpt_every"] == 0 and step < CFG["steps"]:
        torch.save(dict(vis={k: v.cpu() for k, v in VIS.state_dict().items() if k.endswith((".A", ".B"))}, vproj=VPROJ.state_dict(),
                        opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), rng=rng.get_state(),
                        torch_rng=torch.get_rng_state(), log=LOG, step=step), CKPT)
        log(f"checkpoint at step {step}")
    if BUDGET_H and (time.time() - t0) / 3600 > BUDGET_H and step < CFG["steps"]:
        torch.save(dict(vis={k: v.cpu() for k, v in VIS.state_dict().items() if k.endswith((".A", ".B"))}, vproj=VPROJ.state_dict(),
                        opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), rng=rng.get_state(),
                        torch_rng=torch.get_rng_state(), log=LOG, step=step), CKPT)
        log(f"time budget {BUDGET_H} h reached: checkpoint at step {step}/{CFG['steps']}; attach this output and run again to resume")
        stopped = True
        break
if start <= CFG["steps"] and not stopped:
    torch.save({"vis": {k: v.cpu() for k, v in VIS.state_dict().items() if k.endswith((".A", ".B"))}, "vproj": VPROJ.state_dict(),
                "cfg": CFG, "wiring": WIRING}, STATE)
    (OUT / "train_log.json").write_text(json.dumps(LOG, indent=1))
    log("adapter saved ->", STATE.name)
VIS.eval(); VPROJ.eval()
if stopped:                                                     # partial training: no adapted array is extracted in this run
    ENCODERS, EXT_ENCODERS = [], []
    log("partial run: the extraction cells below do nothing; the checkpoint is in the output")
'''

ENC = r'''
# ============================================================================ the adapted encoder in the FLAG_12 encoder interface (MAV-Celeb rows)
class Adapted:
    kind = "face"

    def __init__(self):
        VIS.eval()

    def prep(self, im):
        return PRE(im)

    @torch.no_grad()
    def __call__(self, x, use_amp=True):
        with torch.autocast("cuda", enabled=use_amp and DEV.type == "cuda"):
            return {"": ENC_FWD(x.to(DEV)).float().cpu().numpy()}


NEW = {ARR + "V": Adapted}


def build(name):
    return NEW[name]()


def release(keep_ib=False):
    import gc
    gc.collect()


ARRAYS_NEW = {ARR + "V": ("face", {"": ARR})}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {}
'''

EXT = r'''
# ============================================================================ the adapted CLIP-L of every external face row (stored-array row order)
for src, fk, name in (("v37", FK37, "vox_face"), ("vox2b", FK2B, "vox2b_face"), ("vb", FKVB, "vb_face")):
    f = OUT / f"{name}_{ARR}.npy"
    if f.exists() or not STATE.exists():                       # no finished adapter (time-budget stop): nothing to embed
        continue
    t0 = time.time()
    A = np.full((len(fk), DIM), np.nan, np.float16)
    sub = FT[(FT.src == src) & FT.member.isin(set(MEMBER))]
    if SMOKE:                                                      # smoke test: a few rows of each source
        sub = sub.iloc[:16]
    for c0 in range(0, len(sub), 8192):
        part = sub.iloc[c0:c0 + 8192]
        A[part.row.values] = embed_rows([int(i) for i in part.index]).astype(np.float16)
    np.save(f, A)
    log(f"{name}_{ARR}: {len(sub)}/{len(fk)} rows embedded [{time.time() - t0:.0f}s]")
TL = json.loads((OUT / "train_log.json").read_text()) if (OUT / "train_log.json").exists() else None
CHECKS["facelora3"] = {"arm": ARM, "array": ARR, "pools": {k: len(v) for k, v in POOL.items()}, "held_out": len(held), "cfg": CFG,
                       "wiring": TL["wiring"] if TL else WIRING, "sanity": TL["sanity"] if TL else None,
                       "loss_every_50": TL["loss"][::50] if TL else None}
log("FACELORA-03 sanity EER:", CHECKS["facelora3"]["sanity"])
'''


UPLOAD_CELL = r'''
# 15/10: the adapted arrays go straight to a PRIVATE dataset of the running account (kagglehub authenticates as the notebook's
# owner), so step 2 (FLAG_86) can attach them from other accounts after `kapi.py privatize`. Skipped in smoke runs, without
# FL3_UPLOAD, or when the adapter is not finished (time-budget stop). A failure here never fails the run: the output stays.
UP = os.environ.get("FL3_UPLOAD", "")
done_arrays = sorted(OUT.glob(f"*{ARR}*.npy"))
if UP and not SMOKE and STATE.exists() and len(done_arrays) >= 10:
    try:
        import kagglehub
        stage = Path("/tmp/fl3_upload"); shutil.rmtree(stage, ignore_errors=True); stage.mkdir(parents=True)
        for f in done_arrays + [STATE, OUT / "train_log.json"]:
            if f.exists():
                os.symlink(f, stage / f.name)
        kagglehub.dataset_upload(UP, str(stage), version_notes=f"FACELORA-03 {ARM}: adapted CLIP-L arrays")
        log(f"uploaded {len(done_arrays)} arrays to {UP}")
    except Exception as e:
        log(f"upload to {UP} failed ({e!r}); the arrays are in the notebook output")
else:
    log(f"no upload (FL3_UPLOAD={UP!r}, smoke={SMOKE}, adapter done={STATE.exists()}, arrays={len(done_arrays)})")
'''


INTRO_FB = """
# FLAG 2027 — NB-65f: FACELORA-04 adapter FB4: FaRL ep16 fine-tuned (LoRA) on ~21k identities against frozen SimAM voices (B3 recipe)

**Why.** FACELORA-03 B3 (a CLIP-L LoRA adapter trained this way, used inside the CC head) passed the gate (g +0.46-0.51 over the
frozen control, 4/5 folds) and reached CB 19.77. The CC head's face input has a second encoder, FaRL ep16, still frozen.
FACELORA-02 adapted FaRL on 4.5k identities only (standalone ~0). Same data, batches, loss and schedule as B3; FaRL settings of
FACELORA-02. Registered in `Experiment/FACELORA-04/README.md`.

| | |
|---|---|
| identities | VoxCeleb2 FLAG_37 + FLAG_53 (LID rules of FLAG_42 / FLAG_53) + VoxBlink-clean (LID + noise rule of `vb_rows`): ~21k; 300 VoxCeleb2 identities held out (sanity EER only) |
| batch | 32 identities = **16 VoxCeleb2 + 16 VoxBlink** x 4 (face, voice) pairs of different videos (gender-balanced: 8 male + 8 female identities per source) |
| loss | symmetric multi-positive InfoNCE, tau 0.05 + 0.5 x (InfoNCE within the male rows + within the female rows): same-gender negatives |
| optimiser | AdamW (LoRA 1e-4, voice projection 1e-3, wd 1e-2), 200 warm-up, cosine, **4,600 steps**, fp16, seed 0; checkpoint every 500 steps |
| encoder | FaRL ep16 = openai CLIP ViT-B/16 + FaRL weights (`farl16` of FLAG_40) + LoRA r16 / a32 on c_fc / c_proj of blocks 6-11 (= FACELORA-02's FaRL), gradient checkpointing |

**Wiring check:** at step 0 the adapter IS the frozen FaRL: 64 frames of each source must reproduce the stored `face_farl16.npy`
arrays (cosine >= 0.99), else the notebook stops before training.

**Outputs (`/kaggle/working/feats_facelora4_fb/`):** `face_farlB4_*.npy` / `ext_*_face_farlB4.npy` (MAV-Celeb rows, FLAG_40 order),
`vox_face_farlB4.npy` (FLAG_37 rows), `vox2b_face_farlB4.npy` (FLAG_53 rows), `vb_face_farlB4.npy` (FLAG_46 rows), adapter + train log;
uploaded to the private dataset `FL3_UPLOAD` when set.

**Inputs:** `flag2027-raw-v4-p`, MAV-Celeb v1 / v2 (`chnnhnnguyn/external-data-mavceleb`), `flag2027-ext-meta-p`,
`flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-feats-vox2b-p`, `onnhqunhh/flag2027-lora-frames-p` (frames of FLAG_45 / FLAG_63).
**GPU T4**, **Internet ON** (openai ViT-B/16 + FaRL ep16 weights), **private** (VoxBlink). **Time:** about 3 h (resumable).
"""

FARL_HEAD = r'''
# ============================================================================ FaRL ep16 + LoRA (FACELORA-02's FaRL), fast exact preprocessing, wiring check
import clip as _clip
_base, PRE = _clip.load("ViT-B/16", device="cpu", download_root=str(CACHE / "clip"))    # PRE = openai CLIP preprocess (as FLAG_40)
_sd = torch.load(fetch(FARL16_URL, CACHE / "FaRL-ep16.pth", FARL16_SIZE), map_location="cpu", weights_only=False)["state_dict"]
_vis = [k for k in _base.state_dict() if k.startswith("visual.")]
assert all(k in _sd and _sd[k].shape == _base.state_dict()[k].shape for k in _vis), "FaRL checkpoint does not fit ViT-B/16"
_base.load_state_dict(_sd, strict=False)
VIS = _base.visual.float()
del _base, _sd
LAYERS, MLP_NAMES = VIS.transformer.resblocks, ("c_fc", "c_proj")
ENC_FWD = lambda x: VIS(x)                                                           # = encode_image
DIM = int(VIS.proj.shape[1])
assert DIM == 512, DIM
for p in VIS.parameters():
    p.requires_grad = False
MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)               # openai CLIP normalisation
STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)


'''


def farl_variant(train, model, ext, upload):
    """The FACELORA-03 cells with FaRL ep16 in place of CLIP-L: config line, encoder head, wiring arrays, messages."""
    old = 'enc="clipL", rank=16, alpha=32, blocks=list(range(18, 24))'
    assert train.count(old) == 1
    train = train.replace(old, 'enc="farl16", rank=16, alpha=32, blocks=list(range(6, 12))')
    head_end = model.index("def PRE_FAST(im):")
    assert "def PRE(im):" in model[:head_end] and model.count("face_clipL.npy") == 3
    model = FARL_HEAD + model[head_end:]
    model = model.replace("face_clipL.npy", "face_farl16.npy").replace("CLIPImageProcessor", "the openai CLIP preprocess")
    model = model.replace("CLIP-L", "FaRL")
    assert "CLIPL" not in model and "clipL" not in model and "_proc" not in model, "a CLIP-L leftover in the FaRL encoder cell"
    ext = ext.replace("the adapted CLIP-L", "the adapted FaRL")
    old = 'version_notes=f"FACELORA-03 {ARM}: adapted CLIP-L arrays"'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'version_notes=f"FACELORA-04 {ARM}: adapted FaRL arrays"')
    return train, model, ext, upload


INTRO_C5 = """
# FLAG 2027 — NB-65g: FACELORA-05 adapter C5: a larger CLIP-L LoRA (MLP + attention of blocks 12-23) on ~21k identities (B3 recipe)

**Why.** FACELORA-03 B3 (LoRA r16 on fc1 / fc2 of CLIP-L blocks 18-23, ~1M parameters) passed the gate (g +0.46-0.51 over the frozen
control) and reached CB 19.77; FACELORA-04 (the same on FaRL) added nothing. C5 asks whether CLIP-L has more to give: the adapter
reaches deeper (blocks 12-23) and also adapts attention (q / k / v / out). Everything else is B3. Registered in
`Experiment/FACELORA-05/README.md`.

| | |
|---|---|
| identities | VoxCeleb2 FLAG_37 + FLAG_53 (LID rules of FLAG_42 / FLAG_53) + VoxBlink-clean (LID + noise rule of `vb_rows`): ~21k; 300 VoxCeleb2 identities held out (sanity EER only) |
| batch | 32 identities = **16 VoxCeleb2 + 16 VoxBlink** x 4 (face, voice) pairs of different videos (gender-balanced: 8 male + 8 female identities per source) |
| loss | symmetric multi-positive InfoNCE, tau 0.05 + 0.5 x (InfoNCE within the male rows + within the female rows): same-gender negatives |
| optimiser | AdamW (LoRA 1e-4, voice projection 1e-3, wd 1e-2), 200 warm-up, cosine, **4,600 steps**, fp16, seed 0; checkpoint every 500 steps |
| encoder | CLIP ViT-L/14 @ 32bd642 + LoRA r16 / a32 on **fc1 / fc2 and q / k / v / out_proj of blocks 12-23** (~3.5 M parameters), gradient checkpointing |

**Wiring check:** at step 0 the adapter IS the frozen CLIP-L: 64 frames of each source must reproduce the stored CLIP-L arrays
(cosine >= 0.99), else the notebook stops before training.

**Outputs (`/kaggle/working/feats_facelora5_c5/`):** `face_clipC5_*.npy` / `ext_*_face_clipC5.npy` (MAV-Celeb rows, FLAG_40 order),
`vox_face_clipC5.npy`, `vox2b_face_clipC5.npy`, `vb_face_clipC5.npy`, adapter + train log; uploaded to the private dataset `FL3_UPLOAD`.

**Inputs:** `flag2027-raw-v4-p`, MAV-Celeb v1 / v2 (`chnnhnnguyn/external-data-mavceleb`), `flag2027-ext-meta-p`,
`flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-feats-vox2b-p`, `onnhqunhh/flag2027-lora-frames-p`. **GPU T4**, **Internet ON**,
**private** (VoxBlink). **Time:** about 9-10 h (resumable; `LORA_BUDGET_H` stops training cleanly with a checkpoint).
"""

ATTN_LORA = '''if CFG.get("attn"):                                                       # FACELORA-05: + attention q / k / v / out
    for i in CFG["blocks"]:
        att = LAYERS[i].self_attn
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert isinstance(getattr(att, n), nn.Linear), (i, n, type(getattr(att, n)))
            setattr(att, n, LoRALinear(getattr(att, n), CFG["rank"], CFG["alpha"]))
'''


def c5_variant(train, model, upload):
    """The FACELORA-03 cells with a larger CLIP-L adapter: blocks 12-23, LoRA also on the attention projections."""
    old = 'enc="clipL", rank=16, alpha=32, blocks=list(range(18, 24))'
    assert train.count(old) == 1
    train = train.replace(old, 'enc="clipL", rank=16, alpha=32, blocks=list(range(12, 24)), attn=True')
    old = 'for i in CFG["blocks"]:\n    LAYERS[i] = Ckpt(LAYERS[i])'
    assert model.count(old) == 1
    model = model.replace(old, ATTN_LORA + old)
    old = 'version_notes=f"FACELORA-03 {ARM}: adapted CLIP-L arrays"'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'version_notes=f"FACELORA-05 {ARM}: adapted CLIP-L arrays (MLP + attention, blocks 12-23)"')
    return train, model, upload


INTRO_B3X = """
# FLAG 2027 — NB-65x: B3-EVAL step 1b: the SAVED FACELORA-03 B3 adapter -> `clipB3` of every MAV-Celeb row (no training)

**What.** The CB 19.77 system (FACELORA-03 B3) scores faces with CLIP-L adapted by the B3 LoRA. On new test files (Evaluation) the
faces must go through the SAME adapter. This notebook loads the saved adapter (`facelora_state.pt` of
`nguynchnngn1/flag2027-feats-facelora3-b-p`), skips training, and embeds every MAV-Celeb row found: v4 train, v1 / v2, and every
test list attached (dev and / or Evaluation; set `ONLY_UNDER` in the first cell when both are attached). Package: `BEST/v1977/`.

**Check:** the re-extracted arrays must equal the saved ones of the dataset (cosine >= 0.999 on every row of every array both have);
on Evaluation the new test lists have no reference, the train / v1 / v2 arrays still do.

**Inputs:** as FLAG_65 (`flag2027-raw-v4-p`, MAV-Celeb v1 / v2, `-ext-meta-p`, `-lab-data-p`, `-lab-vb-p`, `-feats-vox2b-p`,
`onnhqunhh/flag2027-lora-frames-p`) + **`nguynchnngn1/flag2027-feats-facelora3-b-p`** (saved adapter + reference arrays) + on
11/11 the organisers' Evaluation data. **GPU T4** (CPU works, slower), **Internet ON** (CLIP-L). **Output:** `feats_clipb3_x/`
(`face_clipB3_<list>.npy`, `ext_*_face_clipB3.npy`) -> attach to FLAG_82b.
"""

B3X_CHECK = r'''
# ============================================================================ B3-EVAL: re-extracted arrays vs the saved ones (no VoxCeleb2 / VoxBlink rows)
ref_dir = next((p.parent for p in FILES.get("face_clipB3_train.npy", []) if OUT.name not in p.as_posix()), None)
assert ref_dir is not None, "attach nguynchnngn1/flag2027-feats-facelora3-b-p"
chk = {}
for f in sorted(OUT.glob("face_clipB3_*.npy")) + sorted(OUT.glob("ext_*_face_clipB3.npy")):
    r = ref_dir / f.name
    if ONLY_UNDER and not (f.name.startswith("ext_") or f.name.endswith("_train.npy")):   # Evaluation lists (11/11)
        chk[f.name] = "Evaluation list: no saved reference"; continue
    if not r.exists():
        chk[f.name] = "no reference (new list)"; continue
    a, b = np.load(f).astype(np.float32), np.load(r).astype(np.float32)
    n = min(len(a), len(b)) if SMOKE else len(b)
    assert len(a) == n or SMOKE, (f.name, a.shape, b.shape)
    a, b = a[:n], b[:n]
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    cos = (l2n_(a[ok]) * l2n_(b[ok])).sum(1)
    chk[f.name] = dict(rows=int(n), finite=int(ok.sum()), mean_cos=round(float(cos.mean()), 6), min_cos=round(float(cos.min()), 6))
log("B3-EVAL re-extracted vs saved:", json.dumps(chk, indent=1))
bad = {k: v for k, v in chk.items() if isinstance(v, dict) and v["min_cos"] < 0.999}
assert not bad, f"the saved adapter does not reproduce the saved arrays: {bad}"
log("OK: attach this output (feats_clipb3_x/) to FLAG_82b; on Evaluation set E2E_SPLIT=eval there")
'''


B3X_PRECOPY = r'''
# B3-EVAL (16/10): the train / v1 / v2 arrays of the SAVED adapter are copied in (same adapter, same frames: the FLAG_12 loop then
# skips them), so only the test lists are embedded here (dev and / or Evaluation): ~5k faces instead of ~28k (CPU ~2.5 h, GPU minutes)
ref_dir = next((p.parent for p in FILES.get("face_clipB3_train.npy", []) if OUT.name not in p.as_posix()), None)
assert ref_dir is not None, "attach nguynchnngn1/flag2027-feats-facelora3-b-p"
for f in [ref_dir / "face_clipB3_train.npy"] + sorted(ref_dir.glob("ext_*_face_clipB3.npy")):
    if not (OUT / f.name).exists():
        shutil.copy(f, OUT / f.name); log("B3-EVAL: saved array copied in:", f.name)
'''

B3X_PRECOPY_ALL = r'''
# CLEANTUNE-01 Evaluation path (2/11): every file of the SAVED clean dataset except the test lists is copied in (train / v1 / v2 /
# VoxCeleb2 / VoxBlink arrays, adapter, bn42_clean.json), so this output alone replaces the dataset in the CC-head run (FLAG_112x);
# only the test lists attached (dev = R11 rehearsal, Evaluation on 11/11) are embedded here through the saved adapter
ref_dir = next((p.parent for p in FILES.get("face_clipB3_train.npy", []) if OUT.name not in p.as_posix()), None)
assert ref_dir is not None and (ref_dir / "bn42_clean.json").exists(), "attach the saved clean adapter dataset (bn42_clean.json)"
for f in sorted(ref_dir.rglob("*")):
    d = OUT / f.relative_to(ref_dir)
    if f.is_file() and not re.match(r"face_clipB3_(no_)?gender_", f.name) and not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, d); log("saved file copied in:", f.relative_to(ref_dir))
'''


INTRO_B3T = """
# FLAG 2027 — NB-65t: TTA-02 step 1: the SAVED FACELORA-03 B3 adapter with horizontal-flip TTA -> `clipB3t` of every face row

**What.** Track 3 (CB 17.70) averages, for every CLIP-family branch, the L2-normalised embeddings of the image and of its horizontal
flip. This notebook loads the saved B3 adapter (`facelora_state.pt` of `nguynchnngn1/flag2027-feats-facelora3-b-p`), skips training
and embeds every face row that way: MAV-Celeb (v4 train, v1 / v2, the test lists) and the VoxCeleb2 / VoxBlink frames.
Registered in `Experiment/TTA-02/README.md`.

**Check:** cosine(`clipB3t`, saved `clipB3`) per array (same adapter; only the flip half differs): mean >= 0.9.

**Inputs:** as FLAG_65 (`flag2027-raw-v4-p`, MAV-Celeb v1 / v2, `-ext-meta-p`, `-lab-data-p`, `-lab-vb-p`, `-feats-vox2b-p`,
`onnhqunhh/flag2027-lora-frames-p`) + `nguynchnngn1/flag2027-feats-facelora3-b-p`. **GPU T4**, **Internet ON** (CLIP-L).
**Output:** `feats_clipb3_t/` (10 arrays), uploaded to `FL3_UPLOAD` when set. **Time:** about 1.5 h.
"""

B3T_CHECK = r'''
# ============================================================================ TTA-02: clipB3t (flip-averaged) vs the saved clipB3 (same adapter)
ref_dir = next((p.parent for p in FILES.get("face_clipB3_train.npy", []) if OUT.name not in p.as_posix()), None)
assert ref_dir is not None, "attach nguynchnngn1/flag2027-feats-facelora3-b-p"
chk = {}
for f in sorted(OUT.glob("*clipB3t*.npy")):
    r = ref_dir / f.name.replace("clipB3t", "clipB3")
    if not r.exists():
        chk[f.name] = "no reference"; continue
    a, b = np.load(f).astype(np.float32), np.load(r).astype(np.float32)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    cos = (l2n_(a[ok]) * l2n_(b[ok])).sum(1)
    chk[f.name] = dict(rows=int(n), finite=int(ok.sum()), mean_cos=round(float(cos.mean()), 4), p05=round(float(np.quantile(cos, .05)), 4))
log("TTA-02 clipB3t vs saved clipB3:", json.dumps(chk, indent=1))
bad = {k: v for k, v in chk.items() if isinstance(v, dict) and v["finite"] and v["mean_cos"] < 0.9}
assert not bad, f"flip-averaged embeddings far from the saved ones: {bad}"
'''


def b3t_variant(model, enc, ext, upload):
    """b3x's saved-adapter loading + flip TTA in both embedding paths (MAV rows: Adapted; frames: embed_rows)."""
    model = b3x_variant(model)
    old = "                out.append(F.normalize(ENC_FWD(X).float(), dim=1).cpu())"
    assert model.count(old) == 1
    model = model.replace(old, "                e = F.normalize(ENC_FWD(X).float(), dim=1) + F.normalize(ENC_FWD(torch.flip(X, dims=[3])).float(), dim=1)\n"
                               "                out.append(F.normalize(e, dim=1).cpu())                     # TTA-02: image + horizontal flip")
    old = '            return {"": ENC_FWD(x.to(DEV)).float().cpu().numpy()}'
    assert enc.count(old) == 1
    enc = enc.replace(old, "            x = x.to(DEV)                                                   # TTA-02: image + horizontal flip\n"
                           "            e = F.normalize(ENC_FWD(x).float(), dim=1) + F.normalize(ENC_FWD(torch.flip(x, dims=[3])).float(), dim=1)\n"
                           '            return {"": F.normalize(e, dim=1).cpu().numpy()}')
    ext = ext.replace("the adapted CLIP-L of every external face row", "the flip-averaged B3 CLIP-L of every external face row")
    old = 'version_notes=f"FACELORA-03 {ARM}: adapted CLIP-L arrays"'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'version_notes=f"TTA-02: B3 CLIP-L with horizontal-flip TTA (clipB3t)"')
    return model, enc, ext, upload


TAP_MODEL = r'''
# ============================================================================ TAP-01: intermediate taps of the B3 CLIP-L and of the frozen FaRL ep16
# CLIP-L (HF): hidden_states[l] = output after l blocks; CLS through post_layernorm. Patch mean after 16 blocks without the 8
# largest-norm tokens (~3 %, the high-norm artifact tokens of Darcet et al.). FaRL (openai CLIP code, LND): [CLS | patch mean]
# after 5 / 9 blocks through ln_post (FaRL's own attribute recipe without the max-pool). Final outputs kept for the wiring check.
TAP_L, TAP_P, TAP_DROP, FARL_L = {"T12": 12, "T16": 16, "T20": 20}, 16, 8, {"T5": 5, "T9": 9}
NAMES = {"": ARR, "T12": "clipT12", "T16": "clipT16", "T20": "clipT20", "P16": "clipP16", "F": "farl16", "FT5": "farlT5", "FT9": "farlT9"}
try:
    import clip                                                   # openai CLIP (FaRL's code base), pip-installed in the first cell
except ModuleNotFoundError:                                       # 29/10 smoke: the pip build failed (setup.py needs pkg_resources)
    import subprocess
    _oc = Path("/tmp/openai_clip")
    if not (_oc / "clip").exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/openai/CLIP.git", str(_oc)], check=True)
        subprocess.run(["git", "-C", str(_oc), "checkout", "-q", "d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"], check=True)
        _f = _oc / "clip/clip.py"
        _f.write_text(_f.read_text().replace("from pkg_resources import packaging", "import packaging, packaging.version"))
    sys.path.insert(0, str(_oc))
    import clip
log("openai CLIP module:", clip.__file__)
FARL = FaRL16Face()                                              # T3 cell: openai ViT-B/16 + FaRL ep16 visual tower, frozen
_fh = {}
for _k, _l in FARL_L.items():
    FARL.m.transformer.resblocks[_l - 1].register_forward_hook(lambda mod, i, o, k=_k: _fh.__setitem__(k, o))


@torch.no_grad()
def all_taps(x):
    """x: CLIP-normalised 224 x 224 batch on DEV -> {NAMES key: (B, d) tensor}."""
    o = VIS["vm"](pixel_values=x, output_hidden_states=True)
    hs, ln = o.hidden_states, VIS["vm"].post_layernorm
    out = {"": VIS["proj"](o.pooler_output)}
    for k, l in TAP_L.items():
        out[k] = ln(hs[l][:, 0])
    p = hs[TAP_P][:, 1:]
    keep = p.norm(dim=-1).argsort(dim=1)[:, :p.shape[1] - TAP_DROP]
    out["P16"] = ln(torch.gather(p, 1, keep[..., None].expand(-1, -1, p.shape[-1])).mean(1))
    out["F"] = FARL.m(x)
    for k in FARL_L:
        h = _fh[k].permute(1, 0, 2)                               # LND -> NLD
        out["F" + k] = torch.cat([FARL.m.ln_post(h[:, 0]), FARL.m.ln_post(h[:, 1:].mean(1))], 1)
    return out


@torch.no_grad()
def embed_taps(rows, bs=128):
    VIS.eval(); out = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for s in range(0, len(rows), bs):
            X = torch.stack(list(ex.map(load_frame, rows[s:s + bs]))).to(DEV)
            with torch.autocast("cuda", enabled=DEV.type == "cuda"):
                o = all_taps(X)
            for k, v in o.items():
                out.setdefault(NAMES[k], []).append(v.float().cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in out.items()}


_t = {NAMES[k]: tuple(v.shape) for k, v in all_taps(load_frame(int(FT[FT.ok].index[0]))[None].to(DEV)).items()}
log("TAP-01 taps of one frame:", _t)
'''

TAP_EXT = r'''
# ============================================================================ TAP-01: every VoxCeleb2 / VoxBlink face row (stored-array row order)
for src, fk, name in (("v37", FK37, "vox_face"), ("vox2b", FK2B, "vox2b_face"), ("vb", FKVB, "vb_face")):
    if all((OUT / f"{name}_{a}.npy").exists() for a in NAMES.values()):
        continue
    t0 = time.time()
    sub = FT[(FT.src == src) & FT.member.isin(set(MEMBER))]
    if SMOKE:                                                      # smoke test: a few rows of each source
        sub = sub.iloc[:16]
    A = {}
    for c0 in range(0, len(sub), 8192):
        part = sub.iloc[c0:c0 + 8192]
        for a, X in embed_taps([int(i) for i in part.index]).items():
            if a not in A:
                A[a] = np.full((len(fk), X.shape[1]), np.nan, np.float16)
            A[a][part.row.values] = X
    for a, X in A.items():
        np.save(OUT / f"{name}_{a}.npy", X)
    log(f"{name}: {len(sub)}/{len(fk)} rows, {len(A)} arrays [{time.time() - t0:.0f}s]")
CHECKS["tap01"] = {"taps": NAMES, "shapes": _t}
'''

TAP_CHECK = r'''
# ============================================================================ TAP-01 wiring: the recomputed final arrays (clipB3, farl16) vs the stored ones
def _stored(name):
    if name in ("vox_face_farl16.npy", "vox2b_face_farl16.npy", "vb_face_farl16.npy"):
        key = {"vox": "vox2t3", "vox2b": "vox2b", "vb": "feats_vb"}[name.split("_face")[0]]
        c = [p for p in FILES.get("face_farl16.npy", []) if key in p.as_posix() and OUT.name not in p.as_posix()]
    else:
        c = [p for p in FILES.get(name, []) if OUT.name not in p.as_posix()]
    return sorted(c, key=lambda p: len(p.as_posix()))[0] if c else None


chk = {}
for f in sorted(OUT.glob("*.npy")):
    if not any(t in f.name for t in (f"_{ARR}", "farl16")) or "farlT" in f.name:
        continue
    r = _stored(f.name)
    if r is None:
        chk[f.name] = "no reference"; continue
    a, b = np.load(f).astype(np.float32), np.load(r, mmap_mode="r")
    n = min(len(a), len(b))
    assert len(a) == len(b) or SMOKE, (f.name, a.shape, b.shape)
    b = np.asarray(b[:n], np.float32); a = a[:n]
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    cos = (l2n_(a[ok]) * l2n_(b[ok])).sum(1)
    chk[f.name] = dict(rows=int(n), finite=int(ok.sum()), mean_cos=round(float(cos.mean()), 6), min_cos=round(float(cos.min()), 6))
log("TAP-01 wiring (recomputed final arrays vs stored):", json.dumps(chk, indent=1))
bad = {k: v for k, v in chk.items() if isinstance(v, dict) and v["finite"] and v["mean_cos"] < (0.999 if ARR in k else 0.995)}
assert not bad, f"the frames / encoders do not reproduce the stored arrays: {bad}"
nan = {f.name: round(float(np.isfinite(np.load(f, mmap_mode="r")[:, 0].astype(np.float32)).mean()), 4) for f in sorted(OUT.glob("*.npy"))}
log("finite share per array:", json.dumps(nan))
CHECKS["tap01_wiring"] = chk
'''

INTRO_TAP = """
# FLAG 2027 — NB-103a: TAP-01 step 1, intermediate taps of the B3 CLIP-L and of FaRL ep16 for every face row

**Why.** In every encoder we use, the final embedding is the decoder of the encoder's own task (CLIP: the last 4 blocks' heads
carry caption semantics; FaRL: layer 9 is best for attributes, layer 5 for geometry). Nine adaptations of the final embedding gave
~0 after the CC head; new information has to come from what the final projection discards. Registered in
`Experiment/TAP-01/README.md` (29/10, before any result).

| array | encoder | tap | dim |
|---|---|---|---|
| `clipT12` / `clipT16` / `clipT20` | CLIP-L + the SAVED B3 adapter | CLS after 12 / 16 / 20 blocks, `post_layernorm` | 1024 |
| `clipP16` | same | patch mean after 16 blocks without the 8 largest-norm tokens, `post_layernorm` | 1024 |
| `farlT5` / `farlT9` | FaRL ep16 (frozen) | [CLS, patch mean] after 5 / 9 blocks, `ln_post` | 1536 |
| `clipB3`, `farl16` | the final outputs | wiring check vs the stored arrays (mean cos >= 0.999 / 0.995) | 768 / 512 |

Rows: MAV-Celeb (v4 train, v1 / v2, the test lists; FLAG_12 loop) and every VoxCeleb2 / VoxBlink face row (stored-array order).
**Inputs:** as FLAG_65 + `nguynchnngn1/flag2027-feats-facelora3-b-p` (saved adapter, stored clipB3). **GPU T4**, **Internet ON**
(CLIP-L, openai ViT-B/16 + FaRL ep16), **private** (VoxBlink). **Output:** `feats_tap01_face/`, uploaded to `FL3_UPLOAD` when set.
**Time:** about 2 h (resumable).
"""


def tap_variant(model, enc, ext, upload):
    """b3x's saved-adapter loading; MAV rows and frames go through all_taps (CLIP-L taps + FaRL taps in one batch)."""
    model = b3x_variant(model)
    old = '            return {"": ENC_FWD(x.to(DEV)).float().cpu().numpy()}'
    assert enc.count(old) == 1
    enc = enc.replace(old, '            return {k: v.float().cpu().numpy() for k, v in all_taps(x.to(DEV)).items()}   # TAP-01')
    old = 'ARRAYS_NEW = {ARR + "V": ("face", {"": ARR})}'
    assert enc.count(old) == 1
    enc = enc.replace(old, 'ARRAYS_NEW = {ARR + "V": ("face", dict(NAMES))}                  # TAP-01: one array per tap')
    old = 'done_arrays = sorted(OUT.glob(f"*{ARR}*.npy"))'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'done_arrays = sorted(OUT.glob("*.npy"))                        # TAP-01: every tap')
    old = 'version_notes=f"FACELORA-03 {ARM}: adapted CLIP-L arrays"'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'version_notes="TAP-01: CLIP-L (B3) taps T12 / T16 / T20 / P16 + FaRL ep16 taps T5 / T9 of every face row"')
    return model, enc, TAP_EXT, upload


INTRO_SA = """
# FLAG 2027 — NB-65s: FACELORA-06 adapter SA: the B3 recipe with South-Asian identities in every batch

**Why.** FACELORA-03 B3 (CB 19.77) drew its 16 VoxCeleb2 + 16 VoxBlink identities per step at random; South-Asian identities are
only ~7 % of the pool, while the test persons (MAV, FLAG) are mostly South-Asian. Track 3 (CB 17.70) keeps a CLIP LoRA trained with
South-Asian batches (`clipsa`). Same data, loss, schedule and adapter as B3; only the batches change. Registered in
`Experiment/FACELORA-06/README.md`.

| | |
|---|---|
| batch | 32 identities = 16 VoxCeleb2 + 16 VoxBlink x 4 (face, voice) pairs of different videos; **half of each source from the South-Asian group**, the rest from the whole pool (no duplicate); gender-balanced within each part |
| South-Asian group | VoxCeleb2: group "SA" of `vox2b_groups.json` + `vox2_groups.json` (~400 ids); VoxBlink: `lan` in {hindi, urdu, tamil, telugu, marathi, gujarati, punjabi, malayalam, kannada, nepali, sinhala, sindhi, pashto} or `loc` in {India, Pakistan, Sri Lanka, Nepal} (~1.1k ids); LID-dropped (Bengali) speakers are never in the pools |
| loss / schedule / encoder | = B3: symmetric multi-positive InfoNCE + 0.5 x same-gender InfoNCE, 4,600 steps, LoRA r16 on fc1 / fc2 of CLIP-L blocks 18-23 |

**Outputs (`/kaggle/working/feats_facelora6_sa/`):** `clipSA` of every MAV-Celeb row and every VoxCeleb2 / VoxBlink face row; uploaded to
`FL3_UPLOAD`. **Inputs:** as FLAG_65. **GPU T4**, **Internet ON**, **private** (VoxBlink). **Time:** about 7 h (resumable).
"""

SA_CELL = r'''
# ============================================================================ FACELORA-06: South-Asian identities of the two pools (registered lists)
VOX2_GRP = __VOX2_GROUPS__
SA_LAN = {"hindi", "urdu", "tamil", "telugu", "marathi", "gujarati", "punjabi", "malayalam", "kannada", "nepali", "sinhala", "sindhi", "pashto"}
SA_LOC = {"India", "Pakistan", "Sri Lanka", "Nepal"}
_vb = VKVB.drop_duplicates("spk")
VB_SA = set(_vb.spk[_vb["lan"].astype(str).str.lower().isin(SA_LAN) | _vb["loc"].astype(str).isin(SA_LOC)].astype(str))
SA_POOL = {"vox": [s for s in POOL["vox"] if VOX2_GRP.get(s.split(":", 1)[1]) == "SA"],
           "vb": [s for s in POOL["vb"] if s.split(":", 1)[1] in VB_SA]}
for k_ in SA_POOL:
    if SMOKE and len(SA_POOL[k_]) < 2:                                # smoke pools are tiny
        SA_POOL[k_] = POOL[k_][:2]
log(f"FACELORA-06 South-Asian identities: vox {len(SA_POOL['vox'])} / {len(POOL['vox'])}, vb {len(SA_POOL['vb'])} / {len(POOL['vb'])} | "
    f"male: vox {sum(GEN[s] == 'm' for s in SA_POOL['vox'])}, vb {sum(GEN[s] == 'm' for s in SA_POOL['vb'])}")
assert SA_POOL["vox"] and SA_POOL["vb"]
'''


def sa_variant(train, loop, upload):
    """FACELORA-03 B3 cells with half of each source's identities per step drawn from the South-Asian group."""
    old = 'enc="clipL", rank=16, alpha=32, blocks=list(range(18, 24))'
    assert train.count(old) == 1
    train = train.replace(old, old + ", sa_frac=0.5")
    old = '    spk = pick_ids(POOL["vox"], P // 2, CFG["sg"] > 0) + pick_ids(POOL["vb"], P - P // 2, CFG["sg"] > 0)'
    assert loop.count(old) == 1
    loop = loop.replace(old, '    spk = pick_sa("vox", P // 2) + pick_sa("vb", P - P // 2)          # FACELORA-06: half of each source South-Asian')
    old = "_POOLG = {}\n"
    assert loop.count(old) == 1
    loop = loop.replace(old, old + '''

def pick_sa(src, n):
    """FACELORA-06: round(n * sa_frac) identities from the South-Asian group, the rest from the whole pool (no duplicate)."""
    a = pick_ids(SA_POOL[src], int(round(n * CFG["sa_frac"])), CFG["sg"] > 0)
    seen = set(a)
    return a + pick_ids([s for s in POOL[src] if s not in seen], n - len(a), CFG["sg"] > 0)
''')
    old = 'version_notes=f"FACELORA-03 {ARM}: adapted CLIP-L arrays"'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'version_notes=f"FACELORA-06 {ARM}: adapted CLIP-L arrays (South-Asian batches)"')
    return train, loop, upload


def b3x_variant(model):
    """FACELORA-03 B3 cells, extraction only: the adapter state comes from the attached dataset, training is skipped."""
    old = "    prev = [p for p in FILES.get(n, []) if p.parent.name == OUT.name]"
    assert model.count(old) == 1
    model = model.replace(old, "    prev = [p for p in FILES.get(n, []) if p.parent.name == OUT.name\n"
                               "            or (n != \"facelora_ckpt.pt\" and re.search(r\"feats[-_]facelora3[-_]b[-_]p\", p.as_posix()))]   # B3-EVAL: saved adapter")
    old = "REF = {"
    assert model.count(old) == 1
    model = model.replace(old, 'assert STATE.exists(), "B3-EVAL: attach nguynchnngn1/flag2027-feats-facelora3-b-p (facelora_state.pt)"\nREF = {')
    return model


# ---------------------------------------------------------------------------- 19/10 "mở rộng" (READMEs written before any result)
INTRO_CO = """
# FLAG 2027 — NB-65c: COTRAIN-01 adapter CO: the C5 recipe (CLIP-L LoRA, MLP + attention of blocks 12-23) towards the ADAPTED voice `sam100fM`

**Why.** C5 adapted CLIP-L towards the frozen voices [sam100p | sam100f] (CB 19.35, the best system without CSLS); VOICELORA-02
step 1 then adapted sam100f towards the frozen C5 face (held-out g EER 14.83 -> 10.83). COTRAIN-01 is the second round of the
alternation: the C5 face recipe again, now towards [sam100p | **sam100fM**]. Only the voice target changes. Registered in
`Experiment/COTRAIN-01/README.md`.

| | |
|---|---|
| identities | as C5 (LID rules of FLAG_42 / FLAG_53 / FLAG_46, VoxBlink noise rule); voice rows without `sam100fM` (clip not in the audio cache) dropped; 300 VoxCeleb2 identities held out (sanity EER only) |
| voice input of the projection | [sam100p (stored) , `sam100fM` = `vox / vox2b / vb` `voice_sam100fL.npy` of `thanhthaoph/flag2027-voicelora2-p`], each L2-normalised; its manifest is checked (`exp = VOICELORA-02`, `face = clipC5`) |
| batch, loss, optimiser | = C5: 16 VoxCeleb2 + 16 VoxBlink identities x 4 pairs of different videos, gender-balanced; InfoNCE tau 0.05 + 0.5 x same-gender; AdamW (LoRA 1e-4, projection 1e-3, wd 1e-2), 200 warm-up, cosine, **4,600 steps**, fp16, seed 0 |
| encoder | = C5: CLIP ViT-L/14 @ 32bd642 + LoRA r16 / a32 on fc1 / fc2 and q / k / v / out_proj of blocks 12-23, gradient checkpointing |

**Wiring check:** at step 0 the adapter IS the frozen CLIP-L (cosine >= 0.99 with the stored arrays). **Step-1 gate (health,
registered):** 4,600 steps done, wiring check passed, held-out g EER at the last step at least 0.5 below step 0 (logged at the end of
section 6).

**Outputs (`/kaggle/working/feats_cotrain1_co/`):** `face_clipCO_*.npy` / `ext_*_face_clipCO.npy` (MAV-Celeb rows, FLAG_40 order),
`vox_face_clipCO.npy`, `vox2b_face_clipCO.npy`, `vb_face_clipCO.npy`, adapter + train log; uploaded to the private dataset `FL3_UPLOAD`.

**Inputs:** `flag2027-raw-v4-p`, MAV-Celeb v1 / v2 (`chnnhnnguyn/external-data-mavceleb`), `flag2027-ext-meta-p`,
`flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-feats-vox2b-p`, `onnhqunhh/flag2027-lora-frames-p`,
`thanhthaoph/flag2027-voicelora2-p`. **GPU T4**, **Internet ON**, **private** (VoxBlink). **Time:** about 9-10 h (as C5; resumable,
`LORA_BUDGET_H` stops training cleanly with a checkpoint).
"""

INTRO_JT = """
# FLAG 2027 — NB-65j: JOINT-01: the C5 face adapter (CLIP-L LoRA) and the unfrozen part of `sam100f` trained JOINTLY

**Why.** C5 adapted the face towards frozen voices; VOICELORA-02 adapted the voice towards the frozen C5 face. JOINT-01 trains both
sides in one loss, so the two spaces meet in the middle instead of one side standing still. Registered in `Experiment/JOINT-01/README.md`.

| | |
|---|---|
| face | = C5: CLIP ViT-L/14 @ 32bd642 + LoRA r16 / a32 on fc1 / fc2 and q / k / v / out_proj of blocks 12-23; lr 1e-4 |
| voice | WeSpeaker SimAM-ResNet100-ft (`sam100fA`), layer3.16-23 + layer4 + pool + bottleneck trainable (as 67b); lr 2e-5 on the same warm-up / cosine schedule; BatchNorm keeps its pretrained statistics (eval mode); voice gradient clipped at norm 5; on the 2nd GPU (T4 x 2), fbank on the GPU |
| voice input of the projection | [sam100p (stored, L2) , sam100f computed live on the clip (L2)]; clip = the centre 3 s in the Bengali-filtered audio cache (`duongphihoang/flag2027-audio-cache-p`), at most 4 clips per identity; voice rows without a cached clip dropped |
| batch, loss, optimiser | = C5: 16 VoxCeleb2 + 16 VoxBlink identities x 4 pairs of different videos, gender-balanced; InfoNCE tau 0.05 + 0.5 x same-gender; AdamW (wd 1e-2), 200 warm-up, cosine, **4,600 steps**, fp16, seed 0; 300 VoxCeleb2 identities held out |

**Wiring checks:** (1) at step 0 the face adapter IS the frozen CLIP-L (cosine >= 0.99 with the stored arrays); (2) the voice net at
step 0 on a cached clip = `SimAMVoice` on the same clip (cosine > 0.999). **Step-1 gate (health, registered):** 4,600 steps done (two
sessions joined by the checkpoint allowed), both wiring checks passed, held-out g EER at the last step at least 0.5 below step 0.

**Outputs (`/kaggle/working/feats_joint1_jt/`):** `face_clipJT_*.npy` / `ext_*_face_clipJT.npy` (MAV-Celeb rows), `vox_face_clipJT.npy`,
`vox2b_face_clipJT.npy`, `vb_face_clipJT.npy`, face adapter + train log, `voicelora_state.pt` (FLAG_67z-compatible: `cfg.exp = JOINT-01`,
`enc = sam100fA`), `vox / vox2b / vb` `voice_sam100fJ.npy` (every cached clip); uploaded to the private dataset `FL3_UPLOAD`.

**Inputs:** `flag2027-raw-v4-p`, MAV-Celeb v1 / v2 (`chnnhnnguyn/external-data-mavceleb`), `flag2027-ext-meta-p`,
`flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-feats-vox2b-p`, `onnhqunhh/flag2027-lora-frames-p`,
`duongphihoang/flag2027-audio-cache-p`. **GPU T4 x 2**, **Internet ON** (WeSpeaker zip), **private** (VoxBlink). **Time:** about
13-15 h in two sessions (`LORA_BUDGET_H` 8.5 in session 1; attach session 1's output to resume).
"""

CO_VOICES = '''def voices(d, sub):
    """COTRAIN-01: [sam100p (stored) | sam100fM = VOICELORA-02's adapted sam100f (voice_sam100fL.npy of flag2027-voicelora2-p)]."""
    return np.concatenate([l2n_(np.load(d / "voice_sam100p.npy").astype(np.float32)),
                           l2n_(np.load(VL2 / sub / "voice_sam100fL.npy").astype(np.float32))], 1)


_vl2 = sorted({p.parent.parent for p in FILES.get("voice_sam100fL.npy", []) if p.parent.name in ("vox", "vox2b", "vb")
               and (p.parent.parent / "voicelora_manifest.json").exists()}, key=lambda p: len(p.as_posix()))
assert _vl2, "attach thanhthaoph/flag2027-voicelora2-p (VOICELORA-02 step 1: vox|vox2b|vb/voice_sam100fL.npy + voicelora_manifest.json)"
VL2 = _vl2[0]
_man = json.loads((VL2 / "voicelora_manifest.json").read_text())
assert _man["cfg"]["exp"] == "VOICELORA-02" and _man["cfg"]["face"] == "clipC5", ("wrong voicelora dataset", VL2, _man["cfg"])
log(f"COTRAIN-01 voice target sam100fM: {VL2} (exp {_man['cfg']['exp']}, face {_man['cfg']['face']}, sanity {_man.get('sanity')})")
'''

CO_NAN = '''_nan = ~np.isfinite(VE).all(1)                                  # COTRAIN-01: voice rows without sam100fM (clip not in the audio cache)
log(f"COTRAIN-01: {int(_nan.sum())}/{len(VE)} voice rows without sam100fM dropped; ok voice rows {int((VT.ok & ~_nan).sum())} "
    f"(was {int(VT.ok.sum())})")
VT["ok"] &= ~_nan
'''

HEALTH = '''_sn = LOG.get("sanity", {})                                     # registered step-1 health gate (COTRAIN-01 / JOINT-01)
if (_sn.get("0") or {}).get("g") is not None and (_sn.get(str(CFG["steps"])) or {}).get("g") is not None:
    _g0, _g1 = _sn["0"]["g"], _sn[str(CFG["steps"])]["g"]
    log(f"step-1 health gate (registered): held-out g EER {_g0:.2f} -> {_g1:.2f} ({_g1 - _g0:+.2f}) -> "
        f"{'PASSES' if _g1 <= _g0 - 0.5 else 'fails'} (needs <= step 0 - 0.5)")
'''


def co_variant(train, loop, upload):
    """COTRAIN-01: the C5 cells (c5_variant first) with the voice target [sam100p | sam100fM]."""
    old = ('def voices(d):\n    return np.concatenate([l2n_(np.load(d / f"voice_{n}.npy").astype(np.float32)) '
           'for n in ("sam100p", "sam100f")], 1)\n')
    assert train.count(old) == 1
    train = train.replace(old, CO_VOICES)
    for src, sub in (("D37", "vox"), ("D2B", "vox2b"), ("DVB", "vb")):
        old = f"voices({src})"
        assert train.count(old) == 1, old
        train = train.replace(old, f'voices({src}, "{sub}")')
    old = 'fv = FT[FT.ok].groupby("id").indices; vv = VT[VT.ok].groupby("id").indices'
    assert train.count(old) == 1
    train = train.replace(old, CO_NAN + old)
    old = "VIS.eval(); VPROJ.eval()\nif stopped:"
    assert loop.count(old) == 1
    loop = loop.replace(old, "VIS.eval(); VPROJ.eval()\n" + HEALTH + "if stopped:")
    old = 'version_notes=f"FACELORA-05 {ARM}: adapted CLIP-L arrays (MLP + attention, blocks 12-23)"'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'version_notes=f"COTRAIN-01 {ARM}: CLIP-L (C5 recipe) adapted towards sam100fM"')
    return train, loop, upload


JT_INROOTS = '''def _inroots():
    """Input roots except the MAV media / raw datasets (an rglob over them stalls for 25+ min on Kaggle: smoke 1, 17/10)."""
    base = Path("/kaggle/input")
    if not base.exists():
        return []
    out = []
    for p in base.iterdir():
        if p.name in ("datasets", "notebooks", "kernels", "models"):
            out += [q for o in p.iterdir() if o.is_dir() for q in o.iterdir() if q.is_dir()]
        elif p.is_dir():
            out.append(p)
    return [p for p in out if not any(k in p.name for k in ("mavceleb", "raw-v4", "ext-meta"))]


'''

JT_PIP = '''    !pip -q install kaldiio silero-vad 2>&1 | tail -1
    !pip -q install --no-deps "git+https://github.com/wenet-e2e/wespeaker.git@9fecd6cb4f47475d01761d87c826298dff4ef18c" 2>&1 | tail -1
'''

JT_DATA = r'''# -- JOINT-01: the voice input of VPROJ = [stored sam100p | sam100f computed live on the clip]; clip = the centre 3 s of the
#    Bengali-filtered audio cache (duongphihoang/flag2027-audio-cache-p), at most vclip_max clips per identity; voice rows without one dropped
SR = 16000
ACROOT = one("tar0000.npz", "/vox/").parent.parent                # the cache root (holds vox/ and vb/), whatever the mount name
JCH = [(p, "vox") for p in sorted((ACROOT / "vox").glob("tar*.npz"))] + [(p, "vb") for p in sorted((ACROOT / "vb").glob("*.npz"))]
if os.environ.get("JT_MAX_CHUNKS"):
    JCH = JCH[:int(os.environ["JT_MAX_CHUNKS"])] + JCH[-int(os.environ["JT_MAX_CHUNKS"]):]
VROW = {}                                                          # (cache kind, spk, video, utt) -> global VT rows
for _off, _vk, _kind in ((0, VK37, "vox"), (len(VK37), VK2B, "vox"), (len(VK37) + len(VK2B), VKVB, "vb")):
    for _r, _k in enumerate(zip(_vk.spk, _vk.video, _vk.utt)):
        VROW.setdefault((_kind,) + tuple(map(str, _k)), []).append(_off + _r)
_vok, _fids = VT.ok.values.copy(), set(FT.id[FT.ok])
CLIPW, _nper = {}, {}                                              # VT row -> int16 centre crop (one array shared by a key's rows)
for _p, _kind in JCH:
    _Z = np.load(_p)
    for _s, _v, _u, _w, _n in zip(_Z["spk"], _Z["video"], _Z["utt"], _Z["wav"], _Z["n"]):
        if int(_n) < SR:                                           # at least 1 s of real audio (as 67b)
            continue
        _ident = ("vb:" if _kind == "vb" else "vox2:") + str(_s)
        _rows = [r for r in VROW.get((_kind, str(_s), str(_v), str(_u)), []) if _vok[r] and r not in CLIPW]
        if not _rows or _ident not in _fids or _nper.get(_ident, 0) >= CFG["vclip_max"]:
            continue
        _a = np.array(_w, np.int16)
        for r in _rows:
            CLIPW[r] = _a
        _nper[_ident] = _nper.get(_ident, 0) + 1
_has = np.zeros(len(VT), bool); _has[list(CLIPW)] = True
log(f"JOINT-01 audio cache {ACROOT}: {len(JCH)} chunks -> {len(_nper)} identities, {sum(_nper.values())} clips "
    f"({sum(a.nbytes for a in {id(a): a for a in CLIPW.values()}.values()) / 2 ** 30:.1f} GB), {len(CLIPW)} voice rows; "
    f"ok voice rows {int((VT.ok & _has).sum())} (was {int(VT.ok.sum())})")
VT["ok"] &= _has
'''

JT_VOICE = r'''
# ============================================================================ JOINT-01: the voice net (sam100f, partly unfrozen as 67b) on the 2nd GPU
DEV2 = torch.device("cuda:1") if torch.cuda.device_count() > 1 else DEV
SV = SimAMVoice(CFG["venc"])
SV.m.set_device(str(DEV2))
NET = SV.m.model.to(DEV2)
NET.eval()                                                         # BN keeps its pretrained running stats (as 67b): never .train()
_vnames = [n for n, _ in NET.named_parameters()]
_vkeys = [k for k in CFG["unfreeze"].split(",") if k]
train_names = [n for n in _vnames if any(n.startswith(k) or f".{k}." in f".{n}" for k in _vkeys)]
_tn = set(train_names)
for n, p_ in NET.named_parameters():
    p_.requires_grad = n in _tn
assert train_names, "unfreeze matched nothing"
_miss = [k for k in _vkeys if not any(n.startswith(k) or f".{k}." in f".{n}" for n in train_names)]
assert not _miss, f"unfreeze keys that match no tensor: {_miss}"
enc_params = [p_ for n, p_ in NET.named_parameters() if n in _tn]
log(f"voice net {CFG['venc']} on {DEV2}: unfrozen {len(train_names)}/{len(_vnames)} tensors, "
    f"{sum(p_.numel() for p_ in enc_params) / 1e6:.2f} M params; first {train_names[:2]} last {train_names[-2:]}")
assert VE.shape[1] == 512, VE.shape
VH = VE.shape[1] // 2                                              # VE[:, :VH] = stored sam100p; the live sam100f replaces VE[:, VH:]
_FB_GPU = [DEV2.type == "cuda"]


def feats(W):
    """int16-scale crops (n, L) -> WeSpeaker fbank + CMN (SimAMVoice's front end) on the voice GPU (registered); CPU fallback logged once."""
    with torch.autocast("cuda", enabled=False):
        return _feats(W)


def _feats(W):
    if _FB_GPU[0]:
        try:
            return torch.stack([SV.m.compute_features(W[i:i + 1].to(DEV2), sample_rate=SR, cmn=True)[0] for i in range(len(W))])
        except Exception as e:
            _FB_GPU[0] = False
            log(f"!! fbank on {DEV2} failed ({e!r}): CPU fbank from here on (a deviation from the README: reported)")
    return torch.stack([SV.m.compute_features(W[i:i + 1].cpu(), sample_rate=SR, cmn=True)[0] for i in range(len(W))]).to(DEV2)


def _vfwd(x):
    out = NET(x)
    out = out[-1] if isinstance(out, tuple) else out
    return F.normalize(out.float(), dim=1)


def live_voice(rows, bs=64, amp=True):
    """sam100f (L2) of the cached clips of these VT rows by the partly trainable net, on DEV2; gradient checkpointing per chunk."""
    NET.eval()
    W = torch.from_numpy(np.stack([CLIPW[int(r)] for r in rows]).astype(np.float32))
    if _FB_GPU[0]:
        W = W.to(DEV2)                                             # one copy per call: the face net runs meanwhile
    out = []
    for s in range(0, len(W), bs):
        x = feats(W[s:s + bs])
        with torch.autocast("cuda", enabled=amp and DEV2.type == "cuda"):
            out.append(checkpoint(_vfwd, x, use_reentrant=False) if torch.is_grad_enabled() else _vfwd(x))
    return torch.cat(out)


def VIN(rows, live=None):
    """VPROJ input of these VT rows on DEV: [stored sam100p | live sam100f] (duplicate rows computed once)."""
    rows = np.asarray(rows, np.int64)
    if live is None:
        u, inv = np.unique(rows, return_inverse=True)
        return VIN(u, live_voice(u))[torch.from_numpy(inv).to(DEV)]
    return torch.cat([torch.from_numpy(VE[rows, :VH]).to(DEV), live.to(DEV).float()], 1)


def vnet_state():
    return {n: p_.detach().cpu() for n, p_ in NET.named_parameters() if n in _tn}


def load_vnet(sd):
    par = dict(NET.named_parameters())
    assert set(sd) == _tn, "voice state does not match the unfrozen tensors"
    with torch.no_grad():
        for n, t in sd.items():
            par[n].copy_(t.to(par[n].device, par[n].dtype))
    log(f"voice net: {len(sd)} trained tensors loaded")


# wiring (registered): the voice net at step 0 on a cached clip == SimAMVoice on the same clip (fp32; GPU fbank vs SimAMVoice's CPU fbank)
_r0 = next(iter(CLIPW))
_w0 = CLIPW[_r0]
with torch.no_grad():
    _a = _vfwd(feats(torch.from_numpy(_w0.astype(np.float32))[None])).cpu().numpy()[0]
    _a16 = live_voice([_r0]).float().cpu().numpy()[0]
_b = np.asarray(SV(_w0.astype(np.float32) / 32768.0)[""], np.float32).ravel()
_b = _b / np.linalg.norm(_b)
WIRING_V = {"cos_fp32": round(float(_a @ _b), 6), "cos_amp": round(float(_a16 @ _b), 6), "fbank_gpu": _FB_GPU[0]}
log("voice wiring at step 0 (training path vs SimAMVoice):", WIRING_V)
assert WIRING_V["cos_fp32"] > 0.999, WIRING_V
'''

JT_EXTRACT = r'''
# ============================================================================ JOINT-01: the trained voice net -> voicelora_state.pt + sam100fJ of every cached clip
if STATE.exists():                                                 # no finished training (time-budget stop): nothing to save / embed
    VS = OUT / "voicelora_state.pt"
    if not VS.exists():                                            # FLAG_67z-compatible (build_vlx_notebook.py j checks cfg.exp)
        torch.save({"net": vnet_state(), "cfg": dict(CFG, exp="JOINT-01", enc=CFG["venc"], face="clipJT", outn="sam100fJ",
                                                     fbank_gpu=_FB_GPU[0]), "unfrozen": train_names}, VS)
        log("voice net saved ->", VS.name)
    VMETA = {"vox": VK37, "vox2b": VK2B, "vb": VKVB}             # = 67b's SRC voice_meta row order
    ROWJ = {s: {tuple(map(str, k)): r for r, k in enumerate(zip(d.spk, d.video, d.utt))} for s, d in VMETA.items()}
    AJ = {s: np.full((len(d), 256), np.nan, np.float16) for s, d in VMETA.items()}
    lim, n_done, t0 = (64 if SMOKE else 0), 0, time.time()
    for p, kind in JCH:
        Z = np.load(p)
        keys, W = [tuple(map(str, k)) for k in zip(Z["spk"], Z["video"], Z["utt"])], Z["wav"]
        for c0 in range(0, len(keys), 64):
            with torch.no_grad():                                  # fp32, as 67b's extraction and SimAMVoice
                E = _vfwd(feats(torch.from_numpy(np.asarray(W[c0:c0 + 64], np.float32)))).cpu().numpy().astype(np.float16)
            for k, e in zip(keys[c0:c0 + 64], E):
                for s in (("vb",) if kind == "vb" else ("vox", "vox2b")):
                    r = ROWJ[s].get(k)
                    if r is not None:
                        AJ[s][r] = e
            n_done += len(E)
            if lim and n_done >= lim:
                break
        if lim and n_done >= lim:
            break
    for s, X in AJ.items():
        (OUT / s).mkdir(exist_ok=True); np.save(OUT / s / "voice_sam100fJ.npy", X)
    CHECKS["joint1_voice"] = {"clips": n_done, "rows": {s: int(np.isfinite(X.astype(np.float32)).all(1).sum()) for s, X in AJ.items()},
                              "wiring_voice": WIRING_V, "fbank_gpu": _FB_GPU[0]}
    log(f"sam100fJ: {n_done} clips [{time.time() - t0:.0f}s]:", CHECKS["joint1_voice"])
else:
    log("no finished training: voice state / sam100fJ not written in this run")
'''


def jt_variant(pip, cell, train, loop, upload):
    """JOINT-01: the C5 cells (c5_variant first) + the unfrozen part of sam100f trained in the same loss (live voice input)."""
    import build_vfeat_notebook as VF                              # T3 cell with the ModelScope retry patch (+ attached-zip lookup)
    old = '    !pip -q install --no-deps "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1" 2>&1 | tail -1\n'
    assert pip.count(old) == 1
    pip = pip.replace(old, old + JT_PIP)
    rg = 'Path("/kaggle/input").rglob(zip_name)'
    assert VF.T3CELL.count(rg) == 1
    cell = JT_INROOTS + VF.T3CELL.replace(rg, "(q for r in _inroots() for q in r.rglob(zip_name))")
    old = "held_out=300, seed=0, flip=0.5, ckpt_every=500)"
    assert train.count(old) == 1
    train = train.replace(old, 'held_out=300, seed=0, flip=0.5, ckpt_every=500,\n           lr_enc=2e-5, venc="sam100fA", '
                               'unfreeze=",".join(f"layer3.{i}" for i in range(16, 24)) + ",layer4,pool,bottleneck", vclip_max=4, '
                               'vgrad_clip=5.0)')
    old = 'fv = FT[FT.ok].groupby("id").indices; vv = VT[VT.ok].groupby("id").indices'
    assert train.count(old) == 1
    train = train.replace(old, JT_DATA + old)
    rep = [("    return X, torch.from_numpy(VE[vi]), torch.tensor(lab), torch.tensor(male)",
            "    return X, torch.tensor(vi), torch.tensor(lab), torch.tensor(male)          # JOINT-01: VT rows; the voice is computed in the step"),
           ("        Vh = dict(zip(vrows, F.normalize(VPROJ(torch.from_numpy(VE[vrows]).to(DEV)).float(), dim=1).cpu().numpy()))",
            "        Vh = dict(zip(vrows, F.normalize(VPROJ(VIN(vrows)).float(), dim=1).cpu().numpy()))"),
           ('opt = torch.optim.AdamW([{"params": lora_params, "lr": CFG["lr_lora"]}, {"params": VPROJ.parameters(), "lr": CFG["lr_proj"]}], weight_decay=CFG["wd"])',
            'opt = torch.optim.AdamW([{"params": lora_params, "lr": CFG["lr_lora"]}, {"params": VPROJ.parameters(), "lr": CFG["lr_proj"]},\n'
            '                         {"params": enc_params, "lr": CFG["lr_enc"]}], weight_decay=CFG["wd"])'),
           ('LOG = {"cfg": CFG, "wiring": WIRING, "loss": [], "sanity": {}, "pools": {k: len(v) for k, v in POOL.items()}}',
            'LOG = {"cfg": CFG, "wiring": WIRING, "wiring_voice": WIRING_V, "loss": [], "sanity": {}, "pools": {k: len(v) for k, v in POOL.items()}}'),
           ('    VIS.load_state_dict({**VIS.state_dict(), **st["vis"]}); VPROJ.load_state_dict(st["vproj"])',
            '    VIS.load_state_dict({**VIS.state_dict(), **st["vis"]}); VPROJ.load_state_dict(st["vproj"]); load_vnet(st["vnet"])'),
           ('    VIS.load_state_dict({**VIS.state_dict(), **ck["vis"]}); VPROJ.load_state_dict(ck["vproj"])',
            '    VIS.load_state_dict({**VIS.state_dict(), **ck["vis"]}); VPROJ.load_state_dict(ck["vproj"]); load_vnet(ck["vnet"])'),
           ("        f = F.normalize(ENC_FWD(X).float(), dim=1); v = F.normalize(VPROJ(V_).float(), dim=1)",
            "        _u, _inv = np.unique(V_.cpu().numpy(), return_inverse=True)\n"
            "        _live = live_voice(_u)                                     # JOINT-01: the voice net first (2nd GPU, runs while the face net runs)\n"
            "        f = F.normalize(ENC_FWD(X).float(), dim=1)\n"
            "        v = F.normalize(VPROJ(VIN(_u, _live)[torch.from_numpy(_inv).to(DEV)]).float(), dim=1)"),
           ("    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()",
            "    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()\n"
            "    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(enc_params, CFG[\"vgrad_clip\"])   # JOINT-01: voice gradient clip (as 67b)\n"
            "    scaler.step(opt); scaler.update(); sched.step()"),
           ('"vproj": VPROJ.state_dict(),', '"vproj": VPROJ.state_dict(), "vnet": vnet_state(),'),
           ("VIS.eval(); VPROJ.eval()\nif stopped:", "VIS.eval(); VPROJ.eval()\n" + HEALTH + "if stopped:")]
    for a, b in rep:
        assert loop.count(a) == 1, a[:80]
        loop = loop.replace(a, b)
    old = "vproj=VPROJ.state_dict(),"
    assert loop.count(old) == 2
    loop = loop.replace(old, "vproj=VPROJ.state_dict(), vnet=vnet_state(),")
    old = 'version_notes=f"FACELORA-05 {ARM}: adapted CLIP-L arrays (MLP + attention, blocks 12-23)"'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'version_notes=f"JOINT-01 {ARM}: CLIP-L (C5 recipe) + sam100f trained jointly; clipJT + voicelora_state.pt + sam100fJ"')
    old = "        for f in done_arrays + [STATE, OUT / \"train_log.json\"]:\n            if f.exists():\n                os.symlink(f, stage / f.name)\n"
    assert upload.count(old) == 1
    upload = upload.replace(old, old + "        for f in [OUT / \"voicelora_state.pt\", *OUT.glob(\"*/voice_sam100fJ.npy\")]:   # JOINT-01: the voice side, sub-folders kept\n"
                            "            if f.exists():\n"
                            "                (stage / f.relative_to(OUT)).parent.mkdir(parents=True, exist_ok=True); os.symlink(f, stage / f.relative_to(OUT))\n")
    return pip, cell, train, loop, upload


# ---------------------------------------------------------------------------- PEFACE-01 step 0 (9/11, README written before any result)
INTRO_PE = """
# FLAG 2027 — NB-129a: PEFACE-01 step 0: FROZEN PE-Core-L14-336 (Meta Perception Encoder) for every face row

**Why (Experiment/PEFACE-01/README.md, registered 9/11 before any result).** The teammate report EXP075 (CB 17.73) uses PE-Core
LoRA adapters (pel1-2) inside EVAL073b, and the RF line found a PE-Core member that passed its OOF gate. PE-Core was never tried in
our pipeline (FLAG_129 reserved on 1/11). Step 0 embeds every face row with the FROZEN encoder; step 0b (FLAG_129b) probes it
(GAPS-01 C) against CLIP-L / B3s / C5s. Nothing is trained here, so no audio of any kind is learnt from.

| | |
|---|---|
| encoder | PE-Core-L14-336, frozen, CLIP image embedding, L2; open_clip `hf-hub:timm/PE-Core-L-14-336` (else tag `meta`), else the timm vision tower; the loader used is logged |
| preprocessing | the encoder's own transform: resize 336 (bicubic), centre crop 336, normalise 0.5 / 0.5 (the 224 x 224 frames are upsampled) |
| rows | MAV-Celeb + organiser lists (FLAG_12 loop, FLAG_40 order), VoxCeleb2 FLAG_37 / FLAG_53, VoxBlink FLAG_46 (stored-array order) |
| filters | the LIDSTRICT-01 pools of B3s / C5s are built (marker `bn42_clean.json` with `lid_strict`) only so that the CA head accepts the arrays under LID_STRICT=1 |

**Health checks (the notebook stops on failure):** batched fp16 vs one-by-one fp32 cosine >= 0.99; zero-shot gender ("a photo of a
man / woman") on 192 labelled VoxCeleb2 / VoxBlink identities >= 0.80 (open_clip loaders only).

**Outputs (`/kaggle/working/feats_peface01_pe0/`):** `face_peL_*.npy` / `ext_*_face_peL.npy`, `vox_face_peL.npy`, `vox2b_face_peL.npy`,
`vb_face_peL.npy`, `bn42_clean.json`, `train_log.json` (health checks, loader); uploaded to the private dataset `FL3_UPLOAD`.

**Inputs (as FLAG_153c):** `flag2027-raw-v4-p`, MAV-Celeb v1 / v2 (`chnnhnnguyn/external-data-mavceleb`), `flag2027-ext-meta-p`,
`flag2027-lab-data-p`, `flag2027-lab-vb-p`, `flag2027-feats-vox2b-p`, `onnhqunhh/flag2027-lora-frames-p`, `ngnnguynchn/flag2027-lid-strict-p`.
**GPU T4**, **Internet ON** (PE-Core weights), **private** (VoxBlink). **Time:** about 2-3 h.
"""

PE_PIP = r'''# PEFACE-01: open_clip + timm recent enough for PE-Core (Perception Encoder); the versions are logged in section 5
if not LOCAL:
    !pip -q install -U "open_clip_torch>=2.32" "timm>=1.0.16" 2>&1 | tail -2
'''

PE_HEAD = r'''
# ============================================================================ PEFACE-01: PE-Core-L14-336, FROZEN (no LoRA: CFG blocks = [])
import open_clip, timm
_pe = _pre_pe = _tok = PE_SRC = None
for _kind, _name in (("open_clip", "hf-hub:timm/PE-Core-L-14-336"), ("open_clip_tag", "PE-Core-L-14-336"),
                     ("timm", "vit_pe_core_large_patch14_336.fb")):
    try:
        if _kind == "open_clip":
            _pe, _pre_pe = open_clip.create_model_from_pretrained(_name, cache_dir=str(CACHE / "pe"))
            _tok = open_clip.get_tokenizer(_name)
        elif _kind == "open_clip_tag":
            _pe, _, _pre_pe = open_clip.create_model_and_transforms(_name, pretrained="meta", cache_dir=str(CACHE / "pe"))
            _tok = open_clip.get_tokenizer(_name)
        else:                                                                      # vision tower only: no zero-shot check
            _pe = timm.create_model(_name, pretrained=True, num_classes=0)
            _pre_pe = timm.data.create_transform(**timm.data.resolve_data_config({}, model=_pe))
        PE_SRC = f"{_kind}:{_name}"
        break
    except Exception as e:
        log(f"PE-Core loader {_kind}:{_name} failed: {e!r}"[:500])
        _pe = _pre_pe = _tok = None
assert _pe is not None, "no PE-Core loader worked (open_clip / timm too old, or Internet off)"
_pe = _pe.float().eval()
for p in _pe.parameters():
    p.requires_grad = False
_CLIP_API = hasattr(_pe, "encode_image")
VIS = _pe.visual if _CLIP_API else _pe
ENC_FWD = (lambda x: _pe.encode_image(x)) if _CLIP_API else (lambda x: _pe(x))
LAYERS, MLP_NAMES = [], ("fc1", "fc2")                                             # no LoRA in step 0
log("PE-Core:", PE_SRC, "| open_clip", open_clip.__version__, "| timm", timm.__version__, "| transform:", _pre_pe)


def PRE(im):
    return _pre_pe(im.convert("RGB"))


_probe = Image.new("RGB", (224, 224), (128, 96, 64))
assert tuple(PRE(_probe).shape) == (3, 336, 336), PRE(_probe).shape
with torch.no_grad():
    DIM = int(ENC_FWD(PRE(_probe)[None]).shape[1])
MEAN = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1); STD = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
log(f"PE-Core embedding dim {DIM}")


'''

PE_PRE_FAST = '''def PRE_FAST(im):
    """PEFACE-01: PE-Core needs 336 x 336, so the 224 x 224 frames take the encoder's full transform (no shortcut)."""
    return PRE(im)
'''

PE_WIRING = r'''WIRING = {}
if not STATE.exists():                                    # PEFACE-01 health checks (README §3): no stored PE arrays to compare with
    rows = [int(i) for i in FT[FT.ok].groupby("src").head(8).index]
    A_ = embed_rows(rows)
    with torch.no_grad():
        B_ = np.concatenate([F.normalize(ENC_FWD(load_frame(i)[None].to(DEV)).float(), dim=1).cpu().numpy() for i in rows])
    c_ = (A_ * B_).sum(1)
    WIRING["batch_fp16_vs_single_fp32"] = dict(rows=len(rows), mean_cos=round(float(c_.mean()), 5), min_cos=round(float(c_.min()), 5))
    if _tok is not None:
        nz = 16 if SMOKE else 96
        okf = FT[FT.ok].groupby("id").head(1)                                      # one face per identity
        gz = np.array([GEN.get(s) for s in okf.id])
        rm = [int(i) for i in okf.index[gz == "m"][:nz]]; rf = [int(i) for i in okf.index[gz == "f"][:nz]]
        with torch.no_grad():
            T_ = F.normalize(_pe.encode_text(_tok(["a photo of a man", "a photo of a woman"])).float(), dim=1).cpu().numpy()
        E_ = embed_rows(rm + rf)
        ism = (E_ @ T_.T).argmax(1) == 0
        acc = float(np.r_[ism[:len(rm)], ~ism[len(rm):]].mean())
        WIRING["zero_shot_gender"] = dict(male=len(rm), female=len(rf), acc=round(acc, 4))
    log("PEFACE-01 health checks (frozen PE-Core):", WIRING)
    assert WIRING["batch_fp16_vs_single_fp32"]["min_cos"] >= 0.99, f"batched / single embeddings differ: {WIRING}"
    assert WIRING.get("zero_shot_gender", {}).get("acc", 1.0) >= 0.80, f"zero-shot gender too low (wrong weights?): {WIRING}"
'''

LOOP_PE = r'''
# ============================================================================ PEFACE-01 step 0: nothing to train (the frozen PE-Core is the encoder)
# STATE only marks "encoder ready" so that sections 8-9 embed every row; it holds no weights (frozen; the loader is in PE_SRC).
LOG = {"cfg": CFG, "wiring": WIRING, "loss": [], "sanity": {}, "pools": {k: len(v) for k, v in POOL.items()}, "pe_src": PE_SRC,
       "frozen": True}
stopped = False
if not STATE.exists():
    torch.save({"vis": {}, "vproj": {}, "cfg": CFG, "wiring": WIRING, "pe_src": PE_SRC, "frozen": True}, STATE)
    (OUT / "train_log.json").write_text(json.dumps(LOG, indent=1))
    log("PEFACE-01: frozen encoder, no training ->", STATE.name)
VIS.eval()
'''


def pe_variant(pip, train, model, loop, ext, upload):
    """PEFACE-01 step 0: the FACELORA-03 cells with a FROZEN PE-Core-L14-336: no LoRA, no training, health checks in place of the
    CLIP-L wiring check, STATE written as an 'encoder ready' marker so that sections 8-9 embed every row."""
    old = "import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers"
    assert pip.count(old) == 1
    pip = pip.replace(old, PE_PIP + old)
    old = 'enc="clipL", rank=16, alpha=32, blocks=list(range(18, 24))'
    assert train.count(old) == 1
    train = train.replace(old, 'enc="pecore", rank=16, alpha=32, blocks=[]')
    i0 = model.index("def PRE_FAST(im):")
    end = "    return (a - MEAN) / STD\n"
    i1 = model.index(end, i0) + len(end)
    assert "def PRE(im):" in model[:i0]
    model = PE_HEAD + PE_PRE_FAST + model[i1:]
    model = model[:model.index('REF = {"v37"')] + PE_WIRING
    model = model.replace("CLIPImageProcessor", "the PE-Core transform").replace("CLIP-L", "PE-Core")
    assert not any(t in model for t in ("clipL", "_proc", "CLIPL", "face_clipL")), "a CLIP-L leftover in the PE-Core cell"
    ext = ext.replace("the adapted CLIP-L", "the frozen PE-Core")
    old = 'version_notes=f"FACELORA-03 {ARM}: adapted CLIP-L arrays"'
    assert upload.count(old) == 1
    upload = upload.replace(old, 'version_notes=f"PEFACE-01 {ARM}: frozen PE-Core-L14-336 arrays (peL)"')
    return pip, train, model, LOOP_PE, ext, upload


def build(v):
    c = VARIANTS[v]
    farl = c.get("enc") == "farl16"
    exp = c.get("exp", "FACELORA-04" if farl else "FACELORA-03")
    pe = exp == "PEFACE-01"
    what = ("FaRL adapted by FACELORA-04" if farl else "frozen PE-Core-L14-336 of PEFACE-01" if pe else "CLIP-L adapted by " + exp)
    env = BR.env(c["out"], [c["arr"] + "V"], f"{what} {c['arm']} ({'no training' if pe else 'trained below'})")
    env = env.replace("if LOCAL and os.environ.get(", f"if {SMOKE} and os.environ.get(")
    env += f'ARM, ARR, SG = "{c["arm"]}", "{c["arr"]}", {c["sg"]}   # {exp} adapter of this notebook\n'
    data = BR.data(c["out"])
    assert data.count("if LOCAL and SMOKE_ROWS:") == 1
    data = data.replace("if LOCAL and SMOKE_ROWS:", f"if {SMOKE} and SMOKE_ROWS:")
    bal = " (gender-balanced: 8 male + 8 female identities per source)" if c["sg"] else ""
    loss = " + 0.5 x (InfoNCE within the male rows + within the female rows): same-gender negatives" if c["sg"] else ""
    intro = (INTRO_PE if pe else INTRO_FB if farl else INTRO_C5 if exp == "FACELORA-05" else INTRO_CO if exp == "COTRAIN-01"
             else INTRO_JT if exp == "JOINT-01" else INTRO_B3X if exp == "B3-EVAL" else INTRO_TAP if exp == "TAP-01"
             else INTRO_B3T if exp == "TTA-02" else INTRO_SA if exp == "FACELORA-06" else INTRO.format(v=v, bal=bal, loss=loss, **c))
    train = TRAIN.replace("__VOX2_GENDER__", (HERE.parent / "Experiment/SCALE-CC-01/vox2_gender.json").read_text(encoding="utf-8").strip())
    if c.get("bn42"):
        for a, b in BN42_FILTER:
            assert train.count(a) == 1, a
            train = train.replace(a, b)
        train += BN42_MARK
        if not pe:                                                    # PEFACE-01: nothing is retrained, its intro says why the pools are built
            intro = INTRO_CT.format(**dict(c, v=v[0], exp=exp)) + intro
    if c.get("strict"):
        assert c.get("bn42") and train.count(LS_ANCHOR) == 1 and train.count(LS_MARK[0]) == 1
        train = train.replace(LS_ANCHOR, LS_ANCHOR + LS_ADD).replace(*LS_MARK)
        if not pe:
            intro = INTRO_LS.format(**c) + intro
    model, ext, upload, enc, loop, pip, cell = MODEL, EXT, UPLOAD_CELL, ENC, LOOP, PIP, CELL
    if farl:
        train, model, ext, upload = farl_variant(train, model, ext, upload)
    if pe:
        pip, train, model, loop, ext, upload = pe_variant(pip, train, model, loop, ext, upload)
    if exp in ("FACELORA-05", "COTRAIN-01", "JOINT-01") or c.get("c5"):
        train, model, upload = c5_variant(train, model, upload)
    if exp == "COTRAIN-01":
        train, loop, upload = co_variant(train, loop, upload)
    if exp == "JOINT-01":
        pip, cell, train, loop, upload = jt_variant(pip, cell, train, loop, upload)
    if exp == "B3-EVAL":
        model = b3x_variant(model)
    if exp == "TTA-02":
        model, enc, ext, upload = b3t_variant(model, enc, ext, upload)
    if exp == "TAP-01":
        model, enc, ext, upload = tap_variant(model, enc, ext, upload)
    if exp == "FACELORA-06":
        train, loop, upload = sa_variant(train, loop, upload)
    if c.get("bn42"):
        old = 'OUT / "train_log.json"]'
        assert upload.count(old) == 1
        upload = upload.replace(old, 'OUT / "train_log.json", OUT / "bn42_clean.json"]')
    enc_name = "FaRL ep16" if farl else "PE-Core-L14-336 (frozen)" if pe else "CLIP-L"
    nb_name, ds = c.get("nb", f"FLAG_65_facelora3_{v}.ipynb"), c.get("ds", f"flag2027-feats-facelora3-{v}-p")
    cells = [("markdown", intro), ("code", env), ("code", pip),
             ("markdown", "## 1. Find the data (FLAG_12 cell): organiser lists with media, v1/v2 rows, resume"), ("code", data),
             ("markdown", "## 2. Audio loading, batching, helpers (FLAG_12 cell)"), ("code", B12.AUDIO12),
             ("markdown", "## 3. Track 3 encoder helpers (CLIP-L id)"), ("code", cell),
             ("markdown", f"## 4. {exp}: identities and frames of the three sources (registered filters)"),
             ("code", train),
             ("markdown", f"## 5. {enc_name} + LoRA, exact fast preprocessing, wiring check"), ("code", model),
             ("markdown", "## 6. Training (registered settings; checkpoint every 500 steps)"), ("code", loop),
             ("markdown", "## 7. The adapted encoder"), ("code", enc),
             ("markdown", f"## 8. `{c['arr']}` of every MAV-Celeb row (FLAG_12 loop, resumable)"), ("code", BR.run(())),
             ("markdown", f"## 9. `{c['arr']}` of every VoxCeleb2 / VoxBlink face row"), ("code", ext),
             ("markdown", "## 10. Manifest"), ("code", BR.done(ds)),
             ("markdown", "## 11. Upload the adapted arrays as a private dataset of this account (FL3_UPLOAD = owner/slug; optional)"),
             ("code", upload)]
    if pe:                                                            # PEFACE-01: section titles of a frozen encoder
        ttl = {"## 5. PE-Core-L14-336 (frozen) + LoRA, exact fast preprocessing, wiring check": "## 5. PE-Core-L14-336 (frozen), preprocessing, health checks",
               "## 6. Training (registered settings; checkpoint every 500 steps)": "## 6. No training (frozen encoder): 'encoder ready' marker",
               "## 7. The adapted encoder": "## 7. The frozen encoder in the FLAG_12 interface"}
        assert all(any(s == k for _, s in cells) for k in ttl), "PEFACE-01: a section title changed"
        cells = [(t, ttl.get(s, s)) for t, s in cells]
    if exp == "B3-EVAL":                                              # extraction only: no VoxCeleb2 / VoxBlink rows, no upload; a check
        i9 = next(i for i, (t, s) in enumerate(cells) if t == "markdown" and s.startswith("## 9."))
        cells = cells[:i9] + [("markdown", "## 9. Check: re-extracted arrays vs the saved B3 arrays"), ("code", B3X_CHECK.replace("clipB3", c["arr"])),
                              ("markdown", "## 10. Manifest"), ("code", BR.done(ds))]
        i8 = next(i for i, (t, s) in enumerate(cells) if t == "markdown" and s.startswith("## 8."))
        cells = cells[:i8] + [("markdown", "## 7b. Saved train / v1 / v2 arrays copied in: only the test lists are embedded"),
                              ("code", (B3X_PRECOPY_ALL if c.get("copy_all") else B3X_PRECOPY).replace("clipB3", c["arr"]))] + cells[i8:]
    if exp == "FACELORA-06":                                          # the South-Asian groups, after the pools are built
        groups = {}
        for f in ("Experiment/SCALE-CC-01/vox2b_groups.json", "Experiment/DSEL-01/vox2_groups.json"):
            groups.update({k: v[0] for k, v in json.loads((HERE.parent / f).read_text(encoding="utf-8")).items()})
        i5 = next(i for i, (t, s) in enumerate(cells) if t == "markdown" and s.startswith("## 5."))
        cells = cells[:i5] + [("markdown", "## 4b. FACELORA-06: South-Asian identities of the pools"),
                              ("code", SA_CELL.replace("__VOX2_GROUPS__", json.dumps(groups, sort_keys=True)))] + cells[i5:]
    if exp == "TAP-01":                                               # the taps after the model; the wiring check before the manifest
        i6 = next(i for i, (t, s) in enumerate(cells) if t == "markdown" and s.startswith("## 6."))
        cells = cells[:i6] + [("markdown", "## 5b. TAP-01: CLIP-L / FaRL taps"), ("code", TAP_MODEL)] + cells[i6:]
        i10 = next(i for i, (t, s) in enumerate(cells) if t == "markdown" and s.startswith("## 10."))
        cells = cells[:i10] + [("markdown", "## 9b. TAP-01 wiring: recomputed final arrays vs the stored ones"), ("code", TAP_CHECK)] + cells[i10:]
    if exp == "TTA-02":                                               # + a check of the flip-averaged arrays before the upload
        i10 = next(i for i, (t, s) in enumerate(cells) if t == "markdown" and s.startswith("## 10."))
        cells = cells[:i10] + [("markdown", "## 9b. Check: clipB3t vs the saved clipB3"), ("code", B3T_CHECK)] + cells[i10:]
    if exp == "JOINT-01":                                             # the voice net after the face model; its extraction after the face rows
        i6 = next(i for i, (t, s) in enumerate(cells) if t == "markdown" and s.startswith("## 6."))
        cells = cells[:i6] + [("markdown", "## 5b. JOINT-01: the voice net (sam100f, layer3.16-23 + layer4 + pool + bottleneck trainable) on the 2nd GPU, wiring check"),
                              ("code", JT_VOICE)] + cells[i6:]
        i10 = next(i for i, (t, s) in enumerate(cells) if t == "markdown" and s.startswith("## 10."))
        cells = cells[:i10] + [("markdown", "## 9b. JOINT-01: the trained voice net (`voicelora_state.pt`) and `sam100fJ` of every cached clip"),
                               ("code", JT_EXTRACT)] + cells[i10:]
    if c.get("saved"):                                               # ALF-01: the saved adapter of another dataset (B3c)
        rx = "feats[-_]" + c["saved"].split("/")[1].replace("flag2027-feats-", "").replace("-", "[-_]")
        n0 = sum(s_.count("feats[-_]facelora3[-_]b[-_]p") for _, s_ in cells)
        cells = [(t, s_.replace("feats[-_]facelora3[-_]b[-_]p", rx).replace("nguynchnngn1/flag2027-feats-facelora3-b-p", c["saved"])
                  .replace("CLIP-L (B3) taps", "CLIP-L (B3c) taps").replace("the SAVED B3 adapter", "the SAVED B3c adapter")) for t, s_ in cells]
        assert n0 >= 1 and sum(s_.count(rx) for _, s_ in cells) == n0, (n0, rx)
    B12.nb(cells, nb_name)
    print(f"{nb_name} built")


if __name__ == "__main__":
    for v in (sys.argv[1:] or list(VARIANTS)):
        build(v)
