"""Generate FLAG_40_t3enc_mav.ipynb (T3ENC-01, V1 of docs/PLAN_VOXBLINK_2026-10-07.md): the encoder family of Track 3's
VoxBlink system (FaRL ep16, CLIP ViT-L/14, WeSpeaker SimAM-ResNet100 pretrained / fine-tuned on VoxBlink2) for the exact rows
of the eda bundle (v4 train + dev lists in organiser order, MAV-Celeb v1/v2 in ext_*_meta order). Its VoxCeleb2 twin is
FLAG_41 (build_vox2t3_notebook.py). Judged locally (Experiment/T3ENC-01).
    python build_t3enc_notebook.py
    FLAG_LOCAL=1 FLAG_IN=<mini> FLAG_OUT=<dir> FLAG_CACHE=<cache> FLAG_SMOKE_ROWS=6 FLAG_T3_PATH=<isolated site dir> \
        python run_notebook_local.py FLAG_40_t3enc_mav.ipynb --jupyter
"""
from pathlib import Path
import build_eval_notebooks as B12
import build_rep_notebooks as BR

HERE = Path(__file__).resolve().parent
CELL = (HERE / "vendor_t3enc/t3enc_cell.py").read_text(encoding="utf-8")

INTRO = """
# FLAG 2027 — NB-40: T3ENC-01, the encoders of Track 3's VoxBlink system on our MAV-Celeb rows

**Why.** A teammate's VoxBlink system (Track 3) reaches about Bangla ng 16 / g 25 on CodaBench, against 18.07 / 28.07 for
our v9. VoxBlink media exist only as an unofficial crawl, so we first test the clean part (plan V1): **are the encoders
and a cross-modal head enough, with MAV-Celeb + VoxCeleb2 as training data?** This notebook embeds every face and voice
of our rows with the same encoder family; FLAG_41 does the same for the ~4,900 VoxCeleb2 identities of FLAG_37.

| array | model | dim |
|---|---|---|
| `face_farl16` | FaRL ViT-B/16, LAION-Face 20M, **ep16** (official GitHub release; our existing `farl` is ep64) | 512 |
| `face_clipL` | `openai/clip-vit-large-patch14` @ 32bd642, image features | 768 |
| `voice_sam100p` | WeSpeaker `voxblink2_samresnet100` (VoxBlink2 pretrained; official ModelScope zip) | 256 |
| `voice_sam100f` | WeSpeaker `voxblink2_samresnet100_ft` (VoxBlink2 -> VoxCeleb2 fine-tune) | 256 |

**Rule check:** frozen pretrained encoders only (VoxBlink2-pretrained audio encoders are named as allowed by the
organisers; face encoders: any). Nothing is trained here; every sample is embedded on its own; the dev lists are only
encoded, never used for anything else.

**Inputs (Add Input):** the raw v4 `train_set` + `dev_set` (as FLAG_33-39), MAV-Celeb v1/v2 + `flag2027-ext-meta`.
Optional: this notebook's previous output, to resume. **Settings:** GPU T4, **Internet ON** (GitHub, Hugging Face,
ModelScope).

**Time:** about 3-4 h: faces ~30 min (two models), each voice model re-reads ~51,000 audio files (~1-1.5 h each, as
FLAG_34). Resumable: if the session stops, Save Version, attach the output, run again.

**Output:** `/kaggle/working/feats_t3enc/` -> dataset **`flag2027-feats-t3enc`** (about 0.3 GB).
"""

PIP = r'''# WeSpeaker (commit 9fecd6c, as Track 3) and openai CLIP (commit d50d76d, FaRL's code base): --no-deps, their pins would
# replace Kaggle's torch; the few runtime deps by hand.
if not LOCAL:
    !pip -q install ftfy regex kaldiio silero-vad 2>&1 | tail -1
    # 2/11: FLAG_50's fix of 29/10 (BEST/v1977/README.md §6; the FLAG_47b re-run smoke lost wespeaker the same way): one pip call
    # per package; the openai CLIP build fails on Kaggle's current image (setup.py needs pkg_resources) and, in a shared call, took wespeaker down with it (FLAG_103c smoke). CLIP then comes from a git clone (FLAG_103a smoke).
    !pip -q install --no-deps "git+https://github.com/wenet-e2e/wespeaker.git@9fecd6cb4f47475d01761d87c826298dff4ef18c" 2>&1 | tail -2
    !pip -q install --no-deps "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1" 2>&1 | tail -2
    try:
        import clip
    except ModuleNotFoundError:
        import subprocess
        _oc = Path("/tmp/openai_clip")
        if not (_oc / "clip").exists():
            subprocess.run(["git", "clone", "-q", "https://github.com/openai/CLIP.git", str(_oc)], check=True)
            subprocess.run(["git", "-C", str(_oc), "checkout", "-q", "d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"], check=True)
            _f = _oc / "clip/clip.py"
            _f.write_text(_f.read_text().replace("from pkg_resources import packaging", "import packaging, packaging.version"))
        sys.path.insert(0, str(_oc))
        import clip
    import importlib.util                       # not `import wespeaker`: its frontends need s3prl (SimAMVoice stubs them first)
    assert importlib.util.find_spec("wespeaker") is not None, "wespeaker did not install"
    print("openai CLIP:", clip.__file__, "| wespeaker:", importlib.util.find_spec("wespeaker").origin, flush=True)
else:                                           # local smoke test: wespeaker + its deps in an isolated folder
    for p_ in reversed([p for p in os.environ.get("FLAG_T3_PATH", "").split(";") if p]):
        sys.path.insert(0, p_)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, transformers
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEV,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
'''

PROBE = BR.PROBE_LIB + r'''
# Sanity probe, WITHIN each modality (the four encoders live in different spaces, so no zero-shot face <-> voice score):
# v4 train, two rows of the same person vs two rows of different persons of the same gender, cosine EER.
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
        for kind, stem in (("face", "farl16"), ("face", "clipL"), ("voice", "sam100p"), ("voice", "sam100f")):
            A = load_arr(stem, kind, "train")
            if A is not None:
                A = l2n(A); rows.append(dict(array=f"{kind}_{stem}", same_gender_identity_eer=round(eer(np.sum(A[pr[:, 0]] * A[pr[:, 1]], 1), y), 2)))
PR = pd.DataFrame(rows); print(PR.to_string(index=False) if len(PR) else "probe: nothing to score (smoke test rows)")
CHECKS["probe_within_modality"] = rows
CHECKS["t3enc"] = dict(T3_STATS)
# Reading guide: v4 train rows of one person may share a session, so these EERs are optimistic; expected order of
# magnitude: faces a few % (FaRL / CLIP are not face-ID models: higher than ArcFace ~1), SimAM voices a few %.
'''


if __name__ == "__main__":
    cells = BR.nb_cells(INTRO, BR.env("feats_t3enc", ["farl16V", "clipLV", "sam100pA", "sam100fA"], "faces first (cheap), then voices"),
                        PIP, "feats_t3enc", [("code", CELL)], PROBE, "flag2027-feats-t3enc")
    B12.nb(cells, "FLAG_40_t3enc_mav.ipynb")
