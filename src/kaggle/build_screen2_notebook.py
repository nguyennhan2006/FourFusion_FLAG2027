"""Generate FLAG_47b_voice_screen.ipynb (ENC-SCREEN-02, 12/10): the two voice encoders of Track 3's SUB21a that we never had,
WeSpeaker voxceleb_resnet293_LM (FLAG_47's run failed on the zip layout, loader fixed since) and voxceleb_ECAPA1024_LM, on our
MAV-Celeb rows (v4 train, dev lists, v1/v2). Screened locally with ENC-SCREEN-01's A-arm head; only a winner is extracted on
VoxCeleb2 / VoxBlink afterwards (rule in Experiment/ENC-SCREEN-02/README.md).
    python build_screen2_notebook.py
Smoke test on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6
"""
from pathlib import Path
import build_eval_notebooks as B12
import build_rep_notebooks as BR
import build_t3enc_notebook as T3

HERE = Path(__file__).resolve().parent
CELL = (HERE / "vendor_t3enc/t3enc_cell.py").read_text(encoding="utf-8")
SMOKE = '(LOCAL or os.environ.get("FLAG_SMOKE") == "1")'
OUT = "feats_screen2"

SCREEN2 = r'''# ============================================================================ ENC-SCREEN-02 voice encoders (frozen, WeSpeaker VoxCeleb2 LM)
# Track 3's SUB21a voice branch: cat4 = L2(sam100p) | L2(sam100f) | L2(rn293) | L2(ecapa1024) (their report of 11/10).
#   rn293A      voxceleb_resnet293_LM (official ModelScope zip; files named <model>.pt / .yaml, renamed by wespeaker_dir), 256-d
#   ecapa1024A  voxceleb_ECAPA1024_LM (official ModelScope zip), 192-d
# Same wespeaker path as sam100p / sam100f: int16-scaled 16 kHz waveform, kaldi fbank 80 + CMN, one forward per file.
SAM_ZIPS.update({"rn293A": ("voxceleb_resnet293_LM.zip", 124116126), "ecapa1024A": ("voxceleb_ECAPA1024_LM.zip", 67665905)})
NEW = {"rn293A": lambda: SimAMVoice("rn293A"), "ecapa1024A": lambda: SimAMVoice("ecapa1024A")}
ARRAYS_NEW = {"rn293A": ("voice", {"": "rn293"}), "ecapa1024A": ("voice", {"": "ecapa1024"})}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {}
'''

INTRO = """
# FLAG 2027 — NB-47b: ENC-SCREEN-02, ResNet293-LM and ECAPA1024-LM on our MAV-Celeb rows

**Why.** Track 3's best system (SUB21a, CB 20.18) concatenates four voice encoders; we only ever had two of them (SimAM-ResNet100
p / ft). T3REPRO-01 rebuilt their system without the other two (ResNet293-LM, ECAPA1024-LM). This notebook embeds both on our
rows so ENC-SCREEN-01's local A-arm screen can tell whether they add anything before any VoxCeleb2 / VoxBlink extraction.

| array | model | dim |
|---|---|---|
| `voice_rn293` | WeSpeaker `voxceleb_resnet293_LM` (VoxCeleb2, official ModelScope zip) | 256 |
| `voice_ecapa1024` | WeSpeaker `voxceleb_ECAPA1024_LM` (VoxCeleb2, official ModelScope zip) | 192 |

**Rule check:** frozen voice encoders pretrained on VoxCeleb (allowed by the organisers); no training; every file on its own.

**Inputs (Add Input):** the raw v4 `train_set` + `dev_set`, MAV-Celeb v1/v2 + `flag2027-ext-meta` (as FLAG_40 / FLAG_47).
Optional: this notebook's previous output, to resume. **Settings:** GPU T4, **Internet ON** (ModelScope, GitHub).

**Time:** each voice model re-reads ~51,000 audio files, ~1-1.5 h each: about 2.5-3 h. Resumable.

**Output:** `/kaggle/working/feats_screen2/` -> dataset **`flag2027-feats-screen2`** (private).
"""

PROBE = BR.PROBE_LIB + r'''
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
        for kind, stem in (("voice", "rn293"), ("voice", "ecapa1024")):
            A = load_arr(stem, kind, "train")
            if A is not None:
                A = l2n(A); rows.append(dict(array=f"{kind}_{stem}", same_gender_identity_eer=round(eer(np.sum(A[pr[:, 0]] * A[pr[:, 1]], 1), y), 2)))
PR = pd.DataFrame(rows); print(PR.to_string(index=False) if len(PR) else "probe: nothing to score (smoke test rows)")
CHECKS["probe_within_modality"] = rows
'''


def cells():
    out = BR.nb_cells(INTRO, BR.env(OUT, ["rn293A", "ecapa1024A"], "voices only, ResNet293 first"),
                      T3.PIP, OUT, [("code", CELL), ("code", SCREEN2)], PROBE, "flag2027-feats-screen2")
    res, n_env, n_data = [], 0, 0
    for kind, src in out:
        if kind == "code" and "if LOCAL and os.environ.get(" in src:          # env cell: smoke switches on Kaggle CPU too
            src = src.replace("if LOCAL and os.environ.get(", f"if {SMOKE} and os.environ.get("); n_env += 1
        if kind == "code" and "if LOCAL and SMOKE_ROWS:" in src:               # data cell: row cut on Kaggle CPU too
            src = src.replace("if LOCAL and SMOKE_ROWS:", f"if {SMOKE} and SMOKE_ROWS:"); n_data += 1
        res.append((kind, src))
    assert n_env == 1 and n_data == 1, (n_env, n_data)
    return res


if __name__ == "__main__":
    B12.nb(cells(), "FLAG_47b_voice_screen.ipynb")
