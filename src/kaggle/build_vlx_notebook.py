"""Generate FLAG_67x_sam100fL_mav.ipynb (VOICELORA-01 step 1b, README addendum 17/10 17:00 before any result): FLAG_67a's adapted
sam100f (`voicelora_state.pt`: layer4 + pooling + bottleneck of WeSpeaker SimAM-ResNet100 ft, trained towards the frozen B3 face)
on every MAV-Celeb voice row: v4 train, the dev lists (or Evaluation lists: ONLY_UNDER), v1/v2. Same cut as the training crops:
the centre 3 s of the clip, zero-padded at the end when shorter (the audio cache's layout). Same FLAG_12 loop / row order / file
names as FLAG_47b (ENC-SCREEN-02), so VOICELORA-01's run.py reads it like ENC-SCALE-01 reads feats_screen2.
    python build_vlx_notebook.py
Smoke test on Kaggle CPU (kapi --smoke): FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6 with FLAG_67a's smoke output attached (a state file).
"""
import build_eval_notebooks as B12
import build_rep_notebooks as BR
import build_t3enc_notebook as T3
import build_vfeat_notebook as VF                                      # T3 cell with the ModelScope retry patch

SMOKE = '(LOCAL or os.environ.get("FLAG_SMOKE") == "1")'
_RG = 'Path("/kaggle/input").rglob(zip_name)'
assert VF.T3CELL.count(_RG) == 1
T3CELL = VF.T3CELL.replace(_RG, "(q for r in _inroots() for q in r.rglob(zip_name))")   # _inroots: defined in the VLX cell
OUT = "feats_vlx"

VLX = r'''# ============================================================================ VOICELORA-01 adapted sam100f (FLAG_67a), frozen here
# sam100fA (WeSpeaker SimAM-ResNet100 ft) with FLAG_67a's adapted tensors copied in; FLAG_67a trained (and extracted the auxiliary
# rows) on centre 3 s crops of int16 audio, zero-padded at the end: the same cut here. Output: the 256-d embedding (no FPROJ).
import torch


def _inroots():
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


class AdaptedVoice(SimAMVoice):
    CROP = 3 * 16000

    def __init__(self):
        print(f"AdaptedVoice: input roots scanned {[r.name for r in _inroots()]}", flush=True)
        super().__init__("sam100fA")
        st = [p for p in os.environ.get("VL_STATE", "").split(os.pathsep) if p]
        st += sorted(str(p) for r in _inroots() for p in r.rglob("voicelora_state.pt"))
        if not st and os.environ.get("FLAG_SMOKE") == "1":           # wiring smoke without a state file: the frozen sam100f
            print("!! SMOKE STAND-IN: no voicelora_state.pt attached, AdaptedVoice = frozen sam100f (centre 3 s)", flush=True)
            T3_STATS["sam100fLA"] = {"state": None, "smoke_standin": True}
            return
        assert st, "attach FLAG_67a's output (voicelora_state.pt) or set VL_STATE"
        S = torch.load(st[0], map_location="cpu", weights_only=False)
        if "__EXP__":                                                  # the state must come from the registered step 1
            assert (S.get("cfg") or {}).get("exp") == "__EXP__", ("wrong voicelora_state.pt", st[0], S.get("cfg"))
        par = dict(self.m.model.named_parameters())
        miss = [n for n in S["net"] if n not in par]
        assert not miss, f"adapted tensors without a SimAM parameter: {miss[:5]}"
        with torch.no_grad():
            for n, t in S["net"].items():
                assert par[n].shape == t.shape, (n, par[n].shape, t.shape)
                par[n].copy_(t.to(par[n].device, par[n].dtype))
        self.m.model.eval()
        T3_STATS["sam100fLA"] = {"state": st[0], "tensors": len(S["net"]), "cfg": S.get("cfg"), "crop_s": self.CROP / 16000}
        print(f"AdaptedVoice: {len(S['net'])} adapted tensors from {st[0]} (cfg {S.get('cfg')})", flush=True)

    @torch.no_grad()
    def __call__(self, w, use_amp=True):
        w = np.asarray(w, np.float32)
        L = self.CROP
        if len(w) > L:
            o = (len(w) - L) // 2
            w = w[o:o + L]
        elif len(w) < L:
            w = np.pad(w, (0, L - len(w)))
        return super().__call__(w, use_amp)


NEW = {"sam100fLA": AdaptedVoice}
ARRAYS_NEW = {"sam100fLA": ("voice", {"": "sam100fL"})}
MAIN_NEW = {k: "" for k in NEW}
RESHAPE = {}
print("VLX cell ready: encoders", list(NEW), flush=True)
'''

INTRO = """
# FLAG 2027 — NB-67x: VOICELORA-01 step 1b, the adapted voice encoder `sam100fL` on every MAV-Celeb voice

**Why.** FLAG_67a adapted the last stage + pooling + embedding layer of SimAM-ResNet100 ft (`sam100f`) towards the frozen B3 face
on ~21k VoxCeleb2 / VoxBlink identities (Bengali speakers never in the audio cache). Step 2 (CC head [face] x [sam100p | sam100f |
sam100fL]) needs the same encoder on our rows: v4 train, the test lists, v1/v2.

| array | model | cut | dim |
|---|---|---|---|
| `voice_sam100fL` | `sam100f` + FLAG_67a's `voicelora_state.pt` (frozen here) | centre 3 s (as the training crops) | 256 |

**Rule check:** no Bengali voice was used to train the adapter (the audio cache holds only speakers kept by each source's LID rule);
the test files are only embedded, one file at a time, with a frozen model (level 0). Declare the adapted encoder in the system
description.

**Inputs (Add Input):** `ngnnguynchn/flag2027-raw-v4-p`, `chnnhnnguyn/external-data-mavceleb`, `ngnnguynchn/flag2027-ext-meta-p`,
FLAG_67a's output dataset (`voicelora_state.pt`). Evaluation (11/11): + the Evaluation data, `ONLY_UNDER`. **GPU T4**, **Internet ON**
(WeSpeaker zip). **Time:** about 1 h (one voice model, ~51,000 files). Resumable.

**Output:** `/kaggle/working/feats_vlx/` -> dataset **`flag2027-feats-vlx`** (private).
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
        A = load_arr("sam100fL", "voice", "train")
        if A is not None:
            A = l2n(A); rows.append(dict(array="voice_sam100fL", same_gender_identity_eer=round(eer(np.sum(A[pr[:, 0]] * A[pr[:, 1]], 1), y), 2)))
PR = pd.DataFrame(rows); print(PR.to_string(index=False) if len(PR) else "probe: nothing to score (smoke test rows)")
CHECKS["probe_within_modality"] = rows
'''


def cells():
    return cells_ds("flag2027-feats-vlx")


def cells_ds(ds):
    key = [k for k in ("sam100fLA", "sam100pLA", "sam100fDA", "sam100fJA") if f'NEW = {{"{k}"' in VLX][0]
    out = BR.nb_cells(INTRO, BR.env(OUT, [key], "the adapted voice only"),
                      T3.PIP, OUT, [("code", T3CELL), ("code", VLX)], PROBE, ds)
    res, n_env, n_data = [], 0, 0
    for kind, src in out:
        if kind == "code" and "if LOCAL and os.environ.get(" in src:          # env cell: smoke switches on Kaggle CPU too
            src = src.replace("if LOCAL and os.environ.get(", f"if {SMOKE} and os.environ.get("); n_env += 1
        if kind == "code" and "if LOCAL and SMOKE_ROWS:" in src:               # data cell: row cut on Kaggle CPU too
            src = src.replace("if LOCAL and SMOKE_ROWS:", f"if {SMOKE} and SMOKE_ROWS:"); n_data += 1
        res.append((kind, src))
    assert n_env == 1 and n_data == 1, (n_env, n_data)
    return res


# step 1b of the 19/10 "mở rộng" experiments (READMEs written before any result): same notebook, another state / encoder / array
VARIANTS = {
    "x": dict(nb="FLAG_67x_sam100fL_mav.ipynb", enc="sam100fA", key="sam100fLA", arr="sam100fL", exp="", out="feats_vlx",
              ds="flag2027-feats-vlx", src="FLAG_67a"),
    "p": dict(nb="FLAG_67z_sam100pL_mav.ipynb", enc="sam100pA", key="sam100pLA", arr="sam100pL", exp="VOICELORA-03", out="feats_vlx3",
              ds="flag2027-feats-vlx3", src="FLAG_67c"),
    "d": dict(nb="FLAG_67z_sam100fD_mav.ipynb", enc="sam100fA", key="sam100fDA", arr="sam100fD", exp="VOICELORA-04", out="feats_vlx4",
              ds="flag2027-feats-vlx4", src="FLAG_67d / 67dh (the lr arm chosen by the registered rule)"),
    "j": dict(nb="FLAG_67z_sam100fJ_mav.ipynb", enc="sam100fA", key="sam100fJA", arr="sam100fJ", exp="JOINT-01", out="feats_vlxj",
              ds="flag2027-feats-vlxj", src="FLAG_65j (JOINT-01)"),
}


def build(v):
    global OUT, VLX, INTRO, PROBE
    V = VARIANTS[v]
    keep = OUT, VLX, INTRO, PROBE
    enc0 = "sam100p" if V["enc"] == "sam100pA" else "sam100f"
    OUT = V["out"]
    VLX = (keep[1].replace('super().__init__("sam100fA")', f'super().__init__("{V["enc"]}")').replace("sam100fLA", V["key"])
           .replace('{"": "sam100fL"}', '{"": "%s"}' % V["arr"]).replace("FLAG_67a", V["src"])
           .replace("frozen sam100f", f"frozen {enc0}").replace("__EXP__", V["exp"]))
    if v != "x":
        INTRO = (keep[2].replace("NB-67x: VOICELORA-01 step 1b", f"NB-67z: {V['exp']} step 1b").replace("sam100fL", V["arr"])
                 .replace("FLAG_67a", V["src"]).replace("flag2027-feats-vlx", V["ds"]).replace("feats_vlx/", V["out"] + "/")
                 .replace("ft (`sam100f`)", f"(`{enc0}`)").replace("| `sam100f` + ", f"| `{enc0}` + "))
    PROBE = keep[3].replace("sam100fL", V["arr"])
    try:
        B12.nb(cells_ds(V["ds"]), V["nb"])
    finally:
        OUT, VLX, INTRO, PROBE = keep
    print(V["nb"], "built")


if __name__ == "__main__":
    for v in (__import__("sys").argv[1:] or ["x"]):
        build(v)
