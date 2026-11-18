r"""CLEANTUNE-01 round 1 (registered 31/10 in README.md before any result): the CC head on Bengali-clean rows and adapters (BN42=1:
FLAG_53 rows also lose FLAG_42's flagged speakers; only adapter arrays marked bn42_clean.json, FLAG_109), one change per arm
against the in-run control B0c (= the B3 recipe: face [farl16 | clipB3], voice [sam100p | sam100f], SUB21A + V4CFG, aux VoxCeleb2 +
VoxBlink, 3 seeds):
  g1  B0c, DRm (dropout .45 / .20), DRp (.70 / .40), EPm (50 epochs), EPp (120), WDp (weight decay 1e-3)
  g2  B0c, AWm (aux weight .5), AIp (128 aux ids), BSp (128 ids per batch), DIMp (dim 1024), T (temp .2 -> .1), M (margin + LM-FT)
  g3  B0c, C5 (face clipC5), CAT4 (voice + rn293 + ecapa1024), TAPv (voice + samS4 PCA 512), C5CAT4; wider voice -> dropout .45.
      g3 rows: auxiliary voices without every extra block are left out in EVERY g3 arm (so g3's B0c is its own control).
    CT_GROUP=g1 FOLDS=1,2 python run.py      (fold jobs; tables ct01<group>_fold<s>)
    python run.py merge ; python run.py report   (ct01_fold<s> = the three groups side by side -> rules -> ct_round2.json)
Env: CT_GROUP (g1 / g2 / g3), TR_SEEDS (3), TR_MAX_STEPS (smoke), BN42_STANDIN=1 (smoke only: unmarked adapter arrays),
CT_STANDIN=1 (smoke only: g3 extra voice blocks = copies of [sam100p | sam100f] when their datasets are not attached).
"""
import importlib.util, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
os.environ["BN42"] = "1"                                              # every arm, every group: the Bengali-clean rows (README §1)
SMOKE = bool(os.environ.get("TR_MAX_STEPS"))
assert SMOKE or os.environ.get("BN42_STANDIN") != "1", "BN42_STANDIN is for smoke tests only"
assert SMOKE or os.environ.get("CT_STANDIN") != "1", "CT_STANDIN is for smoke tests only"
GROUP = os.environ.get("CT_GROUP", "g1")
T3S = os.environ.get("T3S") == "1"                                    # 4/11 T3S-01 (opt-in, Experiment/T3S-01/README.md): Track-3 recipe branches; unset = unchanged bit for bit
T3S_PART = os.environ.get("T3S_PART", "") if T3S else ""              # T3S-01: the face of one job (tables ct01r2<face>_fold<s>); "" = every face (report / zip)
TAG = f"ct01{GROUP}{T3S_PART}"
ARMS_OF = {"g1": ("B0c", "DRm", "DRp", "EPm", "EPp", "WDp"),
           "g2": ("B0c", "AWm", "AIp", "BSp", "DIMp", "T", "M"),
           "g3": ("B0c", "C5", "CAT4", "TAPv", "C5CAT4")}
GROUPS = tuple(ARMS_OF)
VARS = {"B0c": {}, "DRm": dict(face_dropout=.45, voice_dropout=.20), "DRp": dict(face_dropout=.70, voice_dropout=.40),
        "EPm": dict(epochs=50), "EPp": dict(epochs=120), "WDp": dict(wd=1e-3),
        "AWm": dict(aux_weight=.5), "AIp": dict(aux_ids=128), "BSp": dict(ids_per_batch=128), "DIMp": dict(dim=1024),
        "T": dict(lc_temp_end=.1), "M": dict(lc_m_hi=.25, lc_ft_m=.3, lc_ft_lr=.1)}
CONFLICTS = (("DRm", "DRp"), ("EPm", "EPp"))                          # same parameter: keep the larger gain (README §2 rule 4)


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


WHV3 = os.environ.get("WHV3") == "1"                                 # 3/11 WHISPER-02 (opt-in): Whisper-large-v3 voice block (FLAG_160); unset = unchanged bit for bit
WIDE = {"CAT4", "TAPv", "C5CAT4", "C5CAT4W", "C5WH"}                                     # C4t's rule (ENC-SCALE-01): the wider voice gets dropout .45
PLAN = json.loads((HERE / os.environ.get("CT_PLAN", "ct_round2.json")).read_text()) if GROUP == "r2" else None   # 3/11 CT_PLAN: CLEANTUNE-02's plan (opt-in)
FW = ("g3" if PLAN["I"] != "B3c" else "g1") if PLAN else GROUP         # round 2 runs in the framework of its input I*
if FW == "g3":
    os.environ["VL2_TAG"] = TAG
    V1 = _load("voicelora01run", "VOICELORA-01/run.py")              # FACELORA-05 faces [.. | clipB3 | clipC5] + extra-voice hooks
    F5, FL, M, C1 = V1.F5, V1.FL, V1.M, V1.C1
    M.TAG = C1.TAG = FL.TAG = F5.TAG = TAG
    XD = {"samS4": 512, "rn293": 256, "ecapa1024": 192}               # extra voice blocks after [sam100p | sam100f] (512)
    M.VOICE.update({"v_pf": np.r_[0:512], "v_tap": np.r_[0:1024], "v_cat4": np.r_[0:512, 1024:1472]})
    if WHV3:
        XD["whv3"] = 1280                                             # last block: columns 1472:2752 of the g3 voice rows
        M.VOICE.update({"v_cat4w": np.r_[0:512, 1024:2752], "v_wh": np.r_[1472:2752]})
    if T3S:
        assert not WHV3, "T3S-01 and WHISPER-02 blocks are not combined"
        XD.update(wsv=512, rdn=192)                                   # T3S-01: WavLM-Base+-SV / ReDimNet b6 (SCOUT-01): MAV rows only, zeros on auxiliary rows
    INPUT = {"B3c": ("FB3", "v_pf"), "C5": ("FC5", "v_pf"), "CAT4": ("FB3", "v_cat4"), "TAPv": ("FB3", "v_tap"),
             "C5CAT4": ("FC5", "v_cat4"), "C5CAT4W": ("FC5", "v_cat4w"), "C5WH": ("FC5", "v_wh")}
    GARMS = {"B0c": "B3c", "C5": "C5", "CAT4": "CAT4", "TAPv": "TAPv", "C5CAT4": "C5CAT4"}
else:
    os.environ.setdefault("FL3_ARMS", "B3")
    FL = _load("facelora03run", "FACELORA-03/run.py")
    M, C1 = FL.M, FL.C1
    M.TAG = C1.TAG = FL.TAG = TAG
    INPUT = {"B3c": ("B3", "cat")}
    GARMS = {a: "B3c" for a in ARMS_OF.get(GROUP, ())}
CFG = dict(M.t3r.SUB21A, **FL.V4CFG)


def arm_cfg(inp, knobs):
    """B0c's recipe + the input's voice-dropout rule + the knobs; a dropout knob on a wide voice keeps the +.15 (README §6c)."""
    c = dict(CFG, **({"voice_dropout": .45} if inp in WIDE else {}))
    for k in knobs:
        c.update(VARS[k])
        if inp in WIDE and "voice_dropout" in VARS[k]:
            c["voice_dropout"] = VARS[k]["voice_dropout"] + .15
    return c


def r2_spec(plan):
    """README §2 round 2: B0c, I* alone, the best single knob (B3c input), CMB-ALL, CMB-TOP2; duplicates run once."""
    I, SA = plan["I"], list(plan["S_A"])
    spec, seen = {"B0c": ("B3c", ())}, {("B3c", ())}
    for name, key in (("I", (I, ())), ("S1", ("B3c", tuple(SA[:1]))), ("CA", (I, tuple(SA))), ("C2", (I, tuple(SA[:2])))):
        if key not in seen:
            spec[name] = key; seen.add(key)
    if WHV3:                                                          # WHISPER-02: CAW = CA + whv3 block, CAWO = whv3 alone as the voice
        spec["CAW"], spec["CAWO"] = ("C5CAT4W", tuple(SA)), ("C5WH", tuple(SA))
    if os.environ.get("CT_R2_EXTRA") == "F1S" and not any(v[0] == "C5CAT4" and set(v[1]) == {"T", "AIp"} for v in spec.values()):
        spec["F1S"] = ("C5CAT4", ("T", "AIp"))                        # 3/11 (CLEANTUNE-02 README §1b): F1s's recipe run in the same round, opt-in
    only = os.environ.get("CT_R2_ONLY")                               # 3/11 (F1s): opt-in, only these arms (e.g. "CA"); unset = unchanged
    return {a: v for a, v in spec.items() if a in only.split(",")} if only else spec


OFF = {}
if GROUP == "r2":
    SPEC = r2_spec(PLAN)
    CANDS = [a for a in SPEC if a != "B0c"]
    for a, (inp, kn) in SPEC.items():                                 # fresh seeds (+100); 'b' copies (+200) of every candidate (CT20)
        for name, off in ((a, 100),) + (((a + "b", 200),) if a != "B0c" and not T3S else ()):   # T3S-01: control CA without its 'b' copy
            M.PAIRS[name] = INPUT[inp]
            M.ARMS[name] = dict(aux=("vox", "vb"), pairs=[name], seeds=M.N_SEEDS, cfg=arm_cfg(inp, kn))
            OFF[name] = off
    M.ARMS = {a: M.ARMS[a] for a in OFF}
    _ta = M.train_arm

    def _train_arm(R, main, aux_src, arm, seed0, device, tlog):
        return _ta(R, main, aux_src, arm, seed0 + OFF.get(arm, 0), device, tlog)
    M.train_arm = _train_arm                                          # T3RECIPE-01 run() / final() look it up at call time
else:
    M.PAIRS.update({a: INPUT[g] for a, g in GARMS.items()})
    M.ARMS = {a: dict(aux=("vox", "vb"), pairs=[a], seeds=M.N_SEEDS, cfg=arm_cfg(g, [a] if a in VARS and a != "B0c" else []))
              for a, g in GARMS.items()}
M.DECISION = FL.DECISION = "B0c"
if FW == "g3":
    F5.DECISION = V1.DECISION = "B0c"
MAVONLY, _MAV = set(), {}
if T3S:                                                               # T3S-01 README §1: one Track-3 projector (SUB21A = sub60's trainer dict) per face x voice
    assert GROUP == "r2" and FW == "g3", "T3S-01 runs in the round-2 g3 framework (CT_GROUP=r2, ct_round2.json I = C5CAT4)"
    _o = dict(zip(XD, (512 + np.cumsum([0] + list(XD.values()))[:-1]).tolist()))
    _b = lambda k: np.r_[_o[k]:_o[k] + XD[k]]
    T3S_F = {"farl": FL.COLS["farl16"], "clip": FL.COLS["clipL"], "b3s": FL.COLS["clipB3"], "c5s": FL.COLS["clipC5"]}
    T3S_V = {"sim": np.r_[0:256], "r293": _b("rn293"), "ecapa": _b("ecapa1024"), "cat4": M.VOICE["v_cat4"], "wsv": _b("wsv"), "rdn": _b("rdn")}
    NS3 = int(os.environ.get("T3S_SEEDS", "5"))
    if T3S_PART not in ("", "farl"):
        M.ARMS = {}                                                   # the control CA (F1s's head) trains once per fold, in the farl job
    for f in ([T3S_PART] if T3S_PART else T3S_F):
        for v in T3S_V:
            a = f"T3S_{f}_{v}"
            M.FACE[f"t3s_{f}"], M.VOICE[f"t3s_{v}"], M.PAIRS[a] = T3S_F[f], T3S_V[v], (f"t3s_{f}", f"t3s_{v}")
            M.ARMS[a] = dict(aux=() if v in ("wsv", "rdn") else ("vox", "vb"), pairs=[a], seeds=NS3,
                             cfg=dict(M.t3r.SUB21A, face_dropout=.55 if f in ("b3s", "c5s") else .70))
            OFF[a] = 300
            if v in ("wsv", "rdn"):
                MAVONLY.add(a)
    M.N_SEEDS = max(M.N_SEEDS, NS3)                                   # CA keeps its seeds=3 (min(N_SEEDS, arm seeds))
    _so3, _ta3 = M.sources, M.train_arm

    def _sources3(R, T, main_rows, aux, device):
        main, out = _so3(R, T, main_rows, aux, device)
        TF, TV, TS = main_rows                                        # wsv / rdn exist for v4 + MAV v1 / v2 only: those branches train here, no aux
        _MAV["src"] = M.t3r.Source(TF, TV, TS, device, male=R.GMAP)
        R.log(f"T3S-01 MAV-only rows (wsv / rdn branches): {len(_MAV['src'].ids)} ids / {_MAV['src'].n_rows} rows")
        return main, out

    def _train_arm3(R, main, aux_src, arm, seed0, device, tlog):
        return _ta3(R, _MAV["src"] if arm in MAVONLY else main, aux_src, arm, seed0, device, tlog)
    M.sources, M.train_arm = _sources3, _train_arm3
l2n = FL.l2n
print(f"CLEANTUNE-01 {TAG}: BN42=1, framework {FW}, arms {list(M.ARMS)}" + (f", spec {SPEC}" if GROUP == "r2" else ""), flush=True)

# -- g3 extra voice blocks: samS4 (FLAG_103c, PCA 512 fitted on auxiliary rows only, TAP-01's), rn293 / ecapa1024 (ENC-SCALE-01's
#    arrays: MAV = flag2027-feats-screen2-p, auxiliary = FLAG_63w flag2027-aux-voice-feats-p)
_X, _P, _D = {}, {}, {}
STANDIN = os.environ.get("CT_STANDIN") == "1"


def _find(name):
    if name not in _D:
        c = [Path(p) for p in os.environ.get("CT_DIRS", "").split(os.pathsep) if p]
        if Path("/kaggle/input").exists():
            k = len(Path(name).parts)
            c += [Path(*f.parts[:len(f.parts) - k]) for f in Path("/kaggle/input").rglob(name)]
        hit = [d for d in dict.fromkeys(c) if (d / name).exists()]
        _D[name] = hit[0] if hit else None
        print(f"CLEANTUNE-01 {name}: {_D[name]}", flush=True)
        assert _D[name] is not None or STANDIN, f"attach the dataset holding {name} (CT_DIRS / /kaggle/input)"
    return _D[name]


def _find_all(name):
    c = [Path(p) for p in os.environ.get("CT_DIRS", "").split(os.pathsep) if p]
    if Path("/kaggle/input").exists():
        k = len(Path(name).parts)
        c += [Path(*f.parts[:len(f.parts) - k]) for f in Path("/kaggle/input").rglob(name)]
    return [d for d in dict.fromkeys(c) if (d / name).exists()]


def _wh_aux(src, V512):
    """WHISPER-02: whv3 (1280) of one auxiliary source, rows = voice_meta.csv; FLAG_160 lanes b + c merged (finite rows of each)."""
    ds = _find_all(f"{src}/voice_whv3.npy")
    if not ds:
        assert STANDIN, f"attach the FLAG_160 b + c datasets ({src}/voice_whv3.npy)"
        print(f"!! CLEANTUNE-01 SMOKE STAND-IN: {src} whv3", flush=True)
        return _fill(V512, XD["whv3"])
    X = None
    for d in ds:
        a = np.load(d / src / "voice_whv3.npy").astype(np.float32)
        if X is None:
            X = a
        else:
            assert a.shape == X.shape, (src, a.shape, X.shape)
            m = ~np.isfinite(X[:, 0]); X[m] = a[m]
    print(f"CLEANTUNE-01 whv3 {src}: {len(ds)} dataset(s), finite rows {int(np.isfinite(X[:, 0]).sum())} of {len(X)}", flush=True)
    return l2n(X)


def _wh_mav(part, a):
    d = _find("voice_whv3_train.npy")
    if d is None:
        return _fill(a, XD["whv3"])
    sub = ("v1_complete", "v2_complete")
    x = np.load(d / "voice_whv3_train.npy") if part == "v4" else np.concatenate([np.load(d / f"ext_{s}_voice_whv3.npy") for s in sub])
    x = l2n(x.astype(np.float32))
    assert x.shape == (len(a), XD["whv3"]) and np.isfinite(x).all(), (part, x.shape, len(a))
    return x


def _pca(d):
    """PCA 512 of samS4, fitted on 10,000 random finite rows of EACH auxiliary source after per-row L2; seed 0 (TAP-01's)."""
    if "samS4" not in _P:
        from sklearn.decomposition import PCA
        rng, X = np.random.default_rng(0), []
        for src in ("vox", "vox2b", "vb"):
            a = np.load(d / src / "voice_samS4.npy", mmap_mode="r")
            ok = np.flatnonzero(np.isfinite(np.asarray(a[:, 0], np.float32)))
            X.append(np.asarray(a[np.sort(rng.choice(ok, min(10000, len(ok)), replace=False))], np.float32))
        X = l2n(np.concatenate(X))
        pc = PCA(XD["samS4"], svd_solver="randomized", random_state=0).fit(X)
        _P["samS4"] = (pc.mean_.astype(np.float32), pc.components_.astype(np.float32))
        print(f"CLEANTUNE-01 PCA samS4: {len(X)} auxiliary rows, {X.shape[1]} -> {XD['samS4']}, variance kept "
              f"{pc.explained_variance_ratio_.sum():.3f}", flush=True)
    return _P["samS4"]


def _s4(files):
    d = _find("vb/voice_samS4.npy")
    mu, W = _pca(d)
    out = []
    for f in files:
        a = np.load(d / f, mmap_mode="r")
        for i in range(0, len(a), 8192):
            out.append((l2n(np.asarray(a[i:i + 8192], np.float32)) - mu) @ W.T)
    return l2n(np.concatenate(out).astype(np.float32))


def _fill(base, w):
    assert STANDIN, "stand-in blocks are for smoke tests only"
    return l2n(np.tile(base, (1, -(-w // base.shape[1])))[:, :w].astype(np.float32))


def aux_extra(src, V512):
    """[samS4 | rn293 | ecapa1024] of one auxiliary source (rows = its voice_meta.csv; NaN = not available -> row left out)."""
    if src not in _X:
        if _find("vb/voice_samS4.npy") is None or _find("vb/voice_rn293.npy") is None:
            print(f"!! CLEANTUNE-01 SMOKE STAND-IN: {src} extra voice blocks", flush=True)
            _X[src] = np.concatenate([_fill(V512, w) for w in XD.values()], 1)
        else:
            vf = _find("vb/voice_rn293.npy")
            man = json.loads((vf / "vfeat_manifest.json").read_text())
            assert man.get("complete") or STANDIN, f"FLAG_63w output incomplete: {man.get('coverage')}"
            _X[src] = np.concatenate([_s4([f"{src}/voice_samS4.npy"])] +
                                     [l2n(np.load(vf / src / f"voice_{n}.npy").astype(np.float32)) for n in ("rn293", "ecapa1024")], 1)
        if WHV3 and _X[src].shape[1] == sum(XD.values()) - XD["whv3"]:
            wh = _wh_aux(src, V512)                                   # every row the other blocks have must have whv3 (README WHISPER-02 §2)
            assert np.isfinite(wh[:, 0])[np.isfinite(_X[src][:, 0])].all() or STANDIN, f"{src}: whv3 does not cover the rows of the other blocks"
            _X[src] = np.concatenate([_X[src], wh], 1)
        if T3S and _X[src].shape[1] == sum(XD.values()) - XD["wsv"] - XD["rdn"]:   # T3S-01: no wsv / rdn for auxiliary rows (their branches use none); zeros keep every row
            _X[src] = np.concatenate([_X[src], np.zeros((len(V512), XD["wsv"] + XD["rdn"]), np.float32)], 1)
        assert _X[src].shape == (len(V512), sum(XD.values())), (src, _X[src].shape, len(V512))
        print(f"CLEANTUNE-01 {src}: {len(V512)} voice rows, with every extra block {np.isfinite(_X[src]).all(1).mean():.4f}", flush=True)
    return _X[src]


def mav_extra(part, a):
    """[samS4 | rn293 | ecapa1024] of the MAV rows of one part (v4 / ext = v1 then v2); every row must be finite."""
    if _find("vb/voice_samS4.npy") is None or _find("voice_rn293_train.npy") is None:
        print(f"!! CLEANTUNE-01 SMOKE STAND-IN: MAV {part} extra voice blocks", flush=True)
        return np.concatenate([_fill(a, w) for w in XD.values()], 1)
    sub = ("v1_complete", "v2_complete")
    s4 = _s4(["voice_samS4_train.npy"] if part == "v4" else [f"ext_{s}_voice_samS4.npy" for s in sub])
    sc = _find("voice_rn293_train.npy")
    ex = [s4]
    for n in ("rn293", "ecapa1024"):
        x = np.load(sc / f"voice_{n}_train.npy") if part == "v4" else np.concatenate([np.load(sc / f"ext_{s}_voice_{n}.npy") for s in sub])
        ex.append(l2n(x.astype(np.float32)))
    ex = np.concatenate(ex + ([_wh_mav(part, a)] if WHV3 else []) + (_scout_mav(part, a) if T3S else []), 1)
    assert ex.shape == (len(a), sum(XD.values())) and np.isfinite(ex).all(), (part, ex.shape, len(a))
    return ex


SCOUT = {"wsv": "wavlmsv", "rdn": "rdn6vox"}                          # T3S-01: SCOUT-01 feats_scout arrays (ngnnguynchn/flag2027-feats-scout-wsvrdn-p)


def _scout_mav(part, a):
    """T3S-01: [wsv | rdn] of the MAV rows of one part (v4 / ext = v1 then v2), rows as ENC-SCREEN-02's (same index_train.txt)."""
    out, sc = [], _find("voice_rn293_train.npy")
    for k, n in SCOUT.items():
        d = _find(f"voice_{n}_train.npy")
        if d is None:
            out.append(_fill(a, XD[k])); continue
        assert (d / "index_train.txt").read_text() == (sc / "index_train.txt").read_text(), f"{n}: train rows differ from ENC-SCREEN-02's"
        x = np.load(d / f"voice_{n}_train.npy") if part == "v4" else np.concatenate([np.load(d / f"ext_{s}_voice_{n}.npy") for s in ("v1_complete", "v2_complete")])
        out.append(l2n(x.astype(np.float32)))
        assert out[-1].shape == (len(a), XD[k]) and np.isfinite(out[-1]).all(), (k, part, out[-1].shape, len(a))
    return out


def _scout_test(name, V, idx):
    """T3S-01: [wsv | rdn] of one test list (index checked against the trial list)."""
    out = []
    for k, n in SCOUT.items():
        d = _find(f"voice_{n}_{name}.npy")
        if d is None:
            out.append(_fill(V, XD[k])); continue
        assert [l.split()[0] for l in (d / f"index_{name}.txt").read_text().splitlines() if l.strip()] == idx, f"{name}: {n} rows do not match"
        out.append(l2n(np.load(d / f"voice_{n}_{name}.npy").astype(np.float32)))
    return out


if FW == "g3":
    V1.aux_extra, V1.mav_extra = aux_extra, mav_extra                  # VOICELORA-01's hooks look them up at call time


def merge():
    """ct01_fold<s> = the SAFE-04 rows with tr<arm> of every group side by side (row order checked); B0c -> trB0c_<group>."""
    from lab_fuse import SP
    key = ["fold", "data", "protocol", "y", "fi", "vi"]
    for s in range(1, 6):
        out = None
        for g in GROUPS:
            P = pd.read_parquet(SP / "out/safe04" / f"ct01{g}_fold{s}.parquet")
            if out is None:
                out = P[key].copy()
            assert len(P) == len(out) and all((P[k].values == out[k].values).all() for k in ("y", "fi", "vi")), f"row order {g} fold {s}"
            for a in ARMS_OF[g]:
                out[f"trB0c_{g}" if a == "B0c" else f"tr{a}"] = P[f"tr{a}"].values
        out.to_parquet(SP / "out/safe04" / f"ct01_fold{s}.parquet")
        print(f"merged fold {s}: {len(out)} rows, {sum(c.startswith('tr') for c in out.columns)} arm columns", flush=True)


def report():
    """README §2: signal per arm vs its group's B0c (CA-FUSE, purged, g cells), noise floor g1 vs g2, then the round-2 plan."""
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    base = S6 + E.V11_T3
    S2, persons = E.stack02(), E.fold_persons()
    arms = {f"B0c_{g}": base + [f"trB0c_{g}"] for g in GROUPS}
    arms.update({a: base + [f"tr{a}"] for g in GROUPS for a in ARMS_OF[g][1:]})
    cap = {}
    out = SP / "out/safe04" / "ct01_report.txt"
    v = E.gate("ct01", arms, "B0c_g1", standalone=[st[-1] for st in arms.values()], out=out, ci_arms=["B0c_g1"],
               prep=lambda D: cap.setdefault("D", D))
    D = cap["D"]
    pg = lambda a_col, b_col: FL.paired_g(D, S2, persons, base + [a_col], base + [b_col])
    fmt = lambda p: ", ".join(f"f{int(s)} {g:+.2f}" for s, g in p.items()) + f" | mean {p.mean():+.2f}, {int((p > 0).sum())}/5"
    nf_p = pg("trB0c_g2", "trB0c_g1")
    nf = abs(float(nf_p.mean()))
    thr = max(.10, nf)
    txt = [f"\n== CLEANTUNE-01 round 1 (BN42=1, no CSLS)",
           f"noise floor B0c_g2 vs B0c_g1 (same rows, other run): {fmt(nf_p)} -> threshold max(.10, |mean|) = {thr:.2f}",
           f"reported only (g3 rows differ): B0c_g3 vs B0c_g1 {fmt(pg('trB0c_g3', 'trB0c_g1'))}"]
    res = {}
    for g in GROUPS:
        for a in ARMS_OF[g][1:]:
            p = pg(f"tr{a}", f"trB0c_{g}")
            hold = {k: round(v[a][k]["all"] - v[f"B0c_{g}"][k]["all"], 3) for k in ("hold_urdu", "hold_hindi")}
            sig = float(p.mean()) >= thr and int((p > 0).sum()) >= 3 and min(hold.values()) >= -.2
            res[a] = dict(group=g, delta=round(float(p.mean()), 3), wins=int((p > 0).sum()),
                          per_fold={int(s): round(float(x), 3) for s, x in p.items()}, hold=hold, signal=bool(sig))
            txt.append(f"{g} {a:6s} - B0c_{g} (g cells): {fmt(p)} | held-out urdu {hold['hold_urdu']:+.2f}, hindi "
                       f"{hold['hold_hindi']:+.2f} | signal {sig}")
    inp = [a for a in ARMS_OF["g3"][1:] if res[a]["signal"]]
    I = max(inp, key=lambda a: res[a]["delta"]) if inp else "B3c"
    SA = [a for a in ARMS_OF["g1"][1:] + ARMS_OF["g2"][1:] if res[a]["signal"]]
    for pair in CONFLICTS:
        if all(a in SA for a in pair):
            SA.remove(min(pair, key=lambda a: res[a]["delta"]))
    SA.sort(key=lambda a: -res[a]["delta"])
    plan = dict(threshold=round(thr, 3), noise_floor=round(nf, 3), I=I, S_A=SA, CMB_ALL=SA, CMB_TOP2=SA[:2],
                single=SA[:1], run_round2=bool(SA) or I != "B3c", results=res,
                note="CMB_* = the input I plus these knobs (README §2); I = B3c means the B0c face / voice blocks")
    txt.append(f"ROUND-2 PLAN (README §2): I* = {I} | S_A = {SA} | CMB-ALL = {I} + {SA} | CMB-TOP2 = {I} + {SA[:2]} | "
               f"run round 2: {plan['run_round2']}" + ("" if plan["run_round2"] else " -> final system = B0c (F0 only)"))
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    for d in (HERE, SP / "out/safe04"):
        (d / "ct_round2.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    (HERE / "report.txt").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def reportw():
    """WHISPER-02 (Experiment/WHISPER-02/README.md §3, registered before any result): CAW and CAWO against the in-run control CA (same
    rows, same seeds, same 3-seed offset), g cells of SAFE-04, purged, CA-FUSE C = 1. PASS = g >= +0.3, >= 4/5 folds, held-out >= -0.2."""
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    base = S6 + E.V11_T3
    S2, persons = E.stack02(), E.fold_persons()
    cap = {}

    def prep(D):
        for a in ("CA", "CAW", "CAWO"):
            D[f"tr{a}6"] = (D[f"tr{a}"] + D[f"tr{a}b"]) / 2           # 6-seed mean = mean of the two 3-seed means
        cap["D"] = D
        return D
    arms = {a: base + [f"tr{a}"] for a in ("CA", "CAW", "CAWO", "CA6", "CAW6")}
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    v = E.gate(TAG, arms, "CA", standalone=[st[-1] for st in arms.values()], out=out, ci_arms=["CAW"], prep=prep)
    D = cap["D"]
    pg = lambda a, b: FL.paired_g(D, S2, persons, arms[a], arms[b])
    hold = lambda a, b: {k: round(v[a][k]["all"] - v[b][k]["all"], 3) for k in ("hold_urdu", "hold_hindi")}
    fmt = lambda p: ", ".join(f"f{int(s)} {g:+.2f}" for s, g in p.items()) + f" | mean {p.mean():+.2f}, {int((p > 0).sum())}/5"
    txt = ["\n== WHISPER-02 (LID_STRICT, no CSLS): g cells, > 0 = first better"]
    res = {}
    for a, b in (("CAW", "CA"), ("CAWO", "CA"), ("CAW6", "CA6")):
        p, h = pg(a, b), hold(a, b)
        ok = float(p.mean()) >= .3 and int((p > 0).sum()) >= 4 and min(h.values()) >= -.2
        res[a] = dict(vs=b, g=round(float(p.mean()), 3), wins=int((p > 0).sum()), per_fold={int(s): round(float(x), 3) for s, x in p.items()}, hold=h, passed=bool(ok))
        txt.append(f"{a} - {b}: {fmt(p)} | held-out {h} -> {'PASS' if ok else 'FAIL'} (g >= +0.3, >= 4/5, held-out >= -0.2)")
    txt.append("CAW - CA per g cell and fold\n" + _per_cell(D, S2, persons, arms["CAW"], arms["CA"]).round(2).to_string())
    dec = dict(rule="WHISPER-02 README §3", passed=res["CAW"]["passed"], final_arm="CAW", final_spec=["C5CAT4W", list(PLAN["S_A"])], seeds=10,
               results=res, note="CAWO / CAW6 are reported only; the CB one-shot goes to CAW 10 seeds if passed")
    txt.append(f"DECISION: {json.dumps(dec)}")
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    for d in (HERE, SP / "out/safe04"):
        (d / "wh02_decision.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
    (HERE / "reportw.txt").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def _per_cell(D, S2, persons, a_cols, b_cols):
    """CA-FUSE(a) vs CA-FUSE(b), purged per fold: EER(b) - EER(a) per (fold, g cell); > 0 = a better (TAP-01's)."""
    from lab_fuse import table
    D = D.copy(); D["a_"], D["b_"] = np.nan, np.nan
    for s in sorted(D.fold.unique()):
        te = D[D.fold == s]; pool = S2.purge(D[D.fold != s], persons[int(s)])
        dz = S2.Design(pool, te, list(dict.fromkeys(a_cols + b_cols)))
        D.loc[te.index, "a_"], D.loc[te.index, "b_"] = dz.score(a_cols, 1.0), dz.score(b_cols, 1.0)
    T_ = table(D, {"a": lambda g: g["a_"].values, "b": lambda g: g["b_"].values})
    T_ = T_[T_.cell.str.endswith(" g")]
    return (T_["b"] - T_["a"]).groupby([T_.cell, T_.fold]).mean().unstack("fold")


def report2():
    """README §2 round 2: SEL2 (nested over the candidates) vs B0c (fresh seeds), STACK, SEED -> ct_decision.json."""
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    base = S6 + E.V11_T3
    S2, persons = E.stack02(), E.fold_persons()
    cap = {}

    def prep(D):
        ch, det = C1.nested_choice(D, S2, persons, tuple(CANDS), base)
        D["trSEL2"] = np.nan
        for s, a in ch.items():
            D.loc[D.fold == s, "trSEL2"] = D.loc[D.fold == s, f"tr{a}"]
        for a in CANDS:                                               # 6-seed mean = mean of the two 3-seed means (CT20)
            D[f"tr{a}6"] = (D[f"tr{a}"] + D[f"tr{a}b"]) / 2
        cap.update(D=D, choice=ch, detail=det)
        return D
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    D0 = E.safe04(TAG)                                                # the STACK columns need the choice first: compute it once here
    prep(D0)
    chosen = max(set(cap["choice"].values()), key=list(cap["choice"].values()).count)
    stack_extra = ["trB0c"] + (["trI"] if "I" in SPEC else [])
    arms = {"B0c": base + ["trB0c"], "SEL2": base + ["trSEL2"], "STACK": base + ["trSEL2"] + stack_extra,
            **{a: base + [f"tr{a}"] for a in CANDS}, f"{chosen}6": base + [f"tr{chosen}6"]}
    v = E.gate(TAG, arms, "SEL2", standalone=[st[-1] for st in arms.values()], out=out, ci_arms=["SEL2", "STACK"], prep=prep)
    D = cap["D"]
    pg = lambda a, b: FL.paired_g(D, S2, persons, arms[a], arms[b])
    hold = lambda a, b: {k: round(v[a][k]["all"] - v[b][k]["all"], 3) for k in ("hold_urdu", "hold_hindi")}
    fmt = lambda p: ", ".join(f"f{int(s)} {g:+.2f}" for s, g in p.items()) + f" | mean {p.mean():+.2f}, {int((p > 0).sum())}/5"
    txt = [f"\n== CLEANTUNE-01 round 2 (BN42=1, fresh seeds, no CSLS): spec {SPEC}",
           f"SEL2 choice per fold {cap['choice']} (inner gains {json.dumps({int(k): {q: round(x, 3) for q, x in d.items()} for k, d in cap['detail'].items()})})"]
    for a in CANDS:
        txt.append(f"{a:4s} - B0c (g cells): {fmt(pg(a, 'B0c'))} | held-out {hold(a, 'B0c')}")
    P = _per_cell(D, S2, persons, arms["SEL2"], arms["B0c"])
    txt.append("SEL2 - B0c per g cell and fold (> 0 = SEL2 better)\n" + P.round(2).to_string())
    p, h = pg("SEL2", "B0c"), hold("SEL2", "B0c")
    ok = float(p.mean()) >= .3 and int((p > 0).sum()) >= 4 and min(h.values()) >= -.2
    txt.append(f"RULE system (SEL2 - B0c): g {float(p.mean()):+.2f} (>= +0.3), {int((p > 0).sum())}/5 folds (>= 4), held-out {h} "
               f"(>= -0.2) -> {'PASS' if ok else 'FAIL'}")
    ref = "SEL2" if ok else "B0c"
    ps, hs = pg("STACK", ref), hold("STACK", ref)
    sok = float(ps.mean()) >= .3 and int((ps > 0).sum()) >= 4 and min(hs.values()) >= -.2
    txt.append(f"RULE STACK (STACK = SEL2 + {stack_extra} - {ref}): {fmt(ps)}, held-out {hs} -> {'PASS' if sok else 'FAIL'}")
    p6 = pg(f"{chosen}6", chosen)
    seeds = 10 if float(p6.mean()) >= 0 else 5
    txt.append(f"RULE SEED ({chosen}: 6 - 3 seeds): {fmt(p6)} -> final head {seeds} seeds")
    final_arm = chosen if ok else "B0c"
    dec = dict(system_pass=ok, final_arm=final_arm, final_spec=SPEC.get(final_arm), stack_pass=sok, stack_streams=stack_extra if sok else [],
               stack_arm=chosen if sok else None,                     # STACK = SEL2 (this arm) + stack_streams, also when system_pass fails
               seeds=seeds, choice={int(k): a for k, a in cap["choice"].items()}, sel2_g=round(float(p.mean()), 3),
               f1=bool(ok or sok), note="F1 only when system_pass or stack_pass (README §3); else F0 (B0c) is the system")
    if os.environ.get("CT_PLAN"):                                     # 3/11 (CLEANTUNE-02 README §1b): F2s vs F1s on the same tables
        f1s = next((a for a, v in SPEC.items() if v[0] == "C5CAT4" and set(v[1]) == {"T", "AIp"}), None)
        if f1s and final_arm not in ("B0c", f1s):
            pf, hf = pg(final_arm, f1s), hold(final_arm, f1s)
            okf = float(pf.mean()) >= .3 and int((pf > 0).sum()) >= 4 and min(hf.values()) >= -.2
            dec["vs_f1s"] = dict(f1s_arm=f1s, g=round(float(pf.mean()), 3), wins=int((pf > 0).sum()), hold=hf, passes=bool(okf))
            txt.append(f"RULE F2s ({final_arm} - {f1s} = F1s recipe): {fmt(pf)}, held-out {hf} -> {'PASS' if okf else 'FAIL'}")
        elif f1s:
            dec["vs_f1s"] = dict(f1s_arm=f1s, same_recipe=final_arm == f1s, passes=False)
            txt.append(f"F2s check: final arm {final_arm} {'IS the F1s recipe: CLEANTUNE-02 confirms F1s' if final_arm == f1s else '= B0c'}")
    txt.append(f"DECISION: {json.dumps(dec)}")
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    for d in (HERE, SP / "out/safe04"):
        (d / "ct_decision.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
    (HERE / "report2.txt").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def patch_test_cat4():
    """2/11 (F1 = CA): e2e_lib.t3_split (test voice [sam100p | sam100f], 512) -> the 1472 columns of the g3 rows: samS4 slot zero
    (v_cat4 skips it), then [rn293 | ecapa1024] of the test lists from ENC-SCREEN-02 as ENC-SCALE-01's patch_test_voice (ids checked)."""
    import e2e_lib as E
    split0 = E.t3_split

    def t3_split(c):
        F, V, m = split0(c)
        name, S = c["key"].replace("/", "_"), _find("voice_rn293_train.npy")
        idx = [l.split()[0] for l in (S / f"index_{name}.txt").read_text().splitlines() if l.strip()]
        assert len(idx) == len(V) and list(map(str, c["pid"])) == idx, f"{c['key']}: ENC-SCREEN-02 rows do not match the trial list"
        ex = [l2n(np.load(S / f"voice_{n}_{name}.npy").astype(np.float32)) for n in ("rn293", "ecapa1024")]
        if WHV3:
            Sw = _find(f"voice_whv3_{name}.npy")
            if Sw is None:
                ex.append(_fill(V, XD["whv3"]))
            else:
                ix = Sw / f"index_{name}.txt"
                if ix.exists():
                    assert [l.split()[0] for l in ix.read_text().splitlines() if l.strip()] == idx, f"{c['key']}: whv3 rows do not match the trial list"
                ex.append(l2n(np.load(Sw / f"voice_whv3_{name}.npy").astype(np.float32)))
        if T3S:
            ex += _scout_test(name, V, idx)
        V = np.concatenate([V, np.zeros((len(V), XD["samS4"]), np.float32)] + ex, 1)
        assert V.shape[1] == 512 + sum(XD.values()) and np.isfinite(V).all(), (c["key"], V.shape)
        return F, V, m

    E.t3_split = t3_split


def final():
    """FINAL_ARM (default B0c) on every train person, CT_FINAL_SEEDS (5) seeds, test streams of the split (T3RECIPE-01 final)."""
    arm = os.environ.get("FINAL_ARM", "B0c")
    ns = int(os.environ.get("CT_FINAL_SEEDS", "5"))
    M.N_SEEDS = max(M.N_SEEDS, ns)
    M.ARMS[arm] = dict(M.ARMS[arm], seeds=ns)
    M.DECISION = FL.DECISION = arm
    if FW == "g3":
        F5.DECISION = V1.DECISION = arm
        assert M.PAIRS[arm][1] in ("v_pf", "v_cat4", "v_cat4w", "v_wh"), "a TAPv final needs the samS4 block of the test lists: not built (README §3)"
        FL.patch(); F5.patch_test(); V1.patch_voice()
        if M.PAIRS[arm][1] in ("v_cat4", "v_cat4w", "v_wh"):
            patch_test_cat4()
    else:
        FL.patch(); FL.patch_test()
    print(f"CLEANTUNE-01 final head: {arm} ({M.PAIRS[arm]}), {ns} seeds, cfg {M.ARMS[arm]['cfg']}", flush=True)
    M.final()


def fit_en_clean(col):
    """CLEAN-EN-01's English fuser (logistic C = 1 on [col | ImageBind | age] x conditions) on EN-T3-01's English validation table
    with `col` of this run's SAFE-04 tables attached per fold (as CLEAN-EN-01 table_b3)."""
    import e2e_lib as E
    from lab_fuse import SP
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    EN = _load("ent301", "EN-T3-01/run.py")
    D = EN.table_en()
    key = ["fold", "data", "y", "vi", "fi"]
    B = pd.concat([pd.read_parquet(SP / f"out/safe04/{TAG}_fold{s}.parquet", columns=["data", "y", "vi", "fi", col]).assign(fold=s)
                   for s in range(1, 6)], ignore_index=True).drop_duplicates(key)
    n = len(D)
    D = D.merge(B, on=key, how="left", validate="many_to_one")
    assert len(D) == n and np.isfinite(D[col]).all(), f"{col} attach"
    m = E.mav_flagged(D) if "data" in D else np.zeros(len(D), bool)   # 5/11: no fuser fitted on a Whisper-flagged MAV voice (LID_STRICT)
    if m.any():
        print(f"English fuser: {int(m.sum())} pairs with a Whisper-flagged MAV voice left out (LID_STRICT)", flush=True)
        D = D[~m].reset_index(drop=True)
    st = [col, "CTRL_ib", "CTRL_age"]
    X = E.design(D[st].values, D[EN.CO].values); sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), D.y.values)
    print(f"English fuser CLEAN[{col}]: {X.shape[1]} inputs on {len(D)} trials {D.cell.value_counts().to_dict()}", flush=True)
    return dict(streams=st, conditions=EN.CO, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                intercept=float(lr.intercept_[0]), n_trials=int(len(D)), C=1.0)


def zip_ct():
    """F0 (FINAL_ARM = B0c, no F0_ZIP): Bangla = B3's fuser form (S6 + v11 T3 + trB0c) fitted on this run's tables, English =
    CLEAN-EN-01's form on trB0c. F1 (F0_ZIP set): Bangla of FINAL_ARM, English files byte-identical to F0 (README §3). A passing
    STACK needs the test streams of every stacked head: not built here (only if it passes)."""
    import zipfile
    import e2e_lib as E
    from stream_gate import S6
    arm = os.environ.get("FINAL_ARM", "B0c")
    col, od = f"tr{arm}", M.out_dir()
    E.check_v11_fuser()
    Z = np.load(od / f"{TAG}_final_{E.SPLIT}.npz")
    comps = E.components()
    new = {i: np.asarray(Z[f"{col}_{i}"], np.float64) for i in comps}
    assert all(len(new[i]) == len(c["pid"]) and np.isfinite(new[i]).all() for i, c in comps.items())
    bn = E.fit_fuser(TAG, S6 + E.V11_T3 + [col])
    scores = E.bangla_scores(bn, {i: v for i, v in new.items() if not comps[i]["is_en"]})
    f0 = os.environ.get("F0_ZIP")
    zp = od / f"submission_cleantune01_{'F1_' + arm if f0 else 'F0'}_{E.SPLIT}.zip"
    lines = [f"Bangla fuser S6 + v11 T3 + {col} fitted on {TAG} tables"]
    if not f0:
        assert arm == "B0c", "F0 is the B0c head"
        en = fit_en_clean(col)
        for i, c in comps.items():
            if c["is_en"]:
                Zc = np.column_stack([new[i], c["en7"][:, 4], c["en7"][:, 5]])
                fn = lambda r, Zc=Zc, C=c["enco"]: E.apply_fuser(en, Zc if r is None else Zc[r], C if r is None else C[r])
                scores[i], dl0 = E.level0_selftest(fn, len(c["pid"]))
                assert dl0 < 1e-9 and np.isfinite(scores[i]).all(), (c["key"], dl0)
                lines.append(f"{c['key']}: English CLEAN[{col}], level-0 half-file |diff| {dl0:.1e}")
        (od / "cleantune01_fuser_english.json").write_text(json.dumps(en, indent=1))
    else:
        for i, c in comps.items():
            if c["is_en"]:
                scores[i] = np.zeros(len(c["pid"]))                   # replaced by F0's bytes below
    E.write_zip(zp, scores)
    if f0:                                                            # English files: F0's bytes
        tmp = zp.with_suffix(".tmp.zip")
        with zipfile.ZipFile(zp) as a, zipfile.ZipFile(f0) as b, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as w:
            for c in comps.values():
                w.writestr(c["zip_name"], b.read(c["zip_name"]) if c["is_en"] else a.read(c["zip_name"]))
        tmp.replace(zp)
    (od / f"cleantune01_fuser_bangla_{arm}.json").write_text(json.dumps(bn, indent=1))
    rep = E.zip_report(zp, "v11")
    ref = FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None
    lines += [f"-> {zp.name}: {rep}", f"correlation per file with the CB 20.18 zip: {ref}"]
    print("\n".join(lines), flush=True)
    (od / f"cleantune01_zip_{arm}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- T3S-01 (Experiment/T3S-01/README.md §2-3): per-face job tables merged, the branch LR nested in fold, gate, final, zip
def _t3s_groups(cols):
    br = sorted(c for c in cols if c.startswith("trT3S_") and c.count("_") == 2)
    return {"trT3S": br, "trT3Sw": [c for c in br if c.endswith("_wsv")], "trT3Sr": [c for c in br if c.endswith("_rdn")]}


def _lr(trn, cols):
    """The T3S mixer: logistic C = 1 on the standardised branch columns of `trn` -> scoring function (decision value)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(trn[cols].values)
    m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(trn[cols].values), trn.y.values)
    return lambda df: m.decision_function(sc.transform(df[cols].values))


def merge_t3s():
    """ct01r2<face>_fold<s> (one table per face job) side by side -> ct01r2_fold<s> (row order checked)."""
    from lab_fuse import SP
    key = ["fold", "data", "protocol", "y", "fi", "vi"]
    for s in range(1, 6):
        parts = sorted((SP / "out/safe04").glob(f"{TAG}[a-z]*_fold{s}.parquet"))
        assert {p.name[len(TAG):].split("_fold")[0] for p in parts} == set(T3S_F), f"fold {s}: job tables {[p.name for p in parts]}"
        out = None
        for p in parts:
            P = pd.read_parquet(p)
            if out is None:
                out = P[key].copy()
            assert len(P) == len(out) and all((P[k].values == out[k].values).all() for k in ("y", "fi", "vi")), f"row order {p.name}"
            for c in P.columns:
                if c.startswith("tr"):
                    out[c] = P[c].values
        out.to_parquet(SP / "out/safe04" / f"{TAG}_fold{s}.parquet")
        print(f"T3S-01 merged fold {s}: {len(parts)} job tables, {len(out)} rows, {sum(c.startswith('tr') for c in out.columns)} columns", flush=True)


def _with_t3s(D, S2, persons):
    """trT3S / trT3Sw / trT3Sr = the branch LR fitted on the purged other folds (fixed columns: standalone table, final fuser rows)."""
    import e2e_lib as E
    ok = ~E.mav_flagged(D)                                            # 5/11: LR training rows without a Whisper-flagged MAV voice (LID_STRICT)
    for c, cols in _t3s_groups(D.columns).items():
        D[c] = np.nan
        for s in sorted(D.fold.unique()):
            te = (D.fold == s).values
            D.loc[te, c] = _lr(S2.purge(D[~te & ok], persons[int(s)]), cols)(D[te])
    return D


def _nested_design(S2, persons, groups):
    """S2.Design that replaces a T3S column by its value nested in the split it is given (README §1 'học lồng trong fold'): training
    rows get the LR of the other inner folds (purged of the inner fold's persons), test rows the LR of every training row."""
    D0, cache = S2.Design, {}

    class Design(D0):
        def __init__(self, trn, tst, streams):
            hit = [c for c in streams if c in groups]
            if hit:
                keep = list(dict.fromkeys(list(streams) + S2.CO + ["y", "fold", "fp", "vp"] + [b for c in hit for b in groups[c]]))
                trn, tst = trn[keep].copy(), tst[keep].copy()
                for c in hit:
                    k = (c, len(trn), int(trn.index.values.sum()), len(tst), int(tst.index.values.sum()))
                    if k not in cache:
                        a = np.full(len(trn), np.nan)
                        for t in sorted(trn.fold.unique()):
                            it = (trn.fold == t).values
                            a[it] = _lr(S2.purge(trn[~it], persons[int(t)]), groups[c])(trn[it])
                        cache[k] = (a, _lr(trn, groups[c])(tst))
                    trn[c], tst[c] = cache[k]
            super().__init__(trn, tst, streams)
    return Design


def reportt3s():
    """T3S-01 README §2-3 (registered 4/11 before any result): F1s = CA-FUSE(S6 + v11 T3 + trCA) of the in-run control CA; F1s + T3S
    adds trT3S (LR of the 24 branches, nested in fold). PASS = (F1s + T3S) - F1s on the g cells >= +0.3, >= 4/5 folds, held-out
    Urdu / Hindi >= -0.2. Reported only: wsv / rdn branches alone, T3S in place of CA, standalone EER, error correlation T3S / CA."""
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    merge_t3s()
    base = S6 + E.V11_T3
    S2, persons = E.stack02(), E.fold_persons()
    G = _t3s_groups(pd.read_parquet(SP / "out/safe04" / f"{TAG}_fold1.parquet").columns)
    assert len(G["trT3S"]) == len(T3S_F) * len(T3S_V), G["trT3S"]
    S2.Design = _nested_design(S2, persons, G)
    E.stack02 = lambda: S2                                            # E.gate builds its CA-FUSE designs with the nested T3S columns
    cap = {}

    def prep(D):
        cap["D"] = _with_t3s(D, S2, persons)
        return cap["D"]
    arms = {"F1s": base + ["trCA"], "F1sT": base + ["trCA", "trT3S"], "F1sTw": base + ["trCA", "trT3Sw"],
            "F1sTr": base + ["trCA", "trT3Sr"], "T3Sonly": base + ["trT3S"]}
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    v = E.gate(TAG, arms, "F1sT", standalone=["trCA", "trT3S", "trT3Sw", "trT3Sr"] + G["trT3S"], out=out, ci_arms=["F1sT"], prep=prep)
    D = cap["D"]
    pg = lambda a, b: FL.paired_g(D, S2, persons, arms[a], arms[b])
    hold = lambda a, b: {k: round(v[a][k]["all"] - v[b][k]["all"], 3) for k in ("hold_urdu", "hold_hindi")}
    fmt = lambda p: ", ".join(f"f{int(s)} {g:+.2f}" for s, g in p.items()) + f" | mean {p.mean():+.2f}, {int((p > 0).sum())}/5"
    txt = ["\n== T3S-01 (LID_STRICT, no CSLS, no AVSpeech / LRS3; wsv / rdn branches on v4 + MAV only): g cells, > 0 = first better"]
    res = {}
    for a in ("F1sT", "F1sTw", "F1sTr", "T3Sonly"):
        p, h = pg(a, "F1s"), hold(a, "F1s")
        ok = float(p.mean()) >= .3 and int((p > 0).sum()) >= 4 and min(h.values()) >= -.2
        res[a] = dict(vs="F1s", g=round(float(p.mean()), 3), wins=int((p > 0).sum()), per_fold={int(s): round(float(x), 3) for s, x in p.items()},
                      hold=h, passed=bool(ok))
        txt.append(f"{a} - F1s: {fmt(p)} | held-out {h} -> {'PASS' if ok else 'FAIL'}" + ("   <- DECISION" if a == "F1sT" else "   (reported only)"))
    gc = sorted(c for c in D.cell.unique() if c.endswith(" g"))
    lo, hi = S2.boot(D.rename(columns={"F1s|oof": "ref_", "F1sT|oof": "cand_"}), "ref_", "cand_", gc, "both")
    txt.append(f"CI95 (person bootstrap) of (F1s + T3S) - F1s on the g cells: [{lo:+.2f}, {hi:+.2f}]")
    txt.append("(F1s + T3S) - F1s per g cell and fold\n" + _per_cell(D, S2, persons, arms["F1sT"], arms["F1s"]).round(2).to_string())
    ec = {}
    for cell, g in D.groupby("cell"):                                  # error correlation = within-class correlation of the two scores
        r = [np.corrcoef(g.loc[g.y == y, "trT3S"], g.loc[g.y == y, "trCA"])[0, 1] for y in (0, 1) if (g.y == y).sum() > 2]
        ec[cell] = round(float(np.mean(r)), 3)
    txt.append(f"error correlation trT3S / trCA (within-class Pearson, OOF): {ec}")
    dec = dict(rule="T3S-01 README §3", passed=res["F1sT"]["passed"], results=res, ci95=[round(float(lo), 3), round(float(hi), 3)],
               error_corr=ec, branches=G["trT3S"], note="F1sTw / F1sTr / T3Sonly reported only; passed -> final F1sT + U2 = F1sT + RFS, one CB each")
    txt.append(f"DECISION: {json.dumps(dec)}")
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    for d in (HERE, SP / "out/safe04"):
        (d / "t3s_decision.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
    (HERE / "report_t3s.txt").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")


def final_t3s():
    """T3S-01 README §3 (only after the gate passes): the branches of this job (T3S_PART) on every train person, NS3 seeds, test
    streams of the split -> t3s_<arm>_final_<split>.npz (T3RECIPE-01 final, one call per branch). F1s's CA head is reused (zip_t3s)."""
    FL.patch(); F5.patch_test(); V1.patch_voice(); patch_test_cat4()
    for a in [a for a in M.ARMS if a.startswith("T3S_")]:
        M.DECISION = FL.DECISION = F5.DECISION = V1.DECISION = a
        M.TAG = f"t3s_{a}"
        print(f"T3S-01 final branch {a}: {M.PAIRS[a]}, {M.ARMS[a]['seeds']} seeds, aux {M.ARMS[a]['aux']}, cfg {M.ARMS[a]['cfg']}", flush=True)
        M.final()


def zip_t3s():
    """T3S-01 README §3: the F1sT zip. Bangla = CA-FUSE(S6 + v11 T3 + trCA + trT3S) refitted on every SAFE-04 row of this run's tables
    (trT3S rows = the purged other-fold LR); test trCA = F1s's final streams (its ct01r2_final_<split>.npz, attached), test trT3S = the
    branch LR refitted on every row, applied to t3s_<arm>_final_<split>.npz. English files = F1s's bytes (its submission zip, attached)."""
    import zipfile
    import e2e_lib as E
    from stream_gate import S6
    od = M.out_dir()
    merge_t3s()
    S2, persons = E.stack02(), E.fold_persons()
    s0 = E.safe04
    E.safe04 = lambda tag: _with_t3s(s0(tag), S2, persons)            # E.fit_fuser reads the tables through safe04
    D = E.safe04(TAG)
    br = _t3s_groups(D.columns)["trT3S"]
    mix = _lr(D[~E.mav_flagged(D)], br)                               # 5/11: no mixer fitted on a Whisper-flagged MAV voice (LID_STRICT)
    comps = E.components()
    Z = np.load(_find(f"ct01r2_final_{E.SPLIT}.npz") / f"ct01r2_final_{E.SPLIT}.npz")
    new = {}
    for i, c in comps.items():
        if not c["is_en"]:
            B = pd.DataFrame({b: np.load(od / f"t3s_{b[2:]}_final_{E.SPLIT}.npz")[f"{b}_{i}"] for b in br})
            new[i] = np.column_stack([Z[f"trCA_{i}"], mix(B)])
            assert new[i].shape == (len(c["pid"]), 2) and np.isfinite(new[i]).all(), c["key"]
    bn = E.fit_fuser(TAG, S6 + E.V11_T3 + ["trCA", "trT3S"])
    scores = E.bangla_scores(bn, new)
    scores.update({i: np.zeros(len(c["pid"])) for i, c in comps.items() if c["is_en"]})   # replaced by F1s's bytes below
    zp = od / f"submission_t3s01_F1sT_{E.SPLIT}.zip"
    E.write_zip(zp, scores)
    n1 = f"submission_cleantune01_F1_CA_{E.SPLIT}.zip"
    d1, tmp = (_find_all(n1) or [None])[0], zp.with_suffix(".tmp.zip")   # _find asserts when nothing is found
    if d1 is not None:
        f1 = d1 / n1
    else:   # 5/11: Kaggle unzips a zip inside an uploaded dataset -> rebuild the zip from the folder of its 4 txt files
        d1 = _find(n1[:-4] + "/gender/sub_score_v4_English_heard.txt")
        assert d1 is not None, f"{n1}: neither the zip nor its extracted folder is attached"
        x1, f1 = d1 / n1[:-4], od / f"f1s_input_{n1}"
        with zipfile.ZipFile(f1, "w", zipfile.ZIP_DEFLATED) as w1:
            for c in comps.values():
                w1.write(x1 / c["zip_name"], c["zip_name"])
        print(f"F1s zip rebuilt from the extracted folder {x1}", flush=True)
    rd1 = lambda n, f1=f1: zipfile.ZipFile(f1).read(n)
    with zipfile.ZipFile(zp) as a, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as w:
        for c in comps.values():
            w.writestr(c["zip_name"], rd1(c["zip_name"]) if c["is_en"] else a.read(c["zip_name"]))
    tmp.replace(zp)
    (od / "t3s01_fuser_bangla.json").write_text(json.dumps(dict(bangla=bn, t3s_branches=br), indent=1))
    rep = E.zip_report(zp, "v11")
    lines = [f"Bangla fuser S6 + v11 T3 + trCA + trT3S fitted on {TAG} tables; trCA test = F1s ({f1.parent}); English = F1s bytes",
             f"-> {zp.name}: {rep}", f"correlation per file with the F1s zip: {FL.cmp_zip(zp, f1)}"]
    print("\n".join(lines), flush=True)
    (od / "t3s01_zip.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit():
    """README §1 S0c (A0) on the rows the heads train on (the training functions themselves): no Bengali-flagged speaker (FLAG_42,
    FLAG_53, VoxBlink lists), every adapter array marked bn42_clean.json, no AVSpeech / LRS3 input attached. Fails loudly."""
    (V1.patch_voice if FW == "g3" else FL.patch)()
    import run as T3
    d42, d53 = T3.dropped_speakers(), C1.drop53()
    strict = os.environ.get("LID_STRICT") == "1"                       # LIDSTRICT-01 (2/11): bn + as rule
    assert len(d42) == (T3.strict_counts()[0] if strict else 25) and d42 <= d53, "BN42 must merge FLAG_42's list into drop53()"
    VB = M.vb_module().VB
    dvb = set(json.loads((VB / "lid_summary.json").read_text())["speakers_dropped_ids"]) | T3.strict_drop("vb")
    _, _, S2 = T3.vox_rows(T3.t3("face", "vox"), T3.t3("voice", "vox"))
    S2 = S2[~np.isin(S2, ["vox2:" + x for x in d53])]                  # = FACELORA-03 sources_fl3 on the FLAG_37 rows
    _, _, S3 = FL.new_rows4(("new", "more"), print)
    _, _, S4 = FL.vb_rows4(print)
    vox = {s.split(":", 1)[1] for s in np.concatenate([S2, S3])}
    vb = {s.split(":", 1)[1] for s in S4}
    bad = sorted(vox & (d42 | d53)) + sorted(vb & dvb)
    mk = {a: FL.adir(a) / "bn42_clean.json" for a in sorted(FL.LOAD)}
    marks = {a: json.loads(p.read_text()) if p.exists() else None for a, p in mk.items()}
    inp = Path("/kaggle/input")
    names = [p.name for p in inp.iterdir()] + [q.name for p in inp.iterdir() if p.is_dir() for q in p.iterdir()] if inp.exists() else []
    avs = sorted({n for n in names if "avs" in n.lower() or "lrs3" in n.lower()})
    lines = [f"A0 audit ({TAG}): VoxCeleb2 speakers {len(vox)} (FLAG_37 + FLAG_53), VoxBlink {len(vb)}; Bengali-flagged among them "
             f"{len(bad)} {bad[:5]}; lists FLAG_42 {len(d42)}, FLAG_53+42 {len(d53)}, VoxBlink {len(dvb)}",
             f"A0 adapters: {marks}", f"A0 inputs named avs / lrs3: {avs}",
             "A0 by construction: S6 = MAV-only members (BEST/v7_safe), v11 T3 head built with VOX_LID (FLAG_42 filter), ImageBind / "
             "age frozen pretrained; fusers fitted on SAFE-04 validation rows only (no dev / test row, no test statistic, no CSLS)"]
    print("\n".join(lines), flush=True)
    (M.out_dir() / f"{TAG}_audit_A0.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = not bad and not avs and all(m and m.get("exp") == "CLEANTUNE-01" for m in marks.values())
    assert ok or os.environ.get("BN42_STANDIN") == "1", "A0 FAILED"
    print(f"A0 {'PASSED' if ok else 'not passed (smoke stand-in: unmarked adapters)'}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "audit":
        audit()
    elif cmd == "merge":
        merge()
    elif cmd == "report":
        (reportt3s() if T3S else reportw() if WHV3 else report2()) if GROUP == "r2" else report()
    elif cmd == "final":
        final_t3s() if T3S else final()
    elif cmd == "zip":
        zip_t3s() if T3S else zip_ct()
    else:
        (V1.patch_voice if FW == "g3" else FL.patch)()
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            M.run(s)
