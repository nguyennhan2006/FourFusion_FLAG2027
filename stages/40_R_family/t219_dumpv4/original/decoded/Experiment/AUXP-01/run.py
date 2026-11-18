r"""AUXP-01 (registered 13/10 in README.md before any result): AVSpeech (COLLECT-01) and LRS3 (COLLECT-02) pseudo-identities as
two more AUXILIARY sources (own AAM heads) in the CC recipe on SCALE-CC-02's data D4, re-tuned for the larger data (user rule
CLAUDE.md 4.7). Every arm = SUB21A + dropout .55 / .30 (SCALE-CC-02 V4, the tuned recipe on D4), 1 seed, the same seed:
  B   aux vox + vb (control, tuned on the old data)      V0  aux vox + vb + avs + lrs3
  V1  + aux_ids 64    V2  + epochs 80    V3  + aux_weight .5    V4  + aux_ids 64, ids_per_batch 64, epochs 80
SEL = nested choice among V0-V4 (SCALE-CC-01's nested_choice). SCALE-CC-02's run.py is loaded as a module.
    FOLDS=1,2,3,4,5 python run.py ; python run.py report ; E2E_DIR=... python run.py final ; python run.py zip
Env: TR_SEEDS (5, final head), TR_MAX_STEPS (smoke tests only), AUXP_SOURCES (default avs,lrs3; smoke tests may drop one).
"""
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
_s = importlib.util.spec_from_file_location("scalecc02run", EXP / "SCALE-CC-02/run.py")
C2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(C2)
C1, M = C2.C1, C2.M
M.TAG = C1.TAG = TAG = "auxp01"                                       # C1's report / chosen / use_final read C1.TAG
TUNED = dict(face_dropout=.55, voice_dropout=.30)                     # SCALE-CC-02 V4
NEW = tuple(x for x in os.environ.get("AUXP_SOURCES", "avs,lrs3").split(",") if x)
C1.VARIANTS = VARIANTS = {"V0": {}, "V1": dict(aux_ids=64), "V2": dict(epochs=80), "V3": dict(aux_weight=.5),
                          "V4": dict(aux_ids=64, ids_per_batch=64, epochs=80)}
M.ARMS = {"B": dict(aux=("vox", "vb"), pairs=["CC"], seeds=1, cfg=dict(M.t3r.SUB21A, **TUNED)),
          **{v: dict(aux=("vox", "vb") + NEW, pairs=["CC"], seeds=1, cfg=dict(M.t3r.SUB21A, **TUNED, **c)) for v, c in VARIANTS.items()}}
M.DECISION = "V0"                                                     # extra columns only; the decision is SEL (report)
M.BASE_CFG = M.t3r.SUB21A
M.N_SEEDS = int(os.environ.get("TR_SEEDS", "5"))
_P = {}
l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def feats_dir(name):
    """feats_<name>/ of the collection notebook: Experiment/COLLECT-0x/ locally, or any attached notebook output on Kaggle."""
    roots = [EXP / "COLLECT-01", EXP / "COLLECT-02", Path(os.environ.get("FLAG_IN", "/kaggle/input")), Path("/kaggle/input")]
    for r in roots:
        if r.exists():
            for p in sorted(r.rglob("face_meta.csv")):                # feats_<name>/ (notebook output) or flag2027-feats-<name>-p/
                if p.parent.name == f"feats_{name}" or f"feats-{name}-" in str(p.parent).replace("\\", "/"):
                    return p.parent
    raise FileNotFoundError(f"feats_{name}/ (face_meta.csv) not found: attach the COLLECT notebook output")


def pseudo_rows(name, log=print):
    """Rows of a pseudo-identity source: identities with keep = True (identities.csv), voice halves 1 / 2 of the clip, half j with
    face j mod (faces of the clip)."""
    if name in _P:
        return _P[name]
    import pandas as pd
    d = feats_dir(name)
    dt = {"spk": str, "video": str, "utt": str, "clip_id": str, "yid": str}
    fm, vm = pd.read_csv(d / "face_meta.csv", dtype=dt), pd.read_csv(d / "voice_meta.csv", dtype=dt)
    F = np.concatenate([l2n(np.load(d / f"face_{n}.npy").astype(np.float32)) for n in ("farl16", "clipL")], 1)
    V = np.concatenate([l2n(np.load(d / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
    assert len(fm) == len(F) and len(vm) == len(V), (name, len(fm), len(F), len(vm), len(V))
    assert fm.spk.str.startswith(name + ":").all() and vm.spk.str.startswith(name + ":").all(), name
    ids = pd.read_csv(d / "identities.csv", dtype=dt, usecols=["spk", "keep"])
    keep = set(ids.spk[ids.keep.astype(bool)])                         # registered: Bengali, Bengali neighbours, near-duplicates out
    face_of = fm.groupby("spk").indices
    vh = vm[vm.half > 0] if "half" in vm else vm
    Fi, Vi, S = [], [], []
    for s, idx in sorted(vh.groupby("spk").indices.items()):
        if s not in face_of or s not in keep:
            continue
        cand = sorted(face_of[s])
        for j, v in enumerate(sorted(vh.index.values[idx])):
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append(s)
    log(f"{name} rows: {len(S)} rows, {len(set(S))} pseudo-identities ({d})")
    _P[name] = (F[Fi], V[Vi], np.array(S))
    return _P[name]


def sources_auxp(R, T, main_rows, aux, device):
    """D4 (SCALE-CC-02: main v4 + MAV + VoxCeleb2 SA; aux VoxCeleb2 + VoxBlink) + one aux head per pseudo-identity source."""
    main, out = C2.sources_d4(R, T, main_rows, tuple(a for a in aux if a in ("vox", "vb")), device)
    for k in [a for a in aux if a not in ("vox", "vb")]:
        F, V, S = pseudo_rows(k, R.log)
        out.append((k, M.t3r.Source(F, V, S, device)))
    R.log(f"AUXP-01: aux " + ", ".join(f"{k} {len(a.ids)} ids / {a.n_rows} rows" for k, a in out)
          + f" | total {len(main.ids) + sum(len(a.ids) for _, a in out)} identities")
    return main, out


M.sources = sources_auxp


def report():
    """SCALE-CC-01's report (SEL over V0-V4, gate vs v11, rule 3 vs CC5) + rule 1 (SEL - B) and V0 - B on the same paired table."""
    from lab_fuse import SP
    D3 = C1.D3
    cap, _paired = {}, D3.paired

    def paired_b(D, S2, persons, sets):
        sets = dict(sets, B=sets["v11"] + ["trB"])
        cap["D"] = _paired(D, S2, persons, sets)
        return cap["D"]
    D3.paired = paired_b
    try:
        verdict = C1.report()
    finally:
        D3.paired = _paired
    D = cap["D"]
    txt, res = ["\n== AUXP-01 rule 1: value of AVSpeech + LRS3 (tuned vs tuned), g cells per fold"], {}
    for a, b in (("SEL", "B"), ("V0", "B")):
        per = D3.g_gain(D, a, b)
        res[(a, b)] = per
        txt.append(f"{a} - {b}: " + ", ".join(f"fold {int(s)} {g:+.2f}" for s, g in per.items())
                   + f" | mean {per.mean():+.2f}, folds better {int((per > 0).sum())}/5, worse {int((per < 0).sum())}/5")
    for a, b in (("SEL", "B"),):
        per = D3.g_gain(D, a, b, by="face_sex").groupby(level=1).mean()
        txt.append(f"by face sex {a} - {b}: " + ", ".join(f"{k} {g:+.2f}" for k, g in per.items()))
    c = res[("SEL", "B")]
    v = ("HELPS" if c.mean() >= 0.3 and (c > 0).sum() >= 4 else
         "HURTS" if c.mean() <= -0.3 and (c < 0).sum() >= 4 else "NO DIFFERENCE")
    txt.append(f"RULE 1 (mean SEL - B >= +0.3 and better at >= 4/5 folds): AVSpeech + LRS3 {v}")
    print("\n".join(txt))
    with open(SP / "out/safe04" / f"{TAG}_report.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    return verdict


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "report":
        report()
    elif cmd == "final":
        print("final head: variant", C1.use_final(), flush=True)
        M.final()
    elif cmd == "zip":
        C1.use_final()
        M.zipit()
    elif cmd == "rows":
        for k in NEW:
            pseudo_rows(k)
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            M.run(s)
