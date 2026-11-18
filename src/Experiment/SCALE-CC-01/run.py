r"""SCALE-CC-01 (registered 12/10 in README.md before any result): as many identities as we have T3 features for, in the CC recipe
(Track 3's projector): main = v4 + MAV + every South-Asian VoxCeleb2 speaker (FLAG_37 + FLAG_53) + the VoxBlink South-Asian
speakers; VoxCeleb2 aux = the other VoxCeleb2 speakers (FLAG_37 rows + FLAG_53 `more` rows + FLAG_53 `new` speakers); VoxBlink aux =
the other VoxBlink speakers. Five hyper-parameter variants re-tuned for the larger data (1 seed each, the same seed), nested
selection SEL, final head = the variant with the best oof g gain, 5 seeds.
DSEL-03's run.py is loaded as a module (VoxBlink SA speakers, controls, paired gains; it loads DSEL-02 -> T3RECIPE-01 as M).
    FOLDS=1,2,3,4,5 python run.py ; python run.py report ; E2E_DIR=... python run.py final ; python run.py zip
Env: TR_SEEDS (5, final head), TR_MAX_STEPS (smoke tests only).
"""
import os, sys, json, importlib.util, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
_s = importlib.util.spec_from_file_location("dsel03run", EXP / "DSEL-03/run.py")
D3 = importlib.util.module_from_spec(_s); _s.loader.exec_module(D3)
M = D3.M                                                              # T3RECIPE-01 run.py
M.TAG = TAG = "scalecc01"
VARIANTS = {"V0": {}, "V1": dict(aux_ids=64), "V2": dict(epochs=80), "V3": dict(ids_per_batch=64, epochs=80),
            "V4": dict(face_dropout=.55, voice_dropout=.30)}
M.ARMS = {v: dict(aux=("vox", "vb"), pairs=["CC"], seeds=1, cfg=dict(M.t3r.SUB21A, **c)) for v, c in VARIANTS.items()}
M.DECISION = "V0"                                                     # extra columns only; the decision is SEL (report)
M.BASE_CFG = M.t3r.SUB21A
M.N_SEEDS = int(os.environ.get("TR_SEEDS", "5"))
_C = {}


def vox2b_dir():
    """FLAG_53's VoxCeleb2 arrays: Experiment/SCALE-ID-01/feats_vox2b, or flag2027-feats-vox2b-p under /kaggle/input."""
    if "d" not in _C:
        c = [EXP / "SCALE-ID-01/feats_vox2b"]
        if Path("/kaggle/input").exists():
            c += [p.parent for p in Path("/kaggle/input").rglob("voice_lid_probs.npy") if p.parent.name == "feats_vox2b"]
        hit = [d for d in c if (d / "voice_meta.csv").exists() and (d / "face_farl16.npy").exists()]
        assert hit, "attach flag2027-feats-vox2b-p (FLAG_53 output)"
        _C["d"] = hit[0]
    return _C["d"]


def drop53():
    """FLAG_53's Bengali LID list; with BN42=1 (CLEANTUNE-01, 31/10) also FLAG_42's (VOX_LID): 12 of its speakers sit in FLAG_53 rows."""
    d = set(json.loads((vox2b_dir() / "lid_summary.json").read_text())["speakers_dropped_ids"])
    if os.environ.get("BN42") == "1":
        p = os.environ.get("VOX_LID")
        assert p and Path(p).exists(), "BN42=1 needs VOX_LID (FLAG_42 lid_summary.json)"
        d42 = set(json.loads(Path(p).read_text())["speakers_dropped_ids"])
        assert len(d42) == 25, f"FLAG_42 list has {len(d42)} speakers, expected 25"
        d |= d42
    if os.environ.get("LID_STRICT") == "1":                           # LIDSTRICT-01 (2/11): + the bn + as rule (implies BN42)
        assert os.environ.get("BN42") == "1", "LID_STRICT=1 needs BN42=1"
        s = importlib.util.spec_from_file_location("t3enc01run", EXP / "T3ENC-01/run.py"); m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        d |= m.strict_drop("vox2")
    return d


def new_rows(kinds, log=print):
    """= SCALE-ID-01 new_rows: every voice clip of a kept speaker with a face of ANOTHER video of the same speaker (any face if
    it has one video), FLAG_53 rows of the given kinds, FLAG_53's LID rule applied."""
    import pandas as pd
    VB2 = vox2b_dir()
    l2n = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    fm = pd.read_csv(VB2 / "face_meta.csv", dtype=str)
    vm = pd.read_csv(VB2 / "voice_meta.csv", dtype={"spk": str, "video": str, "utt": str, "kind": str})
    F = np.concatenate([l2n(np.load(VB2 / f"face_{n}.npy").astype(np.float32)) for n in ("farl16", "clipL")], 1)
    V = np.concatenate([l2n(np.load(VB2 / f"voice_{n}.npy").astype(np.float32)) for n in ("sam100p", "sam100f")], 1)
    assert len(fm) == len(F) and len(vm) == len(V)
    fk, vk = fm.kind.isin(kinds).values, vm.kind.isin(kinds).values
    fm, F, vm, V = fm[fk].reset_index(drop=True), F[fk], vm[vk].reset_index(drop=True), V[vk]
    keep = (set(fm.spk) & set(vm.spk)) - drop53()
    face_of, fvid, vvid = fm.groupby("spk").indices, fm.video.values, vm.video.values
    Fi, Vi, S = [], [], []
    for s, idx in sorted(vm.groupby("spk").indices.items()):
        if s not in keep:
            continue
        for j, v in enumerate(sorted(idx)):
            cand = [f for f in face_of[s] if fvid[f] != vvid[v]] or list(face_of[s])
            Fi.append(cand[j % len(cand)]); Vi.append(v); S.append("vox2:" + s)
    log(f"FLAG_53 rows {kinds}: {len(S)} rows, {len(set(S))} speakers")
    return F[Fi], V[Vi], np.array(S)


def sources_cc(R, T, main_rows, aux, device):
    """Main = fold's v4 + MAV + VoxCeleb2 SA (FLAG_37 + FLAG_53) + VoxBlink SA; aux = the rest of VoxCeleb2 / VoxBlink."""
    t3r = M.t3r
    TF, TV, TS = main_rows
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))         # FLAG_37 rows, FLAG_42 filter
    ok = ~np.isin(S2, ["vox2:" + x for x in drop53()])                       # FLAG_53's LID rule also drops old rows
    F3, V3, S3 = new_rows(("new", "more"), R.log)
    FX, VX, SX = np.concatenate([F2[ok], F3]), np.concatenate([V2[ok], V3]), np.concatenate([S2[ok], S3])
    grp = {k: v[0] for k, v in json.loads((HERE / "vox2b_groups.json").read_text()).items()}
    grp.update({k: v[0] for k, v in json.loads(D3.D2.GROUPS.read_text()).items()})   # FLAG_37 speakers keep DSEL-01's group
    sa = np.array([grp.get(s.split(":", 1)[1], "OTH") == "SA" for s in SX])
    gm = dict(R.GMAP)
    gm.update({"vox2:" + s: g for s, g in json.loads((HERE / "vox2_gender.json").read_text()).items()})
    VBM = M.vb_module()
    FB, VB_, SB, info = VBM.vb_rows(R.log)
    sab_spk, vgen = D3.vb_sa_speakers(VBM)
    sab = np.array([s.split(":", 1)[1] in sab_spk for s in SB])
    gm.update({"vb:" + s: g for s, g in vgen.items()})
    MS = np.concatenate([np.asarray(TS).astype(str), SX[sa].astype(str), SB[sab].astype(str)])
    missing = sorted({s for s in MS if s not in gm})
    assert not missing, f"main identities without gender: {missing[:5]} ({len(missing)})"
    main = t3r.Source(np.concatenate([TF, FX[sa], FB[sab]]), np.concatenate([TV, VX[sa], VB_[sab]]), MS, device, male=gm)
    assert main.male is not None and main.male.any() and (~main.male).any()
    out = []
    if "vox" in aux:
        out.append(("vox", t3r.Source(FX[~sa], VX[~sa], SX[~sa], device)))
    if "vb" in aux:
        out.append(("vb", t3r.Source(FB[~sab], VB_[~sab], SB[~sab], device)))
    R.log(f"SCALE-CC-01 D3: main {len(main.ids)} ids / {main.n_rows} rows ({int(main.male.sum())} male; VoxCeleb2 SA "
          f"{len(set(SX[sa]))}, VoxBlink SA {len(set(SB[sab]))}) | aux " + ", ".join(f"{k} {len(a.ids)} ids / {a.n_rows} rows" for k, a in out)
          + f" | total {len(main.ids) + sum(len(a.ids) for _, a in out)} identities")
    return main, out


M.sources = sources_cc


def nested_choice(D, S2, persons, variants, base):
    """= DSEL-01 / HEADSWEEP-01 nested_choice: for every outer fold s, the variant with the best mean g-cell gain of CA-FUSE(base +
    tr<v>) over CA-FUSE(base) on the other folds t (pool = folds != s, t; purged of the persons of s and t)."""
    from lab_fuse import table
    sets = {"ref": base, **{v: base + [f"tr{v}"] for v in variants}}
    union = list(dict.fromkeys(c for st in sets.values() for c in st))
    folds = sorted(int(x) for x in D.fold.unique())
    choice, detail = {}, {}
    for s in folds:
        gains = {v: [] for v in variants}
        for t in [t for t in folds if t != s]:
            pool = S2.purge(D[(D.fold != s) & (D.fold != t)], persons[s] | persons[t])
            te = S2.purge(D[D.fold == t], persons[s])
            dz = S2.Design(pool, te, union)
            sc = {k: dz.score(st, 1.0) for k, st in sets.items()}
            X = te.assign(**{f"{k}_": v for k, v in sc.items()})
            T_ = table(X, {k: (lambda g, k=k: g[f"{k}_"].values) for k in sets})
            T_ = T_[T_.cell.str.endswith(" g")]
            for v in variants:
                gains[v].append(float((T_["ref"] - T_[v]).mean()))
        mean = {v: float(np.mean(g)) for v, g in gains.items()}
        best = max(variants, key=lambda v: (round(mean[v], 6), -list(variants).index(v)))
        choice[s], detail[s] = best, mean
    return choice, detail


def report():
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    base = S6 + E.V11_T3
    S2, persons = E.stack02(), E.fold_persons()
    V = list(VARIANTS)
    sel = {}

    def prep(D):
        D = D3.attach(D3.attach(D, "t3recipe02", "trCC5"), "dsel02", "trSAM")
        ch, det = nested_choice(D, S2, persons, V, base)
        D["trSEL"] = np.nan
        for s, v in ch.items():
            m = D.fold == s
            D.loc[m, "trSEL"] = D.loc[m, f"tr{v}"]
        sel.update(choice=ch, detail=det, D=D)
        return D
    arms = {"SEL": base + ["trSEL"], **{v: base + [f"tr{v}"] for v in V}, "SAM": base + ["trSAM"], "CC5": base + ["trCC5"]}
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    verdict = E.gate(TAG, arms, "SEL", standalone=[f"tr{v}" for v in V] + ["trSEL", "trSAM", "trCC5"], out=out,
                     ci_arms=["SEL", "V0"], prep=prep)
    full = {v: verdict[v]["oof"]["g"] for v in V}
    final_v = max(V, key=lambda v: (round(full[v], 6), -V.index(v)))
    sets = {"v11": base, "SEL": base + ["trSEL"], "CC5": base + ["trCC5"], "SAM": base + ["trSAM"], **{v: base + [f"tr{v}"] for v in V}}
    D = D3.paired(sel["D"], S2, persons, sets)
    D["face_sex"] = np.where(D.pf_m > 0.5, "male", "female")
    txt = ["\n== nested choice per outer fold (mean inner g-cell gain vs v11 of every variant):"]
    txt += [f"  fold {s}: {v}   " + " ".join(f"{k} {g:+.2f}" for k, g in sorted(sel["detail"][s].items(), key=lambda t: -t[1]))
            for s, v in sel["choice"].items()]
    txt.append(f"== variant of the final head (best oof g gain, all 5 folds): {final_v} (g {full[final_v]:+.2f})")
    txt.append("\n== paired CA-FUSE differences (purged, oof), g cells per fold")
    res = {}
    for a, b in [("SEL", "CC5"), ("SEL", "SAM"), (V[0], "SAM")] + [(v, V[0]) for v in V[1:]]:   # V[0]: first variant (V0 in SCALE-CC-01 / 02, AUXP-01)
        per = D3.g_gain(D, a, b)
        res[(a, b)] = per
        txt.append(f"{a} - {b}: " + ", ".join(f"fold {int(s)} {g:+.2f}" for s, g in per.items())
                   + f" | mean {per.mean():+.2f}, folds better {int((per > 0).sum())}/5")
    txt.append("\n== diagnostic: g-cell gain by the face's sex (pf_m > 0.5 = male), mean over folds")
    for a, b in (("SEL", "v11"), ("CC5", "v11"), ("SEL", "CC5")):
        per = D3.g_gain(D, a, b, by="face_sex").groupby(level=1).mean()
        txt.append(f"{a} - {b}: " + ", ".join(f"{k} {g:+.2f}" for k, g in per.items()))
    c = res[("SEL", "CC5")]
    repl = bool(verdict["SEL"]["passes"] and c.mean() >= 0.3 and int((c > 0).sum()) >= 4)
    txt.append(f"\nRULE 2 (gate SEL vs v11): {'PASSES' if verdict['SEL']['passes'] else 'fails'}")
    txt.append(f"RULE 3 (replace CC5 by SEL: gate + SEL - CC5 >= +0.3, >= 4/5 folds): {'YES -> final head ' + final_v + ' + one CB run' if repl else 'no'}")
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    (SP / "out/safe04" / f"{TAG}_selection.json").write_text(json.dumps(dict(
        choice={str(k): v for k, v in sel["choice"].items()}, final=final_v, oof_g=full, decision=verdict["SEL"]["passes"],
        replace_cc5=repl, sel_minus_cc5=float(c.mean())), indent=1))
    return verdict


def chosen():
    from lab_fuse import SP
    for p in (M.out_dir() / f"{TAG}_selection.json", SP / "out/safe04" / f"{TAG}_selection.json"):
        if p.exists():
            return json.loads(p.read_text())["final"]
    raise FileNotFoundError("run the report first: the final head uses the variant it chooses")


def use_final():
    """Arms / decision of the final head: the chosen variant with N_SEEDS seeds."""
    from lab_fuse import SP
    src = SP / "out/safe04" / f"{TAG}_selection.json"
    if src.exists() and not (M.out_dir() / src.name).exists():
        shutil.copy(src, M.out_dir() / src.name)
    v = chosen()
    M.ARMS = {v: dict(M.ARMS[v], seeds=M.N_SEEDS)}
    M.DECISION = v
    return v


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "report":
        report()
    elif cmd == "final":
        print("final head: variant", use_final(), flush=True)
        M.final()
    elif cmd == "zip":
        use_final()
        M.zipit()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            M.run(s)
