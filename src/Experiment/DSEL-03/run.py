r"""DSEL-03 (registered 12/10 in README.md before any result): SAM of DSEL-02 (CC5 + the 305 VoxCeleb2 South-Asian speakers in
the MAIN identities) + the VoxBlink-clean South-Asian speakers (VoxBlink's own loc / lan labels) moved from the VoxBlink auxiliary
head into the main identities as well. One arm SAM2. Controls (paired): trCC5 (T3RECIPE-02) and trSAM (DSEL-02).
DSEL-02's run.py is loaded as a module (it loads T3RECIPE-01's run.py as M).
    FOLDS=1,2,3,4,5 python run.py ; python run.py report ; E2E_DIR=... python run.py final ; python run.py zip
Env: TR_SEEDS (5), TR_MAX_STEPS (smoke tests only).
"""
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
_s = importlib.util.spec_from_file_location("dsel02run", EXP / "DSEL-02/run.py")
D2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(D2)
M = D2.M                                                              # T3RECIPE-01 run.py (as patched by DSEL-02)
M.TAG = TAG = "dsel03"
M.ARMS = {"SAM2": dict(aux=("vox", "vb"), pairs=["CC"])}             # one pair -> the same seeds as SAM / CC5
M.DECISION = DECISION = "SAM2"
M.BASE_CFG = M.t3r.SUB21A
M.N_SEEDS = int(os.environ.get("TR_SEEDS", "5"))
SA_LOC = {"India", "Pakistan", "Nepal", "Sri Lanka", "Bangladesh", "Bhutan", "Maldives"}
SA_LAN = {"hindi", "urdu", "tamil", "telugu", "punjabi", "malayalam", "kannada", "marathi", "gujarati", "nepali", "sinhala",
          "bengali", "sindhi", "oriya", "assamese"}


def vb_sa_speakers(VBM):
    """VoxBlink speakers of the South-Asian group by VoxBlink's OWN metadata (loc / lan) -> (set of speaker ids, {spk: m/f})."""
    import pandas as pd
    vm = pd.read_csv(VBM.VB / "voice_meta.csv", dtype={"video": str, "utt": str})
    sp = vm.drop_duplicates("spk").set_index("spk")
    sa = {s for s, r in sp.iterrows() if r["loc"] in SA_LOC or r["lan"] in SA_LAN}
    gen = {s: ("m" if str(g).lower().startswith("m") else "f") for s, g in sp.gender.items()}
    return sa, gen


def sources_sam2(R, T, main_rows, aux, device):
    """Main = the fold's v4 + MAV rows + VoxCeleb2 SA speakers + VoxBlink SA speakers; aux = the rest of each source."""
    import pandas as pd
    t3r = M.t3r
    TF, TV, TS = main_rows
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    grp = {k: v[0] for k, v in json.loads(D2.GROUPS.read_text()).items()}
    sa = np.array([grp.get(s.split(":", 1)[1], "OTH") == "SA" for s in S2])
    fm = pd.read_csv(T.VK / "face_meta.csv", dtype={"video": str, "utt": str})
    gm = dict(R.GMAP); gm.update({"vox2:" + s: g for s, g in zip(fm.spk, fm.gender)})
    VBM = M.vb_module()
    FB, VB_, SB, info = VBM.vb_rows(R.log)
    sab_spk, vgen = vb_sa_speakers(VBM)
    sab = np.array([s.split(":", 1)[1] in sab_spk for s in SB])
    gm.update({"vb:" + s: g for s, g in vgen.items()})
    MS = np.concatenate([np.asarray(TS).astype(str), S2[sa].astype(str), SB[sab].astype(str)])
    missing = sorted({s for s in MS if s not in gm})
    assert not missing, f"main identities without gender: {missing[:5]} ({len(missing)})"
    main = t3r.Source(np.concatenate([TF, F2[sa], FB[sab]]), np.concatenate([TV, V2[sa], VB_[sab]]), MS, device, male=gm)
    assert main.male is not None and main.male.any() and (~main.male).any()
    nb = sorted(set(SB[sab]))
    R.log(f"SAM2: {len(set(S2[sa]))} VoxCeleb2 SA speakers ({int(sa.sum())} rows) + {len(nb)} VoxBlink SA speakers "
          f"({int(sab.sum())} rows; female {sum(vgen[s.split(':', 1)[1]] == 'f' for s in nb)}) moved into the main identities")
    out = []
    if "vox" in aux:
        out.append(("vox", t3r.Source(F2[~sa], V2[~sa], S2[~sa], device)))
    if "vb" in aux:
        out.append(("vb", t3r.Source(FB[~sab], VB_[~sab], SB[~sab], device)))
    return main, out


M.sources = sources_sam2


def attach(D, tag, col):
    """Column `col` of experiment `tag` (same trial tables, row-aligned): <scratch>/out/safe04 or the attached inputs."""
    import pandas as pd
    from lab_fuse import SP
    fs = sorted((SP / "out/safe04").glob(f"{tag}_fold*.parquet"))
    if len(fs) < 5 and Path("/kaggle/input").exists():
        fs = sorted({f.name: f for f in Path("/kaggle/input").rglob(f"{tag}_fold*.parquet")}.values(), key=lambda f: f.name)
    assert len(fs) == 5, f"{tag} fold tables not found ({len(fs)})"
    C = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    C = C[C.fold.isin(D.fold.unique())].reset_index(drop=True)
    assert len(C) == len(D) and (C.y.values == D.y.values).all() and (C.vi.values == D.vi.values).all(), f"row alignment {tag}"
    D[col] = C[col].values
    return D


def eer(s, y):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


def paired(D, S2, persons, sets):
    """CA-FUSE score of every stream set (oof, purged, C = 1) -> columns sc_<name>."""
    D = D.copy()
    allc = list(dict.fromkeys(c for st in sets.values() for c in st))
    for k in sets:
        D[f"sc_{k}"] = np.nan
    for s in sorted(D.fold.unique()):
        te = D[D.fold == s]; pool = S2.purge(D[D.fold != s], persons[int(s)])
        dz = S2.Design(pool, te, allc)
        for k, st in sets.items():
            D.loc[te.index, f"sc_{k}"] = dz.score(st, 1.0)
    return D


def g_gain(D, a, b, by=None):
    """Mean over folds and g cells of EER(b) - EER(a) (+ = a better), optionally per value of column `by`."""
    G = D[D.protocol == "g"]
    keys = ["fold", "data"] + ([by] if by else [])
    rows = []
    for k, g in G.groupby(keys):
        if g.y.nunique() == 2:
            rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), gain=eer(g[f"sc_{b}"].values, g.y.values) - eer(g[f"sc_{a}"].values, g.y.values)))
    import pandas as pd
    T_ = pd.DataFrame(rows)
    return T_.groupby(["fold"] + ([by] if by else [])).gain.mean()


def report():
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    base = S6 + E.V11_T3
    holder = {}

    def prep(D):
        D = attach(attach(D, "t3recipe02", "trCC5"), "dsel02", "trSAM"); holder["D"] = D
        return D
    arms = {"SAM2": base + ["trSAM2"], "SAM": base + ["trSAM"], "CC5": base + ["trCC5"], "SAM2_s0": base + ["trSAM2_s0"]}
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    v = E.gate(TAG, arms, DECISION, standalone=["trSAM2", "trSAM", "trCC5"], out=out, ci_arms=["SAM2", "SAM"], prep=prep)
    S2, persons = E.stack02(), E.fold_persons()
    D = paired(holder["D"], S2, persons, {"v11": base, "SAM2": base + ["trSAM2"], "SAM": base + ["trSAM"], "CC5": base + ["trCC5"]})
    D["face_sex"] = np.where(D.pf_m > 0.5, "male", "female")
    txt = ["\n== paired CA-FUSE differences (purged, oof), g cells per fold"]
    res = {}
    for a, b in (("SAM2", "CC5"), ("SAM2", "SAM"), ("SAM", "CC5")):
        per = g_gain(D, a, b)
        res[(a, b)] = per
        txt.append(f"{a} - {b}: " + ", ".join(f"fold {int(s)} {g:+.2f}" for s, g in per.items())
                   + f" | mean {per.mean():+.2f}, folds better {int((per > 0).sum())}/5")
    txt.append("\n== diagnostic: g-cell gain by the face's sex (pf_m > 0.5 = male), mean over folds")
    for a, b in (("SAM2", "v11"), ("SAM", "v11"), ("SAM2", "SAM")):
        per = g_gain(D, a, b, by="face_sex").groupby(level=1).mean()
        txt.append(f"{a} - {b}: " + ", ".join(f"{k} {g:+.2f}" for k, g in per.items()))
    c = res[("SAM2", "CC5")]
    repl = v["SAM2"]["passes"] and c.mean() >= 0.3 and int((c > 0).sum()) >= 4
    txt.append(f"\nRULE 1 (gate SAM2 vs v11): {'PASSES' if v['SAM2']['passes'] else 'fails'}")
    txt.append(f"RULE 2 (replace CC5 by SAM2: gate + SAM2 - CC5 >= +0.3, >= 4/5 folds): {'YES -> one CB run' if repl else 'no'}")
    print("\n".join(txt))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "report":
        report()
    elif cmd == "final":
        M.final()
    elif cmd == "zip":
        M.zipit()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            M.run(s)
