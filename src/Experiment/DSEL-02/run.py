r"""DSEL-02 (registered 12/10 in README.md before any result): CC5 (T3RECIPE-02) with the 305 VoxCeleb2 South-Asian speakers
moved from the VoxCeleb2 auxiliary source into the MAIN identities (gender-balanced batches + same-gender SupCon). One arm SAM.
Control = trCC5 of T3RECIPE-02 (same trial tables). Same machinery as T3RECIPE-01 (its run.py loaded as a module).
    VOX_LID=... FOLDS=1,2,3,4,5 python run.py ; python run.py report ; E2E_DIR=... python run.py final ; python run.py zip
Env: TR_SEEDS (5), TR_MAX_STEPS (smoke tests only).
"""
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
spec = importlib.util.spec_from_file_location("t3recipe01run", EXP / "T3RECIPE-01/run.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
M.TAG = TAG = "dsel02"
M.ARMS = {"SAM": dict(aux=("vox", "vb"), pairs=["CC"])}
M.DECISION = DECISION = "SAM"
M.BASE_CFG = M.t3r.SUB21A
M.N_SEEDS = int(os.environ.get("TR_SEEDS", "5"))
GROUPS = EXP / "DSEL-01/vox2_groups.json"


def sources_sam(R, T, main_rows, aux, device):
    """Main = the fold's v4 + MAV rows + the VoxCeleb2 South-Asian speakers; VoxCeleb2 aux = the other speakers."""
    import pandas as pd
    t3r = M.t3r
    TF, TV, TS = main_rows
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))
    grp = {k: v[0] for k, v in json.loads(GROUPS.read_text()).items()}
    sa = np.array([grp.get(s.split(":", 1)[1], "OTH") == "SA" for s in S2])
    fm = pd.read_csv(T.VK / "face_meta.csv", dtype={"video": str, "utt": str})
    gm = dict(R.GMAP); gm.update({"vox2:" + s: g for s, g in zip(fm.spk, fm.gender)})
    main = t3r.Source(np.concatenate([TF, F2[sa]]), np.concatenate([TV, V2[sa]]),
                      np.concatenate([np.asarray(TS).astype(str), S2[sa].astype(str)]), device, male=gm)
    assert main.male is not None and main.male.any() and (~main.male).any()
    R.log(f"SAM: {len(set(S2[sa]))} VoxCeleb2 South-Asian speakers ({int(sa.sum())} rows) moved into the main identities")
    out = []
    if "vox" in aux:
        out.append(("vox", t3r.Source(F2[~sa], V2[~sa], S2[~sa], device)))
    if "vb" in aux:
        FB, VB_, SB, info = M.vb_module().vb_rows(R.log)
        out.append(("vb", t3r.Source(FB, VB_, SB, device)))
    return main, out


M.sources = sources_sam


def with_cc5(D):
    """Attach trCC5 of T3RECIPE-02 (same trial tables, row-aligned) as the control column."""
    import pandas as pd
    from lab_fuse import SP
    fs = sorted((SP / "out/safe04").glob("t3recipe02_fold*.parquet"))
    if len(fs) < 5 and Path("/kaggle/input").exists():          # Kaggle: the attached T3RECIPE-02 output (kernel source)
        fs = sorted({f.name: f for f in Path("/kaggle/input").rglob("t3recipe02_fold*.parquet")}.values(), key=lambda f: f.name)
    assert len(fs) == 5, f"T3RECIPE-02 fold tables not found ({len(fs)})"
    C = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    C = C[C.fold.isin(D.fold.unique())].reset_index(drop=True)
    assert len(C) == len(D) and (C.y.values == D.y.values).all() and (C.vi.values == D.vi.values).all(), "row alignment"
    D["trCC5"] = C.trCC5.values
    return D


def paired_g(D, S2, persons, a_cols, b_cols):
    """Per-fold g-cell gain of CA-FUSE(a) over CA-FUSE(b), purged (+ = a better)."""
    from lab_fuse import table
    D = D.copy(); D["a_"], D["b_"] = np.nan, np.nan
    for s in sorted(D.fold.unique()):
        te = D[D.fold == s]; pool = S2.purge(D[D.fold != s], persons[int(s)])
        dz = S2.Design(pool, te, list(dict.fromkeys(a_cols + b_cols)))
        D.loc[te.index, "a_"], D.loc[te.index, "b_"] = dz.score(a_cols, 1.0), dz.score(b_cols, 1.0)
    T_ = table(D, {"a": lambda g: g["a_"].values, "b": lambda g: g["b_"].values})
    T_ = T_[T_.cell.str.endswith(" g")]
    per = (T_["b"] - T_["a"]).groupby(T_.fold).mean()
    return per


def report():
    import e2e_lib as E
    from stream_gate import S6
    from lab_fuse import SP
    base = S6 + E.V11_T3
    holder = {}

    def prep(D):
        D = with_cc5(D); holder["D"] = D
        return D
    arms = {"SAM": base + ["trSAM"], "CC5": base + ["trCC5"], "SAM_s0": base + ["trSAM_s0"]}
    out = SP / "out/safe04" / f"{TAG}_report.txt"
    v = E.gate(TAG, arms, DECISION, standalone=["trSAM", "trCC5"], out=out, ci_arms=["SAM", "CC5"], prep=prep)
    per = paired_g(holder["D"], E.stack02(), E.fold_persons(), base + ["trSAM"], base + ["trCC5"])
    ok2 = per.mean() >= 0.3 and int((per > 0).sum()) >= 4
    txt = ("\n== SAM - CC5 (paired, purged), g cells per fold: " + ", ".join(f"fold {s} {g:+.2f}" for s, g in per.items())
           + f" | mean {per.mean():+.2f}, folds better {int((per > 0).sum())}/5 -> "
           + ("REPLACES CC5 (with the gate)" if (ok2 and v['SAM']['passes']) else "does not replace CC5") + "\n")
    print(txt)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(txt)


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
