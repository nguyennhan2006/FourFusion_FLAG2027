r"""SCALE-CC-02 (registered 12/10 in README.md before any result): SCALE-CC-01 with the VoxBlink South-Asian speakers kept in the
VoxBlink AUXILIARY head (DSEL-03: in the main identities they dilute the MAV identities and hurt the g cells). Main = v4 + MAV +
every South-Asian VoxCeleb2 speaker (FLAG_37 + FLAG_53); VoxCeleb2 aux = the other VoxCeleb2 speakers (FLAG_37 + FLAG_53 rows);
VoxBlink aux = every VoxBlink-clean speaker. Same five variants, nested selection, rules and final head as SCALE-CC-01, whose
run.py is loaded as a module (it loads DSEL-03 -> DSEL-02 -> T3RECIPE-01 as M).
    FOLDS=1,2,3,4,5 python run.py ; python run.py report ; E2E_DIR=... python run.py final ; python run.py zip
Env: TR_SEEDS (5, final head), TR_MAX_STEPS (smoke tests only).
"""
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
_s = importlib.util.spec_from_file_location("scalecc01run", EXP / "SCALE-CC-01/run.py")
C1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(C1)
M = C1.M
M.TAG = C1.TAG = TAG = "scalecc02"                                     # C1's report / chosen / use_final read C1.TAG
# amendment 12/10 (README), before any SCALE-CC-02 result: V5 = V1 + V3 + V4 (each helped on its own in SCALE-CC-01)
C1.VARIANTS["V5"] = dict(aux_ids=64, ids_per_batch=64, epochs=80, face_dropout=.55, voice_dropout=.30)
M.ARMS["V5"] = dict(aux=("vox", "vb"), pairs=["CC"], seeds=1, cfg=dict(M.t3r.SUB21A, **C1.VARIANTS["V5"]))


def sources_d4(R, T, main_rows, aux, device):
    """Main = fold's v4 + MAV + VoxCeleb2 SA (FLAG_37 + FLAG_53); aux = the other VoxCeleb2 speakers, and ALL of VoxBlink."""
    t3r = M.t3r
    TF, TV, TS = main_rows
    F2, V2, S2 = T.vox_rows(T.t3("face", "vox"), T.t3("voice", "vox"))         # FLAG_37 rows, FLAG_42 filter
    ok = ~np.isin(S2, ["vox2:" + x for x in C1.drop53()])                    # FLAG_53's LID rule also drops old rows
    F3, V3, S3 = C1.new_rows(("new", "more"), R.log)
    FX, VX, SX = np.concatenate([F2[ok], F3]), np.concatenate([V2[ok], V3]), np.concatenate([S2[ok], S3])
    grp = {k: v[0] for k, v in json.loads((EXP / "SCALE-CC-01/vox2b_groups.json").read_text()).items()}
    grp.update({k: v[0] for k, v in json.loads(C1.D3.D2.GROUPS.read_text()).items()})   # FLAG_37 speakers keep DSEL-01's group
    sa = np.array([grp.get(s.split(":", 1)[1], "OTH") == "SA" for s in SX])
    gm = dict(R.GMAP)
    gm.update({"vox2:" + s: g for s, g in json.loads((EXP / "SCALE-CC-01/vox2_gender.json").read_text()).items()})
    MS = np.concatenate([np.asarray(TS).astype(str), SX[sa].astype(str)])
    missing = sorted({s for s in MS if s not in gm})
    assert not missing, f"main identities without gender: {missing[:5]} ({len(missing)})"
    main = t3r.Source(np.concatenate([TF, FX[sa]]), np.concatenate([TV, VX[sa]]), MS, device, male=gm)
    assert main.male is not None and main.male.any() and (~main.male).any()
    out = []
    if "vox" in aux:
        out.append(("vox", t3r.Source(FX[~sa], VX[~sa], SX[~sa], device)))
    if "vb" in aux:
        FB, VB_, SB, info = M.vb_module().vb_rows(R.log)
        out.append(("vb", t3r.Source(FB, VB_, SB, device)))
    n_mav = len(set(np.asarray(TS).astype(str)))
    R.log(f"SCALE-CC-02 D4: main {len(main.ids)} ids / {main.n_rows} rows ({int(main.male.sum())} male; MAV {n_mav} = "
          f"{100 * n_mav / len(main.ids):.0f} %, VoxCeleb2 SA {len(set(SX[sa]))}) | aux " + ", ".join(f"{k} {len(a.ids)} ids / {a.n_rows} rows" for k, a in out)
          + f" | total {len(main.ids) + sum(len(a.ids) for _, a in out)} identities")
    return main, out


M.sources = sources_d4

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "report":
        C1.report()
    elif cmd == "final":
        print("final head: variant", C1.use_final(), flush=True)
        M.final()
    elif cmd == "zip":
        C1.use_final()
        M.zipit()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            M.run(s)
