r"""FLAG_110 (31/10): CLEANTUNE-01 round 1 (Experiment/CLEANTUNE-01/README.md, registered 31/10 before any result): the CC head on
Bengali-clean rows and adapters (BN42=1, FLAG_109 arrays), one change per arm against the in-run control B0c.
    FLAG_110_ct01_g1.ipynb / _g2 / _g3   fold jobs of one group (GPU T4 x2): --env 'LAB_FOLDS=1,2' / '3,4' / '5'
    FLAG_110r_ct01_report.ipynb          CPU: the 15 fold tables attached -> merge -> report -> ct_round2.json
    python kaggle/build_flag110.py
Smoke (Kaggle CPU), group notebooks: --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1'
-> fold 1 tiny (old unmarked adapters allowed), folds 2-5 = noisy copies of the B3 tables; report notebook: --smoke 'FL3Z_SMOKE=1'
-> every group table a noisy copy of the B3 tables."""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_e2e_lab as BL
from build_flag93 import GATHER

EXP = "CLEANTUNE-01"
ARMS_OF = {"g1": ("B0c", "DRm", "DRp", "EPm", "EPp", "WDp"),
           "g2": ("B0c", "AWm", "AIp", "BSp", "DIMp", "T", "M"),
           "g3": ("B0c", "C5", "CAT4", "TAPv", "C5CAT4")}
CODE = (["Experiment/CLEANTUNE-01/run.py", "Experiment/VOICELORA-01/run.py", "Experiment/FACELORA-05/run.py"] + BL.EXPS["FACELORA-03"]["code"]
        + ["Experiment/EN-T3-01/run.py", "Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip"])   # F0's English fit + the 20.18 zip
if (HERE.parent / "Experiment/CLEANTUNE-01/ct_round2.json").exists():   # round 2 / F1 notebooks: the round-1 plan travels with the code
    CODE.append("Experiment/CLEANTUNE-01/ct_round2.json")
_CP = os.environ.get("CT_PLAN")   # 3/11 (CLEANTUNE-02): build-time opt-in; plan path relative to Experiment/CLEANTUNE-01/, e.g. ../CLEANTUNE-02/ct_round2.json
if _CP:
    CODE.append("Experiment/" + os.path.normpath("CLEANTUNE-01/" + _CP).replace("\\", "/"))
OPT = dict(BL.OPTIONAL_DIRS, **{"Experiment/CAFUSE-EN/feats_av_full": [EXP]})   # flag2027-lab-avfull-p: AudioCLIP of the English table (F0)
files = BL.CODE_COMMON + [f for f in CODE if f not in BL.CODE_COMMON]
blob, md5 = BL.code_blob(files)
SETUP = BL.SETUP.replace("__CODE_B64__", blob).replace("__CODE_MD5__", json.dumps(md5, indent=1)).replace(
    "__DATA_DIRS__", json.dumps(BL.L.DATA_DIRS)).replace("__OPTIONAL_DIRS__", json.dumps(OPT))


def env(tag, group):
    e = BL.ENV.replace("__EXP__", EXP).replace("__TAG__", tag).replace("__FAR__", "True").replace("__EXP_CODE__", json.dumps(CODE))
    old = 'FOLDS = [int(x) for x in os.environ.get("LAB_FOLDS", "1,2,3,4,5").split(",") if x]'
    assert old in e
    return e.replace(old, old + '\nSMOKE = os.environ.get("FL3Z_SMOKE") == "1"                       # wiring test: fold 1 only\n'
                     'FOLDS = FOLDS[:1] if SMOKE else FOLDS\n'
                     f'os.environ["CT_GROUP"] = "{group}"                                   # CLEANTUNE-01 group of this notebook\n'
                     'os.environ["BN42"] = "1"                                         # Bengali-clean rows + FLAG_109 adapters only'
                     + (f'\nos.environ["CT_PLAN"] = {_CP!r}\nos.environ["CT_R2_EXTRA"] = "F1S"' if _CP else ''))   # CLEANTUNE-02 plan + F1s arm


STANDIN = r'''
# smoke only: group tables not trained here = noisy copies of FACELORA-03's B3 tables (wiring of gather / merge / report)
def standin(tag, arms, seed):
    import numpy as np
    for s in range(1, 6):
        f = SP / "out/safe04" / f"{tag}_fold{s}.parquet"
        if not f.exists():
            d = pd.read_parquet(SP / "out/safe04" / f"facelora03_B3_fold{s}.parquet")
            rng = np.random.RandomState(100 * seed + s)
            for a in arms:
                d[f"tr{a}"] = d["trB3"].values + 0.05 * rng.randn(len(d))
            d.drop(columns=[c for c in d.columns if c.startswith("trB3")]).to_parquet(f)
            prog["done"] = sorted(set(prog["done"]) | {f"{tag}_fold{s}"})
            log(f"SMOKE: {tag} fold {s} = B3 stand-in")
'''

GROUP_STANDIN = STANDIN + r'''
if SMOKE:
    standin(TAG, __ARMS__, 0)
'''

GATHER_CT = r'''
# the 15 fold tables of the three group notebooks (attached outputs or the tables dataset) + FACELORA-03's B3 reference tables
import pandas as pd
for f in sorted(IN.rglob("ct01g[123]_fold[1-5].parquet")):
    if not (SP / "out/safe04" / f.name).exists():
        shutil.copy(f, SP / "out/safe04" / f.name); log(f"gathered {f.name} from {f.parent}")
for f in sorted(IN.rglob("facelora03_B3_fold*.parquet")):
    shutil.copy(f, SP / "out/safe04" / f.name)
''' + STANDIN + r'''
if SMOKE:
    for i, (g, arms) in enumerate(__ARMS_OF__.items()):
        standin(f"ct01{g}", arms, i)
have = sorted(p.name for p in (SP / "out/safe04").glob("ct01g*_fold*.parquet"))
log(f"group fold tables: {len(have)} / 15")
assert len(have) == 15, f"attach the three groups' fold tables: have {have}"
'''

MERGE_REPORT = r'''
# CLEANTUNE-01 merge (three groups side by side) + registered round-1 rules -> ct_round2.json
for cmd in ("merge", "report"):
    r = subprocess.run([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP, cmd], cwd=str(WORK), env=ENVV,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("\n".join(l for l in r.stdout.splitlines() if "Warning" not in l))
    assert r.returncode == 0, r.stderr[-3000:]
for name in ["ct01_report.txt", "ct_round2.json"] + [f"ct01_fold{s}.parquet" for s in range(1, 6)]:
    shutil.copy(SP / "out/safe04" / name, OUTD / name)
print(json.dumps(json.loads((OUTD / "ct_round2.json").read_text()), indent=1)[:3000])
'''

INTRO = """
# FLAG 2027 — NB-110{sfx}: CLEANTUNE-01 round 1{what}

**Why (Experiment/CLEANTUNE-01/README.md, registered 31/10 before any result).** The 20.18 system's adapters and CC head saw 12
Bengali-flagged VoxCeleb2 speakers (FLAG_53 rows were filtered with FLAG_53's own list only). Every arm here runs on the clean rows
(`BN42=1`) and the FLAG_109 adapters (arrays marked `bn42_clean.json`), and changes ONE thing against the in-run control **B0c**
(= the B3 recipe): {arms}. Round 2 combines the arms with a signal (rules in the README).
**Inputs:** the DSC datasets (no AVSpeech / LRS3) + `{b3c}`{more} + `ngnnguynchn/flag2027-facelora03-tables-p`. Internet OFF.
"""


def group_nb(g):
    tag = f"ct01{g}"
    more = (" + `" + "` + `".join(["<owner>/flag2027-feats-ct-c5c-p", "dnq1209/flag2027-feats-tap01-voice-p",
                                    "ngnnguynchn/flag2027-feats-screen2-p", "hthlin/flag2027-aux-voice-feats-p"]) + "`") if g == "g3" else ""
    intro = INTRO.format(sfx=f" {g}", what=f", group {g}: fold jobs of `LAB_FOLDS` (GPU T4 x2)", arms=", ".join(ARMS_OF[g][1:]),
                         b3c="<owner>/flag2027-feats-ct-b3c-p", more=more)
    cells = [("markdown", intro), ("code", env(tag, g)),
             ("markdown", "## 1. Code (embedded) + data links + SAFE-04 reference tables"), ("code", SETUP),
             ("markdown", "## 2. Resume (optional) + fold tables of the other runs"), ("code", BL.RESUME), ("code", GATHER),
             ("markdown", "## 3. Jobs: the folds of LAB_FOLDS, one per GPU at a time"), ("code", BL.JOBS),
             ("code", GROUP_STANDIN.replace("__ARMS__", repr(ARMS_OF[g]))),
             ("markdown", "## 4. Summary (the report runs in FLAG_110r with the three groups)"), ("code", BL.SUMMARY)]
    return cells, f"FLAG_110_ct01_{g}.ipynb"


def report_nb():
    intro = INTRO.format(sfx="r", what=": merge + report (CPU)", arms="every arm of g1 / g2 / g3",
                         b3c="<owner>/flag2027-feats-ct-b3c-p", more=" + the tables dataset of the 9 runs")
    cells = [("markdown", intro), ("code", env("ct01", "g1")),
             ("markdown", "## 1. Code (embedded) + data links + SAFE-04 reference tables"), ("code", SETUP),
             ("markdown", "## 2. Resume (optional) + the 15 group fold tables"), ("code", BL.RESUME),
             ("code", GATHER_CT.replace("__ARMS_OF__", repr(ARMS_OF))),
             ("markdown", "## 3. Merge + registered round-1 rules (README §2) -> ct_round2.json"), ("code", MERGE_REPORT),
             ("markdown", "## 4. Summary"), ("code", BL.SUMMARY)]
    return cells, "FLAG_110r_ct01_report.ipynb"


if __name__ == "__main__":
    for cells, nb in [group_nb(g) for g in ARMS_OF] + [report_nb()]:
        BL.B12.nb(cells, nb)
        print(f"{nb}: embedded {len(md5)} code files ({len(blob) / 1e3:.0f} kB base64)")
