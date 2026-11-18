r"""FLAG_112 (31/10): CLEANTUNE-01 end-to-end heads + dev zips (Experiment/CLEANTUNE-01/README.md §3).
    FLAG_112_f0.ipynb   F0 = CLEAN-B0c: B0c head on every train person (5 seeds, BN42=1, FLAG_109 b3c arrays incl. the dev lists),
                        Bangla fuser S6 + v11 T3 + trB0c fitted on the ct01g1 tables, English CLEAN form on trB0c -> zip
    FLAG_112_f1.ipynb   F1 (only after a passing round 2): FINAL_ARM of ct_decision.json on the ct01r2 tables, English bytes of F0
    python kaggle/build_flag112.py f0   |   python kaggle/build_flag112.py f1 <ct_decision.json>
    FLAG_112_f0x / f1x  (2/11) the Evaluation path of F0 / F1: NO training, the saved final models of the f0 / f1 run score the test
                        files of E2E_SPLIT (dev = R11 rehearsal, compared with that run's zip; eval on 11/11); faces = FLAG_65xb / 65xc,
                        F1's extra voices = a FLAG_47b re-run    python kaggle/build_flag112.py f0x | f1x <ct_decision.json>
Inputs: the DSC datasets + the FLAG_109 b3c dataset (+ c5c for a C5 input) + the round's fold tables (F0: ct01g1, F1: ct01r2)
(+ F1: the F0 zip). GPU T4 (one head of 5 / 10 seeds); smoke: --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;CT_FINAL_SEEDS=1;BN42_STANDIN=1'.
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_e2e_lab as BL
import build_flag110 as B110

EXP = "CLEANTUNE-01"

GATHER_T = r'''
# the round's 5 fold tables (attached outputs / tables dataset) + FACELORA-03's B3 tables; smoke: stand-ins
import pandas as pd
for f in sorted(IN.rglob(f"{TAG}_fold[1-5].parquet")):
    if not (SP / "out/safe04" / f.name).exists():
        shutil.copy(f, SP / "out/safe04" / f.name); log(f"gathered {f.name} from {f.parent}")
for f in sorted(IN.rglob("facelora03_B3_fold*.parquet")):
    shutil.copy(f, SP / "out/safe04" / f.name)
''' + B110.STANDIN + r'''
if SMOKE:
    standin(TAG, [os.environ["FINAL_ARM"]], 0)
have = sorted(p.name for p in (SP / "out/safe04").glob(f"{TAG}_fold*.parquet"))
assert len(have) == 5, f"attach the 5 fold tables of {TAG}: have {have}"
F0Z = sorted(IN.rglob(f"submission_cleantune01_F0_{SPLIT}.zip"))      # 2/11: the split's F0 zip (eval: the f0x run on the Evaluation data)
if not F0Z:   # 3/11 (CLEANTUNE-02 F2s): the F0s zip travels in a dataset as <name>.zip.bin (Kaggle unpacks .zip files on upload)
    for q in sorted(IN.rglob(f"submission_cleantune01_F0_{SPLIT}.zip.bin")):
        shutil.copy(q, WORK / q.name[:-4]); F0Z = [WORK / q.name[:-4]]; break
if os.environ.get("CT_F1") == "1":
    assert len(F0Z) == 1, f"F1 needs exactly one F0 zip of split {SPLIT} attached (English bytes): {F0Z}"
    os.environ["F0_ZIP"] = ENVV["F0_ZIP"] = str(F0Z[0])
'''

FINAL_ZIP = r'''
# A0 audit on the training rows, final head (GPU 0), then the zip (README §1 S0c, §3)
for cmd in ("audit", "final", "zip"):
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP, cmd], cwd=str(WORK),
                         env=dict(ENVV, SAFE04_THREADS=str(os.cpu_count() or 4), CUDA_VISIBLE_DEVICES="0" if NGPU else ""),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-60:]
        if "Warning" not in line and "warnings.warn" not in line:
            print(f"[{cmd}]", line, end="", flush=True)
    if p.wait() != 0:
        print("".join(tail)); raise RuntimeError(f"{EXP} {cmd} failed")
    log(f"{cmd} done [{time.time() - t0:.0f}s]")
ZP = sorted(OUTD.rglob("submission_cleantune01_*.zip"))
assert len(ZP) == 1, ZP
shutil.copy(ZP[0], WORK / "submission.zip"); shutil.copy(ZP[0], OUTD / ZP[0].name) if ZP[0].parent != OUTD else None
log("submission:", ZP[0].name, "sha256", hashlib.sha256(ZP[0].read_bytes()).hexdigest()[:16])
'''

EVALPATH = r'''
# Evaluation path / R11 (2/11, docs/PLAN_EVAL_F0F1_2026-11-02.md): NO training. The saved final models of the F0 / F1 run (attached
# kernel output) score the test files of the split; faces of the test lists = FLAG_65xb / 65xc outputs (saved B3c / C5c adapters,
# every other array copied in), F1's rn293 / ecapa1024 = a FLAG_47b re-run. Dev = R11: the zip is compared with that run's zip.
PROG = OUTD / "progress.json"
prog = {"exp": EXP, "split": SPLIT, "done": [], "seconds": {}, "files": {}, "code_md5": CODE_MD5, "evalpath": True}
F1 = os.environ.get("CT_F1") == "1"
mf = sorted(IN.rglob(f"{TAG}_final_models.pt"))
assert len(mf) == 1, f"attach exactly one output holding {TAG}_final_models.pt (F0: nguynchnngn/flag-2027-nb-112-f0, F1: -112-f1): {mf}"
shutil.copy(mf[0], OUTD / mf[0].name); log("saved final models:", mf[0])
for arr, d in (("clipB3", "feats_clipb3c_x"), ("clipC5", "feats_clipc5c_x"))[:2 if F1 else 1]:
    hit = sorted({q.parent for q in IN.rglob(f"face_{arr}_train.npy")})
    assert len(hit) == 1 and d in hit[0].as_posix() and (hit[0] / "bn42_clean.json").exists(),         f"{arr}: attach only the {d} output (FLAG_65x{'b' if arr == 'clipB3' else 'c'}), not the saved dataset: {hit}"
    log(f"{arr} of the test lists: {hit[0]}")
if F1:
    hit = sorted({q.parent for q in IN.rglob("voice_rn293_train.npy")})
    assert len(hit) == 1 and "flag2027-feats-screen2-p" not in hit[0].as_posix(), f"F1: attach only the FLAG_47b re-run output: {hit}"
    log(f"rn293 / ecapa1024 of the test lists: {hit[0]}")
REFZ = None
if SPLIT == "dev":
    name = f"submission_cleantune01_{'F1_' + os.environ['FINAL_ARM'] if F1 else 'F0'}_dev.zip"
    ref = sorted(mf[0].parent.glob(name)); REFZ = WORK / "r11_reference.zip"
    xd = mf[0].parent / name[:-4]   # Kaggle unzips a zip inside an uploaded dataset zip -> folder of the 4 txt files (EVALPATH-U 4/11)
    if not ref and xd.is_dir():
        import zipfile
        tx = sorted(xd.rglob("*.txt")); assert len(tx) == 4, f"R11: extracted reference {xd} has {len(tx)} txt files"
        with zipfile.ZipFile(REFZ, "w", zipfile.ZIP_DEFLATED) as zr:
            for f in tx: zr.write(f, f.relative_to(xd).as_posix())
        log(f"R11 reference rebuilt from the extracted folder {xd}")
    else:
        assert ref, f"R11: {name} next to the saved models is the reference"
        shutil.copy(ref[0], REFZ)
'''

R11 = r'''
# R11 (dev only): re-extracted faces / voices + saved heads vs the zip of the run that trained them (fusers refitted on the same tables)
if REFZ is not None:
    import io, zipfile
    import numpy as np                                             # 2/11: 112-f0x-r11 v1 stopped here (NameError) after its zip
    from scipy.stats import spearmanr
    with zipfile.ZipFile(ZP[0]) as a, zipfile.ZipFile(REFZ) as b:
        only = os.environ.get("R11_ONLY", "")   # 5/11 EVALPATH-U: F0s's dev zip used the OLD e2e-dev S6 / T3 (no CLEANCOMP) -> f0sx checks English only
        for n in sorted(x for x in b.namelist() if x.endswith(".txt")):
            if only and only not in n:
                log(f"R11 {n}: skipped (R11_ONLY={only}: the reference was built with other components)"); continue
            x = pd.read_csv(io.BytesIO(a.read(n)), sep=" ", header=None, dtype={0: str})
            y = pd.read_csv(io.BytesIO(b.read(n)), sep=" ", header=None, dtype={0: str})
            assert (x[0].values == y[0].values).all(), n
            dd, rho = np.abs(x[1].values - y[1].values), spearmanr(x[1].values, y[1].values)[0]
            log(f"R11 {n}: max |diff| {dd.max():.1e}, mean {dd.mean():.1e}, Spearman {rho:.6f}{' (same bytes)' if a.read(n) == b.read(n) else ''}")
            assert rho >= float(os.environ.get("R11_MIN_RHO", "0.9999")), (n, rho)
    log("R11 PASSED: the Evaluation path reproduces the dev zip")
'''

INTRO = """
# FLAG 2027 — NB-112 {which}: CLEANTUNE-01 end-to-end head + dev zip

**What (Experiment/CLEANTUNE-01/README.md §3, registered 31/10).** {what}
Bengali-clean rows (`BN42=1`) and FLAG_109 adapters only (arrays marked `bn42_clean.json`). No CSLS, no AVSpeech / LRS3, no dev
data in any fit (the dev lists are only scored). **Inputs:** DSC datasets + FLAG_109 output(s) + the fold tables of `{tag}`{more}.
"""


def build(which, dec=None):
    ev, which = which.endswith("x"), which[:2]
    if which == "f0":
        tag, group, arm, seeds, more = "ct01g1", "g1", "B0c", 5, ""
        what = ("F0 = CLEAN-B0c: the B0c head (= the B3 recipe on the clean rows) on every train person, 5 seeds; Bangla fuser "
                "S6 + v11 T3 + trB0c fitted on the round-1 g1 tables; English = CLEAN-EN-01's form (trB0c + ImageBind + age).")
    else:
        tag, group, arm, seeds, more = "ct01r2", "r2", dec["final_arm"], dec["seeds"], " + the F0 zip"
        what = (f"F1: the round-2 arm `{arm}` ({dec['final_spec']}) on every train person, {seeds} seeds; Bangla fuser S6 + v11 T3 + "
                f"tr{arm} fitted on the round-2 tables; English files byte-identical to F0.")
    e = B110.env(tag, group) + (f'\nos.environ.setdefault("FINAL_ARM", "{arm}")\nos.environ.setdefault("CT_FINAL_SEEDS", "{seeds}")'
                                + ('\nos.environ["CT_F1"] = "1"' if which == "f1" else ""))
    if ev:
        what = ("**Evaluation path (2/11): no training.** " + what + " The saved final models (attached f0 / f1 output) score the test "
                "files of `E2E_SPLIT` (dev = R11 rehearsal, compared with that run's zip; `eval` on 11/11 with FLAG_50's output).")
        more += " + the f0 / f1 output (saved models) + FLAG_65xb (+ 65xc, FLAG_47b re-run for F1) outputs, not the saved datasets"
    cells = [("markdown", INTRO.format(which=which.upper() + ("X" if ev else ""), what=what, tag=tag, more=more)), ("code", e),
             ("markdown", "## 1. Code (embedded) + data links + SAFE-04 reference tables"), ("code", B110.SETUP),
             ("markdown", "## 2. Resume (optional) + the round's fold tables"), ("code", BL.RESUME), ("code", GATHER_T),
             ("markdown", "## 3. Final head + zip"), ("code", FINAL_ZIP),
             ("markdown", "## 4. Summary"), ("code", BL.SUMMARY)] if not ev else [("markdown", INTRO.format(which=which.upper() + "X", what=what, tag=tag, more=more)), ("code", e),
             ("markdown", "## 1. Code (embedded) + data links + SAFE-04 reference tables"), ("code", B110.SETUP),
             ("markdown", "## 2. Saved models, re-extracted test arrays, the round's fold tables"), ("code", EVALPATH), ("code", GATHER_T),
             ("markdown", "## 3. Saved head on the test files + zip"), ("code", FINAL_ZIP),
             ("markdown", "## 3b. R11 (dev): vs the zip of the run that trained the models"), ("code", R11),
             ("markdown", "## 4. Summary"), ("code", BL.SUMMARY)]
    nb = f"FLAG_112_{which}{'x' if ev else ''}{os.environ.get('NBSUF', '')}.ipynb"   # NBSUF: opt-in (CLEANTUNE-02 F2s = _f2s), default unchanged
    BL.B12.nb(cells, nb)
    print(f"{nb} built ({arm}, {seeds} seeds, tables {tag})")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "f0"
    build(w, json.loads(Path(sys.argv[2]).read_text()) if w.startswith("f1") else None)
