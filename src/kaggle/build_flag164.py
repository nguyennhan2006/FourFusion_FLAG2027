r"""FLAG_164 (3/11, nhat ky 362): CLEANCOMP-01 (Experiment/CLEANCOMP-01/README.md): the SAFE-04 fold tables of F1's S6 bridge, rel / relg
and v11's T3 head, learned again under LID_STRICT=1. FLAG_48-style lab notebook (FLAG_48 code + CLEANCOMP-01/run.py), one process per fold.
    python kaggle/build_flag164.py                 -> kaggle/FLAG_164_cleancomp01.ipynb   (a = fold tables)
    python kaggle/build_flag164.py final           -> kaggle/FLAG_164f_cleancomp01_final.ipynb (b = final artefacts, see README section 4)
Inputs: ngnnguynchn/flag2027-lab-data-p + flag2027-eda-bundle-p + ngnnguynchn/flag2027-lid-strict-p (v2). Env: LID_STRICT=1;LAB_FOLDS=s."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_lab_notebook as L
import build_eval_notebooks as B12

EXP = "CLEANCOMP-01"
EXTRA = ["Experiment/CLEANCOMP-01/run.py"]
if len(sys.argv) > 1 and sys.argv[1] == "final":
    EXTRA += ["Experiment/CLEANCOMP-01/final.py", "Experiment/E2E/v11_head.py", "BEST/v7_safe/safe_lib.py"]
blob, md5 = L.code_blob(EXTRA)
setup = L.SETUP.replace("__CODE_B64__", blob).replace("__CODE_MD5__", json.dumps(md5, indent=1)).replace(
    "__DATA_DIRS__", json.dumps(L.DATA_DIRS)).replace("__OPTIONAL_DIRS__", json.dumps(L.OPTIONAL_DIRS))
env = L.ENV.replace('os.environ.get("LAB_EXP", "VBSEL-01")', f'os.environ.get("LAB_EXP", "{EXP}")')
assert f'"{EXP}"' in env
mark = lambda name, code: f'print("[cell] {name}", flush=True)\n' + code.lstrip("\n")
INTRO = """
# FLAG 2027 - NB-164: CLEANCOMP-01, F1's S6 / rel / relg / v11 T3 head under LID_STRICT=1 (GPU T4)

Registered 3/11 (Experiment/CLEANCOMP-01/README.md): same code and seeds as the SAFE-04 tables of the old components; the only
change is the Bengali list (lid_strict.json: VoxCeleb2 'vox2' for the T3 head, MAV 'mav_files' for the S6 bridge and the rel anchors).
One process per fold (LAB_FOLDS), table `cleancomp01_fold<s>.parquet`. Inputs: flag2027-lab-data-p, flag2027-eda-bundle-p, flag2027-lid-strict-p.
"""
RUN_FINAL = r"""
# the final artefacts (Experiment/CLEANCOMP-01/final.py): one process, lab_entry.py CLEANCOMP-01 with CC_MODE=final
FIN = OUTD / "final"; FIN.mkdir(exist_ok=True)
t0 = time.time()
p = subprocess.Popen([sys.executable, "-u", str(LAB / "lab_entry.py"), EXP], cwd=str(WORK), env=dict(ENVV, CC_MODE="final", CC_OUT=str(FIN)),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
tail = []
for line in p.stdout:
    tail = (tail + [line])[-40:]
    if "Warning" not in line and "warnings.warn" not in line:
        print(line, end="", flush=True)
if p.wait() != 0:
    print("".join(tail)); raise RuntimeError(f"{EXP} final failed")
need = ["s007_xall.pt", "final_info.json", "v11_head/v11_t3head.pt", "v11_head/t3_streams.npz"]
assert all((FIN / n).exists() for n in need), [n for n in need if not (FIN / n).exists()]
print("[final] artefacts:", {n: (FIN / n).stat().st_size for n in need}, f"[{time.time() - t0:.0f}s]", flush=True)
"""
FINAL_MODE = len(sys.argv) > 1 and sys.argv[1] == "final"
run_cell, name = (RUN_FINAL, "FLAG_164f_cleancomp01_final.ipynb") if FINAL_MODE else (L.RUN, "FLAG_164_cleancomp01.ipynb")
cells = [("markdown", INTRO), ("code", mark("env", env)), ("markdown", "## 1. Code + data links"), ("code", mark("setup", setup)),
         ("markdown", "## 2. Resume"), ("code", mark("resume", L.RESUME)), ("markdown", "## 3. " + ("Final artefacts" if FINAL_MODE else "Folds")),
         ("code", mark("final-artefacts" if FINAL_MODE else "folds", run_cell)),
         ("markdown", "## 4. Check"), ("code", mark("check", L.FINAL))]
B12.nb(cells, name)
print(f"{name}: {len(md5)} code files ({len(blob) / 1e3:.0f} kB base64)")
