r"""FLAG_111 (31/10): CLEANTUNE-01 round 2 (Experiment/CLEANTUNE-01/README.md §2, §6c): the arms of ct_round2.json (written by the
round-1 report, FLAG_110r) with fresh seeds, a 'b' copy of every candidate (CT20), the in-run control B0c; with the 5 folds the
report: SEL2 vs B0c (system rule), STACK, SEED -> ct_decision.json. Needs Experiment/CLEANTUNE-01/ct_round2.json (embedded).
    python kaggle/build_flag111.py
GPU runs with --env 'LAB_FOLDS=1,2' / '3,4' / '5', then a CPU run with --env 'LAB_FOLDS=' and the three outputs' tables attached.
Smoke (Kaggle CPU): --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1'."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PLAN = HERE.parent / "Experiment/CLEANTUNE-01/ct_round2.json"
assert PLAN.exists(), "run the round-1 report first (FLAG_110r) and copy ct_round2.json into Experiment/CLEANTUNE-01/"
import build_e2e_lab as BL
import build_flag110 as B110
from build_flag93 import GATHER, REPORT

plan = json.loads(PLAN.read_text())
assert plan["run_round2"], "the round-1 rules say: no round 2 (final system = B0c, F0 only)"
assert "Experiment/CLEANTUNE-01/ct_round2.json" in B110.CODE
sys.path.insert(0, str(HERE.parent / "Experiment/CLEANTUNE-01"))
I, SA = plan["I"], plan["S_A"]
spec = {"B0c": ("B3c", ())}
seen = {("B3c", ())}
for name, key in (("I", (I, ())), ("S1", ("B3c", tuple(SA[:1]))), ("CA", (I, tuple(SA))), ("C2", (I, tuple(SA[:2])))):
    if key not in seen:
        spec[name] = key; seen.add(key)
ARMS = [a for a in spec] + [a + "b" for a in spec if a != "B0c"]
TAG = "ct01r2"
G3 = I != "B3c"

INTRO = f"""
# FLAG 2027 — NB-111: CLEANTUNE-01 round 2 (combinations; fresh seeds)

**Plan (ct_round2.json, from the round-1 rules):** I* = `{I}`, S_A = {SA}. Arms: {json.dumps({k: [v[0], list(v[1])] for k, v in spec.items()})}
(+ a 'b' copy with other seeds of every candidate). Decision **SEL2** (nested) vs **B0c**: g >= +0.3, >= 4/5 folds, held-out
>= -0.2; STACK and SEED rules in `Experiment/CLEANTUNE-01/README.md` (registered before any result). BN42=1, FLAG_109 adapters only.
**Runs:** `--env 'LAB_FOLDS=1,2'`, `'3,4'`, `'5'` (GPU T4 x2), then `--env 'LAB_FOLDS='` (CPU) with the fold tables attached.
**Inputs:** DSC + FLAG_109 b3c{' + c5c + tap01-voice + screen2 + aux-voice-feats' if G3 else ''} + `flag2027-facelora03-tables-p`. Internet OFF.
"""

cells = [("markdown", INTRO), ("code", B110.env(TAG, "r2")),
         ("markdown", "## 1. Code (embedded, with ct_round2.json) + data links + SAFE-04 reference tables"), ("code", B110.SETUP),
         ("markdown", "## 2. Resume (optional) + fold tables of the other runs"), ("code", BL.RESUME), ("code", GATHER),
         ("markdown", "## 3. Jobs: the folds of LAB_FOLDS, one per GPU at a time"), ("code", BL.JOBS),
         ("code", B110.GROUP_STANDIN.replace("__ARMS__", repr(tuple(ARMS)))),
         ("markdown", "## 4. Registered round-2 rules (all 5 folds) -> ct_decision.json"), ("code", REPORT),
         ("code", 'import shutil\nfor n in ("ct_decision.json",):\n    f = SP / "out/safe04" / n\n    if f.exists(): shutil.copy(f, OUTD / n); print(f.read_text())'),
         ("markdown", "## 5. Summary"), ("code", BL.SUMMARY)]

if __name__ == "__main__":
    BL.B12.nb(cells, "FLAG_111_ct01_r2.ipynb")
    print(f"FLAG_111_ct01_r2.ipynb: arms {ARMS}, framework {'g3' if G3 else 'g1'}")
