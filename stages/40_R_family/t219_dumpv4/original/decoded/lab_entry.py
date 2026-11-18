"""Kaggle entry of the E2E lab: python lab_entry.py <experiment dir> [folds | report | final | zip]
Same code path as on the local machine, with one change: SAFE-04's ECAPA-6144 voice stream is a (rows, 0) placeholder in
the data bundle and is removed here, so an experiment that uses it fails loudly instead of running on empty features."""
import os, sys, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parent
run_py = ROOT / "Experiment" / sys.argv[1] / "run.py"
os.environ["SAFE04_NOMAIN"] = "1"
sys.path.insert(0, str(ROOT / "Experiment/SAFE-04"))
import rel_sweep5 as R
assert R.VOICE["e6144"][0].shape[1] == 0, "the data bundle should carry the ECAPA-6144 placeholder"
del R.VOICE["e6144"]
sys.argv = [str(run_py)] + sys.argv[2:]
runpy.run_path(str(run_py), run_name="__main__")
