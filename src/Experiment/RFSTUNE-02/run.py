"""RFSTUNE-02 (README registered 12/11 06:40 before any number): round 2 around d1, report = RFSTUNE-01/run.py with these parameters.
    python lab_entry.py RFSTUNE-02 report        (Kaggle CPU, FLAG_176; tables rfu<arm>lab_fold<s> of FLAG_175 in <SP>/out/safe04)
Q1 d1 - a0 (repeat), Q2 f6v5 / f8v3 - a0 (which dropout), Q3 SEL2 - d1 (better point); noise floor d1r - d1. Validation only."""
import importlib.util, sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("rfstune01", Path(__file__).resolve().parent.parent / "RFSTUNE-01" / "run.py")
M = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M)
M.PREFIX, M.TAG, M.CONTROL = "rfu", "rfstune2", "a0"
M.ARMS = ["a0", "d1", "d1r", "f6v5", "f8v3", "f7v4", "f5v2", "f4v1", "d1p1", "d1s2", "d1l1"]   # = kaggle/build_flag173.py ARMS2
M.NOISE = ("d1r", "d1")
M.PAIRS = [("d1", "a0"), ("f6v5", "a0"), ("f8v3", "a0"), ("f6v5", "d1"), ("f8v3", "d1"), ("f7v4", "d1"), ("f5v2", "d1"), ("f4v1", "d1"),
           ("d1p1", "d1"), ("d1s2", "d1"), ("d1l1", "d1")]
M.SEL_REFS = ["d1", "a0"]

if __name__ == "__main__":
    M.report()
