r"""BIGB-01 (registered 08/10 in README.md): the T3 head with more identities per InfoNCE batch, plus step-matched controls.
Reuses ALLNEG-01/run.py (same data, rows, L0 streams, CA-FUSE gate, purge); only the arm list, tag and decision arm differ.
    VOX_LID=<FLAG_42 lid_summary.json> FOLDS=1,2,3,4,5 python run.py ; python run.py report > report.txt
"""
import os, sys, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("allneg01run", HERE.parents[0] / "ALLNEG-01/run.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
assert A.LID, "BIGB-01 is registered on the Bengali-filtered VoxCeleb2 rows: set VOX_LID"
A.TAG, A.DECISION = "bigb01f", "P512"
A.ARMS = {"B": dict(), "P256": dict(P=256, K=2), "B2x": dict(steps=2464), "P512": dict(P=512, K=2), "P256x2": dict(P=256, K=2, steps=2464),
          "P1024": dict(P=1024, K=2)}
if os.environ.get("AN_ARMS"):
    A.ARMS = {k: v for k, v in A.ARMS.items() if k in os.environ["AN_ARMS"].split(",")}


def report():
    import pandas as pd
    from stream_gate import load
    D, P = load(A.TAG), load("allneg01f")
    assert (D.y.values == P.y.values).all() and (D.vi.values == P.vi.values).all()
    if "anP256_deep" in D:
        print(f"wiring: corr(anP256_deep, ALLNEG-01 anP256_deep) = {np.corrcoef(D.anP256_deep, P.anP256_deep)[0, 1]:.4f} "
              "(1.0000 on the local CPU; a GPU run is not bit-identical, expect >= 0.99)")
    A.report()
    if "anB_deep" in D:
        print("\n######## sensitivity: reference = this run's own B (same device), not v10 (registered gate above is vs v10)")
        A.REF_FROM_B = True
        A.report()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        for s in [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]:
            A.run(s)
