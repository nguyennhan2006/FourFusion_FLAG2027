"""Pull the exact notebook (+ kernel-metadata.json + log) of Kaggle runs that are not in the FACTOR_ENFUSE package, so that
stages.py can embed the code that actually ran (src "nb:<slug>").

    python release/flag2027_e2e/pull_runs.py owner/slug [owner/slug ...]     -> release/flag2027_e2e/src_nb/<slug>/

Credentials stay inside kaggle/kapi.py (never printed). The account used is the run's owner."""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kaggle"))
import kapi  # noqa: E402

DEST = HERE / "src_nb"
accs = kapi.accounts()
for k in sys.argv[1:]:
    owner, slug = k.split("/")
    acct = owner if owner in accs else next(iter(accs))
    d = DEST / slug
    if (d / "kernel-metadata.json").exists() and list(d.glob("*.ipynb")):
        print("have", k); continue
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    rc, t = kapi.cli(acct, ["kernels", "pull", k, "-p", str(d), "-m"], accs, timeout=600)
    o = d / "out"; o.mkdir()
    rc2, t2 = kapi.cli(acct, ["kernels", "output", k, "-p", str(o), "-o", "--file-pattern", r".*\.log$"], accs, timeout=1800)
    for lg in o.glob("*.log"):
        shutil.move(str(lg), d / f"{slug}.log")
    shutil.rmtree(o, ignore_errors=True)
    nb = list(d.glob("*.ipynb"))
    m = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8")) if (d / "kernel-metadata.json").exists() else {}
    print(k, "ok" if (rc == 0 and nb) else f"FAILED {(t + t2)[-300:]}", "| datasets", m.get("dataset_sources"), "| kernels", m.get("kernel_sources"),
          "| gpu", m.get("enable_gpu"), "| log", (d / f"{slug}.log").exists())
