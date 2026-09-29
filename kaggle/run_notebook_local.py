"""Run a Kaggle notebook's code cells locally, in order, in one namespace (smoke test before handing it over).

    FLAG_LOCAL=1 FLAG_IN=<mini inputs> FLAG_OUT=<dir> FLAG_CACHE=<model cache> python run_notebook_local.py NB.ipynb [--upto N]

IPython shell lines (`!pip ...`) become `pass`; they sit under `if not LOCAL:` in our notebooks anyway.
--jupyter: remove __main__.__file__, as under Jupyter (transformers 5.0 reads it for classes defined in the notebook).
"""
import json, sys, time, traceback

path = sys.argv[1]
upto = int(sys.argv[sys.argv.index("--upto") + 1]) if "--upto" in sys.argv else None
cells = [c for c in json.load(open(path, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
if "--jupyter" in sys.argv:
    del sys.modules["__main__"].__file__
for i, c in enumerate(cells[:upto]):
    src = "".join(c["source"])
    src = "\n".join((ln[:len(ln) - len(ln.lstrip())] + "pass") if ln.lstrip().startswith("!") else ln
                    for ln in src.splitlines())
    t = time.time()
    print(f"\n######## cell {i} ########", flush=True)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except Exception:
        traceback.print_exc()
        print(f"######## cell {i} FAILED after {time.time() - t:.0f}s", flush=True)
        sys.exit(1)
    print(f"######## cell {i} ok in {time.time() - t:.0f}s", flush=True)
print("\nALL CELLS OK")
