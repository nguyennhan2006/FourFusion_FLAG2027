"""11/11 Evaluation run: two dev-only reference checks crashed on the Evaluation lists (EXPERIMENT_LOG 403).

  FLAG_50 cell "drift of the fresh T3 arrays": compared the Evaluation arrays with the dev components of
      flag2027-e2e-dev (392 vs 1008 rows) -> skipped when ONLY_UNDER is set, and when the shapes differ.
  FLAG_65x* cell "B3-EVAL: re-extracted arrays vs the saved ones": compared the Evaluation test-split arrays with
      the saved DEV arrays of the same name (716 vs 1468 rows) -> with ONLY_UNDER set only the train / ext_*
      arrays are compared (the adapter check stays), the test-split arrays are reported as "no saved reference".

ONLY_UNDER = None (dev, R11) leaves both cells exactly as before. Idempotent: run again = no change.
    python kaggle/patch_evalphase_checks.py
"""
import json
from pathlib import Path

K = Path(__file__).parent
FIXES = [
    ('if E2EDEV is not None and any(k.startswith("no_gender/") for k in S):',
     'if E2EDEV is not None and any(k.startswith("no_gender/") for k in S) and not ONLY_UNDER:   # dev only (11/11)'),
    ("            if p.exists():\n                T3DRIFT[f\"{f['key']} {kind}_{n}\"]",
     "            if p.exists() and np.load(p, mmap_mode=\"r\").shape == zc[f\"{kind}_{n}_{f['i']}\"].shape:\n"
     "                T3DRIFT[f\"{f['key']} {kind}_{n}\"]"),
    ('    r = ref_dir / f.name\n    if not r.exists():',
     '    r = ref_dir / f.name\n'
     '    if ONLY_UNDER and not (f.name.startswith("ext_") or f.name.endswith("_train.npy")):   # Evaluation lists (11/11)\n'
     '        chk[f.name] = "Evaluation list: no saved reference"; continue\n'
     '    if not r.exists():'),
]
NBS = ["FLAG_50_e2e_v11.ipynb", "FLAG_65x_clipb3_extract.ipynb", "FLAG_65xb_clipb3c_extract.ipynb", "FLAG_65xc_clipc5c_extract.ipynb"]
PYS = ["build_e2e_v11.py", "build_facelora3_notebook.py"]


def fix(src):
    n = 0
    for old, new in FIXES:
        if new.split("\n")[0] in src and old not in src.replace(new, ""):
            continue                     # already patched
        c = src.count(old)
        if c:
            src = src.replace(old, new); n += c
    return src, n


for name in NBS:
    p = K / name
    nb = json.loads(p.read_text(encoding="utf-8")); tot = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s, n = fix("".join(cell["source"]))
        if n:
            cell["source"] = s.splitlines(keepends=True); tot += n
            compile(s, name, "exec")     # the patched cell still parses
    if tot:
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{name}: {tot} replacement(s)")
for name in PYS:
    p = K / name
    s, n = fix(p.read_text(encoding="utf-8"))
    if n:
        p.write_text(s, encoding="utf-8")
    print(f"{name}: {n} replacement(s) (builder source)")
