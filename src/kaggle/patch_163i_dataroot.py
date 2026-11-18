"""15/11: make DATA_ROOT of a patched FLAG_163i notebook work with both Kaggle mount layouts, /kaggle/input/datasets/<owner>/<slug>
and /kaggle/input/<slug>. thanhhuyen11/flag-2027-nb-163i-rfsfxm256-eval (15/11 12:00) failed with "expected one trial list, found []"
because that run mounted the old layout. Run after patch_163i_eval.py:
    python kaggle/patch_163i_dataroot.py <patched FLAG_163i notebook>"""
import json
import re
import sys

p = sys.argv[1]
nb = json.load(open(p, encoding="utf-8"))
n = 0
for c in nb["cells"]:
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    m = re.search(r"^DATA_ROOT = Path\('(/kaggle/input/datasets/[^/']+/([^/']+))'\).*$", src, flags=re.M)
    if not m:
        continue
    new = (f"DATA_ROOT = next((q for q in (Path('{m.group(1)}'), Path('/kaggle/input/{m.group(2)}')) if q.exists()), Path('{m.group(1)}'))"
           "   # Evaluation lists, both mount layouts (patch_163i_dataroot.py)")
    src = src[:m.start()] + new + src[m.end():]
    c["source"] = src.splitlines(keepends=True)
    n += 1
assert n == 1, f"DATA_ROOT line (new mount layout) found {n} times"
json.dump(nb, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DATA_ROOT patched for both mount layouts")
