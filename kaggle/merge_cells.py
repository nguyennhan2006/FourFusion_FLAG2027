"""Per-cell merge of already-scored submission zips (no retraining).

usage: python merge_cells.py OUT.zip ng_en=A.zip ng_bn=B.zip g_en=C.zip g_bn=D.zip
Each cell file is copied verbatim, so every source keeps its own (lower = same) orientation.
"""
import sys, zipfile
CELL = {"ng_en": "no_gender/sub_score_v4_English_heard.txt", "ng_bn": "no_gender/sub_score_v4_Bangla_unheard.txt",
        "g_en": "gender/sub_score_v4_English_heard.txt", "g_bn": "gender/sub_score_v4_Bangla_unheard.txt"}
out, pairs = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
assert set(pairs) == set(CELL), f"need all of {sorted(CELL)}"
with zipfile.ZipFile(out, "w") as zo:
    for k, fn in CELL.items():
        data = zipfile.ZipFile(pairs[k]).read(fn)
        assert len(data.splitlines()) > 100, (k, pairs[k])
        zo.writestr(fn, data)
        print(f"{k:6s} <- {pairs[k]}")
print("wrote", out)
