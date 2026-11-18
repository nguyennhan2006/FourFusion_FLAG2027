"""Build the FourFusion 16.18 end-to-end bundle from orch.py (orchestrator), viz.py (plots) and stages.py (stage table).

    python release/flag2027_e2e/build_notebook.py        # needs the package E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl (FF_PKG)

Outputs (release/flag2027_e2e/):
  FourFusion_1618_kaggle.ipynb   the notebook to run on Kaggle (< 1 MB: the Kaggle API refuses notebooks above ~1 MB);
                                 each stage's code comes from the code dataset, checked by sha256
  FourFusion_1618_full.ipynb     the same notebook with every stage's code inline (reading copy; runs locally too)
  code/                          the code dataset: <stage>.ipy (original cells, patched, separated by CELL_SEP) +
                                 ff_e2e_code_manifest.json (sha256 per stage) + README.md
In the full notebook each original cell is `print("[cell] ...") / keep_code(...) / if False: <original cell, indented>`;
keep_code writes the block into the stage file. run(<stage>) executes the stage file in a fresh process."""
import hashlib
import importlib.util
import json
import pprint
import re
import shutil
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ff_stages", HERE / "stages.py")
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)
CODE_DIR = HERE / "code"
CELL_SEP = "\n# <<<FF-CELL>>>\n"          # must equal orch.CELL_SEP
J = lambda x: "".join(x) if isinstance(x, list) else x
KIND = {"infer": "suy luận (Evaluation)", "train": "học trọng số / bảng", "data": "trích đặc trưng từ media"}
assert CELL_SEP in (HERE / "orch.py").read_text(encoding="utf-8").replace('"\\n# <<<FF-CELL>>>\\n"', CELL_SEP)


def cell_title(src):
    first = src.lstrip().splitlines()[0] if src.strip() else ""
    m = re.match(r'print\("\[(?:cell|boot)\] ?([^"]*)"', first)
    return (m.group(1) if m else first)[:90]


def stage_cells(st):
    """Original code cells of a stage with the patches applied -> (sources, notes on what changed)."""
    notes = []
    if st["src"] == "new":
        srcs = [s.replace("\r\n", "\n") for s in st["cells"]]
    else:
        if st["src"].startswith("nb:"):
            nb = json.loads(next((ST.SRC_NB / st["src"][3:]).glob("*.ipynb")).read_text(encoding="utf-8"))
        else:
            nb = json.loads((ST.PKG / "notebooks" / st["src"][4:]).read_text(encoding="utf-8"))
        srcs = []
        for c in nb["cells"]:
            if c["cell_type"] != "code" or not J(c["source"]).strip():
                continue
            s = J(c["source"]).replace("\r\n", "\n")
            if any(re.search(rx, cell_title(s)) for rx in st.get("drop_cells", [])):
                notes.append(f"bỏ cell gốc `{cell_title(s)}`")
                continue
            srcs.append(s)
    for old, new, count in ST.GLOBAL_PATCHES + st.get("patches", []):
        hits = sum(s.count(old) for s in srcs)
        if count is not None:
            assert hits == count, f"{st['name']}: patch {old[:60]!r} found {hits} times, expected {count}"
        if hits:
            srcs = [s.replace(old, new) for s in srcs]
            notes.append(f"thay `{old[:80]}` → `{new[:80]}` ({hits} chỗ)")
    return srcs, notes


def table(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


# ---------------------------------------------------------------- code dataset (one file per stage, checked by sha256)
STAGE_CODE = {}
shutil.rmtree(CODE_DIR, ignore_errors=True)
CODE_DIR.mkdir()
manifest = {}
for st in ST.S:
    srcs, notes = stage_cells(st)
    text = "".join(s.rstrip("\n") + CELL_SEP for s in srcs)
    assert not re.search(r"KGAT_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{30,}", text), f"{st['name']}: chuỗi giống token trong mã stage: không dựng"
    (CODE_DIR / f"{st['name']}.ipy").write_text(text, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    STAGE_CODE[st["name"]] = (srcs, notes, sha)
    manifest[st["name"]] = {"sha256": sha, "cells": len(srcs), "bytes": len(text.encode("utf-8")), "source": st["src"], "changes": notes}
(CODE_DIR / "ff_e2e_code_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
(CODE_DIR / "README.md").write_text(
    "# Mã các stage của notebook FourFusion 16.18 end-to-end\n\nMỗi file `<stage>.ipy` = các cell của notebook Kaggle gốc (nguyên văn, chỉ áp các "
    "thay đổi ghi trong `ff_e2e_code_manifest.json`), ngăn bởi dòng `# <<<FF-CELL>>>`. Notebook `FourFusion_1618_kaggle.ipynb` nạp file theo "
    "sha256; `FourFusion_1618_full.ipynb` chứa cùng mã ngay trong cell.\n", encoding="utf-8")


def build(full):
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    plan_rows = [[f"`{s['name']}`", KIND[s["kind"]], "GPU" if s["gpu"] else "CPU", f"{s['minutes']:.0f}", s["title"]] for s in ST.S]
    md(f"""
# FourFusion · hệ 16.18 (C1FIX-01): một notebook từ dữ liệu tới zip nộp

**Hệ:** FACTOR_ENFUSE (c5rl) với LoRA C1 của ImageBind-audio đã sửa khoá; Evaluation **16.18**
(không ràng buộc giới / English 13.27 · không ràng buộc giới / Bangla 11.72 · cùng giới / English 20.19 · cùng giới / Bangla 19.55).
Mỗi cặp được chấm độc lập (không chuẩn hoá điểm, không cohort, không dùng dev); zip lưu −logit.

| file thi | logit (cao = cùng người) |
|---|---|
| Bangla | F1s′ + RFX |
| English | (bộ trộn Bangla của F1s′ áp lên cặp English) + RFX |

{"**Bản đầy đủ (đọc):** mã của từng stage nằm ngay trong các cell (khối `if False:`)." if full else "**Bản chạy trên Kaggle:** mã của từng stage nằm trong **dataset mã** (`ff_e2e_code_manifest.json`, nạp theo sha256), vì Kaggle API không nhận notebook lớn hơn khoảng 1 MB. Bản `FourFusion_1618_full.ipynb` chứa cùng mã ngay trong cell để đọc."}

## Cách notebook chạy

- Mỗi lượt Kaggle gốc của hệ là một **stage**. Mã của stage là **mã gốc nguyên văn** (chỉ áp các thay đổi ghi ở đầu stage), chạy trong
  một tiến trình riêng bằng `run(<stage>)`.
- Mỗi stage thấy một `/kaggle/input` riêng (đúng bố cục Kaggle, chỉ gồm input của nó) và một `/kaggle/working` riêng. Những gì stage để lại
  là **artifact**, mang đúng tên nguồn Kaggle gốc (`datasets/<chủ>/<tên>`, `notebooks/<chủ>/<tên>`).
- Một artifact được lấy theo thứ tự: làm ra trong phiên này → output của phiên trước của chính notebook này (gắn vào input) →
  **dataset / output trung gian bạn đã gắn sẵn** → chạy stage sinh ra nó. Không gắn gì thì notebook tự tính từ dữ liệu gốc.
- Phiên Kaggle tối đa 12 giờ. Stage nào không kịp thì để phiên sau: gắn **output của phiên này** vào input rồi chạy lại notebook. Chia cho
  nhiều tài khoản chạy song song: đặt `ONLY` khác nhau ở mỗi phiên, phiên cuối gắn output của tất cả.

## Config (cell ngay dưới, hoặc biến môi trường `FF_*`)

| biến | ý nghĩa |
|---|---|
| `MODE` | `"infer"`: chỉ suy luận, không học lại (trọng số / bảng đã học phải gắn vào input); `"train"`: học lại mọi thứ còn thiếu |
| `TARGET` | artifact cần có cuối phiên (mặc định zip nộp `{ST.K['final']}`) |
| `USE_IMPORTED` | `True`: dùng artifact đã gắn sẵn; `False`: bỏ qua mọi thứ đã gắn trừ dữ liệu gốc (tính lại hết) |
| `FORCE` | stage / artifact phải tính lại dù đã có |
| `ONLY` | chỉ chạy các stage này trong phiên (chia việc song song) |
| `TIME_BUDGET_H` | thôi nhận stage mới khi phiên đã dùng gần hết thời gian |
| `SMOKE` | chạy thử với tham số nhỏ của từng notebook gốc (CPU được) |

## Stage

{table(plan_rows, ["stage", "loại", "máy", "phút (lượt gốc)", "việc"])}

**Dữ liệu gốc** (luôn phải gắn): {", ".join(f"`{a}`" for a in ST.RAW)}.
""")
    code(r'''
print("[cell] 00 config", flush=True)
import os
MODE = os.environ.get("FF_MODE", "infer")                 # "infer" | "train"
TARGET = os.environ.get("FF_TARGET", "notebooks/ff/fourfusion-1618-final")
USE_IMPORTED = os.environ.get("FF_USE_IMPORTED", "1") == "1"
FORCE = [x for x in os.environ.get("FF_FORCE", "").split(",") if x]
ONLY = [x for x in os.environ.get("FF_ONLY", "").split(",") if x]
TIME_BUDGET_H = float(os.environ.get("FF_TIME_BUDGET_H", "11.3"))
SMOKE = os.environ.get("FF_SMOKE", "0") == "1"
print({"MODE": MODE, "TARGET": TARGET, "USE_IMPORTED": USE_IMPORTED, "FORCE": FORCE, "ONLY": ONLY, "TIME_BUDGET_H": TIME_BUDGET_H, "SMOKE": SMOKE})
''')
    reg = {s["name"]: {k: s[k] for k in ("kernel", "kind", "gpu", "minutes", "title", "inputs", "exports", "env", "smoke_env", "scratch") if k in s}
           for s in ST.S}
    code('print("[cell] 01 bảng stage", flush=True)\n'
         f"RAW = {pprint.pformat(ST.RAW, width=150)}\n"
         f"K = {pprint.pformat(ST.K, width=150)}\n"
         f"C1FIX_ART, C1FIX_LORA = {ST.C1FIX_ART!r}, {ST.C1FIX_LORA!r}\n"
         f"S = {pprint.pformat(reg, width=150, sort_dicts=False)}\n"
         "print(len(S), 'stage')")
    for f in ("orch.py", "viz.py"):
        for part in re.split(r"^# %% ", (HERE / f).read_text(encoding="utf-8"), flags=re.M)[1:]:
            title, body = part.split("\n", 1)
            code(f'print("[cell] {title.strip()}", flush=True)\n{body}')
    code('''print("[cell] kế hoạch phiên", flush=True)
REFERENCE_ZIPS = {"16.18 (C1FIX-01, đã nộp)": find_ref("submission_FACTOR_ENFUSE_c5rl_C1FIX.zip"),
                  "16.99 (FACTOR_ENFUSE, đã nộp)": find_ref("posteval_FACTOR_ENFUSE_c5rl.zip")}
show_plan()''')

    for st in ST.S:
        srcs, notes, sha = STAGE_CODE[st["name"]]
        inp_rows = []
        for e in st["inputs"]:
            vid, aid = (e if isinstance(e, tuple) else (e, e))
            prod = next((s["name"] for s in ST.S if s["kernel"] == aid or aid in s.get("exports", {})), None)
            how = "dữ liệu gốc" if aid in ST.RAW else (f"stage `{prod}`" if prod else "gắn vào input (chưa có stage sinh ra)")
            inp_rows.append([f"`{aid}`" + ("" if vid == aid else f" (gắn ở `{vid}`)"), how])
        src_txt = ("notebook gốc trong gói `" + st["src"][4:] + "`" if st["src"].startswith("pkg:") else
                   "notebook gốc kéo từ Kaggle (`kernels pull`) `" + st["src"][3:] + "`" if st["src"].startswith("nb:") else "mới (ghi ở dưới)")
        md(f"""
---
## Stage `{st['name']}` · {st['title']}

{st['role']}

**Loại:** {KIND[st['kind']]} · **máy:** {'GPU' if st['gpu'] else 'CPU'} · **lượt gốc:** {st['minutes']:.0f} phút · **mã:** {src_txt},
{len(srcs)} cell, sha256 `{sha[:16]}` · **output:** `{st['kernel']}`{''.join(f', `{a}`' for a in st.get('exports', {}))}

{table(inp_rows, ["input", "nguồn"])}

**Thay đổi so với mã gốc:** {'; '.join(notes) if notes else 'không có'} (ngoài việc `/kaggle/input`, `/kaggle/working` trỏ vào thư mục riêng của stage).
""")
        if full:
            for k, s in enumerate(srcs):
                body = "\n".join(("    " + l) if l else "" for l in s.rstrip("\n").split("\n"))
                t = cell_title(s).replace('"', "'").replace("\\", "/")
                code(f'print("[cell] {st["name"]} · mã cell gốc {k + 1}/{len(srcs)}: {t}", flush=True)\n'
                     f'keep_code("{st["name"]}", append={k > 0})\nif False:\n{body}')
        code(f'print("[cell] {st["name"]}: chạy", flush=True)\nrun("{st["name"]}", "{sha}", {len(srcs)})   # mã stage: {len(srcs)} cell, sha256 {sha[:16]}')
        if st.get("viz"):
            md(f"### Visualize `{st['name']}`")
            code(f'print("[cell] {st["name"]}: visualize", flush=True)\n' + st["viz"].strip("\n"))

    md("---\n## Kết thúc phiên")
    code('''print("[cell] tổng kết phiên", flush=True)
report()
fin = out(TARGET)
if fin is not None and (fin / "submission.zip").exists():
    shutil.copy2(fin / "submission.zip", KOUT / "submission.zip")
    print("zip nộp:", KOUT / "submission.zip")''')
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = HERE / ("FourFusion_1618_full.ipynb" if full else "FourFusion_1618_kaggle.ipynb")
    nbf.write(nb, out)
    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.2f} MB, {len(cells)} cells)")
    return out


k = build(full=False)
build(full=True)
assert k.stat().st_size < 1_000_000, "bản Kaggle vượt ~1 MB: Kaggle API sẽ từ chối"
print(f"code/: {len(manifest)} stage, {sum(v['bytes'] for v in manifest.values()) / 1e6:.2f} MB")
