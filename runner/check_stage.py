"""Rebuild each stage's stage.ipy from its original notebook + the changes listed in stage.json, and compare (DESIGN 2.2, 2.3).

    python runner/check_stage.py                 # every stage found under the stage roots
    python runner/check_stage.py e163i t163l_orig [--json] [--diff]

Canonical rebuild (the rule every work package follows; same as release/flag2027_e2e/build_notebook.stage_cells):
  1. code cells of the original notebook with a non-empty source, in order; source = "".join(lines), CRLF -> LF;
  2. drop_cells: an int = 0-based index in that list; a string = regex searched in the cell title (the name of a first-line
     print("[cell] ...") / print("[boot] ...")) or in the first line; a dict {"index": i} / {"match": rx};
  3. patches, in order: {"old", "new", "count", optional "regex": true}; `old` must occur exactly `count` times summed over all
     cells (count null = any number) and is replaced in every cell. "cell" / "line_in_cell" are informational; only "cell_only"
     (index in the list after dropping) restricts a patch to one cell;
  4. insert_cells (optional): {"at": index | "start" | "end", "source": "..."};
  5. stage.ipy = "".join(cell.rstrip("\\n") + CELL_SEP for cell in cells), CELL_SEP = "\\n# <<<FF-CELL>>>\\n".
The original notebook is original/<file> (SOURCE.json 'notebook' / 'file' / 'files', else the only *.ipynb in original/). When a
notebook cannot be published (it embeds organisers' dev data), it may live outside the repository: $FF_PRIVATE/<stage id>/<file>
or <FF_DATA root>/_private_originals/<stage id>/<file>.
Every stage.ipy is also decoded (runner/payloads.py): a literal /kaggle/input inside embedded code without
runner_requirements.literal_kaggle_input in stage.json is an error, and embedded data files must match configs/data_map.json.
Stages with origin.exactness 'new' are checked for the cell rule (every cell starts with print("[cell] ...", flush=True); cell
magics such as %%writefile are exempt) and 'local-script' stages for the SHA-256 of the files listed in SOURCE.json.
Exit code 1 when an exact / builder stage does not rebuild, or a required file is missing."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ffcore as C  # noqa: E402

REBUILD_KINDS = ("exact-package", "exact-pulled", "exact-staging", "exact-executed-copy", "builder-regenerated")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def notebook_code_cells(nb_path):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    out = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        s = c.get("source", "")
        s = "".join(s) if isinstance(s, list) else s
        if not s.strip():
            continue
        out.append(s.replace("\r\n", "\n"))
    return out


def apply_rules(cells, sj):
    """-> (cells, notes, errors)."""
    notes, errors = [], []
    drops = sj.get("drop_cells") or []
    keep = []
    for i, s in enumerate(cells):
        hit = None
        fl = s.lstrip().splitlines()[0] if s.strip() else ""
        for d in drops:
            if isinstance(d, int) and d == i:
                hit = f"index {i}"
            elif isinstance(d, str) and (re.search(d, C.cell_title(s)) or re.search(d, fl)):
                hit = f"regex {d!r}"
            elif isinstance(d, dict) and (d.get("index") == i or (d.get("match") and (re.search(d["match"], C.cell_title(s)) or re.search(d["match"], fl)))):
                hit = f"{d}"
        if hit:
            notes.append(f"dropped cell {i} ({hit}): {C.cell_title(s)[:60]!r}")
        else:
            keep.append(s)
    cells = keep
    for k, p in enumerate(sj.get("patches") or []):
        if isinstance(p, (list, tuple)):
            p = {"old": p[0], "new": p[1], "count": p[2] if len(p) > 2 else None}
        old, new, count = p.get("old"), p.get("new", ""), p.get("count", 1)
        # "cell" / "line_in_cell" are informational (they point into the original notebook); a patch applies to every cell, as in
        # release/flag2027_e2e/build_notebook.py. Only "cell_only" (index in the code-cell list after dropping) restricts it.
        idx = [p["cell_only"]] if isinstance(p.get("cell_only"), int) else range(len(cells))
        if p.get("regex"):
            hits = sum(len(re.findall(old, cells[i])) for i in idx if i < len(cells))
        else:
            hits = sum(cells[i].count(old) for i in idx if i < len(cells))
        if count is not None and hits != count:
            errors.append(f"patch {k} found {hits} times, stage.json says {count}: {str(old)[:70]!r}")
        for i in idx:
            if i < len(cells):
                cells[i] = re.sub(old, new, cells[i]) if p.get("regex") else cells[i].replace(old, new)
        notes.append(f"patch {k}: {hits} x {str(old)[:50]!r}")
    for ins in sj.get("insert_cells") or []:
        at = ins.get("at", "end")
        pos = 0 if at == "start" else len(cells) if at == "end" else int(at)
        cells.insert(pos, ins["source"].replace("\r\n", "\n"))
        notes.append(f"inserted cell at {at}")
    return cells, notes, errors


def build_ipy(cells):
    return "".join(s.rstrip("\n") + C.CELL_SEP for s in cells)


def build_recipe(sdir, sj):
    """stage.json "build" with recipe ff-stage-ipy/1 (used by some work packages): the cells of build.sequence in order, {"orig": i}
    = cell i of the original notebook (index in nb["cells"]; code cell, CRLF -> LF), {"new": "cells/<file>"} = a new cell file;
    every patch applies to the original cells only (count = total over them). -> (text, notes, errors, base path)."""
    b = sj["build"]
    base = sdir / b["base"]
    nb = json.loads(base.read_text(encoding="utf-8"))
    by_idx = {}
    for i, c in enumerate(nb.get("cells", [])):
        s = c.get("source", "")
        s = "".join(s) if isinstance(s, list) else s
        if c.get("cell_type") == "code" and s.strip():
            by_idx[i] = s.replace("\r\n", "\n")
    seq, notes, errors = [], [], []
    for item in b["sequence"]:
        if "orig" in item:
            if item["orig"] not in by_idx:
                errors.append(f"build.sequence: original cell {item['orig']} is not a non-empty code cell")
                continue
            seq.append(["orig", by_idx[item["orig"]]])
        elif "new" in item:
            p = sdir / item["new"]
            if not p.exists():
                errors.append(f"build.sequence: missing {item['new']}")
                continue
            seq.append(["new", p.read_text(encoding="utf-8").replace("\r\n", "\n")])
    for k, p in enumerate(sj.get("patches") or []):
        old, new, count = p.get("old"), p.get("new", ""), p.get("count", 1)
        hits = sum(s.count(old) for kind, s in seq if kind == "orig")
        if count is not None and hits != count:
            errors.append(f"patch {k} found {hits} times in the original cells, stage.json says {count}: {str(old)[:70]!r}")
        for x in seq:
            if x[0] == "orig":
                x[1] = x[1].replace(old, new)
        notes.append(f"patch {k}: {hits} x {str(old)[:50]!r}")
    notes.append(f"recipe {b.get('recipe')}: {sum(k == 'orig' for k, _ in seq)} original + {sum(k == 'new' for k, _ in seq)} new cells")
    return "".join(s.rstrip("\n") + C.CELL_SEP for _k, s in seq), notes, errors, base


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def find_original(sdir, sid, src, origin):
    od = sdir / "original"
    names = []
    for key in ("notebook", "file", "notebook_file", "copied_file"):
        if isinstance(src.get(key), str):
            names.append(src[key])
    f = src.get("files")
    if isinstance(f, list):
        for x in f:
            if isinstance(x, str):
                names.append(x)
            elif isinstance(x, dict):
                names += [x[k] for k in ("name", "path", "file", "dest") if isinstance(x.get(k), str)]
    elif isinstance(f, dict):
        names += list(f)
    for key in ("original_notebook", "notebook"):
        if isinstance(origin.get(key), str):
            names.append(origin[key])
    names += [s for s in _strings(src) if s.endswith(".ipynb")]
    names += [s for s in _strings(origin) if s.endswith(".ipynb")]
    cands, seen = [], set()
    roots = [od]
    if os.environ.get("FF_PRIVATE"):
        roots.append(Path(os.environ["FF_PRIVATE"]) / sid)
    for r in C.split_paths(os.environ.get("FF_DATA", "")):
        roots.append(r / "_private_originals" / sid)
    for n in names:
        if not n.endswith(".ipynb"):
            continue
        b = Path(n.replace("\\", "/")).name
        for r in roots:
            p = r / b
            if p.exists() and p not in seen:
                seen.add(p)
                cands.append(p)
    if not cands:
        cands = sorted(od.glob("*.ipynb"))
    return cands


def recorded_hashes(src):
    hx = set()
    for s in _strings(src):
        s = s.strip().lower()
        if re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{64}", s):
            hx.add(s)
    return hx


DMAP = {}                                                        # configs/data_map.json artifacts (set by main)


def embedded_checks(r, st, ipy):
    """Code embedded in stage.ipy as base64 payloads (runner/payloads.py): (1) a literal /kaggle/input in it needs
    stage.json runner_requirements.literal_kaggle_input (the runner's string rewrite cannot reach decoded code; the runner also
    detects it at run time, but the declaration with its 'why' belongs in stage.json); (2) data files carried inside it must match
    their configs/data_map.json 'embedded/<path>' entry (sha256)."""
    import payloads as PL                                        # noqa: E402  (runner/payloads.py)
    lit = PL.scan_stage_code(ipy)
    L = [u for u in lit["uses"] if u["kind"] == "literal"]
    r["literal"] = {"declared": st.literal_kaggle_input, "declared_false": getattr(st, "literal_declared_false", False),
                    "literal_uses": len(L), "env_defaults": lit["env_defaults"]}
    if L and getattr(st, "literal_declared_false", False):
        ex = ", ".join(sorted({f"{u['member'].split('/')[-1]}:{u['line']}" for u in L})[:4])
        r["notes"].append(f"embedded code has {len(L)} literal /kaggle/input use(s) ({ex}); stage.json declares "
                          "runner_requirements.literal_kaggle_input=false with a 'why' (reviewed: unreachable or redirected), honoured by the runner")
    elif L and not st.literal_kaggle_input:
        ex = ", ".join(sorted({f"{u['member'].split('/')[-1]}:{u['line']}" for u in L})[:4])
        r["errors"].append(f"embedded code reads the literal /kaggle/input ({len(L)} use(s): {ex}) but stage.json has no "
                           "runner_requirements.literal_kaggle_input: declare it with a 'why' (stage owner); the runner treats the stage "
                           "as literal at run time meanwhile")
    if lit["env_defaults"]:
        r["notes"].append(f"embedded module default os.environ.get(X, '/kaggle/input') for {lit['env_defaults']}: the runner exports "
                          "X=<the stage's input view> when the stage does not set X")
    for name, (h, _n) in PL.embedded_files(ipy).items():
        e = DMAP.get("embedded/" + name)
        if e is None:
            r["notes"].append(f"embedded data file {name} (sha256 {h[:16]}...) has no configs/data_map.json entry 'embedded/{name}'")
        elif e.get("sha256") != h:
            r["errors"].append(f"embedded data file {name}: sha256 {h[:16]}... != configs/data_map.json {str(e.get('sha256'))[:16]}...")


def check(sid, st, args):
    sdir = st.dir
    r = {"stage": sid, "exactness": st.origin.get("exactness", "?"), "status": "?", "notes": [], "errors": []}
    for f in ("stage.json", st.code_name, "CHANGES.md", "original/SOURCE.json"):
        if not (sdir / f).exists():
            r["errors"].append(f"missing {f}")
    sj = C.jload(sdir / "stage.json")
    try:
        src = C.jload(sdir / "original/SOURCE.json")
    except Exception:
        src = {}
    ipy = (sdir / st.code_name).read_text(encoding="utf-8") if (sdir / st.code_name).exists() else None
    ex = r["exactness"]
    if ex in REBUILD_KINDS and isinstance(sj.get("build"), dict) and sj["build"].get("sequence") and (sdir / sj["build"].get("base", "")).is_file():
        built, notes, errs, base = build_recipe(sdir, sj)
        r["notes"] += notes
        r["errors"] += errs
        hx = recorded_hashes(src) | recorded_hashes(st.origin) | recorded_hashes(sj["build"])
        m5, s2 = md5_file(base), C.sha256_file(base)
        if hx and not (m5 in hx or s2 in hx):
            r["errors"].append(f"original {base.name}: md5 / sha256 not recorded in SOURCE.json / origin / build")
        norm = ipy.replace("\r\n", "\n") if ipy is not None else None
        r["status"] = "NO-STAGE-IPY" if norm is None else ("EXACT (recipe)" if built == norm else "MISMATCH")
        if r["status"] == "MISMATCH":
            r["errors"].append(f"recipe rebuild sha256 {C.sha256_text(built)[:12]} vs stage.ipy {C.sha256_text(norm)[:12]}")
        exp = sj["build"].get("stage_ipy_sha256")
        if exp and norm is not None and C.sha256_text(norm) != exp:
            r["errors"].append(f"stage.ipy sha256 differs from build.stage_ipy_sha256 {exp[:12]}")
        return r
    if ex in REBUILD_KINDS:
        cands = find_original(sdir, sid, src, st.origin)
        if not cands:
            r["errors"].append("no original notebook found (original/*.ipynb, $FF_PRIVATE/<id>/, FF_DATA/_private_originals/<id>/)")
            r["status"] = "NO-ORIGINAL"
            return r
        if len(cands) > 1:
            slug = (st.origin.get("run") or "").split("/")[-1]
            pick = [c for c in cands if slug and slug in c.name]
            r["notes"].append(f"{len(cands)} candidate notebooks, using {(pick or cands)[0].name}")
            cands = pick or cands
        nb = cands[0]
        hx = recorded_hashes(src) | recorded_hashes(st.origin)
        m5, s2 = md5_file(nb), C.sha256_file(nb)
        if hx:
            ok = m5 in hx or s2 in hx or any(m5.startswith(h) or s2.startswith(h) for h in hx if len(h) >= 12)
            r["notes"].append(f"original {nb.name}: md5 {m5[:12]} sha256 {s2[:12]} -> {'matches' if ok else 'NOT among'} the hashes recorded in SOURCE.json / origin")
            if not ok:
                r["errors"].append("original notebook hash not recorded in SOURCE.json / stage.json origin")
        cells, notes, errs = apply_rules(notebook_code_cells(nb), sj)
        r["notes"] += notes
        r["errors"] += errs
        built = build_ipy(cells)
        if ipy is None:
            r["status"] = "NO-STAGE-IPY"
            return r
        norm = ipy.replace("\r\n", "\n")
        if built == ipy:
            r["status"] = "EXACT"
        elif built == norm:
            r["status"] = "EXACT (after CRLF -> LF)"
        elif built.rstrip("\n") == norm.rstrip("\n") or built[:-len(C.CELL_SEP)] == norm:
            r["status"] = "OK (trailing separator differs)"
        else:
            r["status"] = "MISMATCH"
            a, b = built.split(C.CELL_SEP), norm.split(C.CELL_SEP)
            r["errors"].append(f"rebuilt {len(a)} parts vs stage.ipy {len(b)} parts; sha256 {C.sha256_text(built)[:12]} vs {C.sha256_text(norm)[:12]}")
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    r["errors"].append(f"first differing cell: {i}")
                    if args.diff:
                        d = list(difflib.unified_diff(x.splitlines(), y.splitlines(), "rebuilt", "stage.ipy", lineterm="", n=1))
                        r["diff"] = d[:60]
                    break
        r["sha256_stage_ipy"] = C.sha256_text(norm)
    elif ex == "new":
        if ipy is not None:
            bad = []
            for i, c in enumerate(x for x in ipy.split(C.CELL_SEP) if x.strip()):
                fl = c.lstrip("\n").splitlines()[0] if c.strip() else ""
                if fl.startswith("%%"):
                    continue
                if not re.match(r'print\(\s*["\']\[cell\] ', fl):
                    bad.append(i)
            r["status"] = "OK (new code)" if not bad else "CELL-RULE"
            if bad:
                r["errors"].append(f"cells {bad} do not start with print(\"[cell] ...\", flush=True)")
    elif ex == "local-script":
        # every list of dicts in SOURCE.json whose items carry a sha256 and a path-like key (work packages use files / tree_files /
        # provenance_files / scripts with name / path / copy / repo_path / repo_file); look for the file in the stage folder,
        # original/ and the repository root
        n = ok = absent = 0
        for key, val in src.items():
            if not isinstance(val, list):
                continue
            for x in val:
                if not isinstance(x, dict) or not x.get("sha256"):
                    continue
                name = next((x[k] for k in ("name", "path", "file", "copy", "repo_path", "repo_file", "dest") if isinstance(x.get(k), str)), None)
                if not name:
                    continue
                rel = name.replace("\\", "/")
                cands = [sdir / rel, sdir / "original" / Path(rel).name, sdir / Path(rel).name, C.REPO / rel]
                p = next((c for c in cands if c.is_file()), None)
                if p is None:
                    absent += 1
                    continue
                n += 1
                same = C.sha256_file(p) == x["sha256"]
                ok += same
                if not same:
                    r["notes"].append(f"{key}: {rel} sha256 differs from SOURCE.json (patched copy?)")
        r["status"] = (f"OK ({ok}/{n} listed files match" + (f", {absent} not in the repository" if absent else "") + ")" if n and ok == n else
                       "UNCHECKED (no file list with sha256)" if not n else f"CHECK ({ok}/{n} listed files match)")
    else:
        r["status"] = f"UNCHECKED (exactness {ex!r})"
    if ipy is not None:
        embedded_checks(r, st, ipy)
    if r["errors"] and r["status"] in ("EXACT", "EXACT (after CRLF -> LF)"):
        r["status"] += " with problems"
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--diff", action="store_true", help="print the first differing cell as a unified diff")
    a = ap.parse_args(argv)
    R = C.Runner(C.Config(), quiet=True)
    DMAP.update(R.dmap)
    ids = a.stages or [s for s in R.order_ref if s in R.S and not R.S[s].skeleton] + sorted(s for s in R.S if s not in R.order_ref and not R.S[s].skeleton)
    res = []
    for sid in ids:
        if sid not in R.S or R.S[sid].skeleton:
            res.append({"stage": sid, "status": "NO-STAGE-JSON", "errors": ["no stage.json in this repository"], "notes": []})
            continue
        res.append(check(sid, R.S[sid], a))
    if a.json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
    else:
        for r in res:
            print(f"{r['stage']:22s} {r.get('exactness', '-'):22s} {r['status']}")
            for n in r["notes"]:
                print(f"    . {n}")
            for e in r["errors"]:
                print(f"    ! {e}")
            for line in r.get("diff", []):
                print(f"      {line}")
        miss = sum(1 for s in R.order_ref if s in R.S and R.S[s].skeleton)
        print(f"\n{len(res)} stage(s) checked; {miss} stage(s) of configs/pipeline_1354.json still without stage.json")
    bad = [r for r in res if r["status"] in ("MISMATCH", "NO-ORIGINAL", "NO-STAGE-IPY", "CELL-RULE", "NO-STAGE-JSON") or
           (r.get("exactness") in REBUILD_KINDS and r["errors"])]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
