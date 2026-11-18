#!/usr/bin/env python3
"""Regenerate the generated parts of the documentation from the configuration and the stage folders (no data, no network).

    python tools/build_docs.py            # docs/STAGES.md, docs/CHANGES_FROM_ORIGINAL.md and the generated tables of
                                          # DATA.md, CHECKPOINTS.md, LICENSES.md (between <!-- BEGIN GENERATED: x --> markers)
    python tools/build_docs.py --check    # exit 1 when a generated part is out of date (nothing written)

Sources: configs/data_map.json (artifacts), stages/**/stage.json (consumers, producers), tools/fetch_models.py MODELS (pretrained
models), every stages/**/CHANGES.md, runner/ffrun.py docs (stage table). Hand-written text outside the markers is kept.
Written by the integration fixer, 18/11/2026 (INTEGRATION_r1 B2)."""
import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def stages():
    out = {}
    for p in sorted(glob.glob(str(REPO / "stages/*/*/stage.json"))) + sorted(glob.glob(str(REPO / "tools/stages/*/*/stage.json"))):
        d = jload(p)
        out[d["id"]] = (Path(p).parent, d)
    return out


def consumers(S):
    c = {}
    for sid, (_d, s) in S.items():
        for e in s.get("inputs", []) or []:
            a = e if isinstance(e, str) else (e[1] if isinstance(e, (list, tuple)) else e.get("artifact"))
            c.setdefault(a, set()).add(sid)
    return c


def short(h, n=16):
    return f"`{h[:n]}…`" if isinstance(h, str) and len(h) > n else (f"`{h}`" if h else "")


def esc(s):
    return str(s or "").replace("|", "\\|").replace("\n", " ")


def data_table(A, C, classes):
    rows = ["| artifact id | class | path under FF_DATA | producing stage | private | sha256 (zip / file) | consumers |",
            "|---|---|---|---|---|---|---|"]
    for a, e in sorted(A.items(), key=lambda kv: (classes.index(kv[1].get("class")) if kv[1].get("class") in classes else 99, kv[0])):
        if e.get("class") not in classes:
            continue
        h = e.get("zip_sha256") or e.get("sha256") or ""
        if not h and isinstance(e.get("files"), dict) and e["files"]:
            k = sorted(e["files"])[0]
            h = f"{k}={e['files'][k]}"
        hs = short(h) if h and "=" not in h else (f"`{h.split('=')[0]}` {short(h.split('=', 1)[1])}" if h else "")
        cs = sorted(C.get(a, []))
        rows.append(f"| `{a}` | {e.get('class')} | {('`' + e['path'] + '`') if e.get('path') else '-'} | {e.get('producer') or '-'} | "
                    f"{'yes' if e.get('private') else 'no'} | {hs} | {esc(', '.join(cs[:6]) + (' …' if len(cs) > 6 else ''))} |")
    return "\n".join(rows)


def checkpoint_table(A, C, S):
    rows = ["| artifact id | path under FF_DATA | producing stage (kind, device, minutes of the original run) | files with sha256 | used by | status |",
            "|---|---|---|---|---|---|"]
    for a, e in sorted(A.items()):
        if e.get("class") != "checkpoint":
            continue
        p = e.get("producer")
        st = S.get(p, (None, {}))[1] if p else {}
        comp = st.get("compute") or {}
        prod = f"{p} ({st.get('kind', '?')}, {str(comp.get('device', '?')).upper()}, {comp.get('minutes_original', '?')})" if p else "-"
        fl = e.get("files") or {}
        fs = "; ".join(f"`{k}` {short(v)}" for k, v in sorted(fl.items())[:3]) if isinstance(fl, dict) else ""
        if not fs and e.get("zip_sha256"):
            fs = f"zip {short(e['zip_sha256'])}"
        status = e.get("status") or ("saved copy expected on Drive" if e.get("path") else "made in the session")
        rows.append(f"| `{a}` | {('`' + e['path'] + '`') if e.get('path') else '-'} | {esc(prod)} | {esc(fs)} | "
                    f"{esc(', '.join(sorted(C.get(a, []))))} | {esc(status)[:160]} |")
    return "\n".join(rows)


def license_table():
    spec = importlib.util.spec_from_file_location("ff_fetch_models", REPO / "tools/fetch_models.py")
    m = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(m)
    rows = ["| model | group | source (pinned) | licence (as recorded in tools/fetch_models.py) | used by |", "|---|---|---|---|---|"]
    for x in m.MODELS:
        src = x.get("url") or x.get("repo") or x.get("zip") or x.get("name")
        if x.get("repo") and x.get("revision"):
            src = f"{x['repo']}@{x['revision']}"
        rows.append(f"| `{x['name']}` | {x.get('group', '')} | {esc(src)[:110]} | {esc(x.get('license', 'not recorded'))} | {esc(x.get('used_by', ''))[:140]} |")
    return "\n".join(rows)


def fill(path, key, body, check):
    p = REPO / path
    t = p.read_text(encoding="utf-8")
    a, b = f"<!-- BEGIN GENERATED: {key} -->", f"<!-- END GENERATED: {key} -->"
    i, j = t.find(a), t.find(b)
    assert i >= 0 and j > i, f"{path}: markers for {key} not found"
    new = t[:i + len(a)] + "\n" + body.rstrip("\n") + "\n" + t[j:]
    if new != t and not check:
        p.write_text(new, encoding="utf-8")
    return new != t


def changes_doc(S):
    out = ["# Changes from the original runs (all stages)", "",
           "Generated by `python tools/build_docs.py` from every `stages/**/CHANGES.md` (headings shifted by two levels). Changes the runner",
           "makes at run time (path rewrites, upload switches, pip modes, environment) are not in any `stage.ipy`; they are listed once in",
           "[`runner/RUNTIME_CHANGES.md`](../runner/RUNTIME_CHANGES.md). Change types follow DESIGN 2.5: `path`, `url`, `upload`,",
           "`dev-exclusion`, `wiring`, `variant`, `pin`, `save`, `bugfix`.", "",
           "| # | stage | group | exactness | CHANGES.md |", "|---|---|---|---|---|"]
    order = list(S)
    for k, sid in enumerate(order, 1):
        d, s = S[sid]
        rel = os.path.relpath(d / "CHANGES.md", REPO / "docs").replace("\\", "/")
        out.append(f"| {k} | [`{sid}`](#{sid.lower()}) | {s.get('group', '')} | {(s.get('origin') or {}).get('exactness', '?')} | [{rel}]({rel}) |")
    for sid in order:
        d, s = S[sid]
        f = d / "CHANGES.md"
        txt = f.read_text(encoding="utf-8").replace("\r\n", "\n") if f.exists() else "(missing)"
        txt = re.sub(r"^(#{1,4}) ", lambda m: "#" * min(len(m.group(1)) + 2, 6) + " ", txt, flags=re.M)
        out += ["", f"## {sid}", "", f"`{os.path.relpath(f, REPO).replace(chr(92), '/')}` ({s.get('group', '')}, {(s.get('origin') or {}).get('exactness', '?')})", "", txt.rstrip()]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    S = stages()
    A = jload(REPO / "configs/data_map.json")["artifacts"]
    C = consumers(S)
    stale = []
    if fill("DATA.md", "raw", data_table(A, C, ["raw", "collected"]), a.check):
        stale.append("DATA.md raw")
    if fill("DATA.md", "intermediate", data_table(A, C, ["intermediate", "run_output", "score", "wiring", "embedded", "forbidden", "pretrained"]), a.check):
        stale.append("DATA.md intermediate")
    if fill("CHECKPOINTS.md", "checkpoints", checkpoint_table(A, C, S), a.check):
        stale.append("CHECKPOINTS.md")
    if fill("LICENSES.md", "models", license_table(), a.check):
        stale.append("LICENSES.md")
    cd = changes_doc(S)
    p = REPO / "docs/CHANGES_FROM_ORIGINAL.md"
    if not p.exists() or p.read_text(encoding="utf-8") != cd:
        stale.append("docs/CHANGES_FROM_ORIGINAL.md")
        if not a.check:
            p.write_text(cd, encoding="utf-8")
    if not a.check:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", FF_MODELS="off", FF_DATA=os.environ.get("FF_DATA", ""))
        subprocess.run([sys.executable, str(REPO / "runner/ffrun.py"), "docs", "--out", str(REPO / "docs/STAGES.md")], check=True, env=env, cwd=REPO)
    print(("out of date: " + ", ".join(stale)) if stale else "generated docs up to date", flush=True)
    return 1 if (a.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
