#!/usr/bin/env python3
"""SHA-256 manifest of the data folder FF_DATA (= the Google Drive folder FourFusion_FLAG2027_data/), and checks against it.

    python tools/drive_manifest.py make  <FF_DATA> [--out <FF_DATA>/MANIFEST.sha256] [--only 03_checkpoints,05_scores] [--resume]
    python tools/drive_manifest.py check <FF_DATA> [--manifest <FF_DATA>/MANIFEST.sha256] [--only ...]
    python tools/drive_manifest.py datamap <FF_DATA>                 # configs/data_map.json: which artifacts are present, file hashes

`make` writes lines "<sha256>  <path relative to FF_DATA>" (the sha256sum format; `sha256sum -c MANIFEST.sha256` works from inside
FF_DATA) sorted by path, plus MANIFEST.json (per top folder: files, bytes). --resume keeps hashes of files whose size and mtime did
not change since the previous manifest (MANIFEST.cache.json), so re-running after an upload only hashes new files.
`datamap` lists every artifact of configs/data_map.json with its FF_DATA path, whether it is there, and compares the files whose
SHA-256 is recorded in the map (full hashes only). Standard library only; reads files in 4 MB blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_NAMES = {"MANIFEST.sha256", "MANIFEST.json", "MANIFEST.cache.json", ".DS_Store", "desktop.ini", "Thumbs.db"}


def sha256(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def walk(root, only=None):
    root = Path(root)
    tops = [root / t for t in only] if only else [root]
    for t in tops:
        if t.is_file():
            yield t
            continue
        for d, dirs, files in os.walk(t):
            dirs.sort()
            for f in sorted(files):
                if f not in SKIP_NAMES and not f.endswith(".part"):
                    yield Path(d) / f


def cmd_make(a):
    root = Path(a.data).resolve()
    out = Path(a.out) if a.out else root / "MANIFEST.sha256"
    cache_p = out.with_name("MANIFEST.cache.json")
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if a.resume and cache_p.exists() else {}
    only = [x for x in (a.only or "").split(",") if x]
    rows, new_cache, per_top, t0, nbytes = [], {}, {}, time.time(), 0
    for i, p in enumerate(walk(root, only)):
        rel = p.relative_to(root).as_posix()
        st = p.stat()
        c = cache.get(rel)
        h = c["sha256"] if c and c["bytes"] == st.st_size and c["mtime"] == int(st.st_mtime) else sha256(p)
        new_cache[rel] = {"sha256": h, "bytes": st.st_size, "mtime": int(st.st_mtime)}
        rows.append((rel, h))
        nbytes += st.st_size
        top = rel.split("/")[0]
        per_top.setdefault(top, {"files": 0, "bytes": 0})
        per_top[top]["files"] += 1
        per_top[top]["bytes"] += st.st_size
        if i % 500 == 0:
            print(f"  {i} files, {nbytes / 2**30:.1f} GB, {time.time() - t0:.0f} s: {rel[:90]}", flush=True)
    if only and out.exists():                                      # merge: keep the lines of the other top folders
        keep = [ln.split("  ", 1) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
        keep = [(r, h) for h, r in keep if r.split("/")[0] not in only]
        rows = keep + rows
    rows.sort()
    out.write_text("".join(f"{h}  {r}\n" for r, h in rows), encoding="utf-8")
    cache_p.write_text(json.dumps(new_cache if not only else {**cache, **new_cache}, indent=0), encoding="utf-8")
    (out.with_name("MANIFEST.json")).write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": "FF_DATA",
                                                            "files": len(rows), "per_top_folder": per_top}, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} files, {nbytes / 2**30:.2f} GB hashed in this run)")
    return 0


def cmd_check(a):
    root = Path(a.data).resolve()
    man = Path(a.manifest) if a.manifest else root / "MANIFEST.sha256"
    only = [x for x in (a.only or "").split(",") if x]
    bad = missing = ok = 0
    for ln in man.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        h, rel = ln.split("  ", 1)
        if only and rel.split("/")[0] not in only:
            continue
        p = root / rel
        if not p.is_file():
            missing += 1
            print(f"MISSING  {rel}")
        elif sha256(p) != h:
            bad += 1
            print(f"MISMATCH {rel}")
        else:
            ok += 1
    print(f"{ok} ok, {bad} mismatch, {missing} missing")
    return 0 if not bad and not missing else 1


def cmd_datamap(a):
    roots = [Path(x).resolve() for x in a.data.split(os.pathsep)]
    dm = json.loads((REPO / "configs" / "data_map.json").read_text(encoding="utf-8"))["artifacts"]
    present = 0
    for aid, e in dm.items():
        if not e.get("path"):
            continue
        hit = next((r / e["path"] for r in roots if (r / e["path"]).exists()), None)
        present += hit is not None
        flags = ",".join(k for k in ("private", "voxblink", "dev_data") if e.get(k))
        print(f"{'OK  ' if hit else '----'} {e['class']:12s} {e['path']:58s} {aid}{('  [' + flags + ']') if flags else ''}")
        if hit is None:
            continue
        for f, h in (e.get("files") or {}).items():
            name = f.split(" (")[0]
            p = hit / name if hit.is_dir() else hit
            if not p.is_file() and hit.is_dir():
                c = sorted(hit.rglob(Path(name).name))
                p = c[0] if c else p
            if p.is_file() and len(h) == 64:
                got = sha256(p)
                print(f"       {'sha256 ok ' if got == h else 'SHA256 BAD'} {name}")
            else:
                print(f"       not found  {name}")
    print(f"\n{present} of {sum(1 for e in dm.values() if e.get('path'))} mapped artifacts present under {[str(r) for r in roots]}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("data")
    p.add_argument("--out")
    p.add_argument("--only", help="comma-separated top folders (e.g. 03_checkpoints,05_scores)")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("data")
    p.add_argument("--manifest")
    p.add_argument("--only")
    p = sub.add_parser("datamap")
    p.add_argument("data", help="FF_DATA (several roots separated by os.pathsep)")
    a = ap.parse_args(argv)
    return {"make": cmd_make, "check": cmd_check, "datamap": cmd_datamap}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
