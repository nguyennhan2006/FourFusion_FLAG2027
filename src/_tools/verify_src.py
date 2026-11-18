"""Check repo/src/ against SRC_MANIFEST.json: every listed file present with the recorded sha256, and no unlisted files.

    python src/_tools/verify_src.py                      # anyone: integrity of src/
    python src/_tools/verify_src.py --against <root>     # team side: are the copies still identical to the team repository?

Exit code 0 when everything matches. Standard library only (Python >= 3.8)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=HERE.parent, help="the src/ folder (default: parent of _tools/)")
    ap.add_argument("--against", type=Path, help="team repository root: also compare each copy with its source file")
    a = ap.parse_args()
    src = a.src.resolve()
    man = json.loads((src / "SRC_MANIFEST.json").read_text(encoding="utf-8"))
    entries = [(e["path"], e["sha256"], "copy") for e in man["files"]] + \
              [(e["path"], e["sha256"], "derived") for e in man["derived"]] + \
              [(e["path"], e["sha256"], "tool") for e in man["tools_and_docs"]]
    bad, missing = [], []
    for rel, want, kind in entries:
        p = src / rel
        if not p.is_file():
            missing.append(rel)
        elif sha256(p) != want:
            bad.append(rel)
    listed = {rel for rel, _, _ in entries} | {"SRC_MANIFEST.json"}
    extra = sorted(f.relative_to(src).as_posix() for f in src.rglob("*")
                   if f.is_file() and "__pycache__" not in f.parts and f.relative_to(src).as_posix() not in listed)
    drift = []
    if a.against:
        root = a.against.resolve()
        for e in man["files"]:
            s = root / e["path"]
            if not s.is_file() or sha256(s) != e["sha256"]:
                drift.append(e["path"])
        for e in man["derived"]:
            s = root / e["derived_from"]["path"]
            if not s.is_file() or sha256(s) != e["derived_from"]["sha256"]:
                drift.append(e["derived_from"]["path"])
    print(f"[verify] {len(entries)} listed files: {len(entries) - len(bad) - len(missing)} ok, {len(bad)} sha256 mismatch, {len(missing)} missing; "
          f"{len(extra)} unlisted" + (f"; {len(drift)} differ from the source tree" if a.against else ""))
    for name, lst in (("mismatch", bad), ("missing", missing), ("unlisted", extra), ("source drift", drift)):
        for rel in lst:
            print(f"[verify] {name}: {rel}")
    sys.exit(0 if not (bad or missing or extra or drift) else 1)


if __name__ == "__main__":
    main()
