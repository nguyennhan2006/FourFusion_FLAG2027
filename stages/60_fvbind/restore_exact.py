r"""Rebuild the byte-exact Kaggle notebooks of group 60_fvbind from their redacted copies.

Every notebook of the FLAG_163 family (FLAG_163 / 189 / 192 / 193) embeds the organisers' file meta_file_train_set.csv
(MAV-Celeb v4 train-set metadata), written by the setup cell as train/gender.csv from a gzip (level 9, mtime 0) + base64
literal. The repository does not redistribute organisers' data, so the copies under */original/ have exactly that one
literal removed (a byte splice; nothing else changed). Every removal is recorded under "redactions" in a SOURCE.json
(offset, length, sha256 of the removed literal, md5 / sha256 of the exact and of the redacted file).

Given your copy of the organisers' file (sha256 fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030), this
script re-creates the literal, re-inserts it and checks the md5 / sha256 of every result:

    python stages/60_fvbind/restore_exact.py --csv <FF_DATA>/00_raw/btc_mavceleb_v4/meta_file_train_set.csv --out <dir>

Writes <dir>/<path of the redacted copy, ".redacted" removed>. Exit code 1 when any check fails. Standard library only.

The gzip member is built explicitly (fixed 10-byte header 1f8b0800 00000000 02ff = magic, deflate, no flags, mtime 0,
XFL 2, OS 255, then a raw deflate stream at level 9, then CRC32 / ISIZE). This is byte for byte what
gzip.compress(csv, compresslevel=9, mtime=0) writes on CPython >= 3.13; CPython 3.11 / 3.12 write a platform OS byte
instead of 255 for mtime=0, so the explicit header removes that dependence. The deflate stream itself comes from the
zlib library of the interpreter. Verified 7 / 7 byte-exact on: CPython 3.14.0, Windows, zlib-ng 2.2.4 (zlib API 1.3.1)
on 2026-11-18. If a check fails, the script says whether the header or the deflate stream differs; another zlib build
(for example a different zlib-ng / zlib major version) can produce a different but equivalent deflate stream.
"""
import argparse
import base64
import gzip
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

META_SHA = "fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030"
GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"   # header of the removed literal (its base64 starts 'H4sIAAAAAAAC/')


def gzip_member(data):
    """= gzip.compress(data, compresslevel=9, mtime=0) of CPython >= 3.13, independent of the interpreter's gzip module."""
    return GZIP_HEADER + zlib.compress(data, 9, -15) + struct.pack("<LL", zlib.crc32(data) & 0xffffffff, len(data) & 0xffffffff)


def zlib_info():
    return f"Python {sys.version.split()[0]} ({sys.platform}), zlib {zlib.ZLIB_VERSION}" + \
        (f", zlib-ng {zlib.ZLIBNG_VERSION}" if hasattr(zlib, "ZLIBNG_VERSION") else "")


def main():
    print("[restore] group 60_fvbind: exact notebooks from the redacted copies", flush=True)
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True, help="the organisers' meta_file_train_set.csv")
    ap.add_argument("--out", required=True, help="output folder (outside the repository)")
    a = ap.parse_args()
    print(f"[restore] {zlib_info()}", flush=True)
    csv = Path(a.csv).read_bytes()
    if hashlib.sha256(csv).hexdigest() != META_SHA:
        sys.exit(f"{a.csv}: sha256 {hashlib.sha256(csv).hexdigest()} is not the organisers' file {META_SHA}")
    raw = gzip_member(csv)
    assert gzip.decompress(raw) == csv, "the gzip member does not decompress to the organisers' file"
    lit = base64.b64encode(raw)
    out, bad, n, lit_bad = Path(a.out), 0, 0, set()
    for sj in sorted(here.rglob("SOURCE.json")):
        for r in json.loads(sj.read_text(encoding="utf-8")).get("redactions", []):
            n += 1
            red = (sj.parent / r["repo_file"]).read_bytes()
            red_ok = hashlib.sha256(red).hexdigest() == r["redacted_sha256"]
            lit_ok = hashlib.sha256(lit).hexdigest() == r["removed_literal_sha256"] and len(lit) == r["removed_length"]
            if not lit_ok:
                lit_bad.add((r["removed_literal_sha256"], r["removed_length"]))
            off = r["byte_offset_in_exact_file"]
            exact = red[:off] + lit + red[off:]
            ok = red_ok and lit_ok and hashlib.md5(exact).hexdigest() == r["exact_md5"] and hashlib.sha256(exact).hexdigest() == r["exact_sha256"]
            dst = out / sj.parent.relative_to(here) / r["exact_name"]
            if ok:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(exact)
            bad += not ok
            why = "" if ok else ("  [redacted copy changed]" if not red_ok else "  [literal differs]" if not lit_ok else "  [result differs]")
            print(f"  {'OK  ' if ok else 'FAIL'} {sj.parent.relative_to(here).as_posix()}/{r['exact_name']}  md5 {hashlib.md5(exact).hexdigest()}"
                  f" (expected {r['exact_md5']}){why}", flush=True)
    for h, length in sorted(lit_bad):
        print(f"[restore] literal check failed (expected sha256 {h[:16]}..., {length} characters; got {hashlib.sha256(lit).hexdigest()[:16]}..., "
              f"{len(lit)}): the 10-byte gzip header is fixed by this script, so the deflate stream of this zlib build "
              f"({zlib_info()}) differs from the one of the original run; use an interpreter with the zlib build listed in the docstring", flush=True)
    print(f"[restore] {n - bad} of {n} notebooks restored byte-exact into {out}", flush=True)
    sys.exit(1 if bad or n == 0 else 0)


if __name__ == "__main__":
    main()
