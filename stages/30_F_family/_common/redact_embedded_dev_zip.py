#!/usr/bin/env python3
"""Remove the embedded DEV score zip from an exact Kaggle notebook of the F1s' family, deterministically.

Why (rule: no organiser dev data in the published repository): the executed notebooks of five runs of the 13.54 chain
(t155s_f12 / t155s_f34 / t155s_f5 = FLAG_111, t112sf0s / t112sf1s = FLAG_112, and the FLAG_110 g1 candidate) carry their
code as a base64 zip in `CODE_B64 = "..."`. That zip also holds `Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip`
(sha256 b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d): per-pair scores on the organisers' DEV lists.
Its only reader is `zip_ct()` of `Experiment/CLEANTUNE-01/run.py` (embedded copy, line 561):
    ref = FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None
and `FACELORA-03/run.py cmp_zip` returns None when the file does not exist (line 406). So the removal changes one printed
report line ("correlation per file with the CB 20.18 zip: None") and nothing else: no array, model, table or score.

How (byte-level, deterministic, no recompression):
  1. the base64 string of `CODE_B64` is decoded; the zip is rebuilt by copying the RAW local entries (header + compressed
     data) of every other member in their original order and rewriting the central directory with the new offsets;
  2. the new zip is re-encoded in base64 (no line breaks, as the builder did) and substituted for the old string in the
     notebook FILE (byte-level replacement of exactly one occurrence: every other byte of the notebook is unchanged);
  3. checks: the rebuilt zip opens, testzip() is clean, the member list is the old one minus the removed member, every
     kept member has the same bytes (md5, as the notebook's own CODE_MD5 check) and the same ZipInfo fields.
The notebook's CODE_MD5 dict is left as it was: its check loop runs over the members present in the zip
(`for f in z.namelist(): assert md5 == CODE_MD5[f]`), so it still passes; the stale entry is only printed.

    python redact_embedded_dev_zip.py <exact.ipynb> <out.redacted.ipynb> [--member <name>]... [--report <json>]

Light operation (base64 + zip directory rewrite of a < 1 MB notebook); no network, no model."""
import argparse
import base64
import hashlib
import io
import json
import re
import struct
import sys
import zipfile
from pathlib import Path

DEV_ZIP = "Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip"
DEV_ZIP_SHA256 = "b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d"
B64_RX = re.compile(r'CODE_B64 = "([A-Za-z0-9+/=]+)"')


def _cd_records(data):
    """[(name, local_header_offset, raw central-directory record)] in central-directory order + EOCD fields."""
    eocd = data.rfind(b"PK\x05\x06")
    assert eocd >= 0, "no end-of-central-directory record"
    sig, disk, cd_disk, n_disk, n_total, cd_size, cd_off, clen = struct.unpack("<IHHHHIIH", data[eocd:eocd + 22])
    assert disk == cd_disk == 0 and n_disk == n_total and n_total != 0xFFFF and cd_off != 0xFFFFFFFF, "multi-disk / ZIP64 not handled"
    comment = data[eocd + 22:eocd + 22 + clen]
    recs, p = [], cd_off
    for _ in range(n_total):
        assert data[p:p + 4] == b"PK\x01\x02", f"bad central record at {p}"
        fn_len, ex_len, cm_len = struct.unpack("<HHH", data[p + 28:p + 34])
        lho = struct.unpack("<I", data[p + 42:p + 46])[0]
        name = data[p + 46:p + 46 + fn_len].decode("utf-8")
        rec = data[p:p + 46 + fn_len + ex_len + cm_len]
        recs.append((name, lho, rec))
        p += len(rec)
    assert p == cd_off + cd_size, "central directory size mismatch"
    return recs, cd_off, comment


def drop_members(data, drop):
    """Zip bytes without the members in `drop`; kept members' raw local entries copied unchanged."""
    recs, cd_off, comment = _cd_records(data)
    starts = sorted(lho for _, lho, _ in recs) + [cd_off]
    extent = {lho: (lho, starts[starts.index(lho) + 1]) for _, lho, _ in recs}
    body, cd = bytearray(), bytearray()
    kept = 0
    for name, lho, rec in recs:
        if name in drop:
            continue
        a, b = extent[lho]
        assert data[a:a + 4] == b"PK\x03\x04", f"bad local header for {name}"
        new_off = len(body)
        body += data[a:b]
        cd += rec[:42] + struct.pack("<I", new_off) + rec[46:]
        kept += 1
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, kept, kept, len(cd), len(body), len(comment)) + comment
    return bytes(body + cd + eocd)


def md5(b):
    return hashlib.md5(b).hexdigest()


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def redact(src, dst, members=(DEV_ZIP,)):
    raw = Path(src).read_bytes()
    text = raw.decode("utf-8")
    nb = json.loads(text)
    srcs = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    hits = [m.group(1) for s in srcs for m in B64_RX.finditer(s)]
    assert len(hits) == 1, f"expected exactly one CODE_B64 string, found {len(hits)}"
    old_b64 = hits[0]
    assert text.count(old_b64) == 1, "the base64 string must occur exactly once in the notebook file"
    old_zip = base64.b64decode(old_b64)
    zo = zipfile.ZipFile(io.BytesIO(old_zip))
    names = zo.namelist()
    removed = []
    for m in members:
        assert m in names, f"{m} is not in the embedded zip"
        removed.append(dict(name=m, size=zo.getinfo(m).file_size, sha256=sha256(zo.read(m)), md5=md5(zo.read(m))))
    new_zip = drop_members(old_zip, set(members))
    zn = zipfile.ZipFile(io.BytesIO(new_zip))
    assert zn.testzip() is None, "rebuilt zip fails testzip()"
    assert zn.namelist() == [n for n in names if n not in members], "member list changed"
    for n in zn.namelist():
        a, b = zo.getinfo(n), zn.getinfo(n)
        assert zo.read(n) == zn.read(n), f"member {n} changed"
        assert (a.compress_type, a.CRC, a.file_size, a.compress_size, a.date_time) == (b.compress_type, b.CRC, b.file_size, b.compress_size, b.date_time), n
    new_b64 = base64.b64encode(new_zip).decode("ascii")
    out = text.replace(old_b64, new_b64).encode("utf-8")
    json.loads(out.decode("utf-8"))                                     # still a valid notebook JSON
    Path(dst).write_bytes(out)
    return dict(source=str(src), source_md5=md5(raw), source_sha256=sha256(raw), output=str(dst), output_md5=md5(out),
                output_sha256=sha256(out), removed=removed,
                embedded_zip=dict(members_before=len(names), members_after=len(zn.namelist()), sha256_before=sha256(old_zip),
                                  sha256_after=sha256(new_zip), kept_members_byte_identical=True),
                method="raw local-entry copy + rewritten central directory; byte-level replacement of the one CODE_B64 string")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--member", action="append", default=None, help=f"member to remove (default {DEV_ZIP})")
    ap.add_argument("--report", default=None, help="write the JSON report here as well")
    a = ap.parse_args()
    rep = redact(a.src, a.dst, a.member or [DEV_ZIP])
    for r in rep["removed"]:
        if r["name"] == DEV_ZIP and r["sha256"] != DEV_ZIP_SHA256:
            print(f"!! {DEV_ZIP} sha256 {r['sha256']} differs from the known {DEV_ZIP_SHA256}", file=sys.stderr)
    s = json.dumps(rep, indent=1)
    print(s)
    if a.report:
        Path(a.report).write_text(s + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
