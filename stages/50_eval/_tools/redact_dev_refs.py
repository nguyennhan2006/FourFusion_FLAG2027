r"""Remove the dev-phase score data that three exact Evaluation-run notebooks carry (stages e50cc, e112f0sx, e112f1sx), so that the
published repository holds no dev data. The Evaluation numbers do not depend on what is removed (see below). Pure Python, no
third-party package; deterministic (same input bytes -> same output bytes).

    python redact_dev_refs.py <stage> <exact.ipynb> <out.ipynb>              write the redacted notebook
    python redact_dev_refs.py <stage> <exact.ipynb> --check <redacted.ipynb> verify that a redacted copy derives from the exact one

<stage> is e50cc, e112f0sx or e112f1sx. <exact.ipynb> is the exact executed notebook of the team package
(E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/2_eval/0N_*.ipynb, Kaggle-pulled JSON); its sha256 is checked (EXACT below).
The exact notebooks themselves are kept only in private storage (they contain the dev data).

What is removed and why the Evaluation run never reads it
  e50cc      code cell "32 the submitted zips (dev references)": three base64 zips of the team's dev-phase submissions (score files
             of the dev trial lists: v9, v11, v12) that the cell wrote to repo/reference/. They are only compared with the run's own
             zips when the run is on dev: cell 33 `for name, ref in (REF_ZIPS.items() if is_dev else [])`, with
             `is_dev = any(...) and not ONLY_UNDER` and ONLY_UNDER = 'evaluation_phase' in cell 02. The cell keeps its first four
             lines (REF_ZIPS = {}) and its last line; a comment records the md5 of what was removed.
  e112f0sx,  code cell "03 embedded code": member Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip of the base64 code zip
  e112f1sx   CODE_B64 (a dev-phase score zip). The only reader is Experiment/CLEANTUNE-01/run.py line 635,
             `FL.cmp_zip(zp, EXP / "CLEAN-EN-01/submission_clean_en01_dev.zip") if E.SPLIT == "dev" else None`; these runs set
             E2E_SPLIT = 'eval' (cell 01). The zip is rebuilt by copying the local header + compressed data of every other member and
             their central-directory records byte for byte (only offsets change; no recompression), so each remaining member keeps
             its md5, which the notebook itself checks against CODE_MD5 (CODE_MD5 still lists the removed file; the check only covers
             the members present). One comment line is inserted before CODE_B64.

The notebook JSON is written back exactly as the Kaggle-pulled copies are serialised (json.dumps with separators (",", ":"),
ensure_ascii=False, no final newline): only the edited cell's "source" string changes."""
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

EXACT = {   # sha256 of the exact executed notebooks (team package E:/FLAG2027_submit/FACTOR_ENFUSE_c5rl/notebooks/2_eval/)
    "e50cc": "49022347acefbc24eb43a2370e263d7aa6c081438c18bc0a23d62db5ef7abad1",      # 04_flag-2027-nb-50cc-e2e-eval.ipynb
    "e112f0sx": "0e0aeae47716e6a7dfa658ad560c33c2ccbcd88dd6c8790c0c6bd43cc00daae6",   # 05_flag-2027-nb-112f0sx-eval.ipynb
    "e112f1sx": "4b746a07ce9b72e12cc253e364e5b4b5400a2c230e479e4e6eaccaf12b6b74ad",   # 06_flag-2027-nb-112f1sx-eval.ipynb
}
FIRST_50 = 'print("[cell] 32 the submitted zips (dev references)", flush=True)'
REF_50 = {"v9": ("repo/reference/submission_CA_EN_BN.zip", "f6d77418f3a44f2273ab5fa921d8059d"),
          "v11": ("repo/reference/submission_v11.zip", "bc741c4a56c310c7e0b134c35a64a8db"),
          "v12": ("repo/reference/submission_v12.zip", "a7323cc4a6a26feff0a111745f27694a")}
FIRST_112 = 'print("[cell] 03 embedded code (= the local files, md5 below) - /kaggle/worki", flush=True)'
DROP_112 = "Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip"
DROP_112_MD5 = "ae902d1c7e87783bf88e345b459e39f2"


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def md5(b):
    return hashlib.md5(b).hexdigest()


# ------------------------------------------------------------------ zip member removal by raw copy (no recompression)
def drop_zip_members(raw, drop):
    """-> (new zip bytes, {name: removed member bytes}). Plain zip only (no zip64, no multi-disk), as written by Python's zipfile."""
    eocd = raw.rfind(b"PK\x05\x06")
    assert eocd >= 0 and len(raw) - eocd >= 22, "end of central directory not found"
    disk, cd_disk, n_disk, n_total, cd_size, cd_off, c_len = struct.unpack("<HHHHIIH", raw[eocd + 4:eocd + 22])
    assert disk == cd_disk == 0 and n_disk == n_total, "multi-disk zip not supported"
    assert cd_off + cd_size == eocd, "unexpected data between the central directory and its end record"
    comment = raw[eocd + 22:eocd + 22 + c_len]
    records, p = [], cd_off
    for _ in range(n_total):
        assert raw[p:p + 4] == b"PK\x01\x02", "bad central directory record"
        (flags, csize, n_len, x_len, k_len, loc) = (struct.unpack("<H", raw[p + 8:p + 10])[0], struct.unpack("<I", raw[p + 20:p + 24])[0],
                                                   *struct.unpack("<HHH", raw[p + 28:p + 34]), struct.unpack("<I", raw[p + 42:p + 46])[0])
        assert csize != 0xFFFFFFFF and loc != 0xFFFFFFFF, "zip64 not supported"
        name = raw[p + 46:p + 46 + n_len].decode("utf-8" if flags & 0x800 else "cp437")
        end = p + 46 + n_len + x_len + k_len
        records.append((name, raw[p:end], loc, csize, flags))
        p = end
    names = [r[0] for r in records]
    assert len(set(names)) == len(names), "duplicate member names"
    assert set(drop) <= set(names), f"members not found: {set(drop) - set(names)}"
    src = zipfile.ZipFile(io.BytesIO(raw))
    removed = {n: src.read(n) for n in drop}
    out, cd = io.BytesIO(), io.BytesIO()
    for name, rec, loc, csize, flags in records:
        if name in drop:
            continue
        assert raw[loc:loc + 4] == b"PK\x03\x04", f"bad local header of {name}"
        ln, lx = struct.unpack("<HH", raw[loc + 26:loc + 30])
        end = loc + 30 + ln + lx + csize
        if flags & 0x08:                                    # data descriptor after the compressed data
            end += 16 if raw[end:end + 4] == b"PK\x07\x08" else 12
        new_loc = out.tell()
        out.write(raw[loc:end])
        cd.write(rec[:42] + struct.pack("<I", new_loc) + rec[46:])
    cd_off2, cd_bytes = out.tell(), cd.getvalue()
    out.write(cd_bytes)
    n2 = n_total - len(drop)
    out.write(b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, n2, n2, len(cd_bytes), cd_off2, len(comment)) + comment)
    new = out.getvalue()
    chk = zipfile.ZipFile(io.BytesIO(new))                  # the rebuilt zip reads back, CRCs pass, kept members unchanged
    assert chk.testzip() is None and chk.namelist() == [n for n in names if n not in drop]
    assert all(chk.read(n) == src.read(n) for n in chk.namelist())
    return new, removed


# ------------------------------------------------------------------ the two rules
def _code_cell(nb, first_line):
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and isinstance(c["source"], str) and c["source"].split("\n", 1)[0] == first_line]
    assert len(hits) == 1, f"expected one code cell starting with {first_line!r}, found {len(hits)}"
    return hits[0]


def redact_e50cc(nb):
    c = _code_cell(nb, FIRST_50)
    lines = c["source"].split("\n")
    assert lines[1].startswith("# ===") and lines[2] == "import base64, hashlib" and lines[3] == "REF_ZIPS = {}", lines[:4]
    assert lines[-1] == "print('references:', {k: v.name for k, v in REF_ZIPS.items()})", lines[-1]
    found, i = {}, 4                                        # parse the three  _b = ( "...", ... )  blocks and their md5 asserts
    while i < len(lines) - 1:
        assert lines[i] == "_b = (", (i, lines[i][:60])
        j = i + 1
        while not lines[j].rstrip().endswith('")'):
            j += 1
        blob = "".join(re.findall(r'"([A-Za-z0-9+/=]+)"', "\n".join(lines[i + 1:j + 1])))
        m1 = re.fullmatch(r"REF_ZIPS\['(v\d+)'\] = WORKDIR / '([^']+)'; REF_ZIPS\['v\d+'\]\.parent\.mkdir\(parents=True, exist_ok=True\)", lines[j + 1])
        m2 = re.fullmatch(r"REF_ZIPS\['(v\d+)'\]\.write_bytes\(base64\.b64decode\(_b\)\); assert hashlib\.md5\(REF_ZIPS\['v\d+'\]\.read_bytes\(\)\)\.hexdigest\(\) == '([0-9a-f]{32})'", lines[j + 2])
        assert m1 and m2 and m1.group(1) == m2.group(1), (lines[j + 1][:80], lines[j + 2][:80])
        data = base64.b64decode(blob, validate=True)
        assert md5(data) == m2.group(2), f"{m1.group(1)}: decoded payload md5 differs from the cell's own assert"
        found[m1.group(1)] = (m1.group(2), m2.group(2))
        i = j + 3
    assert found == REF_50, found
    note = ["# FF-REDACTED (FourFusion reproduction repository; change type dev-exclusion; see stages/50_eval/e50cc/CHANGES.md).",
            "# The original cell embedded three dev-phase score zips (base64) and wrote them to disk, md5-checked:"]
    note += [f"#   REF_ZIPS['{k}'] = WORKDIR / '{p}'  md5 {h}" for k, (p, h) in found.items()]
    note += ["# They are compared only when the run is on dev (cell 33: `REF_ZIPS.items() if is_dev`; is_dev is False because",
             "# ONLY_UNDER = 'evaluation_phase'), so the Evaluation numbers do not depend on them. Removed because dev data must not",
             "# be published. REF_ZIPS stays an empty dict."]
    c["source"] = "\n".join(lines[:4] + note + [lines[-1]])
    return {"cell": FIRST_50, "removed_payloads": {k: {"written_to": p, "md5": h} for k, (p, h) in found.items()}}


def redact_e112(nb):
    c = _code_cell(nb, FIRST_112)
    src = c["source"]
    ms = list(re.finditer(r'^CODE_B64 = "([A-Za-z0-9+/=]+)"$', src, re.M))
    assert len(ms) == 1, f"CODE_B64 line found {len(ms)} times"
    m = ms[0]
    raw = base64.b64decode(m.group(1), validate=True)
    new_raw, removed = drop_zip_members(raw, {DROP_112})
    assert md5(removed[DROP_112]) == DROP_112_MD5, md5(removed[DROP_112])
    md5_table = dict(re.findall(r'^ "([^"]+)": "([0-9a-f]{32})",?$', src, re.M))      # the notebook's own CODE_MD5
    kept = zipfile.ZipFile(io.BytesIO(new_raw))
    assert md5_table.get(DROP_112) == DROP_112_MD5 and all(md5(kept.read(n)) == md5_table[n] for n in kept.namelist())
    note = ("# FF-REDACTED (FourFusion reproduction repository; change type dev-exclusion; see this stage's CHANGES.md): member "
            f"{DROP_112} (md5 {DROP_112_MD5}, a dev-phase score zip) was removed from CODE_B64; every other member is copied byte "
            "for byte. Its only reader is Experiment/CLEANTUNE-01/run.py line 635, when E2E_SPLIT == 'dev'; this run sets "
            "E2E_SPLIT = 'eval'. CODE_MD5 below still lists it; the md5 check covers the members present.")
    c["source"] = src[:m.start()] + note + "\n" + 'CODE_B64 = "' + base64.b64encode(new_raw).decode("ascii") + '"' + src[m.end():]
    return {"cell": FIRST_112, "removed_member": DROP_112, "removed_md5": DROP_112_MD5, "code_zip_sha256_before": sha256(raw),
            "code_zip_sha256_after": sha256(new_raw), "members_kept": len(kept.namelist())}


RULES = {"e50cc": redact_e50cc, "e112f0sx": redact_e112, "e112f1sx": redact_e112}


def redact(stage, exact_bytes, strict=True):
    """-> (redacted notebook bytes, report dict)."""
    if strict:
        assert sha256(exact_bytes) == EXACT[stage], f"{stage}: input sha256 {sha256(exact_bytes)} is not the exact notebook {EXACT[stage]}"
    text = exact_bytes.decode("utf-8")
    nb = json.loads(text)
    ser = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False)
    assert ser(nb) == text, "the notebook is not in the compact Kaggle-pulled serialisation; refusing to re-serialise"
    before = [c["source"] for c in nb["cells"]]
    rep = RULES[stage](nb)
    after = [c["source"] for c in nb["cells"]]
    assert sum(a != b for a, b in zip(before, after)) == 1                   # exactly one cell changed
    out = ser(nb).encode("utf-8")
    rep.update(stage=stage, exact_sha256=sha256(exact_bytes), exact_md5=md5(exact_bytes), redacted_sha256=sha256(out), redacted_md5=md5(out))
    return out, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=sorted(RULES))
    ap.add_argument("exact")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--check", metavar="REDACTED")
    ap.add_argument("--not-strict", action="store_true", help="do not require the exact sha256 (diagnostics only)")
    a = ap.parse_args()
    data, rep = redact(a.stage, Path(a.exact).read_bytes(), strict=not a.not_strict)
    if a.check:
        have = Path(a.check).read_bytes()
        ok = have == data
        print(json.dumps(rep, indent=1))
        print("OK: the redacted copy derives from the exact notebook" if ok else
              f"MISMATCH: {a.check} sha256 {sha256(have)} != {rep['redacted_sha256']}")
        sys.exit(0 if ok else 1)
    if not a.out:
        ap.error("give <out.ipynb> or --check <redacted.ipynb>")
    Path(a.out).write_bytes(data)
    print(json.dumps(rep, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
