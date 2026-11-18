"""WP-F tools for stages/40_R_family (standard library only; no network, no Kaggle, no training).
  python stages/40_R_family/_wpf_tools.py check                        rebuild every stage.ipy from original/ + patches, verify hashes,
                                                                       redactions, the LoRA C1 variant guard (forbidden inputs, saved_only + require_files on the C1 input,
                                                                       t163l_orig force_only, JSON-only post_checks, runner_requirements), syntax, secret scan
  python stages/40_R_family/_wpf_tools.py redact <stage> <exact.ipynb>  re-create original/*.redacted.ipynb from a private exact copy
  python stages/40_R_family/_wpf_tools.py decode <stage> <out_dir>      write the code embedded in the stage notebook
The helpers below are the ones that built the stage folders (18/11/2026, WP-F)."""
import base64, gzip, hashlib, io, json, re, struct, zipfile
from pathlib import Path

CELL_SEP = "\n# <<<FF-CELL>>>\n"
J = lambda x: "".join(x) if isinstance(x, list) else x

# organisers' file embedded in every FLAG_163-family notebook as "train/gender.csv" (= meta_file_train_set.csv)
GENDER_SHA256 = "fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030"
GENDER_PLACEHOLDER = "FF-REDACTED:organisers-file:meta_file_train_set.csv:560-bytes:sha256=" + GENDER_SHA256
# dev score zip embedded in the code zip (CODE_B64) of the FLAG_174-family report notebooks (184, 219)
DEVZIP_MEMBER = "Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip"
DEVZIP_SHA256 = "b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d"

BLOB = re.compile(r'\("([^"]+)",\s*"([A-Za-z0-9+/=]{40,})"\)')
CODEB64 = re.compile(r'CODE_B64 = "([A-Za-z0-9+/=]+)"')


def sha256b(b):
    return hashlib.sha256(b).hexdigest()


def md5b(b):
    return hashlib.md5(b).hexdigest()


def code_sources(nb):
    """(cell index, source) of every non-empty code cell, CRLF -> LF (same rule as release/flag2027_e2e/build_notebook.py)."""
    return [(i, J(c["source"]).replace("\r\n", "\n")) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and J(c["source"]).strip()]


def zip_drop_member(zb, name):
    """Remove one member from a zip WITHOUT recompressing the others: the kept local records are copied byte for byte,
    the central directory is rewritten with shifted offsets. Deterministic on any machine (no zlib involved)."""
    eocd = zb.rfind(b"PK\x05\x06")
    assert eocd >= 0, "no end of central directory"
    (_, _, _, _, n_all, cd_size, cd_off, clen) = struct.unpack("<IHHHHIIH", zb[eocd:eocd + 22])
    comment = zb[eocd + 22:eocd + 22 + clen]
    ents, p = [], cd_off
    for _ in range(n_all):
        assert zb[p:p + 4] == b"PK\x01\x02", "bad central directory entry"
        fl, el, cl = struct.unpack("<HHH", zb[p + 28:p + 34])
        lho = struct.unpack("<I", zb[p + 42:p + 46])[0]
        nm = zb[p + 46:p + 46 + fl].decode("utf-8")
        ents.append(dict(name=nm, cd=zb[p:p + 46 + fl + el + cl], lho=lho))
        p += 46 + fl + el + cl
    offs = sorted(e["lho"] for e in ents) + [cd_off]
    span = {o: (o, offs[k + 1]) for k, o in enumerate(offs[:-1])}
    drop = [e for e in ents if e["name"] == name]
    assert len(drop) == 1, f"member {name!r} found {len(drop)} times"
    d0, d1 = span[drop[0]["lho"]]
    body, cds = bytearray(), bytearray()
    newoff = {}
    for o in offs[:-1]:
        if o == d0:
            continue
        newoff[o] = len(body)
        body += zb[o:span[o][1]]
    for e in ents:
        if e["name"] == name:
            continue
        cd = bytearray(e["cd"])
        cd[42:46] = struct.pack("<I", newoff[e["lho"]])
        cds += cd
    end = struct.pack("<IHHHHIIH", 0x06054b50, 0, 0, n_all - 1, n_all - 1, len(cds), len(body), clen) + comment
    out = bytes(body + cds + end)
    with zipfile.ZipFile(io.BytesIO(out)) as z:                          # self-check: readable, every kept member intact
        assert z.testzip() is None
        with zipfile.ZipFile(io.BytesIO(zb)) as z0:
            keep = [n for n in z0.namelist() if n != name]
            assert z.namelist() == keep
            for n in keep:
                assert z.read(n) == z0.read(n)
    return out


def redact(raw, spec):
    """Redacted copy of an exact notebook (bytes in, bytes out). spec: {"gender": bool, "devzip": bool}.
    Only string literals are replaced, in the raw JSON text; everything else stays byte-identical."""
    text = raw.decode("utf-8")
    nb = json.loads(text)
    recs = []
    if spec.get("gender"):
        hits = [(i, m) for i, s in code_sources(nb) for m in BLOB.finditer(s) if m.group(1) == "train/gender.csv"]
        assert len(hits) == 1, f"gender blob found {len(hits)} times"
        i, m = hits[0]
        lit = m.group(2)
        dec = gzip.decompress(base64.b64decode(lit))
        assert sha256b(dec) == GENDER_SHA256, "embedded gender.csv is not the organisers' meta_file_train_set.csv"
        assert text.count(lit) == 1, "gender literal not unique in the raw notebook"
        text = text.replace(lit, GENDER_PLACEHOLDER)
        recs.append(dict(cell=i, kind="redaction", what='string literal of the ("train/gender.csv", ...) entry of the embedded-file loop',
                         original_literal_chars=len(lit), original_literal_sha256=sha256b(lit.encode()),
                         decoded_bytes=len(dec), decoded_sha256=sha256b(dec),
                         decoded_is="organisers' file meta_file_train_set.csv (sha256 identical)", replacement=GENDER_PLACEHOLDER))
    if spec.get("devzip"):
        hits = [(i, m) for i, s in code_sources(nb) for m in CODEB64.finditer(s)]
        assert len(hits) == 1, f"CODE_B64 found {len(hits)} times"
        i, m = hits[0]
        lit = m.group(1)
        zb = base64.b64decode(lit)
        with zipfile.ZipFile(io.BytesIO(zb)) as z:
            dz = z.read(DEVZIP_MEMBER)
        assert sha256b(dz) == DEVZIP_SHA256, "unexpected dev zip member"
        nz = zip_drop_member(zb, DEVZIP_MEMBER)
        nlit = base64.b64encode(nz).decode()
        assert text.count(lit) == 1, "CODE_B64 literal not unique in the raw notebook"
        text = text.replace(lit, nlit)
        recs.append(dict(cell=i, kind="dev-exclusion", what=f"member {DEVZIP_MEMBER} removed from the code zip CODE_B64 (other members copied byte for byte)",
                         original_literal_chars=len(lit), original_literal_sha256=sha256b(lit.encode()),
                         original_zip_sha256=sha256b(zb), removed_member_sha256=sha256b(dz), removed_member_md5=md5b(dz),
                         removed_member_bytes=len(dz), new_zip_sha256=sha256b(nz), new_literal_chars=len(nlit),
                         new_literal_sha256=sha256b(nlit.encode())))
    out = text.encode("utf-8")
    json.loads(out)
    return out, recs


def build_ipy(nb_bytes, patches, drop_cells=()):
    """stage.ipy text = original code cells (CRLF -> LF, drop_cells removed) with the patches applied, each cell + CELL_SEP."""
    nb = json.loads(nb_bytes.decode("utf-8"))
    srcs = []
    for _, s in code_sources(nb):
        first = s.lstrip().splitlines()[0] if s.strip() else ""
        if any(re.search(rx, first) for rx in drop_cells):
            continue
        srcs.append(s)
    for p in patches:
        hits = sum(s.count(p["old"]) for s in srcs)
        assert hits == p["count"], f"patch {p['old'][:70]!r}: found {hits} times, expected {p['count']}"
        srcs = [s.replace(p["old"], p["new"]) for s in srcs]
    return "".join(s.rstrip("\n") + CELL_SEP for s in srcs)


def decode_payloads(nb_bytes):
    """Every embedded payload: ("rel", "<gzip+base64>") pairs and the CODE_B64 zip -> {name: bytes}."""
    nb = json.loads(nb_bytes.decode("utf-8"))
    out = {}
    for i, s in code_sources(nb):
        for m in BLOB.finditer(s):
            rel = m.group(1)
            name = rel if ("/" in rel or "." in rel) else f"Experiment/{rel}/run.py"    # FLAG_174 report cell: ("<EXP>", blob) -> run.py
            out[name] = gzip.decompress(base64.b64decode(m.group(2)))
        for m in CODEB64.finditer(s):
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1)))) as z:
                for n in z.namelist():
                    out[n] = z.read(n)
    return out


# ================================================================================================ group self-check (CLI)
HERE = Path(__file__).resolve().parent
LORA_ORIG = "datasets/onnhqunhh/flag2027-rfs-lora-p"
C1_SHA256 = "b97372e3504a131d0a969e478a855868095c67db4b3c0c133ef770c00e40991c"   # lora_C1.pt of the ORIGINAL LoRA C1 (DESIGN 2.7)
FORBIDDEN = {"datasets/ngnnguynchn/flag2027-rfs-lorafix1-p", "datasets/ngnnguynchn/flag2027-c1fix-lora-p", "datasets/ngnnguynchn/flag2027-c1fix-rfx-art-p",
             "datasets/ngnnguynchn/flag2027-rfsc4lf-art-p", "datasets/ngnnguynchn/flag2027-rf092-dev-emb-p"}
SECRET = [re.compile(x) for x in (r"hf_[A-Za-z0-9]{30,}", r"KAGGLE_KEY", r'"key"\s*:', r"ghp_[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}",
                                     r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"xox[baprs]-[A-Za-z0-9-]{10,}", r"kaggle\.json")]


def stages():
    return sorted(p.parent for p in HERE.glob("*/stage.json"))


def check(verbose=True):
    """Rebuild every stage.ipy from origin.original_notebook + patches and compare; check SOURCE.json / DECODED.json hashes, the
    redactions, the C1 variant guard, Python syntax of every cell and a secret scan of every file of the group."""
    bad = []
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    for sd in stages():
        sj = json.loads((sd / "stage.json").read_text(encoding="utf-8"))
        if (sj.get("origin") or {}).get("exactness") == "new":   # integration fixer r1 (18/11): new-code stage (t3s01_trca), checked by runner/check_stage.py
            say(f"{sd.name:15s} new code: skipped here (runner/check_stage.py checks the [cell] rule)")
            continue
        nbb = (sd / sj["origin"]["original_notebook"]).read_bytes()
        ipy = build_ipy(nbb, sj["patches"], sj.get("drop_cells", []))
        cur = (sd / "stage.ipy").read_bytes()
        ok_ipy = ipy.encode("utf-8") == cur and sha256b(cur) == sj["stage_ipy_sha256"]
        for c in ipy.split(CELL_SEP):
            if c.strip():
                compile(c, f"{sd.name}/stage.ipy", "exec")
        src = json.loads((sd / "original/SOURCE.json").read_text(encoding="utf-8"))
        ok_src = all(sha256b((HERE / f["path"]).read_bytes()) == f["sha256"] for f in src["files"])
        dec = json.loads((sd / "original/decoded/DECODED.json").read_text(encoding="utf-8"))
        ok_dec = all(sha256b((sd / e["path"]).read_bytes()) == e["sha256"] for e in dec["files"].values() if e.get("extracted"))
        # redactions: no organisers' gender blob, no dev zip member left in any notebook copy of the stage
        ok_red = True
        for nbp in sd.rglob("*.ipynb"):
            b = nbp.read_bytes(); nb = json.loads(b.decode("utf-8"))
            for _, s in code_sources(nb):
                if any(m.group(1) == "train/gender.csv" for m in BLOB.finditer(s)):
                    ok_red = False
                for m in CODEB64.finditer(s):
                    with zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1)))) as z:
                        ok_red &= DEVZIP_MEMBER not in z.namelist()
            ok_red &= GENDER_SHA256 not in [sha256b(v) for v in decode_payloads(b).values()]
        # variant guard (DESIGN 2.7)
        arts = {i["artifact"] for i in sj["inputs"]}
        ok_var = not (arts & FORBIDDEN)
        if sj.get("variant", {}).get("lora_c1") == "original" and sj["id"] != "t163l_orig":
            ok_var &= LORA_ORIG in arts
        # LoRA C1 policy (fixer 18/11, README "LoRA C1 policy"): consumers mount the saved original only (resolve saved_only) with the hash
        # required at input (variant.require_files); t163l_orig runs only when forced; post_checks are JSON checks (the runner reads them
        # as JSON), never a bare sha256; every stage states runner_requirements.literal_kaggle_input
        for i in [i for i in sj["inputs"] if i["artifact"] == LORA_ORIG]:
            ok_var &= i.get("resolve") == "saved_only"
            ok_var &= any(r.get("artifact") == LORA_ORIG and r.get("file") == "lora_C1.pt" and r.get("sha256") == C1_SHA256
                          for r in sj.get("variant", {}).get("require_files", []))
        pcs = sj.get("post_checks", [])
        ok_var &= all(("json_path" in c or "equals" in c) and "sha256" not in c for c in pcs)
        ok_var &= isinstance(sj.get("runner_requirements", {}).get("literal_kaggle_input"), bool)
        if sj["id"] == "t163l_orig":
            ok_var &= sj.get("force_only") is True
        st = dict(ipy=ok_ipy, source=ok_src, decoded=ok_dec, redaction=ok_red, variant=ok_var)
        say(f"{sd.name:14s} " + " ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in st.items()) + f"  ipy {sha256b(cur)[:16]}")
        bad += [f"{sd.name}:{k}" for k, v in st.items() if not v]
    hits = []
    for p in HERE.rglob("*"):
        if p.is_file() and p.name != Path(__file__).name and "__pycache__" not in p.parts:
            t = p.read_bytes().decode("utf-8", "replace")
            hits += [f"{p.relative_to(HERE)}: {r.pattern}" for r in SECRET if r.search(t)]
    say("secret scan:", "clean" if not hits else hits)
    bad += hits
    say("RESULT:", "ALL OK" if not bad else f"FAIL {bad}")
    return not bad


def redo_redaction(stage, exact):
    """Re-create the redacted copy from an exact private copy and compare (proves original/*.redacted.ipynb derives from it)."""
    sd = HERE / stage
    src = json.loads((sd / "original/SOURCE.json").read_text(encoding="utf-8"))
    raw = Path(exact).read_bytes()
    for f in src["files"]:
        d = f.get("derived_from")
        if d and d["md5"] == md5b(raw):
            spec = {"gender": any(r["kind"] == "redaction" for r in f["redactions"]), "devzip": any(r["kind"] == "dev-exclusion" for r in f["redactions"])}
            red, _ = redact(raw, spec)
            same = sha256b(red) == f["sha256"]
            print(f"{exact}: md5 {md5b(raw)} = SOURCE.json derived_from; redacted sha256 {sha256b(red)[:16]} {'==' if same else '!='} {f['path']}")
            return same
    print(f"{exact}: md5 {md5b(raw)} matches no derived_from entry of {stage}/original/SOURCE.json")
    return False


def decode(stage, outdir):
    """Write every payload embedded in the stage's redacted notebook (embedded code; gender.csv / dev zip are not there)."""
    sd = HERE / stage
    sj = json.loads((sd / "stage.json").read_text(encoding="utf-8"))
    for n, b in decode_payloads((sd / sj["origin"]["original_notebook"]).read_bytes()).items():
        p = Path(outdir) / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
        print(f"{sha256b(b)}  {n}")


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a[:1] == ["check"]:
        sys.exit(0 if check() else 1)
    elif a[:1] == ["redact"] and len(a) == 3:
        sys.exit(0 if redo_redaction(a[1], a[2]) else 1)
    elif a[:1] == ["decode"] and len(a) == 3:
        decode(a[1], a[2])
    else:
        print("usage: python _wpf_tools.py check | redact <stage_id> <exact_notebook.ipynb> | decode <stage_id> <out_dir>")
        sys.exit(2)
