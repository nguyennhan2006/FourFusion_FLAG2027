"""Compare the code that actually ran (the executed copies kept under repo/stages/) with the src/ copies.

    python src/_tools/scan_stage_code.py [--repo <repo root>] [--out src/_tools/stage_code_scan.json]

Read-only on stages/ and src/; writes one JSON file (default: src/_tools/stage_code_scan.json), which
build_src_copy.py merges into SRC_MANIFEST.json (files[].executed_versions with "origin": "stage-scan").

What is scanned, for every stage folder stages/<group>/<id>/:
  1. every file whose name is also the name of a src/ copy (decoded / embedded payloads written by the stage packages,
     plain executed copies in original/, and the stage's own runnable copies), matched to a src/ path by the longest
     common path suffix, then by content when the name is ambiguous;
  2. every executed notebook (*.ipynb) under original/: base64 payloads in code cells (plain, gzip, zlib, or a zip whose
     members are listed), named by the string literal just before the payload (e.g. ("train/rfs_train.py", "<b64>"),
     ("FUSENET-01", "<b64>"), open('flag_extract.py', 'wb')...), else by content;
  3. code cells of those notebooks whose source is (nearly) a src/ file pasted as a cell (e.g. kaggle/vendor_t3enc/
     t3enc_cell.py): at least 80 % of the non-blank lines shared in both directions.
Each hit records the in-repo location, sha256 / md5 (and the LF form), whether it equals the src/ copy byte for byte,
after CRLF->LF, only in the final newline (same-lines), or differs; for a difference a diff summary (executed -> src
copy; lines_only_in_copy / lines_only_in_executed; hunks_ignoring_indent_and_blank_lines tells a pure re-indentation or
wrapper apart from a code change). stage_exactness is origin.exactness of the stage (builder-regenerated notebooks are
NOT executed copies).
Locations: original (the executed copy of the stage run), original-evidence / original-staging / original-pre-staging
(other runs or other copies kept as evidence), stage-copy (the stage's runnable copy; may carry CHANGES.md edits).
Standard library only (Python >= 3.8)."""
import argparse
import ast
import base64
import binascii
import collections
import difflib
import gzip
import hashlib
import io
import json
import re
import zipfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", ".git"}
TEXT_EXT = {".py", ".json", ".sh", ".csv", ".ipynb", ".txt", ".ipy", ".md"}
MAX_BYTES = 8_000_000
B64_RUN = re.compile(r"(?:[A-Za-z0-9+/]{40,}(?:\\n|\s)*){4,}[A-Za-z0-9+/]*={0,2}")   # last line of a wrapped payload may be short
STR_LIT = re.compile(r"""(?:"([^"\n\\]{1,200})"|'([^'\n\\]{1,200})')""")
EXP_ID = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+")
RAW_LIT = re.compile(r"= r'''(.*?)'''\n", re.S)


def h(b):
    lf = b.replace(b"\r\n", b"\n")
    d = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "md5": hashlib.md5(b).hexdigest()}
    if lf != b:
        d["sha256_lf"] = hashlib.sha256(lf).hexdigest()
    return d


def lf_text(b):
    try:
        return b.replace(b"\r\n", b"\n").decode("utf-8")
    except UnicodeDecodeError:
        return None


def load_src(src):
    out = {}
    for f in sorted(src.rglob("*")):
        if not f.is_file() or SKIP_DIRS & set(f.parts):
            continue
        rel = f.relative_to(src).as_posix()
        if rel.startswith("_tools/") or rel in ("README.md", "SRC_MANIFEST.json", ".gitattributes") or rel.endswith(".reduced.jsonl"):
            continue
        b = f.read_bytes()
        t = lf_text(b)
        out[rel] = {"bytes": b, "sha256": hashlib.sha256(b).hexdigest(),
                    "sha256_lf": hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest(), "text": t}
    return out


def diff_summary(exe_text, src_text):
    a, b = exe_text.splitlines(), src_text.splitlines()
    hunks, plus, minus, first = 0, 0, 0, []
    for g in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(0):
        hunks += 1
        i1, i2, j1, j2 = g[0][1], g[-1][2], g[0][3], g[-1][4]
        minus += sum(i2_ - i1_ for tag, i1_, i2_, j1_, j2_ in g if tag in ("replace", "delete"))
        plus += sum(j2_ - j1_ for tag, i1_, i2_, j1_, j2_ in g if tag in ("replace", "insert"))
        if len(first) < 4:
            sample = next((b[j] for tag, _, _, j1_, j2_ in g if tag in ("replace", "insert") for j in range(j1_, j2_) if b[j].strip()), None)
            if sample is None:
                sample = next((("-" + a[i]) for tag, i1_, i2_, _, _ in g if tag in ("replace", "delete") for i in range(i1_, i2_) if a[i].strip()), "")
            first.append(f"executed {i1 + 1}-{i2} / copy {j1 + 1}-{j2}: {sample.strip()[:100]}")
    return {"hunks": hunks, "lines_only_in_copy": plus, "lines_only_in_executed": minus,
            "executed_lines": len(a), "copy_lines": len(b), "first_hunks": first}


def compare(b, src_rel, srcs):
    s = srcs[src_rel]
    d = h(b)
    lf = hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()
    if d["sha256"] == s["sha256"]:
        m = "exact"
    elif lf == s["sha256_lf"]:
        m = "lf-equal"
    else:
        m = "differs"
    rec = {**d, "src_path": src_rel, "src_sha256": s["sha256"], "match": m}
    if m == "differs":
        et = lf_text(b)
        if et is not None and s["text"] is not None:
            ds = diff_summary(et, s["text"])
            if ds["hunks"] == 0:
                rec["match"] = "same-lines"      # only a final newline (or other line-terminator detail) differs
            else:
                rec["diff"] = ds
                ni = diff_summary("\n".join(x.strip() for x in et.splitlines() if x.strip()),
                                  "\n".join(x.strip() for x in s["text"].splitlines() if x.strip()))
                rec["diff"]["hunks_ignoring_indent_and_blank_lines"] = ni["hunks"]
    return rec


class Matcher:
    def __init__(self, srcs):
        self.srcs = srcs
        self.by_sha = {}
        for p, s in srcs.items():
            self.by_sha.setdefault(s["sha256"], p)
            self.by_sha.setdefault(s["sha256_lf"], p)
        self.base = collections.defaultdict(list)
        for p in srcs:
            self.base[p.rsplit("/", 1)[-1]].append(p)

    def by_path(self, name):
        n = name.replace("\\", "/").strip().lstrip("./")
        parts = [x for x in n.split("/") if x]
        if not parts:
            return None, []
        for k in range(len(parts), 0, -1):
            suf = "/".join(parts[-k:])
            c = [p for p in self.srcs if p == suf or p.endswith("/" + suf)]
            if len(c) == 1:
                return c[0], c
            if len(c) > 1:
                return None, c
        return None, []

    def by_hash(self, b):
        for x in (hashlib.sha256(b).hexdigest(), hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()):
            if x in self.by_sha:
                return self.by_sha[x]
        return None

    def by_content(self, b, cands=None):
        sha, lf = hashlib.sha256(b).hexdigest(), hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()
        for x in (sha, lf):
            if x in self.by_sha and (cands is None or self.by_sha[x] in cands):
                return self.by_sha[x]
        t = lf_text(b)
        if t is None or len(t) < 400:
            return None
        best, bl = None, 0
        for p in (cands or self.srcs):
            st = self.srcs[p]["text"]
            if not st:
                continue
            n = 0
            lim = min(len(st), len(t), 4000)
            while n < lim and st[n] == t[n]:
                n += 1
            if n > bl:
                best, bl = p, n
        return best if bl >= 300 else None

    def resolve(self, name, b):
        p, c = self.by_path(name) if name else (None, [])
        if p:
            return p, "path"
        if EXP_ID.fullmatch(name or "") and f"Experiment/{name}/run.py" in self.srcs:
            return f"Experiment/{name}/run.py", "experiment-id"
        q = self.by_content(b, c or None)
        if q:
            return q, "content"
        return None, ("ambiguous-name" if c else "no-src")


def location(parts):
    if "original" not in parts:
        return "stage-copy"
    for tag, lab in (("evidence", "original-evidence"), ("pre_staging", "original-pre-staging"), ("staging", "original-staging")):
        if tag in parts:
            return lab
    return "original"


def decode_payload(s64):
    raw = re.sub(r"\\n|\s", "", s64)
    try:
        b = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return []
    if b[:2] == b"\x1f\x8b":
        try:
            return [("gzip", None, gzip.decompress(b))]
        except OSError:
            return []
    if b[:1] == b"\x78":
        try:
            return [("zlib", None, zlib.decompress(b))]
        except zlib.error:
            pass
    if b[:4] == b"PK\x03\x04":
        try:
            z = zipfile.ZipFile(io.BytesIO(b))
            return [("zip-member", i.filename, z.read(i)) for i in z.infolist() if not i.is_dir()]
        except zipfile.BadZipFile:
            return []
    return [("b64", None, b)]


def name_before(src_text, start):
    """Name of a payload: the string just before it in a ("name", "<b64>") pair, else the file of the nearest open('name', ...)."""
    pre = src_text[max(0, start - 400):start].rstrip()
    pre = re.sub(r"""(['"]{1,3})\s*$""", "", pre)   # the payload's own opening quote(s)
    pre = re.sub(r"""''\.join\(\s*$|\.join\(\s*$""", "", pre).rstrip()
    pre = re.sub(r"""(['"]{1,3})\s*$""", "", pre)
    m = re.search(r"""["']([^"'\n\\]{1,200})["']\s*,\s*$""", pre)
    if m:
        return m.group(1)
    opens = re.findall(r"""open\(\s*['"]([^'"\n]{1,200})['"]""", pre)
    return opens[-1] if opens else None


def big_strings(s):
    """(variable, value) of every str constant >= 2000 chars assigned in a cell (shell / magic lines replaced by pass)."""
    lines = [(l[:len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("!", "%")) else l for l in s.splitlines()]
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) \
                and len(node.value.value) >= 2000:
            t = node.targets[0]
            out.append((t.id if isinstance(t, ast.Name) else None, node.value.value))
    return out


def cell_sources(nb):
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        s = c.get("source", "")
        yield i, ("".join(s) if isinstance(s, list) else s)


def build_line_index(srcs):
    idx, sizes = collections.defaultdict(set), {}
    for p, s in srcs.items():
        if not s["text"] or not p.endswith((".py", ".sh")):
            continue
        lines = {x.strip() for x in s["text"].splitlines() if x.strip()}
        if len(lines) < 15:
            continue
        sizes[p] = len(lines)
        for x in lines:
            idx[x].add(p)
    return idx, sizes


def scan(repo):
    stages, src = repo / "stages", repo / "src"
    srcs = load_src(src)
    M = Matcher(srcs)
    lidx, lsizes = build_line_index(srcs)
    srcvar = {"SRC_" + re.sub(r"[^A-Z0-9]", "_", p.upper()): p for p in srcs}   # e50cc: SRC_KAGGLE_FLAG_LIB_PY = r'''...'''
    records, unmapped, seen = [], [], set()

    def add(rec):
        key = (rec["stage"], rec["container"], rec["kind"], rec.get("member"), rec.get("cell"), rec["sha256"])
        if key not in seen:
            seen.add(key)
            records.append(rec)

    stage_dirs = sorted({p.parent for p in stages.rglob("stage.json")})
    for sd in stage_dirs:
        try:
            sj = json.loads((sd / "stage.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            sj = {}
        sid = sj.get("id") or sd.name
        exactness = (sj.get("origin") or {}).get("exactness")
        for f in sorted(sd.rglob("*")):
            if not f.is_file() or SKIP_DIRS & set(f.parts) or f.suffix.lower() not in TEXT_EXT or f.stat().st_size > MAX_BYTES:
                continue
            if f.name in ("stage.json", "SOURCE.json", "DECODED.json", "EMBEDDED.json", "CHANGES.md"):
                continue
            rel_stage = f.relative_to(sd).as_posix()
            parts = rel_stage.split("/")[:-1]
            loc = location(parts)
            container = f.relative_to(repo).as_posix()
            b = f.read_bytes()
            base_rec = {"stage": sid, "stage_exactness": exactness, "location": loc, "container": container}
            # 1. the file itself
            if f.name in M.base:
                p, how = M.resolve(rel_stage, b)
                if p:
                    add({**base_rec, "kind": "file", **compare(b, p, srcs), "matched_by": how})
            # 2. / 3. executed notebooks
            if f.suffix.lower() == ".ipynb" and loc.startswith("original"):
                try:
                    nb = json.loads(b.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                nbh = hashlib.md5(b).hexdigest()
                for ci, s in cell_sources(nb):
                    for m in B64_RUN.finditer(s):
                        if len(m.group()) < 400:
                            continue
                        hint = name_before(s, m.start())
                        for kind, member, pb in decode_payload(m.group()):
                            nm = member or hint
                            p, how = M.resolve(nm, pb)
                            r = {**base_rec, "container_md5": nbh, "cell": ci, "kind": "blob-" + kind, "member": nm}
                            if p:
                                add({**r, **compare(pb, p, srcs), "matched_by": how})
                            else:
                                d = h(pb)
                                key = (sid, container, ci, nm, d["sha256"])
                                if key not in seen:
                                    seen.add(key)
                                    unmapped.append({**r, **d, "reason": how})
                    for var, val in big_strings(s):    # source text kept in a string constant (CORE_SOURCE = '...', SRC_<PATH> = r'''...''')
                        vb = val.encode("utf-8")
                        p = srcvar.get(var or "") or M.by_hash(vb) or M.by_hash(vb + b"\n")   # name, else exact / LF hash only
                        if p:
                            add({**base_rec, "container_md5": nbh, "cell": ci, "kind": "cell-string-constant", "member": var,
                                 **compare(vb, p, srcs), "matched_by": "variable-name" if srcvar.get(var or "") else "content-hash"})
                    cl = {x.strip() for x in s.splitlines() if x.strip()}
                    if len(cl) < 15:
                        continue
                    cnt = collections.Counter(p for x in cl for p in lidx.get(x, ()))
                    for p, n in cnt.most_common(3):
                        if n >= 0.8 * lsizes[p] and n >= 0.8 * len(cl):
                            kind, cb = "cell-source", s.encode("utf-8")
                            lit = RAW_LIT.search(s)       # module written by the cell from a r'''...''' string (e50cc style)
                            if lit and len(lit.group(1)) > 0.8 * len(s):
                                kind, cb = "cell-string-literal", lit.group(1).encode("utf-8")
                            add({**base_rec, "container_md5": nbh, "cell": ci, "kind": kind, "member": None,
                                 **compare(cb, p, srcs), "matched_by": "lines", "shared_lines": n,
                                 "copy_nonblank_lines": lsizes[p], "cell_nonblank_lines": len(cl)})
    records.sort(key=lambda r: (r["src_path"], r["stage"], r["container"], r.get("cell") or -1))
    summ = collections.Counter((r["location"], r["match"]) for r in records)
    return {
        "schema": "fourfusion-stage-code-scan/1",
        "about": __doc__.split("\n")[0],
        "stages_scanned": len(stage_dirs),
        "src_files": len(srcs),
        "summary": {f"{a}:{b}": n for (a, b), n in sorted(summ.items())},
        "records": records,
        "unmapped_payloads": sorted(unmapped, key=lambda r: (r["stage"], r["container"], r["cell"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, default=HERE.parent.parent, help="repo root (holds stages/ and src/)")
    ap.add_argument("--out", type=Path, default=HERE / "stage_code_scan.json")
    a = ap.parse_args()
    res = scan(a.repo.resolve())
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(res["records"]), "unmapped_payloads": len(res["unmapped_payloads"]), **res["summary"]}, indent=1))


if __name__ == "__main__":
    main()
