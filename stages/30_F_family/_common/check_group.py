#!/usr/bin/env python3
"""Self-check of the stages in stages/30_F_family/ (WP-E). Light: reads files, hashes, base64 decode, ast.parse. Nothing is executed.

For every <stage>/stage.json:
  1. stage.ipy sha256 / cell count = stage.json "stage_ipy";
  2. stage built from an original notebook: stage.ipy == code cells of origin.original_in_repo (CRLF -> LF, empty cells dropped,
     drop_cells removed) with stage.json "patches" applied (each exactly `count` times), then "insert_cells" (as
     runner/check_stage.py), joined with CELL_SEP;
  3. original/ files: sha256 = SOURCE.json (repo_sha256 for the notebook, source_sha256 for byte-identical copies);
     with --exact, the redacted copies are re-made from origin.source_file (if readable; else an explicit 'skipped' line) into a
     temporary file outside the repository and must give the same bytes;
  4. every cell parses (IPython `!` / `%` lines blanked first); every cell of a "new" stage opens with print("[cell] ...");
  5. no input is a forbidden 13.54 artifact (DESIGN 2.7); no CODE_B64 in the group carries the dev score zip;
  6. secret patterns in every file and in every decoded CODE_B64 member; JSON files parse; code/CODE_MANIFEST.json sha256 match.
    python check_group.py [--exact]          exit 1 on any failure"""
import sys
sys.dont_write_bytecode = True        # no __pycache__ inside the repository tree (the redaction tool is imported below)
import ast, base64, hashlib, importlib.util, io, json, os, re, tempfile, zipfile
from pathlib import Path

GRP = Path(__file__).resolve().parents[1]
CELL_SEP = "\n# <<<FF-CELL>>>\n"
FORBIDDEN = {"datasets/ngnnguynchn/flag2027-rfs-lorafix1-p", "datasets/ngnnguynchn/flag2027-c1fix-lora-p", "datasets/ngnnguynchn/flag2027-c1fix-rfx-art-p",
             "datasets/ngnnguynchn/flag2027-rfsc4lf-art-p", "datasets/ngnnguynchn/flag2027-rf092-dev-emb-p"}
SECRET = {"hf_token": "hf" + r"_[A-Za-z0-9]{30,}", "kaggle_key": "KAGGLE" + "_KEY|" + "KGAT" + r"_[A-Za-z0-9]{10,}",   # split so this file
          "key_json": r"\"key\"\s*:\s*\"[A-Za-z0-9]{20,}", "github": "ghp" + r"_[A-Za-z0-9]{20,}", "aws": "AKIA" + r"[0-9A-Z]{16}"}  # never matches itself
DEV_ZIP = "Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip"
J = lambda x: "".join(x) if isinstance(x, list) else x
sha = lambda b: hashlib.sha256(b).hexdigest()
FAIL = []


def bad(msg):
    FAIL.append(msg); print("  FAIL", msg)


def cells_of(nb_path, drop=()):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        s = J(c["source"]).replace("\r\n", "\n")
        if c["cell_type"] != "code" or not s.strip():
            continue
        first = s.lstrip().splitlines()[0] if s.strip() else ""
        if any(re.search(rx, first) for rx in drop):
            continue
        out.append(s)
    return out


def parse_ok(cell):
    src = "\n".join((l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l) for l in cell.split("\n"))
    ast.parse(src)


def b64_members(text, path=None):
    """Members of every CODE_B64 zip in a stage.ipy / .py text, or (path *.ipynb) in the joined code-cell sources of the notebook:
    in the raw notebook JSON the quotes are escaped, so the regex runs on the parsed cell text."""
    if path is not None and str(path).endswith(".ipynb"):
        try:
            nb = json.loads(text)
            text = "\n".join(J(c.get("source", "")) for c in nb.get("cells", []) if c.get("cell_type") == "code")
        except ValueError:
            pass
    out = []
    for m in re.finditer(r'CODE_B64 = "([A-Za-z0-9+/=]{400,})"', text):
        z = zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1))))
        out += [(n, z.read(n)) for n in z.namelist()]
    return out


def main(exact=False):
    spec = importlib.util.spec_from_file_location("redact", GRP / "_common/redact_embedded_dev_zip.py")
    RD = importlib.util.module_from_spec(spec); spec.loader.exec_module(RD)
    stages = sorted(p for p in GRP.iterdir() if (p / "stage.json").exists())
    for d in stages:
        print(f"== {d.name}")
        sj = json.loads((d / "stage.json").read_text(encoding="utf-8"))
        src = json.loads((d / "original/SOURCE.json").read_text(encoding="utf-8"))
        text = (d / "stage.ipy").read_bytes()
        if sha(text) != sj["stage_ipy"]["sha256"] or text.decode("utf-8").count(CELL_SEP) != sj["stage_ipy"]["cells"]:
            bad(f"{d.name}: stage.ipy differs from stage.json stage_ipy")
        cells = text.decode("utf-8").split(CELL_SEP)
        assert cells[-1] == "", f"{d.name}: stage.ipy must end with the cell separator"
        cells = cells[:-1]
        o = sj["origin"]
        if o.get("original_in_repo", "").endswith(".ipynb"):
            reb = cells_of(d / o["original_in_repo"], sj.get("drop_cells", []))
            for p in sj.get("patches", []):
                hits = sum(c.count(p["old"]) for c in reb)
                if hits != p["count"]:
                    bad(f"{d.name}: patch found {hits} times, expected {p['count']}: {p['old'][:60]!r}")
                reb = [c.replace(p["old"], p["new"]) for c in reb]
            for ins in sj.get("insert_cells") or []:                       # = runner/check_stage.py apply_rules step 4
                at = ins.get("at", "end")
                reb.insert(0 if at == "start" else len(reb) if at == "end" else int(at), ins["source"].replace("\r\n", "\n"))
            if "".join(c.rstrip("\n") + CELL_SEP for c in reb).encode("utf-8") != text:
                bad(f"{d.name}: stage.ipy is not original/ + patches + insert_cells")
            else:
                print(f"  stage.ipy = {o['original_in_repo']} + {len(sj.get('patches', []))} patch(es) + "
                      f"{len(sj.get('insert_cells') or [])} inserted cell(s): OK ({len(reb)} cells)")
        for f in src.get("files", []):
            p = d / f["repo_path"]
            if not p.exists():
                bad(f"{d.name}: missing {f['repo_path']}"); continue
            want = f.get("repo_sha256") or (f.get("source_sha256") if f.get("byte_identical_to_source") else f.get("sha256"))
            if want and sha(p.read_bytes()) != want:
                bad(f"{d.name}: {f['repo_path']} sha256 differs from SOURCE.json")
            if exact and f.get("byte_identical_to_source") is False and not Path(f.get("copied_from", "")).exists():
                print(f"  --exact: re-redaction of {f['repo_path']} skipped (source not readable here: {f.get('copied_from')})")
            elif exact and f.get("byte_identical_to_source") is False:
                fd, tmpn = tempfile.mkstemp(suffix=".ipynb"); os.close(fd); tmp = Path(tmpn)      # outside the repository
                try:
                    rep = RD.redact(f["copied_from"], tmp)
                    if rep["output_sha256"] != sha(p.read_bytes()):
                        bad(f"{d.name}: re-redaction of {f['copied_from']} gives another file")
                    else:
                        print(f"  redaction re-made from the exact notebook: identical ({rep['output_sha256'][:16]})")
                finally:
                    tmp.unlink(missing_ok=True)
        for k, c in enumerate(cells):
            try:
                parse_ok(c)
            except SyntaxError as e:
                bad(f"{d.name}: cell {k + 1} does not parse: {e}")
            if o.get("exactness") == "new" and not c.lstrip().startswith('print("[cell]'):
                bad(f"{d.name}: new cell {k + 1} does not open with print(\"[cell] ...\")")
        for i in sj["inputs"]:
            if i["artifact"] in FORBIDDEN:
                bad(f"{d.name}: forbidden 13.54 input {i['artifact']}")
        cm = d / "code/CODE_MANIFEST.json"
        if cm.exists():
            for rel, meta in json.loads(cm.read_text(encoding="utf-8")).items():
                if sha((d / "code" / rel).read_bytes()) != meta["sha256"]:
                    bad(f"{d.name}: code/{rel} sha256 differs from CODE_MANIFEST.json")
            for k, c in enumerate((d / "code").rglob("*.py")):
                try:
                    ast.parse(c.read_text(encoding="utf-8"))
                except SyntaxError as e:
                    bad(f"{d.name}: {c} does not parse: {e}")
    print("== files of the group (secrets, dev zip, JSON)")
    n = nb_scanned = 0
    for f in sorted(p for p in GRP.rglob("*") if p.is_file()):
        if "__pycache__" in f.parts:
            bad(f"stray cache file {f}"); continue
        b = f.read_bytes(); n += 1
        t = b.decode("utf-8", errors="replace")
        mem = b64_members(t, f) if "CODE_B64" in t else []
        blobs = [(f.name, t)] + [(f"{f.name}:{m}", x.decode('utf-8', 'replace')) for m, x in mem]
        for name, tx in blobs:
            for k, rx in SECRET.items():
                if re.search(rx, tx):
                    bad(f"secret pattern {k} in {f.relative_to(GRP)} ({name})")
        if any(m == DEV_ZIP for m, _ in mem):
            bad(f"{f.relative_to(GRP)} still embeds {DEV_ZIP}")
        nb_scanned += bool(mem) and f.suffix == ".ipynb"
        if f.suffix == ".json":
            json.loads(t)
    print(f"  {n} files scanned ({nb_scanned} notebooks with CODE_B64 decoded from their parsed cells)")
    print("RESULT:", "FAIL" if FAIL else "OK", f"({len(stages)} stages)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main("--exact" in sys.argv))
