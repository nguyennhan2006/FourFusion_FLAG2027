r"""Self-check of the stages of group 50_eval (pure Python, read-only, seconds). Exit code 0 = every check passed.

    python stages/50_eval/_tools/selfcheck_50_eval.py [--exact-dir DIR] [--release DIR] [--skip-staging]

  always   for e65xbs, e65xcs, e47rs, e50cc, e112f0sx, e112f1sx, e163i: the notebook in original/ has the sha256 of stage.json; stage.ipy
           is rebuilt from it (code cells with non-empty source, CRLF -> LF, the stage.json patches each hitting exactly `count` times,
           each cell + "\n# <<<FF-CELL>>>\n"; = release/flag2027_e2e/build_notebook.py stage_cells) and compared byte for byte;
           the files listed in original/SOURCE.json have their sha256;
           for a_factor: tree/ files vs stage.json tree_sha256, every stage.ipy cell parses and opens with print("[cell] ...", flush=True);
           every stage.json has the fields of DESIGN 2.2 and id == folder; 13.54 variant guard (e163i mounts the original-C1 artifact,
           no stage mounts a forbidden artifact); secret-like strings and dev-score payloads (also inside base64 zips) are absent;
           no ZIP_PREFIXES line of a stage still unpacks dev_set.zip (patch no_dev_zip);
           staging: when the folder repo/_staging_backup exists (it does in the staging tree, not in the published repository), every
           file outside its git-ignored _PRIVATE_NOT_FOR_PUBLICATION/ holds no dev score payload (base64 zips with dev score
           members, or with the md5 of a payload removed by redact_dev_refs.py); --skip-staging leaves this check out.
  --exact-dir DIR   folder with the exact executed notebooks of the team package (FACTOR_ENFUSE_c5rl/notebooks/2_eval): their sha256,
           the redaction of e50cc / e112f0sx / e112f1sx re-done and compared with original/, and the code of each exact notebook
           (only the patches that release/flag2027_e2e also applies, i.e. without "in_release": false) compared with stage.json
           release_e2e_code.sha256.
  --release DIR     release/flag2027_e2e/code: the <stage>.ipy files there have the sha256 recorded in stage.json.
  --exact-dir and --release need team-private files that are not part of the published repository (the exact notebooks embed dev
  score data; release/flag2027_e2e/code/e50cc.ipy too): the public check is the form without them."""
import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
import re
import sys
import zipfile
from pathlib import Path

G = Path(__file__).resolve().parents[1]                   # stages/50_eval
CELL_SEP = "\n# <<<FF-CELL>>>\n"
E_STAGES = ["e65xbs", "e65xcs", "e47rs", "e50cc", "e112f0sx", "e112f1sx", "e163i"]
REQUIRED = ["id", "group", "title", "kind", "role_in_1354", "compute", "origin", "inputs", "outputs", "env", "smoke_env", "patches",
            "drop_cells", "neutralize_upload", "pip_policy", "variant", "skip_with", "verify", "stage_ipy_sha256", "todo"]
ORIGINAL_C1_ART = "datasets/ngnnguynchn/flag2027-rfsfxc5rl-art-p"
FORBIDDEN = ["datasets/ngnnguynchn/flag2027-rfs-lorafix1-p", "datasets/ngnnguynchn/flag2027-c1fix-lora-p", "datasets/ngnnguynchn/flag2027-c1fix-rfx-art-p",
             "datasets/ngnnguynchn/flag2027-rfsc4lf-art-p", "datasets/ngnnguynchn/flag2027-rf092-dev-emb-p"]
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|KAGGLE_KEY\s*[=:]\s*['\"]?[0-9a-f]{20,}"
                    r"|\"key\"\s*:\s*\"[0-9a-f]{20,}\"|-----BEGIN [A-Z ]*PRIVATE KEY-----")
DEV_MEMBER = re.compile(r"(_dev\.zip$|sub_score_v4_.*\.txt$)")
B64_RUN = re.compile(r'((?:"[A-Za-z0-9+/=]+"[\s,]*)+)')       # one literal, or several separated by blanks / commas (_b = ("..", ".."))
STAGING = G.parents[1] / "_staging_backup"                   # repo/_staging_backup (staging tree only)
PRIVATE = "_PRIVATE_NOT_FOR_PUBLICATION"
J = lambda x: "".join(x) if isinstance(x, list) else x
sha = lambda b: hashlib.sha256(b).hexdigest()
FAIL = []


def check(ok, what):
    print(("PASS  " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAIL.append(what)


def code_text(nb_bytes, patches):
    nb = json.loads(nb_bytes.decode("utf-8"))
    srcs = [J(c["source"]).replace("\r\n", "\n") for c in nb["cells"] if c["cell_type"] == "code" and J(c["source"]).strip()]
    for p in patches:
        hits = sum(s.count(p["old"]) for s in srcs)
        if hits != p["count"]:
            raise ValueError(f"patch {p['old'][:50]!r} hits {hits}, expected {p['count']}")
        srcs = [s.replace(p["old"], p["new"]) for s in srcs]
    return "".join(s.rstrip("\n") + CELL_SEP for s in srcs)


def b64_zips(text):
    """Base64 string literals (joined when adjacent) of >= 2000 characters that decode to a zip -> list of raw bytes."""
    out = []
    for m in B64_RUN.finditer(text):
        blob = "".join(re.findall(r'"([A-Za-z0-9+/=]+)"', m.group(1)))
        if len(blob) < 2000:
            continue
        try:
            raw = base64.b64decode(blob)
        except Exception:
            continue
        if raw[:2] == b"PK":
            out.append(raw)
    return out


def payload_scan(text, where):
    """base64 zips inside a text: no member that looks like a dev score file, no secret-like string inside members."""
    for raw in b64_zips(text):
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile:
            check(False, f"{where}: unreadable base64 zip payload ({len(raw)} bytes): inspect by hand")
            continue
        bad = [n for n in z.namelist() if DEV_MEMBER.search(n)]
        check(not bad, f"{where}: base64 zip ({len(z.namelist())} members) holds no dev score file {bad if bad else ''}")
        sec = [n for n in z.namelist() if SECRET.search(z.read(n).decode("utf-8", "replace"))]
        check(not sec, f"{where}: no secret-like string inside the embedded zip members {sec if sec else ''}")


def dev_payloads(text, known_md5):
    """-> descriptions of the dev score payloads among the base64 zips of a text (no check() call; used for the staging scan)."""
    found = []
    for raw in b64_zips(text):
        h = hashlib.md5(raw).hexdigest()
        if h in known_md5:
            found.append(f"zip md5 {h} (a payload removed by redact_dev_refs.py)")
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
            bad = [n for n in z.namelist() if DEV_MEMBER.search(n) or hashlib.md5(z.read(n)).hexdigest() in known_md5]
        except (zipfile.BadZipFile, RuntimeError, ValueError, EOFError):
            found.append(f"unreadable base64 zip ({len(raw)} bytes): inspect by hand")
            continue
        if bad:
            found.append(f"zip md5 {h} with dev member(s) {bad[:4]}")
    return found


def staging_scan(known_md5):
    """Files of repo/_staging_backup outside _PRIVATE_NOT_FOR_PUBLICATION/ (they would be committed with the tree) hold no dev payload."""
    if not STAGING.is_dir():
        print("SKIP  staging scan: no repo/_staging_backup (published repository)")
        return
    leaks = []
    for p in sorted(STAGING.rglob("*")):
        rel = p.relative_to(STAGING).as_posix()
        if not p.is_file() or rel.split("/")[0] == PRIVATE or p.suffix.lower() not in (".ipynb", ".ipy", ".py", ".json", ".txt", ".sh", ".md"):
            continue
        t = p.read_bytes().decode("utf-8", "replace")
        if p.suffix.lower() == ".ipynb":
            try:
                t = "\n".join(J(c.get("source", "")) for c in json.loads(t)["cells"])
            except (ValueError, KeyError, TypeError):
                pass
        leaks += [f"_staging_backup/{rel}: {d}" for d in dev_payloads(t, known_md5)]
    for x in leaks:
        print("      ", x)
    check(not leaks, f"staging scan: no dev score payload in _staging_backup outside {PRIVATE}/ ({len(leaks)} found; move those "
                     f"files into {PRIVATE}/ or out of the tree before any commit)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exact-dir")
    ap.add_argument("--release")
    ap.add_argument("--skip-staging", action="store_true", help="leave out the scan of repo/_staging_backup")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("redact_dev_refs", G / "_tools/redact_dev_refs.py")
    RD = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(RD)
    for sid in E_STAGES + ["a_factor"]:
        d = G / sid
        st = json.loads((d / "stage.json").read_text(encoding="utf-8"))
        check(st.get("id") == sid and st.get("group") == "50_eval" and all(k in st for k in REQUIRED),
              f"{sid}: stage.json fields (missing: {[k for k in REQUIRED if k not in st]})")
        ipy = (d / "stage.ipy").read_bytes()
        check(sha(ipy) == st["stage_ipy_sha256"], f"{sid}: stage.ipy sha256 = stage.json stage_ipy_sha256")
        mounted = [i["artifact"] for i in st["inputs"]]
        check(not [x for x in mounted if x in FORBIDDEN], f"{sid}: no forbidden (C1 fix / dev) artifact among the inputs")
        if sid in E_STAGES:
            nbp = d / st["origin"]["notebook"]
            nb = nbp.read_bytes()
            check(sha(nb) == st["origin"]["notebook_sha256"], f"{sid}: original notebook sha256")
            check(len(list((d / "original").glob("*.ipynb"))) == 1, f"{sid}: exactly one notebook in original/")
            try:
                rebuilt = code_text(nb, st["patches"]).encode("utf-8")
            except ValueError as e:
                rebuilt = b""
                print("      ", e)
            check(rebuilt == ipy, f"{sid}: stage.ipy rebuilt from original/ + patches is byte-identical")
            zp = [ln for ln in ipy.decode("utf-8").split("\n") if ln.startswith("ZIP_PREFIXES =")]
            if zp:
                check(all('"dev_set"' not in ln.split("#")[0] for ln in zp), f"{sid}: ZIP_PREFIXES never unpacks dev_set.zip (patch no_dev_zip)")
            src = json.loads((d / "original/SOURCE.json").read_text(encoding="utf-8"))
            for f in src["files"]:
                check(sha((d / "original" / f["path"]).read_bytes()) == f["sha256"], f"{sid}: original/{f['path']} sha256 = SOURCE.json")
            cells_src = "\n".join(J(c["source"]) for c in json.loads(nb.decode("utf-8"))["cells"])
            payload_scan(cells_src, f"{sid}: original notebook (cell sources)")
            if a.exact_dir:
                ex = Path(a.exact_dir) / Path(st["origin"]["source_file"]).name
                exb = ex.read_bytes()
                check(sha(exb) == st["origin"]["exact_notebook_sha256"], f"{sid}: exact notebook {ex.name} sha256")
                if st["origin"].get("redacted"):
                    red, _ = RD.redact(sid, exb)
                    check(red == nb, f"{sid}: redaction of the exact notebook reproduces original/ byte for byte")
                rel_patches = [q for q in st["patches"] if q.get("in_release", True)]
                check(sha(code_text(exb, rel_patches).encode("utf-8")) == st["release_e2e_code"]["sha256"],
                      f"{sid}: code of the exact notebook (+ patches) = release/flag2027_e2e/code/{sid}.ipy sha256")
            if a.release:
                rp = Path(a.release) / f"{sid}.ipy"
                rb = rp.read_bytes()
                check(sha(rb) == st["release_e2e_code"]["sha256"], f"{sid}: {rp.name} sha256 as recorded")
                check((rb == ipy) == st["release_e2e_code"]["equal_to_stage_ipy"], f"{sid}: equality with {rp.name} as recorded")
        else:
            for rel, h in st["tree_sha256"].items():
                check(sha((d / rel).read_bytes()) == h, f"{sid}: {rel} sha256")
            src = json.loads((d / "original/SOURCE.json").read_text(encoding="utf-8"))
            for f in src["provenance_files"]:
                check(sha((d / f["path"]).read_bytes()) == f["sha256"], f"{sid}: {f['path']} sha256 = SOURCE.json")
            cells = [c for c in ipy.decode("utf-8").split(CELL_SEP) if c.strip()]
            for k, c in enumerate(cells, 1):
                try:
                    ast.parse(c); okp = True
                except SyntaxError:
                    okp = False
                check(okp and re.match(r'print\("\[cell\] [^"]+", flush=True\)', c.split("\n", 1)[0]) is not None,
                      f"{sid}: new cell {k}/{len(cells)} parses and opens with print(\"[cell] ...\", flush=True)")
        if sid == "e163i":
            check(ORIGINAL_C1_ART in mounted and st["variant"].get("lora_c1") == "original"
                  and any(r["file"] == "lora_C1.pt" and r["sha256"].startswith("b97372e3504a131d") for r in st["variant"]["require_files"]),
                  "e163i: 13.54 variant: artifact with the ORIGINAL LoRA C1 (lora_C1.pt b97372e3...) mounted, check declared")
    if not a.skip_staging:
        staging_scan({h for _p, h in RD.REF_50.values()} | {RD.DROP_112_MD5})
    for p in sorted(G.rglob("*")):
        if p.is_file():
            t = p.read_bytes().decode("utf-8", "replace")
            check(not SECRET.search(t), f"secret scan: {p.relative_to(G).as_posix()}")
    print(f"\n{'ALL CHECKS PASSED' if not FAIL else str(len(FAIL)) + ' CHECK(S) FAILED'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
