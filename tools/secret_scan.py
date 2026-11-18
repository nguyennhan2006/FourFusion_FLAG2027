#!/usr/bin/env python3
"""Scan files before they are published: secrets, embedded organisers' / dev data, data-like files (DESIGN 2.8, user rule 6).

    python tools/secret_scan.py [PATH ...] [--json] [--exclude GLOB ...] [--max-mb 50]

Default PATH = the repository root. For every text file (and every code cell AND output of a notebook) the scanner looks for
secret patterns; every base64 literal of >= 200 characters is decoded, gunzipped / unzipped when it is a gzip or zip payload, and
scanned again (recursively), with the SHA-256 of every decoded payload / zip member compared with known organisers' and dev files.
It never prints a secret: only the pattern name, the file, the line and a masked preview.
Severity: HIGH = secret, the team's credential stores, organisers' / dev data, VoxBlink-named embedded arrays or a _PRIVATE* folder (exit code 1); MED = data-like file (arrays, weights, media, zips) or a file
larger than --max-mb; LOW = generic 'token = "..."' assignment worth a look. Standard library only."""
from __future__ import annotations

import argparse
import base64
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERNS = {                                                     # name -> (regex, severity)
    "hf_token": (r"hf_[A-Za-z0-9]{30,}", "HIGH"),
    "kaggle_access_token": (r"KGAT_[A-Za-z0-9]{10,}", "HIGH"),
    "kaggle_json_key": (r'"key"\s*:\s*"[0-9a-f]{32}"', "HIGH"),
    "kaggle_key_assignment": (r"KAGGLE_KEY\s*[=:]\s*['\"]?[0-9a-f]{20,}", "HIGH"),
    "github_token": (r"\bgh[pousr]_[A-Za-z0-9]{36,}\b", "HIGH"),
    "github_pat": (r"github_pat_[A-Za-z0-9_]{40,}", "HIGH"),
    "aws_access_key": (r"\bAKIA[0-9A-Z]{16}\b", "HIGH"),
    "private_key_block": (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "HIGH"),
    "slack_token": (r"\bxox[abprs]-[A-Za-z0-9-]{10,}", "HIGH"),
    "sk_style_key": (r"\bsk-[A-Za-z0-9_-]{32,}\b", "HIGH"),
    "google_api_key": (r"\bAIza[0-9A-Za-z_-]{35}\b", "HIGH"),
    "token_assignment": (r"(?i)\b(?:token|secret|password|passwd|api_key|apikey)\s*[=:]\s*['\"][A-Za-z0-9_\-]{20,}['\"]", "LOW"),
    "credential_store": (r"(?i)(upar_accounts|flag_secrets\.env|\.config[/\\]overleaf)", "HIGH"),      # the team's own secret stores
    "credential_location": (r"(?i)(\.kaggle[/\\]|kaggle\.json)", "MED"),                          # usually an instruction: check no key follows
}
KNOWN_SHA = {                                                    # sha256 -> what (never in the public repository)
    "b4b216c2eaaf7cbfb3441a1d5800ff1ccc538ea3d7ad8a51496aafa19aa0657d": "dev score zip Experiment/CLEAN-EN-01/submission_clean_en01_dev.zip",
    "fd7991a3353b9f90b2120b8d79fdf50ebaef6438c2e128be162ffbbf6d29c030": "organisers' meta_file_train_set.csv",
    "1803739f059860d5aede064839a17b01c6297f3160ab80cc9d3c929b27ae2ad9": "organisers' train_set.zip",
    "f29cc4139bc3e40bf9e9cd6533b6822e41296de972589d85ac71953cd1a92983": "organisers' dev_set.zip",
    "c9efa2100dc54188fbd9739970a98c1cd966985d9c6f7d46148e5a3af96fc9f9": "organisers' evaluation_phase.zip",
    "a70762a3fb52af6bbf52c14a86c79a4d074ec319b78d6ffedcdf195d71f9381e": "eval_set.zip (organisers' Evaluation data)",
}
DATA_MEMBER = re.compile(r"(?i)(meta_file_train_set|(^|/)gender\.csv$|(^|[/_])dev(_set)?[/._]|submission_[^/]*_dev\b|_dev\.zip$|\.wav$|\.mp4$|\.m4a$|\.jpg$)")
# arrays / weights / tables embedded in code: MED (data belongs in FF_DATA), HIGH when the member name points at VoxBlink
ARRAY_MEMBER = re.compile(r"(?i)\.(npy|npz|pt|pth|ckpt|parquet|pkl|joblib|safetensors|onnx|h5)$")
VB_NAME = re.compile(r"(?i)(^|[/_.-])(vb|voxblink)([/_.0-9-]|$)")
PRIVATE_DIR = re.compile(r"(?i)(^|/)_PRIVATE[^/]*(/|$)")
DATA_EXT = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin", ".wav", ".mp4", ".m4a", ".flac", ".zip",
            ".tar", ".gz", ".7z", ".h5", ".pkl", ".joblib"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXT = {".py", ".ipy", ".ipynb", ".json", ".md", ".txt", ".sh", ".yaml", ".yml", ".cfg", ".toml", ".csv", ".tsv", ".tex", ".html", ".js",
            ".css", ".ini", ".log", ".jsonl", "", ".gitignore", ".dockerfile"}
B64 = re.compile(r"[\"']([A-Za-z0-9+/=\\n\s]{200,})[\"']")
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def mask(s):
    s = s.strip()
    return (s[:6] + "..." + s[-2:]) if len(s) > 12 else "***"


class Scanner:
    def __init__(self, max_mb=50, excludes=()):
        self.findings = []
        self.max_mb = max_mb
        self.excludes = list(excludes)
        self.payloads = 0

    def add(self, sev, kind, where, detail):
        self.findings.append({"severity": sev, "kind": kind, "where": where, "detail": detail})

    def scan_text(self, text, where, depth=0):
        if depth == 0 and where.split(":")[0].endswith("secret_scan.py"):
            return                                               # the scanner's own pattern definitions
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > 20000:                                # base64 lines: patterns can still hide inside, keep scanning
                pass
            for name, (rx, sev) in PATTERNS.items():
                for m in re.finditer(rx, line):
                    self.add(sev, name, f"{where}:{i}", f"masked {mask(m.group(0))}")
        if depth > 3:
            return
        for m in B64.finditer(text):
            blob = re.sub(r"\\n|\s", "", m.group(1))
            if len(blob) < 200 or len(blob) % 4 not in (0,):
                continue
            try:
                raw = base64.b64decode(blob, validate=True)
            except Exception:
                continue
            self.payloads += 1
            self.scan_bytes(raw, f"{where} [base64 @{m.start()}]", depth + 1)

    def scan_bytes(self, raw, where, depth):
        h = hashlib.sha256(raw).hexdigest()
        if h in KNOWN_SHA:
            self.add("HIGH", "known_data_file", where, KNOWN_SHA[h])
        if raw[:2] == b"\x1f\x8b":
            try:
                raw2 = gzip.decompress(raw)
                h2 = hashlib.sha256(raw2).hexdigest()
                if h2 in KNOWN_SHA:
                    self.add("HIGH", "known_data_file", where + " [gzip]", KNOWN_SHA[h2])
                self.scan_bytes(raw2, where + " [gzip]", depth + 1)
            except Exception:
                pass
            return
        if raw[:4] == b"PK\x03\x04":
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for info in z.infolist():
                        if info.is_dir():
                            continue
                        b = z.read(info)
                        hm = hashlib.sha256(b).hexdigest()
                        w = f"{where} [zip member {info.filename}]"
                        if hm in KNOWN_SHA:
                            self.add("HIGH", "known_data_file", w, KNOWN_SHA[hm])
                        if DATA_MEMBER.search(info.filename):
                            self.add("HIGH", "data_like_member", w, f"{info.file_size} bytes (organisers' / dev / media data embedded?)")
                        if ARRAY_MEMBER.search(info.filename):
                            vb = bool(VB_NAME.search(info.filename))
                            self.add("HIGH" if vb else "MED", "embedded_array_member", w,
                                     f"{info.file_size} bytes ({'VoxBlink-named ' if vb else ''}arrays / weights embedded in code: data belongs in FF_DATA)")
                        self.scan_bytes(b, w, depth + 1)
            except Exception:
                pass
            return
        try:
            t = raw.decode("utf-8")
        except UnicodeDecodeError:
            return
        self.scan_text(t, where, depth)

    def scan_notebook(self, p, rel):
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            self.scan_text(p.read_text(encoding="utf-8", errors="replace"), rel)
            return
        for k, c in enumerate(nb.get("cells", [])):
            src = c.get("source", "")
            self.scan_text("".join(src) if isinstance(src, list) else src, f"{rel} cell {k}")
            for o in c.get("outputs", []) or []:
                txt = o.get("text") or ""
                txt = "".join(txt) if isinstance(txt, list) else txt
                for v in (o.get("data") or {}).values():
                    txt += "\n" + ("".join(v) if isinstance(v, list) else str(v))
                if txt.strip():
                    self.scan_text(txt, f"{rel} cell {k} output")
        meta = json.dumps(nb.get("metadata", {}))
        self.scan_text(meta, f"{rel} metadata")

    def scan_path(self, root):
        root = Path(root)
        files = [root] if root.is_file() else []
        if root.is_dir():
            for d, dirs, fs in os.walk(root):
                dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
                files += [Path(d) / f for f in sorted(fs)]
        priv = set()
        for p in files:
            rel = p.relative_to(root).as_posix() if root.is_dir() else p.name
            if any(fnmatch.fnmatch(rel, g) for g in self.excludes):
                continue
            m = PRIVATE_DIR.search(rel)
            if m:
                top = rel[:m.end()].rstrip("/")
                if top not in priv:
                    priv.add(top)
                    self.add("HIGH", "private_folder", top, "a _PRIVATE* folder inside the scanned tree: never publish it (exclude it explicitly with --exclude)")
            ext = p.suffix.lower()
            size = p.stat().st_size
            if size > self.max_mb * 2**20:
                self.add("MED", "large_file", rel, f"{size / 2**20:.1f} MB")
            if ext in DATA_EXT:
                h = None
                if size < 2**31:
                    h = hashlib.sha256(p.read_bytes()).hexdigest() if size < 512 * 2**20 else None
                if h in KNOWN_SHA:
                    self.add("HIGH", "known_data_file", rel, KNOWN_SHA[h])
                self.add("MED", "data_like_file", rel, f"{ext} {size} bytes (data / weights / archives belong in FF_DATA, not in the repository)")
                if ext == ".zip" and size < 256 * 2**20:
                    self.scan_bytes(p.read_bytes(), rel, 1)
                continue
            if ext in IMG_EXT:
                if size > 2 * 2**20:
                    self.add("MED", "large_image", rel, f"{size / 2**20:.1f} MB")
                continue
            if ext == ".ipynb":
                self.scan_notebook(p, rel)
            elif ext in TEXT_EXT or size < 2 * 2**20:
                try:
                    t = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                h = hashlib.sha256(t.encode("utf-8")).hexdigest()
                if h in KNOWN_SHA:
                    self.add("HIGH", "known_data_file", rel, KNOWN_SHA[h])
                self.scan_text(t, rel)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[str(REPO)])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--exclude", action="append", default=[], help="glob on the path relative to the scanned root (repeatable)")
    ap.add_argument("--max-mb", type=float, default=50)
    a = ap.parse_args(argv)
    S = Scanner(a.max_mb, a.exclude)
    for p in a.paths:
        S.scan_path(p)
    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    S.findings.sort(key=lambda f: (order[f["severity"]], f["where"]))
    if a.json:
        print(json.dumps({"findings": S.findings, "decoded_payloads": S.payloads}, indent=1))
    else:
        for f in S.findings:
            print(f"{f['severity']:4s} {f['kind']:22s} {f['where']}  {f['detail']}")
        n = {s: sum(f["severity"] == s for f in S.findings) for s in order}
        print(f"\n{n['HIGH']} HIGH, {n['MED']} MED, {n['LOW']} LOW finding(s); {S.payloads} embedded payload(s) decoded")
    return 1 if any(f["severity"] == "HIGH" for f in S.findings) else 0


if __name__ == "__main__":
    sys.exit(main())
