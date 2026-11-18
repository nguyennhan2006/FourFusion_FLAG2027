"""Decode the code that stage.ipy files embed (base64 -> gzip / zip -> text, recursively) and find literal /kaggle/input paths.

The runner rewrites the strings '/kaggle/input' and '/kaggle/working' in stage.ipy text only. Code that a stage writes to disk
from an embedded base64 payload (rfs_train.py, CLEANTUNE-01/run.py, T3ENC-01/run.py, build_l0_notebooks.py, ...) still holds the
literal paths. This module finds them so that the runner (runner/ffcore.py Runner.literal_info) and the checker
(runner/check_stage.py) can treat such a stage correctly:

  * kind 'env_default': os.environ.get("X", "/kaggle/input...") / os.getenv("X", "/kaggle/input...") -> the runner exports
    X=<the stage's input view> when the stage does not set X itself (the value the default would have had in the original run,
    where /kaggle/input held exactly the stage's inputs);
  * kind 'literal': any other string constant starting with /kaggle/input (rglob fallbacks, `or [Path("/kaggle/input")]`) ->
    the stage needs the literal path (runner_requirements.literal_kaggle_input; Linux: /kaggle/input is pointed at the stage's
    input view for the duration of the stage; Kaggle: a session holding exactly the stage's inputs).

Docstrings and comments are ignored (ast). Modules that do not parse as Python fall back to a line scan (comment lines skipped).
Standard library only."""
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import io
import re
import zipfile

B64 = re.compile(r"[\"']([A-Za-z0-9+/=\\n\s]{200,})[\"']")
LIT = "/kaggle/input"


def decoded_members(text, name="stage.ipy", depth=0, out=None):
    """-> list of (member name, bytes, text or None) for every payload embedded in `text` (the top level itself excluded)."""
    out = [] if out is None else out
    if depth > 5:
        return out
    for m in B64.finditer(text):
        blob = re.sub(r"\\n|\s", "", m.group(1))
        if len(blob) < 200 or len(blob) % 4:
            continue
        try:
            raw = base64.b64decode(blob, validate=True)
        except Exception:
            continue
        _bytes(raw, f"{name}|b64", depth + 1, out)
    return out


def _bytes(raw, name, depth, out):
    if depth > 5:
        return
    if raw[:2] == b"\x1f\x8b":
        try:
            _bytes(gzip.decompress(raw), name + "|gz", depth + 1, out)
        except Exception:
            pass
        return
    if raw[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for info in z.infolist():
                    if not info.is_dir():
                        _bytes(z.read(info), info.filename, depth + 1, out)
        except Exception:
            pass
        return
    try:
        t = raw.decode("utf-8")
    except UnicodeDecodeError:
        out.append((name, raw, None))
        return
    out.append((name, raw, t))
    decoded_members(t, name, depth, out)


def _is_environ(node):
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (isinstance(node, ast.Name) and node.id == "environ")


def literal_uses(text):
    """-> list of {"line", "kind": "literal" | "env_default", "env": X | None, "value"} for one decoded module."""
    try:
        tree = ast.parse(text)
    except Exception:
        tree = None
    uses = []
    if tree is None:
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if LIT in s and not s.startswith("#"):
                m = re.search(r"(?:environ\.get|getenv)\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*,\s*['\"]" + re.escape(LIT), s)
                uses.append({"line": i, "kind": "env_default" if m else "literal", "env": m.group(1) if m else None, "value": s[:120]})
        return uses
    doc = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            doc.add(id(body[0].value))
    handled = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        f = node.func
        is_get = isinstance(f, ast.Attribute) and f.attr == "get" and _is_environ(f.value)
        is_getenv = (isinstance(f, ast.Attribute) and f.attr == "getenv") or (isinstance(f, ast.Name) and f.id == "getenv")
        a0, a1 = node.args[0], node.args[1]
        if (is_get or is_getenv) and isinstance(a1, ast.Constant) and isinstance(a1.value, str) and a1.value.startswith(LIT):
            env = a0.value if isinstance(a0, ast.Constant) and isinstance(a0.value, str) else None
            uses.append({"line": a1.lineno, "kind": "env_default" if env else "literal", "env": env, "value": a1.value})
            handled.add(id(a1))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(LIT) and id(node) not in doc \
                and id(node) not in handled:
            uses.append({"line": node.lineno, "kind": "literal", "env": None, "value": node.value[:120]})
    return sorted(uses, key=lambda u: u["line"])


def scan_stage_code(text):
    """-> {"members": n, "uses": [{"member", "sha256", ...use}], "literal": bool, "env_defaults": sorted env names} for a stage.ipy."""
    uses, n = [], 0
    for name, raw, t in decoded_members(text):
        n += 1
        if t is None or LIT not in t:
            continue
        h = hashlib.sha256(raw).hexdigest()
        for u in literal_uses(t):
            uses.append({"member": name, "sha256": h, **u})
    return {"members": n, "uses": uses, "literal": any(u["kind"] == "literal" for u in uses),
            "env_defaults": sorted({u["env"] for u in uses if u["kind"] == "env_default" and u["env"]})}


def embedded_files(text, pattern=r"\.(json|csv|tsv|txt|npy|npz|pt|pth|parquet|pkl|safetensors|onnx)$"):
    """-> {member name: (sha256, bytes)} of data-like members embedded in a stage.ipy (requirements.txt excluded)."""
    rx = re.compile(pattern, re.I)
    out = {}
    for name, raw, _t in decoded_members(text):
        if rx.search(name) and not name.endswith("requirements.txt"):
            out[name] = (hashlib.sha256(raw).hexdigest(), len(raw))
    return out
