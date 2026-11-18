"""Build repo/src/ (team side, run once): byte-for-byte copies of the team repository's builders, notebook modules,
local scripts and chain scripts at their original relative paths, the reduced Kaggle push log, and SRC_MANIFEST.json.

    python src/_tools/build_src_copy.py --source-root <team repository root> [--dest <repo>/src] [--stages <repo>/stages] [--dry-run]

Read-only on the source tree and on stages/. Writes only under --dest (default: the parent of this _tools/ folder).
The file list, stage usage, cited executed versions and notes come from src_curation.json (hand-curated; see its
"about" field). After copying, scan_stage_code.py compares every executed copy kept under stages/ (decoded / embedded
payloads, base64 / gzip / zlib / zip blobs and pasted cells of the executed notebooks) with the copies; its result
(_tools/stage_code_scan.json) is merged into files[].executed_versions ("origin": "stage-scan", full sha256, in-repo
location, diff summary) and files[].other_copies_in_repo.
Every copied file is re-read after writing and its sha256 compared with the source; a mismatch stops the run.
A file that matches a secret pattern is not copied and stops the run (nothing is redacted silently). Generated text
(the reduced log, the scan, the manifest) must not contain a local user-home path either.
Standard library only (Python >= 3.8)."""
import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True   # importing scan_stage_code must not leave __pycache__ under src/
sys.path.insert(0, str(HERE))
import scan_stage_code as SCAN  # noqa: E402

SECRET = {   # DESIGN 2.8 patterns (+ a few common token shapes and the credential folder); any hit stops the copy
    "hf_token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "kaggle_key_var": re.compile(rb"KAGGLE" rb"_KEY"),   # split so that secret scanners do not flag this line
    "json_key": re.compile(rb'"key"\s*:'),
    "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    "aws_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "kaggle_token": re.compile(rb"KGAT_[A-Za-z0-9]{10,}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "kaggle_credential_dir": re.compile(rb"\.kaggle[/\\]"),
}
# a local user-home path: allowed inside verbatim copies (logic fidelity; listed in the manifest), never in generated text
HOME_PATH = re.compile(rb"[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s\"']+")


def digest(b):
    lf = b.replace(b"\r\n", b"\n")
    n_crlf, n_lf = b.count(b"\r\n"), b.count(b"\n")
    eol = "none" if n_lf == 0 else ("CRLF" if n_crlf == n_lf else ("LF" if n_crlf == 0 else "mixed"))
    d = {"bytes": len(b), "lines": n_lf, "eol": eol, "sha256": hashlib.sha256(b).hexdigest(), "md5": hashlib.md5(b).hexdigest()}
    if eol in ("CRLF", "mixed"):
        d["sha256_lf"] = hashlib.sha256(lf).hexdigest()
    return d


def git(root, *args):
    r = subprocess.run(["git", "-C", str(root), "-c", "core.quotepath=off", *args], capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else ""


def git_info(root, paths):
    tracked = set(git(root, "ls-files", "--", *paths).splitlines())
    status = {}
    for line in git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths).splitlines():
        status[line[3:]] = line[:2]
    out = {}
    for p in paths:
        code = status.get(p, "  " if p in tracked else "??")
        d = {"tracked": p in tracked, "status": {"  ": "clean", " M": "modified", "M ": "modified", "MM": "modified", "??": "untracked"}.get(code, code.strip())}
        if p in tracked:
            d["last_commit"] = git(root, "log", "-1", "--format=%h %ad", "--date=short", "--", p).strip()
            if d["status"] == "modified":
                ns = git(root, "diff", "--numstat", "--", p).split()
                if len(ns) >= 2:
                    d["diff_vs_head"] = f"+{ns[0]} -{ns[1]}"
        out[p] = d
    head = git(root, "rev-parse", "HEAD").strip()
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    return out, head, branch


def check_versions(entries, d):
    res = []
    for e in entries:
        e = dict(e)
        if "prefix" in e:
            key = {"md5": "md5", "sha256": "sha256", "sha256_lf": "sha256_lf"}[e["algo"]]
            val = d.get(key) or (d["sha256"] if key == "sha256_lf" else "")   # an LF-only file: sha256_lf == sha256
            e["matches_this_copy"] = val.startswith(e["prefix"].lower())
            if "lf_prefix" in e:
                e["lf_matches_this_copy"] = (d.get("sha256_lf") or d["sha256"]).startswith(e["lf_prefix"].lower())
        else:
            e["matches_this_copy"] = None
        e.setdefault("origin", "citation" if "prefix" in e else "claim")
        res.append(e)
    return res


def rewrite_path(s, rules):
    if re.match(r"^[A-Za-z]:\\", s):          # Windows path written with backslashes: compare with forward slashes
        s = s.replace("\\", "/")
    for rx, rep in rules:
        s2 = re.sub(rx, rep, s)
        if s2 != s:
            return s2
    return s


def reduce_kapi(src_bytes, spec):
    """One JSON line per kept record of the team's push log. Kept: records whose id (spec['record_filter']['id_fields']:
    kernel, dataset, privatized) matches one of spec['record_filter']['keep_if_id_matches'] (FLAG 2027 runs, uploads and
    privatize actions); dropped: other projects' records and records without any id. Paths in 'source' / 'files' are
    rewritten by spec['path_rewrite'] (repo-relative, or a neutral placeholder for local folders)."""
    keep, drop = spec["keep"], set(spec["drop"])
    filt = [re.compile(x) for x in spec["record_filter"]["keep_if_id_matches"]]
    rules = [(re.compile(a), b) for a, b in spec["path_rewrite"]]
    out_lines, n_in, unknown, dropped = [], 0, set(), {"no_id": 0, "id_not_flag2027": 0}
    for i, raw in enumerate(src_bytes.decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        n_in += 1
        rec = json.loads(raw)
        unknown |= set(rec) - set(keep) - drop
        ids = [rec.get(k) for k in spec["record_filter"]["id_fields"] if rec.get(k)]
        if not ids:
            dropped["no_id"] += 1
            continue
        if not any(rx.search(x) for rx in filt for x in ids):
            dropped["id_not_flag2027"] += 1
            continue
        red = {"line": i}
        red.update({k: rec[k] for k in keep if k in rec})
        if isinstance(red.get("source"), str):
            red["source"] = rewrite_path(red["source"], rules)
        if isinstance(red.get("files"), list):
            red["files"] = [rewrite_path(x, rules) if isinstance(x, str) else x for x in red["files"]]
        out_lines.append(json.dumps(red, ensure_ascii=False))
    if unknown:
        raise SystemExit(f"kapi_runs.jsonl has fields that are neither kept nor dropped: {sorted(unknown)}")
    return ("\n".join(out_lines) + "\n").encode("utf-8"), len(out_lines), n_in, dropped


def where(r):
    s = r["container"]
    if r.get("cell") is not None:
        s += f" cell {r['cell']}"
    if r.get("member"):
        s += f" [{r['member']}]"
    return s


EXECUTED_EXACTNESS_EXCLUDED = {"builder-regenerated", "new", None}


def merge_scan(path, d, curated, recs, scan_notes):
    """executed_versions = stage-scan groups (executed copies under stages/<g>/<id>/original/, one entry per sha256)
    + curated entries; a curated hash that equals a scan group is folded into it (its citation kept)."""
    groups, other = {}, []
    for r in recs:
        executed = r["location"] == "original" and r["stage_exactness"] not in EXECUTED_EXACTNESS_EXCLUDED
        if not executed:
            o = {"stage": r["stage"], "location": r["location"] if r["location"] != "original" else "original (builder-regenerated, not executed)",
                 "in_repo": where(r), "kind": r["kind"], "sha256": r["sha256"], "match": r["match"]}
            if r.get("diff"):
                o["diff_hunks"] = r["diff"]["hunks"]
            other.append(o)
            continue
        g = groups.setdefault(r["sha256"], {"origin": "stage-scan", "stages": set(), "algo": "sha256", "value": r["sha256"],
                                            "md5": r["md5"], "match": r["match"], "matches_this_copy": r["match"] == "exact",
                                            "stage_exactness": set(), "kinds": set(), "in_repo": [], "diff": r.get("diff")})
        if r.get("sha256_lf"):
            g["sha256_lf"] = r["sha256_lf"]
        g["stages"].add(r["stage"])
        g["stage_exactness"].add(r["stage_exactness"])
        g["kinds"].add(r["kind"])
        if where(r) not in g["in_repo"]:
            g["in_repo"].append(where(r))
    out = []
    for sha, g in groups.items():
        g["stages"], g["stage_exactness"], g["kinds"] = sorted(g["stages"]), sorted(g["stage_exactness"]), sorted(g["kinds"])
        if g["match"] == "lf-equal":
            g["lf_matches_this_copy"] = True
        note = scan_notes.get(sha[:12])
        if note:
            g["note"] = note
        if not g["diff"]:
            del g["diff"]
        out.append(g)
    rest = []
    for c in curated:
        hit = None
        if "prefix" in c:
            p = c["prefix"].lower()
            for g in out:
                if (c["algo"] == "sha256" and g["value"].startswith(p)) or (c["algo"] == "md5" and g["md5"].startswith(p)) or \
                        (c["algo"] == "sha256_lf" and (g.get("sha256_lf") or g["value"]).startswith(p)):
                    hit = g
                    break
        if hit:
            cit = {k: v for k, v in c.items() if k not in ("matches_this_copy", "lf_matches_this_copy", "origin")}
            hit.setdefault("citations", []).append(cit)
            extra = sorted(set(c.get("stages", [])) - set(hit["stages"]))
            if extra:
                hit["stages_cited_but_not_in_repo_copies"] = sorted(set(hit.get("stages_cited_but_not_in_repo_copies", [])) | set(extra))
        else:
            rest.append(c)
    out.sort(key=lambda g: (g["stages"][0] if g["stages"] else "", g["value"]))
    return out + rest, sorted(other, key=lambda o: (o["stage"], o["in_repo"]))


def home_of(path, homes):
    best = None
    for pre, art, ff in homes:
        if path == pre or path.startswith(pre.rstrip("/") + "/"):
            if best is None or len(pre) > len(best[0]):
                best = (pre, art, ff)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-root", required=True, type=Path, help="root of the team repository (read-only)")
    ap.add_argument("--dest", type=Path, default=HERE.parent, help="the repo's src/ folder (default: parent of _tools/)")
    ap.add_argument("--stages", type=Path, default=None, help="the repo's stages/ folder (default: <dest>/../stages)")
    ap.add_argument("--dry-run", action="store_true", help="check and hash everything, write nothing")
    a = ap.parse_args()
    src, dest = a.source_root.resolve(), a.dest.resolve()
    repo = dest.parent
    stages_dir = (a.stages or repo / "stages").resolve()
    cur = json.loads((HERE / "src_curation.json").read_text(encoding="utf-8"))
    paths = list(cur["design_9_4"]) + [p for p in cur["added"] if p not in cur["design_9_4"]]
    assert len(paths) == len(set(paths)), "duplicate paths in src_curation.json"

    # 1. secret scan of every source file before anything is written
    hits = []
    for p in paths + [v["from"] for v in cur["derived"].values()]:
        b = (src / p).read_bytes()
        for name, rx in SECRET.items():
            if rx.search(b):
                hits.append((p, name))
    if hits:
        raise SystemExit(f"secret-like strings found, nothing copied: {hits}")

    # 2. byte-for-byte copies
    gi, head, branch = git_info(src, paths + [v["from"] for v in cur["derived"].values()])
    files, local_paths = [], []
    for p in paths:
        s, t = src / p, (dest / p).resolve()
        if dest not in t.parents:
            raise SystemExit(f"refusing to write outside {dest}: {t}")
        b = s.read_bytes()
        if not a.dry_run:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)
            if hashlib.sha256(t.read_bytes()).digest() != hashlib.sha256(b).digest():
                raise SystemExit(f"copy check failed: {p}")
        n_home = len(HOME_PATH.findall(b))
        if n_home:
            local_paths.append({"path": p, "occurrences": n_home})
        d = digest(b)
        e = {"path": p, **d, "source_mtime": dt.datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
             "git": gi[p], "in_design_9_4": p in cur["design_9_4"]}
        if p in cur["added"]:
            e["added_reason"] = cur["added"][p]
        e["role"] = cur["roles"][p]
        e["stages"] = cur["stages"].get(p, {})
        e["_curated_versions"] = check_versions(cur["executed_versions"].get(p, []), d)
        if p in cur["notes"]:
            e["notes"] = cur["notes"][p]
        files.append(e)

    # 3. derived files (the reduced Kaggle push log)
    derived = []
    for out_rel, spec in cur["derived"].items():
        sb = (src / spec["from"]).read_bytes()
        rb, n, n_in, dropped = reduce_kapi(sb, spec)
        bad = [name for name, rx in SECRET.items() if rx.search(rb)] + (["home_path"] if HOME_PATH.search(rb) else [])
        if bad:
            raise SystemExit(f"secret-like strings or local home paths in {out_rel}: {bad}")
        t = (dest / out_rel).resolve()
        if dest not in t.parents:
            raise SystemExit(f"refusing to write outside {dest}: {t}")
        if not a.dry_run:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(rb)
        derived.append({"path": out_rel, **digest(rb), "records": n, "records_in_source": n_in, "records_dropped": dropped,
                        "derived_from": {"path": spec["from"], **digest(sb), "git": gi[spec["from"]]},
                        "record_filter": spec["record_filter"], "path_rewrite": spec["path_rewrite"],
                        "kept_fields": spec["keep"], "dropped_fields": spec["drop"], "added_fields": spec["add"], "why": spec["why"],
                        "stages_note": spec["stages_note"]})

    # 4. scan of the executed copies under stages/ (needs the copies written above)
    scan_info = {"stages_dir": None}
    scan_recs = {}
    if stages_dir.is_dir() and not a.dry_run:
        res = SCAN.scan(repo)
        txt = json.dumps(res, indent=1, ensure_ascii=False) + "\n"
        if HOME_PATH.search(txt.encode("utf-8")) or any(rx.search(txt.encode("utf-8")) for rx in SECRET.values()):
            raise SystemExit("secret-like strings or local home paths in the stage scan")
        (HERE / "stage_code_scan.json").write_text(txt, encoding="utf-8")
        cur_sha = {f["path"]: f["sha256"] for f in files}
        stale = [r["src_path"] for r in res["records"] if cur_sha.get(r["src_path"]) != r["src_sha256"]]
        if stale:
            raise SystemExit(f"stage scan compared against other bytes than the copies: {sorted(set(stale))[:5]}")
        for r in res["records"]:
            scan_recs.setdefault(r["src_path"], []).append(r)
        scan_info = {"stages_dir": "stages/", "stages_scanned": res["stages_scanned"], "records": len(res["records"]),
                     "by_location_and_match": res["summary"], "unmapped_payloads": res["unmapped_payloads"],
                     "file": "_tools/stage_code_scan.json"}

    # 5. merge cited / claimed versions with the scan
    notes = cur.get("scan_notes", {})
    ignored = {(x["path"], x["stage"]) for x in cur.get("scan_pairs_ignored", [])}
    missing_pairs = []
    for f in files:
        ev, other = merge_scan(f["path"], f, f.pop("_curated_versions"), scan_recs.get(f["path"], []), notes.get(f["path"], {}))
        if ev:
            f["executed_versions"] = ev
        if other:
            f["other_copies_in_repo"] = other
        for g in ev:
            if g.get("origin") == "stage-scan":
                for s in g["stages"]:
                    if s not in f["stages"] and (f["path"], s) not in ignored:
                        missing_pairs.append({"path": f["path"], "stage": s})

    # 6. not_copied: FF_DATA home (configs/data_map.json code) of each reference
    homes = cur.get("not_copied_homes", [])
    not_copied = []
    for e in cur["not_copied"]:
        e = dict(e)
        h = home_of(e["path"], homes)
        if h:
            e["artifact"], e["ff_data"] = h[1], h[2]
        elif e["class"] in ("data", "bundle-or-score-zip", "learned-parameters", "data-table", "intermediate", "dev-data"):
            e.setdefault("ff_data", None)
            e.setdefault("home_note", "no FF_DATA home in configs/data_map.json")
        not_copied.append(e)

    # 7. executed notebooks whose CODE_MD5 dictionaries were read: their in-repo (redacted) copies
    in_repo_nb = {}
    for rl in scan_recs.values():
        for r in rl:
            if r.get("container_md5") and r["location"].startswith("original"):
                in_repo_nb.setdefault(r["stage"], {})[r["container"]] = r["container_md5"]
    scanned_nb = []
    for e in cur.get("executed_notebooks_scanned", []):
        e = dict(e)
        e["in_repo"] = [{"path": k, "md5": v} for k, v in sorted(in_repo_nb.get(e["stage"], {}).items())]
        scanned_nb.append(e)

    # 8. tools / docs of src/ itself (everything under dest that is not a copy, a derived file or the manifest)
    listed = {f["path"] for f in files} | {d["path"] for d in derived} | {"SRC_MANIFEST.json"}
    tools = []
    if dest.exists():
        for f in sorted(dest.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                rel = f.relative_to(dest).as_posix()
                if rel not in listed:
                    if not (rel.startswith("_tools/") or rel in ("README.md", ".gitattributes")):
                        raise SystemExit(f"stray file under {dest} (not in the copy list, not a tool / doc): {rel}")
                    b = f.read_bytes()
                    bad = [n for n, rx in SECRET.items() if rx.search(b)] + (["home_path"] if HOME_PATH.search(b) else [])
                    if bad:
                        raise SystemExit(f"secret-like strings or local home paths in {rel}: {bad}")
                    tools.append({"path": rel, **digest(b)})

    by_stage = {}
    for f in files:
        for s, u in f["stages"].items():
            by_stage.setdefault(s, []).append({"path": f["path"], "usage": u})
    evs = [v for f in files for v in f.get("executed_versions", [])]
    ev = [v for v in evs if v.get("matches_this_copy") is not None]
    summary = {
        "files_copied": len(files), "files_in_design_9_4": sum(f["in_design_9_4"] for f in files),
        "files_added_beyond_design": sum(not f["in_design_9_4"] for f in files),
        "derived_files": len(derived), "tool_and_doc_files": len(tools),
        "git_status": {k: sum(1 for f in files if f["git"]["status"] == k) for k in sorted({f["git"]["status"] for f in files})},
        "eol": {k: sum(1 for f in files if f["eol"] == k) for k in sorted({f["eol"] for f in files})},
        "roles": {k: sum(1 for f in files if f["role"] == k) for k in sorted({f["role"] for f in files})},
        "executed_versions_checked": len(ev), "executed_versions_matching_copy": sum(1 for v in ev if v["matches_this_copy"]),
        "executed_versions_from_stage_scan": sum(1 for v in evs if v.get("origin") == "stage-scan"),
        "executed_versions_from_stage_scan_differing": sum(1 for v in evs if v.get("origin") == "stage-scan" and v["match"] not in ("exact", "lf-equal")),
        "executed_versions_claims_without_hash": sum(1 for v in evs if v.get("matches_this_copy") is None),
        "files_with_a_differing_executed_version": sorted({f["path"] for f in files for v in f.get("executed_versions", [])
                                                           if v.get("matches_this_copy") is False and not v.get("lf_matches_this_copy")
                                                           and v.get("match") != "same-lines"}),
        "scan_pairs_not_in_stage_index": missing_pairs,
        "copies_with_local_home_paths": local_paths,
        "executed_notebooks_scanned": len(scanned_nb),
        "not_copied_references": len(not_copied), "stages_indexed": len(by_stage),
    }
    man = {
        "schema": "fourfusion-src-manifest/2",
        "created": dt.date.today().isoformat(),
        "about": ("Byte-for-byte copies of the team repository's builders, notebook modules, local scripts and chain scripts, at their "
                  "original relative paths. Read-only reference: no stage imports src/; every stage carries the code that actually ran "
                  "(stages/<group>/<id>/). files[].executed_versions lists the versions of each file found in executed copies: "
                  "'stage-scan' entries come from an automatic comparison with every executed copy kept under stages/<group>/<id>/original/ "
                  "(decoded / embedded payloads, base64 / gzip / zlib / zip blobs and pasted cells; full sha256, in-repo location, diff "
                  "summary); 'citation' entries are hashes read from build documents or CODE_MD5 dictionaries; 'claim' entries have no "
                  "hash. A stage that ran a file but left no executed copy in stages/ (e.g. a builder-regenerated notebook) has no "
                  "stage-scan entry, so a missing entry is not proof that this copy ran. See src/README.md."),
        "source": {"repository": "team repository (private)", "git_head": head, "branch": branch,
                   "git_note": "most files were never committed (git status 'untracked'); sha256 below are of the working-tree bytes at copy time"},
        "copy_rule": "bytes identical to the source file at copy time (re-read and sha256-checked); src/.gitattributes disables line-ending conversion",
        "usage_vocabulary": {
            "builds": "builder that generates the stage notebook (the stage may run an executed copy instead; then the builder is a cross-check)",
            "builder-import": "module imported (directly or transitively) by a builder of the stage",
            "builder-input": "file read by the current builder and embedded in the generated notebook (embedded does not mean executed)",
            "patches-input": "file read by a patch script at patch time and pasted into the stage notebook",
            "runs": "code executed in the stage (embedded in the executed notebook or run as a local script), per DESIGN / layer maps / stage scan",
            "patches": "script that patched the stage notebook before the push",
            "check": "reference output the stage must reproduce",
            "pushed-by": "chain script that pushed / orchestrated the stage's Kaggle run (provenance only; do not run)",
            "wiring": "produces data that only satisfy presence checks (not in the numbers)",
            "reference": "not executed; reference for stage.json / runner construction",
        },
        "executed_versions_fields": {
            "origin": "stage-scan | citation | claim",
            "value / md5 / sha256_lf": "hashes of the executed bytes (stage-scan)",
            "prefix / algo": "cited hash prefix and its algorithm (citation)",
            "match": "exact | lf-equal (equal after CRLF->LF) | same-lines (only the final newline differs) | differs",
            "in_repo": "where the executed bytes are in this repository (container [cell n] [member])",
            "diff": "executed -> this copy: hunks, lines_only_in_copy, lines_only_in_executed, hunks_ignoring_indent_and_blank_lines, first_hunks",
            "citations": "curated citations whose hash equals this scan entry (layer maps, CODE_MD5, build documents)",
        },
        "summary": summary,
        "stage_scan": scan_info,
        "files": files,
        "derived": derived,
        "tools_and_docs": tools,
        "by_stage": dict(sorted(by_stage.items())),
        "stage_notes": cur.get("stage_notes", {}),
        "not_copied": not_copied,
        "executed_notebooks_scanned": scanned_nb,
        "embedded_members_not_in_src": cur.get("embedded_members_not_in_src", []),
        "scan_pairs_ignored": cur.get("scan_pairs_ignored", []),
    }
    if a.dry_run:
        print(json.dumps(summary, indent=1))
        return
    txt = json.dumps(man, indent=1, ensure_ascii=False) + "\n"
    bad = [n for n, rx in SECRET.items() if rx.search(txt.encode("utf-8"))] + (["home_path"] if HOME_PATH.search(txt.encode("utf-8")) else [])
    if bad:
        raise SystemExit(f"secret-like strings or local home paths in SRC_MANIFEST.json: {bad}")
    (dest / "SRC_MANIFEST.json").write_text(txt, encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=1))
    for k in ("scan_pairs_not_in_stage_index", "copies_with_local_home_paths"):
        print(k, json.dumps(summary[k], ensure_ascii=False))


if __name__ == "__main__":
    main()
