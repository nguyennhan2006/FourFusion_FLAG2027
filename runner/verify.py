"""Check a submission zip (or any score zip) against the reference hashes, or compare two score zips.

    python runner/verify.py zip <zip> [--expected xjf_r10_trsig]          # SHA-256 of every member vs configs/expected_sha256.json
    python runner/verify.py compare <zip> <reference.zip> [--min-spearman 0.9999]
    python runner/verify.py file <path> (--sha256 HEX | --expected KEY[/SUBKEY])
    python runner/verify.py list                                          # keys of configs/expected_sha256.json

Score files are lines "<trial id> <value>" (the zip stores -score: lower = same person). `compare` aligns the two files of the
same member by trial id and reports the number of pairs, missing ids, Spearman and Pearson correlation and the largest absolute
difference. Standard library only (Spearman / Pearson implemented here), so it runs anywhere.
Exit code 0 when every check passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPECTED = REPO / "configs" / "expected_sha256.json"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def expected(key, path=EXPECTED):
    node = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in key.split("/"):
        node = node[k]
    return node


def zip_members(zp):
    with zipfile.ZipFile(zp) as z:
        return {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}


def check_members(zp, exp_members):
    """-> (all_ok, rows) comparing the SHA-256 of every member with exp_members {name: sha256}."""
    got = zip_members(zp)
    rows, ok = [], True
    for name, h in exp_members.items():
        if name not in got:
            rows.append(f"missing  {name}")
            ok = False
            continue
        g = sha256_bytes(got[name])
        same = g == h
        ok &= same
        rows.append(f"{'OK      ' if same else 'MISMATCH'} {name}  {g[:16]}... {'==' if same else '!='} {h[:16]}...")
    for name in sorted(set(got) - set(exp_members)):
        rows.append(f"extra    {name} (not in the reference)")
        ok = False
    return ok, rows


def parse_scores(b):
    ids, vals = [], []
    for line in b.decode("utf-8").split("\n"):
        if not line.strip():
            continue
        a = line.split()
        ids.append(a[0])
        vals.append(float(a[1]))
    return ids, vals


def _rank(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a, b):
    return pearson(_rank(a), _rank(b))


def compare_zips(za, zb):
    """-> list of {member, n, missing_in_a, missing_in_b, spearman, pearson, max_abs_diff, identical_bytes}."""
    A, B = zip_members(za), zip_members(zb)
    rows = []
    for name in sorted(set(A) | set(B)):
        if name not in A or name not in B:
            rows.append({"member": name, "n": 0, "missing_in_a": name not in A, "missing_in_b": name not in B,
                         "spearman": float("nan"), "pearson": float("nan"), "max_abs_diff": float("nan"), "identical_bytes": False})
            continue
        ia, va = parse_scores(A[name])
        ib, vb = parse_scores(B[name])
        mb = dict(zip(ib, vb))
        common = [i for i in ia if i in mb]
        x = [v for i, v in zip(ia, va) if i in mb]
        y = [mb[i] for i in common]
        rows.append({"member": name, "n": len(common), "missing_in_a": len(set(ib) - set(ia)), "missing_in_b": len(set(ia) - set(ib)),
                     "spearman": spearman(x, y), "pearson": pearson(x, y),
                     "max_abs_diff": max((abs(p - q) for p, q in zip(x, y)), default=float("nan")),
                     "identical_bytes": A[name] == B[name]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("zip")
    p.add_argument("zip")
    p.add_argument("--expected", default="xjf_r10_trsig", help="key in configs/expected_sha256.json (default xjf_r10_trsig)")
    p.add_argument("--expected-file", default=str(EXPECTED))
    p = sub.add_parser("compare")
    p.add_argument("zip")
    p.add_argument("reference")
    p.add_argument("--min-spearman", type=float, default=0.9999)
    p = sub.add_parser("file")
    p.add_argument("path")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sha256")
    g.add_argument("--expected")
    p.add_argument("--expected-file", default=str(EXPECTED))
    sub.add_parser("list")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        d = json.loads(EXPECTED.read_text(encoding="utf-8"))
        for k, v in d.items():
            if not k.startswith("_"):
                print(f"{k:16s} {v.get('what', '') if isinstance(v, dict) else ''}"[:160])
        return 0
    if a.cmd == "zip":
        node = expected(a.expected, a.expected_file)
        zh = sha256_file(a.zip)
        ok, rows = check_members(a.zip, node["members"])
        print(f"{a.zip}\n  zip sha256 {zh}" + (" (= reference)" if zh == node.get("zip") else f" (reference {node.get('zip', '?')[:16]}...: "
              "a rewritten zip differs by its header times; members decide)"))
        for r in rows:
            print("  " + r)
        if "zip_member_date_time" in node:
            with zipfile.ZipFile(a.zip) as z:
                dts = {tuple(i.date_time) for i in z.infolist()}
                order = [i.filename for i in z.infolist()]
            print(f"  member times {sorted(dts)} (reference {tuple(node['zip_member_date_time'])}); "
                  f"order {'as reference' if order == node.get('member_order', order) else 'differs: ' + str(order)}")
        print("ALL MEMBERS MATCH" if ok else "MEMBER MISMATCH")
        return 0 if ok else 1
    if a.cmd == "compare":
        rows = compare_zips(a.zip, a.reference)
        ok = True
        for r in rows:
            good = r["n"] > 0 and not r["missing_in_a"] and not r["missing_in_b"] and r["spearman"] >= a.min_spearman
            ok &= good
            print(f"  {'OK  ' if good else 'FAIL'} {r['member']:45s} n={r['n']:4d} missing a/b={r['missing_in_a']}/{r['missing_in_b']} "
                  f"Spearman {r['spearman']:.6f} Pearson {r['pearson']:.6f} max|diff| {r['max_abs_diff']:.6g}"
                  f"{'  (identical bytes)' if r['identical_bytes'] else ''}")
        print(f"{'PASS' if ok else 'FAIL'} (Spearman >= {a.min_spearman} for every member)")
        return 0 if ok else 1
    if a.cmd == "file":
        h = sha256_file(a.path)
        exp = a.sha256
        if a.expected:
            node = expected(a.expected, a.expected_file)
            exp = node.get("zip") if isinstance(node, dict) else node
        same = h == exp
        print(f"{'OK' if same else 'MISMATCH'} {a.path} {h} {'==' if same else '!='} {exp}")
        return 0 if same else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
