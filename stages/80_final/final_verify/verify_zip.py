"""Stage final_verify (new code, FourFusion 13.54 reproduction repo): compare a rebuilt submission zip with the submitted one.

    python verify_zip.py --zip xjf_r10_trsig.zip --expected <repo>/assembly/expected.json [--ref <submitted zip>] [--strict] [--out report.json]

Checks, per member (file inside the zip):
  * sha256 of the member bytes against expected.json["target"]["members"]          -> "byte-identical" or not
  * when --ref is given (the submitted xjf_r10_trsig.zip, sha256 b4354eaf...): same pair ids, Spearman, max / mean |difference|
The sha256 of the WHOLE zip is printed for information only: zipfile writes the current time into every member header, so a rebuilt zip
never has the submitted zip's sha256 unless it is written with the same ZipInfo fields and the same deflate library (assembly/README.md).
Byte-identical members are expected only in mode (c) (assemble from stored scores); modes (a) / (b) re-run GPU stages, which are not
bit-deterministic: there the criterion is Spearman >= 0.9999 per file (DESIGN 5.2), reported, not enforced unless --strict-spearman.
Exit code: 0, except 1 when --strict and a member is not byte-identical, or --strict-spearman and a file is below the threshold, or the
zip is malformed (member list, pair counts, non-finite values)."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def pairs(b):
    t = [l.split() for l in b.decode().splitlines() if l.strip()]
    assert all(len(x) == 2 for x in t), "every line must be '<pair id> <value>'"
    return [x[0] for x in t], np.array([float(x[1]) for x in t])


def main():
    ap = argparse.ArgumentParser(description="per-member sha256 / Spearman check of a rebuilt submission zip")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--expected", required=True, help="assembly/expected.json of the repo")
    ap.add_argument("--ref", default=None, help="the submitted zip (optional)")
    ap.add_argument("--strict", action="store_true", help="fail unless every member is byte-identical to the submission")
    ap.add_argument("--strict-spearman", type=float, default=None, help="fail when a file's Spearman vs --ref is below this value")
    ap.add_argument("--out", default=None, help="write the report as JSON")
    a = ap.parse_args()

    E = json.loads(Path(a.expected).read_text(encoding="utf-8"))["target"]
    exp_members, counts = E["members"], E["pairs"]
    zb = Path(a.zip).read_bytes()
    rep = {"zip": str(a.zip), "zip_sha256": sha256_bytes(zb), "zip_sha256_submitted": E["sha256"],
           "zip_identical": sha256_bytes(zb) == E["sha256"], "members": {}}
    ok_format, all_identical, spear_ok = True, True, True
    with zipfile.ZipFile(a.zip) as z:
        names = [i.filename for i in z.infolist()]
        if sorted(names) != sorted(exp_members):
            print(f"FORMAT: members {names} != expected {sorted(exp_members)}", flush=True); ok_format = False
        ref = zipfile.ZipFile(a.ref) if a.ref else None
        for n in names:
            b = z.read(n); h = sha256_bytes(b)
            ids, v = pairs(b)
            m = {"sha256": h, "expected_sha256": exp_members.get(n), "byte_identical": h == exp_members.get(n),
                 "pairs": len(ids), "expected_pairs": counts.get(n), "finite": bool(np.isfinite(v).all()), "unique_ids": len(set(ids)) == len(ids)}
            ok_format &= m["finite"] and m["unique_ids"] and m["pairs"] == m["expected_pairs"]
            all_identical &= m["byte_identical"]
            if ref is not None and n in ref.namelist():
                rids, rv = pairs(ref.read(n))
                if set(rids) == set(ids):
                    rmap = dict(zip(rids, rv)); r = np.array([rmap[p] for p in ids])
                    m["vs_ref"] = {"same_ids": True, "spearman": float(pd.Series(v).corr(pd.Series(r), method="spearman")),
                                   "max_abs_diff": float(np.abs(v - r).max()), "mean_abs_diff": float(np.abs(v - r).mean()),
                                   "ref_member_sha256": sha256_bytes(ref.read(n))}
                    if a.strict_spearman is not None and m["vs_ref"]["spearman"] < a.strict_spearman:
                        spear_ok = False
                else:
                    m["vs_ref"] = {"same_ids": False}; ok_format = False
            rep["members"][n] = m
            line = f"{n}: {m['pairs']} pairs, sha256 {h[:16]}... {'== submitted' if m['byte_identical'] else '!= submitted (' + str(m['expected_sha256'])[:16] + '...)'}"
            if "vs_ref" in m and m["vs_ref"].get("same_ids"):
                line += f" | vs ref: Spearman {m['vs_ref']['spearman']:.6f}, max|diff| {m['vs_ref']['max_abs_diff']:.2e}"
            print(line, flush=True)
    rep.update(format_ok=bool(ok_format), all_members_identical=bool(all_identical), spearman_ok=bool(spear_ok))
    print(f"whole zip sha256 {rep['zip_sha256']} ({'identical to' if rep['zip_identical'] else 'differs from'} the submitted "
          f"{E['sha256'][:16]}...; expected to differ unless written with the fixed ZipInfo fields, see assembly/README.md)", flush=True)
    print("RESULT:", "all four members byte-identical to the submission" if all_identical else "members differ from the submission",
          "| format OK" if ok_format else "| FORMAT PROBLEM", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    if not ok_format or (a.strict and not all_identical) or not spear_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
