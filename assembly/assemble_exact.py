"""Mode (c) of the FourFusion 13.54 reproduction: rebuild the submitted Evaluation zip xjf_r10_trsig.zip (CodaBench submission 1054704,
Evaluation 13.54) from the STORED per-pair scores of the original runs. CPU, numpy / pandas / scipy / pyarrow, a few seconds.

    python assembly/assemble_exact.py [--inputs DIR] [--out DIR] [--no-check-inputs]

  --inputs  folder with the ten stored files (default: <each FF_DATA entry>/05_scores, then assembly/inputs); a file may also sit in a
            sub-folder of DIR, as long as its name is unique there
  --out     new, empty work folder (default: $FF_WORK/assembly_<time> or ./ff_work/assembly_<time>); nothing is ever overwritten

The ten files and their sha256 are listed in assembly/expected.json ("inputs"); each was made by a stage of the chain:
  f1s_eval.zip, components_eval.npz, e2e_manifest.json, cleantune01_fuser_bangla_CA.json, ct01r2_final_eval.npz   (F1s' Evaluation path:
  e50cc, e112f1sx); rfx_eval.zip (e163i, RFX c5rl with the ORIGINAL ImageBind-audio LoRA C1); rfsfxfvb_eval.zip (e163i_fvb, RFX'
  = RFX + FVBIND voice block B2, original LoRA C1); xjf_scores.npz (dp_train: DP branch scores); xjfdump_v4.parquet (t219_dumpv4:
  held-out-train scores for the scale constants); consts.json (dp_train, cross-check only).

Steps (each checked against assembly/expected.json):
  1. rebuild_final.py (verbatim copy of release/factor_enfuse/rebuild_final.py = rebuild/rebuild_final.py of the team package)
     on the F1s' files + rfx_eval.zip                     -> posteval_FACTOR_ENFUSE_c5rl.zip   (all four members must be identical)
  2. the same with rfx_eval.zip := rfsfxfvb_eval.zip      -> posteval_FVBIND_ENFUSE_fvb.zip    (its gender/Bangla member must be
     identical; it is the only member that enters the final zip; the other three are compared and reported)
  3. stages/70_dp/dp_sigma/dp_sigma.py on xjfdump_v4.parquet -> SIG = "5.0983,3.8376"        (must be equal)
  4. stages/80_final/final_blend/blend.py 1.0 with XJF_SIG = SIG -> zips/xjf_r10_trsig.zip  (the four members must be identical)
  5. information only: the four members re-written with the submitted zip's ZipInfo fields (time 2026-11-16 11:08:08, made on
     Windows, deflate) -> zips/xjf_r10_trsig_fixed_zipinfo.zip; its whole-file sha256 equals b4354eaf... only when the deflate
     library gives the same bytes (the original was written by Python 3.14 on Windows, zlib-ng 2.2.4)
Exit code 0 when steps 1-4 pass. A report is written to <out>/assembly_report.json."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXP = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
REBUILD = HERE / "rebuild_final.py"
SIGMA = REPO / "stages" / "70_dp" / "dp_sigma" / "dp_sigma.py"
BLEND = REPO / "stages" / "80_final" / "final_blend" / "blend.py"
F1S_FILES = ["f1s_eval.zip", "components_eval.npz", "e2e_manifest.json", "cleantune01_fuser_bangla_CA.json", "ct01r2_final_eval.npz"]


def sha256_file(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def members(zp):
    with zipfile.ZipFile(zp) as z:
        order = [i.filename for i in z.infolist()]
        return {n: z.read(n) for n in order}, order


def npz_arrays_sha256(p):
    import numpy as np
    X = np.load(p)
    return {k: hashlib.sha256(np.ascontiguousarray(X[k]).tobytes()).hexdigest() for k in X.files}


def locate(root, name):
    """root/name, else the only file of that name below root (None when absent or ambiguous; ambiguity is printed)."""
    p = root / name
    if p.is_file():
        return p
    hits = sorted(q for q in root.rglob(name) if q.is_file())
    if len(hits) > 1:
        print(f"  {name}: {len(hits)} candidates under {root} ({[str(q) for q in hits[:4]]}); put the right one at {root / name}", flush=True)
    return hits[0] if len(hits) == 1 else None


def find_inputs(cli):
    names = list(EXP["inputs"])
    need = [n for n in names if not EXP["inputs"][n].get("optional")]
    if cli:
        roots = [Path(cli)]
    else:
        roots = [Path(r) / "05_scores" for r in os.environ.get("FF_DATA", "").split(os.pathsep) if r] + [HERE / "inputs"]
    tried = []
    for r in roots:
        if not r.is_dir():
            tried.append(f"{r} (no folder)"); continue
        got = {n: locate(r, n) for n in names}
        miss = [n for n in need if got[n] is None]
        if not miss:
            return r, {n: p for n, p in got.items() if p is not None}
        tried.append(f"{r} (missing {miss})")
    sys.exit("input files not found; tried:\n  " + "\n  ".join(tried) + "\nSee assembly/inputs/README.md for where they come from.")


def check_inputs(inp, strict):
    rows, ok = {}, True
    for n, spec in EXP["inputs"].items():
        p = inp.get(n)
        if p is None:
            rows[n] = {"present": False}; continue
        h = sha256_file(p)
        if "sha256" in spec:
            good = h == spec["sha256"]
        else:                                                       # xjf_scores.npz: English-only file or the one with Bangla keys too
            good = h in spec["sha256_any_of"]
            if not good:
                arr = npz_arrays_sha256(p)
                good = all(arr.get(k) == v for k, v in spec["arrays_sha256"].items())
        rows[n] = {"path": str(p), "sha256": h, "ok": bool(good)}
        print(f"  input {n:34s} {'OK ' if good else 'BAD'} sha256 {h[:16]}...", flush=True)
        ok &= bool(good) or bool(spec.get("optional"))
    if strict and not ok:
        sys.exit("input check failed (a stored file is not the one of the submission); use --no-check-inputs to go on anyway")
    return rows, ok


def compare(zp, expected, label):
    mem, order = members(zp)
    res = {}
    for n, h in expected.items():
        got = hashlib.sha256(mem[n]).hexdigest() if n in mem else None
        res[n] = {"sha256": got, "expected": h, "identical": got == h}
        print(f"  {label} {n:45s} {'identical' if got == h else 'DIFFERENT'} ({(got or 'absent')[:16]}...)", flush=True)
    return res, order


def run(cmd, env=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    env = dict(env or os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    print("    " + out.strip().replace("\n", "\n    "), flush=True)
    return r.returncode, out


def rebuild(out, tag, inp, rfx_src, zip_name):
    d = out / f"{tag}_in"
    d.mkdir()
    for n in F1S_FILES:
        shutil.copyfile(inp[n], d / n)
    shutil.copyfile(rfx_src, d / "rfx_eval.zip")                    # rebuild_final.py reads IN/rfx_eval.zip
    z = out / zip_name
    code, _ = run([sys.executable, REBUILD, d, z, out / "_no_reference_zip_here.zip"])   # no reference: compared below by sha256
    assert code == 0 and z.is_file(), f"rebuild_final.py failed for {tag}"
    return z


def main():
    ap = argparse.ArgumentParser(description="rebuild xjf_r10_trsig.zip (FourFusion 13.54) byte for byte from stored per-pair scores")
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-check-inputs", action="store_true")
    a = ap.parse_args()
    import zlib
    root, inp = find_inputs(a.inputs)
    out = Path(a.out) if a.out else Path(os.environ.get("FF_WORK") or "ff_work") / time.strftime("assembly_%Y%m%d_%H%M%S")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: give a new --out (zips are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    R = {"inputs_root": str(root), "out": str(out), "python": sys.version.split()[0],
         "zlib": {"version": zlib.ZLIB_RUNTIME_VERSION, "zlib_ng": getattr(zlib, "ZLIBNG_VERSION", None)}}
    print(f"inputs: {root}\nwork:   {out}", flush=True)

    print("[0] stored inputs", flush=True)
    R["inputs"], R["inputs_ok"] = check_inputs(inp, strict=not a.no_check_inputs)

    print("[1] FACTOR_ENFUSE (F1s' + RFX; English through the F1s' Bangla fuser)", flush=True)
    fz = rebuild(out, "factor", inp, inp["rfx_eval.zip"], "posteval_FACTOR_ENFUSE_c5rl.zip")
    R["factor_enfuse"], _ = compare(fz, EXP["factor_enfuse"]["members"], "FACTOR_ENFUSE")
    ok1 = all(v["identical"] for v in R["factor_enfuse"].values())

    print("[2] FVBIND_ENFUSE (F1s' + RFX'; only gender/Bangla is used)", flush=True)
    vz = rebuild(out, "fvbind", inp, inp["rfsfxfvb_eval.zip"], "posteval_FVBIND_ENFUSE_fvb.zip")
    R["fvbind_enfuse"], _ = compare(vz, EXP["fvbind_enfuse"]["members"], "FVBIND_ENFUSE")
    ok2 = all(R["fvbind_enfuse"][n]["identical"] for n in EXP["fvbind_enfuse"]["required_members"])
    if ok2 and not all(v["identical"] for v in R["fvbind_enfuse"].values()):
        print("  note: a member of FVBIND_ENFUSE that does not enter the final zip differs (DESIGN N5); the final zip is unaffected", flush=True)

    print("[3] scale constants SIG (dp_sigma)", flush=True)
    code, _ = run([sys.executable, SIGMA, "--dump", inp["xjfdump_v4.parquet"], "--out", out / "sigma.json", "--expect", EXP["sigma"]["xjf_sig"], "--strict"])
    sig = json.loads((out / "sigma.json").read_text(encoding="utf-8")) if (out / "sigma.json").is_file() else {}
    R["sigma"] = {k: sig.get(k) for k in ("xjf_sig", "full", "per_replicate", "matches_expected")}
    ok3 = code == 0 and sig.get("xjf_sig") == EXP["sigma"]["xjf_sig"]

    print("[4] final blend (blend.py, r = 1.0)", flush=True)
    env = dict(os.environ, DP_POSTEVAL_DIR=str(out), DP_FVBIND_DIR=str(out), DP_SCORES=str(inp["xjf_scores.npz"]),
               DP_ZIP_DIR=str(out / "zips"), XJF_SIG=EXP["sigma"]["xjf_sig"], PYTHONIOENCODING="utf-8")
    env.pop("XJF_BASE", None); env.pop("DP_SIGMA_JSON", None)
    code, _ = run([sys.executable, BLEND, str(EXP["r"])], env=env)
    final = out / "zips" / EXP["target"]["zip"]
    assert code == 0 and final.is_file(), "blend.py failed"
    R["final"], order = compare(final, EXP["target"]["members"], "FINAL")
    ok4 = all(v["identical"] for v in R["final"].values())
    R["final_zip_sha256"] = sha256_file(final)

    print("[5] information: whole-zip sha256 with the submitted ZipInfo fields", flush=True)
    zi_spec = EXP["target"]["zipinfo"]
    mem, _ = members(final)
    fx = out / "zips" / "xjf_r10_trsig_fixed_zipinfo.zip"
    with zipfile.ZipFile(fx, "w", zipfile.ZIP_DEFLATED) as w:
        for n in zi_spec["order"]:
            zi = zipfile.ZipInfo(n, date_time=tuple(zi_spec["date_time"]))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = zi_spec["create_system"]
            zi.external_attr = zi_spec["external_attr"]
            w.writestr(zi, mem[n])
    R["fixed_zipinfo_zip_sha256"] = sha256_file(fx)
    R["fixed_zipinfo_zip_identical"] = R["fixed_zipinfo_zip_sha256"] == EXP["target"]["sha256"]
    print(f"  {fx.name}: sha256 {R['fixed_zipinfo_zip_sha256']} "
          f"({'IDENTICAL to the submitted zip' if R['fixed_zipinfo_zip_identical'] else 'differs from ' + EXP['target']['sha256'][:16] + '... (deflate library)'}"
          f"; zlib {R['zlib']})", flush=True)

    R["steps_ok"] = {"1_factor_enfuse": ok1, "2_fvbind_gender_bangla": ok2, "3_sigma": ok3, "4_final_members": ok4}
    (out / "assembly_report.json").write_text(json.dumps(R, indent=1) + "\n", encoding="utf-8")
    good = ok1 and ok2 and ok3 and ok4
    print(("OK: the four members of " if good else "FAILED: not all members of ") + f"{final} equal the submission "
          f"(sha256 {EXP['target']['sha256'][:16]}..., CodaBench {EXP['target']['codabench_submission']}); report {out / 'assembly_report.json'}",
          flush=True)
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
