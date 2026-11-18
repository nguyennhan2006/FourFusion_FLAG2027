#!/usr/bin/env python3
"""Self-test of the runner's 13.54 guards on a toy repository (no data, no GPU, no network; a few seconds of CPU).

    python runner/selftest.py [--dir SCRATCH] [--keep]

It copies the REAL configs/variant_1354.json and configs/modes.json into a temporary toy repository with four tiny stages and
checks, with the real runner code:
  1. FF_MODE=train: the lineage root t163l_orig writes a NEW lora_C1.pt (bytes differ from b97372e3...) and a passing dry check
     -> accepted, the new hash and root_check=passed are recorded in _DONE.json;
  2. a downstream stage that mounts the made LoRA and copies it into its output -> accepted (lineage + propagation);
  3. a stage that writes a lora_C1.pt of its own (not copied from an input) -> VariantError;
  4. the root stage with a failing dry check (209 clips) -> VariantError;
  5. FF_MODE=infer: a saved copy whose lora_C1.pt is not the original -> VariantError at the input;
  6. FF_SMOKE=1: the root stage without a dry check -> accepted, reported only;
  7. variant_check: lora_c1 'original, through ...' / 'not_used (...)' accepted, 'key-fixed (lf)' refused;
  8. verify_stage: 'expected' as a member dict, a bare SHA-256 and a dotted JSON subset never raise.
Exit code 0 when every case behaves as expected."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ffcore as C  # noqa: E402

SEP = C.CELL_SEP
DRY_OK = {"ok": True, "got": {"clips": 210, "persons": 70, "clips_by_source": {"mav:v1": 210}, "persons_by_source": {"mav:v1": 70}}}
DRY_BAD = {"ok": False, "got": {"clips": 209, "persons": 70, "clips_by_source": {"mav:v1": 209}, "persons_by_source": {"mav:v1": 70}}}


def cell(name, body):
    return f'print("[cell] {name}", flush=True)\n{body}'


def stage(root, sid, kind, inputs, kernel, exports, cells, **extra):
    d = root / "stages" / "40_toy" / sid
    d.mkdir(parents=True, exist_ok=True)
    sj = {"id": sid, "group": "40_toy", "title": f"toy {sid}", "kind": kind, "compute": {"device": "cpu", "minutes_original": 0.1},
          "inputs": [{"artifact": a} for a in inputs], "outputs": {"kernel": kernel, "exports": exports}, "origin": {"exactness": "new"}}
    sj.update(extra)
    (d / "stage.json").write_text(json.dumps(sj, indent=1), encoding="utf-8")
    (d / "stage.ipy").write_text(SEP.join(cells) + SEP, encoding="utf-8")


def build_repo(root, dry):
    shutil.rmtree(root, ignore_errors=True)
    (root / "configs").mkdir(parents=True)
    for f in ("variant_1354.json", "modes.json"):
        shutil.copy2(HERE.parent / "configs" / f, root / "configs" / f)
    lora = "datasets/onnhqunhh/flag2027-rfs-lora-p"
    dm = {"artifacts": {lora: {"path": "03_checkpoints/rfs_lora_c1_original", "class": "checkpoint", "producer": "t163l_orig", "private": True}}}
    (root / "configs/data_map.json").write_text(json.dumps(dm, indent=1), encoding="utf-8")
    (root / "configs/pipeline_1354.json").write_text(json.dumps({"stage_roots": ["stages"], "stages": [], "default_target": "toy_prop"}), encoding="utf-8")
    wr = ("import json, os\nfrom pathlib import Path\nW = Path('/kaggle/working')\n(W / 'rfs_lora').mkdir(exist_ok=True)\n"
          "smoke = os.environ.get('FF_SMOKE') == '1'\n"
          "(W / 'rfs_lora' / 'lora_C1.pt').write_bytes(b'retrained LoRA C1 ' + (b'smoke' if smoke else b'real'))\n"
          f"if not smoke:\n    json.dump({dry!r}, open(W / 'ff_lora_drycheck.json', 'w'))\n")
    stage(root, "t163l_orig", "train", [], "notebooks/toy/root", {lora: "rfs_lora"}, [cell("root: train LoRA", wr)])
    cp = ("import shutil\nfrom pathlib import Path\nsrc = next(Path('/kaggle/input').rglob('lora_C1.pt'))\n(Path('/kaggle/working') / 'art').mkdir()\n"
          "shutil.copy(src, Path('/kaggle/working') / 'art' / 'lora_C1.pt')\n")
    stage(root, "toy_prop", "train", [lora], "notebooks/toy/prop", {}, [cell("prop: copy LoRA", cp)],
          variant={"lora_c1": "original, through the toy root"})
    own = "from pathlib import Path\n(Path('/kaggle/working') / 'lora_C1.pt').write_bytes(b'some other LoRA')\n"
    stage(root, "toy_bad", "train", [lora], "notebooks/toy/bad", {}, [cell("bad: own LoRA", own)])
    stage(root, "toy_infer", "infer", [lora], "notebooks/toy/infer", {}, [cell("infer: read LoRA", "print('ok')")])


def runner(root, work, data, mode, smoke=False):
    os.environ.pop("FF_SMOKE", None)
    cfg = C.Config(repo=str(root), work=str(work), data=str(data), mode=mode, pip="skip", models="off", smoke=smoke, use_saved="none" if mode == "train" else "all")
    return C.Runner(cfg, quiet=True)


def expect_error(fn, kind):
    try:
        fn()
    except kind as ex:
        return True, str(ex)[:160]
    except Exception as ex:                                         # noqa: BLE001
        return False, f"{type(ex).__name__}: {str(ex)[:160]}"
    return False, "no error raised"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="scratch folder (default: a new temporary folder)")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    base = Path(a.dir) if a.dir else Path(tempfile.mkdtemp(prefix="ff_selftest_"))
    base.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    res = []
    try:
        # 1 + 2 + 3: train mode, lineage
        root, work, data = base / "repo", base / "work", base / "data"
        data.mkdir(parents=True, exist_ok=True)
        build_repo(root, DRY_OK)
        R = runner(root, work, data, "train")
        try:
            R.run_stage(R.S["t163l_orig"])
            info = C.jload(R.art_dir("datasets/onnhqunhh/flag2027-rfs-lora-p") / "_DONE.json")
            lin = info.get("variant_lineage", {}).get("lora_c1", {})
            res.append(("1 train: lineage root accepted, hash recorded", lin.get("root_check") == "passed" and len(lin.get("sha256", [])) == 1,
                        json.dumps(lin)[:160]))
        except Exception as ex:                                     # noqa: BLE001
            res.append(("1 train: lineage root accepted, hash recorded", False, f"{type(ex).__name__}: {ex}"[:200]))
        try:
            R.run_stage(R.S["toy_prop"])
            res.append(("2 train: downstream copy accepted", True, ""))
        except Exception as ex:                                     # noqa: BLE001
            res.append(("2 train: downstream copy accepted", False, f"{type(ex).__name__}: {ex}"[:200]))
        ok, msg = expect_error(lambda: R.run_stage(R.S["toy_bad"]), C.VariantError)
        res.append(("3 train: own lora_C1.pt refused", ok, msg))
        # 4: failing dry check
        root4, work4 = base / "repo4", base / "work4"
        build_repo(root4, DRY_BAD)
        R4 = runner(root4, work4, data, "train")
        ok, msg = expect_error(lambda: R4.run_stage(R4.S["t163l_orig"]), C.VariantError)
        res.append(("4 train: failing dry check refused", ok, msg))
        # 5: infer mode, wrong saved copy
        sv = data / "03_checkpoints/rfs_lora_c1_original"
        sv.mkdir(parents=True, exist_ok=True)
        (sv / "lora_C1.pt").write_bytes(b"not the original")
        R5 = runner(root, base / "work5", data, "infer")
        ok, msg = expect_error(lambda: R5.run_stage(R5.S["toy_infer"]), C.VariantError)
        res.append(("5 infer: non-original saved LoRA refused", ok, msg))
        shutil.rmtree(sv, ignore_errors=True)
        # 6: smoke root without dry check
        R6 = runner(root, base / "work6", data, "train", smoke=True)
        try:
            R6.run_stage(R6.S["t163l_orig"])
            res.append(("6 smoke: placeholder LoRA reported only", True, ""))
        except Exception as ex:                                     # noqa: BLE001
            res.append(("6 smoke: placeholder LoRA reported only", False, f"{type(ex).__name__}: {ex}"[:200]))
        # 7: lora_c1 field
        vals = {"original, through rfx_eval.zip of e163i": True, "not_used (no LoRA input)": True, "not used": True,
                "key-fixed (lf)": False, "C1FIX-01 fixed LoRA": False, "something else": False}
        bad7 = [v for v, want in vals.items() if R.lora_c1_value_ok(v)[0] != want]
        res.append(("7 variant_check lora_c1 normalisation", not bad7, f"wrong: {bad7}" if bad7 else ""))
        # 8: verify shapes
        kd = R.art_dir("notebooks/toy/prop")
        kd.mkdir(parents=True, exist_ok=True)
        zp = kd / "x.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("a.txt", "hello")
        (kd / "m.json").write_text(json.dumps({"files": {"lora_C1.pt": "abc"}, "checks": {"ok": True}}), encoding="utf-8")
        st = R.S["toy_prop"]
        st.verify = [{"file": "x.zip", "check": "members_sha256", "expected": {"a.txt": hashlib.sha256(b"hello").hexdigest()}},
                     {"file": "x.zip", "check": "sha256", "expected": "0" * 64},
                     {"file": "m.json", "check": "json", "expected": {"files.lora_C1.pt": "abc", "checks.ok": True}},
                     {"file": "m.json", "check": "sha256", "expected": None}]
        try:
            ver = R.verify_stage(st)
            st_ = [s for s, _m in ver]
            res.append(("8 verify_stage shapes", st_[:1] == ["OK"] and "WARN" in st_ and st_.count("OK") >= 2 and "FAIL" not in st_,
                        "; ".join(f"{s} {m[:50]}" for s, m in ver)))
        except Exception as ex:                                     # noqa: BLE001
            res.append(("8 verify_stage shapes", False, f"{type(ex).__name__}: {ex}"[:200]))
    finally:
        if not a.keep:
            shutil.rmtree(base, ignore_errors=True)
    for name, ok, msg in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({msg})" if msg else ""))
    bad = [r for r in res if not r[1]]
    print(f"\n{len(res) - len(bad)}/{len(res)} cases as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
