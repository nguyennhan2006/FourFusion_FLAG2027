#!/usr/bin/env python3
"""FourFusion FLAG 2027 - command line of the 13.54 reproduction chain.

    python runner/ffrun.py plan   [--mode infer|train|assemble] [--target T,...] [--assume-saved] [--json]
    python runner/ffrun.py run    [--mode ...] [--target ...] [--only S,...] [--force S,...] [--smoke] [--pip original|pinned|skip] [--dry]
    python runner/ffrun.py dag    [--compare-skeleton] [--json]       # whole-graph checks: cycles, duplicate producers, unresolved inputs
    python runner/ffrun.py locate ARTIFACT [...]                       # where an artifact would come from under the current settings
    python runner/ffrun.py status                                       # stage status recorded in FF_WORK/state.json
    python runner/ffrun.py check-stage [S ...] [--diff] [--json]        # = runner/check_stage.py
    python runner/ffrun.py verify zip ZIP [--expected KEY] | compare ZIP REF [--min-spearman X] | file ...   # = runner/verify.py
    python runner/ffrun.py docs   [--out FILE]                          # markdown table of every stage (for docs/STAGES.md)
    python runner/ffrun.py selftest [--dir D] [--keep]                  # = runner/selftest.py (variant guard + verify shapes, toy repo)

Every option has an environment twin (FF_MODE, FF_TARGET, FF_ONLY, FF_FORCE, FF_SMOKE, FF_PIP, FF_DATA, FF_WORK, FF_USE_SAVED,
FF_TIME_BUDGET_H, FF_MODELS); see runner/ffcore.py. Typical uses:
    FF_DATA=/data/ff FF_WORK=/work python runner/ffrun.py run --mode assemble          # (c) byte-exact zip from saved scores
    FF_DATA=/data/ff FF_WORK=/work python runner/ffrun.py run --mode infer             # (b) Evaluation media + checkpoints
    FF_DATA=/data/ff FF_WORK=/work python runner/ffrun.py run --mode train --only t153c   # (a) one session of the full rebuild"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ffcore as C  # noqa: E402


def cfg_from(a, **extra):
    over = {}
    for k in ("mode", "target", "only", "force", "pip", "data", "work", "use_saved", "models"):
        v = getattr(a, k, None)
        if v:
            over[k] = v
    if getattr(a, "smoke", False):
        over["smoke"] = True
    over.update(extra)
    return C.Config(**over)


def add_common(p):
    p.add_argument("--mode", choices=["train", "infer", "assemble"])
    p.add_argument("--target", help="stage id(s) or artifact id(s), comma separated (default final_verify)")
    p.add_argument("--data", help="FF_DATA (data roots separated by os.pathsep)")
    p.add_argument("--work", help="FF_WORK")
    p.add_argument("--use-saved", dest="use_saved", choices=["all", "collected", "none"])
    p.add_argument("--force", help="stage / artifact ids to recompute")
    p.add_argument("--models", help="FF_MODELS: pretrained folder | auto | off")


def cmd_plan(a):
    R = C.Runner(cfg_from(a, assume_saved=a.assume_saved))
    for w in R.warnings:
        print("WARNING", w)
    order, blocked = R.make_plan()
    dag = R.dag_check()
    if a.json:
        print(json.dumps({"config": R.cfg.describe(), "use_saved": R.use_saved, "order": order,
                          "blocked": [{"artifact": x, "for": v, "why": w} for x, v, w in blocked],
                          "dag": {k: v for k, v in dag.items() if k != "warnings"}}, indent=1, ensure_ascii=False))
        return 0 if not dag["cycles"] else 1
    R.print_plan(order, blocked)
    exp = R.mode.get("expected_stages")
    if exp:
        miss, extra = [s for s in exp if s not in order], [s for s in order if s not in exp]
        print(f"\nconfigs/modes.json expected_stages for FF_MODE={R.cfg.mode}: "
              + ("plan matches" if not miss and not extra else f"missing {miss} | not expected {extra}")
              + ("" if a.assume_saved else "  (without --assume-saved the plan depends on what exists under FF_DATA)"))
    print(f"\nDAG over {dag['stages']} stages ({dag['with_stage_json']} with stage.json, the others from the skeleton): "
          f"{'no cycle' if not dag['cycles'] else 'CYCLES ' + str(dag['cycles'])}; "
          f"{len(dag['duplicates'])} unresolved duplicate producer(s); {len(dag['unresolved_inputs'])} unresolved input(s); "
          f"{len(dag['forbidden_inputs'])} forbidden input(s); {len(dag['variant_errors'])} static variant error(s)")
    print_req1(dag)
    for sid, e in dag["variant_errors"]:
        print(f"  VARIANT ERROR {sid}: {e}")
    return 0 if not dag["cycles"] and not dag["variant_errors"] else 1


def print_req1(dag):
    """Requirement 1 (start from raw data): consumed artifacts that are not raw / collected / pretrained and that no stage of the
    repository produces, or that the modes always take from a frozen copy."""
    n1, n2 = dag["nonraw_without_producer"], dag["default_saved_frozen"]
    print(f"\nREQUIREMENT-1 GAPS: {len(n1)} consumed intermediate artifact(s) without a producing stage (always taken frozen from FF_DATA):")
    for x in n1:
        print(f"  {x['artifact']:62s} {x['class']:11s} -> {', '.join(x['consumers'][:6])}{' ...' if len(x['consumers']) > 6 else ''}")
        if x.get("gap"):
            print(f"      {str(x['gap'])[:220]}")
    print(f"{len(n2)} artifact(s) with a producing stage that the modes take frozen by default (rebuild with FF_FORCE=<producer>):")
    for x in n2:
        print(f"  {x['artifact']:62s} producer {x['producer']:16s} ({x['why']})")
    if dag.get("embedded_data"):
        print(f"{len(dag['embedded_data'])} data file(s) embedded in stage code (configs/data_map.json class 'embedded'):")
        for x in dag["embedded_data"]:
            print(f"  {x['artifact']:62s} producer {x.get('producer') or '-':12s} carried by {len(x.get('carried_by') or [])} stage(s)")


def cmd_dag(a):
    R = C.Runner(cfg_from(a))
    rep = R.dag_check(compare_skeleton=a.compare_skeleton)
    if a.json:
        print(json.dumps(rep, indent=1, ensure_ascii=False))
    else:
        print(f"{rep['stages']} stages, {rep['with_stage_json']} with stage.json")
        for k in ("cycles", "duplicates", "unresolved_inputs", "forbidden_inputs", "variant_errors", "unknown_artifacts", "skeleton_diff",
                  "unconsumed_data_map_ids", "warnings"):
            v = rep[k]
            print(f"{k}: {len(v)}")
            for x in (v.items() if isinstance(v, dict) else v):
                print(f"   {x}")
        print_req1(rep)
    bad = rep["cycles"] or rep["duplicates"] or rep["forbidden_inputs"] or rep["variant_errors"]
    return 1 if bad else 0


def cmd_run(a):
    loaded = C.load_kaggle_secrets()
    cfg = cfg_from(a)
    R = C.Runner(cfg)
    print("config:", json.dumps(cfg.describe(), ensure_ascii=False), "| secrets from Kaggle store:", loaded or "none", flush=True)
    for w in R.warnings:
        print("WARNING", w)
    try:
        st = R.run(dry=a.dry)
    except (C.StageBlocked, C.VariantError) as e:
        print(f"\nSTOPPED: {type(e).__name__}: {e}", flush=True)
        R.save_state()
        return 2
    if a.dry:
        return 0
    bad = [s for s, v in st.items() if v not in ("done", "not_selected")]
    return 0 if not bad else 3


def cmd_locate(a):
    R = C.Runner(cfg_from(a))
    for aid in a.artifacts:
        p, how = R.locate(aid)
        prod = R.producer_of(aid)
        print(f"{aid}\n   class {R.cls(aid)} | producer {prod} | saved copy allowed now: {R.saved_allowed(aid)} | "
              f"{'FOUND (' + how + ') ' + str(p) if p else 'not found'}")
        if not p:
            for c in R.saved_candidates(aid)[:8]:
                print(f"      looked at {c}")
    return 0


def cmd_status(a):
    R = C.Runner(cfg_from(a))
    R.report()
    return 0


def cmd_docs(a):
    R = C.Runner(cfg_from(a))
    md = ("# Stages of the 13.54 chain\n\nGenerated by `python runner/ffrun.py docs` from every `stage.json` (stages without one come from "
          "the skeleton in `configs/pipeline_1354.json`).\n\n" + R.stages_markdown())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")
        print("wrote", a.out)
    else:
        print(md)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check-stage"]:
        import check_stage
        return check_stage.main(argv[1:])
    if argv[:1] == ["verify"]:
        import verify
        return verify.main(argv[1:])
    if argv[:1] == ["selftest"]:
        import selftest
        return selftest.main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    add_common(p)
    p.add_argument("--assume-saved", action="store_true", help="plan as if every artifact the mode may take from FF_DATA were there")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("run")
    add_common(p)
    p.add_argument("--only")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pip", choices=["original", "pinned", "skip"])
    p.add_argument("--dry", action="store_true", help="print the plan only")
    p = sub.add_parser("dag")
    add_common(p)
    p.add_argument("--compare-skeleton", action="store_true")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("locate")
    add_common(p)
    p.add_argument("artifacts", nargs="+")
    p = sub.add_parser("status")
    add_common(p)
    p = sub.add_parser("docs")
    add_common(p)
    p.add_argument("--out")
    sub.add_parser("check-stage", help="see runner/check_stage.py --help")
    sub.add_parser("verify", help="see runner/verify.py --help")
    sub.add_parser("selftest", help="see runner/selftest.py --help (13.54 guard self-test on a toy repository)")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "dag": cmd_dag, "locate": cmd_locate, "status": cmd_status, "docs": cmd_docs}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
