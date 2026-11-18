print("[cell] r07: overlap.py (SET-07, paths as arguments), threshold margins and comparison with the list hard-coded in the chain", flush=True)
# New cell. With --train-only the list depends only on THR, the v4-train centroid threshold (midpoint of the same-person p1 and the
# different-person p99.5 cosine on v4 train). The SET-07 run did not save THR; the same rule printed 0.624 in FLAG_08 (log 37), and
# the original overlap.csv brackets it: THR in [0.6212 (v2_complete:id0060, not flagged), 0.6503 (v1_complete:shaukat_tarin, flagged)).
# The margin is small (~0.003 below THR), so regenerated VGG-Face arrays (stage r08 on another GPU) could move a person across it.
# This stage is evidence only (the chain hard-codes the two keys): a difference limited to persons within NEAR of THR is reported
# as a warning; any other difference fails (unless FF_SMOKE=1). r07_check.json holds MAV person keys and scores only (no v4 ids).
import json, re, shutil
NEAR = 0.01
cmd = [sys.executable, "-u", str(STAGE_DIR / "overlap.py"), "--data", str(DATA), "--feats-ext", str(FEATS_EXT),
       "--lib", str(STAGE_DIR / "lib"), "--out", str(SCRATCH / "overlap.csv")] + ([] if WITH_DEV else ["--train-only"])
print("run:", " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
THR = None
for line in p.stdout:
    if "v4train_id" in line or re.search(r"\bid\d{3}\b", line):    # overlap.py prints its top rows incl. the v4 id: keep them out
        line = "[row printed by overlap.py hidden: it names a v4 train identity]\n"
    print(line, end="", flush=True)
    m = re.search(r"thresholds: centroid ([0-9.]+)", line)
    if m:
        THR = float(m.group(1))
assert p.wait() == 0, "overlap.py failed"
import pandas as pd
O = pd.read_csv(SCRATCH / "overlap.csv")
flagged = sorted(O.person[O.overlap.astype(bool)])
HARD_CODED = sorted({"v1_complete:shaukat_tarin", "v2_complete:id0073"})   # kaggle/build_l0_notebooks.py:92,1309; BEST/v7_safe/safe_lib.py:23
fl = O[O.overlap.astype(bool)]
nf = O[~O.overlap.astype(bool)].sort_values("max_cos_v4train", ascending=False)
rep = {"mode": "with dev rule (original)" if WITH_DEV else "train-only", "persons": int(len(O)),
       "thr_centroid_printed": THR, "thr_expected": {"log_37": 0.624, "bracket_original": [0.6212396621704102, 0.6503481268882751]},
       "flagged": {k: round(float(v), 6) for k, v in zip(fl.person, fl.max_cos_v4train)},
       "highest_not_flagged": {k: round(float(v), 6) for k, v in zip(nf.person[:2], nf.max_cos_v4train[:2])},
       "hard_coded": HARD_CODED, "same_as_hard_coded": flagged == HARD_CODED}
if THR is not None:
    rep["margin_lowest_flagged_minus_thr"] = round(float(fl.max_cos_v4train.min()) - THR, 6) if len(fl) else None
    rep["margin_thr_minus_highest_not_flagged"] = round(THR - float(nf.max_cos_v4train.max()), 6) if len(nf) else None
diff = sorted(set(flagged) ^ set(HARD_CODED))
near = THR is not None and all(abs(float(O.max_cos_v4train[O.person == k].iloc[0]) - THR) < NEAR for k in diff if (O.person == k).any())
rep["differing_persons"] = diff
rep["difference_only_near_threshold"] = bool(diff) and bool(near)
(WORK / "r07_check.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1), flush=True)
shutil.rmtree(SCRATCH, ignore_errors=True)                         # organisers' csv files + the linkage table: never kept
print("scratch removed:", SCRATCH, flush=True)
if rep["difference_only_near_threshold"]:
    print(f"!! WARNING: the list differs from the hard-coded one only by person(s) within {NEAR} of THR {THR}: {diff} "
          "(regenerated VGG-Face arrays); the chain is unaffected (it hard-codes the list)", flush=True)
assert rep["same_as_hard_coded"] or rep["difference_only_near_threshold"] or os.environ.get("FF_SMOKE") == "1", \
    f"overlap list differs from the list the chain hard-codes: {diff}"
