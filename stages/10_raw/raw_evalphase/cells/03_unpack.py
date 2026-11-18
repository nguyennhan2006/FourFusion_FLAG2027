print("[cell] raw_evalphase: unpack as the Kaggle dataset view (flag2027-evalphase-p = evaluation_phase/...)", flush=True)
# Kaggle extracted the uploaded eval_set.zip on upload: the dataset the Evaluation notebooks mounted holds evaluation_phase/... at its
# root (no zip). The export datasets/ngnnguynchn/flag2027-evalphase-p is that view; eval_set.zip itself stays in this stage's output
# (the DP branch reads the zip: stages 70_dp).
VIEW = OUT / "flag2027-evalphase-p"
if VIEW.exists():
    import shutil
    shutil.rmtree(VIEW)
VIEW.mkdir(parents=True)
with zipfile.ZipFile(EVAL_SET) as z:
    z.extractall(VIEW)
n = sum(1 for q in VIEW.rglob("*") if q.is_file())
lists = sorted(q.relative_to(VIEW).as_posix() for q in VIEW.glob("evaluation_phase/*/*_test.txt"))
REP.update(view_files=n, view_lists=lists, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
(OUT / "raw_evalphase_check.json").write_text(json.dumps(REP, indent=1))
print(f"dataset view: {n} files; lists {lists}", flush=True)
assert n == len(EXP) and len(lists) == 4
print("raw_evalphase: OK", flush=True)
