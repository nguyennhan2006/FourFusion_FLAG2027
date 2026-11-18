print("[cell] raw_evalphase: build eval_set.zip with build_evalphase.py (original local script, unchanged)", flush=True)
# Original run: on the team machine, 11/11/2026, `python -I Experiment/EVALPATH-U/build_evalphase.py <btc zip> <out zip>`.
# The script only renames <Lang>_valid -> <Lang>_test, fixes 416 list lines of gender/English (space instead of '/') and keeps the
# organiser lists under evaluation_phase/btc_original/; ids, row order and media bytes are unchanged.
EVAL_SET = OUT / "eval_set.zip"
if EVAL_SET.exists():
    EVAL_SET.unlink()                                           # never append to a partial zip of an interrupted run
cmd = [sys.executable, "-I", "-u", str(STAGE_DIR / "build_evalphase.py"), str(BTC), str(EVAL_SET)]
print("run:", " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
assert p.wait() == 0, "build_evalphase.py failed"
REP["eval_set_zip_sha256"] = sha256_file(EVAL_SET)
REP["eval_set_zip_sha256_equals_submitted"] = REP["eval_set_zip_sha256"] == EVAL_SET_SHA256_SUBMITTED
print("eval_set.zip sha256", REP["eval_set_zip_sha256"],
      "(= submitted copy)" if REP["eval_set_zip_sha256_equals_submitted"]
      else "(differs from a70762a3... as expected: zipfile.writestr stamps the build time; members are compared next)", flush=True)
