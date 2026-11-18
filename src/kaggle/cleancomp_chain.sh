#!/bin/bash
# 3/11 CLEANCOMP-01 (FLAG_164, nhat ky 362; Experiment/CLEANCOMP-01/README.md): when the three GPU runs are COMPLETE -> download, merge into the
# strict e2e bundle (build_tables.py + build_dev.py on CPU), create the private datasets flag2027-cleancomp-e2e-p / -final-p, privatize.
# Then F1s: f1s_chain pushes 112f1s itself once its inputs are ready AND the bundle is readable; if its attempt was blocked ("F1s: 0 / 1 pushed" in its
# log) this chain pushes 112f1s (ACC = locnguyentruongan, the account of F0s). Each stage once; ERROR / CANCEL / failed step -> one log line, stop
# (no retry). Polls every 10 min, gives up after 24 h.   bash kaggle/cleancomp_chain.sh >> <log>
cd "$(dirname "$0")/.." || exit 1
K="python kaggle/kapi.py"
A2=locnguyentruongan
st() { timeout 120 $K status "$1" 2>&1 | grep -o 'COMPLETE\|ERROR\|CANCEL[A-Z]*\|RUNNING\|QUEUED' | head -1; }
ok_ds() { timeout 200 $K access $A2 "$1" 2>&1 | grep -q ": readable"; }
log() { echo "$(date '+%m-%d %H:%M') $*"; }
fin() { [ -n "$1" ] && [ "$1" != RUNNING ] && [ "$1" != QUEUED ] && [ "$1" != COMPLETE ]; }
KF1=nguynlcnhtho/flag-2027-nb-164-cleancomp01-f123; KF2=dngxunqun/flag-2027-nb-164-cleancomp01-f45; KFIN=dngxunqun/flag-2027-nb-164f-cleancomp01-final
F0K=$A2/flag-2027-nb-112s-f0s; F1K=$A2/flag-2027-nb-112s-f1s
CCE=ngnnguynchn/flag2027-cleancomp-e2e-p; CCF=ngnnguynchn/flag2027-cleancomp-final-p
W=${CC_WORK:-/c/Users/ASUS/AppData/Local/Temp/claude/25100f4d054c6988/0f0ca738-bd3a-47cc-b0db-1af8914f37cf/scratchpad/cleancomp}
F1LOG=/c/Users/ASUS/AppData/Local/Temp/claude/25100f4d054c6988/0f0ca738-bd3a-47cc-b0db-1af8914f37cf/scratchpad/chains/f1s_chain.log
mkdir -p "$W"
stage=0   # 0 waiting for the GPU runs, 1 datasets built, 2 F1s pushed / seen, 9 stopped
t0=$(date +%s)
log "cleancomp chain start (pid $$)"
if ok_ds $CCE && ok_ds $CCF; then log "datasets $CCE / $CCF already readable by $A2: stage 1"; stage=1; fi
while [ $(( $(date +%s) - t0 )) -lt 86400 ]; do
  if [ $stage = 0 ]; then
    s1=$(st $KF1); s2=$(st $KF2); s3=$(st $KFIN)
    if fin "$s1" || fin "$s2" || fin "$s3"; then log "a CLEANCOMP run ended badly ($s1 / $s2 / $s3): stopped, F1s not unblocked"; stage=9
    elif [ "$s1" = COMPLETE ] && [ "$s2" = COMPLETE ] && [ "$s3" = COMPLETE ]; then
      rm -rf "$W/out" "$W/bundle" "$W/final"; mkdir -p "$W/out" "$W/final"
      okb=1
      for k in $KF1 $KF2; do $K output $k --dest "$W/out/$(basename $k)" --pattern "cleancomp01_fold[0-9]\.parquet" > /dev/null 2>&1; done
      find "$W/out" -name "cleancomp01_fold*.parquet" -exec cp {} "$W/out/" \;
      [ "$(ls "$W/out"/cleancomp01_fold*.parquet 2>/dev/null | wc -l)" = 5 ] || { log "fold tables: not 5 files"; okb=0; }
      $K output $KFIN --dest "$W/final/raw" --pattern "final" > /dev/null 2>&1
      fd=$(dirname "$(find "$W/final/raw" -name s007_xall.pt | head -1)")
      [ -f "$fd/s007_xall.pt" ] && [ -f "$fd/v11_head/t3_streams.npz" ] || { log "final artefacts not found under $W/final/raw"; okb=0; }
      if [ $okb = 1 ]; then
        PYTHONIOENCODING=utf-8 python -P Experiment/CLEANCOMP-01/build_tables.py "$W/out" "$W/bundle" > "$W/build_tables.log" 2>&1 || { log "build_tables failed ($W/build_tables.log)"; okb=0; }
      fi
      if [ $okb = 1 ]; then
        PYTHONIOENCODING=utf-8 python -P Experiment/CLEANCOMP-01/build_dev.py "$fd" "$W/bundle" > "$W/build_dev.log" 2>&1 || { log "build_dev failed ($W/build_dev.log)"; okb=0; }
      fi
      if [ $okb = 1 ]; then
        python -P -c "
import zipfile, pathlib, sys
W = pathlib.Path(sys.argv[1]); fd = pathlib.Path(sys.argv[2])
z = zipfile.ZipFile(W / 'flag2027-cleancomp-e2e.zip', 'w', zipfile.ZIP_DEFLATED)
[z.write(f, f.relative_to(W / 'bundle').as_posix()) for f in sorted((W / 'bundle').rglob('*')) if f.is_file()]; z.close()
z = zipfile.ZipFile(W / 'flag2027-cleancomp-final.zip', 'w', zipfile.ZIP_DEFLATED)
[z.write(fd / n, n) for n in ('s007_xall.pt', 'final_info.json', 'v11_head/v11_t3head.pt', 'v11_head/t3_streams.npz')]; z.close()
" "$(cygpath -w "$W")" "$(cygpath -w "$fd")" || okb=0
      fi
      if [ $okb = 1 ]; then
        $K dscreate "$(cygpath -w "$W/flag2027-cleancomp-e2e.zip")" --account ngnnguynchn --slug flag2027-cleancomp-e2e-p > "$W/dscreate_e2e.log" 2>&1 || okb=0
        $K dscreate "$(cygpath -w "$W/flag2027-cleancomp-final.zip")" --account ngnnguynchn --slug flag2027-cleancomp-final-p > "$W/dscreate_final.log" 2>&1 || okb=0
      fi
      if [ $okb = 1 ]; then
        sleep 240; $K privatize $CCE > /dev/null 2>&1; $K privatize $CCF > /dev/null 2>&1; sleep 60
        if ok_ds $CCE && ok_ds $CCF; then log "datasets $CCE / $CCF created, privatized, readable by $A2"; stage=1
        else log "datasets created but not readable by $A2 yet: run privatize on each of $CCE and $CCF"; stage=9; fi
      else log "building the bundle failed: logs in $W; stopped"; stage=9; fi
    fi
  fi
  if [ $stage = 1 ]; then
    sf=$(st $F1K)
    if [ -n "$sf" ]; then log "F1s kernel exists ($sf): pushed by f1s_chain with the strict bundle; nothing to do"; stage=2
    elif grep -q "F1s: 0 / 1 pushed" "$F1LOG" 2>/dev/null; then
      if [ "$(st $F0K)" = COMPLETE ] && timeout 200 $K access $A2 ngnnguynchn/flag2027-ct02s-r2ca-tables-p 2>&1 | grep -q ": readable"; then
        ACC=$A2 bash kaggle/sat_push.sh 112f1s 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') F1s (cleancomp chain): {} / 1 pushed"
        stage=2
      else log "F1s was blocked but F0s / r2ca tables are not ready: wait"; fi
    fi
  fi
  [ $stage = 2 ] || [ $stage = 9 ] && break
  sleep 600
done
log "cleancomp chain end: stage $stage"
