#!/bin/bash
# 3/11 PLAN_KEODIEM (docs/PLAN_KEODIEM_2026-11-03.md; user: "còn gì cứ đẩy và chạy hết ngay"): waits for the other session's clean
# adapters (FLAG_153b B3s, FLAG_153c C5s) and for MLLM-01 session 2, then pushes, each lane once:
#   B3s done            -> privatize B3s, rebuild FLAG_110, CLEANTUNE-02 g1 + g2 (6 runs)
#   B3s + C5s done      -> privatize C5s, rebuild FLAG_110 / FLAG_156, CLEANTUNE-02 g3 (3 runs) + FACENEG-01 (3 runs)
#   MLLM-01 s2 done     -> privatize its tables, FLAG_127 report (TABLES=mllm01)
# A lane whose kernel ends in ERROR / CANCEL, or whose dataset is not readable after COMPLETE, is logged and left to the user.
# One line per event; gives up after 14 h.   bash kaggle/keodiem_chain.sh >> <log>
cd "$(dirname "$0")/.." || exit 1
K="python kaggle/kapi.py"
st() { timeout 120 $K status "$1" 2>&1 | grep -o 'COMPLETE\|ERROR\|CANCEL[A-Z]*\|RUNNING\|QUEUED' | head -1; }
ok_ds() { timeout 200 $K access nguynthanhnhnn "$1" 2>&1 | grep -q ": readable"; }
log() { echo "$(date '+%m-%d %H:%M') $*"; }
B3K=phanvnhtrng/flag-2027-nb-153b-b3s; C5K=quynhnhu82/flag-2027-nb-153c-c5s; MLK=khanhnguyenm841/flag-2027-nb-147-mllm01-ft-r2
B3D=phanvnhtrng/flag2027-feats-ls-b3s-p; C5D=quynhnhu82/flag2027-feats-ls-c5s-p; MLD=khanhnguyenm841/flag2027-mllm01-ft-p
lb=0; lc=0; lm=0; t0=$(date +%s)
log "chain start (pid $$)"
while [ $((lb + lc + lm)) -lt 3 ] && [ $(( $(date +%s) - t0 )) -lt 50400 ]; do
  sb=$(st $B3K); sc=$(st $C5K); sm=$(st $MLK)
  if [ $lb = 0 ] && [ "$sb" = COMPLETE ]; then
    $K privatize $B3D > /dev/null 2>&1
    if ok_ds $B3D; then
      python kaggle/build_flag110.py > /dev/null 2>&1
      LS=1 CTB=$B3D CTG="g1 g2" ACCS="nguynthanhnhnn hdonqu thanhtruc02 rehoanzun nngocnh nguyenlethaonhi123" SLUGP=flag-2027-nb-154-ct02 \
        bash kaggle/sat_push.sh 110 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') CLEANTUNE-02 g1 + g2: {} / 6 pushed"
    else log "B3s COMPLETE but $B3D not readable: left to the user"; fi
    lb=1
  elif [ $lb = 0 ] && [ -n "$sb" ] && [ "$sb" != RUNNING ] && [ "$sb" != QUEUED ]; then log "B3s ended $sb: left to the user"; lb=1; fi
  if [ $lc = 0 ] && [ "$sc" = COMPLETE ] && [ $lb = 1 ]; then
    $K privatize $C5D > /dev/null 2>&1
    if ok_ds $C5D && ok_ds $B3D; then
      python kaggle/build_flag110.py > /dev/null 2>&1; python kaggle/build_flag156.py > /dev/null 2>&1
      LS=1 CTB=$B3D CTC=$C5D CTG=g3 ACCS="letranchauanh anhkieeu qunhhnghv" SLUGP=flag-2027-nb-154-ct02 \
        bash kaggle/sat_push.sh 110 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') CLEANTUNE-02 g3: {} / 3 pushed"
      CTB=$B3D CTC=$C5D ACCS="phannguyngiahuy11a5 nguynlynnnhi vantiennhuannguyen" \
        bash kaggle/sat_push.sh 156 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') FACENEG-01: {} / 3 pushed"
    else log "C5s COMPLETE but $C5D (or B3s) not readable: left to the user"; fi
    lc=1
  elif [ $lc = 0 ] && [ -n "$sc" ] && [ "$sc" != RUNNING ] && [ "$sc" != QUEUED ] && [ "$sc" != COMPLETE ]; then log "C5s ended $sc: left to the user"; lc=1; fi
  if [ $lm = 0 ] && [ "$sm" = COMPLETE ]; then
    $K privatize $MLD > /dev/null 2>&1
    if ok_ds $MLD; then
      ACC=trngsnam TABLES=mllm01 TABDS=$MLD SLUG=flag-2027-nb-127-reports-mllm01 bash kaggle/sat_push.sh 127 2>&1 \
        | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') MLLM-01 report: {} / 1 pushed"
    else log "MLLM-01 s2 COMPLETE but $MLD not readable (upload failed?): left to the user"; fi
    lm=1
  elif [ $lm = 0 ] && [ -n "$sm" ] && [ "$sm" != RUNNING ] && [ "$sm" != QUEUED ]; then log "MLLM-01 s2 ended $sm: left to the user"; lm=1; fi
  [ $((lb + lc + lm)) -lt 3 ] && sleep 600
done
log "chain end: B3s lane $lb, C5s lane $lc, MLLM lane $lm"
