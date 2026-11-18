#!/bin/bash
# 3/11 lane F1s (nhat ky 355): builds the clean F1s system once its inputs exist. Each lane runs once; ERROR / CANCEL / unreadable dataset /
# failed gather -> one log line, lane stops (no retry). Polls every 10 min, gives up after 20 h.   bash kaggle/f1s_chain.sh >> <log>
#   lane R (CA tables): B3s + C5s COMPLETE -> 111s (CLEANTUNE-01 r2, only the CA arm, LID_STRICT=1) 3 GPU runs
#                       -> the 3 COMPLETE -> 111sr gather ct01r2 -> dataset ct02s-r2ca -> (with F0s COMPLETE) 112f1s = F1s
#   lane G (F0s)      : CLEANTUNE-02 g1 f12/f34/f5 (keodiem_chain's runs) COMPLETE -> 111sg gather ct01g1 -> dataset ct02s-g1
#                       -> 112f0s = F0s (B0s head, English bytes for F1s), same account as F1s (--kernels)
# GPU accounts: nguyenchonnhan (r2 f12, f34), locnguyentruongan (r2 f5, F0s, F1s). Zip of F1s: output of locnguyentruongan/flag-2027-nb-112s-f1s
# (submission_cleantune01_F1_CA_dev.zip). Does not touch keodiem_chain / rf_iblora_chain.
cd "$(dirname "$0")/.." || exit 1
K="python kaggle/kapi.py"
A1=nguyenchonnhan; A2=locnguyentruongan
st() { timeout 120 $K status "$1" 2>&1 | grep -o 'COMPLETE\|ERROR\|CANCEL[A-Z]*\|RUNNING\|QUEUED' | head -1; }
ok_ds() { for a in $A1 $A2; do timeout 200 $K access $a "$1" 2>&1 | grep -q ": readable" || return 1; done; }
log() { echo "$(date '+%m-%d %H:%M') $*"; }
fin() { [ -n "$1" ] && [ "$1" != RUNNING ] && [ "$1" != QUEUED ] && [ "$1" != COMPLETE ]; }   # ended badly
B3K=phanvnhtrng/flag-2027-nb-153b-b3s; C5K=quynhnhu82/flag-2027-nb-153c-c5s
B3D=phanvnhtrng/flag2027-feats-ls-b3s-p; C5D=quynhnhu82/flag2027-feats-ls-c5s-p
G1=("nguynthanhnhnn/flag-2027-nb-154-ct02-g1-f12" "hdonqu/flag-2027-nb-154-ct02-g1-f34" "thanhtruc02/flag-2027-nb-154-ct02-g1-f5")
R2=("$A1/flag-2027-nb-155s-ct02ca-f12" "$A1/flag-2027-nb-155s-ct02ca-f34" "$A2/flag-2027-nb-155s-ct02ca-f5")
F0K=$A2/flag-2027-nb-112s-f0s; F1K=$A2/flag-2027-nb-112s-f1s
CTTS=ngnnguynchn/flag2027-ct02s-g1-tables-p; CT2S=ngnnguynchn/flag2027-ct02s-r2ca-tables-p
export CTB=$B3D CTC=$C5D LS=1
# stage per lane: 0 waiting, 1 pushed/gathered, 9 stopped
r=0; g=0; r2g=0; f1=0; f0p=0; t0=$(date +%s)
log "f1s chain start (pid $$)"
while [ $(( $(date +%s) - t0 )) -lt 72000 ]; do
  sb=$(st $B3K); sc=$(st $C5K)
  # --- lane R: push the CA-only round 2
  if [ $r = 0 ]; then
    if [ "$sb" = COMPLETE ] && [ "$sc" = COMPLETE ]; then
      $K privatize $B3D $C5D > /dev/null 2>&1
      if ok_ds $B3D && ok_ds $C5D; then
        python kaggle/build_flag111.py > /dev/null 2>&1; python kaggle/build_flag112.py f0 > /dev/null 2>&1
        python kaggle/build_flag112.py f1 Experiment/CLEANTUNE-01/ct_decision.json > /dev/null 2>&1
        ACCS="$A1 $A1 $A2" bash kaggle/sat_push.sh 111s 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') 111s CA round 2: {} / 3 pushed"
        r=1
      else log "B3s / C5s COMPLETE but datasets not readable by $A1 and $A2: lane R stopped"; r=9; fi
    elif fin "$sb" || fin "$sc"; then log "B3s '$sb' / C5s '$sc' ended badly: lanes R, G stopped"; r=9; g=9; fi
  fi
  # --- lane G: gather g1 tables, push F0s
  if [ $g = 0 ]; then
    n=0; bad=0; for k in "${G1[@]}"; do s=$(st $k); [ "$s" = COMPLETE ] && n=$((n + 1)); fin "$s" && bad=1; done
    if [ $bad = 1 ]; then log "a CLEANTUNE-02 g1 run ended badly: lane G stopped"; g=9
    elif [ $n = 3 ]; then
      if RUNS="${G1[*]}" bash kaggle/sat_push.sh 111sg > /dev/null 2>&1; then
        log "g1 tables gathered -> $CTTS (waiting for it to be ready)"; sleep 240; $K privatize $CTTS > /dev/null 2>&1; g=1
      else log "g1 gather failed: lane G stopped"; g=9; fi
    fi
  fi
  if [ $g = 1 ] && [ $f0p = 0 ] && [ "$sb" = COMPLETE ]; then
    if ok_ds $CTTS && ok_ds $B3D; then
      python kaggle/build_flag112.py f0 > /dev/null 2>&1
      ACC=$A2 bash kaggle/sat_push.sh 112f0s 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') F0s: {} / 1 pushed"
      f0p=1
    else log "$CTTS not readable: F0s not pushed, lane G stopped"; g=9; f0p=9; fi
  fi
  # --- lane R part 2: gather r2 tables, push F1s when F0s is COMPLETE
  if [ $r = 1 ] && [ $r2g = 0 ]; then
    n=0; bad=0; for k in "${R2[@]}"; do s=$(st $k); [ "$s" = COMPLETE ] && n=$((n + 1)); fin "$s" && bad=1; done
    if [ $bad = 1 ]; then log "a 111s run ended badly: F1s stopped"; r2g=9
    elif [ $n = 3 ]; then
      if RUNS="${R2[*]}" bash kaggle/sat_push.sh 111sr > /dev/null 2>&1; then
        log "r2 CA tables gathered -> $CT2S"; sleep 240; $K privatize $CT2S > /dev/null 2>&1; r2g=1
      else log "r2 gather failed: F1s stopped"; r2g=9; fi
    fi
  fi
  if [ $r2g = 1 ] && [ $f1 = 0 ]; then
    s0=$(st $F0K)
    if [ "$s0" = COMPLETE ]; then
      if ok_ds $CT2S; then
        ACC=$A2 bash kaggle/sat_push.sh 112f1s 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') F1s: {} / 1 pushed"
        f1=1
      else log "$CT2S not readable: F1s not pushed"; f1=9; fi
    elif fin "$s0"; then log "F0s ended '$s0': F1s not pushed"; f1=9; fi
  fi
  if [ $f1 = 1 ]; then
    s1=$(st $F1K)
    if [ "$s1" = COMPLETE ]; then log "F1s COMPLETE: python kaggle/kapi.py output $F1K --dest <dir> --pattern submission"; f1=2
    elif fin "$s1"; then log "F1s ended '$s1': find the cause first (LIDSTRICT-01 README s6)"; f1=2; fi
  fi
  # stop when nothing can progress
  { [ $f1 = 2 ] || [ $f1 = 9 ] || [ $r = 9 ] || [ $r2g = 9 ]; } && { [ $f0p != 0 ] || [ $g = 9 ]; } && break
  sleep 600
done
log "f1s chain end: R $r, G $g, F0s push $f0p, r2 tables $r2g, F1s $f1"
