#!/bin/bash
# 11/11 Evaluation run of U' = F1s' + RFS (+ US / U4 from RFSG), docs/PLAN_EVAL_U_2026-11-11.md sections 1-3b. Polls every 10 min.
#   lane G (EA, at most 2 GPU sessions): 50cc, 47rs, 65xbs, 65xcs pushed in that order as slots free up ("GPU session count" = retry)
#   lane X0: 65xbs + 50cc COMPLETE -> 112f0sx (CPU)      lane X1: 112f0sx + 65xcs + 47rs + 50cc COMPLETE -> 112f1sx (CPU)
#   lane Z: 112f1sx / RFS / RFSG COMPLETE -> download -> U' (EVALPATH-U/make_eval_zip.sh) and US / U4 (OBJFAM-01/make_eval_zips.sh)
# RFS / RFSG are pushed by hand (kaggle/rfs_eval_push.sh); this chain only downloads them. ERROR / CANCEL -> one log line, that lane stops.
# A kernel that already exists (any status) counts as pushed, so a restart resumes. Gives up after 14 h.
#   bash kaggle/evalrun_chain.sh >> Experiment/EVALPATH-U/eval_out/evalrun_chain.log
cd "$(dirname "$0")/.." || exit 1
K="python kaggle/kapi.py"
export EA=${EA:-thanhhuyen11} SUF=eval EVALDS=${EVALDS:-ngnnguynchn/flag2027-evalphase-p} ONLY_UNDER=${ONLY_UNDER:-evaluation_phase}
P=$EA/flag-2027-nb
K50=$P-50cc-e2e-$SUF; K47=$P-47rs-screen2-$SUF; K65B=$P-65xbs-b3sx-$SUF; K65C=$P-65xcs-c5sx-$SUF; K0=$P-112f0sx-$SUF; K1=$P-112f1sx-$SUF
KR=ngnnguynchn/flag-2027-nb-163i-rfs-eval; KG=ngnnguynchn/flag-2027-nb-163i-rfsg-eval
O=Experiment/EVALPATH-U/eval_out; W=$O/work; mkdir -p "$W"
CNT='no_gender/sub_score_v4_English_heard.txt=392,no_gender/sub_score_v4_Bangla_unheard.txt=802,gender/sub_score_v4_English_heard.txt=416,gender/sub_score_v4_Bangla_unheard.txt=716'
st() { timeout 120 $K status "$1" 2>&1 | grep -o 'COMPLETE\|ERROR\|CANCEL[A-Z]*\|RUNNING\|QUEUED' | head -1; }
log() { echo "$(date '+%m-%d %H:%M') $*"; }
bad() { [ -n "$1" ] && [ "$1" != RUNNING ] && [ "$1" != QUEUED ] && [ "$1" != COMPLETE ]; }
push() {   # push <step> [VAR=val ...] -> PUSHR = ok | slot | fail
  local s=$1; shift; local o; o=$(env "$@" bash kaggle/sat_push.sh "$s" 2>&1 | grep -v Warning | tail -1); log "push $s: $o"
  case "$o" in *successfully*) PUSHR=ok;; *"GPU session count"*|*"Maximum"*|*"maximum"*) PUSHR=slot;; *) PUSHR=fail;; esac; }
dl() {   # dl <kernel> <dir> <pattern> <file glob> -> path of the file (empty if none)
  rm -rf "$W/$2"; mkdir -p "$W/$2"; $K output "$1" --dest "$W/$2" --pattern "$3" > /dev/null 2>&1; find "$W/$2" -name "$4" | head -1; }
declare -A G; zf=0; zr=0; zg=0; zu=0; zs=0; x0=0; x1=0
t0=$(date +%s); log "evalrun chain start (pid $$): EA $EA, EVALDS $EVALDS, ONLY_UNDER $ONLY_UNDER"
while [ $(( $(date +%s) - t0 )) -lt 50400 ]; do
  s50=$(st $K50); s47=$(st $K47); s65b=$(st $K65B); s65c=$(st $K65C); s0=$(st $K0); s1=$(st $K1); sr=$(st $KR); sg=$(st $KG)
  # --- lane G
  for step in 50cc:$K50:$s50 47rs:$K47:$s47 65xbs:$K65B:$s65b 65xcs:$K65C:$s65c; do
    IFS=: read -r n k v <<< "$step"
    [ -n "$v" ] && [ -z "${G[$n]}" ] && { log "$n exists ($v)"; G[$n]=1; }
    if bad "$v" && [ "${G[$n]}" != 9 ]; then log "!! $n ended $v (log: $K logs $k)"; G[$n]=9; fi
    if [ -z "${G[$n]}" ]; then push $n; case $PUSHR in ok) G[$n]=1;; slot) break;; fail) G[$n]=9;; esac; fi
  done
  # --- lane X0 / X1
  if [ $x0 = 0 ] && [ -z "$s0" ] && [ "$s65b" = COMPLETE ] && [ "$s50" = COMPLETE ]; then
    push 112f0sx E2E_SPLIT=eval "KX=$K65B $K50"; case $PUSHR in ok) x0=1;; *) x0=9;; esac; fi
  [ -n "$s0" ] && [ $x0 = 0 ] && x0=1
  if [ $x1 = 0 ] && [ -z "$s1" ] && [ "$s0" = COMPLETE ] && [ "$s65c" = COMPLETE ] && [ "$s47" = COMPLETE ] && [ "$s50" = COMPLETE ]; then
    push 112f1sx E2E_SPLIT=eval "KX=$K0 $K65B $K65C $K47 $K50"; case $PUSHR in ok) x1=1;; *) x1=9;; esac; fi
  [ -n "$s1" ] && [ $x1 = 0 ] && x1=1
  for p in "112f0sx:$s0" "112f1sx:$s1" "rfs:$sr" "rfsg:$sg"; do n=${p%%:*}; v=${p#*:}
    if bad "$v" && [[ " ${seen:-} " != *" $n "* ]]; then log "!! $n ended $v"; seen="${seen:-} $n"; fi; done
  # --- lane Z: downloads
  if [ $zf = 0 ] && [ "$s1" = COMPLETE ]; then
    z=$(dl $K1 f1 submission_cleantune01_F1_CA "submission_cleantune01_F1_CA_*.zip")
    if [ -n "$z" ]; then cp "$z" "$O/f1s_eval.zip"; zf=1; log "F1s' zip: $(basename "$z") -> $O/f1s_eval.zip"
      timeout 300 $K logs $K1 --tail 400000 2>&1 | grep -a -m6 "A0 PASSED\|A0 FAIL\|unreadable" | sed 's/^/    112f1sx log: /'
      c=$(dl $K50 c50 checks.json checks.json); [ -n "$c" ] && log "50cc checks: $(grep -o '"level0_half_file_max_abs_diff"[^,}]*' "$c")"
    else log "!! 112f1sx COMPLETE but no F1 zip in its output"; zf=9; fi; fi
  if [ $zr = 0 ] && [ "$sr" = COMPLETE ]; then
    z=$(dl $KR rfs submission submission.zip); r=$(dl $KR rfsrep run_report run_report.json)
    if [ -n "$z" ]; then cp "$z" "$O/rfs_eval.zip"; zr=1; log "RFS zip -> $O/rfs_eval.zip; run_report: $([ -n "$r" ] && grep -o '"level0"[^,}]*' "$r")"
    else log "!! RFS COMPLETE but no submission.zip"; zr=9; fi; fi
  if [ $zg = 0 ] && [ "$sg" = COMPLETE ]; then
    z=$(dl $KG rfsg submission submission.zip); r=$(dl $KG rfsgrep run_report run_report.json)
    if [ -n "$z" ]; then cp "$z" "$O/rfsg_eval.zip"; zg=1; log "RFSG zip -> $O/rfsg_eval.zip; run_report: $([ -n "$r" ] && grep -o '"level0"[^,}]*' "$r")"
    else log "!! RFSG COMPLETE but no submission.zip"; zg=9; fi; fi
  bad "$sr" && [ $zr = 0 ] && zr=9; bad "$sg" && [ $zg = 0 ] && zg=9
  # --- lane Z: U' and US / U4
  if [ $zu = 0 ] && [ $zf = 1 ] && [ $zr = 1 ]; then
    o=$(bash Experiment/EVALPATH-U/make_eval_zip.sh "$O/f1s_eval.zip" "$O/rfs_eval.zip" "$O/submission_U_eval.zip" eval --counts "$CNT" 2>&1)
    echo "$o" | sed 's/^/    /'; v=$(echo "$o" | grep -cx OK); log "U' eval zip: $O/submission_U_eval.zip, OK lines $v/3"; zu=1; fi
  if [ $zs = 0 ] && [ $zf = 1 ] && [ $zr = 1 ] && [ $zg = 1 ]; then
    mkdir -p "$O/objfam"; o=$(bash Experiment/OBJFAM-01/make_eval_zips.sh "$O/f1s_eval.zip" "$O/rfs_eval.zip" "$O/rfsg_eval.zip" "$O/objfam" eval --counts "$CNT" 2>&1)
    echo "$o" | tail -12 | sed 's/^/    /'; log "US / U4: $(echo "$o" | tail -1) ($O/objfam)"; zs=1; fi
  # stop when nothing can progress
  f1dead=0; { [ $x0 = 9 ] || [ $x1 = 9 ] || [ $zf = 9 ] || bad "$s0" || bad "$s1" || [ "${G[50cc]}" = 9 ] || [ "${G[47rs]}" = 9 ] || [ "${G[65xbs]}" = 9 ] || [ "${G[65xcs]}" = 9 ]; } && f1dead=1
  if { [ $zu = 1 ] || [ $f1dead = 1 ] || [ $zr = 9 ]; } && { [ $zs = 1 ] || [ $zg = 9 ] || [ $f1dead = 1 ] || [ $zr = 9 ]; } && [ $zr != 0 ] && [ $zg != 0 ]; then break; fi
  sleep 600
done
log "evalrun chain end: G 50cc=${G[50cc]} 47rs=${G[47rs]} 65xbs=${G[65xbs]} 65xcs=${G[65xcs]}, 112f0sx $x0, 112f1sx $x1, zips F1s $zf RFS $zr RFSG $zg, U' $zu, US/U4 $zs"
