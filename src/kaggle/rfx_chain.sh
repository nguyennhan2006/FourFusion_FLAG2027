#!/bin/bash
# FACTOR-01 step 3 (13/11, Experiment/FACTOR-01/README.md section 4; gate passed, log 428): final RFX = RFS recipe a0 + extra blocks c5rl.
#   smoke CPU FLAG_163 P (stand-in blocks) -> P GPU on account A (+ the FACTOR-01 block datasets) -> privatize the artifact
#   -> FLAG_163i (patch_163i_eval + patch_163i_factor) on account EA, which owns the Evaluation kernels of the F1s' path
#      (47rs = rn293 / ecapa1024, 65xcs = C5s faces of the Evaluation lists) -> U_X = F1s' + RFX and ENFUSE_X, frozen in POSTEVAL-01.
#   Nothing is submitted. Start through WMI (CLAUDE.md section 7):
#   bash kaggle/rfx_chain.sh >> Experiment/FACTOR-01/rfx_chain.log 2>> Experiment/FACTOR-01/rfx_chain.err
cd "$(dirname "$0")/.." || exit 1
K="python kaggle/kapi.py"
A=${A:-ngnnguynchn}; EA=${EA:-thanhhuyen11}; NAME=${NAME:-fxc5rl}; CFGJ=${CFGJ:-'{"add": ["f_c5s", "v_rl"]}'}; TAG=${TAG:-c5rl}
BUN=ngnnguynchn/flag2027-rfs-bundle-p; LID=ngnnguynchn/flag2027-lid-strict-p; LORAD=ngnnguynchn/flag2027-rfs-lorafix1-p
XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
KX="$EA/flag-2027-nb-47rs-screen2-eval,$EA/flag-2027-nb-65xcs-c5sx-eval"
XDS="$XDS${XDS_ADD:+,$XDS_ADD}"; KX="$KX${KX_ADD:+,$KX_ADD}"   # WHL-01 (15/11): extra block datasets / Evaluation kernels; unset = as before
ART=$A/flag2027-rfs$NAME-art-p
PK=$A/flag-2027-nb-163-rfs$NAME-p; SK=$PK-smoke; EK=$EA/flag-2027-nb-163i-rfs$NAME-eval${EVSUF:-}
EVALDS=ngnnguynchn/flag2027-evalphase-p; DATA_ROOT=/kaggle/input/datasets/ngnnguynchn/flag2027-evalphase-p
O=${O:-Experiment/FACTOR-01}; EO=Experiment/EVALPATH-U/eval_out; PZ=Experiment/POSTEVAL-01/zips
ZP=${ZPFX:-FACTOR}; ZU=posteval_${ZP}_U_$TAG.zip; ZE=posteval_${ZP}_ENFUSE_$TAG.zip   # FUSEDIM-02 (15/11): ZPFX / O; unset = as before
[ -n "${W:-}" ] && export XT_W=$W ENF_W=$W   # FUSEDIM-02: F1s' share w of the U / ENFUSE sums (XTEAM-03, ENFUSE-U); unset = 1:1
CNT='no_gender/sub_score_v4_English_heard.txt=392,no_gender/sub_score_v4_Bangla_unheard.txt=802,gender/sub_score_v4_English_heard.txt=416,gender/sub_score_v4_Bangla_unheard.txt=716'
st() { timeout 120 $K status "$1" 2>&1 | grep -o 'COMPLETE\|ERROR\|CANCEL[A-Z]*\|RUNNING\|QUEUED' | head -1; }
fin() { [ -n "$1" ] && [ "$1" != RUNNING ] && [ "$1" != QUEUED ] && [ "$1" != COMPLETE ]; }
log() { echo "$(date '+%m-%d %H:%M') $*"; }
dlzip() { local d; d=$(mktemp -d); $K output "$1" --dest "$d" --pattern submission > /dev/null 2>&1; find "$d" -name "submission.zip" | head -1; }
mkdir -p $O/out
s=${SO:-0}; t0=$(date +%s)
log "rfx chain start (pid $$): $NAME cfg $CFGJ; P on $A, Evaluation inference on $EA (+ $KX)"
while [ $(( $(date +%s) - t0 )) -lt 72000 ] && [ $s != 9 ] && [ $s != 4 ]; do
  case $s in
  0) $K push kaggle/FLAG_163_rfs.ipynb --account $A --slug flag-2027-nb-163-rfs$NAME-p --internet --datasets "$BUN,$LID,$LORAD" \
       --smoke "FLAG_SMOKE=1;RFS_STEPS=40;RFS_NTRIAL=150;RFS_MODE=P;RFS_CFG=$CFGJ" 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') smoke P $NAME: {} pushed"
     s=1 ;;
  1) x=$(st $SK)
     if [ "$x" = COMPLETE ]; then
       $K push kaggle/FLAG_163_rfs.ipynb --account $A --slug flag-2027-nb-163-rfs$NAME-p --gpu --internet --force --datasets "$BUN,$LID,$LORAD,$XDS" \
         --env "RFS_MODE=P;RFS_CFG=$CFGJ;RFS_UPLOAD=$ART" 2>&1 | grep -c "successfully pushed" | xargs -I{} echo "$(date '+%m-%d %H:%M') P $NAME GPU: {} pushed"
       s=2
     elif fin "$x"; then log "smoke P ended '$x' (kapi.py logs $SK): stopped"; s=9; fi ;;
  2) x=$(st $PK)
     if [ "$x" = COMPLETE ]; then
       sleep 240; $K privatize $ART > /dev/null 2>&1; sleep 60
       if timeout 200 $K access $EA $ART 2>&1 | grep -q ": readable"; then
         D=$(mktemp -d); command -v cygpath > /dev/null && D=$(cygpath -m "$D")
         MSYS_NO_PATHCONV=1 python kaggle/patch_163i_eval.py "$D/FLAG_163i_rfs_infer.ipynb" --data-root "$DATA_ROOT" > /dev/null && python kaggle/patch_163i_factor.py "$D/FLAG_163i_rfs_infer.ipynb" > /dev/null \
           && python kaggle/patch_163i_dataroot.py "$D/FLAG_163i_rfs_infer.ipynb" > /dev/null || { log "patch failed: stopped"; s=9; continue; }
         o=$($K push "$D/FLAG_163i_rfs_infer.ipynb" --account $EA --slug "${EK#*/}" --gpu --internet --datasets "$ART,ngnnguynchn/flag2027-raw-v4-p,chnnhnnguyn/external-data-mavceleb,$EVALDS" \
           --kernels "$KX" 2>&1 | grep -v Warning | tail -1)
         log "infer $NAME eval on $EA: $o"
         echo "$o" | grep -q "successfully pushed" && s=3   # 15/11: a refused push (e.g. 2 GPU sessions busy on $EA) is retried at the next poll
       else log "$ART not readable by $EA: stopped"; s=9; fi
     elif fin "$x"; then log "P $NAME ended '$x' (kapi.py logs $PK): stopped"; s=9; fi ;;
  3) x=$(st $EK)
     if [ "$x" = COMPLETE ]; then
       z=$(dlzip $EK)
       if [ -z "$z" ]; then log "infer eval COMPLETE but no submission.zip: stopped"; s=9; continue; fi
       cp "$z" $O/out/rfs${NAME}_eval.zip
       bash Experiment/EVALPATH-U/make_eval_zip.sh $EO/f1s_eval.zip $O/out/rfs${NAME}_eval.zip $PZ/$ZU eval --counts "$CNT" 2>&1 | sed 's/^/    /'
       python - "$PZ/$ZU" <<'PY'
import hashlib, json, sys
from pathlib import Path
dst = Path(sys.argv[1]); m = Path("Experiment/POSTEVAL-01/manifest.json"); M = json.loads(m.read_text(encoding="utf-8"))
M["zips"][dst.name] = dict(sha256=hashlib.sha256(dst.read_bytes()).hexdigest(), note="rfx_chain (FACTOR-01 step 3): U_X = F1s' + RFX, frozen before any label")
m.write_text(json.dumps(M, indent=1), encoding="utf-8"); print("frozen", dst, M["zips"][dst.name]["sha256"][:16])
PY
       python Experiment/EVALPATH-U/check_eval_zip.py "$PZ/$ZU" --counts "$CNT" --peer $EO/submission_U_eval.zip 2>&1 | tail -1 | sed 's/^/    check U: /'
       PYTHONIOENCODING=utf-8 ENF_RFS=$O/out/rfs${NAME}_eval.zip ENF_U=$PZ/$ZU ENF_OUT=$PZ/$ZE python Experiment/ENFUSE-U/run.py 2>&1 | tail -2 | sed 's/^/    enfuse: /'
       python Experiment/EVALPATH-U/check_eval_zip.py "$PZ/$ZE" --counts "$CNT" --peer $EO/submission_U_eval.zip 2>&1 | tail -1 | sed 's/^/    check ENFUSE: /'
       log "frozen $PZ/$ZU and $PZ/$ZE (NOT submitted)"; s=4
     elif fin "$x"; then log "infer $NAME eval ended '$x' (kapi.py logs $EK): stopped"; s=9; fi ;;
  esac
  [ $s = 4 ] || [ $s = 9 ] || sleep 600
done
log "rfx chain end: stage $s"
