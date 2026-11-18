#!/bin/bash
# RFS-01 buoc 2 (nhat ky 361): FLAG_163 L (LoRA C1) -> P (oof + projector + bo tron + artifact) -> notebook suy luan RF tren dev voi artifact moi -> tai zip.
# Dieu kien dau: dataset bundle `ngnnguynchn/flag2027-rfs-bundle-p` (ext.npz + v4.npz theo 'Hop dong bundle' cua Experiment/RFS-01/README.md, tu FLAG_162 + FLAG_160)
# doc duoc boi tai khoan L va P; neu khong co sau 12 h chuoi dung (ghi log, de lai cho nguoi dung).
# Moi su kien mot dong; ket thuc khi het viec hoac gap loi. Chay nen:  bash kaggle/rfs_chain.sh >> <log>
cd "$(dirname "$0")/.." || exit 1
K="python kaggle/kapi.py"
log() { echo "$(date '+%m-%d %H:%M') $*"; }
st() { timeout 120 $K status "$1" 2>&1 | grep -o 'COMPLETE\|ERROR\|CANCEL[A-Z]*\|RUNNING\|QUEUED' | head -1; }
ok_ds() { local a=$1; shift; local o; o=$(timeout 200 $K access "$a" "$@" 2>&1); [ "$(echo "$o" | grep -c ': readable')" = "$#" ]; }
waitk() { local t0=$(date +%s); while true; do s=$(st "$1"); case "$s" in COMPLETE) return 0;; ERROR|CANCEL*) log "$1 ended $s"; return 1;; esac; [ $(( $(date +%s) - t0 )) -gt "$2" ] && { log "$1 qua $2 s: $s"; return 1; }; sleep 600; done; }
LA=onnhqunhh; PA=ngnnguynchn; IA=ngnnguynchn
BUN=ngnnguynchn/flag2027-rfs-bundle-p; LID=ngnnguynchn/flag2027-lid-strict-p; CACHE=duongphihoang/flag2027-audio-cache-p
MAV=chnnhnnguyn/external-data-mavceleb; RAW=ngnnguynchn/flag2027-raw-v4-p
LORAD=$LA/flag2027-rfs-lora-p; ARTD=$PA/flag2027-rfs-art-p
LK=$LA/flag-2027-nb-163-rfs-l; PK=$PA/flag-2027-nb-163-rfs-p; IK=$IA/flag-2027-nb-163i-rfs-dev
log "chain start (pid $$)"
t0=$(date +%s)
RFSD="nguynchnngn1/flag2027-feats-rfs-a-p phanththui/flag2027-feats-rfs-b-p phanvnnnnn6789/flag2027-feats-rfs-c-p"
WHD="tranthuytien2511/flag2027-feats-whv3-a-p thanhhuyen11/flag2027-feats-whv3-b-p phuonggmaii/flag2027-feats-whv3-c-p"
EXM=ngnnguynchn/flag2027-ext-meta-p; BA=ngnnguynchn; BK=$BA/flag-2027-nb-163-rfs-b
# B: ghep FLAG_162 + FLAG_160 thanh bundle (CPU); doi toi khi du 6 dataset dac trung (dataset chi xuat hien sau khi luot tuong ung tai len xong)
until [ $(( $(date +%s) - t0 )) -gt 43200 ] || ok_ds $BA $RFSD $WHD $EXM $RAW; do
  for d in $RFSD $WHD; do $K privatize $d > /dev/null 2>&1; done; sleep 900
done
ok_ds $BA $RFSD $WHD $EXM $RAW || { log "dac trung FLAG_162 / FLAG_160 chua doc duoc sau 12 h: de lai cho nguoi dung"; exit 0; }
log "dac trung du: day luot B ghep bundle ($BK)"
$K push kaggle/FLAG_163_rfs.ipynb --account $BA --slug flag-2027-nb-163-rfs-b --internet --force --env "RFS_MODE=B;RFS_UPLOAD=$BUN"   --datasets $(echo $RFSD $WHD $EXM $RAW | tr ' ' ',') 2>&1 | grep -q "successfully pushed" || { log "push B that bai"; exit 1; }
sleep 300; waitk $BK 7200 || exit 1   # sleep: the previous version status stays visible 1-2 min after a push
$K privatize $BUN > /dev/null 2>&1; ok_ds $LA $BUN $LID $CACHE $MAV $RAW && ok_ds $PA $BUN $LID || { log "B COMPLETE nhung $BUN khong doc duoc: de lai cho nguoi dung"; exit 1; }
if [ "${RFS_SKIP_L:-0}" = 1 ]; then log "RFS_SKIP_L=1: dung lai LoRA da co ($LORAD)"; else   # L khong doc f_xf / v_xv
log "bundle doc duoc: day luot L ($LK)"
$K push kaggle/FLAG_163_rfs.ipynb --account $LA --slug flag-2027-nb-163-rfs-l --gpu --internet --env "RFS_MODE=L;RFS_UPLOAD=$LORAD" \
  --datasets $BUN,$LID,$CACHE,$MAV,$RAW 2>&1 | grep -q "successfully pushed" || { log "push L that bai"; exit 1; }
waitk $LK 28800 || exit 1
fi
$K privatize $LORAD > /dev/null 2>&1; ok_ds $PA $LORAD || { log "L COMPLETE nhung $LORAD khong doc duoc: de lai cho nguoi dung"; exit 1; }
log "L xong; day luot P ($PK)"
$K push kaggle/FLAG_163_rfs.ipynb --account $PA --slug flag-2027-nb-163-rfs-p --gpu --internet --force --env "RFS_MODE=P;RFS_UPLOAD=$ARTD" \
  --datasets $BUN,$LID,$LORAD 2>&1 | grep -q "successfully pushed" || { log "push P that bai"; exit 1; }
sleep 300; waitk $PK 21600 || exit 1
$K privatize $ARTD > /dev/null 2>&1; ok_ds $IA $ARTD $RAW $MAV || { log "P COMPLETE nhung $ARTD khong doc duoc: de lai cho nguoi dung"; exit 1; }
log "P xong; day notebook suy luan RF tren dev voi artifact moi ($IK)"
$K push kaggle/FLAG_163i_rfs_infer.ipynb --account $IA --slug flag-2027-nb-163i-rfs-dev --gpu --internet --datasets $ARTD,$RAW,$MAV 2>&1 | grep -q "successfully pushed" || { log "push infer that bai"; exit 1; }
waitk $IK 14400 || exit 1
mkdir -p Experiment/RFS-01/out && for p in submission report; do $K output $IK --dest Experiment/RFS-01/out --pattern $p > /dev/null 2>&1; done   # one pattern per call: kaggle.cmd breaks on '|'
log "chain xong: zip dev o Experiment/RFS-01/out/ (nop CB: viec cua nguoi dung; luat doc o README)"
