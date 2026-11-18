#!/usr/bin/env bash
# Runbook 16-17/10 (docs/PLAN_SAT_2026-10-17.md §3): every push of the plan as one named step, with its datasets and accounts.
#   bash kaggle/sat_push.sh <step> [ACC=<account>]        e.g.  bash kaggle/sat_push.sh 86t   |   ACC=trngsnam bash kaggle/sat_push.sh 86s
# STEP=2 runs the second half of the two-part steps (92r, 93r). Each step checks its input first (the run it waits for must be COMPLETE). After a push: check `[boot]` within 10 min
# (bash kaggle/sat_push.sh status); silent -> stop it and push again on the spare account (KAGGLE_API_RULES §6b).
set -euo pipefail
cd "$(dirname "$0")/.."
K="python kaggle/kapi.py"
DS73="ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-eda-bundle-p,ngnnguynchn/flag2027-e2e-dev-p,ngnnguynchn/flag2027-cc5-oof-p,ngnnguynchn/flag2027-dsel02-oof-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-feats-avs-p,ngnnguynchn/flag2027-feats-lrs3-p"
B3=nguynchnngn1/flag2027-feats-facelora3-b-p
TAB=ngnnguynchn/flag2027-facelora03-tables-p
DSC=$(echo "$DS73" | tr "," " " | xargs -n1 | grep -v -e feats-avs -e feats-lrs3 | paste -sd, -)
AVF=ngnnguynchn/flag2027-lab-avfull-p
SCR2=ngnnguynchn/flag2027-feats-screen2-p
# outputs of tonight's runs (auto-uploaded by the notebooks; `privatize` shares them with every account of the env file)
TTA=duongphihoang/flag2027-feats-clipb3-tta-p          # FLAG_65t
SA=lkhcvit/flag2027-feats-facelora6-sa-p               # FLAG_65s
C5=nguyentruonglong/flag2027-feats-facelora5-c5-p      # FLAG_65g r2
VF=hthlin/flag2027-aux-voice-feats-p                   # FLAG_63w
CTB=${CTB:-ngnnguynchn/flag2027-feats-ct-b3c-p}         # CLEANTUNE-01 FLAG_109b (Bengali-clean B3 adapter arrays)
CTC=${CTC:-nammo0/flag2027-feats-ct-c5c-p}              # CLEANTUNE-01 FLAG_109c (Bengali-clean C5 adapter arrays)
CTT=${CTT:-ngnnguynchn/flag2027-ct01-tables-p}          # CLEANTUNE-01 round-1 fold tables (15), made by step 110r STEP=1
                                                        # (F0 needs only ct01g1: CTT=ngnnguynchn/flag2027-ct01g1-tables-p, step 110g1)
VL=berryyusilee/flag2027-voicelora-p                   # FLAG_67a (VOICELORA-01 step 1)
VLX=ngnnguynchn/flag2027-feats-vlx-p                   # FLAG_67x output (step 1b), made by step 67xd
VL2ACC=${VL2ACC:-thanhthaoph}                          # FLAG_67b account (VOICELORA-02 step 1)
VL2=$VL2ACC/flag2027-voicelora2-p                      # FLAG_67b output (uploaded by the notebook)
VLX2=ngnnguynchn/flag2027-feats-vlx2-p                 # FLAG_67y output (step 1b), made by step 67yd
VL3ACC=${VL3ACC:-onnhqunhh}                            # FLAG_67c account (VOICELORA-03 step 1)
VL3=$VL3ACC/flag2027-voicelora3-p                      # FLAG_67c output (uploaded by the notebook)
VLX3=ngnnguynchn/flag2027-feats-vlx3-p                 # FLAG_67z p output (step 1b), made by step 67zpd
VL4ACC=${VL4ACC:-dnq1209}                              # FLAG_67d account (VOICELORA-04 step 1, chosen arm lr 2e-5)
VL4=$VL4ACC/flag2027-voicelora4-p                      # FLAG_67d output (uploaded by the notebook)
VLX4=ngnnguynchn/flag2027-feats-vlx4-p                 # FLAG_67z d output (step 1b), made by step 67zdd
JTACC=${JTACC:-trngminhhieu}                         # FLAG_65j account (JOINT-01 step 1, run flag-2027-nb-65j-joint1-r5; r4 on thanhthaoph deleted 20/10 05:36, §6d)
JT=$JTACC/flag2027-feats-joint1-p                    # FLAG_65j output (uploaded by the notebook): face_clipJT_*, voicelora_manifest.json, voice_sam100fJ.npy
VLXJ=ngnnguynchn/flag2027-feats-vlxj-p                 # FLAG_67z j output (step 1b), made by step 67zjd
ACACHE=duongphihoang/flag2027-audio-cache-p
MAVRAW="ngnnguynchn/flag2027-raw-v4-p,chnnhnnguyn/external-data-mavceleb,ngnnguynchn/flag2027-ext-meta-p"
CO=lkhcvit/flag2027-feats-cotrain1-co-p               # FLAG_65c output (COTRAIN-01 step 1)
VL2T=ngnnguynchn/flag2027-voicelora02-tables-p        # FLAG_98 fold tables (trR5 reference column)
D99="$DS73,$C5,$TAB,$AVF,$VL2,$VLX2,$VL3,$VLX3,$CO,$VL2T"   # FLAG_99 vl3,co; vl4 adds EXTRA99=$VL4,$VLX4, jt the joint1 + flag99_vl3co tables
RUNS="${RUNS:-hthlin/flag-2027-nb-63w-aux-voice lkhcvit/flag-2027-nb-65s-facelora6-sa duongphihoang/flag-2027-nb-65t-clipb3-tta
nguyentruonglong/flag-2027-nb-65g-facelora5-c5-r2 duongphihoang/flag-2027-nb-63v-audio-cache nguynchnngn1/flag-2027-nb-86c-csls01
nguynchnngn1/flag-2027-nb-93-sgw01-f1234}"   # env RUNS overrides (110g1 / 110r need their own list)

need_complete() {   # need_complete <owner/slug>
  s=$($K status "$1" 2>&1 | tail -1)
  case "$s" in *COMPLETE*) echo "ok: $1 complete" ;; *) echo "!! $1 not complete yet: $s"; exit 1 ;; esac
}

gather_tables() {   # gather_tables <tag> <dataset slug> <owner/kernel>... : fold tables of finished runs -> one private dataset
  # (a notebook cannot read another account's private notebook output through --kernels: the tables travel as a dataset)
  tag=$1; ds=$2; shift 2
  d=$(mktemp -d); mkdir -p "$d/e2e_out"
  for k in "$@"; do
    need_complete "$k"
    $K output "$k" --dest "$d/$(basename "$k")" --pattern "${PAT:-${tag}_fold[0-9]\.parquet}" >/dev/null   # PAT / GLOB (T3S-01): other table names
    find "$d/$(basename "$k")" -name "${GLOB:-${tag}_fold*.parquet}" -exec cp {} "$d/e2e_out/" \;
  done
  ls "$d/e2e_out"
  n=$(ls "$d/e2e_out" | wc -l); [ "$n" -eq "${NTAB:-5}" ] || { echo "!! $n fold tables (${NTAB:-5} needed)"; exit 1; }
  (cd "$d" && python -c "import zipfile,pathlib; z=zipfile.ZipFile('${tag}_tables.zip','w'); [z.write(f, 'tables/'+f.name) for f in sorted(pathlib.Path('e2e_out').glob('*.parquet'))]; z.close()")
  $K dscreate "$(cygpath -w "$d/${tag}_tables.zip" 2>/dev/null || echo "$d/${tag}_tables.zip")" --account ngnnguynchn --slug "$ds"   # Windows path (17/10: the POSIX one failed)
  echo "wait until the dataset is ready (1-3 min), then: $K privatize ngnnguynchn/$ds"
}

STEP_="${1:-}"
case "${1:-}" in
  status)
    for k in $RUNS ${EXTRA:-}; do $K alive "$k" 2>&1 | tail -2 | cut -c1-200; done ;;

  86t)   # TTA-02 step 2: after FLAG_65t (~17:30 16/10); ~2.5-3 h GPU T4 x2
    need_complete duongphihoang/flag-2027-nb-65t-clipb3-tta
    $K privatize $TTA
    $K push kaggle/FLAG_86t_tta02.ipynb --account "${ACC:-duongphihoang}" --slug flag-2027-nb-86t-tta02 --gpu \
      --datasets "$DS73,$B3,$TTA,$TAB,$AVF" ;;

  86s)   # FACELORA-06 step 2: after FLAG_65s (~22:30 16/10); ~2.5-3 h
    need_complete lkhcvit/flag-2027-nb-65s-facelora6-sa
    $K privatize $SA
    $K push kaggle/FLAG_86s_facelora06_sa.ipynb --account "${ACC:-nguynchnngn1}" --slug flag-2027-nb-86s-fl6-sa --gpu \
      --datasets "$DS73,$B3,$SA,$TAB,$AVF" ;;

  86g)   # FACELORA-05 step 2: after FLAG_65g r2 (~21:40 16/10); ~2.5-3 h
    need_complete nguyentruonglong/flag-2027-nb-65g-facelora5-c5-r2
    $K privatize $C5
    $K push kaggle/FLAG_86g_facelora05_c5.ipynb --account "${ACC:-nguyentruonglong}" --slug flag-2027-nb-86g-fl5-c5 --gpu \
      --datasets "$DS73,$B3,$C5,$TAB,$AVF" ;;

  92)    # ENC-SCALE-01 stage B: after FLAG_63w (~00:30 17/10); three fold groups, ~4.2 h each, then `92r`
    need_complete hthlin/flag-2027-nb-63w-aux-voice
    [ "${SKIP_PRIV:-}" = 1 ] || $K privatize $VF
    a=(${ACCS:-berryyusilee nammo0 thanhthaoph}); f=("1,2" "3,4" "5"); n=(f12 f34 f5)
    for i in 0 1 2; do
      if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${n[$i]}" ]; then continue; fi        # ONLY=f12 / f34 / f5: one fold group
      $K push kaggle/FLAG_92_encscale01.ipynb --account "${a[$i]}" --slug "flag-2027-nb-92-encscale01-${n[$i]}" --gpu \
        --datasets "$DS73,$B3,$TAB,$SCR2,$VF" --env "LAB_FOLDS=${f[$i]}"
    done ;;

  92r)   # ENC-SCALE-01: (1) tables of the three 92 runs -> dataset; (2) privatize; (3) gather + merge + report on CPU
    a=(${ACCS:-berryyusilee nammo0 thanhthaoph})
    if [ "${STEP:-1}" = 1 ]; then
      gather_tables encscale01 flag2027-encscale01-tables-p "${a[0]}/flag-2027-nb-92-encscale01-f12" \
        "${a[1]}/flag-2027-nb-92-encscale01-f34" "${a[2]}/flag-2027-nb-92-encscale01-f5"
      echo "then: STEP=2 bash kaggle/sat_push.sh 92r"
    else
      $K privatize ngnnguynchn/flag2027-encscale01-tables-p
      $K push kaggle/FLAG_92_encscale01.ipynb --account "${ACC:-nguynlcnhtho}" --slug flag-2027-nb-92-encscale01-report \
        --datasets "$DS73,$B3,$TAB,$SCR2,$VF,ngnnguynchn/flag2027-encscale01-tables-p" --env "LAB_FOLDS="
    fi ;;

  93)    # SGW-01 fold 5 (folds 1-4: nguynchnngn1/flag-2027-nb-93-sgw01-f1234, pushed 16/10 16:54, ~6.2 h); ~3.1 h GPU
    $K push kaggle/FLAG_93_sgw01.ipynb --account "${ACC:-trngsnam}" --slug flag-2027-nb-93-sgw01-f5 --gpu       --datasets "$DS73,$B3,$TAB" --env "LAB_FOLDS=5" ;;

  93r)   # SGW-01: (1) tables of f1234 + f5 -> dataset; (2) privatize + gather + report on CPU
    if [ "${STEP:-1}" = 1 ]; then
      gather_tables sgw01 flag2027-sgw01-tables-p nguynchnngn1/flag-2027-nb-93-sgw01-f1234 "${ACC5:-nguyenchonnhan}/flag-2027-nb-93-sgw01-f5"
      echo "then: STEP=2 bash kaggle/sat_push.sh 93r"
    else
      $K privatize ngnnguynchn/flag2027-sgw01-tables-p
      $K push kaggle/FLAG_93_sgw01.ipynb --account "${ACC:-nguynlcnhtho}" --slug flag-2027-nb-93-sgw01-report \
        --datasets "$DS73,$B3,$TAB,ngnnguynchn/flag2027-sgw01-tables-p" --env "LAB_FOLDS="
    fi ;;

  csls)  # CSLS-01 candidate: final B3 head (5 seeds) + trB3 / trCS of the dev files + zip (GPU ~1.5 h); resumes the 5 folds
         # of the 16/10 run (dataset flag2027-csls01-tables-p: e2e_out/ of that run; --kernels of another account's private
         # notebook is NOT readable). The zip goes to CodaBench ONLY after the user accepts the train-bank reading (README CSLS-01).
    $K push kaggle/FLAG_86c_csls01.ipynb --account "${ACC:-nguyentruonglong}" --slug flag-2027-nb-86c-csls01-final --gpu       --datasets "$DS73,$B3,$AVF,ngnnguynchn/flag2027-csls01-tables-p" --env "E2E_ALLOW_CODE_CHANGE=1" ;;

  94)    # CSLS-02: one GPU run per fold (~45 min each) on the accounts with 1-2.5 h left (16/10 evening); then `94r`
    a=(${ACCS:-trngminhhieu locnguyentruongan chnnhnnguyn berryyusilee nguynlcnhtho})
    for s in 1 2 3 4 5; do
      $K push kaggle/FLAG_94_csls02.ipynb --account "${a[$((s-1))]}" --slug "flag-2027-nb-94-csls02-f$s" --gpu \
        --datasets "$DS73,$B3,ngnnguynchn/flag2027-csls01-tables-p" --env "LAB_FOLDS=$s"
    done ;;

  94r)   # CSLS-02: (1) tables of the five fold runs -> dataset; (2) privatize + gather + report on CPU
    a=(${ACCS:-trngminhhieu locnguyentruongan chnnhnnguyn berryyusilee nguynlcnhtho})
    if [ "${STEP:-1}" = 1 ]; then
      gather_tables csls02 flag2027-csls02-tables-p "${a[0]}/flag-2027-nb-94-csls02-f1" "${a[1]}/flag-2027-nb-94-csls02-f2" \
        "${a[2]}/flag-2027-nb-94-csls02-f3" "${a[3]}/flag-2027-nb-94-csls02-f4" "${a[4]}/flag-2027-nb-94-csls02-f5"
      echo "then: STEP=2 bash kaggle/sat_push.sh 94r"
    else
      $K privatize ngnnguynchn/flag2027-csls02-tables-p
      $K push kaggle/FLAG_94_csls02.ipynb --account "${ACC:-trngsnam}" --slug flag-2027-nb-94-csls02-report \
        --datasets "$DS73,$B3,ngnnguynchn/flag2027-csls01-tables-p,ngnnguynchn/flag2027-csls02-tables-p" --env "LAB_FOLDS="
    fi ;;

  86w)   # R1C-01 R1all again once the C5 / T / SA fold tables exist (CPU, minutes): attach the 86g / 86t / 86s outputs
    $K push kaggle/FLAG_86w_fixw01_r1c01.ipynb --account "${ACC:-nguynlcnhtho}" --slug flag-2027-nb-86w-fixw01-r1c01 --force \
      --datasets "$DS73,$TAB" ${KERNELS:+--kernels "$KERNELS"} ;;


  67x)   # VOICELORA-01 step 1b: after FLAG_67a (~01:00 18/10) passes its step-1 gate (train_log.json / log line "step-1 gate"); ~1 h GPU
    need_complete berryyusilee/flag-2027-nb-67a-voicelora
    $K privatize $VL
    if [ "${SMOKE:-0}" = 1 ]; then            # CPU wiring test first (SMOKE=1 bash kaggle/sat_push.sh 67x): ~10 min once started
      $K push kaggle/FLAG_67x_sam100fL_mav.ipynb --account "${ACC:-nguynlcnhtho}" --slug flag-2027-nb-67x-vlx --internet \
        --datasets "$MAVRAW,$VL" --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6'
    else
      $K push kaggle/FLAG_67x_sam100fL_mav.ipynb --account "${ACC:-nguynlcnhtho}" --slug flag-2027-nb-67x-vlx --gpu ${GPUT:+--accelerator $GPUT} --internet \
        --datasets "$MAVRAW,$VL"
    fi ;;

  67xd)  # FLAG_67x output -> private dataset flag2027-feats-vlx-p (small: ~51k x 256)
    k="${ACC:-nguynlcnhtho}/flag-2027-nb-67x-vlx"; need_complete "$k"
    d=$(mktemp -d); $K output "$k" --dest "$d/out" --pattern "feats_vlx/" >/dev/null
    v=$(find "$d/out" -type d -name feats_vlx | head -1); [ -n "$v" ] && [ -f "$v/voice_sam100fL_train.npy" ] || { echo "!! no feats_vlx/voice_sam100fL_train.npy in the output"; exit 1; }
    (cd "$v/.." && python -c "import zipfile,pathlib; z=zipfile.ZipFile('feats_vlx.zip','w'); [z.write(f, str(f)) for f in sorted(pathlib.Path('feats_vlx').rglob('*')) if f.is_file()]; z.close()")
    $K find flag2027-feats-vlx-p
    $K dscreate "$(cygpath -w "$v/../feats_vlx.zip")" --account ngnnguynchn --slug flag2027-feats-vlx-p
    echo "wait 1-3 min, then: $K privatize $VLX" ;;

  97)    # VOICELORA-01 step 2: three fold groups (GPU T4 x2, ~3-3.5 h each), after 67xd; then `97r`
    a=(${ACCS:-nguynlcnhtho nguynchnngn1 trngsnam})
    for i in 0 1 2; do
      f=$(echo "1,2 3,4 5" | cut -d' ' -f$((i+1))); t=$(echo "f12 f34 f5" | cut -d' ' -f$((i+1)))
      $K push kaggle/FLAG_97_voicelora01.ipynb --account "${a[$i]}" --slug "flag-2027-nb-97-voicelora01-$t" --gpu \
        --datasets "$DS73,$B3,$C5,$TAB,$AVF,$VL,$VLX" --env "LAB_FOLDS=$f"
    done ;;

  97r)   # STEP=1: gather the fold tables (dataset); STEP=2: report + final head (5 seeds) + zips on GPU (~1.5 h)
    a=(${ACCS:-nguynlcnhtho nguynchnngn1 trngsnam})
    if [ "${STEP:-1}" = 1 ]; then
      gather_tables voicelora01 flag2027-voicelora01-tables-p "${a[0]}/flag-2027-nb-97-voicelora01-f12" \
        "${a[1]}/flag-2027-nb-97-voicelora01-f34" "${a[2]}/flag-2027-nb-97-voicelora01-f5"
      echo "then: STEP=2 bash kaggle/sat_push.sh 97r"
    else
      $K privatize ngnnguynchn/flag2027-voicelora01-tables-p
      $K push kaggle/FLAG_97_voicelora01.ipynb --account "${ACC:-ngnnguynchn}" --slug flag-2027-nb-97-voicelora01-final --gpu \
        --datasets "$DS73,$B3,$C5,$TAB,$AVF,$VL,$VLX,ngnnguynchn/flag2027-voicelora01-tables-p" --env "LAB_FOLDS="
    fi ;;

  67b)   # VOICELORA-02 step 1: sam100f -> frozen C5 face, + layer3 blocks 16-23 (~5.5 h GPU); SMOKE=1 first (Kaggle CPU)
    D67="$ACACHE,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-feats-vox2b-p,$C5"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_67b_voicelora2.ipynb --account "$VL2ACC" --slug flag-2027-nb-67b-voicelora2-smoke --internet \
        --datasets "$D67" --smoke 'FLAG_SMOKE=1;VL_STEPS_A=3;VL_STEPS_B=3;VL_MAX_CHUNKS=4;VL_EXTRACT_MAX=64'
    else
      $K push kaggle/FLAG_67b_voicelora2.ipynb --account "$VL2ACC" --slug flag-2027-nb-67b-voicelora2 --gpu ${GPUT:+--accelerator $GPUT} --internet \
        --datasets "$D67" --env "VL_UPLOAD=$VL2"
    fi ;;

  67y)   # VOICELORA-02 step 1b: FLAG_67x notebook with FLAG_67b's state (only $VL2 attached, never $VL); ~1 h GPU
    need_complete "$VL2ACC/flag-2027-nb-67b-voicelora2"
    $K privatize $VL2
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_67x_sam100fL_mav.ipynb --account "${ACC:-$VL2ACC}" --slug flag-2027-nb-67y-vlx2-smoke --internet \
        --datasets "$MAVRAW,$VL2" --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6'
    else
      $K push kaggle/FLAG_67x_sam100fL_mav.ipynb --account "${ACC:-$VL2ACC}" --slug ${S67Y:-flag-2027-nb-67y-vlx2-r2} --gpu ${GPUT:+--accelerator $GPUT} --internet \
        --datasets "$MAVRAW,$VL2"
    fi ;;

  67yd)  # FLAG_67y output -> private dataset flag2027-feats-vlx2-p
    k="${ACC:-$VL2ACC}/${S67Y:-flag-2027-nb-67y-vlx2-r2}"; need_complete "$k"
    d=$(mktemp -d); $K output "$k" --dest "$d/out" --pattern "feats_vlx/" >/dev/null
    v=$(find "$d/out" -type d -name feats_vlx | head -1); [ -n "$v" ] && [ -f "$v/voice_sam100fL_train.npy" ] || { echo "!! no feats_vlx/voice_sam100fL_train.npy in the output"; exit 1; }
    grep -rl "VOICELORA-02" "$d/out" >/dev/null || { echo "!! no VOICELORA-02 cfg in the output / kernel log"; exit 1; }
    (cd "$v/.." && python -c "import zipfile,pathlib; z=zipfile.ZipFile('feats_vlx.zip','w'); [z.write(f, str(f)) for f in sorted(pathlib.Path('feats_vlx').rglob('*')) if f.is_file()]; z.close()")
    $K find flag2027-feats-vlx2-p
    $K dscreate "$(cygpath -w "$v/../feats_vlx.zip")" --account ngnnguynchn --slug flag2027-feats-vlx2-p
    echo "wait 1-3 min, then: $K privatize $VLX2" ;;

  67zp)  # VOICELORA-03 step 1b: FLAG_67z notebook (sam100pL of every MAV voice) with FLAG_67c's state (only $VL3 attached); ~1 h GPU
    need_complete "$VL3ACC/flag-2027-nb-67c-voicelora3"
    $K privatize $VL3
    $K push kaggle/FLAG_67z_sam100pL_mav.ipynb --account "${ACC:-$VL3ACC}" --slug ${S67ZP:-flag-2027-nb-67z-vlx3} --gpu ${GPUT:+--accelerator $GPUT} --internet \
      --datasets "$MAVRAW,$VL3" ;;

  67zpd) # FLAG_67z p output -> private dataset flag2027-feats-vlx3-p
    k="${ACC:-$VL3ACC}/${S67ZP:-flag-2027-nb-67z-vlx3}"; need_complete "$k"
    d=$(mktemp -d); $K output "$k" --dest "$d/out" --pattern "feats_vlx3/" | tail -2
    v=$(find "$d/out" -type d -name feats_vlx3 | head -1); [ -n "$v" ] && [ -f "$v/voice_sam100pL_train.npy" ] || { echo "!! no feats_vlx3/voice_sam100pL_train.npy in the output"; exit 1; }
    grep -rl "VOICELORA-03" "$d/out" >/dev/null || { echo "!! no VOICELORA-03 cfg in the output / kernel log"; exit 1; }
    (cd "$v/.." && python -c "import zipfile,pathlib; z=zipfile.ZipFile('feats_vlx3.zip','w'); [z.write(f, str(f)) for f in sorted(pathlib.Path('feats_vlx3').rglob('*')) if f.is_file()]; z.close()")
    $K find flag2027-feats-vlx3-p
    $K dscreate "$(cygpath -w "$v/../feats_vlx3.zip")" --account ngnnguynchn --slug flag2027-feats-vlx3-p
    echo "wait 1-3 min, then: $K privatize $VLX3" ;;

  67zd)  # VOICELORA-04 step 1b: FLAG_67z notebook (sam100fD of every MAV voice) with FLAG_67d's state (the chosen arm); ~1 h GPU
    need_complete "$VL4ACC/flag-2027-nb-67d-voicelora4"
    $K privatize $VL4
    $K push kaggle/FLAG_67z_sam100fD_mav.ipynb --account "${ACC:-$VL4ACC}" --slug ${S67ZD:-flag-2027-nb-67z-vlx4} --gpu ${GPUT:+--accelerator $GPUT} --internet \
      --datasets "$MAVRAW,$VL4" ;;

  67zdd) # FLAG_67z d output -> private dataset flag2027-feats-vlx4-p
    k="${ACC:-$VL4ACC}/${S67ZD:-flag-2027-nb-67z-vlx4}"; need_complete "$k"
    d=$(mktemp -d); $K output "$k" --dest "$d/out" --pattern "feats_vlx4/" | tail -2
    v=$(find "$d/out" -type d -name feats_vlx4 | head -1); [ -n "$v" ] && [ -f "$v/voice_sam100fD_train.npy" ] || { echo "!! no feats_vlx4/voice_sam100fD_train.npy in the output"; exit 1; }
    grep -rl "VOICELORA-04" "$d/out" >/dev/null || { echo "!! no VOICELORA-04 cfg in the output / kernel log"; exit 1; }
    (cd "$v/.." && python -c "import zipfile,pathlib; z=zipfile.ZipFile('feats_vlx4.zip','w'); [z.write(f, str(f)) for f in sorted(pathlib.Path('feats_vlx4').rglob('*')) if f.is_file()]; z.close()")
    $K find flag2027-feats-vlx4-p
    $K dscreate "$(cygpath -w "$v/../feats_vlx4.zip")" --account ngnnguynchn --slug flag2027-feats-vlx4-p
    echo "wait 1-3 min, then: $K privatize $VLX4" ;;

  67zj)  # JOINT-01 step 1b: FLAG_67z notebook (sam100fJ of every MAV voice) with FLAG_65j's state; ~1 h GPU
    need_complete "$JTACC/${S65J2:-flag-2027-nb-65j-joint1-r5}"
    $K privatize $JT
    $K push kaggle/FLAG_67z_sam100fJ_mav.ipynb --account "${ACC:-$JTACC}" --slug ${S67ZJ:-flag-2027-nb-67z-vlxj} --gpu ${GPUT:+--accelerator $GPUT} --internet       --datasets "$MAVRAW,$JT" ;;

  67zjd) # FLAG_67z j output -> private dataset flag2027-feats-vlxj-p
    k="${ACC:-$JTACC}/${S67ZJ:-flag-2027-nb-67z-vlxj}"; need_complete "$k"
    d=$(mktemp -d); $K output "$k" --dest "$d/out" --pattern "feats_vlxj/" | tail -2
    v=$(find "$d/out" -type d -name feats_vlxj | head -1); [ -n "$v" ] && [ -f "$v/voice_sam100fJ_train.npy" ] || { echo "!! no feats_vlxj/voice_sam100fJ_train.npy in the output"; exit 1; }
    grep -rl "JOINT-01" "$d/out" >/dev/null || { echo "!! no JOINT-01 cfg in the output / kernel log"; exit 1; }
    (cd "$v/.." && python -c "import zipfile,pathlib; z=zipfile.ZipFile('feats_vlxj.zip','w'); [z.write(f, str(f)) for f in sorted(pathlib.Path('feats_vlxj').rglob('*')) if f.is_file()]; z.close()")
    $K find flag2027-feats-vlxj-p
    $K dscreate "$(cygpath -w "$v/../feats_vlxj.zip")" --account ngnnguynchn --slug flag2027-feats-vlxj-p
    echo "wait 1-3 min, then: $K privatize $VLXJ" ;;

  # EXTRA99 per group (run.py finds adapters / extras by rglob under /kaggle/input, so only the datasets must be attached):
  #   G99=vl4  EXTRA99="$VL4,$VLX4"                  (VL4: fD aux voices [manifest exp VOICELORA-04]; VLX4: fD of MAV [feats-vlx4])
  #   G99=jt   EXTRA99="$JT,$VLXJ,ngnnguynchn/flag2027-flag99-vl3co-tables-p"
  #            (JT: face_clipJT_* + fJ aux voices [manifest exp JOINT-01]; VLXJ: fJ of MAV [feats-vlxj]; vl3co tables: trOR reference column, made by `G99=vl3,co STEP=1 99r`)
  #   e.g.  G99=jt EXTRA99="$JT,$VLXJ,ngnnguynchn/flag2027-flag99-vl3co-tables-p" bash kaggle/sat_push.sh 99   (same EXTRA99 for 99r)
  99s)   # FLAG_99 smoke (Kaggle CPU, fold 1 + stand-ins) for group G99 before the GPU folds; ACC required
    G=${G99:-vl3,co}; s0=flag-2027-nb-99-$(echo "$G" | tr -d ,)
    $K push kaggle/FLAG_99_vl3co.ipynb --account "${ACC:?set ACC=<account>}" --slug "${S99S:-$s0}" --datasets "$D99${EXTRA99:+,$EXTRA99}" \
      --smoke "F99_GROUPS=$G;FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;VL2_STANDIN=1;VL2_FINAL_SEEDS=1" ;;
  99)    # FLAG_99 step 2 (FLAG99-S2/run.py), groups G99 (default vl3,co): three fold groups (GPU T4 x2, ~5 h each); then `99r`
    G=${G99:-vl3,co}; t0=flag99_$(echo "$G" | tr -d ,); nb=kaggle/FLAG_99_vl3co.ipynb; s0=flag-2027-nb-99-$(echo "$G" | tr -d ,)
    a=(${ACCS:-trngminhhieu nammo0 hieuhau})
    for i in 0 1 2; do
      f=$(echo "1,2 3,4 5" | cut -d' ' -f$((i+1))); t=$(echo "f12 f34 f5" | cut -d' ' -f$((i+1)))
      $K push $nb --account "${a[$i]}" --slug "$s0-$t" --gpu --datasets "$D99${EXTRA99:+,$EXTRA99}" --env "LAB_FOLDS=$f;F99_GROUPS=$G"
    done ;;

  99r)   # STEP=1: gather the fold tables (dataset); STEP=2: report + final heads (5 seeds) + zips on GPU
    G=${G99:-vl3,co}; t0=flag99_$(echo "$G" | tr -d ,); s0=flag-2027-nb-99-$(echo "$G" | tr -d ,); ds=flag2027-$(echo "$t0" | tr _ -)-tables-p
    a=(${ACCS:-trngminhhieu nammo0 hieuhau})
    if [ "${STEP:-1}" = 1 ]; then
      gather_tables "$t0" "$ds" "${a[0]}/${S12:-$s0-f12}" "${a[1]}/${S34:-$s0-f34}" "${a[2]}/${S5:-$s0-f5}"   # S12 / S34 / S5: re-pushed slug (§6d)
      echo "then: STEP=2 bash kaggle/sat_push.sh 99r"
    else
      $K privatize ngnnguynchn/$ds
      $K push kaggle/FLAG_99_vl3co.ipynb --account "${ACC:-hthlin}" --slug "$s0-final" --gpu \
        --datasets "$D99${EXTRA99:+,$EXTRA99},ngnnguynchn/$ds" --env "LAB_FOLDS=;F99_GROUPS=$G"
    fi ;;

  101)   # DABL-01 stage 1 (DSC = DS73 without the AVSpeech / LRS3 features, rule 28/10): SMOKE=1 first (Kaggle CPU); then 20 GPU runs = 4 blocks x 5 folds (~2.3 h each), one account each
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_101_dabl01.ipynb --account "${ACC:-nguyentruonglong}" --slug flag-2027-nb-101-dabl01-smoke         --datasets "$DSC,$B3,$TAB" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;DABL_BLOCK=0;LAB_FOLDS=1'
    else
      a=(${ACCS:?ACCS = 20 accounts, block-major: b0f1 b0f2 ... b3f5})
      i=0
      for b in 0 1 2 3; do for f in 1 2 3 4 5; do
        $K push kaggle/FLAG_101_dabl01.ipynb --account "${a[$i]}" --slug "flag-2027-nb-101-dabl01-b${b}f${f}" --gpu           --datasets "$DSC,$B3,$TAB" --env "DABL_BLOCK=$b;LAB_FOLDS=$f"
        i=$((i+1))
      done; done
    fi ;;

  101r)  # DABL-01 stage 1 report: STEP=1 the 20 block tables -> dataset; STEP=2 privatize + merge + report on CPU (~1.5 h)
    if [ "${STEP:-1}" = 1 ]; then
      ks=(${KS101:-trngsnam/flag-2027-nb-101-dabl01-b0f2 nguynlcnhtho/flag-2027-nb-101-dabl01-b0f3 nguyentruonglong/flag-2027-nb-101-dabl01-b0f5
          nguynchnngn1/flag-2027-nb-101-dabl01-b1f2 berryyusilee/flag-2027-nb-101-dabl01-b1f3 chnnhnnguyn/flag-2027-nb-101-dabl01-b2f5
          hthlin/flag-2027-nb-101-dabl01-b3f4 nguynlcnhtho/flag-2027-nb-101-dabl01-b1f5 nguynchnngn1/flag-2027-nb-101-dabl01-b2f34
          berryyusilee/flag-2027-nb-101-dabl01-b3f12 chnnhnnguyn/flag-2027-nb-101-dabl01-b3f3 thanhthaoph/flag-2027-nb-101-dabl01-b3f5
          duongphihoang/flag-2027-nb-101-dabl01-b2f1 nguynchnngn/flag-2027-nb-101-dabl01-b2f2 trngminhhieu/flag-2027-nb-101-dabl01-b0f1
          dnq1209/flag-2027-nb-101-dabl01-b0f4 hieuhau/flag-2027-nb-101-dabl01-b1f1 locnguyentruongan/flag-2027-nb-101-dabl01-b1f4})   # 29/10 05:29 placement (RUNS.md)
      NTAB=20 gather_tables "dabl01b[0-9]" flag2027-dabl01-tables-p "${ks[@]}"
      echo "then: STEP=2 bash kaggle/sat_push.sh 101r"
    else
      $K privatize ngnnguynchn/flag2027-dabl01-tables-p
      $K push kaggle/FLAG_101_dabl01.ipynb --account "${ACC:-nguyentruonglong}" --slug flag-2027-nb-101-dabl01-report         --datasets "$DSC,$B3,$TAB,ngnnguynchn/flag2027-dabl01-tables-p" --env "LAB_FOLDS="
    fi ;;

  104)   # TAP-01 step 2 (FLAG_104, DSC + the FLAG_103a / 103c tap datasets): ACCS = "f12 f34 f5" accounts (~5-6 / 5-6 / 3 h GPU)
    a=(${ACCS:?ACCS = 3 accounts: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
    TAPDS="thanhthaoph/flag2027-feats-tap01-face-p,dnq1209/flag2027-feats-tap01-voice-p"
    for i in 0 1 2; do
      $K push kaggle/FLAG_104_tap01.ipynb --account "${a[$i]}" --slug "flag-2027-nb-104-tap01-${s[$i]}" --gpu \
        --datasets "$DSC,$B3,$TAB,$TAPDS" --env "LAB_FOLDS=${f[$i]}"
    done ;;
  104r)  # TAP-01: STEP=1 fold tables (OWN="acc_f12 acc_f34 acc_f5") -> dataset; STEP=2 report on CPU
    if [ "${STEP:-1}" = 1 ]; then
      o=(${OWN:?OWN = the 3 accounts of the f12 f34 f5 runs})
      gather_tables tap01 flag2027-tap01-tables-p "${o[0]}/flag-2027-nb-104-tap01-f12" "${o[1]}/flag-2027-nb-104-tap01-f34" "${o[2]}/flag-2027-nb-104-tap01-f5"
      echo "then: STEP=2 bash kaggle/sat_push.sh 104r"
    else
      $K privatize ngnnguynchn/flag2027-tap01-tables-p
      $K push kaggle/FLAG_104_tap01.ipynb --account "${ACC:-nguynchnngn}" --slug flag-2027-nb-104-tap01-report \
        --datasets "$DSC,$B3,$TAB,thanhthaoph/flag2027-feats-tap01-face-p,dnq1209/flag2027-feats-tap01-voice-p,ngnnguynchn/flag2027-tap01-tables-p" --env "LAB_FOLDS="
    fi ;;
  105tab)  # unsubmit (29/10): PROJ-01's 5 fold tables (in its report kernel) -> private dataset (other accounts cannot read the kernel)
    gather_tables proj01 flag2027-proj01-tables-p ngnnguynchn/flag-2027-nb-96-proj01-report ;;
  105)   # unsubmit (29/10): final head (5 seeds) + dev zip of HARDNEG-AGE-01 B2a / SGW-01 W1 / PROJ-01 OR (GPU ~1.5 h each)
    a=(${A105:?A105 = 3 accounts: hna01 sgw01 proj01}); BASE="$DSC,$B3,$TAB,$AVF"; only="${ONLY105:-hna01 sgw01 proj01}"
    case " $only " in *" hna01 "*) $K push kaggle/FLAG_105_hna01_final.ipynb --account "${a[0]}" --slug flag-2027-nb-105-hna01-final --gpu \
      --datasets "$BASE,ngnnguynchn/flag2027-hna01-tables-p" --env "FINAL_ARM=B2a" ;; esac
    case " $only " in *" sgw01 "*) $K push kaggle/FLAG_105_sgw01_final.ipynb --account "${a[1]}" --slug flag-2027-nb-105-sgw01-final --gpu \
      --datasets "$BASE,ngnnguynchn/flag2027-sgw01-tables-p" --env "FINAL_ARM=W1" ;; esac
    case " $only " in *" proj01 "*) $K push kaggle/FLAG_105_proj01_final.ipynb --account "${a[2]}" --slug flag-2027-nb-105-proj01-final --gpu \
      --datasets "$BASE,ngnnguynchn/flag2027-proj01-tables-p" --env "FINAL_ARM=OR" ;; esac ;;
  110)   # CLEANTUNE-01 round 1 (FLAG_110, DSC + FLAG_109 outputs): CTG (default "g1 g2 g3"), ACCS = 3 accounts per group
         # in order (f12 f34 f5 of the first group, then of the next); g1 / g2 need CTB, g3 also CTC
         # SPLIT=5 (31/10): one fold per run (f1 ... f5), ACCS = 5 accounts per group; 110r takes any number of runs (15 tables)
         # CLEANTUNE-02 (3/11): LS=1 CTB=<B3s> CTC=<C5s> SLUGP=flag-2027-nb-154-ct02; SMOKE=1 = one CPU wiring run per group
    a=(${ACCS:?ACCS = 3 accounts per group: f12 f34 f5 (SPLIT=5: 5 accounts, f1 ... f5)})
    G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"
    if [ "${SPLIT:-3}" = 5 ]; then f=(1 2 3 4 5); s=(f1 f2 f3 f4 f5); else f=("1,2" "3,4" "5"); s=(f12 f34 f5); fi; i=0
    for g in ${CTG:-g1 g2 g3}; do   # not GROUPS: a read-only bash array (the user's group ids)
      x=""; [ "$g" = g3 ] && x=",$G3X"
      ds="$DSC,$CTB,$TAB,$AVF$x${LS:+,ngnnguynchn/flag2027-lid-strict-p}"   # LS=1: CLEANTUNE-02 (LID_STRICT=1; CTB / CTC = B3s / C5s)
      if [ "${SMOKE:-0}" = 1 ]; then   # Kaggle CPU wiring test of group g: fold 1, 2 steps, stand-in adapters / voice blocks
        $K push kaggle/FLAG_110_ct01_$g.ipynb --account "${a[$i]}" --slug "${SLUGP:-flag-2027-nb-110-ct01}-$g-smoke" --datasets "$ds" \
          --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1${LS:+;LID_STRICT=1}${QE-;T3R_QF=1;T3R_EMA=0.9995}"
        i=$((i + 1)); continue
      fi
      for j in $(seq 0 $((${#f[@]} - 1))); do
        $K push kaggle/FLAG_110_ct01_$g.ipynb --account "${a[$i]}" --slug "${SLUGP:-flag-2027-nb-110-ct01}-$g-${s[$j]}" --gpu \
          --datasets "$ds" --env "LAB_FOLDS=${f[$j]}${LS:+;LID_STRICT=1}${QE-;T3R_QF=1;T3R_EMA=0.9995}"   # AVF: OPT makes feats_av_full required (F0 English fit)
        i=$((i + 1))
      done
    done ;;
  110r)  # CLEANTUNE-01: STEP=1 the 15 fold tables of the 9 FLAG_110 runs (RUNS = "owner/kernel ...") -> $CTT; STEP=2 report (CPU)
    if [ "${STEP:-1}" = 1 ]; then
      d=$(mktemp -d); mkdir -p "$d/t"
      for k in ${RUNS:?RUNS = the 9 owner/kernel of FLAG_110}; do
        need_complete "$k"
        $K output "$k" --dest "$d/$(basename "$k")" --pattern "ct01g[123]_fold[0-9]\.parquet" >/dev/null
        find "$d/$(basename "$k")" -name "ct01g*_fold*.parquet" -exec cp {} "$d/t/" \;
      done
      ls "$d/t"; n=$(ls "$d/t" | wc -l); [ "$n" -eq 15 ] || { echo "!! $n fold tables (15 needed)"; exit 1; }
      (cd "$d" && python -c "import zipfile,pathlib; z=zipfile.ZipFile('ct01_tables.zip','w'); [z.write(f, 'tables/'+f.name) for f in sorted(pathlib.Path('t').glob('*.parquet'))]; z.close()")
      $K dscreate "$(cygpath -w "$d/ct01_tables.zip" 2>/dev/null || echo "$d/ct01_tables.zip")" --account ngnnguynchn --slug "${CTT#*/}"
      echo "wait until the dataset is ready (1-3 min), then: STEP=2 bash kaggle/sat_push.sh 110r"
    else
      $K privatize "$CTT"
      $K push kaggle/FLAG_110r_ct01_report.ipynb --account "${ACC:-nguynchnngn}" --slug flag-2027-nb-110r-ct01-report \
        --datasets "$DSC,$TAB,$CTB,$CTT,$AVF" --env "LAB_FOLDS="   # AVF: feats_av_full is required by OPT
    fi ;;
  156)   # FACENEG-01 (FLAG_156, 3/11): CA + 8 bank arms, CLEANTUNE-01 g3 framework, LID_STRICT=1; CTB / CTC = B3s / C5s;
         # ACCS = 3 accounts f12 f34 f5 (SMOKE=1: one Kaggle CPU wiring run, stand-in adapters / voice blocks)
    a=(${ACCS:?ACCS = 3 accounts: f12 f34 f5 (SMOKE=1: one)}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
    ds="$DSC,$CTB,$TAB,$AVF,$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF,ngnnguynchn/flag2027-lid-strict-p"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_156_faceneg01.ipynb --account "${a[0]}" --slug flag-2027-nb-156-faceneg01-smoke --datasets "$ds" \
        --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1;LID_STRICT=1${QE-;T3R_QF=1;T3R_EMA=0.9995}"
    else
      for j in 0 1 2; do
        $K push kaggle/FLAG_156_faceneg01.ipynb --account "${a[$j]}" --slug "flag-2027-nb-156-faceneg01-${s[$j]}" --gpu \
          --datasets "$ds" --env "LAB_FOLDS=${f[$j]};LID_STRICT=1${QE-;T3R_QF=1;T3R_EMA=0.9995}"
      done
    fi ;;
  111)   # CLEANTUNE-01 round 2 (FLAG_111, built from ct_round2.json): SMOKE=1 (Kaggle CPU) | GPU: ACCS = 3 accounts f12 f34 f5 |
         # STEP=r1: RUNS = the 3 GPU runs -> $CT2 tables dataset; STEP=r2: CPU report (LAB_FOLDS=) -> ct_decision.json
         # G3=1 when I* != B3c (build_flag111 prints "framework g3"): the c5c / tap / screen2 / aux-voice arrays are attached
    CT2=${CT2:-ngnnguynchn/flag2027-ct01r2-tables-p}
    x=""; [ "${G3:-0}" = 1 ] && x=",$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_111_ct01_r2.ipynb --account "${ACC:-nguyentruonglong}" --slug flag-2027-nb-111-ct01-r2-smoke \
        --datasets "$DSC,$CTB,$TAB,$AVF$x" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1'
    elif [ "${STEP:-}" = r1 ]; then
      gather_tables ct01r2 "${CT2#*/}" ${RUNS:?RUNS = the 3 FLAG_111 GPU runs}
    elif [ "${STEP:-}" = r2 ]; then
      $K privatize "$CT2"
      $K push kaggle/FLAG_111_ct01_r2.ipynb --account "${ACC:-nguynchnngn}" --slug flag-2027-nb-111r-ct01-report \
        --datasets "$DSC,$CTB,$TAB,$AVF$x,$CT2" --env "LAB_FOLDS="
    else
      a=(${ACCS:?ACCS = 3 accounts: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
      for i in 0 1 2; do
        $K push kaggle/FLAG_111_ct01_r2.ipynb --account "${a[$i]}" --slug "flag-2027-nb-111-ct01-r2-${s[$i]}" --gpu \
          --datasets "$DSC,$CTB,$TAB,$AVF$x" --env "LAB_FOLDS=${f[$i]}"
      done
    fi ;;
  110g1) # CLEANTUNE-01 (31/10): the 5 ct01g1 fold tables of G1 (RUNS = its owner/kernel list) -> dataset for F0, before G2 / G3 end
    gather_tables ct01g1 flag2027-ct01g1-tables-p ${RUNS:?RUNS = the G1 owner/kernel list} ;;
  121)   # ENSEL-01 + COMPL-01 (FLAG_121, CPU, 1/11): merge round-1 tables ($CTT) -> English re-selection + zip, Bangla complement screen
    $K push kaggle/FLAG_121_ensel_compl.ipynb --account "${ACC:-nguynchnngn}" --slug "${SLUG:-flag-2027-nb-121-ensel-compl}" \
      --datasets "$DSC,$TAB,$CTB,$CTT,$AVF" --env "LAB_FOLDS=" ;;
  128)   # FUSEML-01 (FLAG_128, CPU, 1/11): merge round-1 tables ($CTT) -> constrained fusers vs CA-FUSE
    $K push kaggle/FLAG_128_fuseml.ipynb --account "${ACC:-nguynchnngn}" --slug "${SLUG:-flag-2027-nb-128-fuseml}" \
      --datasets "$DSC,$TAB,$CTB,$CTT,$AVF" --env "LAB_FOLDS=" ;;
  122)   # BASEGAP-01 (FLAG_122, CPU, 1/11): FOP baseline gap; one job per fold, ACCS = 5 accounts (folds 0..4); SMOKE=1 first
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_122_basegap01.ipynb --account "${ACC:-berryyusilee}" --slug flag-2027-nb-122-basegap01 \
        --datasets trngsnam/flag-input --smoke 'BG_SMOKE=1'
    else
      a=(${ACCS:?ACCS = 5 accounts, one per fold 0..4})
      for k in 0 1 2 3 4; do
        $K push kaggle/FLAG_122_basegap01.ipynb --account "${a[$k]}" --slug "flag-2027-nb-122-basegap01-f$k" \
          --datasets trngsnam/flag-input --env "BG_FOLDS=$k"
      done
    fi ;;
  135)   # BACKTEST-01 (FLAG_135, CPU, 2/11): validation measures vs the CB direction of past English decisions; inputs as 123
    if [ "${SMOKE:-0}" = 1 ]; then x=(--smoke "FL3Z_SMOKE=1"); else x=(--env "LAB_FOLDS="); fi
    $K push kaggle/FLAG_135_backtest01.ipynb --account "${ACC:-trngsnam}" --slug "${SLUG:-flag-2027-nb-135-backtest01}" \
      --datasets "$DSC,$TAB,$CTB,$CTT,$AVF" "${x[@]}" ;;
  132)   # UTTSTAB-01 (FLAG_132, GPU + Internet, 1/11): utterance-stability arms vs in-run B0c; SMOKE=1 first (CPU), then ACCS = f12 f34 f5
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_132_uttstab01.ipynb --account "${ACC:?ACC = the smoke account}" --slug "${SLUG:-flag-2027-nb-132-uttstab01-smoke}" \
        --internet --datasets "$DSC,$CTB,$TAB,$AVF,$ACACHE" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;UTT_AUX_MAX=256'
    else
      a=(${ACCS:?ACCS = 3 accounts with GPU quota: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
      for j in 0 1 2; do
        $K push kaggle/FLAG_132_uttstab01.ipynb --account "${a[$j]}" --slug "flag-2027-nb-132-uttstab01-${s[$j]}" --gpu --internet \
          --datasets "$DSC,$CTB,$TAB,$AVF,$ACACHE" --env "LAB_FOLDS=${f[$j]}"
      done
    fi ;;
  141)   # SESSPROJ-01 (FLAG_141, GPU, 2/11): 5 folds of B0c + within-person projections, gate in the notebook;
         # SMOKE=1 first (CPU, ACC: fold 1 + B3 stand-ins for folds 2-5), then ACC = one account with GPU quota
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_141_sessproj01.ipynb --account "${ACC:?ACC = the smoke account}" --slug "${SLUG:-flag-2027-nb-141-sessproj01}" \
        --datasets "$DSC,$CTB,$TAB,$AVF" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1"
    else
      $K push kaggle/FLAG_141_sessproj01.ipynb --account "${ACC:?ACC = one account with GPU quota}" --slug "${SLUG:-flag-2027-nb-141-sessproj01}" \
        --gpu --datasets "$DSC,$CTB,$TAB,$AVF"
    fi ;;
  136)   # KFOLD-01 (FLAG_136, GPU, 2/11): 256 jobs (scheme, partition, fold) over 19 notebooks, job j -> notebook (j - 1) mod 19;
         # SMOKE=1 first (CPU, ACC + JOB: 28 = MC5 r0 fold 1 wiring check, 27 = K10 r0 fold 10), then ACCS = 19 accounts
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_136_kfold01.ipynb --account "${ACC:?ACC = the smoke account}" --slug "${SLUG:-flag-2027-nb-136-kfold01-j${JOB:-28}}" \
        --datasets "$DSC,$CTB,$TAB,$AVF" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=${JOB:-28}"
    else
      a=(${ACCS:?ACCS = 19 accounts with GPU quota}); [ ${#a[@]} -eq 19 ] || { echo "!! ${#a[@]} accounts (19 needed)"; exit 1; }
      for i in $(seq 0 18); do
        f=$(python -c "print(','.join(str(j) for j in range(1, 257) if (j - 1) % 19 == $i))")
        $K push kaggle/FLAG_136_kfold01.ipynb --account "${a[$i]}" --slug "flag-2027-nb-136-kfold01-n$((i + 1))" --gpu \
          --datasets "$DSC,$CTB,$TAB,$AVF" --env "LAB_FOLDS=$f"
      done
    fi ;;
  136x)  # KFOLD-01: re-run of the jobs a notebook skipped at its time budget (JOBS = comma list, e.g. 254,255), one GPU run (2/11)
    $K push kaggle/FLAG_136_kfold01.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-136-kfold01-rerun}" --gpu --datasets "$DSC,$CTB,$TAB,$AVF" --env "LAB_FOLDS=${JOBS:?JOBS}" ;;
  133)   # CFGOPT-01 (FLAG_133, GPU, 1/11): two groups of 8 Sobol configs vs in-run B0c; SMOKE=1 first (CPU, group a), then
         # ACCS = 6 accounts: a-f12 a-f34 a-f5 b-f12 b-f34 b-f5
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_133_cfgopt01_a.ipynb --account "${ACC:?ACC = the smoke account}" --slug "${SLUG:-flag-2027-nb-133-cfgopt01-smoke}" \
        --datasets "$DSC,$CTB,$TAB,$AVF" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1'
    else
      a=(${ACCS:?ACCS = 6 accounts with GPU quota}); f=("1,2" "3,4" "5"); s=(f12 f34 f5); j=0
      for g in a b; do
        for i in 0 1 2; do
          $K push "kaggle/FLAG_133_cfgopt01_$g.ipynb" --account "${a[$j]}" --slug "flag-2027-nb-133-cfgopt01$g-${s[$i]}" --gpu \
            --datasets "$DSC,$CTB,$TAB,$AVF" --env "LAB_FOLDS=${f[$i]}"
          j=$((j + 1))
        done
      done
    fi ;;
  134g)  # FLAG_134 inputs (1/11): fold tables of UTTSTAB-01 (RUNS132 = the 3 FLAG_132 kernels) and / or CFGOPT-01 (RUNS133A /
         # RUNS133B = the 3 kernels of group a / b) -> one private dataset per table tag; set the RUNS of what has finished
    [ -n "${RUNS132:-}" ] && gather_tables uttstab01 flag2027-uttstab01-tables-p $RUNS132
    [ -n "${RUNS133A:-}" ] && gather_tables cfgopt01a flag2027-cfgopt01a-tables-p $RUNS133A
    [ -n "${RUNS133B:-}" ] && gather_tables cfgopt01b flag2027-cfgopt01b-tables-p $RUNS133B
    true ;;
  134)   # UTTSTAB-01 / CFGOPT-01 reports (FLAG_134, CPU, 1/11); SMOKE=1 = stand-in tables, else the datasets of step 134g
         # (TABLES = the table tags whose dataset exists; the notebook reports every experiment with all its tables)
    if [ "${SMOKE:-0}" = 1 ]; then x=(--smoke "FL3Z_SMOKE=1"); t=""; else x=(--env "LAB_FOLDS="); t=""
      for g in ${TABLES:-uttstab01 cfgopt01a cfgopt01b}; do t="$t,ngnnguynchn/flag2027-$g-tables-p"; done; fi
    $K push kaggle/FLAG_134_reports.ipynb --account "${ACC:-trngsnam}" --slug "${SLUG:-flag-2027-nb-134-reports}" \
      --datasets "$DSC,$TAB,$CTB,$AVF$t" "${x[@]}" ;;
  139)   # EDA-F1RISK (FLAG_139, CPU, 2/11) on CLEANTUNE-01's round-2 tables; SMOKE=1 = stand-in tables
    if [ "${SMOKE:-0}" = 1 ]; then x=(--smoke "FL3Z_SMOKE=1"); t=""; else x=(--env "LAB_FOLDS="); t=",ngnnguynchn/flag2027-ct01r2-tables-p"; fi
    $K push kaggle/FLAG_139_f1risk.ipynb --account "${ACC:-nguynchnngn}" --slug "${SLUG:-flag-2027-nb-139-f1risk}" \
      --datasets "$DSC,$TAB,$CTB,$AVF$t" "${x[@]}" ;;
  127g)  # FLAG_127 inputs (1/11): the fold tables of CCPROP-01 (RUNS125 = the FLAG_125 kernels) and / or SHORT-03 (RUNS124 = the
         # FLAG_124 kernels) -> one private dataset per experiment; set only the RUNS of the experiment that has finished
    [ -n "${RUNS125:-}" ] && gather_tables ccprop01 flag2027-ccprop01-tables-p $RUNS125
    [ -n "${RUNS124:-}" ] && gather_tables short03 flag2027-short03-tables-p $RUNS124
    [ -n "${RUNS138:-}" ] && gather_tables spkf01 flag2027-spkf01-tables-p $RUNS138   # 2/11: SPKF-01 (FLAG_138 f12 / f34 / f5)
    [ -n "${RUNS145:-}" ] && gather_tables uncscore01 flag2027-uncscore01-tables-p $RUNS145   # 2/11: UNCSCORE-01 (FLAG_145 f12 / f34 / f5)
    true ;;
  127)   # CCPROP-01 / SHORT-03 reports (FLAG_127, CPU, 1/11); SMOKE=1 = stand-in tables, else the datasets of step 127g
         # (TABLES = the tag(s) whose dataset exists, default both; the notebook reports every experiment with 5 tables)
    if [ "${SMOKE:-0}" = 1 ]; then x=(--smoke "FL3Z_SMOKE=1"); t=""; else x=(--env "LAB_FOLDS="); t=""
      for g in ${TABLES:-ccprop01 short03}; do t="$t,ngnnguynchn/flag2027-$g-tables-p"; done; [ -n "${TABDS:-}" ] && t=",$TABDS"; fi   # TABDS: explicit dataset list (XENC / MLLM upload their own)
    $K push kaggle/FLAG_127_reports.ipynb --account "${ACC:-trngsnam}" --slug "${SLUG:-flag-2027-nb-127-reports}" \
      --datasets "$DSC,$TAB,$CTB,$AVF$t" "${x[@]}" ;;
  123)   # ENABL-01 (FLAG_123, CPU, 1/11): English stream ablations with full AudioCLIP + H-ANCHOR; inputs as 121; SMOKE=1 = stand-in tables
    if [ "${SMOKE:-0}" = 1 ]; then x=(--smoke "FL3Z_SMOKE=1"); else x=(--env "LAB_FOLDS="); fi
    $K push kaggle/FLAG_123_enabl01.ipynb --account "${ACC:-trngsnam}" --slug "${SLUG:-flag-2027-nb-123-enabl01}" \
      --datasets "$DSC,$TAB,$CTB,$CTT,$AVF" "${x[@]}" ;;
  124)   # SHORT-03 (FLAG_124, GPU + Internet, 1/11): CC head B0c on full / 4 s / 2 s voices; SMOKE=1 first (CPU), then ACCS = f12 f34 f5
    MAV="chnnhnnguyn/external-data-mavceleb,ngnnguynchn/flag2027-ext-meta-p,trngsnam/flag-input"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_124_short03.ipynb --account "${ACC:-hieuhau}" --slug "${SLUG:-flag-2027-nb-124-short03}" --internet \
        --datasets "$DSC,$CTB,$TAB,$AVF,$MAV" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1'
    else
      a=(${ACCS:?ACCS = 3 accounts with GPU quota: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
      for j in 0 1 2; do
        $K push kaggle/FLAG_124_short03.ipynb --account "${a[$j]}" --slug "flag-2027-nb-124-short03-${s[$j]}" --gpu --internet \
          --datasets "$DSC,$CTB,$TAB,$AVF,$MAV" --env "LAB_FOLDS=${f[$j]}"
      done
    fi ;;
  126)   # GEO3D-01 gate 1 (FLAG_126, CPU + Internet, 1/11): 3DDFA_V2 shape vs T3 voice; SMOKE=1 first
    if [ "${SMOKE:-0}" = 1 ]; then x=(--smoke "GEO_SMOKE=1"); else x=(); fi
    $K push kaggle/FLAG_126_geo3d01.ipynb --account "${ACC:-trngminhhieu}" --slug "${SLUG:-flag-2027-nb-126-geo3d01}" --internet \
      --datasets "ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-eda-bundle-p,ngnnguynchn/flag2027-ext-meta-p,chnnhnnguyn/external-data-mavceleb,trngsnam/flag-input" \
      "${x[@]}" ;;
  137)   # SPKF-01 step 1 (FLAG_137, GPU + Internet, 2/11): SpeakingFaces rows (HF range reads) -> SPKF_UPLOAD; SMOKE=1 first (CPU)
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_137_spkf01_feats.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-137-spkf01-smoke}" --internet \
        --datasets "$CTB" --smoke 'FLAG_SMOKE=1;SF_SMOKE_SUBS=3;SF_SMOKE_CMDS=6'
    else
      $K push kaggle/FLAG_137_spkf01_feats.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-137-spkf01}" --gpu --internet \
        --datasets "$CTB" --secret HF_TOKEN --env "SPKF_UPLOAD=${ACC}/flag2027-feats-spkf-p;AVS_THREADS=8"
    fi ;;
  138)   # SPKF-01 step 2 (FLAG_138, GPU, 2/11): sf aux source w .3 / 1 / 3 vs in-run B0c; SMOKE=1 first (CPU, stand-in sf rows unless
         # SPK = the FLAG_137 dataset exists), then ACCS = f12 f34 f5 with SPK set
    SPK=${SPK:-}
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_138_spkf01.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-138-spkf01-smoke}" \
        --datasets "$DSC,$CTB,$TAB,$AVF${SPK:+,$SPK}" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;SPKF_STANDIN=1'
    else
      a=(${ACCS:?ACCS = 3 accounts with GPU quota: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
      for j in 0 1 2; do
        $K push kaggle/FLAG_138_spkf01.ipynb --account "${a[$j]}" --slug "flag-2027-nb-138-spkf01-${s[$j]}" --gpu \
          --datasets "$DSC,$CTB,$TAB,$AVF,${SPK:?SPK = owner/flag2027-feats-spkf-p}" --env "LAB_FOLDS=${f[$j]}"
      done
    fi ;;
  125)   # CCPROP-01 (FLAG_125, GPU, 1/11): SUBCENTER-01 + GENDHEAD-01 vs in-run B0c; SMOKE=1 first (CPU), then ACCS = f12 f34 f5
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_125_ccprop01.ipynb --account "${ACC:-hieuhau}" --slug flag-2027-nb-125-ccprop01 \
        --datasets "$DSC,$CTB,$TAB,$AVF" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1'
    else
      a=(${ACCS:?ACCS = 3 accounts with GPU quota: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
      for j in 0 1 2; do
        $K push kaggle/FLAG_125_ccprop01.ipynb --account "${a[$j]}" --slug "flag-2027-nb-125-ccprop01-${s[$j]}" --gpu \
          --datasets "$DSC,$CTB,$TAB,$AVF" --env "LAB_FOLDS=${f[$j]}"
      done
    fi ;;
  112f1) # CLEANTUNE-01 F1 (FLAG_112_f1, 1/11): round-2 arm of ct_decision.json (CA = C5CAT4 + T + AIp, 10 seeds) on the ct01r2
         # tables + g3 arrays; English bytes from the F0 zip = output of nguynchnngn/flag-2027-nb-112-f0 (--kernels: same account)
    CT2=${CT2:-ngnnguynchn/flag2027-ct01r2-tables-p}; x=",$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_112_f1.ipynb --account "${ACC:-nguynchnngn}" --slug flag-2027-nb-112-f1-smoke --kernels nguynchnngn/flag-2027-nb-112-f0 \
        --datasets "$DSC,$CTB,$TAB,$CT2,$AVF$x" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;CT_FINAL_SEEDS=1;BN42_STANDIN=1'
    else
      $K push kaggle/FLAG_112_f1.ipynb --account "${ACC:-nguynchnngn}" --slug flag-2027-nb-112-f1 --gpu --kernels nguynchnngn/flag-2027-nb-112-f0 \
        --datasets "$DSC,$CTB,$TAB,$CT2,$AVF$x"
    fi ;;
  65xb|65xc|47r) # CLEANTUNE-01 Evaluation path (2/11, docs/PLAN_EVAL_F0F1_2026-11-02.md), one account for the whole chain (kernel
         # outputs are read by FLAG_112x on the same account): saved B3c / C5c adapter -> faces of the test lists | FLAG_47b re-run ->
         # rn293 / ecapa1024 of every MAV row. Dev = R11; on 11/11 add the Evaluation data (EVALDS) and set ONLY_UNDER in the first cell
    D65="$MAVRAW,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-feats-vox2b-p,onnhqunhh/flag2027-lora-frames-p"
    case "$STEP_" in
      65xb) nb=FLAG_65xb_clipb3c_extract.ipynb; s=flag-2027-nb-65xb-b3cx; ds="$D65,$CTB" ;;
      65xc) nb=FLAG_65xc_clipc5c_extract.ipynb; s=flag-2027-nb-65xc-c5cx; ds="$D65,$CTC" ;;
      47r)  nb=FLAG_47b_voice_screen.ipynb; s=flag-2027-nb-47r-screen2; ds="$MAVRAW" ;;
    esac
    ds="$ds${EVALDS:+,$EVALDS}"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/$nb --account "${ACC:-nguynchnngn}" --slug "$s-smoke" --internet --datasets "$ds" --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6'
    else
      $K push kaggle/$nb --account "${ACC:-nguynchnngn}" --slug "${SLUG:-$s-r11}" --gpu --internet --datasets "$ds"
    fi ;;
  112f0x|112f1x) # CLEANTUNE-01 Evaluation path (2/11): saved F0 / F1 heads (no training) on the split's test files + zip; dev = R11
         # (compared with the f0 / f1 zip). KX = the kernels: f0x "<112-f0> <65xb>", f1x "<112-f1> <112-f0x> <65xb> <65xc> <47r>";
         # no saved b3c / c5c / screen2 datasets (the re-extracted outputs replace them); on 11/11 + FLAG_50's output, E2E_SPLIT=eval
    a=${ACC:-nguynchnngn}
    if [ "$STEP_" = 112f0x ]; then nb=FLAG_112_f0x.ipynb; s=flag-2027-nb-112-f0x; ds="$DSC,$TAB,${CTT:-ngnnguynchn/flag2027-ct01g1-tables-p},$AVF"
    else nb=FLAG_112_f1x.ipynb; s=flag-2027-nb-112-f1x; ds="$DSC,$TAB,ngnnguynchn/flag2027-ct01r2-tables-p,$AVF,dnq1209/flag2027-feats-tap01-voice-p,$VF"; fi
    kx=$(echo ${KX:?KX = the kernels} | tr " " ",")
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/$nb --account "$a" --slug "$s-smoke" --kernels "$kx" --datasets "$ds" --smoke 'FL3Z_SMOKE=1;FLAG_SMOKE=1'
    else
      $K push kaggle/$nb --account "$a" --slug "${SLUG:-$s-r11}" $([ "${NOGPU:-0}" = 1 ] || echo --gpu) --kernels "$kx" --datasets "$ds"         --env "${XENV:-E2E_SPLIT=dev}"   # NOGPU=1: CPU (no training here; frees the account's 2 GPU slots)
    fi ;;
  112f0) # CLEANTUNE-01 F0 (FLAG_112_f0): audit A0 + B0c final head (5 seeds) + zip; needs $CTT (ct01g1 tables) and $CTB
    $K push kaggle/FLAG_112_f0.ipynb --account "${ACC:?ACC = one account with GPU quota}" --slug flag-2027-nb-112-f0 --gpu \
      --datasets "$DSC,$CTB,$TAB,$CTT,$AVF" ;;   # AVF: AudioCLIP arrays of the English validation table (EN-T3-01 table_en)
  102)   # LOSSCUR-01 (FLAG_102, DSC: no AVSpeech / LRS3): SMOKE=1 first (Kaggle CPU); then 3 GPU runs, ACCS = "f12 f34 f5" accounts
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_102_losscur01.ipynb --account "${ACC:-nguyentruonglong}" --slug flag-2027-nb-102-lc01-smoke \
        --datasets "$DSC,$B3,$TAB" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1'
    else
      a=(${ACCS:?ACCS = 3 accounts: f12 f34 f5})
      f=("1,2" "3,4" "5"); s=(f12 f34 f5)
      for i in 0 1 2; do
        $K push kaggle/FLAG_102_losscur01.ipynb --account "${a[$i]}" --slug "flag-2027-nb-102-lc01-${s[$i]}" --gpu \
          --datasets "$DSC,$B3,$TAB" --env "LAB_FOLDS=${f[$i]}"
      done
    fi ;;
  102r)  # LOSSCUR-01: STEP=1 fold tables of the three FLAG_102 runs (OWN="acc_f12 acc_f34 acc_f5") -> dataset; STEP=2 report on CPU
    if [ "${STEP:-1}" = 1 ]; then
      o=(${OWN:?OWN = the 3 accounts of the f12 f34 f5 runs})
      gather_tables lc01 flag2027-lc01-tables-p "${o[0]}/flag-2027-nb-102-lc01-f12" "${o[1]}/flag-2027-nb-102-lc01-f34" "${o[2]}/flag-2027-nb-102-lc01-f5"
      echo "then: STEP=2 bash kaggle/sat_push.sh 102r"
    else
      $K privatize ngnnguynchn/flag2027-lc01-tables-p
      $K push kaggle/FLAG_102_losscur01.ipynb --account "${ACC:-nguynchnngn}" --slug flag-2027-nb-102-lc01-report \
        --datasets "$DSC,$B3,$TAB,ngnnguynchn/flag2027-lc01-tables-p" --env "LAB_FOLDS="
    fi ;;
  100r)  # HARDNEG-AGE-01: STEP=1 fold tables of the three FLAG_100 runs -> dataset; STEP=2 privatize + gather + report on CPU
    if [ "${STEP:-1}" = 1 ]; then
      gather_tables hna01 flag2027-hna01-tables-p nguynchnngn1/flag-2027-nb-100-hna01-f12 nguynchnngn/flag-2027-nb-100-hna01-f34 nguyenchonnhan/flag-2027-nb-100-hna01-f5
      echo "then: STEP=2 bash kaggle/sat_push.sh 100r"
    else
      $K privatize ngnnguynchn/flag2027-hna01-tables-p
      $K push kaggle/FLAG_100_hna01.ipynb --account "${ACC:-nguynchnngn}" --slug flag-2027-nb-100-hna01-report \
        --datasets "$DS73,$B3,$TAB,ngnnguynchn/flag2027-hna01-tables-p" --env "LAB_FOLDS="
    fi ;;

  98)    # VOICELORA-02 step 2: three fold groups (GPU T4 x2, ~3 h each), after 67yd; then `98r`
    a=(${ACCS:-duongphihoang locnguyentruongan chnnhnnguyn})
    for i in 0 1 2; do
      f=$(echo "1,2 3,4 5" | cut -d' ' -f$((i+1))); t=$(echo "f12 f34 f5" | cut -d' ' -f$((i+1)))
      $K push kaggle/FLAG_98_voicelora02.ipynb --account "${a[$i]}" --slug "flag-2027-nb-98-voicelora02-$t" --gpu \
        --datasets "$DS73,$B3,$C5,$TAB,$AVF,$VL2,$VLX2" --env "LAB_FOLDS=$f"
    done ;;

  98r)   # STEP=1: gather the fold tables (dataset); STEP=2: report + final head (5 seeds) + zips on GPU (~1.5 h)
    a=(${ACCS:-duongphihoang locnguyentruongan chnnhnnguyn})
    if [ "${STEP:-1}" = 1 ]; then
      gather_tables voicelora02 flag2027-voicelora02-tables-p "${a[0]}/flag-2027-nb-98-voicelora02-f12" \
        "${a[1]}/flag-2027-nb-98-voicelora02-f34" "${a[2]}/flag-2027-nb-98-voicelora02-f5"
      echo "then: STEP=2 bash kaggle/sat_push.sh 98r"
    else
      $K privatize ngnnguynchn/flag2027-voicelora02-tables-p
      $K push kaggle/FLAG_98_voicelora02.ipynb --account "${ACC:-nguynchnngn}" --slug flag-2027-nb-98-voicelora02-final --gpu \
        --datasets "$DS73,$B3,$C5,$TAB,$AVF,$VL2,$VLX2,ngnnguynchn/flag2027-voicelora02-tables-p" --env "LAB_FOLDS="
    fi ;;

  # ---- 19/10 "mở rộng" (READMEs + LOG 280 written before any run). SMOKE=1 first (Kaggle CPU), then without SMOKE (GPU)
  67c|67d|67dh)   # VOICELORA-03 (sam100p -> frozen C5 face) | VOICELORA-04 (8 clips / identity, 7,000 phase-B steps) | its lr 4e-5 arm
    D67="$ACACHE,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-feats-vox2b-p,$C5"
    case "$STEP_" in
      67c)  nb=FLAG_67c_voicelora3.ipynb; a=${ACC:-onnhqunhh};      slug=flag-2027-nb-67c-voicelora3;  up=flag2027-voicelora3-p;  ex="" ;;
      67d)  nb=FLAG_67d_voicelora4.ipynb; a=${ACC:-dnq1209};        slug=flag-2027-nb-67d-voicelora4;  up=flag2027-voicelora4-p;  ex="" ;;
      67dh) nb=FLAG_67d_voicelora4.ipynb; a=${ACC:-nguyenchonnhan}; slug=flag-2027-nb-67dh-voicelora4h; up=flag2027-voicelora4h-p; ex=";VL_LR_ENC=4e-5" ;;
    esac
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/$nb --account "$a" --slug $slug-smoke --internet \
        --datasets "$D67" --smoke "FLAG_SMOKE=1;VL_STEPS_A=3;VL_STEPS_B=3;VL_MAX_CHUNKS=4;VL_EXTRACT_MAX=64$ex"
    else
      $K push kaggle/$nb --account "$a" --slug $slug --gpu ${GPUT:+--accelerator $GPUT} --internet --datasets "$D67" --env "VL_UPLOAD=$a/$up$ex"
    fi ;;

  65c|65j|65j2)   # COTRAIN-01 (C5 recipe towards sam100fM; needs FLAG_67b's $VL2) | JOINT-01 session 1 (T4 x 2, 8.5 h budget) | session 2 (resume)
    D65="$MAVRAW,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-feats-vox2b-p,onnhqunhh/flag2027-lora-frames-p"
    SM='FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6;SMOKE_IDS=4;LORA_STEPS=3;LORA_P=4;LORA_K=2'
    case "$STEP_" in
      65c)  need_complete "$VL2ACC/flag-2027-nb-67b-voicelora2"; $K privatize $VL2
            nb=FLAG_65c_cotrain1.ipynb; a=${ACC:-lkhcvit}; slug=flag-2027-nb-65c-cotrain1; ds="$D65,$VL2"; up=flag2027-feats-cotrain1-co-p; ex=""; kk="" ;;
      65j)  nb=FLAG_65j_joint1.ipynb; a=${ACC:-nguynchnngn}; slug=flag-2027-nb-65j-joint1; ds="$D65,$ACACHE"; up=flag2027-feats-joint1-p
            ex=";LORA_BUDGET_H=8.5"; kk=""; SM="$SM;JT_MAX_CHUNKS=1" ;;
      65j2) nb=FLAG_65j_joint1.ipynb; a=${ACC:-nguynchnngn}; slug=${S65J2:-flag-2027-nb-65j-joint1-r2}; ds="$D65,$ACACHE"; up=flag2027-feats-joint1-p
            ex=";LORA_BUDGET_H=11"
            if [ -n "$J1CK" ]; then ds="$ds,$J1CK"; kk=""    # session 1's checkpoint as a private dataset: run on another account (KAGGLE_API_RULES §6d)
            else need_complete "$a/flag-2027-nb-65j-joint1"; kk="$a/flag-2027-nb-65j-joint1"; fi ;;
    esac
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/$nb --account "$a" --slug $slug-smoke --internet --datasets "$ds" --smoke "$SM"
    else
      $K push kaggle/$nb --account "$a" --slug $slug --gpu ${GPUT:+--accelerator $GPUT} --internet --datasets "$ds" ${kk:+--kernels "$kk"} --env "FL3_UPLOAD=$a/$up$ex"
    fi ;;

  143)   # XENC-01 step 1 (FLAG_143, T4, Internet): token caches (saved C5c face adapter, SimAM voice) of MAV rows + auxiliary identities
    D65="$MAVRAW,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-feats-vox2b-p,onnhqunhh/flag2027-lora-frames-p"
    a=${ACC:-thanhthaoph}; slug=${SLUG:-flag-2027-nb-143-xenc01-tok}
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_143_xenc01_tok.ipynb --account "$a" --slug $slug-smoke --internet --datasets "$D65,$ACACHE,$CTC"         --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6;SMOKE_IDS=4;JT_MAX_CHUNKS=1'
    else
      $K push kaggle/FLAG_143_xenc01_tok.ipynb --account "$a" --slug $slug --gpu ${GPUT:+--accelerator $GPUT} --internet         --datasets "$D65,$ACACHE,$CTC" --env "FL3_UPLOAD=$a/flag2027-xenc01-tok-p"
    fi ;;

  144)   # XENC-01 step 2 (FLAG_144, T4): 8 cross-encoder arms on the FLAG_143 tokens -> xenc01 trial tables (+ ct01r2 tables);
         # TOK = the token dataset (owner/flag2027-xenc01-tok-p); SMOKE=1: Kaggle CPU on the FLAG_143 smoke output (same account)
    a=${ACC:-thanhthaoph}; slug=${SLUG:-flag-2027-nb-144-xenc01-train}; CT2T=ngnnguynchn/flag2027-ct01r2-tables-p
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_144_xenc01_train.ipynb --account "$a" --slug $slug-smoke --datasets "$CT2T"         --kernels "$a/flag-2027-nb-143-xenc01-tok-smoke" --smoke 'FLAG_SMOKE=1;XENC_STEPS=3;XENC_P=2'
    else
      $K push kaggle/FLAG_144_xenc01_train.ipynb --account "$a" --slug $slug --gpu ${GPUT:+--accelerator $GPUT}         --datasets "$CT2T,${TOK:?TOK = owner/flag2027-xenc01-tok-p}" --env "XENC_UPLOAD=$a/flag2027-xenc01-tables-p${ARMS:+;XENC_ARMS=$ARMS}"
    fi ;;

  146)   # MLLM-01 step 0 (FLAG_146, T4 x 2, Internet): Qwen2.5-Omni-3B fp16 check + zero-shot scores of every SAFE-04 trial / dev test row
    a=${ACC:-onnhqunhh}; slug=${SLUG:-flag-2027-nb-146-mllm01-zs}; ds="$MAVRAW,ngnnguynchn/flag2027-ct01r2-tables-p"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_146_mllm01_zs.ipynb --account "$a" --slug $slug-smoke --internet --datasets "$ds" --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=4'
    else
      $K push kaggle/FLAG_146_mllm01_zs.ipynb --account "$a" --slug $slug --gpu ${GPUT:+--accelerator $GPUT} --internet --datasets "$ds"         --env "MLLM_UPLOAD=$a/flag2027-mllm01-zs-p"
    fi ;;

  145)   # UNCSCORE-01 (FLAG_145, GPU T4 x2, round-2 framework g3): SMOKE=1 (Kaggle CPU) | GPU: ACCS = 3 accounts f12 f34 f5
    x=",$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_145_uncscore01.ipynb --account "${ACC:-berryyusilee}" --slug flag-2027-nb-145-uncscore01-smoke         --datasets "$DSC,$CTB,$TAB,$AVF$x" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1'
    else
      a=(${ACCS:?ACCS = 3 accounts: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
      for i in 0 1 2; do
        $K push kaggle/FLAG_145_uncscore01.ipynb --account "${a[$i]}" --slug "flag-2027-nb-145-uncscore01-${s[$i]}" --gpu           --datasets "$DSC,$CTB,$TAB,$AVF$x" --env "LAB_FOLDS=${f[$i]}"
      done
    fi ;;

  147)   # MLLM-01 step 1 (FLAG_147, T4 x 2, Internet): Qwen2.5-Omni-3B + LoRA on auxiliary pairs; TOK = FLAG_143 token dataset,
         # ZS = FLAG_146 dataset (trMZ tables, optional); session 2: KX=<session 1> or CK=<owner>/flag2027-mllm01-ckpt-p, SLUG=<new>;
         # STEPS = cap on optimiser steps (MLLM_STEPS_MAX; a resume at the cap only scores)
    D65="$MAVRAW,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-feats-vox2b-p,onnhqunhh/flag2027-lora-frames-p"
    a=${ACC:-onnhqunhh}; slug=${SLUG:-flag-2027-nb-147-mllm01-ft}; ds="$D65,$ACACHE,ngnnguynchn/flag2027-ct01r2-tables-p${ZS:+,$ZS}${CK:+,$CK}"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_147_mllm01_ft.ipynb --account "$a" --slug $slug-smoke --internet --datasets "$ds${TOK:+,$TOK}" ${KX:+--kernels "$KX"} --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=4;SMOKE_IDS=4;JT_MAX_CHUNKS=1;MLLM_MICRO=1;MLLM_ACCUM=1'
    else
      $K push kaggle/FLAG_147_mllm01_ft.ipynb --account "$a" --slug $slug --gpu ${GPUT:+--accelerator $GPUT} --internet --datasets "$ds,${TOK:?TOK = owner/flag2027-xenc01-tok-p}" ${KX:+--kernels "$KX"} --env "MLLM_UPLOAD=$a/flag2027-mllm01-ft-p;LORA_BUDGET_H=${BUDGET:-8.5}${STEPS:+;MLLM_STEPS_MAX=$STEPS}"
    fi ;;

  142)   # IBLORA-01 (Experiment/IBLORA-01/README.md): ImageBind LoRA, T4 x 2, LORA_BUDGET_H stop; session 2 = same step with
         # KX=<owner/slug of session 1> SLUG=<new slug> (resumes from its checkpoint, then extracts ibLv / ibLa); on another account:
         # ACC=<acc> CK=<owner>/flag2027-iblora1-ckpt-p SLUG=<new slug> (the checkpoint the notebook uploads at a budget stop)
    D65="$MAVRAW,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-feats-vox2b-p,onnhqunhh/flag2027-lora-frames-p"
    a=${ACC:-trngminhhieu}; slug=${SLUG:-flag-2027-nb-142-iblora01}
    nb=kaggle/FLAG_142${S:+s}_iblora01${S:+s}.ipynb; ls=${S:+,ngnnguynchn/flag2027-lid-strict-p}   # S=1 (3/11): IBLORA-01s, LIDSTRICT-01 list (build_flag142.py s)
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push $nb --account "$a" --slug $slug-smoke --internet --datasets "$D65,$ACACHE$ls" \
        --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6;SMOKE_IDS=4;LORA_STEPS=3;LORA_P=4;LORA_K=2;JT_MAX_CHUNKS=1'
    else
      $K push $nb --account "$a" --slug $slug --gpu ${GPUT:+--accelerator $GPUT} --internet \
        --datasets "$D65,$ACACHE$ls${CK:+,$CK}" ${KX:+--kernels "$KX"} --env "FL3_UPLOAD=$a/flag2027-feats-iblora1${S:+s}-p;LORA_BUDGET_H=${BUDGET:-8.5}"
    fi ;;

  111s|111sg|111sr|112f0s|112f1s) # F1s lane (3/11, nhat ky 355): LID_STRICT=1, CTB / CTC = B3s / C5s, ngnnguynchn/flag2027-lid-strict-p v2
         # 111s: CLEANTUNE-01 r2 with ONLY the CA arm (CT_R2_ONLY=CA; SMOKE=1 | ACCS = f12 f34 f5); 111sr: its 3 runs (RUNS) -> flag2027-ct02s-r2ca-tables-p
         # 111sg: the 3 CLEANTUNE-02 g1 runs -> flag2027-ct02s-g1-tables-p; 112f0s: F0s (ACC) ; 112f1s: F1s (ACC = account of 112f0s, --kernels)
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"
    CT2S=${CT2S:-ngnnguynchn/flag2027-ct02s-r2ca-tables-p}; CTTS=${CTTS:-ngnnguynchn/flag2027-ct02s-g1-tables-p}
    F0K="${ACC:-nguyenchonnhan}/flag-2027-nb-112s-f0s${SMOKE:+-smoke}"
    case "$STEP_" in
      111sg) gather_tables ct01g1 "${CTTS#*/}" ${RUNS:?RUNS = the 3 CLEANTUNE-02 g1 runs} ;;
      111sr) gather_tables ct01r2 "${CT2S#*/}" ${RUNS:?RUNS = the 3 111s GPU runs} ;;
      111s)
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_111_ct01_r2.ipynb --account "${ACC:-nguyenchonnhan}" --slug flag-2027-nb-155s-ct02ca-smoke \
            --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1;LID_STRICT=1;CT_R2_ONLY=CA'
        else
          a=(${ACCS:?ACCS = 3 accounts: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
          for i in 0 1 2; do
            $K push kaggle/FLAG_111_ct01_r2.ipynb --account "${a[$i]}" --slug "flag-2027-nb-155s-ct02ca-${s[$i]}" --gpu \
              --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD" --env "LAB_FOLDS=${f[$i]};LID_STRICT=1;CT_R2_ONLY=CA"
          done
        fi ;;
      112f0s)
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_112_f0.ipynb --account "${ACC:-nguyenchonnhan}" --slug flag-2027-nb-112s-f0s-smoke \
            --datasets "$DSC,$CTB,$TAB,$CTT,$AVF,$SD" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;CT_FINAL_SEEDS=1;BN42_STANDIN=1;LID_STRICT=1'
        else
          $K push kaggle/FLAG_112_f0.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-112s-f0s --gpu \
            --datasets "$DSC,$CTB,$TAB,$CTTS,$AVF,$SD" --env "LID_STRICT=1"
        fi ;;
      112f1s)   # CLEANCOMP-01 (3/11, nhat ky 362): F1s reads the strict S6 / rel / relg / v11-T3 bundle $CCE INSTEAD of flag2027-e2e-dev-p; not readable = not pushed
        python kaggle/build_flag112.py f1 Experiment/CLEANTUNE-01/ct_decision.json > /dev/null   # re-embed the code (e2e_lib with the CLEANCOMP=1 opt-in); same call as f1s_chain
        CCE=${CCE:-ngnnguynchn/flag2027-cleancomp-e2e-p}; DSCC=$(echo "$DSC" | tr ',' '\n' | grep -v 'flag2027-e2e-dev-p' | paste -sd, -)
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_112_f1.ipynb --account "${ACC:-nguyenchonnhan}" --slug ${SLUG:-flag-2027-nb-112s-f1s-smoke} --kernels "$F0K" \
            --datasets "$DSCC,$CCE,$CTB,$TAB,${CT2:-ngnnguynchn/flag2027-ct01r2-tables-p},$AVF,$G3X,$SD" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;CT_FINAL_SEEDS=1;BN42_STANDIN=1;LID_STRICT=1;CLEANCOMP=1'
        elif ! timeout 200 $K access "${ACC:?ACC}" "$CCE" 2>&1 | grep -q ": readable"; then
          echo "!! 112f1s NOT pushed (0 pushed): $CCE (CLEANCOMP-01 strict S6 + v11 T3 components, LIDSTRICT-01 A0 section 5 (3)) is not readable by $ACC yet - see docs/RUNS.md CLEANCOMP-01 / kaggle/cleancomp_chain.sh"
          exit 1
        else
          $K push kaggle/FLAG_112_f1.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-112s-f1s --gpu --kernels "$F0K" \
            --datasets "$DSCC,$CCE,$CTB,$TAB,$CT2S,$AVF,$G3X,$SD" --env "LID_STRICT=1;CLEANCOMP=1"
        fi ;;
    esac ;;
  164|164f)   # CLEANCOMP-01 (3/11, nhat ky 362, Experiment/CLEANCOMP-01/README.md): 164 = fold tables (ACC, FOLDS = "1,2,3" | "4,5"; SMOKE=1 on CPU),
            # 164f = final artefacts (strict s007_xall.pt + v11 T3 head + dev T3 streams, ACC); both under LID_STRICT=1 on flag2027-lab-data-p + eda-bundle-p + lid-strict-p
    CCD="ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-eda-bundle-p,ngnnguynchn/flag2027-lid-strict-p"
    if [ "$STEP_" = 164f ]; then CCD="$CCD,ngnnguynchn/flag2027-e2e-dev-p"; nbk=kaggle/FLAG_164f_cleancomp01_final.ipynb; slug=flag-2027-nb-164f-cleancomp01-final; envs="LID_STRICT=1"
    else nbk=kaggle/FLAG_164_cleancomp01.ipynb; slug=flag-2027-nb-164-cleancomp01-f${FOLDS:-all}; envs="LID_STRICT=1;LAB_FOLDS=${FOLDS:-1,2,3,4,5}"; slug=${slug//,/}; fi
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push $nbk --account "${ACC:-thanhthaoph}" --slug ${SLUG:-$slug} --datasets "$CCD" --smoke "LID_STRICT=1;LAB_FOLDS=1;L0_STEPS=10;L0_NETS=1;SAFE04_THREADS=4${CC_SMOKE:+;CC_SMOKE=1}${CC_MODE:+;CC_MODE=$CC_MODE}" --force
    else
      $K push $nbk --account "${ACC:?ACC}" --slug ${SLUG:-$slug} --gpu --datasets "$CCD" --env "$envs"
    fi ;;

  166|166g|166r|166f)   # WHISPER-02 (3/11, nhat ky 365, Experiment/WHISPER-02/README.md): F1s's CA head + the whv3 voice block (FLAG_160 a / b / c); LID_STRICT=1
            # 166: fold lanes (SMOKE=1 on CPU | ACCS = f12 f34 f5); 166g: gather the 3 runs' tables (RUNS) -> ngnnguynchn/flag2027-wh02-r2-tables-p;
            # 166r: CPU report (ACC) -> wh02_decision.json; 166f: final CAW 10 seeds + zip (ACC; F0S zip dataset F0ZD, tables WHT, CLEANCOMP bundle CCE)
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"
    WHA=${WHA:-tranthuytien2511/flag2027-feats-whv3-a-p}; WHB=${WHB:-thanhhuyen11/flag2027-feats-whv3-b-p}; WHC=${WHC:-phuonggmaii/flag2027-feats-whv3-c-p}
    WHT=${WHT:-ngnnguynchn/flag2027-wh02-r2-tables-p}; WHENV="LID_STRICT=1;WHV3=1;CT_R2_ONLY=CA,CAW,CAWO"
    case "$STEP_" in
      166g) gather_tables ct01r2 "${WHT#*/}" ${RUNS:?RUNS = the 3 166 GPU runs} ;;
      166)
        python kaggle/build_flag166.py > /dev/null
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_166_whisper02.ipynb --account "${ACC:-tranthuytien2511}" --slug ${SLUG:-flag-2027-nb-166-wh02-smoke} \
            --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1;LID_STRICT=1;WHV3=1;CT_R2_ONLY=CA,CAW,CAWO'
        else
          a=(${ACCS:?ACCS = 3 accounts: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
          for i in 0 1 2; do
            $K push kaggle/FLAG_166_whisper02.ipynb --account "${a[$i]}" --slug "flag-2027-nb-166-wh02-${s[$i]}" --gpu \
              --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$WHA,$WHB,$WHC" --env "LAB_FOLDS=${f[$i]};$WHENV"
          done
        fi ;;
      166r)
        python kaggle/build_flag166.py > /dev/null
        $K push kaggle/FLAG_166_whisper02.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-166r-wh02-report \
          --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$WHT" --env "LAB_FOLDS=;$WHENV" ;;
      166f)   # reads the strict S6 / v11-T3 bundle $CCE like 112f1s; F0S English bytes come from the dataset F0ZD (a --kernels input must be the pushing account's own)
        python kaggle/build_flag166.py > /dev/null
        CCE=${CCE:-ngnnguynchn/flag2027-cleancomp-e2e-p}; F0ZD=${F0ZD:-ngnnguynchn/flag2027-wh02-f0s-zip-p}; DSCC=$(echo "$DSC" | tr ',' '\n' | grep -v 'flag2027-e2e-dev-p' | paste -sd, -)
        $K push kaggle/FLAG_166f_whisper02_final.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-166f-whisper02-final --gpu \
          --datasets "$DSCC,$CCE,$CTB,$TAB,$WHT,$F0ZD,$AVF,$G3X,$SD,$WHA,$WHB,$WHC" --env "LID_STRICT=1;CLEANCOMP=1;WHV3=1;CT_R2_ONLY=CAW" ;;
    esac ;;
  167|167g|167r|167f0|167f)   # T3S-01 (4/11, nhat ky 378, Experiment/T3S-01/README.md): Track-3 recipe branches 4 faces x 6 voices x 5 seeds + the CA control; LID_STRICT=1
            # 167: 5 fold runs, one fold each, 4 face jobs 2 per GPU (SMOKE=1 on CPU | ACCS = 5 accounts f1..f5); 167g: the 20 job tables of the 5 runs (RUNS) -> $T3T;
            # 167r: CPU report (ACC) -> t3s_decision.json; 167f0: F1s's final npz + zip (output of $F1K) -> $F1F; 167f: final branches + F1sT zip (ACC; SMOKE=1 on CPU)
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p     # B3s / C5s (LID_STRICT adapters), as F1s
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; F1F=${F1F:-ngnnguynchn/flag2027-t3s01-f1s-final-p}; F1K=${F1K:-locnguyentruongan/flag-2027-nb-112s-f1s}
    CCE=${CCE:-ngnnguynchn/flag2027-cleancomp-e2e-p}; DSCC=$(echo "$DSC" | tr ',' '\n' | grep -v 'flag2027-e2e-dev-p' | paste -sd, -)
    T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"; T3SM="FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;T3S_SEEDS=1;T3S_STAGGER=0;BN42_STANDIN=1;CT_STANDIN=1"
    # 5/11: T3S_PER_GPU=1 (2 jobs per GPU -> 4 data loads at once -> host OOM, exit -9, in all 5 folds of v1)
    case "$STEP_" in
      167g) PAT='ct01r2[a-z0-9]+_fold[0-9]\.parquet' GLOB='ct01r2?*_fold*.parquet' NTAB=20 gather_tables ct01r2 "${T3T#*/}" ${RUNS:?RUNS = the 5 167 GPU runs} ;;
      167)
        python kaggle/build_flag167.py > /dev/null
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_167_t3s01.ipynb --account "${ACC:-phantrnhuyntrn}" --slug ${SLUG:-flag-2027-nb-167-t3s01-smoke} \
            --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "$T3SM;LAB_FOLDS=1;T3S_PARTS=farl,b3s;$T3ENV" ${FORCE:+--force}
        else
          a=(${ACCS:?ACCS = 5 accounts: f1 f2 f3 f4 f5})
          for i in 0 1 2 3 4; do
            $K push kaggle/FLAG_167_t3s01.ipynb --account "${a[$i]}" --slug "flag-2027-nb-167-t3s01-f$((i + 1))" --gpu \
              --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --env "LAB_FOLDS=$((i + 1));$T3ENV"
          done
        fi ;;
      167r)
        python kaggle/build_flag167.py > /dev/null
        $K push kaggle/FLAG_167_t3s01.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-167r-t3s01-report \
          --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force} ;;
      167f0)
        d=$(mktemp -d); need_complete "$F1K"
        for p in 'ct01r2_final_dev\.npz' 'submission_cleantune01_F1_CA_dev\.zip'; do $K output "$F1K" --dest "$d/o" --pattern "$p" > /dev/null; done
        f=$(find "$d/o" -name ct01r2_final_dev.npz | head -1); z=$(find "$d/o" -name submission_cleantune01_F1_CA_dev.zip | head -1)
        [ -n "$f" ] && [ -n "$z" ] || { echo "!! F1s final npz / zip not in $F1K output"; exit 1; }
        $K dscreate "$(cygpath -w "$f" 2>/dev/null || echo "$f")" "$(cygpath -w "$z" 2>/dev/null || echo "$z")" --account ngnnguynchn --slug "${F1F#*/}"
        echo "wait until the dataset is ready (1-3 min), then: $K privatize $F1F" ;;
      167f)
        python kaggle/build_flag167.py > /dev/null
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_167f_t3s01_final.ipynb --account "${ACC:-nguyntrquangtrung}" --slug ${SLUG:-flag-2027-nb-167f-t3s01-final-smoke} \
            --datasets "$DSCC,$CCE,$CTB,$TAB,$F1F,$AVF,$G3X,$SD,$SCT" --smoke "$T3SM;LAB_FOLDS=;CLEANCOMP=1;$T3ENV" ${FORCE:+--force}
        else
          $K push kaggle/FLAG_167f_t3s01_final.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-167f-t3s01-final --gpu \
            --datasets "$DSCC,$CCE,$CTB,$TAB,$T3T,$F1F,$AVF,$G3X,$SD,$SCT" --env "LAB_FOLDS=;CLEANCOMP=1;$T3ENV"
        fi ;;
    esac ;;
  173)   # RFSTUNE-01 (11/11, Experiment/RFSTUNE-01/README.md): RFS-LAB per registered arm (FLAG_170 + arm loop); ACC + PART=1|2 (GPU), SMOKE=1 = Kaggle CPU
    python kaggle/build_flag173.py > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_173_rfstune.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-173-rfstune-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40;RFST_PART=${PART:-1}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_173_rfstune.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-173-rfstune-p${PART:?PART}" --gpu \
        --datasets "ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p" \
        --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  175)   # RFSTUNE-02 (12/11, Experiment/RFSTUNE-02/README.md): round-2 lab (FLAG_175 = FLAG_173 with ARMS2, tables rfu*); ACC + PART=1|2, SMOKE=1
    python kaggle/build_flag173.py 2 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_175_rfstune2.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-175-rfstune2-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40;RFST_PART=${PART:-1}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_175_rfstune2.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-175-rfstune2-p${PART:?PART}" --gpu \
        --datasets "ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p" \
        --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  176)   # RFSTUNE-02 report (CPU, FLAG_176): as 174 with the two 175 runs of ACC as kernel inputs
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 2 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_176_rfstune2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-176-rfstune2-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" \
        --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_176_rfstune2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-176-rfstune2-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" \
        --kernels "$ACC/flag-2027-nb-175-rfstune2-p1,$ACC/flag-2027-nb-175-rfstune2-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  177)   # ENPROG-01 (12/11, Experiment/ENPROG-01/README.md): lab of the 12 block arms (FLAG_177 = FLAG_173 with ARMS3, tables rfe*); ACC + PART=1|2, SMOKE=1
    python kaggle/build_flag173.py 3 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_177_enprog.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-177-enprog-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40;RFST_PART=${PART:-1}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_177_enprog.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-177-enprog-p${PART:?PART}" --gpu         --datasets "ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"         --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  178)   # ENPROG-01 report (CPU, FLAG_178): as 176 with the 173 / 175 / 177 runs of ACC as kernel inputs (tables rft*, rfu*, rfe*)
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 3 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_178_enprog_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-178-enprog-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_178_enprog_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-178-enprog-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-173-rfstune-p1,$ACC/flag-2027-nb-173-rfstune-p2,$ACC/flag-2027-nb-175-rfstune2-p1,$ACC/flag-2027-nb-175-rfstune2-p2,$ACC/flag-2027-nb-177-enprog-p1,$ACC/flag-2027-nb-177-enprog-p2"         --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  179)   # RISKSIM-01 report (CPU, FLAG_179): same inputs as the ENPROG report: the 173 / 175 / 177 runs of ACC as kernel inputs (tables rft*, rfu*, rfe*)
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 4 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_179_risksim_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-179-risksim-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_179_risksim_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-179-risksim-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-173-rfstune-p1,$ACC/flag-2027-nb-173-rfstune-p2,$ACC/flag-2027-nb-175-rfstune2-p1,$ACC/flag-2027-nb-175-rfstune2-p2,$ACC/flag-2027-nb-177-enprog-p1,$ACC/flag-2027-nb-177-enprog-p2"         --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  180)   # HARDNEG-RFS + EN-HEARD (12/11): lab of the 11 arms (FLAG_180 = FLAG_173 with ARMS4, tables rfh*); ACC + PART=1|2, SMOKE=1
    python kaggle/build_flag173.py 4 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_180_engx.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-180-engx-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40;RFST_PART=${PART:-1}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_180_engx.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-180-engx-p${PART:?PART}" --gpu         --datasets "ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"         --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  181)   # HARDNEG-RFS + EN-HEARD report (CPU, FLAG_181): the two 180 runs of ACC as kernel inputs
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 5 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_181_engx_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-181-engx-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_181_engx_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-181-engx-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-180-engx-p1,$ACC/flag-2027-nb-180-engx-p2"         --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  182)   # IBLORA-01 bac 2 report (CPU, FLAG_182, 12/11): F1s' English fuser with ibLoRA; + ct02s-g1 tables + IBLORA-01s arrays, no kernels
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 6 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_182_iblora2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-182-iblora2-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,ngnnguynchn/flag2027-ct02s-g1-tables-p,phantrnanna/flag2027-feats-iblora1s-p"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_182_iblora2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-182-iblora2-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T,ngnnguynchn/flag2027-ct02s-g1-tables-p,phantrnanna/flag2027-feats-iblora1s-p"         --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  183)   # FACTOR-01 (13/11, Experiment/FACTOR-01/README.md): RFS-LAB with F1s' blocks (FLAG_183 = FLAG_173 with ARMS5, tables rfx*); ACC + PART=1|2 (GPU);
          # SMOKE=1 = synthetic bundle (stand-in blocks, Kaggle CPU); SMOKE=real = real data on Kaggle CPU, fold 1, arms a0 + ALL, 40 steps (loaders + identity checks)
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 5 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_183_factor.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-183-factor-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40;RFST_PART=${PART:-1}" ${FORCE:+--force}
    elif [ "${SMOKE:-0}" = real ]; then
      $K push kaggle/FLAG_183_factor.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-183-factor-realsmoke --datasets "$LABDS,$XDS"         --env "RFS_STEPS=40;RFS_LAB_FOLDS=1;RFST_ONLY=${ONLY:-a0,ALL}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_183_factor.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-183-factor-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS"         --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  184)   # FACTOR-01 report (CPU, FLAG_184): step 0 atlas + step 2 gate; the two 183 runs of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 7 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_184_factor_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-184-factor-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_184_factor_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-184-factor-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-183-factor-p1,$ACC/flag-2027-nb-183-factor-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  185)   # FACTOR-02 (13/11, Experiment/FACTOR-02/README.md): RFS-LAB, ARMS6 (FLAG_185 = FLAG_183 with ARMS6, tables rfy*); ACC + PART=1|2 (GPU); SMOKE=1 synthetic
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 6 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_185_factor2.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-185-factor2-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40;RFST_PART=${PART:-1}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_185_factor2.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-185-factor2-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS"         --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  186)   # FACTOR-02 report (CPU, FLAG_186): 183 p1/p2 (+ 185 p1/p2 unless PHASE=1) of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 8 > /dev/null
    KS="$ACC/flag-2027-nb-183-factor-p1,$ACC/flag-2027-nb-183-factor-p2"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_186_factor2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-186-factor2-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    elif [ "${PHASE:-2}" = 1 ]; then
      $K push kaggle/FLAG_186_factor2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-186-factor2-phase1 --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$KS" --env "LAB_FOLDS=;FACTOR2_PHASE=1;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_186_factor2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-186-factor2-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$KS,$ACC/flag-2027-nb-185-factor2-p1,$ACC/flag-2027-nb-185-factor2-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  187)   # TRANSL-01 (14/11, Experiment/TRANSL-01/README.md): RFS-LAB, ARMS7 (FLAG_187 = FLAG_183 with ARMS7, tables rfz*); ACC + PART=1|2 (GPU);
          # SMOKE=1 synthetic bundle, every arm (PART unset), 40 steps
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 7 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_187_transl.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-187-transl-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40${PART:+;RFST_PART=$PART}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_187_transl.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-187-transl-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS"         --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  188)   # TRANSL-01 report (CPU, FLAG_188): 183 p1/p2 (rfxc5rl) + 187 p1/p2 of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 9 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_188_transl_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-188-transl-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      if [ -n "${TDS:-}" ]; then   # 14/11: lab parts on two accounts -> their rfz tables as one private dataset (kaggle/transl_chain.sh)
        $K push kaggle/FLAG_188_transl_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-188-transl-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T,$TDS" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
      else
      $K push kaggle/FLAG_188_transl_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-188-transl-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$([ "$ACC" = phantrnhuyntrn ] && echo "$ACC/flag-2027-nb-183-factor-p1,$ACC/flag-2027-nb-183-factor-p2,")$ACC/flag-2027-nb-187-transl-p1,$ACC/flag-2027-nb-187-transl-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
      fi
    fi ;;
  189)   # FVBIND-01 (14/11, Experiment/FVBIND-01/README.md): ImageBind audio LoRA, big batch / identity anchor; ACC + ARMS (B0,B1 | IB0,B2), GPU + Internet;
          # SMOKE=1 synthetic (stand-in model, every arm); SMOKE=real = real data on Kaggle CPU, stand-in model, fold 1, few persons / rows
    FDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p,duongphihoang/flag2027-audio-cache-p,chnnhnnguyn/external-data-mavceleb,ngnnguynchn/flag2027-raw-v4-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag189.py > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_189_fvbind.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-189-fvbind-smoke --smoke "FLAG_SMOKE=1;FVB_STEPS=12;FVB_ARMS=B0,B1,B2,IB0" ${FORCE:+--force}
    elif [ "${SMOKE:-0}" = real ]; then
      $K push kaggle/FLAG_189_fvbind.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-189-fvbind-realsmoke --datasets "$FDS,$XDS"         --env "FVB_STANDIN=1;FVB_STEPS=6;RFS_LAB_FOLDS=1;FVB_ARMS=${ARMS:-B0,B2,IB0}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_189_fvbind.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-189-fvbind-$(echo "${ARMS:?ARMS}" | tr 'A-Z,' 'a-z-')" --gpu --internet --datasets "$FDS,$XDS"         --env "FVB_ARMS=$ARMS" ${FORCE:+--force}
    fi ;;
  192)   # FVBIND-01 gate 2 step 1 (14/11, README section 3b): block v_fvb from the B2 state; ACC=thanhhuyen11 (owns 189 b2 + 47rs outputs), GPU + Internet;
          # SMOKE=1 synthetic; SMOKE=real = real data on Kaggle CPU, stand-in model, FVB_STANDIN_N rows per part
    FDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p,duongphihoang/flag2027-audio-cache-p,chnnhnnguyn/external-data-mavceleb,ngnnguynchn/flag2027-raw-v4-p,ngnnguynchn/flag2027-evalphase-p"
    FK="$ACC/flag-2027-nb-189-fvbind-b2-ib0,$ACC/flag-2027-nb-47rs-screen2-eval"
    python kaggle/build_flag192.py > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_192_fvbx.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-192-fvbx-smoke --smoke "FLAG_SMOKE=1" ${FORCE:+--force}
    elif [ "${SMOKE:-0}" = real ]; then
      $K push kaggle/FLAG_192_fvbx.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-192-fvbx-realsmoke --datasets "$FDS" --kernels "$FK" --env "FVB_STANDIN=1;FVB_STANDIN_N=${N:-150}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_192_fvbx.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-192-fvbx${SUF:-}" --gpu --internet --datasets "$FDS" --kernels "$FK" ${FORCE:+--force}
    fi ;;
  193)   # FVBIND-01 gate 2 lab (FLAG_193 = FLAG_187 with ARMS8 c0 / fvb, tables rfb*); ACC=thanhhuyen11 (reads the FLAG_192 output), GPU
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 8 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_193_fvbg2.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-193-fvbg2-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_193_fvbg2.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-193-fvbg2 --gpu --datasets "$LABDS,$XDS" --kernels "${FVBX:-$ACC/flag-2027-nb-192-fvbx}" ${FORCE:+--force}
    fi ;;
  194)   # FVBIND-01 gate 2 report (CPU, FLAG_194): FLAG_193 of ACC as kernel input; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 10 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_194_fvbg2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-194-fvbg2-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_194_fvbg2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-194-fvbg2-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-193-fvbg2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  195)   # FUSEDIM-01 (14/11, Experiment/FUSEDIM-01/README.md): RFS-LAB, ARMS9 = c5rl width 64 .. 2048 (FLAG_195, tables rfd*); ACC + PART=1|2 (GPU);
          # SMOKE=1 synthetic bundle (PART unset = every arm), 40 steps
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 9 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_195_fusedim.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-195-fusedim-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40${PART:+;RFST_PART=$PART}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_195_fusedim.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-195-fusedim-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS" --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  196)   # FUSEDIM-01 report (CPU, FLAG_196): the two FLAG_195 runs of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 11 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_196_fusedim_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-196-fusedim-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_196_fusedim_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-196-fusedim-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-195-fusedim-p1,$ACC/flag-2027-nb-195-fusedim-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  197)   # PEFACE-01 stage 1a report (CPU, FLAG_197; README "Bac 1a"): FLAG_193 (rfbc0) of ACC as kernel input + embedded PE-Core CA tables
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 12 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_197_pe1a_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-197-pe1a-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_197_pe1a_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-197-pe1a-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-193-fvbg2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  198)   # FUSEDIM-02 (15/11, Experiment/FUSEDIM-02/README.md): RFS-LAB, ARMS10 = c5rl Matryoshka (FLAG_198, tables rfm*); ACC + PART=1|2 (GPU);
          # SMOKE=1 synthetic bundle (PART unset = every arm), 40 steps
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 10 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_198_mrl.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-198-mrl-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40${PART:+;RFST_PART=$PART}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_198_mrl.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-198-mrl-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS" --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  199)   # FUSEDIM-02 report (CPU, FLAG_199): the two FLAG_195 runs + the two FLAG_198 runs of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 13 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_199_fusedim2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-199-fusedim2-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_199_fusedim2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-199-fusedim2-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-195-fusedim-p1,$ACC/flag-2027-nb-195-fusedim-p2,$ACC/flag-2027-nb-198-mrl-p1,$ACC/flag-2027-nb-198-mrl-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  205)   # RFXARCH-01 (15/11, Experiment/RFXARCH-01/README.md): RFS-LAB, ARMS11 = c5rl arch res / idw (FLAG_205, tables rfa*); ACC + PART=1|2 (GPU); SMOKE=1 synthetic
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 11 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_205_rfxarch.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-205-rfxarch-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40${PART:+;RFST_PART=$PART}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_205_rfxarch.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-205-rfxarch-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS" --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  206)   # RFXARCH-01 report (CPU, FLAG_206): the two FLAG_205 runs of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 14 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_206_rfxarch_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-206-rfxarch-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_206_rfxarch_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-206-rfxarch-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --kernels "$ACC/flag-2027-nb-205-rfxarch-p1,$ACC/flag-2027-nb-205-rfxarch-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  207L)  # LORAFIX-01 (15/11, Experiment/LORAFIX-01/README.md): FLAG_163 mode L with RFS_LORA_FIX=1; V=1 (1 200 steps) | V=2 (3 600 steps); GPU + Internet
    STEPS=$([ "${V:?V}" = 2 ] && echo 3600 || echo 1200)
    $K push kaggle/FLAG_163_rfs.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-163-rfs-lfix$V" --gpu --internet --env "RFS_MODE=L;RFS_UPLOAD=$ACC/flag2027-rfs-lorafix$V-p;RFS_LORA_FIX=1;RFS_LORA_STEPS=$STEPS" --datasets "ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,duongphihoang/flag2027-audio-cache-p,chnnhnnguyn/external-data-mavceleb,ngnnguynchn/flag2027-raw-v4-p" ${FORCE:+--force} ;;
  207)   # LORAFIX-01 lab (FLAG_207, tables rflL<V>): c5rl with the LoRA dataset of V in place of ngnnguynchn/flag2027-rfs-lorafix1-p; ACC + V=1|2 (GPU); SMOKE=1 synthetic
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    LABDS=$(echo "$LABDS" | sed "s#ngnnguynchn/flag2027-rfs-lorafix1-p#${ACC:?ACC}/flag2027-rfs-lorafix${V:-1}-p#")
    python kaggle/build_flag173.py 12 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_207_lorafix.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-207-lorafix-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_207_lorafix.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-207-lorafix-l${V:?V}" --gpu --datasets "$LABDS,$XDS" --env "RFST_ONLY=L$V" ${FORCE:+--force}
    fi ;;
  208)   # RFXFUSE-01 (+ LORAFIX-01 phase B) report (CPU, FLAG_208): FLAG_195 runs of ACC (rfdc0) [+ the two FLAG_207 runs when PHASEB=1]; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 15 > /dev/null
    KS="$ACC/flag-2027-nb-195-fusedim-p1"; [ "${PHASEB:-0}" = 1 ] && KS="$KS,$ACC/flag-2027-nb-207-lorafix-l1,$ACC/flag-2027-nb-207-lorafix-l2"
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_208_rfxfuse_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-208-rfxfuse-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_208_rfxfuse_report.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-208-rfxfuse-report$([ "${PHASEB:-0}" = 1 ] && echo -b)" --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --kernels "$KS" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  212)   # FUSEARCH-01 report (CPU, FLAG_212): FLAG_195 / 198 / 207 runs of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 16 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_212_fusearch_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-212-fusearch-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_212_fusearch_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-212-fusearch-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --kernels "$ACC/flag-2027-nb-195-fusedim-p1,$ACC/flag-2027-nb-195-fusedim-p2,$ACC/flag-2027-nb-198-mrl-p1,$ACC/flag-2027-nb-198-mrl-p2,$ACC/flag-2027-nb-207-lorafix-l1,$ACC/flag-2027-nb-207-lorafix-l2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  213)   # FUSEARCH-02 (15/11, Experiment/FUSEARCH-02/README.md): RFS-LAB, ARMS13 = c5rl seeds 4-6 / 7-9 (FLAG_213, tables rfr*); ACC + PART=1|2 (GPU);
          # SMOKE=1 synthetic bundle (PART unset = every arm), 40 steps
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 13 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_213_seedrep.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-213-seedrep-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40${PART:+;RFST_PART=$PART}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_213_seedrep.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-213-seedrep-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS" --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  214)   # FUSEARCH-02 report (CPU, FLAG_214): FLAG_195 p1 + the two FLAG_213 runs of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 17 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_214_fusearch2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-214-fusearch2-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_214_fusearch2_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-214-fusearch2-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T"         --kernels "$ACC/flag-2027-nb-213-seedrep-p1,$ACC/flag-2027-nb-213-seedrep-p2$([ "${REP0:-1}" = 1 ] && echo ,$ACC/flag-2027-nb-195-fusedim-p1)" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  215)   # FUSEARCH-03 J01 export (CPU, FLAG_215): FLAG_195 p1 of ACC as kernel input; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 18 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_215_fusearch3_export.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-215-fusearch3-export-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_215_fusearch3_export.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-215-fusearch3-export --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --kernels "$ACC/flag-2027-nb-195-fusedim-p1" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  216)   # LANGLANE-01 (15/11, Experiment/LANGLANE-01/README.md): RFS-LAB, ARMS16 = c5rl seeds 10-12 (FLAG_216, table rfrc0t); ACC (GPU, one run);
          # SMOKE=1 synthetic bundle, 40 steps. Push on the account of the FLAG_213 runs (FLAG_217 reads all three as kernel outputs)
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 16 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_216_seedrep3.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-216-seedrep3-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_216_seedrep3.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-216-seedrep3}" --gpu --datasets "$LABDS,$XDS" ${FORCE:+--force}
    fi ;;
  217)   # LANGLANE-01 part B report (CPU, FLAG_217): the two FLAG_213 runs + the FLAG_216 run of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 19 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_217_langlane_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-217-langlane-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_217_langlane_report.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-217-langlane-report}" --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --kernels "$ACC/flag-2027-nb-213-seedrep-p1,$ACC/flag-2027-nb-213-seedrep-p2,$ACC/${K216:-flag-2027-nb-216-seedrep3}" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  220)   # RFXSTD-01 (16/11, Experiment/RFXSTD-01/README.md): RFS-LAB, ARMS20 = c5rl training-loop knobs x 2 replicates + control rerun (FLAG_220, tables rfk*);
          # ACC + PART=1|2 (GPU); SMOKE=1 synthetic bundle (PART unset = every arm), 40 steps. Push on the account of the FLAG_213 runs (FLAG_221 reads them)
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag173.py 20 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_220_rfxstd.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-220-rfxstd-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40${PART:+;RFST_PART=$PART}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_220_rfxstd.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-220-rfxstd-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS" --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  221)   # RFXSTD-01 report (CPU, FLAG_221): the two FLAG_213 runs + the two FLAG_220 runs of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py 21 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_221_rfxstd_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-221-rfxstd-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_221_rfxstd_report.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-221-rfxstd-report}" --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --kernels "$ACC/flag-2027-nb-213-seedrep-p1,$ACC/flag-2027-nb-213-seedrep-p2,$ACC/flag-2027-nb-220-rfxstd-p1,$ACC/flag-2027-nb-220-rfxstd-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  222)   # FUSENET-01 export (CPU, FLAG_222, FUSENET_MODE=export): FLAG_195 p1 of ACC as kernel input (as 215); SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1};FUSENET_MODE=export"
    python kaggle/build_flag174.py 22 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_222_fusenet_export.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-222-fusenet-export-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_222_fusenet_export.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-222-fusenet-export --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --kernels "$ACC/flag-2027-nb-195-fusedim-p1" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  219)   # FUSENET-01 report (CPU, FLAG_219, Experiment/FUSENET-01/README.md): the two FLAG_213 runs + the FLAG_216 run of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}${FMODE:+;FUSENET_MODE=$FMODE}"
    python kaggle/build_flag174.py 20 > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_219_fusenet_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-219-fusenet-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_219_fusenet_report.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-219-fusenet-report}" --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" --kernels "$ACC/flag-2027-nb-213-seedrep-p1,$ACC/flag-2027-nb-213-seedrep-p2,$ACC/${K216:-flag-2027-nb-216-seedrep3}" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  218)   # LANGLANE-01 part A (FLAG_218, CPU): English suite backtest; inputs as 135
    python kaggle/build_flag218.py > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then x=(--smoke "FL3Z_SMOKE=1"); s=-smoke; else x=(--env "LAB_FOLDS="); s=; fi
    $K push kaggle/FLAG_218_langlane_bt.ipynb --account "${ACC:?ACC}" --slug "${SLUG:-flag-2027-nb-218-langlane-bt$s}" \
      --datasets "$DSC,$TAB,$CTB,$CTT,$AVF" "${x[@]}" ${FORCE:+--force} ;;
  190)   # WHY-01 part B (14/11, Experiment/WHY-01/README.md): RFS-LAB, ARMS_WHY of kaggle/build_flag190.py (tables rfw*); ACC + PART=1|2 (GPU);
          # SMOKE=1 synthetic bundle, every arm (PART unset), 40 steps; SMOKE=real = real data on Kaggle CPU, fold 1, arms c0 + samevid (ONLY=), 40 steps
    LABDS="ngnnguynchn/flag2027-rfs-bundle-p,ngnnguynchn/flag2027-lid-strict-p,ngnnguynchn/flag2027-rfs-lorafix1-p,ngnnguynchn/flag2027-ct02s-r2ca-tables-p"
    XDS="quynhnhu82/flag2027-feats-ls-c5s-p,trngsnam/flag2027-feats-peface01-pe0-p,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-feats-vox2b-p,ngnnguynchn/flag2027-lab-vb-p,hthlin/flag2027-aux-voice-feats-p,ngnnguynchn/flag2027-feats-screen2-p,nguynchnngn1/flag2027-feats-rfs-a-p,phanththui/flag2027-feats-rfs-b-p,phanvnnnnn6789/flag2027-feats-rfs-c-p"
    python kaggle/build_flag190.py > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_190_why.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-190-why-smoke --smoke "FLAG_SMOKE=1;RFS_STEPS=40${PART:+;RFST_PART=$PART}" ${FORCE:+--force}
    elif [ "${SMOKE:-0}" = real ]; then
      $K push kaggle/FLAG_190_why.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-190-why-realsmoke --datasets "$LABDS,$XDS"         --env "RFS_STEPS=40;RFS_LAB_FOLDS=1;RFST_ONLY=${ONLY:-c0,samevid}" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_190_why.ipynb --account "${ACC:?ACC}" --slug "flag-2027-nb-190-why-p${PART:?PART}" --gpu --datasets "$LABDS,$XDS"         --env "RFST_PART=$PART" ${FORCE:+--force}
    fi ;;
  191)   # WHY-01 report (CPU, FLAG_191): ct02s-g1 tables (A3) + rfzc0 tables of FLAG_187 (dataset RFZ) + the two 190 runs of ACC as kernel inputs;
          # PHASE=A = part A only (no 190 run needed); SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    G1=ngnnguynchn/flag2027-ct02s-g1-tables-p; RFZ=${RFZ:-phanththui/flag2027-why01-rfzc0-tables-p}
    python kaggle/build_flag191.py > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_191_why_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-191-why-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$G1"         --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    elif [ "${PHASE:-}" = A ]; then
      $K push kaggle/FLAG_191_why_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-191-why-phasea --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T,$G1,$RFZ"         --env "LAB_FOLDS=;WHY_PHASE=A;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_191_why_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-191-why-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T,$G1,$RFZ"         --kernels "$ACC/flag-2027-nb-190-why-p1,$ACC/flag-2027-nb-190-why-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  174)   # RFSTUNE-01 report (CPU, FLAG_174): datasets of 171 minus EDA3, + the two 173 runs of ACC as kernel inputs; SMOKE=1 = stand-ins
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag174.py > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_174_rfstune_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-174-rfstune-report-smoke --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" \
        --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_174_rfstune_report.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-174-rfstune-report --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T" \
        --kernels "$ACC/flag-2027-nb-173-rfstune-p1,$ACC/flag-2027-nb-173-rfstune-p2" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  171)   # EDA-3FAM-01 (5/11, Experiment/EDA-3FAM-01/README.md): CPU report; SMOKE=1 = stand-ins for the T3S / EDA tables; else ACC + T3T + EDA3
    CTB=phanvnhtrng/flag2027-feats-ls-b3s-p; CTC=quynhnhu82/flag2027-feats-ls-c5s-p
    SD=ngnnguynchn/flag2027-lid-strict-p; G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; SCT=ngnnguynchn/flag2027-feats-scout-wsvrdn-p
    T3T=${T3T:-ngnnguynchn/flag2027-t3s01-tables-p}; EDA3=${EDA3:-ngnnguynchn/flag2027-eda3fam-tables-p}
    T3ENV="LID_STRICT=1;T3S=1;CT_R2_ONLY=CA;T3S_PER_GPU=1;NOREL=${NOREL:-1}"
    python kaggle/build_flag171.py > /dev/null
    if [ "${SMOKE:-0}" = 1 ]; then
      $K push kaggle/FLAG_171_eda3fam.ipynb --account "${ACC:-anhkieeu}" --slug ${SLUG:-flag-2027-nb-171-eda3fam-smoke}         --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT" --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;BN42_STANDIN=1;CT_STANDIN=1;LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    else
      $K push kaggle/FLAG_171_eda3fam.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-171-eda3fam         --datasets "$DSC,$CTB,$TAB,$AVF,$G3X,$SD,$SCT,$T3T,$EDA3" --env "LAB_FOLDS=;$T3ENV" ${FORCE:+--force}
    fi ;;
  155|155q|155f0|155f)   # CLEANTUNE-02 round 2 + QFEMA-01 report + F2s (3/11, nhat ky 364; Experiment/CLEANTUNE-02/README.md §1b, §2)
         # 155 : round 2 (FLAG_155, plan = Experiment/CLEANTUNE-02/ct_round2.json): SMOKE=1 | ACCS = 3 accounts f12 f34 f5 | STEP=r1: RUNS = the 3 GPU runs -> $CT2 tables |
         #       STEP=r2: CPU report (ACC) -> ct_decision.json.   CTB / CTC = B3s / C5s (export), LS data always attached.
         # 155q: QFEMA-01 report (FLAG_127q, CPU, ACC) on the round-1 per-group tables $CTT (+ QX = round-2 tables dataset, QT=ct01r2 = only that tag, SLUGQ = slug); SMOKE=1: stand-in
         # 155f0: F0s zip (output of $F0K, default locnguyentruongan/flag-2027-nb-112s-f0s) -> private dataset $F0ZD (English bytes for F2s)
         # 155f: F2s final head + zip (FLAG_112_f1_f2s, built with CT_PLAN / NBSUF; ACC = account, must read $CCE, $CT2, $F0ZD)
    CT2=${CT2:-ngnnguynchn/flag2027-ct02-r2-tables-p}; SD=ngnnguynchn/flag2027-lid-strict-p; F0ZD=${F0ZD:-ngnnguynchn/flag2027-ct02-f0s-zip-p}
    G3X="$CTC,dnq1209/flag2027-feats-tap01-voice-p,$SCR2,$VF"; QEV=${QE-;T3R_QF=1;T3R_EMA=0.9995}
    case "$STEP_" in
      155q)
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_127q_qfema.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-127q-qfema-smoke --datasets "$DSC,$TAB,$CTB,$AVF" --smoke 'FL3Z_SMOKE=1;QE_STANDIN=1'
        else
          $K privatize "$CTT"
          $K push kaggle/FLAG_127q_qfema.ipynb --account "${ACC:?ACC}" --slug "${SLUGQ:-flag-2027-nb-127q-qfema}" --datasets "$DSC,$TAB,$CTB,$AVF,$CTT${QX:+,$QX}" --env "LAB_FOLDS=${QT:+;QE_TAGS=$QT}"
        fi ;;
      155f0)
        d=$(mktemp -d); need_complete "${F0K:-locnguyentruongan/flag-2027-nb-112s-f0s}"
        $K output "${F0K:-locnguyentruongan/flag-2027-nb-112s-f0s}" --dest "$d/o" --pattern "submission_cleantune01_F0_dev\.zip" > /dev/null
        mkdir -p "$d/z"; find "$d/o" -name "submission_cleantune01_F0_dev.zip" -exec cp {} "$d/z/" \;
        [ "$(ls "$d/z" | wc -l)" -eq 1 ] || { echo "!! F0s zip not found"; exit 1; }
        (cd "$d" && python -c "import zipfile,pathlib; z=zipfile.ZipFile('f0zip.zip','w'); [z.write(f, 'f0/'+f.name+'.bin') for f in pathlib.Path('z').glob('*.zip')]; z.close()")   # .zip.bin: Kaggle unpacks nested .zip
        $K dscreate "$(cygpath -w "$d/f0zip.zip" 2>/dev/null || echo "$d/f0zip.zip")" --account ngnnguynchn --slug "${F0ZD#*/}" ;;
      155f)
        CCE=${CCE:-ngnnguynchn/flag2027-cleancomp-e2e-p}; DSCC=$(echo "$DSC" | tr ',' '\n' | grep -v 'flag2027-e2e-dev-p' | paste -sd, -)
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_112_f1_f2s.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-155f-f2s-smoke \
            --datasets "$DSCC,$CCE,$CTB,$TAB,${CT2},$AVF,$G3X,$SD,$F0ZD" --smoke 'FL3Z_SMOKE=1;TR_MAX_STEPS=2;CT_FINAL_SEEDS=1;BN42_STANDIN=1;LID_STRICT=1;CLEANCOMP=1;CT_STANDIN=1'
        elif ! timeout 200 $K access "${ACC:?ACC}" "$CCE" 2>&1 | grep -q ": readable"; then
          echo "!! 155f NOT pushed (0 pushed): $CCE not readable by $ACC"; exit 1
        else
          $K push kaggle/FLAG_112_f1_f2s.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-155f-f2s --gpu \
            --datasets "$DSCC,$CCE,$CTB,$TAB,$CT2,$AVF,$G3X,$SD,$F0ZD" --env "LID_STRICT=1;CLEANCOMP=1"
        fi ;;
      155)
        ds="$DSC,$CTB,$TAB,$AVF,$G3X,$SD"
        if [ "${SMOKE:-0}" = 1 ]; then
          $K push kaggle/FLAG_155_ct02_r2.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-155-ct02-r2-smoke --datasets "$ds" \
            --smoke "FL3Z_SMOKE=1;TR_MAX_STEPS=2;TR_SEEDS=1;LAB_FOLDS=1;BN42_STANDIN=1;CT_STANDIN=1;LID_STRICT=1${QEV}"
        elif [ "${STEP:-}" = r1 ]; then
          gather_tables ct01r2 "${CT2#*/}" ${RUNS:?RUNS = the 3 FLAG_155 GPU runs}
        elif [ "${STEP:-}" = r2 ]; then
          $K privatize "$CT2"
          $K push kaggle/FLAG_155_ct02_r2.ipynb --account "${ACC:?ACC}" --slug flag-2027-nb-155r-ct02-report --datasets "$ds,$CT2" --env "LAB_FOLDS=;LID_STRICT=1"
        else
          a=(${ACCS:?ACCS = 3 accounts: f12 f34 f5}); f=("1,2" "3,4" "5"); s=(f12 f34 f5)
          for i in 0 1 2; do
            $K push kaggle/FLAG_155_ct02_r2.ipynb --account "${a[$i]}" --slug "flag-2027-nb-155-ct02-r2-${s[$i]}" --gpu \
              --datasets "$ds" --env "LAB_FOLDS=${f[$i]};LID_STRICT=1${QEV}"
          done
        fi ;;
    esac ;;
  evalu-heads|65xbs|65xcs|47rs|50cc|112f0sx|112f1sx) # NOREL-01 (5/11): 50cc / 112sx default NOREL=1 (BTC FAQ, no rel / relg / t3_relg). EVALPATH-U (3/11, nhat ky 363, Experiment/EVALPATH-U/README.md, docs/PLAN_EVAL_U_2026-11-11.md): Evaluation path of F1s,
         # R11 on dev. ONE account (EA, default thanhhuyen11) for the extraction + 50cc + 112sx kernels (kernel outputs are read on the same account). Env:
         #   EVALDS = organisers' Evaluation data (dataset), ONLY_UNDER = eval (patched into the notebook copy), SUF = slug suffix (r11u; use eval on 11/11),
         #   SMOKE=1 = Kaggle CPU wiring run, E2E_SPLIT = dev | eval (112sx), KX = kernels for 112sx, GPU=1 for 112sx on GPU.
         # The notebooks are the existing ones (65xb / 65xc / 47b / FLAG_50 / 112_f0x / 112_f1x) copied to a temp file and patched with sed:
         # ONLY_UNDER, and the B3c / C5c dataset names -> B3s / C5s. Nothing in the originals changes.
    EA=${EA:-thanhhuyen11}; SUF=${SUF:-r11u}; SPL=${E2E_SPLIT:-dev}
    B3D=${B3D:-phanvnhtrng/flag2027-feats-ls-b3s-p}; C5D=${C5D:-quynhnhu82/flag2027-feats-ls-c5s-p}
    SD=ngnnguynchn/flag2027-lid-strict-p; CCE=${CCE:-ngnnguynchn/flag2027-cleancomp-e2e-p}; CCF=ngnnguynchn/flag2027-cleancomp-final-p
    HF0=ngnnguynchn/flag2027-evalu-f0s-p; HF1=ngnnguynchn/flag2027-evalu-f1s-p
    D65="$MAVRAW,ngnnguynchn/flag2027-lab-data-p,ngnnguynchn/flag2027-lab-vb-p,ngnnguynchn/flag2027-feats-vox2b-p,onnhqunhh/flag2027-lora-frames-p"
    TD=$(mktemp -d); P_="flag-2027-nb"
    mk() {   # mk <notebook> [sed-expr...]: temp copy in $TD (ONLY_UNDER applied when set)
      local nb=$1; shift; cp "kaggle/$nb" "$TD/$nb"
      if [ -n "${ONLY_UNDER:-}" ]; then sed -i "s/ONLY_UNDER = None /ONLY_UNDER = '${ONLY_UNDER}'/" "$TD/$nb"; fi
      for e in "$@"; do sed -i "$e" "$TD/$nb"; done; }
    case "$STEP_" in
      evalu-heads)   # F0s / F1s outputs (locnguyentruongan) -> two private datasets (a kernel of another account cannot be attached): saved models + dev reference zip
        for h in f0s:ct01g1:F0 f1s:ct01r2:F1_CA; do
          IFS=: read -r n ct z <<< "$h"
          need_complete locnguyentruongan/flag-2027-nb-112s-$n
          d=$(mktemp -d); for pt in "${ct}_final_models" "submission_cleantune01_${z}_dev"; do $K output locnguyentruongan/flag-2027-nb-112s-$n --dest "$d/o" --pattern "$pt" > /dev/null; done   # no '|' in the pattern (Windows)
          mkdir -p "$d/p"; find "$d/o" \( -name "${ct}_final_models.pt" -o -name "submission_cleantune01_${z}_dev.zip" \) -exec cp {} "$d/p/" \;
          [ "$(ls "$d/p" | wc -l)" = 2 ] || { echo "!! $n: expected ${ct}_final_models.pt + the dev zip, got: $(ls "$d/p")"; exit 1; }
          (cd "$d" && python -c "import zipfile,pathlib; z=zipfile.ZipFile('h.zip','w'); [z.write(f, f.name) for f in sorted(pathlib.Path('p').iterdir())]; z.close()")
          $K dscreate "$(cygpath -w "$d/h.zip")" --account ngnnguynchn --slug "flag2027-evalu-$n-p"
        done; echo "then: $K privatize $HF0 ; $K privatize $HF1" ;;
      65xbs) mk FLAG_65xb_clipb3c_extract.ipynb 's/ct\[-_\]b3c\[-_\]p/ls[-_]b3s[-_]p/g' 's#ngnnguynchn/flag2027-feats-ct-b3c-p#'"$B3D"'#g'
         if [ "${SMOKE:-0}" = 1 ]; then   # stand-in: the saved b3c dataset (same layout) with the ORIGINAL notebook
           $K push kaggle/FLAG_65xb_clipb3c_extract.ipynb --account "$EA" --slug flag-2027-nb-65xbs-smoke --internet --datasets "$D65,ngnnguynchn/flag2027-feats-ct-b3c-p" --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6'
         else $K push "$TD/FLAG_65xb_clipb3c_extract.ipynb" --account "$EA" --slug "$P_-65xbs-b3sx-$SUF" --gpu --internet --datasets "$D65,$B3D${EVALDS:+,$EVALDS}"; fi ;;
      65xcs) mk FLAG_65xc_clipc5c_extract.ipynb 's/ct\[-_\]c5c\[-_\]p/ls[-_]c5s[-_]p/g' 's#nammo0/flag2027-feats-ct-c5c-p#'"$C5D"'#g'
         if [ "${SMOKE:-0}" = 1 ]; then
           $K push kaggle/FLAG_65xc_clipc5c_extract.ipynb --account "$EA" --slug flag-2027-nb-65xcs-smoke --internet --datasets "$D65,nammo0/flag2027-feats-ct-c5c-p" --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6'
         else $K push "$TD/FLAG_65xc_clipc5c_extract.ipynb" --account "$EA" --slug "$P_-65xcs-c5sx-$SUF" --gpu --internet --datasets "$D65,$C5D${EVALDS:+,$EVALDS}"; fi ;;
      47rs)  mk FLAG_47b_voice_screen.ipynb    # rn293 / ecapa1024: no adapter involved, so the notebook is the original (only ONLY_UNDER / the Evaluation data differ)
         if [ "${SMOKE:-0}" = 1 ]; then $K push kaggle/FLAG_47b_voice_screen.ipynb --account "$EA" --slug flag-2027-nb-47rs-smoke --internet --datasets "$MAVRAW" --smoke 'FLAG_SMOKE=1;FLAG_SMOKE_ROWS=6'
         else $K push "$TD/FLAG_47b_voice_screen.ipynb" --account "$EA" --slug "$P_-47rs-screen2-$SUF" --gpu --internet --datasets "$MAVRAW${EVALDS:+,$EVALDS}"; fi ;;
      50cc)  # FLAG_50 (S6 + T3 v11 components of the test files) with the strict members: CLEANCOMP-01 final (s007_xall.pt, v11 head reused); XALL_PREFER picks it over best-v7
         # no smoke mode in FLAG_50 (pip cell checked 29/10, BEST/v1977 s6): the dev run IS the check (about 3.5 h, T4 x2)
         mk FLAG_50_e2e_v11.ipynb
         ds="ngnnguynchn/flag2027-raw-v4-p,ngnnguynchn/flag2027-members-p,ngnnguynchn/flag2027-eda-bundle-p,ngnnguynchn/flag2027-best-v7-p,ngnnguynchn/flag2027-lab-data-p,$CCE,$CCF,$SD${EVALDS:+,$EVALDS}"
         $K push "$TD/FLAG_50_e2e_v11.ipynb" --account "$EA" --slug "$P_-50cc-e2e-$SUF" --gpu --internet --datasets "$ds" --env "LID_STRICT=1;CLEANCOMP=1;XALL_PREFER=cleancomp;NOREL=${NOREL:-1}" ;;
      112f0sx|112f1sx)   # saved F0s / F1s heads (datasets $HF0 / $HF1) on the test files of E2E_SPLIT. KX = kernels of $EA: f0sx "<65xbs> <50cc>", f1sx "<112f0sx> <65xbs> <65xcs> <47rs> <50cc>"
         kx=$(echo ${KX:?KX = kernels of EA} | tr " " ",")
         DSCC=$(echo "$DSC" | tr ',' '\n' | grep -v 'flag2027-e2e-dev-p' | paste -sd, -)    # the E2E dir is the 50cc output, never the old bundle
         if [ "$STEP_" = 112f0sx ]; then nb=FLAG_112_f0x.ipynb; ds="$DSCC,$TAB,${CTTS:-ngnnguynchn/flag2027-ct02s-g1-tables-p},$AVF,$SD,$HF0"
         else nb=FLAG_112_f1x.ipynb; ds="$DSCC,$TAB,${CT2S:-ngnnguynchn/flag2027-ct02s-r2ca-tables-p},$AVF,dnq1209/flag2027-feats-tap01-voice-p,$VF,$SD,$HF1"; fi
         if [ "${SMOKE:-0}" = 1 ]; then
           $K push kaggle/$nb --account "$EA" --slug "$P_-${STEP_}-smoke" --kernels "$kx" --datasets "$ds" --smoke "FL3Z_SMOKE=1;FLAG_SMOKE=1;LID_STRICT=1;CLEANCOMP=1;NOREL=${NOREL:-1}"
         else
           $K push kaggle/$nb --account "$EA" --slug "$P_-${STEP_}-$SUF" $([ "${GPU:-0}" = 1 ] && echo --gpu) --kernels "$kx" --datasets "$ds" ${FORCE:+--force} \
             --env "E2E_SPLIT=$SPL;LID_STRICT=1;CLEANCOMP=1;NOREL=${NOREL:-1}${XENV:+;$XENV}"
         fi ;;
    esac ;;
  *) echo "steps: status | 86t | 86s | 86g | 92 | 92r | 93 | 93r | csls | 94 | 94r | 86w | 67x | 67xd | 97 | 97r | 67b | 67y | 67yd | 67zp | 67zpd | 67zd | 67zdd | 98 | 98r | 67c | 67d | 67dh | 65c | 65j | 65j2   (ACC= / ACCS= / KERNELS= override accounts and inputs; GPUT=NvidiaTeslaP100 picks the GPU type; J1CK=<owner/dataset> runs 65j2 from a checkpoint dataset on another account)"; exit 2 ;;
esac
