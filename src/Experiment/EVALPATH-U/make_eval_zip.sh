#!/usr/bin/env bash
# EVALPATH-U: U = F1s + RFS (sum of the two -logit scores, 1:1) from the two ZIPS OF THE SAME SPLIT, then the checks.
#   bash Experiment/EVALPATH-U/make_eval_zip.sh <F1s zip> <RFS zip> <out U zip> [dev | eval] [--map 'devname=evalname,...'] [--counts 'name=n,...'] [--ref <U dev zip>]
# dev:  counts = the four dev files; --ref (optional) = Experiment/UNIFIED-01/submission_unified01_rfs_f1s_dev.zip, Spearman >= 0.999 (R11 of U)
# eval: file names / pair counts unknown until 11/11: pass --counts 'name=n,...' (as in the organisers' list) or read the printed counts and check them by eye.
set -euo pipefail
cd "$(dirname "$0")/../.."
F1=$1; RF=$2; OUT=$3; SPL=${4:-dev}; shift 4 || shift $#
MAP=""; CNT=""; REF=""
while [ $# -gt 0 ]; do case "$1" in --map) MAP=$2; shift 2;; --counts) CNT=$2; shift 2;; --ref) REF=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac; done
C=Experiment/EVALPATH-U/check_eval_zip.py
CA=(); [ "$SPL" = dev ] && CA=(--dev) || { [ -n "$CNT" ] && CA=(--counts "$CNT"); }
echo "== F1s zip";  python $C "$F1" "${CA[@]}" --peer "$RF" | tail -5
echo "== RFS zip";  python $C "$RF" "${CA[@]}" --peer "$F1" | tail -5
python Experiment/XTEAM-03/run.py "$RF" "$F1" "$OUT"
echo "== U zip"
python $C "$OUT" "${CA[@]}" --peer "$F1" "$RF" ${REF:+--ref "$REF" --min 0.999} ${MAP:+--map "$MAP"} | tee /dev/stderr | tail -1 | grep -qx OK
sha256sum "$OUT" | cut -c1-16
