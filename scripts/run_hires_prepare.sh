#!/usr/bin/env bash
# Full-resolution / deformable planes for every kidney, in three parallel workers, then merge the
# 128 px deformable shards.   bash scripts/run_hires_prepare.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
KIDS=$(python3 -c "import json; print(' '.join(sorted(json.load(open('data/trusted_pairs_meta.json'))['kidneys'])))")
set -- $KIDS
A=(); B=(); C=(); i=0
for k in "$@"; do case $((i % 3)) in 0) A+=("$k");; 1) B+=("$k");; 2) C+=("$k");; esac; i=$((i+1)); done
python -I scripts/trusted_hires_prepare.py --kids "${A[@]}" > logs/hires_prep_0.log 2>&1 &
python -I scripts/trusted_hires_prepare.py --kids "${B[@]}" > logs/hires_prep_1.log 2>&1 &
python -I scripts/trusted_hires_prepare.py --kids "${C[@]}" > logs/hires_prep_2.log 2>&1 &
wait
python -I scripts/trusted_hires_prepare.py --merge
grep -h "Dice rigid" logs/hires_prep_*.log | wc -l
echo "hires prepare done"
