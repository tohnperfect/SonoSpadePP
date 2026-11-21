#!/usr/bin/env bash
# M9 / M9c (one real US image of the patient given, learned correspondence + translator) and M9a / M9ac
# (one-shot fine-tuning of M7 / M7c on that image).   bash scripts/run_m9.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
for mode in ct lab; do
  python -I scripts/train_m9.py --mode $mode --seeds 0 1 2 > logs/train_m9_${mode}.log 2>&1 &
done
for mode in ct lab; do
  python -I scripts/adapt_m9a.py --mode $mode --seeds 0 1 2 > logs/adapt_m9a_${mode}.log 2>&1 &
done
wait
grep -h "fold\|wrote\|seed" logs/train_m9_ct.log logs/train_m9_lab.log logs/adapt_m9a_ct.log logs/adapt_m9a_lab.log
echo "run_m9 done"
