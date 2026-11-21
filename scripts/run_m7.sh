#!/usr/bin/env bash
# M7 (whole translator fine-tuned on same-patient pairs, with the CT image) and M7c (no CT), then M8 /
# M8c (each fine-tuned per patient on the other half of its sweep); scored at fixed registration and
# alignment-tolerant.   bash scripts/run_m7.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
for mode in ct lab; do
  python -I scripts/train_m7.py --mode $mode --seeds 0 1 2 > logs/train_m7_${mode}.log 2>&1 &
done
wait
grep -h "fold\|wrote" logs/train_m7_ct.log logs/train_m7_lab.log
for mode in ct lab; do
  python -I scripts/adapt_m8.py --mode $mode --seeds 0 1 2 > logs/adapt_m8_${mode}.log 2>&1 &
done
wait
cat logs/adapt_m8_ct.log logs/adapt_m8_lab.log
P=(); for s in 0 1 2; do
  P+=(--pred m7_s$s=outputs/m7/preds_ct_s$s.npz --pred m7c_s$s=outputs/m7/preds_lab_s$s.npz)
  P+=(--pred m8_s$s=outputs/m8/preds_ct_s$s.npz --pred m8c_s$s=outputs/m8/preds_lab_s$s.npz)
done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --append --out outputs/trusted_def.json
python -I scripts/eval_aligned.py --pred "m6=outputs/m6def/preds_ct_s{s}.npz" --pred "m6c=outputs/m6def/preds_lab_s{s}.npz" \
  --pred "m7=outputs/m7/preds_ct_s{s}.npz" --pred "m7c=outputs/m7/preds_lab_s{s}.npz" \
  --pred "m8=outputs/m8/preds_ct_s{s}.npz" --pred "m8c=outputs/m8/preds_lab_s{s}.npz"
python scripts/summarize_hires.py
echo "run_m7 done"
