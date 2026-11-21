#!/usr/bin/env bash
# Scores the one-image setting (M9) at fixed registration and alignment-tolerant, then summarises.
# Needs run_m9.sh and run_m7.sh (M7/M8 rows) to have finished.   bash scripts/run_m9_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
P=(); for s in 0 1 2; do
  P+=(--pred m9_s$s=outputs/m9/preds_ct_s$s.npz --pred m9c_s$s=outputs/m9/preds_lab_s$s.npz)
  P+=(--pred m9w_s$s=outputs/m9/preds_ctw_s$s.npz --pred m9cw_s$s=outputs/m9/preds_labw_s$s.npz)
  P+=(--pred m9a_s$s=outputs/m9/preds_ct1_s$s.npz --pred m9ac_s$s=outputs/m9/preds_lab1_s$s.npz)
done
P+=(--pred copy_s0=outputs/m9/preds_copy.npz --pred labwarp_s0=outputs/m9/preds_labwarp.npz)
python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --allow-partial --out outputs/trusted_m9.json
python -I scripts/eval_aligned.py --no-b0 --out outputs/trusted_aligned_m9.json \
  --pred "m9=outputs/m9/preds_ct_s{s}.npz" --pred "m9c=outputs/m9/preds_lab_s{s}.npz" \
  --pred "m9w=outputs/m9/preds_ctw_s{s}.npz" --pred "m9cw=outputs/m9/preds_labw_s{s}.npz" \
  --pred "m9a=outputs/m9/preds_ct1_s{s}.npz" --pred "m9ac=outputs/m9/preds_lab1_s{s}.npz"
python -I scripts/eval_aligned.py --no-b0 --seeds 0 --append --out outputs/trusted_aligned_m9.json \
  --pred "copy=outputs/m9/preds_copy.npz" --pred "labwarp=outputs/m9/preds_labwarp.npz"
python -I scripts/summarize_m9.py
echo "run_m9_eval done"
