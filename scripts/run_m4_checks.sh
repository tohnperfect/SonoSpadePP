#!/usr/bin/env bash
# Robustness of "the learned HU curve adds nothing": retrain M4 with 100x weaker penalties, once from
# zero and once starting at M2's hand-set backscatter rule, then score both next to the other rows.
#   bash scripts/run_m4_checks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -I scripts/train_m4.py --mode hu --tag huweak --w-smooth 1e-4 --w-mag 1e-5 --seeds 0 1 2
python -I scripts/train_m4.py --mode hu --tag hum2 --init m2 --w-smooth 1e-4 --w-mag 1e-5 --seeds 0 1 2
ARGS=()
for s in 0 1 2; do ARGS+=(--pred "m4w_s${s}=outputs/m4/preds_huweak_s${s}.npz" --pred "m4m_s${s}=outputs/m4/preds_hum2_s${s}.npz"); done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs.npz "${ARGS[@]}" --append --out outputs/trusted.json
echo "m4 checks done"
