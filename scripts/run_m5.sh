#!/usr/bin/env bash
# M5, patient-specific HU-to-echo mappings, and its HU-free control, for both splits and three GAN
# seeds; then score them next to every other row.   bash scripts/run_m5.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
while pgrep -f "scripts/run_m4_checks.sh" >/dev/null; do sleep 20; done
# the two splits run side by side (the fits are small and latency-bound on MPS)
for split in halves kidney; do
  ( for mode in hu const; do python -I scripts/train_m5.py --mode $mode --split $split --seeds 0 1 2; done ) \
    > logs/train_m5_${split}.log 2>&1 &
done
wait
grep -h "patient fits" logs/train_m5_halves.log logs/train_m5_kidney.log
ARGS=()
for s in 0 1 2; do
  ARGS+=(--pred "m5_s${s}=outputs/m5/preds_hu_halves_s${s}.npz" --pred "m5c_s${s}=outputs/m5/preds_const_halves_s${s}.npz")
  ARGS+=(--pred "m5k_s${s}=outputs/m5/preds_hu_kidney_s${s}.npz" --pred "m5kc_s${s}=outputs/m5/preds_const_kidney_s${s}.npz")
done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs.npz "${ARGS[@]}" --allow-partial --append --out outputs/trusted.json
python -I scripts/summarize_trusted.py --results outputs/trusted.json --out docs/TRUSTED_TABLE.md
python scripts/paper_macros.py
echo "run_m5 done"
