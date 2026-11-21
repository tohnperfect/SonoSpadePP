#!/usr/bin/env bash
# M5 and M5c at 128 px with the deformable registration (both splits, three GAN seeds), starting from
# the deformable M4 / M4c; scored next to the other 128 px deformable rows.   bash scripts/run_m5_def.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
for split in halves kidney; do
  ( for mode in hu const; do
      python -I scripts/train_m5.py --shards data/trusted/pairs_def --pop-suffix def --out outputs/m5def \
        --mode $mode --split $split --seeds 0 1 2
    done ) > logs/train_m5def_${split}.log 2>&1 &
done
wait
grep -h "patient fits" logs/train_m5def_halves.log logs/train_m5def_kidney.log
P=()
for s in 0 1 2; do
  P+=(--pred "m5_s${s}=outputs/m5def/preds_hu_halves_s${s}.npz" --pred "m5c_s${s}=outputs/m5def/preds_const_halves_s${s}.npz")
  P+=(--pred "m5k_s${s}=outputs/m5def/preds_hu_kidney_s${s}.npz" --pred "m5kc_s${s}=outputs/m5def/preds_const_kidney_s${s}.npz")
done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --allow-partial --append --out outputs/trusted_def.json
python scripts/summarize_hires.py
echo "run_m5_def done"
