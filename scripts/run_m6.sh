#!/usr/bin/env bash
# M6 (spatial CT network) and M6c (labels only) at 128 px with the deformable registration, three GAN
# seeds, patient-level 5-fold CV; scored next to the other deformable 128 px rows.   bash scripts/run_m6.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
for mode in ct lab; do
  python -I scripts/train_m6.py --shards data/trusted/pairs_def --mode $mode --seeds 0 1 2 > logs/train_m6_${mode}.log 2>&1 &
done
wait
grep -h "fold 4\|wrote" logs/train_m6_ct.log logs/train_m6_lab.log
while pgrep -f "scripts/run_m5_def.sh" >/dev/null; do sleep 20; done      # one writer of trusted_def.json at a time
P=(); for s in 0 1 2; do P+=(--pred "m6_s${s}=outputs/m6def/preds_ct_s${s}.npz" --pred "m6c_s${s}=outputs/m6def/preds_lab_s${s}.npz"); done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --append --out outputs/trusted_def.json
python scripts/summarize_hires.py
echo "run_m6 done"
