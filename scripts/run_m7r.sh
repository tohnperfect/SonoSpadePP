#!/usr/bin/env bash
# M7r / M7rc: M7 and M7c trained with a joint registration network (RegGAN-style, training only).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
# auxiliary: wait until seed 0 of the B0t control (run_b0t.sh) has finished, so the GPU goes to it first
until grep -q "b0t seed 0 finished" logs/run_b0t.log 2>/dev/null; do sleep 60; done
for mode in ct lab; do
  python -I scripts/train_m7.py --mode $mode --seeds 0 1 2 --reg-net --tag ${mode}reg --out outputs/m7r > logs/train_m7r_${mode}.log 2>&1 &
done
wait
grep -h "fold\|wrote" logs/train_m7r_ct.log logs/train_m7r_lab.log
P=(); for s in 0 1 2; do P+=(--pred m7r_s$s=outputs/m7r/preds_ctreg_s$s.npz --pred m7rc_s$s=outputs/m7r/preds_labreg_s$s.npz); done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --append --out outputs/trusted_def.json
python -I scripts/eval_aligned.py --no-b0 --append --pred "m7r=outputs/m7r/preds_ctreg_s{s}.npz" --pred "m7rc=outputs/m7r/preds_labreg_s{s}.npz"
python scripts/summarize_hires.py
echo "run_m7r done"
