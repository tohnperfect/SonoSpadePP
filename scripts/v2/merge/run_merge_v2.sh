#!/usr/bin/env bash
# Same-patient steps 2-4 with the v2 full model's recipe: B0's recipe plus the USFM-initialized
# S_US (A3, the only tier-0 component that passed). TRUSTED data, runs and
# outputs stay outside git. Step 5 (style codes) is not run: A2 was dropped;
# tier 2 is the one-shot fine-tune only. Resumable.
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
# The study ran in two working copies of this repository: W held the tier-0 data, runs and results, SPP the
# TRUSTED data, runs and outputs. Both default to this repository; set the variables to split them again.
W=${SONOSPADE_V2_ROOT:-$ROOT}; SPP=${SONOSPADEPP_ROOT:-$ROOT}
cd "$SPP"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export SONOSPADE_FM_DIR=$W/data/fm                 # foundation-model weights
PY="$W/.venv/bin/python -u -I"
S=$W/scripts
SUS=runs/s_us_usfm_phys_b0__lab_base.pt
mkdir -p logs outputs/merge_v2 runs/m7
# step 1 (v2 recipe here): S_US from USFM on this repository's sector-warped pairs, v1's recipe otherwise
[ -f $SUS ] || $PY $S/train_s_us.py --pairs data/pairs_train.npz --phys phys_b0 --lab lab_base --init usfm \
    --out $SUS > logs/merge_v2_s_us.log 2>&1
# B0-v2: B0's recipe (run_compare.sh) with that S_US, seeds 0-2
for s in 0 1 2; do
  [ -f runs/b0v2_s$s.pt ] || $PY $S/train_variant.py --variant b0 --pairs data/pairs_train.npz --s-us $SUS \
      --real-dir data/real_pool --iters 1500 --seed $s --out runs/b0v2_s$s.pt > logs/merge_v2_b0_s$s.log 2>&1
done
# step 2: v2 on TRUSTED without fine-tuning (fixed registration and alignment-tolerant)
$PY $S/eval_trusted.py --pairs data/trusted_pairs_def.npz --ckpt b0_v2_s0=runs/b0v2_s0.pt \
    --ckpt b0_v2_s1=runs/b0v2_s1.pt --ckpt b0_v2_s2=runs/b0v2_s2.pt --append --out outputs/trusted_merge_v2.json \
    > logs/merge_v2_eval_b0.log 2>&1
$PY $S/eval_aligned.py --b0 "runs/b0v2_s{s}.pt" --b0-name b0_v2 --append \
    --out outputs/trusted_aligned_merge_v2.json > logs/merge_v2_aligned_b0.log 2>&1
# step 3: B0t-v2 (B0-v2's unpaired recipe on the same TRUSTED planes, approach 1's budget)
[ -f outputs/merge_v2/preds_b0t_v2_s2.npz ] || $PY $S/train_b0t.py --b0 "runs/b0v2_s{seed}.pt" --s-us $SUS \
    --tag b0t_v2 --out outputs/merge_v2 --seeds 0 1 2 > logs/merge_v2_b0t.log 2>&1
# step 4: approach 1 from v2 (M7c started from B0-v2), then tier 2 (one-shot fine-tune of it)
[ -f outputs/merge_v2/preds_m7c_v2_s2.npz ] || $PY $S/train_m7.py --mode lab --b0 "runs/b0v2_s{seed}.pt" \
    --tag m7c_v2 --out outputs/merge_v2 --seeds 0 1 2 > logs/merge_v2_m7c.log 2>&1
[ -f outputs/merge_v2/preds_m9ac_v2_s2.npz ] || $PY $S/adapt_m9a.py --mode lab --b0 "runs/b0v2_s{seed}.pt" \
    --m7-tag m7c_v2 --tag m9ac_v2 --out outputs/merge_v2 --seeds 0 1 2 > logs/merge_v2_m9ac.log 2>&1
P=(); for s in 0 1 2; do
  for n in b0t_v2 m7c_v2 m9ac_v2; do P+=(--pred ${n}_s$s=outputs/merge_v2/preds_${n}_s$s.npz); done
done
$PY $S/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --allow-partial --append \
    --out outputs/trusted_merge_v2.json > logs/merge_v2_eval_preds.log 2>&1
$PY $S/eval_aligned.py --no-b0 --append --out outputs/trusted_aligned_merge_v2.json \
    --pred "b0t_v2=outputs/merge_v2/preds_b0t_v2_s{s}.npz" --pred "m7c_v2=outputs/merge_v2/preds_m7c_v2_s{s}.npz" \
    --pred "m9ac_v2=outputs/merge_v2/preds_m9ac_v2_s{s}.npz" > logs/merge_v2_aligned_preds.log 2>&1
$PY $S/summarize_merge.py > logs/merge_v2_summary.log 2>&1
echo "merge v2 chain done"
