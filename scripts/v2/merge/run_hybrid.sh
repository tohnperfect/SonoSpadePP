#!/usr/bin/env bash
# H-RX, the hybrid: tier 1 with the retrieved, label-warped
# image of another patient as an extra input, seeds 0-2, folds 0-4, then TRUSTED scoring and the comparison with
# tier 1. Starts after the process given as $1 exits
# (one heavy job at a time). Resumable.
set -e
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
# The study ran in two working copies of this repository: W held the tier-0 data, runs and results, SPP the
# TRUSTED data, runs and outputs. Both default to this repository; set the variables to split them again.
W=${SONOSPADE_V2_ROOT:-$ROOT}; SPP=${SONOSPADEPP_ROOT:-$ROOT}
cd "$SPP"
export PYTORCH_ENABLE_MPS_FALLBACK=1 SONOSPADE_FM_DIR=$W/data/fm
PY="$W/.venv/bin/python -u -I"
S=$W/scripts
echo "$(date '+%F %T') hybrid start"
[ -f outputs/m7/exemplars_train.npz ] || $PY $S/build_exemplars.py > logs/build_exemplars.log 2>&1
[ -f outputs/m7/preds_m7r_s2.npz ] || $PY $S/train_m7.py --mode lab --tag m7r --seeds 0 1 2 \
    --exemplars outputs/m7/exemplars_train.npz --exemplars-test outputs/m9/preds_retrieve.npz > logs/m7r_train.log 2>&1
P=(); for s in 0 1 2; do P+=(--pred m7r_s$s=outputs/m7/preds_m7r_s$s.npz); done
$PY $S/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --allow-partial --append \
    --out outputs/trusted_merge_v2.json > logs/m7r_eval.log 2>&1
$PY $S/eval_aligned.py --no-b0 --append --out outputs/trusted_aligned_merge_v2.json \
    --pred "m7r=outputs/m7/preds_m7r_s{s}.npz" > logs/m7r_aligned.log 2>&1
$PY $S/summarize_merge.py > logs/merge_summary_m7r.log 2>&1
awk '/^## hybrid/{p=1} p&&/^## /&&!/hybrid/{p=0} p' docs/TRUSTED_MERGE.md
echo "$(date '+%F %T') hybrid done"
