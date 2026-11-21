#!/usr/bin/env bash
# Paired-method checks: (2) the pix2pix baseline on TRUSTED and (3) tier 1
# on the 60 annotated frames (TRUSTED stays outside git). Starts after the process given as $1 exits (one heavy job at a time). Resumable.
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
mkdir -p logs outputs/m7 runs/m7
echo "$(date '+%F %T') paired checks start"
# (2) pix2pix baseline
[ -f outputs/m7/preds_p2p_s2.npz ] || $PY $S/train_m7.py --mode lab --loss p2p --tag p2p --seeds 0 1 2 \
    > logs/p2p_train.log 2>&1
P=(); for s in 0 1 2; do P+=(--pred p2p_s$s=outputs/m7/preds_p2p_s$s.npz); done
$PY $S/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --allow-partial --append \
    --out outputs/trusted_merge_v2.json > logs/p2p_eval.log 2>&1
$PY $S/eval_aligned.py --no-b0 --append --out outputs/trusted_aligned_merge_v2.json \
    --pred "p2p=outputs/m7/preds_p2p_s{s}.npz" > logs/p2p_aligned.log 2>&1
$PY $S/summarize_merge.py > logs/merge_summary_p2p.log 2>&1
echo "$(date '+%F %T') pix2pix baseline done"
# (3) tier 1 on the 60 annotated frames, against its B0 starts
cd "$W"
for s in 0 1 2; do
  B=$SPP/runs/b0_s$s.pt
  [ -f results/v2/rows/b0pp_s$s.csv ] || $PY scripts/v2/eval_v2.py --model b0pp --seed $s --kind sonospade \
      --ckpt $B --skip-tstr > logs/v2/eval_b0pp_s$s.log 2>&1
  [ -f results/v2/rows/tier1_s$s.csv ] || $PY scripts/v2/eval_v2.py --model tier1 --seed $s --kind paired_lab \
      --ckpt "$B::$SPP/runs/m7/lab_s${s}_f0.pt" --skip-tstr > logs/v2/eval_tier1_s$s.log 2>&1
done
$PY scripts/v2/gate_v2.py --cand tier1 --control b0pp --mode noharm | tee logs/v2/tier1_noharm.log
echo "$(date '+%F %T') paired checks done"
