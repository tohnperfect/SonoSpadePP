#!/usr/bin/env bash
# WP1, second gated setting: DINOv2 E1b, lambda_fm_tissue 0.1, the feasible
# selection run with the lowest validation score across encoders (section 3's rule read across encoders).
# Full pool, seeds 0-2, evaluation frames, then the section 3 gate test against the three v1 retrains.
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
NAME=e1b_dinov2_tis0.1; FLAGS="--fm-disc dinov2 --lambda-fm-tissue 0.1"
mkdir -p runs/v2/wp1_final logs/v2
for S in 0 1 2; do
  OUT=runs/v2/wp1_final/${NAME}_full_s$S.pt
  [ -f $OUT ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split all --seed $S --out $OUT $FLAGS \
      > logs/v2/train_${NAME}_full_s$S.log 2>&1
  [ -f results/v2/rows/wp1_${NAME}_s$S.csv ] || $PY scripts/v2/eval_v2.py --model wp1_${NAME} --seed $S \
      --kind sonospade_v2 --ckpt $OUT --tstr-seeds 0 1 2 > logs/v2/eval_wp1_${NAME}_s$S.log 2>&1
done
$PY scripts/v2/make_v2_tables.py > logs/v2/tables_after_wp1b.log 2>&1
$PY scripts/v2/gate_v2.py --cand wp1_$NAME --control v1 | tee logs/v2/wp1b_gate.log
echo "WP1b chain done"
