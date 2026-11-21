#!/usr/bin/env bash
# WP3 (A3, foundation-model initialization of S_US). One job at a time; resumable.
# 1. train S_US per initialization with v1's protocol; 2. score it (held-out simulator Dice selects; real Dice is
# analysis only); 3. retrain SonoSPADE with each frozen S_US (seed 0, full pool, v1 recipe otherwise, plus any
# WP1 flags passed in WP1_FLAGS); 4. score the generators with eval_v2.py.
#   INITS="usfm dinov2 echocare" WP1_FLAGS="" bash scripts/v2/run_wp3.sh
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
INITS=${INITS:-"usfm dinov2 echocare"}
SEEDS=${SEEDS:-0}
DIR=runs/v2/wp3; mkdir -p $DIR logs/v2
for I in $INITS; do
  [ -f $DIR/s_us_$I.pt ] || $PY scripts/v2/train_sus_v2.py --sus-init $I --out $DIR/s_us_$I.pt > logs/v2/train_sus_$I.log 2>&1
  grep -q "^$I," results/v2/sus.csv 2>/dev/null || $PY scripts/v2/eval_sus_v2.py --name $I --ckpt $DIR/s_us_$I.pt > logs/v2/eval_sus_$I.log 2>&1
done
for I in $INITS; do
  for S in $SEEDS; do
    OUT=$DIR/gen_sus${I}${TAG:-}_s$S.pt
    [ -f $OUT ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --s-us $DIR/s_us_$I.pt --seed $S --out $OUT \
        ${WP1_FLAGS:-} > logs/v2/train_gen_sus${I}${TAG:-}_s$S.log 2>&1
    [ -f results/v2/rows/sus${I}${TAG:-}_s$S.csv ] || $PY scripts/v2/eval_v2.py --model sus${I}${TAG:-} --seed $S \
        --kind sonospade_v2 --ckpt $OUT --tstr-seeds 0 1 2 > logs/v2/eval_sus${I}${TAG:-}_s$S.log 2>&1
  done
done
