#!/usr/bin/env bash
# WP-final: full model = every tier-0 component whose gate passed (plan_final.py reads the gate files),
# leave-one-out ablations, seeds 0-2 on the full pool, evaluated with eval_v2.py (TSTR seeds 0-2); latency of the full
# model against v1 in one process. Models whose flags equal a gated work-package model reuse its runs (aliases).
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
D=runs/v2/final; mkdir -p $D logs/v2
$PY scripts/v2/plan_final.py | tee logs/v2/final_plan.log
grep -v "^#" logs/v2/final_plan.log | while read -r KEY FLAGS; do
  [ -n "$KEY" ] || continue
  for S in 0 1 2; do
    OUT=$D/${KEY}_s$S.pt
    [ -f $OUT ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split all --seed $S --out $OUT $FLAGS \
        > logs/v2/train_${KEY}_s$S.log 2>&1 < /dev/null
    [ -f results/v2/rows/${KEY}_s$S.csv ] || $PY scripts/v2/eval_v2.py --model $KEY --seed $S --kind sonospade_v2 \
        --ckpt $OUT --tstr-seeds 0 1 2 > logs/v2/eval_${KEY}_s$S.log 2>&1 < /dev/null
    if echo "$FLAGS" | grep -q -- "--style"; then
      [ -f results/v2/rows/${KEY}_sampled_s$S.csv ] || $PY scripts/v2/eval_v2.py --model ${KEY}_sampled --seed $S \
          --kind sonospade_v2 --ckpt $OUT --style-mode sample --skip-tstr > logs/v2/eval_${KEY}_sampled_s$S.log 2>&1 < /dev/null
    fi
  done
done
# latency: the full model (its seed-0 checkpoint, wherever it lives) against v1, same process
FULL=$($PY -c "import json; print(json.load(open('results/v2/final_model.json'))['models']['full']['key'])")
case "$FULL" in
  v1) CK="" ;;
  v2) CK=$D/v2_s0.pt ;;
  sus*) CK=runs/v2/wp3/gen_${FULL}_s0.pt ;;
  wp2i_e2a) CK=runs/v2/wp2/e2i_full_s0.pt ;;
  wp1_*) CK=runs/v2/wp1_final/${FULL#wp1_}_full_s0.pt ;;
esac
[ -z "$CK" ] || $PY scripts/v2/bench_latency_v2.py --stage v1_final:sonospade:runs/v2/v1_repro/sonospade_s0.pt \
    --stage v2_final:sonospade_v2:$CK > logs/v2/latency_final.log 2>&1
$PY scripts/v2/make_v2_tables.py > logs/v2/tables_after_final.log 2>&1
echo "WP-final chain done (full model: $FULL)"
