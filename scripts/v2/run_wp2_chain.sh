#!/usr/bin/env bash
# WP2 (A2, per-region style codes) end to end: SEAN-style codes with the default
# weights (style_dim 16, latent regression 1, KL 0.01), full pool, seeds 0-2; each checkpoint scored twice:
# E2a with every tissue at its bank mean, E2b with sampled codes (realism of sampled frames).
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
D=runs/v2/wp2; mkdir -p $D logs/v2
for S in 0 1 2; do
  OUT=$D/e2_sean_full_s$S.pt
  [ -f $OUT ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split all --seed $S --style sean \
      --out $OUT > logs/v2/train_e2_sean_s$S.log 2>&1
  [ -f results/v2/rows/wp2_e2a_s$S.csv ] || $PY scripts/v2/eval_v2.py --model wp2_e2a --seed $S --kind sonospade_v2 \
      --ckpt $OUT --tstr-seeds 0 1 2 > logs/v2/eval_wp2_e2a_s$S.log 2>&1
  [ -f results/v2/rows/wp2_e2b_s$S.csv ] || $PY scripts/v2/eval_v2.py --model wp2_e2b --seed $S --kind sonospade_v2 \
      --ckpt $OUT --style-mode sample --skip-tstr > logs/v2/eval_wp2_e2b_s$S.log 2>&1
done
$PY scripts/v2/make_v2_tables.py > logs/v2/tables_after_wp2.log 2>&1
echo "WP2 chain done"
