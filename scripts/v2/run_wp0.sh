#!/usr/bin/env bash
# WP0: score every v1-setup texture stage with eval_v2.py, then the latency of v1, then the tables and gate.
# Assumes the v1 retrains exist (runs/v2/v1_repro/sonospade_s{0,1,2}.pt, from scripts/v2/train_v2.py
# --variant sonospade or scripts/v1/train_texture_gan.py) and data/ is linked (README, "Data").
# SMOKE=1 runs a tiny version into results/v2/rows_smoke.
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -I"
if [ "${SMOKE:-0}" = "1" ]; then
  EX="--limit 20 --n-locality 3 --tstr-iters 20 --n-boot 20 --n-boot-heavy 5 --n-boot-kid 3 --out-dir results/v2/rows_smoke"
  TS="0"
else
  EX=""
  TS="0 1 2"
fi
mkdir -p logs/v2
ev() { $PY scripts/v2/eval_v2.py "$@" --tstr-seeds $TS $EX; }
ev --model physics --seed 0 --kind physics                                            > logs/v2/eval_physics.log 2>&1
ev --model v1_paper --seed 0 --kind sonospade --ckpt data/v1_ckpt/exp_real_sonospade.pt > logs/v2/eval_v1_paper.log 2>&1
for S in 0 1 2; do
  ev --model v1 --seed $S --kind sonospade --ckpt runs/v2/v1_repro/sonospade_s$S.pt   > logs/v2/eval_v1_s$S.log 2>&1
done
for V in cut cyclegan cyclegan_sc; do
  ev --model ${V}_paper --seed 0 --kind $V --ckpt data/v1_ckpt/exp_real_$V.pt          > logs/v2/eval_${V}_paper.log 2>&1
done
if [ "${SMOKE:-0}" != "1" ]; then
  $PY scripts/v2/bench_latency_v2.py --stage v1_paper:sonospade:data/v1_ckpt/exp_real_sonospade.pt \
      --stage v1_s0:sonospade:runs/v2/v1_repro/sonospade_s0.pt --out results/v2/latency.csv > logs/v2/latency_wp0.log 2>&1
  $PY scripts/v2/make_v2_tables.py --gate wp0
fi
