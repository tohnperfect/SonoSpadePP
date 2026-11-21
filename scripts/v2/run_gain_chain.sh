#!/usr/bin/env bash
# Gain-field translator H-G: selection over gain_sigma in {2, 4, 8}
# on seed 0 / the training split, the pick on the full pool with seeds 0-2, evaluation, gate against v1.
# Starts after the process given as $1 (the USB Gym chain) exits: one heavy job at a time. Resumable.
set -e
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
cd "$(dirname "$0")/../.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
D=runs/v2/gain; SEL=results/v2/selection_gain.csv; mkdir -p $D logs/v2
echo "$(date '+%F %T') gain chain start"
grep -q "^v1sel," $SEL 2>/dev/null || $PY scripts/v2/select_v2.py --name v1sel \
    --ckpt runs/v2/wp1/v1sel_s0.pt --out $SEL > logs/v2/select_v1sel_gain.log 2>&1
for G in 2 4 8; do
  N=gain_s$G
  [ -f $D/${N}_sel.pt ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split train --seed 0 \
      --gain-sigma $G --out $D/${N}_sel.pt > logs/v2/train_${N}_sel.log 2>&1
  grep -q "^$N," $SEL 2>/dev/null || $PY scripts/v2/select_v2.py --name $N --ckpt $D/${N}_sel.pt \
      --out $SEL > logs/v2/select_${N}.log 2>&1
  echo "$(date '+%F %T') selection $N done"
done
BEST=$($PY scripts/v2/pick_wp1.py --selection $SEL --prefix gain_s | tail -1 | awk '{print $1}')
echo "$(date '+%F %T') gain pick: $BEST" | tee logs/v2/gain_pick.log
[ "$BEST" != "NONE" ] || { echo "no feasible gain setting: H-G fails"; exit 0; }
G=${BEST#gain_s}
for S in 0 1 2; do
  OUT=$D/gain_full_s$S.pt
  [ -f $OUT ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split all --seed $S --gain-sigma $G \
      --out $OUT > logs/v2/train_gain_full_s$S.log 2>&1
  [ -f results/v2/rows/gain_s$S.csv ] || $PY scripts/v2/eval_v2.py --model gain --seed $S \
      --kind sonospade_v2 --ckpt $OUT --tstr-seeds 0 1 2 > logs/v2/eval_gain_s$S.log 2>&1
  echo "$(date '+%F %T') full seed $S done"
done
$PY scripts/v2/gate_v2.py --cand gain | tee logs/v2/gain_gate.log
echo "$(date '+%F %T') gain chain done"
