#!/usr/bin/env bash
# WP5 (A5, per-tissue speckle statistics) end to end. Selection runs: seed 0, real-pool
# train split; scored label-free on the validation split (select_v2.py --speckle, results/v2/selection_wp5.csv).
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
D=runs/v2/wp5; SEL=results/v2/selection_wp5.csv; mkdir -p $D logs/v2
run() { local N=$1; shift
  [ -f $D/${N}_s0.pt ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split train --seed 0 \
      --out $D/${N}_s0.pt "$@" > logs/v2/train_${N}_s0.log 2>&1
  grep -q "^$N," $SEL 2>/dev/null || $PY scripts/v2/select_v2.py --speckle --name $N --ckpt $D/${N}_s0.pt --out $SEL \
      > logs/v2/select_${N}.log 2>&1; }
[ -f runs/v2/wp1/v1sel_s0.pt ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split train --seed 0 \
    --out runs/v2/wp1/v1sel_s0.pt > logs/v2/train_v1sel_s0.log 2>&1
grep -q "^v1sel," $SEL 2>/dev/null || $PY scripts/v2/select_v2.py --speckle --name v1sel --ckpt runs/v2/wp1/v1sel_s0.pt \
    --out $SEL > logs/v2/select_v1sel_wp5.log 2>&1
for L in 0.25 0.5 1 2; do run e5a_spk$L --speckle-loss --lambda-speckle $L; done
run e5b_noise --noise-inject
L=$($PY scripts/v2/pick_wp1.py --selection $SEL --prefix e5a_spk --score val_fan_logm_w1 | tail -1 | awk '{print $2}')
[ -n "$L" ] || L=1
run e5c_spk${L}_noise --speckle-loss --lambda-speckle $L --noise-inject
BEST=$($PY scripts/v2/pick_wp1.py --selection $SEL --names e5a_spk$L,e5b_noise,e5c_spk${L}_noise \
       --score val_fan_logm_w1 | tail -1 | awk '{print $1}')
echo "WP5 pick: lambda $L, setting $BEST" | tee logs/v2/wp5_pick.log
[ "$BEST" != "NONE" ] || { echo "no feasible WP5 setting"; exit 0; }
case "$BEST" in
  e5a_*) FLAGS="--speckle-loss --lambda-speckle $L" ;;
  e5b_*) FLAGS="--noise-inject" ;;
  e5c_*) FLAGS="--speckle-loss --lambda-speckle $L --noise-inject" ;;
esac
for S in 0 1 2; do
  OUT=$D/${BEST}_full_s$S.pt
  [ -f $OUT ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split all --seed $S --out $OUT $FLAGS \
      > logs/v2/train_${BEST}_full_s$S.log 2>&1
  [ -f results/v2/rows/wp5_${BEST}_s$S.csv ] || $PY scripts/v2/eval_v2.py --model wp5_$BEST --seed $S \
      --kind sonospade_v2 --ckpt $OUT --tstr-seeds 0 1 2 > logs/v2/eval_wp5_${BEST}_s$S.log 2>&1
done
$PY scripts/v2/make_v2_tables.py > logs/v2/tables_after_wp5.log 2>&1
echo "WP5 chain done"
