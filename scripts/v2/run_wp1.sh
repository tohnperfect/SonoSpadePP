#!/usr/bin/env bash
# WP1 (A1, foundation-model feature discriminator + per-tissue feature matching), selection stage.
# Seed 0 only, trained on the real pool's TRAIN split (296 frames), scored on its VALIDATION split
# (scripts/v2/select_v2.py). One job at a time. Resumable: a run whose checkpoint exists is skipped.
#   ENC=dinov2 bash scripts/v2/run_wp1.sh sweep           # E1a / E1b lambda sweeps for the main encoder
#   ENC=usfm ADV=0.5 TIS=1.0 bash scripts/v2/run_wp1.sh fixed   # E1a/E1b/E1c at chosen lambdas
# SMOKE=1 trains 20 iterations into runs/v2/wp1_smoke.
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
ENC=${ENC:-echocare}
ITERS=1500; DIR=runs/v2/wp1; SEL=results/v2/selection.csv
if [ "${SMOKE:-0}" = "1" ]; then ITERS=20; DIR=runs/v2/wp1_smoke; SEL=results/v2/selection_smoke.csv; fi
mkdir -p $DIR logs/v2

run() {  # name, extra args...
  local NAME=$1; shift
  if [ ! -f $DIR/${NAME}_s0.pt ]; then
    $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split train --seed 0 --iters $ITERS \
        --out $DIR/${NAME}_s0.pt "$@" > logs/v2/train_${NAME}_s0.log 2>&1
    $PY scripts/v2/select_v2.py --name $NAME --ckpt $DIR/${NAME}_s0.pt --out $SEL > logs/v2/select_${NAME}.log 2>&1
  fi
}

run v1sel                                                    # v1 recipe on the selection pool (reference)
case "${1:-sweep}" in
  sweep)
    for L in 0.1 0.5 1 2; do run e1a_${ENC}_adv$L --fm-disc $ENC --lambda-fm-adv $L; done
    for L in 0.1 0.5 1 2; do run e1b_${ENC}_tis$L --fm-disc $ENC --lambda-fm-tissue $L; done
    ;;
  fixed)
    run e1a_${ENC}_adv$ADV --fm-disc $ENC --lambda-fm-adv $ADV
    run e1b_${ENC}_tis$TIS --fm-disc $ENC --lambda-fm-tissue $TIS
    run e1c_${ENC}_adv${ADV}_tis$TIS --fm-disc $ENC --lambda-fm-adv $ADV --lambda-fm-tissue $TIS
    ;;
esac
