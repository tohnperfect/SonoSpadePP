#!/usr/bin/env bash
# WP1 end to end, following the protocol fixed before the runs (no human choice inside the chain):
#  1. EchoCare sweep (E1a, E1b) + v1sel on the selection pool, seed 0
#  2. pick the best lambda_fm_adv and lambda_fm_tissue (pick_wp1.py: feasibility, then validation KID, USFM)
#  3. EchoCare E1c; USFM and DINOv2 E1a / E1b / E1c at EchoCare's lambdas (seed 0)
#  4. pick the EchoCare setting among the best E1a, the best E1b and E1c; train it on the full pool, seeds 0-2
#  5. evaluate those three on the 60 evaluation frames (eval_v2.py), rebuild the tables, apply the gate test
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
mkdir -p logs/v2 runs/v2/wp1_final
ENC=echocare bash scripts/v2/run_wp1.sh sweep
ADV_LINE=$($PY scripts/v2/pick_wp1.py --prefix e1a_echocare_adv | tail -1)
TIS_LINE=$($PY scripts/v2/pick_wp1.py --prefix e1b_echocare_tis | tail -1)
echo "picked: $ADV_LINE | $TIS_LINE" | tee logs/v2/wp1_pick.log
ADV=$(echo "$ADV_LINE" | awk '{print $2}'); TIS=$(echo "$TIS_LINE" | awk '{print $2}')
[ -n "$ADV" ] || ADV=0.5          # no feasible E1a: the arm failed; E1c still runs at the planned start value
[ -n "$TIS" ] || TIS=1
ENC=echocare ADV=$ADV TIS=$TIS bash scripts/v2/run_wp1.sh fixed
ENC=usfm ADV=$ADV TIS=$TIS bash scripts/v2/run_wp1.sh fixed
ENC=dinov2 ADV=$ADV TIS=$TIS bash scripts/v2/run_wp1.sh fixed
BEST=$($PY scripts/v2/pick_wp1.py --names e1a_echocare_adv$ADV,e1b_echocare_tis$TIS,e1c_echocare_adv${ADV}_tis$TIS | tail -1)
echo "final setting: $BEST" | tee -a logs/v2/wp1_pick.log
NAME=$(echo "$BEST" | awk '{print $1}')
[ "$NAME" != "NONE" ] || { echo "no feasible EchoCare setting: WP1 gate fails"; exit 0; }
case "$NAME" in
  e1a_*) FLAGS="--fm-disc echocare --lambda-fm-adv $ADV" ;;
  e1b_*) FLAGS="--fm-disc echocare --lambda-fm-tissue $TIS" ;;
  e1c_*) FLAGS="--fm-disc echocare --lambda-fm-adv $ADV --lambda-fm-tissue $TIS" ;;
esac
echo "$NAME $FLAGS" > runs/v2/wp1_final/SETTING.txt
for S in 0 1 2; do
  OUT=runs/v2/wp1_final/${NAME}_full_s$S.pt
  [ -f $OUT ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split all --seed $S --out $OUT $FLAGS \
      > logs/v2/train_${NAME}_full_s$S.log 2>&1
  [ -f results/v2/rows/wp1_${NAME}_s$S.csv ] || $PY scripts/v2/eval_v2.py --model wp1_${NAME} --seed $S \
      --kind sonospade_v2 --ckpt $OUT --tstr-seeds 0 1 2 > logs/v2/eval_wp1_${NAME}_s$S.log 2>&1
done
$PY scripts/v2/make_v2_tables.py > logs/v2/tables_after_wp1.log 2>&1
$PY scripts/v2/gate_v2.py --cand wp1_$NAME --control v1 | tee logs/v2/wp1_gate.log   # section 3 gate test
echo "WP1 chain done"
