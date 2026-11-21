#!/usr/bin/env bash
# Tier-1 side-effect check (3b): tier 1 against its B0 starts on the same-patient study's own held-out fan render
# (data/dataset/pairs_spp_eval200.npz). Starts after the process given as $1 exits. Resumable.
set -e
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
# The study ran in two working copies of this repository: W held the tier-0 data, runs and results, SPP the
# TRUSTED data, runs and outputs. Both default to this repository; set the variables to split them again.
W=${SONOSPADE_V2_ROOT:-$ROOT}; SPP=${SONOSPADEPP_ROOT:-$ROOT}
cd "$W"
export PYTORCH_ENABLE_MPS_FALLBACK=1 SONOSPADE_FM_DIR=$W/data/fm
PY="$W/.venv/bin/python -u -I"
P=data/dataset/pairs_spp_eval200.npz
echo "$(date '+%F %T') tier1 fan-render check start"
for s in 0 1 2; do
  B=$SPP/runs/b0_s$s.pt
  [ -f results/v2/rows/b0pp_fan_s$s.csv ] || $PY scripts/v2/eval_v2.py --model b0pp_fan --seed $s --kind sonospade \
      --ckpt $B --pairs $P --skip-tstr > logs/v2/eval_b0pp_fan_s$s.log 2>&1
  [ -f results/v2/rows/tier1_fan_s$s.csv ] || $PY scripts/v2/eval_v2.py --model tier1_fan --seed $s --kind paired_lab \
      --ckpt "$B::$SPP/runs/m7/lab_s${s}_f0.pt" --pairs $P --skip-tstr \
      > logs/v2/eval_tier1_fan_s$s.log 2>&1
done
$PY scripts/v2/gate_v2.py --cand tier1_fan --control b0pp_fan --mode noharm | tee logs/v2/tier1_fan_noharm.log
echo "$(date '+%F %T') tier1 fan-render check done"
