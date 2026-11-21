#!/usr/bin/env bash
# Full comparison, resumable (every step skips outputs that exist):
#   volumes -> one pose sweep for all variants -> S_US per (physics, labels) -> variants x seeds -> eval
#
#   bash scripts/run_compare.sh                 # seeds 0 1 2, 1500 iters, 2 jobs at a time
#   SEEDS="0" ITERS=1500 K=2 bash scripts/run_compare.sh
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=${PY:-.venv/bin/python}
TRAIN="s0011 s0058 s0344 s0358 s0461 s0477 s0511 s0513"        # SonoSpade's training CTs
EVAL="s0519 s0522 s0591 s0649 s0703 s0720 s0727 s0762 s0777 s1086 s1373"   # held-out CTs
SEEDS=${SEEDS:-"0 1 2"}
ITERS=${ITERS:-1500}
K=${K:-2}
TSTR_SEEDS=${TSTR_SEEDS:-"0 1 2"}
mkdir -p runs outputs logs

$PY scripts/build_volumes.py --cases $TRAIN $EVAL
[ -f data/pairs_train.npz ] || $PY scripts/make_pairs.py --cases $TRAIN --n-per-case 110 --n-aug 2 \
    --seed 0 --out data/pairs_train.npz
[ -f data/pairs_eval.npz ] || $PY scripts/make_pairs.py --cases $EVAL --n-per-case 30 --n-aug 0 \
    --seed 1 --out data/pairs_eval.npz

# S_US: b0 and m3 share one (same physics, same labels)
for pl in "phys_b0 lab_base" "phys_m1 lab_pp" "phys_m2 lab_base"; do
  set -- $pl
  OUT=runs/s_us_${1}__${2}.pt
  [ -f "$OUT" ] || $PY scripts/train_s_us.py --pairs data/pairs_train.npz --phys "$1" --lab "$2" \
      --out "$OUT" > "logs/s_us_${1}__${2}.log" 2>&1
done

s_us_for() { case $1 in b0|m3) echo runs/s_us_phys_b0__lab_base.pt;; m1) echo runs/s_us_phys_m1__lab_pp.pt;;
             m2) echo runs/s_us_phys_m2__lab_base.pt;; esac; }
CMDS=()
for s in $SEEDS; do for v in b0 m1 m2 m3; do
  OUT=runs/${v}_s${s}.pt
  [ -f "$OUT" ] && continue
  CMDS+=("$PY scripts/train_variant.py --variant $v --pairs data/pairs_train.npz --s-us $(s_us_for $v) \
--real-dir data/real_pool --iters $ITERS --seed $s --out $OUT > logs/${v}_s${s}.log 2>&1")
done; done
if [ ${#CMDS[@]} -gt 0 ]; then
  printf '%s\n' "${CMDS[@]}" | xargs -P "$K" -I{} bash -c '{}'
fi

CK=()
for s in $SEEDS; do for v in b0 m1 m2 m3; do CK+=(--ckpt "${v}_s${s}=runs/${v}_s${s}.pt"); done; done
$PY scripts/eval_compare.py --pairs data/pairs_eval.npz --real-test data/real_test.npz --physics-rows \
    --tstr-seeds $TSTR_SEEDS "${CK[@]}" --out outputs/compare.json | tee logs/eval.log
$PY scripts/summarize.py --results outputs/compare.json --out docs/RESULTS_TABLE.md
