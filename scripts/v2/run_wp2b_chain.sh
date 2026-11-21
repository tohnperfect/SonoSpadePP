#!/usr/bin/env bash
# WP2 fallback, INADE. Selection: seed 0, train split, 1500 iterations,
# lambda_ms in {0.1, 0.3, 1}; feasible = locality (both) and edge IoU at least 95% of v1sel's and content diversity
# (LPIPS of sampled codes) above 0.02; pick the lowest validation KID (USFM). The pick: full pool, seeds 0-2, E2a
# (codes at 0) and E2b (sampled codes), then the WP2 gate.
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -u -I"
D=runs/v2/wp2; SEL=results/v2/selection_wp2.csv; mkdir -p $D logs/v2
grep -q "^v1sel," $SEL 2>/dev/null || $PY scripts/v2/select_v2.py --diversity --name v1sel \
    --ckpt runs/v2/wp1/v1sel_s0.pt --out $SEL > logs/v2/select_v1sel_wp2.log 2>&1
for L in 0.1 0.3 1; do
  N=e2i_ms$L
  [ -f $D/${N}_s0.pt ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split train --seed 0 \
      --style inade --lambda-ms $L --out $D/${N}_s0.pt > logs/v2/train_${N}_s0.log 2>&1
  grep -q "^$N," $SEL 2>/dev/null || $PY scripts/v2/select_v2.py --diversity --name $N --ckpt $D/${N}_s0.pt \
      --out $SEL > logs/v2/select_${N}.log 2>&1
done
BEST=$($PY scripts/v2/pick_wp1.py --selection $SEL --prefix e2i_ms --require diversity_lpips:0.02 | tail -1 | awk '{print $1}')
echo "WP2 INADE pick: $BEST" | tee logs/v2/wp2b_pick.log
[ "$BEST" != "NONE" ] || { echo "no feasible INADE setting: WP2 fails"; exit 0; }
L=${BEST#e2i_ms}
for S in 0 1 2; do
  OUT=$D/e2i_full_s$S.pt
  [ -f $OUT ] || $PY scripts/v2/train_v2.py --variant sonospade_v2 --real-split all --seed $S --style inade \
      --lambda-ms $L --out $OUT > logs/v2/train_e2i_full_s$S.log 2>&1
  [ -f results/v2/rows/wp2i_e2a_s$S.csv ] || $PY scripts/v2/eval_v2.py --model wp2i_e2a --seed $S \
      --kind sonospade_v2 --ckpt $OUT --tstr-seeds 0 1 2 > logs/v2/eval_wp2i_e2a_s$S.log 2>&1
  [ -f results/v2/rows/wp2i_e2b_s$S.csv ] || $PY scripts/v2/eval_v2.py --model wp2i_e2b --seed $S \
      --kind sonospade_v2 --ckpt $OUT --style-mode sample --skip-tstr > logs/v2/eval_wp2i_e2b_s$S.log 2>&1
done
$PY scripts/v2/make_v2_tables.py > logs/v2/tables_after_wp2b.log 2>&1
$PY scripts/v2/gate_v2.py --cand wp2i_e2a --sampled wp2i_e2b --mode wp2 | tee logs/v2/wp2b_gate.log
echo "WP2b chain done"
