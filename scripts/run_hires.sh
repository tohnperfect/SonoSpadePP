#!/usr/bin/env bash
# After scripts/run_hires_prepare.sh: score the fixed rows at full resolution, train M4/M5 (and their
# HU-free controls) at full resolution for both registrations, and rerun the 128 px pipeline with
# the deformable registration (translators B0-M3, M4 and M4c retrained).   bash scripts/run_hires.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
while pgrep -f "scripts/run_hires_prepare.sh" >/dev/null; do sleep 20; done
python -I scripts/eval_hires.py --out outputs/trusted_hires.json > logs/eval_hires_fixed.log 2>&1 &
for reg in rig def; do
  ( for scheme in pop halves kidney; do for mode in hu const; do
      python -I scripts/train_hr.py --scheme $scheme --mode $mode --reg $reg
    done; done ) > logs/train_hr_${reg}.log 2>&1 &
done
( ARGS=()
  for v in b0 m1 m2 m3; do for s in 0 1 2; do ARGS+=(--ckpt "${v}_s${s}=runs/${v}_s${s}.pt"); done; done
  python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz "${ARGS[@]}" --out outputs/trusted_def.json
  python -I scripts/train_m4.py --shards data/trusted/pairs_def --mode hu --tag hudef --out outputs/m4def --seeds 0 1 2
  python -I scripts/train_m4.py --shards data/trusted/pairs_def --mode const --tag constdef --out outputs/m4def --seeds 0 1 2
  P=(); for s in 0 1 2; do P+=(--pred "m4_s${s}=outputs/m4def/preds_hudef_s${s}.npz" --pred "m4c_s${s}=outputs/m4def/preds_constdef_s${s}.npz"); done
  python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --append --out outputs/trusted_def.json
) > logs/def128.log 2>&1 &
wait
P=()
for reg in rig def; do
  P+=(--pred "m4hr@${reg}=outputs/hires/preds_pop_hu_${reg}.npz" --pred "m4chr@${reg}=outputs/hires/preds_pop_const_${reg}.npz")
  P+=(--pred "m5hr@${reg}=outputs/hires/preds_halves_hu_${reg}.npz" --pred "m5chr@${reg}=outputs/hires/preds_halves_const_${reg}.npz")
  P+=(--pred "m5khr@${reg}=outputs/hires/preds_kidney_hu_${reg}.npz" --pred "m5kchr@${reg}=outputs/hires/preds_kidney_const_${reg}.npz")
done
python -I scripts/eval_hires.py --no-fixed --append "${P[@]}" --out outputs/trusted_hires.json > logs/eval_hires_learned.log 2>&1
echo "run_hires done"
