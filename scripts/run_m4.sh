#!/usr/bin/env bash
# After scripts/run_trusted.sh: make sure every shard has its raw slices, train M4 and its HU-free
# control with patient-level 5-fold CV for each GAN seed, then score every row together.
#   bash scripts/run_m4.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
while pgrep -f "scripts/run_trusted.sh" >/dev/null; do sleep 30; done
python -I scripts/trusted_prepare.py                       # regenerates shards without raw slices, merges
python -I scripts/train_m4.py --mode hu --seeds 0 1 2
python -I scripts/train_m4.py --mode const --seeds 0 1 2
ARGS=()
for v in b0 m1 m2 m3; do for s in 0 1 2; do ARGS+=(--ckpt "${v}_s${s}=runs/${v}_s${s}.pt"); done; done
for s in 0 1 2; do ARGS+=(--pred "m4_s${s}=outputs/m4/preds_hu_s${s}.npz" --pred "m4c_s${s}=outputs/m4/preds_const_s${s}.npz"); done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs.npz "${ARGS[@]}" --out outputs/trusted.json
python -I scripts/summarize_trusted.py --results outputs/trusted.json --out docs/TRUSTED_TABLE.md
python scripts/paper_macros.py
echo "run_m4 done"
