#!/usr/bin/env bash
# Scores everything once all training has finished (run_m7.sh, run_m7r.sh, run_m9.sh, run_b0t.sh):
# B0t rows on all 413 planes (fixed and alignment-tolerant), then the one-image setting, then summaries.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
P=(); for s in 0 1 2; do [ -f outputs/b0t/preds_b0t_s$s.npz ] && P+=(--pred b0t_s$s=outputs/b0t/preds_b0t_s$s.npz); done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs_def.npz "${P[@]}" --append --out outputs/trusted_def.json
S=(); for s in 0 1 2; do [ -f outputs/b0t/preds_b0t_s$s.npz ] && S+=($s); done
python -I scripts/eval_aligned.py --no-b0 --append --seeds "${S[@]}" --pred "b0t=outputs/b0t/preds_b0t_s{s}.npz"
python scripts/summarize_hires.py
bash scripts/run_m9_eval.sh
echo "run_final_eval done"
