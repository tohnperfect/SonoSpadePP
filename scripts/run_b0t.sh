#!/usr/bin/env bash
# B0t, the domain-matched control for approach 1: SonoSPADE's unpaired recipe fine-tuned on TRUSTED.
# Seed 0 first (its line "b0t seed 0 finished" lets run_m7r.sh start), then seeds 1 and 2.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -I scripts/train_b0t.py --seeds 0 > logs/train_b0t_s0.log 2>&1
echo "b0t seed 0 finished (exit $?)"
python -I scripts/train_b0t.py --seeds 1 2 > logs/train_b0t_s12.log 2>&1
echo "b0t seeds 1 2 finished (exit $?)"
echo "run_b0t done"
