#!/usr/bin/env bash
# Recompute only the speckle rows of the WP0 stages with the final windows (7 px Nakagami, 8 px speckle size).
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python -I"
sp() { $PY scripts/v2/eval_v2.py --only speckle "$@" 2>&1 | grep -E "replaced|Error"; }
sp --model physics --seed 0 --kind physics
sp --model v1_paper --seed 0 --kind sonospade --ckpt data/v1_ckpt/exp_real_sonospade.pt
for S in 0 1 2; do sp --model v1 --seed $S --kind sonospade --ckpt runs/v2/v1_repro/sonospade_s$S.pt; done
for V in cut cyclegan cyclegan_sc; do sp --model ${V}_paper --seed 0 --kind $V --ckpt data/v1_ckpt/exp_real_$V.pt; done
