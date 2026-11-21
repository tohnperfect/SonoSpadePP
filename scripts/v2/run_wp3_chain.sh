#!/usr/bin/env bash
# WP3 end to end: S_US per init, score, generator seed 0 per init, pick by held-out
# simulator Dice, seeds 1 and 2 for the pick, tables, gate.
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
INITS="usfm dinov2 echocare" SEEDS=0 bash scripts/v2/run_wp3.sh
BEST=$(.venv/bin/python -I scripts/v2/pick_wp3.py usfm dinov2 echocare | tail -1)
echo "WP3 pick: $BEST" | tee logs/v2/wp3_pick.log
[ "$BEST" != "NONE" ] && INITS="$BEST" SEEDS="1 2" bash scripts/v2/run_wp3.sh
.venv/bin/python -I scripts/v2/make_v2_tables.py > logs/v2/tables_after_wp3.log 2>&1
[ "$BEST" = "NONE" ] || .venv/bin/python -I scripts/v2/gate_v2.py --cand sus$BEST --control v1 --mode noharm \
    --sus-name $BEST | tee logs/v2/wp3_gate.log          # gate as fixed before the runs
echo "WP3 chain done"
