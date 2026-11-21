#!/usr/bin/env bash
# The remaining work-package chains, one after another (one heavy job at a time), in the planned order.
# A chain that fails is logged and the next one still runs: WP3, WP2 and WP5 do not read WP1's outcome.
# Every chain is resumable (finished checkpoints and evaluation rows are skipped).
#   bash scripts/v2/run_queue.sh [wp1 wp3 wp2 wp5]
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs/v2
for C in ${@:-wp1 wp3 wp2 wp5}; do
  echo "$(date '+%F %T') start $C" | tee -a logs/v2/queue.log
  bash scripts/v2/run_${C}_chain.sh >> logs/v2/${C}_chain.log 2>&1
  RC=$?; echo "$(date '+%F %T') $C exit $RC" | tee -a logs/v2/queue.log
done
