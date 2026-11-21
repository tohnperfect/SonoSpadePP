#!/usr/bin/env bash
# Dense 128 px training planes (every 1 degree through the kidney), 3 workers.   bash scripts/run_dense_prepare.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
kids=($(python -I -c "import json; print(' '.join(sorted({p['kid'] for p in json.load(open('data/trusted_pairs_meta.json'))['planes']})))"))
for w in 0 1 2; do
  sel=()
  for i in "${!kids[@]}"; do (( i % 3 == w )) && sel+=("${kids[$i]}"); done
  python -I scripts/trusted_dense_prepare.py --kids "${sel[@]}" > logs/dense_prep_$w.log 2>&1 &
done
wait
ls data/trusted/pairs_dense/*.npz | wc -l
echo "run_dense_prepare done"
