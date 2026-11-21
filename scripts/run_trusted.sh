#!/usr/bin/env bash
# TRUSTED same-patient evaluation end to end, after scripts/trusted_totalseg.sh has started:
# cut planes per kidney as CT labels become available, then score every checkpoint.
#   bash scripts/run_trusted.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
N=$(ls data/trusted/raw/*/US_DATA/US_images/*_imgUS.nii.gz | wc -l | tr -d ' ')
for i in $(seq 1 200); do
  python -I scripts/trusted_prepare.py
  have=$(ls data/trusted/pairs/*.npz 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date +%H:%M:%S) shards $have / $N"
  [ "$have" -ge "$N" ] && break
  if ! pgrep -f trusted_totalseg.sh >/dev/null; then python -I scripts/trusted_prepare.py; break; fi
  sleep 60
done
CK=""
for v in b0 m1 m2 m3; do for s in 0 1 2; do CK="$CK --ckpt ${v}_s${s}=runs/${v}_s${s}.pt"; done; done
python -I scripts/eval_trusted.py --pairs data/trusted_pairs.npz $CK --out outputs/trusted.json
python -I scripts/summarize_trusted.py --results outputs/trusted.json --out docs/TRUSTED_TABLE.md
