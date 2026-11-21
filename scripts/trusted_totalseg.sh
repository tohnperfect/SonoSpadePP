#!/usr/bin/env bash
# Label every TRUSTED CT that has an annotated 3D US volume with TotalSegmentator (task "total",
# 1.5 mm), the same tool SonoSpade's training CTs were labeled with. Resumable: a CT whose output
# folder already holds liver.nii.gz is skipped.
#
#   bash scripts/trusted_totalseg.sh [DEVICE]      # DEVICE: mps (default), cpu or gpu
set -euo pipefail
cd "$(dirname "$0")/.."
DEV="${1:-mps}"
ROOT=data/trusted/raw
OUT=data/trusted/totalseg
mkdir -p "$OUT"
for ct in "$ROOT"/*/CT_DATA/CT_images/*.nii.gz; do
  pid=$(basename "$ct" | sed -E 's/^([0-9]+).*/\1/')
  if ! compgen -G "$ROOT/*/US_DATA/US_images/${pid}[LR]_*" >/dev/null; then
    continue                                     # no annotated US for this patient
  fi
  if [ -f "$OUT/$pid/liver.nii.gz" ]; then
    echo "skip $pid"; continue
  fi
  echo "$(date +%H:%M:%S) TotalSegmentator $pid"
  TotalSegmentator -i "$ct" -o "$OUT/$pid" -ta total -d "$DEV" -q
done
echo "done"
