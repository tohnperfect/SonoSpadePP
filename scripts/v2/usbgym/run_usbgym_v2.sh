#!/usr/bin/env bash
# USB Gym closed loop for v2 (protocol fixed before the runs).
# Two new texture stages for USB Gym's `sonospade` slot, both trained with USB Gym's own trainer and recipe
# (scripts/train_texture_gan.py, unchanged: the 1540 curvilinear pairs of its 8 texture-training patients,
# 2x view-augmented, configs/sonospade_curvi.yaml, 1500 iterations, seed 0, the 404-frame public pool), in the
# v2 environment (train_texture_gym.py only widens the segmenter loader):
#   v1r  with USB Gym's published S_US (s_us_grouped_curvi.pt): v1 retrained in this environment (the control)
#   v2   with a USFM-initialized S_US trained on the same non-augmented pairs (2000 iterations, as USB Gym's S_US)
# Then USB Gym's own steps, unchanged, per stage: per-stage learned-mask segmenters (companion repository,
# seeds 0-2, 4000 iterations; the cross-validation and consistency extras of run_seg_v2.sh are skipped, they do
# not enter the closed loop) and both RL arms (CT masks, learned masks; seeds 0-2; run_table4_v2.py, one job at
# a time). Outputs stay in the two repositories' ignored folders; the summary goes to results/v2/usbgym.*.
# Resumable: finished files are skipped.
set -euo pipefail
W=${SONOSPADE_V2_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
# USB Gym (https://github.com/tohnperfect/USBGym) and its segmenter-training companion are separate checkouts.
GYM=${USBG_GYM_ROOT:?set USBG_GYM_ROOT to a USB Gym checkout}
SEGR=${USBG_SEGTRAIN_ROOT:?set USBG_SEGTRAIN_ROOT to the segmenter-training checkout}
export USBG_GYM_ROOT=$GYM
export PYTORCH_ENABLE_MPS_FALLBACK=1 SONOSPADE_FM_DIR=$W/data/fm USBG_SEAT=anatomical
V2PY="$W/.venv/bin/python -u -I"
D=$GYM/data/dataset/v2fm; O=$GYM/outputs/v2fm
mkdir -p $D $O
log() { echo "$(date '+%F %T') $*"; }

cd $GYM
log "S_US from USFM on the gym's curvilinear pairs"
[ -f $D/s_us_usfm_curvi.pt ] || $V2PY $W/scripts/v2/train_sus_v2.py --sus-init usfm \
    --pairs data/dataset/v2/pairs_med_curvi.npz --iters 2000 --seed 0 --out $D/s_us_usfm_curvi.pt > $O/s_us_usfm_curvi.log 2>&1

tex() {   # tag, S_US
  log "texture $1"
  [ -f $D/sonospade_$1_curvi.pt ] || $V2PY $W/scripts/v2/usbgym/train_texture_gym.py --variant sonospade \
      --pairs data/dataset/v2/pairs_med_curvi_aug.npz --real-dir data/real/kaggle_vitale/pool --s-us $2 \
      --config configs/sonospade_curvi.yaml --seed 0 --iters 1500 --out $D/sonospade_$1_curvi.pt \
      --preview $O/sonospade_$1_preview.png > $O/sonospade_$1_curvi.log 2>&1
  mkdir -p $O/textures_$1
  python3 - "$GYM" "$D/sonospade_$1_curvi.pt" "$O/textures_$1/manifest.json" <<'PYEOF'
import json, os, sys
gym, ck, out = sys.argv[1:]
m = {"physics": None, "cut": os.path.join(gym, "data/dataset/v2/cut_curvi.pt"),
     "cyclegan": os.path.join(gym, "data/dataset/v2/cyclegan_curvi.pt"), "sonospade": ck}
json.dump(m, open(out, "w"), indent=2)
PYEOF
}
tex v1r data/dataset/v2/s_us_grouped_curvi.pt
tex v2 $D/s_us_usfm_curvi.pt

seg() {   # tag: run_seg_v2.sh's build, check, train and manifest steps for the sonospade slot
  local tag=$1 NN=data/nnunet_v2fm_$1 OUTS=outputs/v2fm/$1 TX=$D/sonospade_$1_curvi.pt
  cd $SEGR; mkdir -p $NN $OUTS
  export USBG_LEARNED_SEG=0; unset US_SEG_CKPT || true
  for s in 0 1 2; do
    log "segmenter $tag seed $s"
    local id=$((110 + s)) eid=$((210 + s))
    local ds=$NN/Dataset${id}_USOrgansV2_sonospade_s$s eds=$NN/Dataset${eid}_USOrgansV2eval_sonospade_s$s
    local ck=$OUTS/us_seg_sonospade_s$s.pt
    [ -f $ds/dataset.json ] || ./run.sh scripts/build_dataset.py --out $NN --dataset-id $id \
        --name USOrgansV2_sonospade_s$s --translator $TX --cases train --role train --seat-organs liver,kidney \
        --strict-seat --skip-pairs s0058:kidney,s0522:kidney,s0727:kidney --require-seat anatomical --n-poses 27 \
        --seed $s > $OUTS/build_s$s.log 2>&1
    [ -f $eds/dataset.json ] || ./run.sh scripts/build_dataset.py --out $NN --dataset-id $eid \
        --name USOrgansV2eval_sonospade_s$s --translator $TX --cases test --role eval --seat-organs liver,kidney \
        --strict-seat --skip-pairs s0058:kidney,s0522:kidney,s0727:kidney --require-seat anatomical --n-poses 12 \
        --seed $((1000 + s)) --no-real-test > $OUTS/build_eval_s$s.log 2>&1
    ./run.sh scripts/check_manifest.py --datasets $ds --eval-datasets $eds --require-seat anatomical > $OUTS/check_s$s.log 2>&1
    [ -f $ck ] || ./run.sh scripts/train.py --dataset $ds --iters 4000 --seed $s --kidneyus \
        --vitale-split data/real/vitale_split_v2.npz --forbid-test-patients --out $ck > $OUTS/train_s$s.log 2>&1
    ./run.sh scripts/write_seg_manifest.py --dir $OUTS > /dev/null
    ./run.sh scripts/check_manifest.py --histories ${ck%.pt}_history.json --vitale-split data/real/vitale_split_v2.npz \
        --seg-manifest $OUTS/seg_manifest.json --require-seat anatomical > $OUTS/check_hist_s$s.log 2>&1
  done
  unset USBG_LEARNED_SEG
  cd $GYM
}
seg v1r
seg v2

for tag in v1r v2; do
  log "RL arms $tag"
  PYTHONPATH=src .venv/bin/python scripts/run_table4_v2.py --out $O/table4_$tag --tex-manifest $O/textures_$tag/manifest.json \
      --seg-manifest $SEGR/outputs/v2fm/$tag/seg_manifest.json --phases ctmask matched --only sonospade --k 1 \
      > $O/table4_$tag.log 2>&1
  PYTHONPATH=src .venv/bin/python scripts/agg_table4_v2.py $O/table4_$tag > $O/agg_$tag.log 2>&1
done
cd $W && $V2PY scripts/v2/usbgym/summarize_usbgym.py > logs/v2/usbgym_summary.log 2>&1
log "USB Gym v2 chain done"
