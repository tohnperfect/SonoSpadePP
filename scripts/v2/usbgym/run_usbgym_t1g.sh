#!/usr/bin/env bash
# T1G: tier 1 in the USB Gym closed loop (pre-registered
# 2026-11-19). Tier 1's recipe (scripts/train_m7.py --mode lab, seed 0, fold 0, the paper's
# settings) started from v1r, exported to USB Gym's SonoSPADE layout, then the earlier closed-loop pipeline
# unchanged for one new stage "t1g" (run_usbgym_v2.sh's segmenter and RL steps). v1r's arms are reused.
# Gates G1-G3 stop the chain before any RL run. Resumable: finished files are skipped. One heavy job at a time.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
# The study ran in two working copies of this repository: W held the tier-0 data, runs and results, SPP the
# TRUSTED data, runs and outputs. Both default to this repository; set the variables to split them again.
W=${SONOSPADE_V2_ROOT:-$ROOT}; SPP=${SONOSPADEPP_ROOT:-$ROOT}
# USB Gym (https://github.com/tohnperfect/USBGym) and its segmenter-training companion are separate checkouts.
GYM=${USBG_GYM_ROOT:?set USBG_GYM_ROOT to a USB Gym checkout}
SEGR=${USBG_SEGTRAIN_ROOT:?set USBG_SEGTRAIN_ROOT to the segmenter-training checkout}
export USBG_GYM_ROOT=$GYM
export PYTORCH_ENABLE_MPS_FALLBACK=1 SONOSPADE_FM_DIR=$W/data/fm USBG_SEAT=anatomical
V2PY="$W/.venv/bin/python -u -I"; SPPPY="$SPP/.venv/bin/python -u -I"
D=$GYM/data/dataset/v2fm; O=$GYM/outputs/v2fm; L=$W/logs/v2; G=$W/results/v2/gate_t1g.json
V1R=$D/sonospade_v1r_curvi.pt; TX=$D/sonospade_t1g_curvi.pt
mkdir -p $L $O
log() { echo "$(date '+%F %T') $*"; }

log "G3 leakage: v1r's training CTs and TRUSTED against the six USB Gym test patients"
$V2PY - "$GYM/data/dataset/v2/pairs_med_curvi.npz" "$SPP/data/trusted/pairs_dense" "$G" <<'PYEOF'
import glob, json, os, sys
import numpy as np
test = {"s0358", "s0513", "s0519", "s0591", "s0649", "s0762"}
cts = {str(s).split("_")[0].split(":")[0] for s in np.load(sys.argv[1], allow_pickle=False)["sessions"]}
tr = set()
for f in glob.glob(os.path.join(sys.argv[2], "*.npz")):
    tr |= {str(s) for s in np.load(f, allow_pickle=False)["sessions"]}
hit = sorted((cts | tr) & test)
g = {"G3_v1r_training_cts": sorted(cts), "G3_overlap": hit, "G3_pass": not hit}
json.dump(g, open(sys.argv[3], "w"), indent=1)
print(g)
sys.exit(0 if not hit else 1)
PYEOF

cd $SPP
mkdir -p runs/t1g outputs/t1g logs
[ -f runs/t1g/b0_s0.pt ] || cp $V1R runs/t1g/b0_s0.pt          # train_m7 starts from runs/<dir>/b0_s<seed>.pt
log "tier 1 from v1r (train_m7 --mode lab, seed 0, fold 0)"
[ -f runs/t1g/m7/t1g_s0_f0.pt ] || $SPPPY scripts/train_m7.py --mode lab --seeds 0 --folds 0 --runs runs/t1g \
    --out outputs/t1g --tag t1g > $L/t1g_train.log 2>&1
log "v1r's own held-out agreement (same script, 1 step at lr 0)"
[ -f runs/t1g/m7/v1r0_s0.json ] || $SPPPY scripts/train_m7.py --mode lab --seeds 0 --folds 0 --runs runs/t1g \
    --out outputs/t1g --tag v1r0 --steps 1 --lr 0 > $L/t1g_v1r0.log 2>&1
$V2PY - "$SPP/runs/t1g/m7/t1g_s0.json" "$SPP/runs/t1g/m7/v1r0_s0.json" "$G" <<'PYEOF'
import json, sys
t, b = (json.load(open(f))["0"]["heldout_r"] for f in sys.argv[1:3])
g = json.load(open(sys.argv[3])); g.update({"G2_t1g_heldout_r": t, "G2_v1r_heldout_r": b, "G2_pass": t > b})
json.dump(g, open(sys.argv[3], "w"), indent=1)
print({k: v for k, v in g.items() if k.startswith("G2")})
sys.exit(0 if t > b else 1)
PYEOF

cd $W
log "export to USB Gym's layout and G1"
$V2PY scripts/v2/usbgym/export_t1g.py --start $V1R --tier1 $SPP/runs/t1g/m7/t1g_s0_f0.pt --out $TX \
    --pairs $GYM/data/dataset/v2/pairs_med_curvi.npz --gate $G > $L/t1g_export.log 2>&1
cat $L/t1g_export.log
mkdir -p $O/textures_t1g
python3 - "$GYM" "$TX" "$O/textures_t1g/manifest.json" <<'PYEOF'
import json, os, sys
gym, ck, out = sys.argv[1:]
m = {"physics": None, "cut": os.path.join(gym, "data/dataset/v2/cut_curvi.pt"),
     "cyclegan": os.path.join(gym, "data/dataset/v2/cyclegan_curvi.pt"), "sonospade": ck}
json.dump(m, open(out, "w"), indent=2)
PYEOF

tag=t1g; NN=data/nnunet_v2fm_$tag; OUTS=outputs/v2fm/$tag
cd $SEGR; mkdir -p $NN $OUTS
export USBG_LEARNED_SEG=0; unset US_SEG_CKPT || true
for s in 0 1 2; do
  log "segmenter $tag seed $s"
  id=$((110 + s)); eid=$((210 + s))
  ds=$NN/Dataset${id}_USOrgansV2_sonospade_s$s; eds=$NN/Dataset${eid}_USOrgansV2eval_sonospade_s$s
  ck=$OUTS/us_seg_sonospade_s$s.pt
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
log "RL arms $tag (CT masks, learned masks; seeds 0-2)"
PYTHONPATH=src .venv/bin/python scripts/run_table4_v2.py --out $O/table4_$tag --tex-manifest $O/textures_$tag/manifest.json \
    --seg-manifest $SEGR/outputs/v2fm/$tag/seg_manifest.json --phases ctmask matched --only sonospade --k 1 \
    > $O/table4_$tag.log 2>&1
PYTHONPATH=src .venv/bin/python scripts/agg_table4_v2.py $O/table4_$tag > $O/agg_$tag.log 2>&1
cd $W && $V2PY scripts/v2/usbgym/summarize_t1g.py > $L/usbgym_t1g_summary.log 2>&1
cat $L/usbgym_t1g_summary.log
log "T1G chain done"
