#!/usr/bin/env bash
# Link the inputs this repo needs from an existing SonoSpade / USB Gym data folder.
#   SRC=/path/to/data bash scripts/link_data.sh
# Needs, under $SRC:
#   raw/totalseg_small/<case>/{ct.nii.gz,segmentations/}   TotalSegmentator CTs (CC BY 4.0)
#   real/kaggle_vitale/pool/                                real Kaggle ussimandsegm frames (unlabeled pool)
#   eval/kaggle_vitale_test.npz                             the 60 labeled real test frames
# (SonoSpade's scripts/ingest_kaggle_us.py builds the last two from the Kaggle download.)
set -e
SRC=${SRC:?set SRC to the data folder (see the header of this script)}
mkdir -p data/raw
ln -sfn "$SRC/raw/totalseg_small" data/raw/totalseg_small
ln -sfn "$SRC/real/kaggle_vitale/pool" data/real_pool
ln -sfn "$SRC/eval/kaggle_vitale_test.npz" data/real_test.npz
ls -l data data/raw
