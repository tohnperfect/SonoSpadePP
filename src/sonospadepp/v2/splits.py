"""Fixed data splits for SonoSPADE v2 (data/splits_v2.json), built once by scripts/v2/make_splits.py.

* Real pool (unlabeled, 404 Kaggle train-split frames): split by acquisition SEQUENCE (the letter prefix of
  the source file name), never by frame, because consecutive frames of one sequence are near-duplicates.
  Validation sequences are drawn by a seeded permutation until they hold at least 20% of the frames. The
  validation split is used only for model selection (sweeps); final models train on the full pool, as v1.
* Evaluation frames: the 60 annotated Kaggle test-split frames. Never used for training, pseudo-label
  tuning, early stopping or selection. They come from sequences (a, b, g, h) disjoint from the pool.
* Simulator content: v1's protocol scores the first 200 pairs of pairs_med.npz (v1's 8 training CTs).
* USB Gym held-out patients (from the USB Gym datasheet): no v2 model used in the closed-loop evaluation
  may be trained on them.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

USBGYM_TEST_PATIENTS = ["s0358", "s0513", "s0519", "s0591", "s0649", "s0762"]
SASHIMI_CASES = ["s0011", "s0058", "s0344", "s0358", "s0461", "s0477", "s0511", "s0513"]


def sequence_of(fname: str) -> str:
    return re.match(r"([a-zA-Z]+)", os.path.basename(fname)).group(1)


def build(kaggle_src: str, test_npz: str, val_frac: float = 0.2, seed: int = 0) -> dict:
    rus = os.path.join(kaggle_src, "abdominal_US", "abdominal_US", "RUS")
    train_files = sorted(glob.glob(os.path.join(rus, "images", "train", "*.jpg")))   # ingest order
    seqs = [sequence_of(f) for f in train_files]
    uniq = sorted(set(seqs))
    counts = {s: seqs.count(s) for s in uniq}
    order = list(np.random.default_rng(seed).permutation(uniq))
    val_seqs, n = [], 0
    for s in order:
        if n >= val_frac * len(train_files):
            break
        val_seqs.append(str(s))
        n += counts[s]
    pool = [{"index": i, "png": f"{i:05d}.png", "source": os.path.basename(f), "sequence": s,
             "split": "val" if s in val_seqs else "train"} for i, (f, s) in enumerate(zip(train_files, seqs))]
    ev = np.load(test_npz, allow_pickle=True)
    eval_names = [str(x) for x in ev["sessions"]]
    eval_seqs = sorted({sequence_of(x) for x in eval_names})
    assert not set(eval_seqs) & set(uniq), "evaluation sequences overlap the real pool"
    return {
        "schema": "sonospade-v2-splits-1",
        "real_pool": {"n": len(pool), "sequence_counts": counts, "val_sequences": sorted(val_seqs),
                      "val_seed": seed, "val_frac_target": val_frac,
                      "n_train": sum(p["split"] == "train" for p in pool),
                      "n_val": sum(p["split"] == "val" for p in pool), "frames": pool},
        "eval_frames": {"n": len(eval_names), "names": eval_names, "sequences": eval_seqs,
                        "rule": "never used in training, pseudo-label tuning, early stopping or selection"},
        "sim_content": {"pairs": "data/dataset/pairs_med.npz", "cases": SASHIMI_CASES,
                        "eval_first_n": 200},
        "usbgym_test_patients": USBGYM_TEST_PATIENTS,
    }


def load(path: str = "data/splits_v2.json") -> dict:
    with open(path) as fh:
        return json.load(fh)


def pool_indices(splits: dict, which: str) -> list[int]:
    """Indices into the sorted pool PNGs for 'train', 'val' or 'all'."""
    fr = splits["real_pool"]["frames"]
    return [p["index"] for p in fr if which == "all" or p["split"] == which]
