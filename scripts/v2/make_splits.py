#!/usr/bin/env python3
"""Write data/splits_v2.json (see sonospadepp.v2.splits). Run once; the file is committed.

  python -I scripts/v2/make_splits.py --kaggle-src /path/to/ussimandsegm
"""
from __future__ import annotations

import argparse
import json
import os

from sonospadepp.v2 import splits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-src", default=os.environ.get("USSIM_SRC", "data/raw/ussimandsegm"),
                    help="the Kaggle ussimandsegm download (or set USSIM_SRC)")
    ap.add_argument("--test-npz", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--out", default="data/splits_v2.json")
    a = ap.parse_args(argv)
    s = splits.build(a.kaggle_src, a.test_npz)
    with open(a.out, "w") as fh:
        json.dump(s, fh, indent=1)
    rp = s["real_pool"]
    print(f"real pool {rp['n']}: train {rp['n_train']}, val {rp['n_val']} (sequences {rp['val_sequences']})")
    print(f"eval frames {s['eval_frames']['n']} from sequences {s['eval_frames']['sequences']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
