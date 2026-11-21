#!/usr/bin/env python3
"""Build the HU-keeping label volumes (sonospadepp.ctvol) for TotalSegmentator cases.

  python scripts/build_volumes.py --cases s0011 s0058 --raw data/raw/totalseg_small --out data/pp_cache
"""
from __future__ import annotations

import argparse
import json
import os
import time

from sonospadepp.ctvol import build_pp_volume, save_pp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--raw", default="data/raw/totalseg_small")
    ap.add_argument("--out", default="data/pp_cache")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    summary = {}
    for case in args.cases:
        path = os.path.join(args.out, f"{case}.npz")
        if os.path.exists(path) and not args.force:
            print(f"{case}: cached"); continue
        t0 = time.time()
        d = os.path.join(args.raw, case)
        vol = build_pp_volume(os.path.join(d, "ct.nii.gz"), os.path.join(d, "segmentations"))
        save_pp(vol, path)
        m = vol.meta
        summary[case] = {k: m[k] for k in ("gas_frac_of_gi", "vessel_frac_of_liver", "cyst_vox",
                                          "bone_rule_exempted_vox", "ref_hu")}
        print(f"{case}: gas/GI {m['gas_frac_of_gi']:.3f}  CT-vessel/liver {m['vessel_frac_of_liver']:.3f}  "
              f"cyst {m['cyst_vox']}  bone-rule exempted {m['bone_rule_exempted_vox']}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    if summary:
        with open(os.path.join(args.out, "build_summary.json"), "a") as fh:
            fh.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
