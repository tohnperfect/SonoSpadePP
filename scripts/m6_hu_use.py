#!/usr/bin/env python3
"""Does the trained M6 use the CT intensity? For every seed and fold, the held-out planes (every third)
are passed through the trained network twice: with their z and HU, and with z and HU replaced inside
each soft-tissue class by the class median of the plane (CT intensity removed, labels and context
kept). Reported: the mean |change| of the log backscatter factor on the modulated pixels, the spread
(standard deviation) of the factor there for M6 and M6c, and the training agreement of both
(train_m6.py logs, mean of the last 50 steps).

  python -I scripts/m6_hu_use.py      # -> outputs/m6_hu_use.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main():
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_m6 import load_planes
    from sonospadepp import learned_echo as LE
    from sonospadepp.labels import HU_MODULATED
    from sonospadepp.spatial_echo import SpatialEcho

    planes = load_planes("data/trusted/pairs_def")
    out = {"hu_change": [], "sd_m6": [], "sd_m6c": []}
    for seed in (0, 1, 2):
        for fold in range(5):
            mods = {}
            for tag in ("ct", "lab"):
                st = torch.load(f"runs/m6/{tag}_s{seed}_f{fold}.pt")
                m = SpatialEcho(use_ct=st["use_ct"], device="cpu")
                m.net.load_state_dict(st["net"]); m.a.data[:] = st["a"]; m.net.eval()
                mods[tag] = m
            for p in [p for p in planes if LE.fold_of(p["kid"]) == fold][::3]:
                lab, z, hu = p["raw_lab"].astype(np.int64), p["raw_z"], p["raw_hu"]
                mod = np.isin(lab, list(HU_MODULATED))
                z0, hu0 = z.copy(), hu.copy()
                for c in np.unique(lab[mod]):
                    k = lab == c
                    z0[k], hu0[k] = np.median(z[k]), np.median(hu[k])
                f = np.log(mods["ct"].factor_numpy(lab, z, hu))
                f0 = np.log(mods["ct"].factor_numpy(lab, z0, hu0))
                fc = np.log(mods["lab"].factor_numpy(lab, z, hu))
                out["hu_change"].append(float(np.abs(f - f0)[mod].mean()))
                out["sd_m6"].append(float(f[mod].std())); out["sd_m6c"].append(float(fc[mod].std()))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["n_planes"] = len(out["hu_change"])
    for tag, key in (("ct", "train_r_m6"), ("lab", "train_r_m6c")):
        logs = [json.load(open(f"runs/m6/{tag}_s{s}.json")) for s in (0, 1, 2)]
        res[key] = float(np.mean([d[f]["train_r_last50"] for d in logs for f in d]))
        res[f"n_params_{tag}"] = logs[0]["0"]["n_params"]
    json.dump(res, open("outputs/m6_hu_use.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
