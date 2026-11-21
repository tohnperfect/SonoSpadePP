#!/usr/bin/env python3
"""Same-patient evaluation on TRUSTED: does a variant's frame match the real US of the SAME patient
at the SAME plane better than B0's does?

Input: data/trusted_pairs.npz from scripts/trusted_prepare.py (CT-derived inputs and the real US,
pixel-aligned in the Kaggle sector geometry). For every row, per frame, Pearson r between the
smoothed synthetic and smoothed real image over (a) the kidney interior, (b) the kidney plus a band
around it, (c) the whole common fan (metrics.paired_scores). Rows:
  phys_b0 / phys_m1 / phys_m2   untranslated physics renders
  ct_hu                         the smoothed CT slice itself (how much of the US structure the CT
                                intensity predicts linearly; sign included)
  <variant>_s<seed>             each checkpoint; M3 also with its HU channel zeroed

  python scripts/eval_trusted.py --pairs data/trusted_pairs.npz --ckpt b0_s0=runs/b0_s0.pt ... \
      --out outputs/trusted.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/trusted_pairs.npz")
    ap.add_argument("--ckpt", action="append", default=[], help="name=path; name starts with the variant")
    ap.add_argument("--pred", action="append", default=[],
                    help="name=path of train_m4.py predictions, e.g. m4_s0=outputs/m4/preds_hu_s0.npz "
                         "(adds rows phys_<name> and <name>)")
    ap.add_argument("--sigma", type=float, default=1.5, help="smoothing (px) before correlating")
    ap.add_argument("--out", default="outputs/trusted.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="--pred files may leave planes without a prediction (scored as NaN, skipped in summaries)")
    ap.add_argument("--append", action="store_true",
                    help="keep the rows already in --out (same planes) and add or replace only the rows given here")
    args = ap.parse_args(argv)

    from usbg.render_lotus import get_device
    from sonospadepp import metrics as M
    from sonospadepp.variants import VARIANTS, Translator

    dev = get_device()
    d = np.load(args.pairs, allow_pickle=False)
    real = d["real"].astype(np.float32) / 255.0
    kid = d["kidney_us"].astype(bool)
    valid = d["valid"].astype(bool)
    kidney_id = d["sessions"]
    res = {"n_frames": int(len(real)), "kidneys": sorted(set(kidney_id.tolist())), "sigma": args.sigma,
           "kidney_id": kidney_id.tolist(), "rows": {}}
    if args.append and os.path.exists(args.out):
        old = json.load(open(args.out))
        assert old["kidney_id"] == res["kidney_id"] and old["sigma"] == args.sigma, f"{args.out}: different planes or sigma"
        res["rows"] = old["rows"]

    nan_scores = {k: float("nan") for k in ("r_kidney", "r_band", "r_fan", "r_kidney_dt", "r_band_dt")}

    def score(name, fake, extra, keep=None):
        keep = np.ones(len(fake), bool) if keep is None else keep
        per = [M.paired_scores(f, r, k, v, sigma=args.sigma) if ok else nan_scores
               for f, r, k, v, ok in zip(fake, real, kid, valid, keep)]
        row = dict(extra)
        for key in ("r_kidney", "r_band", "r_fan", "r_kidney_dt", "r_band_dt"):
            row[key] = [p[key] for p in per]
        res["rows"][name] = row
        means = {k: np.nanmean(row[k]) for k in ("r_kidney", "r_band", "r_fan", "r_kidney_dt", "r_band_dt")}
        print(f"{name:14s} r_kidney {means['r_kidney']:+.3f}  r_band {means['r_band']:+.3f}  "
              f"r_fan {means['r_fan']:+.3f}  detrended: kidney {means['r_kidney_dt']:+.3f} "
              f"band {means['r_band_dt']:+.3f}", flush=True)

    for key in ("phys_b0", "phys_m1", "phys_m2"):
        if key not in res["rows"]:
            score(key, d[key].astype(np.float32) / 255.0, {"kind": "physics"})
    if "ct_hu" not in res["rows"]:
        score("ct_hu", d["hu_img"].astype(np.float32), {"kind": "reference"})

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        vname = name.split("_")[0]
        v = VARIANTS[vname]
        tr = Translator(path, device=dev)
        phys = d[v["phys"]].astype(np.float32) / 255.0
        labs = d[v["lab"]].astype(np.int64)
        zs = d["z"].astype(np.float32)

        def run(zz):
            return np.concatenate([tr.batch(phys[i:i + 32], labs[i:i + 32],
                                            None if zz is None else zz[i:i + 32])
                                   for i in range(0, len(phys), 32)])
        out = run(zs if v["hu"] else None)
        score(name, out, {"kind": "variant", "variant": name.rsplit("_s", 1)[0], "ckpt": path})
        if v["hu"]:
            score(name + "_hu0", run(np.zeros_like(zs)), {"kind": "ablation", "variant": vname + "_hu0",
                                                          "ckpt": path})
    for spec in args.pred:
        name, path = spec.split("=", 1)
        vname = name.split("_")[0]
        pr = np.load(path, allow_pickle=False)
        assert pr["sessions"].tolist() == kidney_id.tolist(), f"{path}: planes not aligned with {args.pairs}"
        done = pr["done"].astype(bool)
        assert done.all() or args.allow_partial, f"{path}: not every plane has a held-out prediction"
        score("phys_" + name, pr["phys"].astype(np.float32) / 255.0, {"kind": "physics", "variant": "phys_" + vname, "pred": path}, done)
        score(name, pr["gen"].astype(np.float32), {"kind": "variant", "variant": name.rsplit("_s", 1)[0], "pred": path}, done)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
