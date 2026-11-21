#!/usr/bin/env python3
"""Score the full-resolution planes (trusted_hires_prepare.py) for both registrations.

Rows, each scored against the real US of the same plane at 0.3 mm (hires.scores) with smoothing of
1 mm and 2 mm:
  ct_hu      the CT intensity of the plane itself
  phys_b0    physics render from the labels (320 x 512 scan grid, scan-converted)
  phys_m2    M2's HU-modulated render;  phys_m2b  M2 with backscatter modulation only
  --pred     learned renders (train_hr.py), Cartesian uint8, NaN where a plane has no prediction
Output: one row per (name, registration, sigma) in outputs/trusted_hires.json.

  python -I scripts/eval_hires.py [--pred m4hr@def=outputs/hires/preds_pop_hu_def.npz ...] [--append]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import replace

import numpy as np


def load_hires(d="data/trusted/hires"):
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    return [np.load(f, allow_pickle=False) for f in files]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hires", default="data/trusted/hires")
    ap.add_argument("--cache", default="data/trusted/pp_cache")
    ap.add_argument("--pred", action="append", default=[], help="name@reg=path")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--no-fixed", action="store_true", help="only the --pred rows")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--out", default="outputs/trusted_hires.json")
    args = ap.parse_args(argv)

    from usbg import render_lotus as R
    from sonospadepp import hires as H
    from sonospadepp.ctvol import load_pp
    from sonospadepp.physics import maps_from_labels, maps_hu_modulated, render_from_maps

    dev = R.get_device()
    parts = load_hires(args.hires)
    kid_ids = [str(k) for p in parts for k in p["sessions"]]
    _, fan, rad, th = H.cart_grid()
    L, S = parts[0]["pol_lab_rig"].shape[1:]
    s_idx, l_idx = H.polar_index(rad, th, L, S)
    lat_scale = L / 160.0
    res = {"kidney_id": kid_ids, "kidneys": sorted(set(kid_ids)), "mm": H.MM, "rows": {}}
    if args.append and os.path.exists(args.out):
        old = json.load(open(args.out))
        assert old["kidney_id"] == kid_ids, "different planes"
        res["rows"] = old["rows"]

    real = [p["real"][j].astype(np.float32) / 255.0 for p in parts for j in range(len(p["sessions"]))]
    kus = [p["kidney_us"][j] for p in parts for j in range(len(p["sessions"]))]
    valid = [p["valid"][j] for p in parts for j in range(len(p["sessions"]))]

    def score_all(name, reg, imgs, keep=None):
        for sg in args.sigmas:
            row = {"variant": name, "reg": reg, "sigma_mm": sg}
            per = [H.scores(f, r, k, v, rad, sigma_mm=sg) if (keep is None or keep[i]) else None
                   for i, (f, r, k, v) in enumerate(zip(imgs, real, kus, valid))]
            for key in ("r_kidney", "r_band", "r_fan", "r_kidney_dt", "r_band_dt"):
                row[key] = [float("nan") if p is None else p[key] for p in per]
            res["rows"][f"{name}@{reg}@s{sg:g}"] = row
            print(f"{name:10s} {reg} sigma {sg:g} mm: kidney_dt {np.nanmean(row['r_kidney_dt']):+.3f} "
                  f"band_dt {np.nanmean(row['r_band_dt']):+.3f} fan {np.nanmean(row['r_fan']):+.3f}", flush=True)

    if not args.no_fixed:
        refs = {}
        for reg in ("rig", "def"):
            out = {"ct_hu": [], "phys_b0": [], "phys_m2": [], "phys_m2b": []}
            for p in parts:
                pid = str(p["sessions"][0])[:-1]
                if pid not in refs:
                    refs[pid] = load_pp(os.path.join(args.cache, f"{pid}.npz")).ref
                for j in range(len(p["sessions"])):
                    out["ct_hu"].append((np.clip(p[f"cart_hu_{reg}"][j].astype(np.float32), -200, 300) + 200) / 500.0)
                    lab = p[f"pol_lab_{reg}"][j].astype(np.int64)
                    hu = p[f"pol_hu_{reg}"][j].astype(np.float32); z = p[f"pol_z_{reg}"][j].astype(np.float32)
                    cfg = replace(R.RenderConfig(), seed=int(p["seeds"][j]), psf_sigma_lat0=R.RenderConfig().psf_sigma_lat0 * lat_scale)
                    Z0, MU0, BS0 = maps_from_labels(lab, cfg)
                    Zm, _, BSm = maps_hu_modulated(lab, hu, z, refs[pid], cfg)
                    for name, (Z, BS) in (("phys_b0", (Z0, BS0)), ("phys_m2", (Zm, BSm)), ("phys_m2b", (Z0, BSm))):
                        out[name].append(H.scan_convert(render_from_maps(Z, MU0, BS, cfg, dev), s_idx, l_idx, fan))
            for name, imgs in out.items():
                score_all(name, reg, imgs)
    for spec in args.pred:
        key, path = spec.split("=", 1)
        name, reg = key.split("@")
        pr = np.load(path, allow_pickle=False)
        assert pr["sessions"].tolist() == kid_ids, f"{path}: planes not aligned"
        imgs = [a.astype(np.float32) / 255.0 for a in pr["phys"]]
        score_all(name, reg, imgs, pr["done"].astype(bool))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
