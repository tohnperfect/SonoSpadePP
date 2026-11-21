#!/usr/bin/env python3
"""Alignment-tolerant same-plane agreement: does a model's output match the real ultrasound better
than another's once the residual misregistration (median landmark residual about 5.5 mm) is allowed
for?

For every test plane (data/trusted/pairs_def) and every model output, a smooth displacement field
(--grid x --grid control points, bilinear, at most --max-px pixels, 4 px = 5 mm) is fitted to the
output to maximise the detrended agreement with the real US (mean over the eroded kidney and the
kidney + band, 1.5 px smoothing, the loss of M4 to M8). The same procedure is applied to every row,
so a row gains here only if its structure matches the patient once allowed to shift by a few mm.

Rows: b0 (computed from runs/b0_s{seed}.pt) and any --pred name=path_with_{s} (gen of the preds npz;
planes its `done` marks as not predicted are NaN).
Output: outputs/trusted_aligned.json with per-plane r before and after alignment, per region.

  python -I scripts/eval_aligned.py --pred m7=outputs/m7/preds_ct_s{s}.npz --pred m7c=outputs/m7/preds_lab_s{s}.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-shards", default="data/trusted/pairs_def")
    ap.add_argument("--pred", action="append", default=[], help="name=path containing {s} for the seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-b0", action="store_true")
    ap.add_argument("--b0", default="runs/b0_s{s}.pt", help="checkpoint pattern of the row computed from a model")
    ap.add_argument("--b0-name", default="b0", help="row name of that model (merge: b0_v2)")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--max-px", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--out", default="outputs/trusted_aligned.json")
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from usbg import render_lotus as R
    from sonospadepp import learned_echo as LE
    from sonospadepp import metrics as M
    from sonospadepp.variants import Translator
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_m7 import load_planes, prepare_regions, stack

    dev = R.get_device()
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
    rad = M._sector_radius((128, 128))
    kern = LE.gauss_kernel(args.sigma, device=dev)
    planes = load_planes(args.test_shards)
    prepare_regions(planes, rad, args.sigma, t)
    kid_ids = [p["kid"] for p in planes]
    res = {"kidney_id": kid_ids, "kidneys": sorted(set(kid_ids)), "grid": args.grid, "max_px": args.max_px, "rows": {}}
    if args.append and os.path.exists(args.out):
        old = json.load(open(args.out)); assert old["kidney_id"] == kid_ids; res["rows"] = old["rows"]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 128, device=dev), torch.linspace(-1, 1, 128, device=dev), indexing="ij")
    base = torch.stack([xx, yy], -1)[None]
    scale = args.max_px * 2 / 127.0

    def region_r(sm, ids):
        """(B, n_regions) detrended r, NaN where a plane lacks a region."""
        out = torch.full((len(ids), 2), float("nan"), device=dev)
        for b, i in enumerate(ids):
            for j, rg in enumerate(planes[i]["regions"]):
                out[b, j] = LE.detrended_r_t(sm[b], *rg)
        return out

    def align(imgs, keep):
        """imgs (N,128,128) float32 [0,1] -> per-plane r before/after, kidney and band (NaN where not keep)."""
        before = np.full((len(planes), 2), np.nan); after = np.full((len(planes), 2), np.nan)
        ok = [i for i, p in enumerate(planes) if p["regions"] and keep[i]]
        for j in range(0, len(ok), args.batch):
            ids = ok[j:j + args.batch]
            x = t(imgs[ids])[:, None]
            with torch.no_grad():
                before[ids] = region_r(LE.smooth_t(x[:, 0], kern), ids).cpu().numpy()
            prm = torch.zeros(len(ids), 2, args.grid, args.grid, device=dev, requires_grad=True)
            opt = torch.optim.Adam([prm], lr=args.lr)

            def warped():
                d = F.interpolate(prm, size=(128, 128), mode="bilinear", align_corners=True)
                g = base + torch.tanh(d).permute(0, 2, 3, 1) * scale
                return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
            for _ in range(args.steps):
                r = region_r(LE.smooth_t(warped(), kern), ids)
                loss = -torch.nanmean(r, 1).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                after[ids] = region_r(LE.smooth_t(warped(), kern), ids).cpu().numpy()
        return before, after

    rows = []
    if not args.no_b0:
        rows += [(args.b0_name, s, None) for s in args.seeds]
    for spec in args.pred:
        name, path = spec.split("=", 1)
        rows += [(name, s, path.format(s=s)) for s in args.seeds]
    for name, s, path in rows:
        t0 = time.time()
        keep = np.ones(len(planes), bool)
        if path is None:
            tr = Translator(args.b0.format(s=s), device=dev)
            imgs = np.concatenate([tr.batch(stack(planes, list(range(j, min(j + 32, len(planes)))), "phys"),
                                            stack(planes, list(range(j, min(j + 32, len(planes)))), "lab"))
                                   for j in range(0, len(planes), 32)])
        else:
            pr = np.load(path, allow_pickle=False)
            assert pr["sessions"].tolist() == kid_ids, f"{path}: planes not aligned"
            imgs = pr["gen"].astype(np.float32)
            if "done" in pr.files:
                keep = pr["done"].astype(bool)
        before, after = align(imgs.astype(np.float32), keep)
        res["rows"][f"{name}_s{s}"] = {"variant": name, "seed": s, "r_kidney_before": before[:, 0].tolist(),
                                       "r_band_before": before[:, 1].tolist(), "r_kidney_after": after[:, 0].tolist(),
                                       "r_band_after": after[:, 1].tolist()}
        print(f"{name}_s{s}: kidney {np.nanmean(before[:, 0]):+.3f} -> {np.nanmean(after[:, 0]):+.3f}, "
              f"band {np.nanmean(before[:, 1]):+.3f} -> {np.nanmean(after[:, 1]):+.3f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(args.out, "w"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
