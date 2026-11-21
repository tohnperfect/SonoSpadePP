#!/usr/bin/env python3
"""Non-learned baselines for the one-image setting of M9 (the central native plane of each kidney is
the one real US image given; the other six native planes are predicted):

  copy      the given US image itself, unchanged, as the prediction for every other plane;
  labwarp   the given US image warped onto each target plane by a smooth deformation (8 x 8 control
            points, bilinear) fitted so the given plane's label map matches the target's label map
            (soft Dice over the classes present, kidney weighted x3, plus a smoothness penalty), as an
            atlas-based synthesis would.

A learned one-image model has to beat these as well as the models without the image.
Outputs: outputs/m9/preds_copy.npz and preds_labwarp.npz (seed-independent).

  python -I scripts/ref_baselines.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-shards", default="data/trusted/pairs_def")
    ap.add_argument("--out", default="outputs/m9")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5, help="Adam step on control-point displacements (px)")
    ap.add_argument("--w-smooth", type=float, default=0.02)
    ap.add_argument("--kidney-class", type=int, default=None, help="label id of the kidney (default: from metrics)")
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from usbg import render_lotus as R
    from sonospadepp import learned_echo as LE
    from sonospadepp import paired_translator as PT
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_m7 import load_planes
    from train_m9 import native_refs

    dev = R.get_device()
    planes = load_planes(args.test_shards)
    refs = native_refs(planes, args.test_shards)
    kern = LE.gauss_kernel(2.0, device=dev)
    n = len(planes)
    copy = np.zeros((n, 128, 128), np.float16); labwarp = np.zeros_like(copy); done = np.zeros(n, bool)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 128, device=dev), torch.linspace(-1, 1, 128, device=dev), indexing="ij")
    base = torch.stack([xx, yy], -1)[None]

    def onehot(lab):
        x = F.one_hot(torch.as_tensor(lab, device=dev).long().clamp(0, PT.N_LAB - 1), PT.N_LAB).permute(0, 3, 1, 2).float()
        return torch.stack([LE.smooth_t(x[:, c], kern) for c in range(PT.N_LAB)], 1)

    t0 = time.time(); fin = []
    for kid, (r, tg) in sorted(refs.items()):
        ref_real = planes[r]["real"].astype(np.float32)
        copy[tg] = ref_real.astype(np.float16)
        # the kidney class is the label most frequent inside the kidney mask of the given plane
        kc = args.kidney_class if args.kidney_class is not None else int(np.bincount(planes[r]["lab"][planes[r]["kidney"]],
                                                                                   minlength=PT.N_LAB).argmax())
        B = len(tg)
        src = onehot(np.stack([planes[r]["lab"]] * B)); dst = onehot(np.stack([planes[i]["lab"] for i in tg]))
        present = ((src.sum((2, 3)) > 20) | (dst.sum((2, 3)) > 20)).float()
        present[:, 0] = 0                                                  # background
        wcls = present.clone(); wcls[:, kc] *= 3
        prm = torch.zeros(B, 2, args.grid, args.grid, device=dev, requires_grad=True)
        opt = torch.optim.Adam([prm], lr=args.lr)
        for _ in range(args.steps):
            d = F.interpolate(prm, size=(128, 128), mode="bilinear", align_corners=True) * (2 / 127.0)
            g = base + d.permute(0, 2, 3, 1)
            w = F.grid_sample(src, g, mode="bilinear", padding_mode="border", align_corners=True)
            inter = (w * dst).sum((2, 3)); den = w.sum((2, 3)) + dst.sum((2, 3)) + 1e-3
            dice = ((2 * inter / den) * wcls).sum(1) / wcls.sum(1)
            sm = ((prm[:, :, 1:] - prm[:, :, :-1]) ** 2).mean() + ((prm[:, :, :, 1:] - prm[:, :, :, :-1]) ** 2).mean()
            loss = -dice.sum() + args.w_smooth * sm * B
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            d = F.interpolate(prm, size=(128, 128), mode="bilinear", align_corners=True) * (2 / 127.0)
            g = base + d.permute(0, 2, 3, 1)
            img = torch.as_tensor(np.stack([ref_real] * B), device=dev)[:, None]
            labwarp[tg] = F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0].cpu().numpy()
            fin.append(float(dice.mean()))
        done[tg] = True
    print(f"{len(refs)} kidneys, {int(done.sum())} target planes, label Dice after warp {np.mean(fin):.3f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    phys = np.stack([np.round(p["phys"] * 255) for p in planes]).astype(np.uint8)
    sess = np.asarray([p["kid"] for p in planes])
    for name, g in (("copy", copy), ("labwarp", labwarp)):
        np.savez_compressed(os.path.join(args.out, f"preds_{name}.npz"), phys=phys, gen=g, done=done, sessions=sess)
    print(f"wrote {args.out}/preds_copy.npz and preds_labwarp.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
