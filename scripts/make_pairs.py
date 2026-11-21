#!/usr/bin/env python3
"""Render ONE pose sweep into every variant's training/eval arrays, so the variants differ only in
how they use the CT.

For each pose (seeded se(3) perturbation of the seated liver view, as in SonoSpade's make_dataset)
and one mild domain-randomization draw shared by all variants:
  phys_b0  physics from base labels           (b0, m3)
  phys_m1  physics from the M1 labels         (m1)
  phys_m2  HU-modulated physics, base labels  (m2)
  lab_base 13-class base labels (every variant's evaluation target)
  lab_pp   14-class M1 labels
  z        relative HU (float16)
All are resized to 128x128 and warped to the Kaggle sector geometry (n_aug extra view-augmented
copies per pose for training, sharing one warp across channels).

  python scripts/make_pairs.py --cases s0011 s0058 --n-per-case 110 --n-aug 2 --out data/pairs_train.npz
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import numpy as np
from PIL import Image
from usbg import render_lotus as R
from usbg import volume as V
from usbg.contracts_bridge import pose_from_rpy
from usbg.placement import seat_volume_under_probe
from usbg.segmenter import align_label_to_image

from sonospadepp.ctvol import load_pp
from sonospadepp.physics import maps_from_labels, maps_hu_modulated, render_from_maps
from sonospadepp.slicing import slice_all
from sonospadepp.warp import warp_frame


def mild_randomize(base, rng):
    """SonoSpade make_dataset._mild_randomize (v0.2.0), verbatim: vary gain, TGC, PSF and speckle,
    keep the per-tissue acoustic constants."""
    u = rng.uniform
    return replace(base, gain=float(u(0.85, 1.15)), tgc_slope=float(u(0.7, 1.3)),
                   psf_sigma_lat0=float(u(0.9, 1.4)), psf_focus_depth=float(u(0.03, 0.08)),
                   speckle_strength=float(u(0.7, 1.1)), noise_sigma=float(u(0.0, 0.03)),
                   dyn_range_db=float(u(48, 56)), seed=int(rng.integers(2 ** 31 - 1)))


def resize_img(img01, size):
    a = np.clip(np.asarray(img01) * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), np.uint8)


def resize_float(arr_img_orient, size):
    return np.asarray(Image.fromarray(np.asarray(arr_img_orient, np.float32), mode="F")
                      .resize((size, size), Image.BILINEAR), np.float32)


KINDS = {"phys_b0": "image", "phys_m1": "image", "phys_m2": "image",
         "lab_base": "label", "lab_pp": "label", "z": "float"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--cache", default="data/pp_cache")
    ap.add_argument("--n-per-case", type=int, default=110)
    ap.add_argument("--n-aug", type=int, default=0)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--depth-m", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    dev = R.get_device()
    rng = np.random.default_rng(args.seed)
    acc = {k: [] for k in KINDS}
    sessions = []
    for case in args.cases:
        vol = load_pp(os.path.join(args.cache, f"{case}.npz"))
        lv = vol.labeled("base")
        tgt = V.LIVER if lv.has(V.LIVER) else V.SOFT_TISSUE
        W, T_base = seat_volume_under_probe(lv, tgt, standoff_m=0.0, depth_m=args.depth_m)
        for _ in range(args.n_per_case):
            perturb = pose_from_rpy(tx=float(rng.uniform(-0.03, 0.03)), ty=float(rng.uniform(-0.03, 0.03)),
                                    tz=float(rng.uniform(-0.02, 0.02)), rx=float(rng.uniform(-12, 12)),
                                    ry=float(rng.uniform(-12, 12)), rz=float(rng.uniform(-20, 20)))
            T = T_base @ perturb
            lab_b, lab_p, hu, z = slice_all(vol, T, W)
            cfg = mild_randomize(R.RenderConfig(), rng)
            img_b = render_from_maps(*maps_from_labels(lab_b, cfg), cfg, dev)
            img_p = render_from_maps(*maps_from_labels(lab_p, cfg), cfg, dev)
            img_h = render_from_maps(*maps_hu_modulated(lab_b, hu, z, vol.ref, cfg), cfg, dev)
            frame = {
                "phys_b0": resize_img(img_b, args.size), "phys_m1": resize_img(img_p, args.size),
                "phys_m2": resize_img(img_h, args.size),
                "lab_base": align_label_to_image(lab_b, (args.size, args.size)).astype(np.uint8),
                "lab_pp": align_label_to_image(lab_p, (args.size, args.size)).astype(np.uint8),
                "z": resize_float(z.T, args.size),
            }
            views = [(0.0, 0.0, 0.0)] + [(float(rng.uniform(-15, 15)), float(rng.uniform(-0.1, 0.1)),
                                          float(rng.uniform(-0.1, 0.1))) for _ in range(args.n_aug)]
            for rot, tx, ty in views:
                w = warp_frame(frame, KINDS, rot_deg=rot, tx=tx, ty=ty)
                for k in KINDS:
                    acc[k].append(w[k])
                sessions.append(case)
        print(f"{case}: {len(sessions)} frames so far", flush=True)

    out = {k: np.stack(v) for k, v in acc.items()}
    out["z"] = out["z"].astype(np.float16)
    meta = {"cases": args.cases, "n_per_case": args.n_per_case, "n_aug": args.n_aug, "seed": args.seed,
            "size": args.size, "randomize": "mild", "geometry": "kaggle sector"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, sessions=np.asarray(sessions), meta=json.dumps(meta), **out)
    print(f"wrote {len(sessions)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
