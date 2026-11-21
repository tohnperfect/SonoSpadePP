#!/usr/bin/env python3
"""Train one variant (b0 | m1 | m2 | m3) with SonoSPADE's objective and the release's loss weights
(configs/sonospade.yaml, copied from SonoSpade v0.2.0).

  python scripts/train_variant.py --variant m3 --pairs data/pairs_train.npz --s-us runs/s_us_phys_b0__lab_base.pt \
      --real-dir data/real_pool --iters 1500 --seed 0 --out runs/m3_s0.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def load_real_dir(d, size):
    from PIL import Image
    out = []
    for root, _dirs, files in os.walk(d):
        for fn in sorted(files):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                im = Image.open(os.path.join(root, fn)).convert("L").resize((size, size))
                out.append(np.asarray(im, np.float32) / 255.0)
    if not out:
        raise SystemExit(f"no images under {d}")
    return np.asarray(out, np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["b0", "m1", "m2", "m3"])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--s-us", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--config", default="configs/sonospade.yaml")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from usbg import segmenter as SEG
    from usbg import texture_gan as TG
    from usbg.render_lotus import get_device
    from sonospadepp.variants import VARIANTS, HUSonoSPADE, cond_tensor

    v = VARIANTS[args.variant]
    dev = get_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = TG.load_config(args.config)
    mc, lc, tc = cfg.get("model", {}), cfg.get("loss", {}), cfg.get("train", {})
    d = np.load(args.pairs, allow_pickle=False)
    phys = d[v["phys"]].astype(np.float32) / 255.0
    labs = d[v["lab"]].astype(np.int64)
    zs = d["z"].astype(np.float32) if v["hu"] else None
    real = load_real_dir(args.real_dir, phys.shape[1])
    from sonospadepp.v2.fm_seg import load_any_segmenter
    s_us = load_any_segmenter(args.s_us, device=dev)          # v1 U-Net, or a v2 foundation-model S_US
    kw = dict(s_us=s_us, ngf=mc.get("ngf", 32), ndf=mc.get("ndf", 32), n_blocks=mc.get("n_blocks", 6),
              spade_hidden=mc.get("spade_hidden", 64), lambda_nce=lc.get("lambda_nce", 1.0),
              lambda_seg=lc.get("lambda_seg", 5.0), lambda_seg_dice=lc.get("lambda_seg_dice", 1.0),
              lambda_tc=lc.get("lambda_tc", 0.0), lambda_lm=lc.get("lambda_labelmix", 0.0),
              r1_gamma=lc.get("r1_gamma", 0.0), ema_decay=tc.get("ema_decay", 0.999),
              nce_tau=lc.get("nce_tau", 0.07), lr=cfg.get("optim", {}).get("lr", 2e-4))
    gan = HUSonoSPADE(dev, n_lab=v["n_lab"], **kw) if v["hu"] else TG.SonoSPADE(dev, n_classes=v["n_lab"], **kw)
    print(f"[{args.variant}] {v['title']}: pairs {len(phys)}, real {len(real)}, S_US {args.s_us}, "
          f"{args.iters} iters, seed {args.seed}, {dev}", flush=True)

    def t(a):
        return torch.from_numpy(a[:, None].astype(np.float32) * 2 - 1).to(dev)

    t0 = time.time()
    log = []
    for it in range(args.iters):
        pi = rng.integers(0, len(phys), args.batch)
        ri = rng.integers(0, len(real), args.batch)
        cond = cond_tensor(labs[pi], zs[pi] if zs is not None else None, v["n_lab"], v["hu"], dev)
        losses = gan.step(t(phys[pi]), cond, t(real[ri]))
        if (it + 1) % max(1, args.iters // 10) == 0:
            log.append({"iter": it + 1, **losses})
            print(f"  iter {it + 1:5d}  " + "  ".join(f"{k} {x:.3f}" for k, x in losses.items()), flush=True)
    gan.save(args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({"variant": args.variant, "seed": args.seed, "iters": args.iters, "s_us": args.s_us,
                   "pairs": args.pairs, "minutes": round((time.time() - t0) / 60, 1), "log": log}, fh, indent=2)
    print(f"saved -> {args.out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
