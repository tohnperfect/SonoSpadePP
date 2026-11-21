#!/usr/bin/env python3
"""Train a SonoSPADE texture stage for v2 experiments.

  --variant sonospade      runs the vendored v1 script (scripts/v1/train_texture_gan.py) unchanged
                           (backward compatibility: v1 numbers are reproduced by v1's own code)
  --variant sonospade_v2   v1's training loop (same data sampling and RNG use) with the SonoSPADEv2 trainer;
                           the v2 additions are switched on by the flags below

Every run writes its resolved configuration next to the checkpoint (<out>.json) and inside it (run_config).

  python -I scripts/v2/train_v2.py --variant sonospade_v2 --fm-disc dinov2 --lambda-fm-adv 0.5 \
      --lambda-fm-tissue 1.0 --real-split train --seed 0 --out runs/v2/wp1/e1c_dinov2_s0.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1_SCRIPT = os.path.join(HERE, "..", "v1", "train_texture_gan.py")


def run_v1(args):
    spec = importlib.util.spec_from_file_location("v1_train_texture_gan", V1_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = ["--variant", "sonospade", "--pairs", args.pairs, "--real-dir", args.real_dir, "--s-us", args.s_us,
            "--config", args.config, "--iters", str(args.iters), "--seed", str(args.seed), "--out", args.out,
            "--batch", str(args.batch), "--preview", args.preview or os.devnull]
    if args.real_split != "all":
        raise SystemExit("--variant sonospade trains on the full pool only (v1's protocol)")
    return mod.main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sonospade", "sonospade_v2"], default="sonospade_v2")
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--real-dir", default="data/real/kaggle_vitale/pool")
    ap.add_argument("--real-split", choices=["all", "train", "val"], default="all",
                    help="pool subset from data/splits_v2.json: all (final runs), train (selection runs)")
    ap.add_argument("--splits", default="data/splits_v2.json")
    ap.add_argument("--s-us", default="data/dataset/s_us_grouped.pt",
                    help="frozen S_US: v1's U-Net (scratch) or a WP3 FMSegmenter from train_sus_v2.py")
    ap.add_argument("--config", default="configs/sonospade.yaml", help="v1 hyperparameters (unchanged)")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    # A1 (WP1)
    ap.add_argument("--fm-disc", choices=["none", "dinov2", "usfm", "echocare"], default="none")
    ap.add_argument("--lambda-fm-adv", type=float, default=0.0)
    ap.add_argument("--lambda-fm-tissue", type=float, default=0.0)
    ap.add_argument("--fm-layers", type=int, nargs="*", default=None)
    ap.add_argument("--diffaug", default="color,translation,cutout")
    ap.add_argument("--no-fan", action="store_true")
    # A2 (WP2)
    ap.add_argument("--style", choices=["none", "sean", "inade"], default="none")
    ap.add_argument("--lambda-ms", type=float, default=1.0, help="INADE: mode-seeking weight")
    ap.add_argument("--style-dim", type=int, default=16)
    ap.add_argument("--lambda-style-rec", type=float, default=1.0)
    ap.add_argument("--lambda-style-kl", type=float, default=0.01)
    # A5 (WP5)
    ap.add_argument("--speckle-loss", action="store_true")
    ap.add_argument("--lambda-speckle", type=float, default=1.0)
    ap.add_argument("--noise-inject", action="store_true")
    ap.add_argument("--dyn-range-db", type=float, default=60.0)
    ap.add_argument("--gain-sigma", type=float, default=0.0,
                    help="> 0: gain-field generator, the output keeps the render's fine structure (blur sigma, px)")
    args = ap.parse_args(argv)

    if args.variant == "sonospade":
        return run_v1(args)

    import torch
    from usbg import texture_gan as TG
    from usbg import segmenter as SEG
    from usbg.render_lotus import get_device
    from sonospadepp.v2 import splits as SP
    from sonospadepp.v2.evaldata import fan_mask, load_pairs, load_pool
    from sonospadepp.v2.trainer import SonoSPADEv2

    device = get_device()
    torch.manual_seed(args.seed)                                   # v1's order: torch seed, then numpy rng
    rng = np.random.default_rng(args.seed)
    cfg = TG.load_config(args.config)
    mcfg, lcfg, tcfg = cfg.get("model", {}), cfg.get("loss", {}), cfg.get("train", {})
    K = mcfg.get("n_classes", 13)
    imgs, labs, _ = load_pairs(args.pairs)
    idx = None if args.real_split == "all" else SP.pool_indices(SP.load(args.splits), args.real_split)
    real_pool = load_pool(args.real_dir, idx)
    fan = fan_mask(real_pool)
    from sonospadepp.v2.fm_seg import load_any_segmenter
    s_us = load_any_segmenter(args.s_us, device=device)          # v1 U-Net or a WP3 FMSegmenter

    v2 = {"fm_disc": args.fm_disc, "lambda_fm_adv": args.lambda_fm_adv, "lambda_fm_tissue": args.lambda_fm_tissue,
          "fm_layers": args.fm_layers, "diffaug": args.diffaug, "use_fan": not args.no_fan,
          "style": args.style, "style_dim": args.style_dim, "lambda_style_rec": args.lambda_style_rec,
          "lambda_style_kl": args.lambda_style_kl, "lambda_ms": args.lambda_ms,
          "lambda_speckle": args.lambda_speckle if args.speckle_loss else 0.0, "noise_inject": args.noise_inject,
          "dyn_range_db": args.dyn_range_db, "gain_sigma": args.gain_sigma}
    gan = SonoSPADEv2(device, v2=v2, fan=fan, s_us=s_us, n_classes=K, ngf=mcfg.get("ngf", 32),
                      ndf=mcfg.get("ndf", 32), n_blocks=mcfg.get("n_blocks", 6),
                      spade_hidden=mcfg.get("spade_hidden", 64), lambda_nce=lcfg.get("lambda_nce", 1.0),
                      lambda_seg=lcfg.get("lambda_seg", 5.0), lambda_seg_dice=lcfg.get("lambda_seg_dice", 1.0),
                      lambda_tc=lcfg.get("lambda_tc", 0.0), lambda_lm=lcfg.get("lambda_labelmix", 0.0),
                      r1_gamma=lcfg.get("r1_gamma", 0.0), ema_decay=tcfg.get("ema_decay", 0.999),
                      nce_tau=lcfg.get("nce_tau", 0.07), lr=cfg.get("optim", {}).get("lr", 2e-4))
    run_config = {"argv": sys.argv[1:] if argv is None else list(argv), "args": vars(args), "v1_config": cfg,
                  "v2": gan.v2, "n_real": int(len(real_pool)), "device": str(device),
                  "torch": torch.__version__}
    print(f"[sonospade_v2] pairs {len(imgs)}, real {len(real_pool)} ({args.real_split}), device {device}, "
          f"{args.iters} iters, v2 {gan.v2}", flush=True)

    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)

    hist = []
    t0 = time.time()
    for it in range(args.iters):
        pi = rng.integers(0, len(imgs), args.batch)
        ri = rng.integers(0, len(real_pool), args.batch)
        phys = to_t(np.stack([imgs[i] for i in pi], 0)[:, None] * 2 - 1)
        seg = TG.onehot_labels(np.stack([labs[i] for i in pi], 0), K, device)
        real = to_t(np.stack([real_pool[i] for i in ri], 0)[:, None] * 2 - 1)
        losses = gan.step(phys, seg, real)
        if (it + 1) % args.log_every == 0 or it + 1 == args.iters:
            losses["iter"] = it + 1
            losses["sec"] = round(time.time() - t0, 1)
            hist.append(losses)
            print("  " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                   for k, v in losses.items()), flush=True)
    run_config["loss_history"] = hist
    if gan.E_s is not None:                       # style bank from the training pool (no encoder at inference)
        gan.build_style_bank(real_pool)
    run_config["s_us"] = args.s_us
    run_config = json.loads(json.dumps(run_config, default=str))     # plain types only (weights_only loading)
    gan.save(args.out, run_config=run_config)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(run_config, fh, indent=1, default=str)
    print(f"saved -> {args.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
