#!/usr/bin/env python3
"""WP3: train S_US with a foundation-model initialization (--sus-init usfm | dinov2 | ...), with v1's protocol
(scripts/v1/train_segmenter.py): aligned simulator pairs, grouped 9-class scheme, CE + Dice, whole CT cases held
out for validation (val fraction 0.34, seed 0, the same split v1 used), batch 8, the iteration budget of
configs/sonospade.yaml. The encoder is fine-tuned at 0.1x the head's learning rate (or frozen).
--sus-init scratch delegates to the vendored v1 script unchanged.

  python -I scripts/v2/train_sus_v2.py --sus-init usfm --out runs/v2/wp3/s_us_usfm.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sus-init", choices=["scratch", "usfm", "dinov2", "medsam", "biomedparse", "echocare"],
                    required=True)
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--config", default="configs/sonospade.yaml")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--enc-lr-mult", type=float, default=0.1)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.34)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.sus_init == "scratch":
        spec = importlib.util.spec_from_file_location("v1_train_segmenter", os.path.join(HERE, "..", "v1",
                                                                                         "train_segmenter.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main(["--pairs", args.pairs, "--config", args.config, "--group", "--seed", str(args.seed),
                         "--out", args.out, "--preview", os.devnull]
                        + (["--iters", str(args.iters)] if args.iters else []))
    if args.sus_init in ("medsam", "biomedparse"):          # trained on the evaluation frames
        raise SystemExit(f"--sus-init {args.sus_init}: excluded (its training data include the evaluation frames)")

    import torch
    from usbg import segmenter as SEG
    from usbg import texture_gan as TG
    from usbg.dataset import load_pairs_npz
    from usbg.render_lotus import get_device
    from sonospadepp.v2.fm_seg import FMSegmenter

    spec = importlib.util.spec_from_file_location("v1_train_segmenter", os.path.join(HERE, "..", "v1",
                                                                                     "train_segmenter.py"))
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)

    cfg = TG.load_config(args.config)
    scfg = cfg.get("segmenter", {})
    iters = args.iters or scfg.get("iters", 3000)
    lr = args.lr or scfg.get("lr", 5e-4)
    device = get_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    imgs, labs, sessions, _ = load_pairs_npz(args.pairs)
    imgs = imgs.astype(np.float32) / 255.0
    gmap = SEG.grouped_map_array()
    labs = SEG.remap_labels(labs.astype(np.int64), gmap)
    tr_idx, va_idx = v1._session_split(sessions, args.val_frac, rng, mode="session")   # v1's split, same rng
    val_cases = sorted(set(sessions[va_idx].tolist()))
    print(f"[S_US {args.sus_init}] train {len(tr_idx)} / val {len(va_idx)} (held-out cases {val_cases}), "
          f"{iters} iters, device {device}", flush=True)
    seg = FMSegmenter(args.sus_init, device=device, n_classes=SEG.N_GROUPED, group_map=gmap)
    opt = torch.optim.Adam(seg.param_groups(lr, args.enc_lr_mult, args.freeze_encoder))

    def val_dice():
        seg.net.eval()
        with torch.no_grad():
            d = [SEG.dice_score(seg.net(torch.from_numpy(imgs[va_idx[j:j + 8]][:, None]).to(device)),
                                torch.from_numpy(labs[va_idx[j:j + 8]]).to(device), n_classes=SEG.N_GROUPED)
                 for j in range(0, len(va_idx), 8)]
        seg.net.train()
        return float(np.mean(d))

    hist, t0 = [], time.time()
    seg.net.train()
    for it in range(iters):
        bi = rng.choice(tr_idx, size=args.batch, replace=False)
        x = torch.from_numpy(imgs[bi][:, None]).to(device)
        y = torch.from_numpy(labs[bi]).to(device)
        opt.zero_grad()
        loss = SEG.seg_ce_dice_loss(seg.net(x), y, n_classes=SEG.N_GROUPED,
                                    dice_weight=cfg.get("loss", {}).get("lambda_seg_dice", 1.0))
        loss.backward()
        opt.step()
        if (it + 1) % max(1, iters // 10) == 0:
            hist.append({"iter": it + 1, "loss": float(loss.detach()), "val_dice": val_dice(),
                         "sec": round(time.time() - t0, 1)})
            print(f"  {hist[-1]}", flush=True)
    seg.save(args.out)
    final = val_dice()
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({"args": vars(args), "val_cases": val_cases, "history": hist, "heldout_sim_dice": final}, fh,
                  indent=1)
    print(f"saved -> {args.out}; held-out sim Dice {final:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
