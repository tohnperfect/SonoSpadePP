#!/usr/bin/env python3
"""Train the frozen segmenter S_US on one variant's (physics, labels) pairs, grouped scheme.

Same recipe as SonoSpade's scripts/train_segmenter.py --group (U-Net base 32 depth 4, Adam 5e-4,
batch 8, CE + Dice), on this repo's sector-warped pairs. M1's GAS label joins the air group.

  python scripts/train_s_us.py --pairs data/pairs_train.npz --phys phys_b0 --lab lab_base --out runs/s_us_b0.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--phys", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", choices=["scratch", "usfm", "dinov2", "echocare"], default="scratch",
                    help="v2 A3: encoder initialized from a foundation model (sonospadepp.v2.fm_seg), at 0.1x lr")
    ap.add_argument("--enc-lr-mult", type=float, default=0.1)
    args = ap.parse_args(argv)

    import torch
    from usbg import segmenter as SEG
    from usbg.render_lotus import get_device
    from sonospadepp.labels import grouped_map, N_GROUPED

    dev = get_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = np.load(args.pairs, allow_pickle=False)
    imgs = d[args.phys].astype(np.float32) / 255.0
    labs_c = d[args.lab].astype(np.int64)
    gmap = grouped_map(14 if args.lab == "lab_pp" else 13)
    labs = gmap[labs_c]
    sess = d["sessions"]
    cases = sorted(set(sess.tolist()))
    val_case = cases[-1]                                   # hold one CT case out, as the release splits by session
    tr = np.where(sess != val_case)[0]; va = np.where(sess == val_case)[0]
    if len(cases) == 1:                                    # single-case smoke runs: split frames instead
        val_case = f"{cases[0]} (last 20% of frames)"
        cut = int(0.8 * len(sess)); tr, va = np.arange(cut), np.arange(cut, len(sess))
    if args.init == "scratch":
        seg = SEG.Segmenter(device=dev, n_classes=N_GROUPED, base=32, depth=4, group_map=gmap)
        opt = torch.optim.Adam(seg.net.parameters(), lr=args.lr)
    else:
        from sonospadepp.v2.fm_seg import FMSegmenter
        seg = FMSegmenter(args.init, device=dev, n_classes=N_GROUPED, group_map=gmap)
        opt = torch.optim.Adam(seg.param_groups(args.lr, args.enc_lr_mult))
    seg.net.train()
    for it in range(args.iters):
        bi = rng.choice(tr, size=args.batch, replace=False)
        x = torch.from_numpy(imgs[bi][:, None]).to(dev)
        y = torch.from_numpy(labs[bi]).to(dev)
        opt.zero_grad()
        loss = SEG.seg_ce_dice_loss(seg.net(x), y, n_classes=N_GROUPED, dice_weight=1.0)
        loss.backward(); opt.step()
        if (it + 1) % max(1, args.iters // 5) == 0:
            print(f"  iter {it + 1}  loss {float(loss):.4f}", flush=True)
    seg.net.eval()
    dices = []
    with torch.no_grad():
        for j in range(0, len(va), 16):
            vb = va[j:j + 16]
            dices.append(SEG.dice_score(seg.net(torch.from_numpy(imgs[vb][:, None]).to(dev)),
                                        torch.from_numpy(labs[vb]).to(dev), n_classes=N_GROUPED))
    seg.save(args.out)
    dice = float(np.mean(dices))
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({"phys": args.phys, "lab": args.lab, "iters": args.iters, "val_case": val_case, "init": args.init,
                   "heldout_sim_dice": dice}, fh, indent=2)
    print(f"saved S_US -> {args.out}  (held-out case {val_case}, grouped Dice {dice:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
