#!/usr/bin/env python3
"""WP3: score an S_US checkpoint (v1 U-Net or v2 FMSegmenter), writing rows to results/v2/sus.csv.

  heldout_sim_dice   grouped-class Dice on held-out simulator pairs (CTs outside v1's 8 and USB Gym's 6
                     test patients: data/dataset/pairs_heldout_v2.npz). USED FOR SELECTION.
  real_dice_<organ>  pseudo-label Dice on the 60 annotated evaluation frames, per annotated organ, and their
                     macro mean (v1's analysis in the SASHIMI paper). ANALYSIS ONLY: never used to choose models.

  python -I scripts/v2/eval_sus_v2.py --name usfm --ckpt runs/v2/wp3/s_us_usfm.pt
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os

import numpy as np

ORGANS = {1: "liver", 2: "gallbladder", 3: "vessel", 4: "kidney", 7: "spleen"}   # canonical ids


def dice(pred, tgt):
    inter = np.logical_and(pred, tgt).sum()
    s = pred.sum() + tgt.sum()
    return float(2 * inter / s) if s > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", default="data/dataset/pairs_heldout_v2.npz")
    ap.add_argument("--real-pairs", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--out", default="results/v2/sus.csv")
    args = ap.parse_args(argv)

    import torch
    from usbg import segmenter as SEG
    from usbg.render_lotus import get_device
    from sonospadepp.v2.evaldata import load_pairs
    from sonospadepp.v2.fm_seg import load_any_segmenter

    dev = get_device()
    seg = load_any_segmenter(args.ckpt, device=dev)
    seg.net.eval()
    gmap = np.asarray(seg.group_map) if seg.group_map is not None else np.arange(13)

    def predict(imgs):
        out = []
        with torch.no_grad():
            for j in range(0, len(imgs), 16):
                x = torch.from_numpy(np.asarray(imgs[j:j + 16], np.float32))[:, None].to(dev)
                out.append(seg.net(x).argmax(1).cpu().numpy())
        return np.concatenate(out)

    rows = []
    date = datetime.date.today().isoformat()
    hi, hl, _ = load_pairs(args.heldout)
    hp = predict(hi)
    hg = gmap[hl]
    with torch.no_grad():
        d = SEG.dice_score(torch.nn.functional.one_hot(torch.from_numpy(hp), seg.n_classes).permute(0, 3, 1, 2)
                           .float() * 50.0, torch.from_numpy(hg), n_classes=seg.n_classes)
    rows.append(("heldout_sim_dice", "macro", float(d), len(hi)))

    ri, rl, _ = load_pairs(args.real_pairs)
    rp = predict(ri)
    per, perf = {}, {}
    for c, name in ORGANS.items():
        g = int(gmap[c])
        p_, t_ = rp == g, rl == c                     # pooled over the 60 frames: the SASHIMI paper's definition
        per[name] = float((2 * np.logical_and(p_, t_).sum() + 1e-6) / (p_.sum() + t_.sum() + 1e-6))
        rows.append(("real_dice", name, per[name], int(t_.any(axis=(1, 2)).sum())))
        vals = [dice(p == g, t == c) for p, t in zip(rp, rl) if (t == c).any()]
        perf[name] = float(np.mean(vals)) if vals else float("nan")
        rows.append(("real_dice_per_frame", name, perf[name], len(vals)))
    rows.append(("real_dice", "macro", float(np.mean(list(per.values()))), len(ri)))
    rows.append(("real_dice_per_frame", "macro", float(np.nanmean(list(perf.values()))), len(ri)))

    new = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["name", "ckpt", "metric", "tissue", "value", "n", "use", "date"])
        for m, t, v, n in rows:
            w.writerow([args.name, args.ckpt, m, t, round(v, 4), n,
                        "selection" if m == "heldout_sim_dice" else "analysis only", date])
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
