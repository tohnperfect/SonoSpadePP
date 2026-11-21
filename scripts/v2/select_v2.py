#!/usr/bin/env python3
"""Model-selection score on the real-pool VALIDATION split (sequences held out of selection runs' training;
never the evaluation frames). Writes one row per checkpoint to results/v2/selection.csv.

Score (fixed before WP1 ran): fan-masked whole-frame KID (x1e3) between the 108 validation frames and the
first 200 translated content frames, with an extractor NOT used in the model's training loss (USFM for a
DINOv2- or EchoCare-trained model, DINOv2 for a USFM-trained one; both for models without D_fm). Constraints:
label-swap locality (v1 definition and 8 px band, first 40 content frames) and edge IoU at least 95% of the
v1 reference trained on the same selection pool.

  python -I scripts/v2/select_v2.py --name e1a_dinov2_adv0.5 --ckpt runs/v2/wp1/e1a_dinov2_adv0.5_s0.pt
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", default="sonospade_v2")
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--pool", default="data/real/kaggle_vitale/pool")
    ap.add_argument("--splits", default="data/splits_v2.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-locality", type=int, default=40)
    ap.add_argument("--out", default="results/v2/selection.csv")
    ap.add_argument("--speckle", action="store_true",
                    help="also score label-free speckle statistics over the whole fan (WP5 selection); use with a "
                         "separate --out, since it adds columns")
    ap.add_argument("--diversity", action="store_true",
                    help="also score per-tissue diversity of sampled style codes on the content frames (LPIPS, label-"
                         "free, no real frames; WP2 INADE selection); use with a separate --out")
    args = ap.parse_args(argv)

    from usbg import texture_eval as TE
    from usbg.render_lotus import get_device
    from sonospadepp.v2 import features as F
    from sonospadepp.v2 import metrics as M
    from sonospadepp.v2 import splits as SP
    from sonospadepp.v2.evaldata import Stage, fan_mask, load_pairs, load_pool

    device = get_device()
    sp = SP.load(args.splits)
    val = load_pool(args.pool, SP.pool_indices(sp, "val"))
    fan = fan_mask(load_pool(args.pool, SP.pool_indices(sp, "train")))
    phys, labs, _ = load_pairs(args.pairs, args.limit)
    st = Stage(args.kind, args.ckpt, device=device)
    trained = st.translator.config.get("fm_disc", "none") if args.kind == "sonospade_v2" else "none"
    trans = np.stack([st(p, l) for p, l in zip(phys, labs)])
    row = dict(name=args.name, ckpt=args.ckpt, trained_fm=trained, date=datetime.date.today().isoformat())
    for ext in ("usfm", "dinov2"):
        if ext == trained:
            row[f"val_fan_kid_{ext}"] = ""
            continue
        enc = F.FMEncoder(ext, device)
        a = F.extract_global(enc, val * fan)
        b = F.extract_global(enc, trans * fan)
        row[f"val_fan_kid_{ext}"] = round(1e3 * M.kid(a, b)[0], 4)
    row["edge_iou"] = round(float(np.mean([TE.edge_iou(p, t) for p, t in zip(phys, trans)])), 4)
    loc0, loc8 = [], []
    if st.label_aware:
        with st.fixed_noise(0):                       # A5 noise identical in base and swapped outputs
            for img, lab in zip(phys[:args.n_locality], labs[:args.n_locality]):
                loc0.append({c: v["locality"] for c, v in M.label_swap_locality(st, img, lab, band=0).items()})
                loc8.append({c: v["locality"] for c, v in M.label_swap_locality(st, img, lab, band=8).items()})
        row["locality_band0"] = round(float(np.mean(list(TE.summarize_per_class(loc0).values()))), 3)
        row["locality_band8"] = round(float(np.mean(list(TE.summarize_per_class(loc8).values()))), 3)
    fields = ["name", "ckpt", "trained_fm", "val_fan_kid_usfm", "val_fan_kid_dinov2", "edge_iou", "locality_band0",
              "locality_band8", "date"]
    if args.speckle:                     # label-free: the validation frames have no masks, so use the whole fan
        rm = np.concatenate([M.nakagami_windows(v, fan, win=7, stride=3, erode=1)[0] for v in val])
        fm = np.concatenate([M.nakagami_windows(t, fan, win=7, stride=3, erode=1)[0] for t in trans])
        row["val_fan_logm_w1"] = round(M.w1_1d(np.log(rm), np.log(fm)), 4)
        ra = [M.speckle_acov_sum(v, fan, patch=8, stride=4, erode=1) for v in val]
        fa = [M.speckle_acov_sum(t, fan, patch=8, stride=4, erode=1) for t in trans]
        rs, fs = sum(a[0] for a in ra), sum(a[0] for a in fa)
        errs = []
        for b in range(rs.shape[0]):
            for r_, f_ in zip(M.fwhm_from_acov(rs[b]), M.fwhm_from_acov(fs[b])):
                if np.isfinite(r_) and np.isfinite(f_):
                    errs.append(abs(f_ - r_))
        row["val_fan_speckle_size_err"] = round(float(np.mean(errs)), 4) if errs else ""
        fields += ["val_fan_logm_w1", "val_fan_speckle_size_err"]
    if args.diversity and not getattr(getattr(st, "translator", None), "has_style", False):
        row["diversity_lpips"], row["style_swap_locality"] = 0.0, ""        # no codes: one output per label
        fields += ["diversity_lpips", "style_swap_locality"]
    elif args.diversity:                 # synthetic content only: 4 code samples x 20 frames, as eval_v2.py
        from sonospadepp.v2.diversity import eval_diversity

        class _R:
            rows = {}

            def add(self, metric, value, tissue="all", **kw):
                self.rows[(metric, tissue)] = value
        r = _R()
        eval_diversity(r, st.translator, phys, labs, argparse.Namespace(n_locality=args.n_locality, band=8))
        row["diversity_lpips"] = round(float(r.rows.get(("diversity_lpips", "all"), float("nan"))), 5)
        row["style_swap_locality"] = round(float(r.rows.get(("style_swap_locality", "all"), float("nan"))), 4)
        fields += ["diversity_lpips", "style_swap_locality"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
