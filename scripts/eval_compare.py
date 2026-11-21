#!/usr/bin/env python3
"""Score raw physics renders and trained variants on held-out CT content against real Kaggle frames.

Rows: phys_b0 / phys_m1 / phys_m2 (untranslated renders, to separate what the CT does in physics
from what the GAN does) and one row per checkpoint. Resumable: rows already in --out are kept.

  python scripts/eval_compare.py --pairs data/pairs_eval.npz --real-test data/real_test.npz \
      --ckpt b0_s0=runs/b0_s0.pt --ckpt m3_s0=runs/m3_s0.pt --out outputs/compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--real-test", required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="name=path; name starts with the variant")
    ap.add_argument("--physics-rows", action="store_true")
    ap.add_argument("--tstr-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tstr-iters", type=int, default=500)
    ap.add_argument("--no-tstr", action="store_true", help="skip TSTR (fast preview)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from usbg.render_lotus import get_device
    from sonospadepp import metrics as M
    from sonospadepp.variants import VARIANTS, Translator

    dev = get_device()
    d = np.load(args.pairs, allow_pickle=False)
    lab_base = d["lab_base"].astype(np.int64)
    rt = np.load(args.real_test, allow_pickle=False)
    real_imgs = rt["images"].astype(np.float32) / 255.0
    real_labs = rt["labels"].astype(np.int64)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {"rows": {}}
    res["n_eval"] = int(len(lab_base))
    res["cases"] = sorted(set(d["sessions"].tolist()))

    def score(name, trans, phys_in, extra):
        t0 = time.time()
        row = dict(extra)
        row["organ_w1"], row["organ_w1_per"] = M.organ_w1(trans, lab_base, real_imgs, real_labs)
        row["hetero_gap"], row["hetero_per"] = M.hetero_gap(trans, lab_base, real_imgs, real_labs)
        row["edge_iou"] = M.edge_iou(phys_in, trans)
        if args.no_tstr:
            row["tstr_dice"], row["tstr_sd"], row["tstr_runs"] = float("nan"), float("nan"), []
        else:
            row["tstr_dice"], row["tstr_sd"], row["tstr_runs"] = M.tstr(
                trans, lab_base, real_imgs, real_labs, seeds=args.tstr_seeds, iters=args.tstr_iters,
                device=dev)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        res["rows"][name] = row
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"{name:12s} W1 {row['organ_w1']:.3f}  hetero-gap {row['hetero_gap']:.4f}  "
              f"TSTR {row['tstr_dice']:.3f}+-{row['tstr_sd']:.3f}  edge {row['edge_iou']:.3f}"
              + (f"  locality {row['locality']:.1f}" if "locality" in row else "")
              + (f"  HU-sens {row['hu_sensitivity']:.4f}" if "hu_sensitivity" in row else ""), flush=True)

    if args.physics_rows:
        for key in ("phys_b0", "phys_m1", "phys_m2"):
            if key in res["rows"]:
                continue
            p = d[key].astype(np.float32) / 255.0
            score(key, p, p, {"kind": "physics"})

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        if name in res["rows"]:
            continue
        v = VARIANTS[name.split("_")[0]]
        tr = Translator(path, device=dev)
        phys = d[v["phys"]].astype(np.float32) / 255.0
        labs = d[v["lab"]].astype(np.int64)
        zs = d["z"].astype(np.float32)
        trans = np.concatenate([tr.batch(phys[i:i + 32], labs[i:i + 32], zs[i:i + 32] if v["hu"] else None)
                                for i in range(0, len(phys), 32)])
        extra = {"kind": "variant", "variant": name.split("_")[0], "ckpt": path}
        if v["hu"]:
            extra["hu_sensitivity"] = M.hu_sensitivity(tr, phys, labs, zs)
            # label swap with the HU channel held at each frame's own z
            extra["locality"], _ = _locality_with_z(M, tr, phys, labs, zs, v["n_lab"])
        else:
            extra["hu_sensitivity"] = 0.0
            extra["locality"], _ = M.locality(lambda im, lb: tr(im, lb), phys, labs, v["n_lab"])
        score(name, trans, phys, extra)
    return 0


def _locality_with_z(M, tr, phys, labs, zs, n_lab, n=40):
    from usbg import texture_eval as TE
    sens = TE.summarize_per_class([
        {c: v["locality"] for c, v in TE.label_swap_sensitivity(
            (lambda im, lb, _z=z: tr(im, lb, _z)), im, lb, n_classes=n_lab).items()}
        for im, lb, z in zip(phys[:n], labs[:n], zs[:n])])
    return (float(np.mean(list(sens.values()))) if sens else float("nan")), sens


if __name__ == "__main__":
    raise SystemExit(main())
