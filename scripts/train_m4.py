#!/usr/bin/env python3
"""Train M4 (learned HU-to-echogenicity front end, learned_echo.py) on TRUSTED with patient-level
5-fold cross-validation, and predict every held-out plane.

For each GAN seed s and fold k: fit (a, g) on the planes of the other four folds, through the
differentiable render, resize, sector warp and the frozen B0 generator of seed s, maximizing the
same-plane agreement (depth-detrended correlation over the kidney interior and the kidney+band
region, as in metrics.paired_scores). Then render fold k's planes with the learned factors through
the standard numpy pipeline (the same one every other row uses) and translate them with that
generator. --mode const learns the per-tissue offsets only (no HU dependence), the control.

  python -I scripts/train_m4.py --mode hu --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np


def load_planes(shard_dir):
    planes = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "*.npz"))):
        d = np.load(f, allow_pickle=False)
        if "raw_lab" not in d.files:
            raise SystemExit(f"{f} has no raw slices; rerun scripts/trusted_prepare.py")
        for j in range(len(d["sessions"])):
            planes.append({"kid": str(d["sessions"][j]), "j": j, "raw_lab": d["raw_lab"][j], "raw_z": d["raw_z"][j].astype(np.float32),
                           "seed": int(d["raw_seed"][j]), "lab128": d["lab_base"][j], "real": d["real"][j] / 255.0,
                           "kidney": d["kidney_us"][j] > 0, "valid": d["valid"][j] > 0})
    return planes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/trusted/pairs")
    ap.add_argument("--mode", choices=["hu", "const"], default="hu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--w-smooth", type=float, default=1e-2, help="penalty on second differences of g")
    ap.add_argument("--w-mag", type=float, default=1e-3, help="penalty on the magnitudes of a and g")
    ap.add_argument("--init", choices=["zero", "m2"], default="zero",
                    help="m2: start g at M2's hand-set backscatter rule, -0.35 min(|z|, 4)")
    ap.add_argument("--tag", default=None, help="output name (default: --mode)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/m4")
    args = ap.parse_args(argv)

    import torch
    from scipy.ndimage import binary_dilation, binary_erosion
    from usbg import render_lotus as R
    from usbg import texture_gan as TG
    from sonospadepp import learned_echo as LE
    from sonospadepp import metrics as M
    from sonospadepp.physics import maps_from_labels, render_from_maps, render_parts, render_from_parts
    from sonospadepp.variants import Translator
    from sonospadepp.warp import warp_frame
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_pairs import resize_img

    dev = R.get_device()
    planes = load_planes(args.shards)
    rad = M._sector_radius((128, 128))
    post, kern = LE.SectorPost(device=dev), LE.gauss_kernel(args.sigma, device=dev)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)

    # fixed per-plane tensors: render parts, base backscatter, regions with the real residuals
    t0 = time.time()
    for p in planes:
        cfg = replace(R.RenderConfig(), seed=p["seed"])
        Z, MU, BS = maps_from_labels(p["raw_lab"], cfg)
        p["parts"] = render_parts(Z, MU, cfg, dev)
        p["bs"], p["lab_t"], p["z_t"] = t(BS), t(p["raw_lab"], torch.long), t(p["raw_z"])
        p["cond"] = TG.onehot_labels(p["lab128"][None].astype(np.int64), 13, dev)
        k = p["kidney"] & p["valid"]
        regions = [binary_erosion(k, iterations=1) & p["valid"], binary_dilation(k, iterations=4) & p["valid"]]
        real_s = M.smooth(p["real"], args.sigma)
        p["regions"] = []
        for reg in regions:
            if reg.sum() < 50:
                continue
            idx, X = LE.region_design(reg, rad)
            p["regions"].append((t(idx, torch.long), t(X), t(np.linalg.inv(X.T @ X)), t(LE.detrended_resid_np(real_s, idx, X))))
    print(f"{len(planes)} planes from {len({p['kid'] for p in planes})} kidneys prepared in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.runs, "m4"), exist_ok=True)
    for seed in args.seeds:
        tr = Translator(os.path.join(args.runs, f"b0_s{seed}.pt"), device=dev)
        G = tr.G
        for q in G.parameters():
            q.requires_grad_(False)
        phys_out = np.zeros((len(planes), 128, 128), np.uint8)
        gen_out = np.zeros((len(planes), 128, 128), np.float16)
        done = np.zeros(len(planes), bool)
        states = {}
        for fold in args.folds:
            test = [i for i, p in enumerate(planes) if LE.fold_of(p["kid"]) == fold]
            train = [i for i, p in enumerate(planes) if LE.fold_of(p["kid"]) != fold and p["regions"]]
            if not test or not train:
                continue
            torch.manual_seed(1000 * seed + fold)
            rng = np.random.default_rng(1000 * seed + fold)
            model = LE.LearnedEcho(hu_dependent=args.mode == "hu", device=dev)
            if args.init == "m2":
                model.init_m2_rule()
            opt = torch.optim.Adam(model.params(), lr=args.lr)
            t0, hist = time.time(), []
            for step in range(args.steps):
                bi = rng.choice(train, size=min(args.batch, len(train)), replace=False)
                imgs = torch.stack([render_from_parts(planes[i]["bs"] * torch.exp(model.log_factor(planes[i]["lab_t"], planes[i]["z_t"])),
                                                      planes[i]["parts"]) for i in bi])
                phys = post(imgs)
                cond = torch.cat([planes[i]["cond"] for i in bi])
                fake = ((G(phys[:, None] * 2 - 1, cond)[:, 0] + 1) / 2).clamp(0, 1)
                sm = LE.smooth_t(fake, kern)
                rs = [torch.stack([LE.detrended_r_t(sm[b], *reg) for reg in planes[i]["regions"]]).mean()
                      for b, i in enumerate(bi)]
                score = torch.stack(rs).mean()
                loss = -score + model.regularizer(args.w_smooth, args.w_mag)
                opt.zero_grad(); loss.backward(); opt.step()
                hist.append(float(score))
                if (step + 1) % max(1, args.steps // 5) == 0:
                    print(f"  seed {seed} fold {fold} step {step + 1}: train r {np.mean(hist[-20:]):+.3f}", flush=True)
            states[fold] = model.state()
            states[fold].update({"w_smooth": args.w_smooth, "w_mag": args.w_mag, "init": args.init})
            states[fold]["train_r_last20"] = float(np.mean(hist[-20:])); states[fold]["minutes"] = round((time.time() - t0) / 60, 1)
            # held-out planes through the standard pipeline
            for i in test:
                p = planes[i]
                cfg = replace(R.RenderConfig(), seed=p["seed"])
                Z, MU, BS = maps_from_labels(p["raw_lab"], cfg)
                img = render_from_maps(Z, MU, BS * model.factor_numpy(p["raw_lab"], p["raw_z"]), cfg, dev)
                phys_out[i] = warp_frame({"p": resize_img(img, 128)}, {"p": "image"})["p"]
                gen_out[i] = tr.batch(phys_out[i][None] / 255.0, p["lab128"][None])[0]
                done[i] = True
            print(f"seed {seed} fold {fold}: {len(train)} train planes, {len(test)} test planes, "
                  f"{states[fold]['minutes']} min", flush=True)
        tag = f"{args.tag or args.mode}_s{seed}"
        with open(os.path.join(args.runs, "m4", f"{tag}.json"), "w") as fh:
            json.dump(states, fh)
        np.savez_compressed(os.path.join(args.out, f"preds_{tag}.npz"), phys=phys_out, gen=gen_out, done=done,
                            sessions=np.asarray([p["kid"] for p in planes]))
        print(f"wrote {args.out}/preds_{tag}.npz ({int(done.sum())}/{len(planes)} planes predicted)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
