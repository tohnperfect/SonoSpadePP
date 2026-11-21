#!/usr/bin/env python3
"""Train M6 (spatial_echo.SpatialEcho: a CNN from the local CT to the log backscatter factor) and its
labels-only control on TRUSTED, with patient-level 5-fold cross-validation, exactly as train_m4.py:
render from the scan-grid slices, differentiable resize and sector warp, frozen B0 generator of each
seed, loss = negative same-plane agreement (detrended r over the kidney interior and kidney + band)
plus small penalties on the CNN output and the offsets. Held-out planes go through the standard
numpy pipeline.

  python -I scripts/train_m6.py --shards data/trusted/pairs_def --mode ct --seeds 0 1 2
  python -I scripts/train_m6.py --shards data/trusted/pairs_def --mode lab --seeds 0 1 2    # control
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
        for j in range(len(d["sessions"])):
            planes.append({"kid": str(d["sessions"][j]), "raw_lab": d["raw_lab"][j], "raw_z": d["raw_z"][j].astype(np.float32),
                           "raw_hu": d["raw_hu"][j].astype(np.float32), "seed": int(d["raw_seed"][j]),
                           "lab128": d["lab_base"][j], "real": d["real"][j] / 255.0,
                           "kidney": d["kidney_us"][j] > 0, "valid": d["valid"][j] > 0})
    return planes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/trusted/pairs_def")
    ap.add_argument("--mode", choices=["ct", "lab"], default="ct")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-net", type=float, default=1e-3)
    ap.add_argument("--lr-a", type=float, default=0.03)
    ap.add_argument("--w-out", type=float, default=1e-3, help="penalty on the mean squared CNN output")
    ap.add_argument("--w-mag", type=float, default=1e-3, help="penalty on the offsets")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/m6def")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    import torch
    from scipy.ndimage import binary_dilation, binary_erosion
    from usbg import render_lotus as R
    from usbg import texture_gan as TG
    from sonospadepp import learned_echo as LE
    from sonospadepp import metrics as M
    from sonospadepp.physics import maps_from_labels, render_from_maps, render_parts, render_from_parts
    from sonospadepp.spatial_echo import SpatialEcho
    from sonospadepp.variants import Translator
    from sonospadepp.warp import warp_frame
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_pairs import resize_img

    dev = R.get_device()
    planes = load_planes(args.shards)
    rad = M._sector_radius((128, 128))
    post, kern = LE.SectorPost(device=dev), LE.gauss_kernel(args.sigma, device=dev)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
    t0 = time.time()
    for p in planes:
        cfg = replace(R.RenderConfig(), seed=p["seed"])
        Z, MU, BS = maps_from_labels(p["raw_lab"], cfg)
        p["parts"] = render_parts(Z, MU, cfg, dev)
        p["bs"], p["lab_t"], p["z_t"], p["hu_t"] = t(BS), t(p["raw_lab"], torch.long), t(p["raw_z"]), t(p["raw_hu"])
        p["cond"] = TG.onehot_labels(p["lab128"][None].astype(np.int64), 13, dev)
        k = p["kidney"] & p["valid"]
        real_s = M.smooth(p["real"], args.sigma)
        p["regions"] = []
        for reg in (binary_erosion(k, iterations=1) & p["valid"], binary_dilation(k, iterations=4) & p["valid"]):
            if reg.sum() < 50:
                continue
            idx, X = LE.region_design(reg, rad)
            p["regions"].append((t(idx, torch.long), t(X), t(np.linalg.inv(X.T @ X)), t(LE.detrended_resid_np(real_s, idx, X))))
    print(f"{len(planes)} planes prepared in {time.time() - t0:.0f}s", flush=True)

    tag = args.tag or args.mode
    os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.runs, "m6"), exist_ok=True)
    for seed in args.seeds:
        tr = Translator(os.path.join(args.runs, f"b0_s{seed}.pt"), device=dev)
        G = tr.G
        for q in G.parameters():
            q.requires_grad_(False)
        phys_out = np.zeros((len(planes), 128, 128), np.uint8)
        gen_out = np.zeros((len(planes), 128, 128), np.float16)
        done = np.zeros(len(planes), bool)
        log = {}
        for fold in args.folds:
            test = [i for i, p in enumerate(planes) if LE.fold_of(p["kid"]) == fold]
            train = [i for i, p in enumerate(planes) if LE.fold_of(p["kid"]) != fold and p["regions"]]
            torch.manual_seed(1000 * seed + fold); rng = np.random.default_rng(1000 * seed + fold)
            model = SpatialEcho(use_ct=args.mode == "ct", device=dev)
            opt = torch.optim.Adam(model.param_groups(args.lr_net, args.lr_a))
            hist, t1 = [], time.time()
            for step in range(args.steps):
                bi = rng.choice(train, size=min(args.batch, len(train)), replace=False)
                lab = torch.stack([planes[i]["lab_t"] for i in bi]); z = torch.stack([planes[i]["z_t"] for i in bi])
                hu = torch.stack([planes[i]["hu_t"] for i in bi])
                lf, cnn = model.log_factor(lab, z, hu)
                imgs = torch.stack([render_from_parts(planes[i]["bs"] * torch.exp(lf[b]), planes[i]["parts"]) for b, i in enumerate(bi)])
                fake = ((G(post(imgs)[:, None] * 2 - 1, torch.cat([planes[i]["cond"] for i in bi]))[:, 0] + 1) / 2).clamp(0, 1)
                sm = LE.smooth_t(fake, kern)
                score = torch.stack([torch.stack([LE.detrended_r_t(sm[b], *rg) for rg in planes[i]["regions"]]).mean()
                                     for b, i in enumerate(bi)]).mean()
                loss = -score + args.w_out * (cnn ** 2).mean() + args.w_mag * (model.a[model.mod] ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
                hist.append(float(score.detach()))
            log[fold] = {"train_r_last50": float(np.mean(hist[-50:])), "minutes": round((time.time() - t1) / 60, 1),
                         "a": model.a.detach().cpu().tolist(), "n_params": model.n_params()}
            torch.save(model.state(), os.path.join(args.runs, "m6", f"{tag}_s{seed}_f{fold}.pt"))
            model.net.eval()
            for i in test:
                p = planes[i]
                cfg = replace(R.RenderConfig(), seed=p["seed"])
                Z, MU, BS = maps_from_labels(p["raw_lab"], cfg)
                img = render_from_maps(Z, MU, BS * model.factor_numpy(p["raw_lab"], p["raw_z"], p["raw_hu"]), cfg, dev)
                phys_out[i] = warp_frame({"p": resize_img(img, 128)}, {"p": "image"})["p"]
                gen_out[i] = tr.batch(phys_out[i][None] / 255.0, p["lab128"][None])[0]
                done[i] = True
            print(f"seed {seed} fold {fold}: train r {log[fold]['train_r_last50']:+.3f}, {log[fold]['minutes']} min", flush=True)
        json.dump(log, open(os.path.join(args.runs, "m6", f"{tag}_s{seed}.json"), "w"))
        np.savez_compressed(os.path.join(args.out, f"preds_{tag}_s{seed}.npz"), phys=phys_out, gen=gen_out, done=done,
                            sessions=np.asarray([p["kid"] for p in planes]))
        print(f"wrote {args.out}/preds_{tag}_s{seed}.npz ({int(done.sum())}/{len(planes)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
