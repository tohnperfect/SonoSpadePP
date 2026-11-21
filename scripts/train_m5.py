#!/usr/bin/env python3
"""M5: a patient-specific HU-to-echogenicity mapping, fitted to each patient's own ultrasound.

M4 learns one mapping for all patients (per-tissue curves over relative HU) and applies it to a new
patient. M5 starts from that population mapping, for the fold that held the patient out, and fits
the curves again on planes of THIS patient, then renders the patient's other planes with them. The
fitted object is a set of per-tissue curves (no spatial parameters), so it can carry what is
specific to the patient (contrast phase, scanner gain per tissue, the patient's own relation between
CT intensity and echogenicity) but cannot copy image structure from one plane to another.

Splits (both give every test plane a mapping fitted without it):
  halves   each kidney's seven planes are ordered by sweep angle; fit on the first four and test on
           the last three, and the other way round (all 59 kidneys)
  kidney   fit on all planes of the patient's other kidney, test on this kidney (patients with both)
--mode const fits only the per-tissue offsets (no HU), from the population HU-free control: what a
patient-specific brightness calibration gives without the CT intensity.

Loss: the same negative same-plane agreement as M4, plus a pull toward the population mapping
(w_prior * squared distance) and the smoothness penalty on the curves.

  python -I scripts/train_m5.py --mode hu --split halves --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/trusted/pairs")
    ap.add_argument("--mode", choices=["hu", "const"], default="hu")
    ap.add_argument("--split", choices=["halves", "kidney"], default="halves")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--w-prior", type=float, default=1e-2, help="pull toward the population mapping")
    ap.add_argument("--w-smooth", type=float, default=1e-2, help="penalty on second differences of g")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/m5")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N fit/test tasks")
    ap.add_argument("--pop-suffix", default="", help="population M4 runs to start from: runs/m4/<hu|const><suffix>_s<seed>.json "
                    "(e.g. 'def' for the deformable registration)")
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
    from train_m4 import load_planes

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
        p["bs"], p["lab_t"], p["z_t"] = t(BS), t(p["raw_lab"], torch.long), t(p["raw_z"])
        p["cond"] = TG.onehot_labels(p["lab128"][None].astype(np.int64), 13, dev)
        k = p["kidney"] & p["valid"]
        p["regions"] = []
        for reg in (binary_erosion(k, iterations=1) & p["valid"], binary_dilation(k, iterations=4) & p["valid"]):
            if reg.sum() < 50:
                continue
            idx, X = LE.region_design(reg, rad)
            p["regions"].append((t(idx, torch.long), t(X), t(np.linalg.inv(X.T @ X)),
                                 t(LE.detrended_resid_np(M.smooth(p["real"], args.sigma), idx, X))))
    print(f"{len(planes)} planes prepared in {time.time() - t0:.0f}s", flush=True)

    by_kid = {}
    for i, p in enumerate(planes):
        by_kid.setdefault(p["kid"], []).append(i)          # shard order = ascending sweep angle
    tasks = []                                             # (fit plane ids, test plane ids)
    for kid, ids in by_kid.items():
        if args.split == "halves":
            h = (len(ids) + 1) // 2
            tasks += [(ids[:h], ids[h:]), (ids[h:], ids[:h])]
        else:
            other = kid[:-1] + ("L" if kid.endswith("R") else "R")
            if other in by_kid:
                tasks.append((by_kid[other], ids))

    if args.limit:
        tasks = tasks[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    pop_tag = ("hu" if args.mode == "hu" else "const") + args.pop_suffix
    for seed in args.seeds:
        tr = Translator(os.path.join(args.runs, f"b0_s{seed}.pt"), device=dev)
        G = tr.G
        for q in G.parameters():
            q.requires_grad_(False)
        pop = json.load(open(os.path.join(args.runs, "m4", f"{pop_tag}_s{seed}.json")))
        phys_out = np.zeros((len(planes), 128, 128), np.uint8)
        gen_out = np.zeros((len(planes), 128, 128), np.float16)
        done = np.zeros(len(planes), bool)
        fits, t1 = [], time.time()
        for fit_ids, test_ids in tasks:
            fit_ids = [i for i in fit_ids if planes[i]["regions"]]
            if not fit_ids:
                continue
            st = pop[str(LE.fold_of(planes[test_ids[0]]["kid"]))]   # the fold that held this patient out
            model = LE.LearnedEcho.from_state(st, device=dev)
            a0, g0 = model.a.detach().clone(), model.g.detach().clone()
            opt = torch.optim.Adam(model.params(), lr=args.lr)
            m = model.mod
            for step in range(args.steps):
                imgs = torch.stack([render_from_parts(planes[i]["bs"] * torch.exp(model.log_factor(planes[i]["lab_t"], planes[i]["z_t"])),
                                                      planes[i]["parts"]) for i in fit_ids])
                cond = torch.cat([planes[i]["cond"] for i in fit_ids])
                fake = ((G(post(imgs)[:, None] * 2 - 1, cond)[:, 0] + 1) / 2).clamp(0, 1)
                sm = LE.smooth_t(fake, kern)
                score = torch.stack([torch.stack([LE.detrended_r_t(sm[b], *reg) for reg in planes[i]["regions"]]).mean()
                                     for b, i in enumerate(fit_ids)]).mean()
                reg = args.w_prior * ((model.a - a0)[m] ** 2).sum()
                if model.hu:
                    g = model.g[m]
                    reg = reg + args.w_prior * ((model.g - g0)[m] ** 2).sum() \
                        + args.w_smooth * ((g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]) ** 2).sum()
                loss = -score + reg
                opt.zero_grad(); loss.backward(); opt.step()
                if step == 0:
                    s0 = float(score.detach())
            fits.append({"kid": planes[test_ids[0]]["kid"], "fit_r_start": s0, "fit_r_end": float(score.detach()),
                         "state": model.state()})
            for i in test_ids:
                p = planes[i]
                cfg = replace(R.RenderConfig(), seed=p["seed"])
                Z, MU, BS = maps_from_labels(p["raw_lab"], cfg)
                img = render_from_maps(Z, MU, BS * model.factor_numpy(p["raw_lab"], p["raw_z"]), cfg, dev)
                phys_out[i] = warp_frame({"p": resize_img(img, 128)}, {"p": "image"})["p"]
                gen_out[i] = tr.batch(phys_out[i][None] / 255.0, p["lab128"][None])[0]
                done[i] = True
        tag = f"{args.mode}_{args.split}_s{seed}"
        json.dump(fits, open(os.path.join(args.out, f"fits_{tag}.json"), "w"))
        np.savez_compressed(os.path.join(args.out, f"preds_{tag}.npz"), phys=phys_out, gen=gen_out, done=done,
                            sessions=np.asarray([p["kid"] for p in planes]))
        gain = np.mean([f["fit_r_end"] - f["fit_r_start"] for f in fits])
        print(f"seed {seed}: {len(fits)} patient fits in {(time.time() - t1) / 60:.1f} min, mean fit gain {gain:+.3f}, "
              f"{int(done.sum())}/{len(planes)} planes predicted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
