#!/usr/bin/env python3
"""M4 and M5 at full resolution: the learned HU-to-echogenicity mapping (learned_echo.LearnedEcho)
fitted on the 0.3 mm planes, render only (the translators exist at 128 px only).

The loss is the same-plane agreement of hires.scores: the render on the 320 x 512 scan grid is made
differentiable in backscatter (physics.render_from_parts), scan-converted to the 0.3 mm grid
(hires.ScanConvertT), smoothed (sigma 1 mm) and correlated, after removing a quadratic in depth,
with the real US over the kidney interior and the kidney + band.

  --scheme pop     M4: patient-level 5-fold cross-validation (TRUSTED folds); predicts every plane
  --scheme halves  M5: per patient, from the pop model of the fold that held the patient out,
                   fitted on half of each kidney's sweep and tested on the other half
  --scheme kidney  M5: fitted on the patient's other kidney
  --mode const     the HU-free control (per-tissue offsets only)
  --reg rig|def    rigid or deformable registration

  python -I scripts/train_hr.py --scheme pop --mode hu --reg def
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
    ap.add_argument("--hires", default="data/trusted/hires")
    ap.add_argument("--scheme", choices=["pop", "halves", "kidney"], default="pop")
    ap.add_argument("--mode", choices=["hu", "const"], default="hu")
    ap.add_argument("--reg", choices=["rig", "def"], default="def")
    ap.add_argument("--sigma-mm", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=None, help="default 300 (pop) or 100 (per patient)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--w-smooth", type=float, default=1e-2)
    ap.add_argument("--w-mag", type=float, default=1e-3)
    ap.add_argument("--w-prior", type=float, default=1e-2)
    ap.add_argument("--runs", default="runs/hr")
    ap.add_argument("--out", default="outputs/hires")
    args = ap.parse_args(argv)
    steps = args.steps or (300 if args.scheme == "pop" else 100)

    import torch
    from usbg import render_lotus as R
    from sonospadepp import hires as H, learned_echo as LE
    from sonospadepp.physics import maps_from_labels, render_from_maps, render_parts, render_from_parts
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_hires import load_hires

    dev = R.get_device()
    parts = load_hires(args.hires)
    _, fan, rad, th = H.cart_grid()
    L, S = parts[0][f"pol_lab_{args.reg}"].shape[1:]
    s_idx, l_idx = H.polar_index(rad, th, L, S)
    conv = H.ScanConvertT(s_idx, l_idx, fan, S, L, device=dev)
    kern = LE.gauss_kernel(args.sigma_mm / H.MM, device=dev)
    lat = R.RenderConfig().psf_sigma_lat0 * L / 160.0
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
    from scipy.ndimage import gaussian_filter

    planes, t0 = [], time.time()
    for p in parts:
        for j in range(len(p["sessions"])):
            lab = p[f"pol_lab_{args.reg}"][j].astype(np.int64); z = p[f"pol_z_{args.reg}"][j].astype(np.float32)
            cfg = replace(R.RenderConfig(), seed=int(p["seeds"][j]), psf_sigma_lat0=lat)
            Z, MU, BS = maps_from_labels(lab, cfg)
            q = {"kid": str(p["sessions"][j]), "lab": lab, "z": z, "cfg": cfg,
                 "parts": render_parts(Z, MU, cfg, dev), "bs": t(BS), "lab_t": t(lab, torch.long), "z_t": t(z)}
            real = p["real"][j].astype(np.float32) / 255.0
            rs = gaussian_filter(real, args.sigma_mm / H.MM)
            q["regions"] = []
            for reg in H.regions(p["kidney_us"][j], p["valid"][j]):
                if reg.sum() < 200:
                    continue
                idx, X = LE.region_design(reg, rad)
                q["regions"].append((t(idx, torch.long), t(X), t(np.linalg.inv(X.T @ X)), t(LE.detrended_resid_np(rs, idx, X))))
            planes.append(q)
    print(f"{len(planes)} planes ({args.reg}) prepared in {time.time() - t0:.0f}s", flush=True)

    def agreement(model, ids):
        imgs = torch.stack([render_from_parts(planes[i]["bs"] * torch.exp(model.log_factor(planes[i]["lab_t"], planes[i]["z_t"])),
                                              planes[i]["parts"]) for i in ids])
        sm = LE.smooth_t(conv(imgs), kern)
        return torch.stack([torch.stack([LE.detrended_r_t(sm[b], *rg) for rg in planes[i]["regions"]]).mean()
                            for b, i in enumerate(ids)]).mean()

    def predict(model, i):
        q = planes[i]
        Z, MU, BS = maps_from_labels(q["lab"], q["cfg"])
        img = render_from_maps(Z, MU, BS * model.factor_numpy(q["lab"], q["z"]), q["cfg"], dev)
        return np.clip(np.round(H.scan_convert(img, s_idx, l_idx, fan) * 255), 0, 255).astype(np.uint8)

    os.makedirs(args.runs, exist_ok=True); os.makedirs(args.out, exist_ok=True)
    phys = np.zeros((len(planes),) + fan.shape, np.uint8)
    done = np.zeros(len(planes), bool)
    tag = f"{args.scheme}_{args.mode}_{args.reg}"
    log = []
    if args.scheme == "pop":
        states = {}
        for fold in range(5):
            test = [i for i, q in enumerate(planes) if LE.fold_of(q["kid"]) == fold]
            train = [i for i, q in enumerate(planes) if LE.fold_of(q["kid"]) != fold and q["regions"]]
            torch.manual_seed(fold); rng = np.random.default_rng(fold)
            model = LE.LearnedEcho(hu_dependent=args.mode == "hu", device=dev)
            opt = torch.optim.Adam(model.params(), lr=args.lr)
            hist, t1 = [], time.time()
            for step in range(steps):
                bi = rng.choice(train, size=min(args.batch, len(train)), replace=False)
                score = agreement(model, bi)
                loss = -score + model.regularizer(args.w_smooth, args.w_mag)
                opt.zero_grad(); loss.backward(); opt.step()
                hist.append(float(score.detach()))
            states[fold] = model.state(); states[fold]["train_r_last20"] = float(np.mean(hist[-20:]))
            for i in test:
                phys[i] = predict(model, i); done[i] = True
            print(f"fold {fold}: train r {np.mean(hist[-20:]):+.3f}, {(time.time() - t1) / 60:.1f} min", flush=True)
        json.dump(states, open(os.path.join(args.runs, f"{tag}.json"), "w"))
    else:
        pop = json.load(open(os.path.join(args.runs, f"pop_{args.mode}_{args.reg}.json")))
        by_kid = {}
        for i, q in enumerate(planes):
            by_kid.setdefault(q["kid"], []).append(i)
        tasks = []
        for kid, ids in by_kid.items():
            if args.scheme == "halves":
                h = (len(ids) + 1) // 2
                tasks += [(ids[:h], ids[h:]), (ids[h:], ids[:h])]
            else:
                other = kid[:-1] + ("L" if kid.endswith("R") else "R")
                if other in by_kid:
                    tasks.append((by_kid[other], ids))
        t1 = time.time()
        for fit_ids, test_ids in tasks:
            fit_ids = [i for i in fit_ids if planes[i]["regions"]]
            if not fit_ids:
                continue
            model = LE.LearnedEcho.from_state(pop[str(LE.fold_of(planes[test_ids[0]]["kid"]))], device=dev)
            a0, g0 = model.a.detach().clone(), model.g.detach().clone()
            opt = torch.optim.Adam(model.params(), lr=args.lr)
            m = model.mod
            for step in range(steps):
                score = agreement(model, fit_ids)
                reg = args.w_prior * ((model.a - a0)[m] ** 2).sum()
                if model.hu:
                    g = model.g[m]
                    reg = reg + args.w_prior * ((model.g - g0)[m] ** 2).sum() \
                        + args.w_smooth * ((g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]) ** 2).sum()
                loss = -score + reg
                opt.zero_grad(); loss.backward(); opt.step()
                if step == 0:
                    s0 = float(score.detach())
            log.append({"kid": planes[test_ids[0]]["kid"], "fit_r_start": s0, "fit_r_end": float(score.detach())})
            for i in test_ids:
                phys[i] = predict(model, i); done[i] = True
        json.dump(log, open(os.path.join(args.runs, f"fits_{tag}.json"), "w"))
        print(f"{len(log)} patient fits in {(time.time() - t1) / 60:.1f} min, mean fit gain "
              f"{np.mean([f['fit_r_end'] - f['fit_r_start'] for f in log]):+.3f}", flush=True)
    np.savez_compressed(os.path.join(args.out, f"preds_{tag}.npz"), phys=phys, done=done,
                        sessions=np.asarray([q["kid"] for q in planes]))
    print(f"wrote {args.out}/preds_{tag}.npz ({int(done.sum())}/{len(planes)} planes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
