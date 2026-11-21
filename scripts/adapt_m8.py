#!/usr/bin/env python3
"""M8: M7 fine-tuned to each patient on that patient's own ultrasound (patient-specific learning, as
the per-patient fine-tuning of Chen et al., Med Phys 2025), and its control M8c (from M7c, no CT).

For every kidney, its seven native test planes are ordered by sweep angle and halved. Each half is
predicted by a copy of the M7 of the fold that held the patient out, fine-tuned on the kidney's dense
planes (trusted_dense_prepare.py) that lie on the other side of the sweep, at least --gap degrees from
every plane of the half being predicted (so no test plane or its neighbours are seen). Loss: the
same-plane agreement plus an L1 pull toward the starting model's output (the prior, as in M5).

  python -I scripts/adapt_m8.py --mode ct  --seeds 0 1 2
  python -I scripts/adapt_m8.py --mode lab --seeds 0 1 2      # control
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="data/trusted/pairs_dense")
    ap.add_argument("--test-shards", default="data/trusted/pairs_def")
    ap.add_argument("--mode", choices=["ct", "lab"], default="ct")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--w-prior", type=float, default=0.05)
    ap.add_argument("--gap", type=float, default=3.0, help="minimum angle (deg) between fit planes and predicted planes")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/m8")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N kidneys")
    args = ap.parse_args(argv)

    import torch
    from usbg import render_lotus as R
    from sonospadepp import learned_echo as LE
    from sonospadepp import metrics as M
    from sonospadepp import paired_translator as PT
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_m7 import load_planes, prepare_regions, agreement, stack

    dev = R.get_device()
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
    rad = M._sector_radius((128, 128))
    kern = LE.gauss_kernel(args.sigma, device=dev)
    test_all = load_planes(args.test_shards)
    phis = {}
    for f in sorted(glob.glob(os.path.join(args.test_shards, "*.npz"))):
        for p in json.loads(str(np.load(f, allow_pickle=False)["meta"]))["planes"]:
            phis.setdefault(p["kid"], []).append(p["phi_deg"])
    cnt = {}
    for p in test_all:
        j = cnt.get(p["kid"], 0); p["phi"] = float(phis[p["kid"]][j]); cnt[p["kid"]] = j + 1
    dense_all = load_planes(args.dense)
    prepare_regions(test_all, rad, args.sigma, t); prepare_regions(dense_all, rad, args.sigma, t)
    use_ct = args.mode == "ct"
    os.makedirs(args.out, exist_ok=True)

    def run(G, planes, ids):
        c, s = PT.inputs(stack(planes, ids, "phys"), stack(planes, ids, "lab"), stack(planes, ids, "z"),
                         stack(planes, ids, "hu"), use_ct, dev)
        return PT.forward01(G, c, s)

    by_kid = {}
    for i, p in enumerate(test_all):
        by_kid.setdefault(p["kid"], []).append(i)
    dense_by_kid = {}
    for i, p in enumerate(dense_all):
        if p["regions"]:
            dense_by_kid.setdefault(p["kid"], []).append(i)
    kids = sorted(by_kid)[:args.limit] if args.limit else sorted(by_kid)
    for seed in args.seeds:
        gen_out = np.zeros((len(test_all), 128, 128), np.float16)
        done = np.zeros(len(test_all), bool)
        log, t0 = [], time.time()
        states = {}
        for kid in kids:
            fold = LE.fold_of(kid)
            if fold not in states:
                states[fold] = torch.load(os.path.join(args.runs, "m7", f"{args.mode}_s{seed}_f{fold}.pt"), map_location=dev)
            ids = sorted(by_kid[kid], key=lambda i: test_all[i]["phi"])
            h = (len(ids) + 1) // 2
            for half in (ids[:h], ids[h:]):
                if not half:
                    continue
                lo, hi = min(test_all[i]["phi"] for i in half), max(test_all[i]["phi"] for i in half)
                first = half[0] == ids[0]
                fit = [i for i in dense_by_kid.get(kid, [])
                       if (dense_all[i]["phi"] > hi + args.gap if first else dense_all[i]["phi"] < lo - args.gap)]
                G = PT.build(os.path.join(args.runs, f"b0_s{seed}.pt"), dev); G.load_state_dict(states[fold])
                if fit:
                    G0 = PT.build(os.path.join(args.runs, f"b0_s{seed}.pt"), dev); G0.load_state_dict(states[fold]); G0.eval()
                    for q in G0.parameters():
                        q.requires_grad_(False)
                    G.train()
                    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
                    rng = np.random.default_rng(seed * 100000 + int(kid[:-1]) * 10 + (0 if first else 1))
                    s0 = None
                    for step in range(args.steps):
                        bi = rng.choice(fit, size=min(args.batch, len(fit)), replace=False)
                        fake = run(G, dense_all, bi)
                        with torch.no_grad():
                            ref = run(G0, dense_all, bi)
                        score = agreement(LE.smooth_t(fake, kern), dense_all, bi).mean()
                        vm = t(stack(dense_all, bi, "valid").astype(np.float32))
                        loss = -score + args.w_prior * ((fake - ref).abs() * vm).sum() / vm.sum().clamp(min=1)
                        opt.zero_grad(); loss.backward(); opt.step()
                        if s0 is None:
                            s0 = float(score.detach())
                    log.append({"kid": kid, "half": 0 if first else 1, "n_fit": len(fit), "fit_r_start": s0,
                                "fit_r_end": float(score.detach())})
                G.eval()
                with torch.no_grad():
                    gen_out[half] = run(G, test_all, half).cpu().numpy().astype(np.float16)
                done[half] = True
        json.dump(log, open(os.path.join(args.out, f"fits_{args.mode}_s{seed}.json"), "w"))
        phys = np.stack([np.round(p["phys"] * 255) for p in test_all]).astype(np.uint8)
        np.savez_compressed(os.path.join(args.out, f"preds_{args.mode}_s{seed}.npz"), phys=phys, gen=gen_out, done=done,
                            sessions=np.asarray([p["kid"] for p in test_all]))
        gains = [f["fit_r_end"] - f["fit_r_start"] for f in log]
        print(f"seed {seed}: {len(log)} fits in {(time.time() - t0) / 60:.1f} min, mean fit gain {np.mean(gains):+.3f}, "
              f"median fit planes {int(np.median([f['n_fit'] for f in log]))}, {int(done.sum())}/{len(test_all)} predicted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
