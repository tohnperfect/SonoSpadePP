#!/usr/bin/env python3
"""M9a: the one-image setting by one-shot fine-tuning. For every kidney, the M7 of the fold that held
the patient out (M7c for the control M9ac) is fine-tuned on the single given pair (the central native
plane: its CT inputs and its real US) and then predicts the other six native planes. Loss: the
same-plane agreement on the given plane plus an L1 pull toward the starting model's output (as M8).

  python -I scripts/adapt_m9a.py --mode ct  --seeds 0 1 2
  python -I scripts/adapt_m9a.py --mode lab --seeds 0 1 2      # control
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-shards", default="data/trusted/pairs_def")
    ap.add_argument("--mode", choices=["ct", "lab"], default="ct")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--w-prior", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/m9")
    ap.add_argument("--b0", default="runs/b0_s{seed}.pt", help="base checkpoint per seed (merge: the v2 B0)")
    ap.add_argument("--m7-tag", default=None, help="tag of the approach-1 states in runs/m7 (default: --mode)")
    ap.add_argument("--tag", default=None, help="output tag (default: --mode + '1')")
    args = ap.parse_args(argv)

    import torch
    from usbg import render_lotus as R
    from sonospadepp import learned_echo as LE
    from sonospadepp import metrics as M
    from sonospadepp import paired_translator as PT
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_m7 import load_planes, prepare_regions, agreement, stack
    from train_m9 import native_refs

    dev = R.get_device()
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
    kern = LE.gauss_kernel(args.sigma, device=dev)
    planes = load_planes(args.test_shards)
    refs = native_refs(planes, args.test_shards)
    prepare_regions(planes, M._sector_radius((128, 128)), args.sigma, t)
    use_ct = args.mode == "ct"
    os.makedirs(args.out, exist_ok=True)

    def run(G, ids):
        c, s = PT.inputs(stack(planes, ids, "phys"), stack(planes, ids, "lab"), stack(planes, ids, "z"),
                         stack(planes, ids, "hu"), use_ct, dev)
        return PT.forward01(G, c, s)

    for seed in args.seeds:
        gen_out = np.zeros((len(planes), 128, 128), np.float16); done = np.zeros(len(planes), bool)
        log, t0, states = [], time.time(), {}
        for kid, (r, tg) in sorted(refs.items()):
            fold = LE.fold_of(kid)
            if fold not in states:
                states[fold] = torch.load(os.path.join(args.runs, "m7", f"{args.m7_tag or args.mode}_s{seed}_f{fold}.pt"),
                                          map_location=dev)
            G = PT.build(args.b0.format(seed=seed), dev); G.load_state_dict(states[fold])
            if planes[r]["regions"]:
                with torch.no_grad():
                    G.eval(); prior = run(G, [r])
                G.train()
                opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
                vm = t(stack(planes, [r], "valid").astype(np.float32))
                s0 = None
                for _ in range(args.steps):
                    fake = run(G, [r])
                    score = agreement(LE.smooth_t(fake, kern), planes, [r]).mean()
                    loss = -score + args.w_prior * ((fake - prior).abs() * vm).sum() / vm.sum().clamp(min=1)
                    opt.zero_grad(); loss.backward(); opt.step()
                    if s0 is None:
                        s0 = float(score.detach())
                log.append({"kid": kid, "fit_r_start": s0, "fit_r_end": float(score.detach())})
            G.eval()
            with torch.no_grad():
                gen_out[tg] = run(G, tg).cpu().numpy().astype(np.float16)
            done[tg] = True
        tag = args.tag or f"{args.mode}1"
        json.dump(log, open(os.path.join(args.out, f"fits1_{args.mode}_s{seed}.json" if args.tag is None
                                         else f"fits_{tag}_s{seed}.json"), "w"))
        phys = np.stack([np.round(p["phys"] * 255) for p in planes]).astype(np.uint8)
        np.savez_compressed(os.path.join(args.out, f"preds_{tag}_s{seed}.npz"), phys=phys, gen=gen_out, done=done,
                            sessions=np.asarray([p["kid"] for p in planes]))
        print(f"seed {seed}: {len(log)} one-shot fits in {(time.time() - t0) / 60:.1f} min, mean fit gain "
              f"{np.mean([f['fit_r_end'] - f['fit_r_start'] for f in log]):+.3f}, {int(done.sum())}/{len(planes)} predicted",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
