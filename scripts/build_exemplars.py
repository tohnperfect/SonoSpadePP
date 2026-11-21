#!/usr/bin/env python3
"""Retrieved exemplars for the TRUSTED training planes (hybrid H-RX: tier 1 with the retrieved exemplar
as input). For every dense plane and every test fold f other than the plane's own, the most similar label map
(ref_retrieval.similarity) among the dense planes of the folds other than f and of other patients is warped onto the
plane with the label-warp optimizer (ref_retrieval's settings). A (plane, source) pair is warped once and shared by
the folds that pick it. Output outputs/m7/exemplars_train.npz: ex (M,128,128) float16 and idx (5, N) into ex
(-1 for the plane's own fold). TRUSTED-derived: stays outside git.

  python -I scripts/build_exemplars.py      # inside the SonoSpadePP checkout, CPU
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-shards", default="data/trusted/pairs_dense")
    ap.add_argument("--out", default="outputs/m7/exemplars_train.npz")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--w-smooth", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from sonospadepp import learned_echo as LE
    from sonospadepp import paired_translator as PT
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ref_retrieval import similarity
    from train_m7 import load_planes

    dev = args.device
    pool = load_planes(args.pool_shards)
    n = len(pool)
    lab32 = np.stack([p["lab"][::4, ::4] for p in pool])
    fold = np.asarray([LE.fold_of(p["kid"]) for p in pool])
    patient = np.asarray([p["kid"][:-1] for p in pool])
    kcs = [int(np.bincount(p["lab"][p["kidney"]], minlength=PT.N_LAB).argmax()) if p["kidney"].any() else 0 for p in pool]
    t0 = time.time()
    # ranked candidates per plane (other patients), then the first candidate outside each test fold
    src = np.full((5, n), -1, np.int64)
    for i in range(n):
        cand = np.flatnonzero(patient != patient[i])
        order = cand[np.argsort(-similarity(lab32[i], lab32[cand], kcs[i], PT.N_LAB), kind="stable")]
        for f in range(5):
            if f == fold[i]:
                continue
            src[f, i] = order[np.argmax(fold[order] != f)]
    pairs = sorted({(i, int(src[f, i])) for f in range(5) for i in range(n) if src[f, i] >= 0})
    print(f"{n} planes, {len(pairs)} distinct (plane, source) pairs, ranked in {time.time() - t0:.0f}s", flush=True)

    kern = LE.gauss_kernel(2.0, device=dev)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 128, device=dev), torch.linspace(-1, 1, 128, device=dev), indexing="ij")
    base = torch.stack([xx, yy], -1)[None]

    def onehot(lab):
        x = F.one_hot(torch.as_tensor(lab, device=dev).long().clamp(0, PT.N_LAB - 1), PT.N_LAB).permute(0, 3, 1, 2).float()
        return torch.stack([LE.smooth_t(x[:, c], kern) for c in range(PT.N_LAB)], 1)

    ex = np.zeros((len(pairs), 128, 128), np.float16)
    pos = {}
    for c0 in range(0, len(pairs), args.chunk):
        ch = pairs[c0:c0 + args.chunk]
        B = len(ch)
        s_oh = onehot(np.stack([pool[j]["lab"] for _, j in ch])); d_oh = onehot(np.stack([pool[i]["lab"] for i, _ in ch]))
        present = ((s_oh.sum((2, 3)) > 20) | (d_oh.sum((2, 3)) > 20)).float()
        present[:, 0] = 0
        wcls = present.clone()
        for b, (i, _) in enumerate(ch):
            wcls[b, kcs[i]] *= 3
        prm = torch.zeros(B, 2, args.grid, args.grid, device=dev, requires_grad=True)
        opt = torch.optim.Adam([prm], lr=args.lr)
        for _ in range(args.steps):
            d = F.interpolate(prm, size=(128, 128), mode="bilinear", align_corners=True) * (2 / 127.0)
            g = base + d.permute(0, 2, 3, 1)
            w = F.grid_sample(s_oh, g, mode="bilinear", padding_mode="border", align_corners=True)
            inter = (w * d_oh).sum((2, 3)); den = w.sum((2, 3)) + d_oh.sum((2, 3)) + 1e-3
            dice = ((2 * inter / den) * wcls).sum(1) / wcls.sum(1).clamp(min=1e-6)
            sm = ((prm[:, :, 1:] - prm[:, :, :-1]) ** 2).mean() + ((prm[:, :, :, 1:] - prm[:, :, :, :-1]) ** 2).mean()
            loss = -dice.sum() + args.w_smooth * sm * B
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            d = F.interpolate(prm, size=(128, 128), mode="bilinear", align_corners=True) * (2 / 127.0)
            g = base + d.permute(0, 2, 3, 1)
            img = torch.as_tensor(np.stack([pool[j]["real"].astype(np.float32) for _, j in ch]), device=dev)[:, None]
            ex[c0:c0 + B] = F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0].cpu().numpy()
        for b, p in enumerate(ch):
            pos[p] = c0 + b
        if (c0 // args.chunk) % 10 == 0:
            print(f"  {c0 + B}/{len(pairs)} warped, {time.time() - t0:.0f}s", flush=True)
    idx = np.full((5, n), -1, np.int64)
    for f in range(5):
        for i in range(n):
            if src[f, i] >= 0:
                idx[f, i] = pos[(i, int(src[f, i]))]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, ex=ex, idx=idx, src=src, sessions=np.asarray([p["kid"] for p in pool]))
    print(f"wrote {args.out} ({len(pairs)} exemplars, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
