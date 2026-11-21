#!/usr/bin/env python3
"""Retrieved-exemplar warp (H-R): for every TRUSTED test plane, the real
image of another patient whose CT label map is the most similar (top 1 among the dense planes of the patients in the
other folds, tier 1's training planes) is warped onto the test plane through the label maps with the label-warp
reference's optimizer (ref_baselines.py). Non-learned, deterministic. Output outputs/m9/preds_retrieve.npz.

  python -I scripts/ref_retrieval.py      # inside the SonoSpadePP checkout
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def similarity(t32, pool32, kc, n_lab):
    """Class-weighted Dice of one 32x32 label map against a pool (N,32,32): classes present in either map,
    background excluded, the kidney class kc weighted 3."""
    t1 = np.stack([t32 == c for c in range(n_lab)], 0).reshape(n_lab, -1).astype(np.float32)        # (C, P)
    p1 = np.stack([pool32 == c for c in range(n_lab)], 1).reshape(len(pool32), n_lab, -1).astype(np.float32)
    inter = np.einsum("cp,ncp->nc", t1, p1)
    den = t1.sum(1)[None] + p1.sum(2)
    present = (t1.sum(1)[None] > 0) | (p1.sum(2) > 0)
    present[:, 0] = False
    w = present.astype(np.float32)
    w[:, kc] *= 3
    dice = np.where(den > 0, 2 * inter / np.maximum(den, 1e-6), 0.0)
    return (dice * w).sum(1) / np.maximum(w.sum(1), 1e-6)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-shards", default="data/trusted/pairs_def")
    ap.add_argument("--pool-shards", default="data/trusted/pairs_dense")
    ap.add_argument("--out", default="outputs/m9")
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
    from train_m7 import load_planes
    from train_m9 import native_refs

    dev = args.device
    planes = load_planes(args.test_shards)
    refs = native_refs(planes, args.test_shards)
    pool = load_planes(args.pool_shards)
    pool_lab32 = np.stack([p["lab"][::4, ::4] for p in pool])
    pool_fold = np.asarray([LE.fold_of(p["kid"]) for p in pool])
    kern = LE.gauss_kernel(2.0, device=dev)
    n = len(planes)
    out = np.zeros((n, 128, 128), np.float16); done = np.zeros(n, bool)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 128, device=dev), torch.linspace(-1, 1, 128, device=dev), indexing="ij")
    base = torch.stack([xx, yy], -1)[None]

    def onehot(lab):
        x = F.one_hot(torch.as_tensor(lab, device=dev).long().clamp(0, PT.N_LAB - 1), PT.N_LAB).permute(0, 3, 1, 2).float()
        return torch.stack([LE.smooth_t(x[:, c], kern) for c in range(PT.N_LAB)], 1)

    t0 = time.time(); sims, fin, picked = [], [], {}
    for kid, (r, tg) in sorted(refs.items()):
        f = LE.fold_of(kid)
        cand = np.flatnonzero(pool_fold != f)
        kc = int(np.bincount(planes[r]["lab"][planes[r]["kidney"]], minlength=PT.N_LAB).argmax())
        ids = sorted(set(tg) | {r})                      # every native plane of the kidney (the scored ones are tg)
        src_ids = []
        for i in ids:
            s = similarity(planes[i]["lab"][::4, ::4], pool_lab32[cand], kc, PT.N_LAB)
            j = int(cand[int(np.argmax(s))]); src_ids.append(j); sims.append(float(s.max()))
            assert LE.fold_of(pool[j]["kid"]) != f
        picked[kid] = sorted({pool[j]["kid"] for j in src_ids})
        B = len(ids)
        src = onehot(np.stack([pool[j]["lab"] for j in src_ids])); dst = onehot(np.stack([planes[i]["lab"] for i in ids]))
        present = ((src.sum((2, 3)) > 20) | (dst.sum((2, 3)) > 20)).float()
        present[:, 0] = 0
        wcls = present.clone(); wcls[:, kc] *= 3
        prm = torch.zeros(B, 2, args.grid, args.grid, device=dev, requires_grad=True)
        opt = torch.optim.Adam([prm], lr=args.lr)
        for _ in range(args.steps):
            d = F.interpolate(prm, size=(128, 128), mode="bilinear", align_corners=True) * (2 / 127.0)
            g = base + d.permute(0, 2, 3, 1)
            w = F.grid_sample(src, g, mode="bilinear", padding_mode="border", align_corners=True)
            inter = (w * dst).sum((2, 3)); den = w.sum((2, 3)) + dst.sum((2, 3)) + 1e-3
            dice = ((2 * inter / den) * wcls).sum(1) / wcls.sum(1)
            sm = ((prm[:, :, 1:] - prm[:, :, :-1]) ** 2).mean() + ((prm[:, :, :, 1:] - prm[:, :, :, :-1]) ** 2).mean()
            loss = -dice.sum() + args.w_smooth * sm * B
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            d = F.interpolate(prm, size=(128, 128), mode="bilinear", align_corners=True) * (2 / 127.0)
            g = base + d.permute(0, 2, 3, 1)
            img = torch.as_tensor(np.stack([pool[j]["real"].astype(np.float32) for j in src_ids]), device=dev)[:, None]
            out[ids] = F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0].cpu().numpy()
            fin.append(float(dice.mean()))
        done[ids] = True
    print(f"{len(refs)} kidneys, {int(done.sum())} planes, retrieval similarity {np.mean(sims):.3f} "
          f"(min {np.min(sims):.3f}), label Dice after warp {np.mean(fin):.3f}, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    phys = np.stack([np.round(p["phys"] * 255) for p in planes]).astype(np.uint8)
    np.savez_compressed(os.path.join(args.out, "preds_retrieve.npz"), phys=phys, gen=out, done=done,
                        sessions=np.asarray([p["kid"] for p in planes]))
    print(f"wrote {args.out}/preds_retrieve.npz; source patients per kidney: "
          f"{np.mean([len(v) for v in picked.values()]):.1f} on average")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
