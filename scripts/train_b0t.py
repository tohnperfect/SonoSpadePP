#!/usr/bin/env python3
"""B0t, the domain-matched control for approach 1 (M7c): SonoSPADE's own UNPAIRED objective and loss
weights (configs/sonospade.yaml, as train_variant.py), started from B0 and fine-tuned on TRUSTED with
the patient-level 5-fold split, on exactly the training planes M7c uses (the dense planes of the
patients outside the test fold).

The content (physics render + label map) and the real US are drawn independently, as in SonoSPADE's
training: the real images only teach the discriminator what TRUSTED ultrasound looks like. M7c differs
from B0t only in using the pairing (the same-plane agreement loss). So M7c - B0t is the effect of the
paired training, and B0t - B0 the effect of matching TRUSTED's scanner domain.

B0's checkpoint keeps the generator (raw and EMA) but not the discriminator, so a fresh discriminator
is first trained alone for --d-warmup iterations against the frozen B0 output, then both are trained.

  python -I scripts/train_b0t.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def d_only_step(gan, phys, onehot, real):
    """The discriminator half of TG.SonoSPADE.step (balanced per-pixel CE, R1, LabelMix), G untouched."""
    import torch
    with torch.no_grad():
        fake = gan.G(phys, onehot)
        pseudo = gan.S.net(gan._to01(real)).argmax(dim=1)
    B, _, H, W = fake.shape
    gan.opt_d.zero_grad()
    use_r1 = gan.r1_gamma and gan.r1_gamma > 0
    real_in = real.detach().requires_grad_(True) if use_r1 else real
    d_real = gan.D(real_in)
    d_fake = gan.D(fake)
    loss = gan._balanced_ce(d_real, pseudo) + gan._balanced_ce(d_fake, torch.full_like(onehot.argmax(1), gan.n_seg))
    if use_r1:
        grad = torch.autograd.grad(d_real.sum(), real_in, create_graph=True)[0]
        loss = loss + 0.5 * gan.r1_gamma * grad.pow(2).flatten(1).sum(1).mean() / float(H * W)
    if gan.lambda_lm and gan.lambda_lm > 0:
        m = gan._labelmix_mask(B, H, W)
        loss = loss + gan.lambda_lm * (gan.D(m * real + (1 - m) * fake) - (m * d_real.detach() + (1 - m) * d_fake.detach())).pow(2).mean()
    loss.backward(); gan.opt_d.step()
    return float(loss.detach())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", default="data/trusted/pairs_dense")
    ap.add_argument("--test-shards", default="data/trusted/pairs_def")
    ap.add_argument("--s-us", default="runs/s_us_phys_b0__lab_base.pt")
    ap.add_argument("--config", default="configs/sonospade.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--iters", type=int, default=1200, help="as M7c")
    ap.add_argument("--batch", type=int, default=8, help="as M7c")
    ap.add_argument("--d-warmup", type=int, default=200)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/b0t")
    ap.add_argument("--tag", default="b0t")
    ap.add_argument("--b0", default="runs/b0_s{seed}.pt", help="starting checkpoint per seed (merge: the v2 B0)")
    args = ap.parse_args(argv)

    import torch
    from usbg import segmenter as SEG
    from usbg import texture_gan as TG
    from usbg.render_lotus import get_device
    from sonospadepp import learned_echo as LE
    from sonospadepp.variants import cond_tensor
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_m7 import load_planes, stack

    dev = get_device()
    cfg = TG.load_config(args.config)
    mc, lc, tc = cfg.get("model", {}), cfg.get("loss", {}), cfg.get("train", {})
    from sonospadepp.v2.fm_seg import load_any_segmenter
    s_us = load_any_segmenter(args.s_us, device=dev)          # v1 U-Net, or a v2 foundation-model S_US
    kw = dict(s_us=s_us, ngf=mc.get("ngf", 32), ndf=mc.get("ndf", 32), n_blocks=mc.get("n_blocks", 6),
              spade_hidden=mc.get("spade_hidden", 64), lambda_nce=lc.get("lambda_nce", 1.0),
              lambda_seg=lc.get("lambda_seg", 5.0), lambda_seg_dice=lc.get("lambda_seg_dice", 1.0),
              lambda_tc=lc.get("lambda_tc", 0.0), lambda_lm=lc.get("lambda_labelmix", 0.0),
              r1_gamma=lc.get("r1_gamma", 0.0), ema_decay=tc.get("ema_decay", 0.999),
              nce_tau=lc.get("nce_tau", 0.07), lr=cfg.get("optim", {}).get("lr", 2e-4))
    train_all = load_planes(args.train_shards)
    test_all = load_planes(args.test_shards)
    print(f"{len(train_all)} training planes, {len(test_all)} test planes, {args.iters} iters x {args.batch}, "
          f"D warm-up {args.d_warmup}, {dev}", flush=True)
    os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.runs, "b0t"), exist_ok=True)
    t11 = lambda a: torch.from_numpy(np.ascontiguousarray(a)[:, None].astype(np.float32) * 2 - 1).to(dev)

    for seed in args.seeds:
        ck = torch.load(args.b0.format(seed=seed), map_location="cpu", weights_only=True)
        gen_out = np.zeros((len(test_all), 128, 128), np.float16); done = np.zeros(len(test_all), bool)
        log = {}
        for fold in args.folds:
            torch.manual_seed(1000 * seed + fold); rng = np.random.default_rng(1000 * seed + fold)
            train = [i for i, p in enumerate(train_all) if LE.fold_of(p["kid"]) != fold]
            test = [i for i, p in enumerate(test_all) if LE.fold_of(p["kid"]) == fold]
            gan = TG.SonoSPADE(dev, n_classes=13, **kw)
            gan.G.load_state_dict(ck["G_raw"]); gan.G_ema.load_state_dict(ck["G_spade"])
            t1, hist = time.time(), []

            def batch():
                pi = rng.choice(train, args.batch, replace=False); ri = rng.choice(train, args.batch, replace=False)
                return (t11(stack(train_all, pi, "phys")), cond_tensor(stack(train_all, pi, "lab"), None, 13, False, dev),
                        t11(stack(train_all, ri, "real").astype(np.float32)))
            for _ in range(args.d_warmup):
                d_only_step(gan, *batch())
            for it in range(args.iters):
                hist.append(gan.step(*batch()))
            G = gan.G_ema if gan.G_ema is not None else gan.G
            G.eval()
            with torch.no_grad():
                for j in range(0, len(test), 32):
                    ids = test[j:j + 32]
                    out = G(t11(stack(test_all, ids, "phys")), cond_tensor(stack(test_all, ids, "lab"), None, 13, False, dev))
                    gen_out[ids] = ((out[:, 0] + 1) / 2).clamp(0, 1).cpu().numpy().astype(np.float16)
                    done[ids] = True
            gan.save(os.path.join(args.runs, "b0t", f"{args.tag}_s{seed}_f{fold}.pt"))
            last = {k: float(np.mean([h[k] for h in hist[-50:]])) for k in hist[-1]} if hist else {}
            log[fold] = {"last50": last, "minutes": round((time.time() - t1) / 60, 1)}
            print(f"seed {seed} fold {fold}: " + "  ".join(f"{k} {v:.3f}" for k, v in last.items())
                  + f", {log[fold]['minutes']} min", flush=True)
        json.dump(log, open(os.path.join(args.runs, "b0t", f"{args.tag}_s{seed}.json"), "w"), indent=1)
        phys = np.stack([np.round(p["phys"] * 255) for p in test_all]).astype(np.uint8)
        np.savez_compressed(os.path.join(args.out, f"preds_{args.tag}_s{seed}.npz"), phys=phys, gen=gen_out, done=done,
                            sessions=np.asarray([p["kid"] for p in test_all]))
        print(f"wrote {args.out}/preds_{args.tag}_s{seed}.npz ({int(done.sum())}/{len(test_all)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
