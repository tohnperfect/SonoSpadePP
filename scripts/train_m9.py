#!/usr/bin/env python3
"""Train M9 (sonospadepp.exemplar: the translator given one real US image of the same patient) and its
control M9c (identical, CT channels zeroed, so the correspondence comes from the label maps alone),
with the patient-level 5-fold split of TRUSTED and the three B0 seeds.

Training: target and reference are two dense planes (trusted_dense_prepare.py) of the same training
kidney, at least --min-gap degrees apart. Test (the one-image setting): for every kidney of the test
fold, the central native plane (the 4th of 7 by sweep angle) is the given image; the other six native
planes are predicted and scored. The reference planes themselves are not predicted (NaN in the
scores), so every row of this setting is compared on the same 354 planes.

Loss: negative same-plane agreement of the output, plus --w-aux times that of the warped reference W
(so the correspondence is trained directly as well), plus a small L1 pull toward the frozen B0 output.

Outputs: outputs/m9/preds_{tag}_s{seed}.npz (the translator) and preds_{tag}w_s{seed}.npz (W alone).

  python -I scripts/train_m9.py --mode ct  --seeds 0 1 2
  python -I scripts/train_m9.py --mode lab --seeds 0 1 2      # control
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def native_refs(test_all, test_shards):
    """Give the native test planes their sweep angle; return {kid: (ref_id, [target ids])}."""
    phis = {}
    for f in sorted(glob.glob(os.path.join(test_shards, "*.npz"))):
        for p in json.loads(str(np.load(f, allow_pickle=False)["meta"]))["planes"]:
            phis.setdefault(p["kid"], []).append(p["phi_deg"])
    cnt = {}
    for p in test_all:
        j = cnt.get(p["kid"], 0); p["phi"] = float(phis[p["kid"]][j]); cnt[p["kid"]] = j + 1
    by = {}
    for i, p in enumerate(test_all):
        by.setdefault(p["kid"], []).append(i)
    out = {}
    for kid, ids in by.items():
        ids = sorted(ids, key=lambda i: test_all[i]["phi"])
        r = ids[len(ids) // 2]
        out[kid] = (r, [i for i in ids if i != r])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", default="data/trusted/pairs_dense")
    ap.add_argument("--test-shards", default="data/trusted/pairs_def")
    ap.add_argument("--mode", choices=["ct", "lab"], default="ct")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-corr", type=float, default=5e-4)
    ap.add_argument("--w-aux", type=float, default=0.5, help="weight of the warped reference's own agreement")
    ap.add_argument("--w-anchor", type=float, default=0.05, help="L1 pull toward the frozen B0 output")
    ap.add_argument("--min-gap", type=float, default=2.0, help="min angle (deg) between training target and reference")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/m9")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    import torch
    from usbg import render_lotus as R
    from sonospadepp import exemplar as EX
    from sonospadepp import learned_echo as LE
    from sonospadepp import metrics as M
    from sonospadepp import paired_translator as PT
    from sonospadepp.variants import Translator
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_m7 import load_planes, prepare_regions, agreement, stack

    dev = R.get_device()
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
    rad = M._sector_radius((128, 128))
    kern = LE.gauss_kernel(args.sigma, device=dev)
    t0 = time.time()
    train_all = load_planes(args.train_shards)
    test_all = load_planes(args.test_shards)
    refs = native_refs(test_all, args.test_shards)
    prepare_regions(train_all, rad, args.sigma, t); prepare_regions(test_all, rad, args.sigma, t)
    print(f"{len(train_all)} training planes, {len(test_all)} test planes ({sum(len(v[1]) for v in refs.values())} "
          f"targets), prepared in {time.time() - t0:.0f}s", flush=True)
    use_ct = args.mode == "ct"
    tag = args.tag or args.mode
    os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.runs, "m9"), exist_ok=True)
    dense_by_kid = {}
    for i, p in enumerate(train_all):
        dense_by_kid.setdefault(p["kid"], []).append(i)

    def realf(planes, ids):
        return t(stack(planes, ids, "real").astype(np.float32))

    def forward(G, Cr, planes_t, it, planes_r, ir):
        x_t = EX.ct_inputs(stack(planes_t, it, "lab"), stack(planes_t, it, "z"), stack(planes_t, it, "hu"), use_ct, dev)
        x_r = EX.ct_inputs(stack(planes_r, ir, "lab"), stack(planes_r, ir, "z"), stack(planes_r, ir, "hu"), use_ct, dev)
        W, C = Cr(x_t, x_r, t(stack(planes_r, ir, "valid").astype(np.float32)), realf(planes_r, ir))
        c, s = EX.gen_inputs(stack(planes_t, it, "phys"), x_t, W, C, dev)
        return PT.forward01(G, c, s), W

    for seed in args.seeds:
        ck = os.path.join(args.runs, f"b0_s{seed}.pt")
        G0 = Translator(ck, device=dev).G
        for q in G0.parameters():
            q.requires_grad_(False)
        gen_out = np.zeros((len(test_all), 128, 128), np.float16); w_out = np.zeros_like(gen_out)
        done = np.zeros(len(test_all), bool)
        log = {}
        for fold in args.folds:
            test_kids = [k for k in refs if LE.fold_of(k) == fold]
            train = [i for i, p in enumerate(train_all) if LE.fold_of(p["kid"]) != fold and p["regions"]]
            torch.manual_seed(1000 * seed + fold); rng = np.random.default_rng(1000 * seed + fold)
            G = PT.build(ck, dev, n_extra=PT.N_CT + EX.N_REF); G.train()
            Cr = EX.corr_module(dev); Cr.train()
            opt = torch.optim.Adam([{"params": G.parameters(), "lr": args.lr},
                                    {"params": Cr.parameters(), "lr": args.lr_corr}], betas=(0.5, 0.999))
            hist, t1 = [], time.time()
            for step in range(args.steps):
                bi = rng.choice(train, size=min(args.batch, len(train)), replace=False)
                ri = []
                for i in bi:
                    pool = [j for j in dense_by_kid[train_all[i]["kid"]]
                            if abs(train_all[j]["phi"] - train_all[i]["phi"]) >= args.min_gap]
                    ri.append(int(rng.choice(pool)))
                fake, W = forward(G, Cr, train_all, bi, train_all, ri)
                with torch.no_grad():
                    ref = ((G0(t(stack(train_all, bi, "phys"))[:, None] * 2 - 1,
                               EX.ct_inputs(stack(train_all, bi, "lab"), stack(train_all, bi, "z"), stack(train_all, bi, "hu"),
                                            False, dev)[:, :PT.N_LAB])[:, 0] + 1) / 2).clamp(0, 1)
                score = agreement(LE.smooth_t(fake, kern), train_all, bi).mean()
                score_w = agreement(LE.smooth_t(W, kern), train_all, bi).mean()
                vmask = t(stack(train_all, bi, "valid").astype(np.float32))
                anchor = ((fake - ref).abs() * vmask).sum() / vmask.sum().clamp(min=1)
                loss = -score - args.w_aux * score_w + args.w_anchor * anchor
                opt.zero_grad(); loss.backward(); opt.step()
                hist.append((float(score.detach()), float(score_w.detach()), float(anchor.detach())))
            G.eval(); Cr.eval()
            held, held_w = [], []
            with torch.no_grad():
                for kid in test_kids:
                    r, tg = refs[kid]
                    out, W = forward(G, Cr, test_all, tg, test_all, [r] * len(tg))
                    gen_out[tg] = out.cpu().numpy().astype(np.float16); w_out[tg] = W.cpu().numpy().astype(np.float16)
                    done[tg] = True
                    keep = [b for b, i in enumerate(tg) if test_all[i]["regions"]]
                    if keep:
                        held += agreement(LE.smooth_t(out[keep], kern), test_all, [tg[b] for b in keep]).cpu().tolist()
                        held_w += agreement(LE.smooth_t(W[keep], kern), test_all, [tg[b] for b in keep]).cpu().tolist()
            h = np.asarray(hist)
            log[fold] = {"train_r_last50": float(h[-50:, 0].mean()), "train_rw_last50": float(h[-50:, 1].mean()),
                         "anchor_last50": float(h[-50:, 2].mean()), "heldout_r": float(np.mean(held)),
                         "heldout_rw": float(np.mean(held_w)), "scale": float(Cr.log_scale.exp()),
                         "lam": float(Cr.log_lam.exp()), "minutes": round((time.time() - t1) / 60, 1)}
            torch.save({"G": {k: v.detach().cpu() for k, v in G.state_dict().items()},
                        "corr": {k: v.detach().cpu() for k, v in Cr.state_dict().items()}},
                       os.path.join(args.runs, "m9", f"{tag}_s{seed}_f{fold}.pt"))
            L = log[fold]
            print(f"seed {seed} fold {fold}: train r {L['train_r_last50']:+.3f} (W {L['train_rw_last50']:+.3f}), held-out r "
                  f"{L['heldout_r']:+.3f} (W {L['heldout_rw']:+.3f}), scale {L['scale']:.2f}, lam {L['lam']:.3f}, "
                  f"{L['minutes']} min", flush=True)
        json.dump(log, open(os.path.join(args.runs, "m9", f"{tag}_s{seed}.json"), "w"))
        phys = np.stack([np.round(p["phys"] * 255) for p in test_all]).astype(np.uint8)
        sess = np.asarray([p["kid"] for p in test_all])
        np.savez_compressed(os.path.join(args.out, f"preds_{tag}_s{seed}.npz"), phys=phys, gen=gen_out, done=done, sessions=sess)
        np.savez_compressed(os.path.join(args.out, f"preds_{tag}w_s{seed}.npz"), phys=phys, gen=w_out, done=done, sessions=sess)
        print(f"wrote {args.out}/preds_{tag}_s{seed}.npz and preds_{tag}w_s{seed}.npz ({int(done.sum())}/{len(test_all)})",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
