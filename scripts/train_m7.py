#!/usr/bin/env python3
"""Train M7 (paired_translator: the whole SPADE translator fine-tuned on same-patient pairs, with the
CT image in its content path and its conditioning) and its control M7c (identical, CT channels set
to zero), with the patient-level 5-fold split of TRUSTED and the three B0 seeds.

Training planes: the dense planes (trusted_dense_prepare.py, every 1 degree through the kidney) of the
patients outside the test fold. Test planes: the seven native planes per kidney of the test fold
(data/trusted/pairs_def), the planes every other row is scored on.

--reg-net adds a registration network trained with the translator (RegGAN, Kong et al., NeurIPS
2021): from the output and the real US it predicts a smooth displacement (4 x 4 control points, at
most 4 px = 5 mm) that warps the output before the loss, so residual misregistration does not
penalise a correct translation. It is used in training only; outputs are scored unwarped.

Loss: negative same-plane agreement (depth-detrended r of the 1.5 px smoothed output with the
smoothed real US, over the eroded kidney and over the kidney + 4 px band, as for M4 to M6) plus a
small L1 pull toward the frozen B0 output (keeps the texture of a translator trained for realism).

  python -I scripts/train_m7.py --mode ct  --seeds 0 1 2
  python -I scripts/train_m7.py --mode lab --seeds 0 1 2      # control
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

FIELDS = ("phys_b0", "lab_base", "z", "hu_img", "real", "kidney_us", "valid")


def load_planes(shard_dir):
    planes = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "*.npz"))):
        d = np.load(f, allow_pickle=False)
        phi = d["phi_deg"] if "phi_deg" in d.files else np.full(len(d["sessions"]), np.nan, np.float32)
        for j in range(len(d["sessions"])):
            planes.append({"kid": str(d["sessions"][j]), "phi": float(phi[j]),
                           "phys": d["phys_b0"][j].astype(np.float32) / 255.0, "lab": d["lab_base"][j].astype(np.int64),
                           "z": d["z"][j].astype(np.float32), "hu": d["hu_img"][j].astype(np.float32),
                           "real": d["real"][j] / 255.0, "kidney": d["kidney_us"][j] > 0, "valid": d["valid"][j] > 0})
    return planes


def prepare_regions(planes, rad, sigma, t):
    import torch
    from scipy.ndimage import binary_dilation, binary_erosion
    from sonospadepp import learned_echo as LE
    from sonospadepp import metrics as M
    for p in planes:
        k = p["kidney"] & p["valid"]
        real_s = M.smooth(p["real"], sigma)
        p["regions"] = []
        for reg in (binary_erosion(k, iterations=1) & p["valid"], binary_dilation(k, iterations=4) & p["valid"]):
            if reg.sum() < 50:
                continue
            idx, X = LE.region_design(reg, rad)
            p["regions"].append((t(idx, torch.long), t(X), t(np.linalg.inv(X.T @ X)), t(LE.detrended_resid_np(real_s, idx, X))))


def agreement(sm, planes, ids):
    """Mean over planes of the mean over regions of the detrended r (torch, differentiable)."""
    import torch
    from sonospadepp import learned_echo as LE
    return torch.stack([torch.stack([LE.detrended_r_t(sm[b], *rg) for rg in planes[i]["regions"]]).mean()
                        for b, i in enumerate(ids)])


def reg_net(device, grid=4):
    """Small CNN: (output, real) -> (B, 2, grid, grid) displacement parameters; last layer zero (identity)."""
    import torch.nn as nn
    layers, c = [], 2
    for co in (16, 32, 64, 64):
        layers += [nn.Conv2d(c, co, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
        c = co
    last = nn.Conv2d(c, 2, 1); nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(grid), last).to(device)


def warp(img, prm, max_px):
    """img (B,H,W), prm (B,2,g,g) -> img warped by a bilinear displacement field of at most max_px."""
    import torch
    import torch.nn.functional as F
    B, H, W = img.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=img.device), torch.linspace(-1, 1, W, device=img.device),
                            indexing="ij")
    d = torch.tanh(F.interpolate(prm, size=(H, W), mode="bilinear", align_corners=True)) * (max_px * 2 / (W - 1))
    g = torch.stack([xx, yy], -1)[None] + d.permute(0, 2, 3, 1)
    return F.grid_sample(img[:, None], g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]


def patch_disc(in_ch, device):
    """pix2pix's 70x70 PatchGAN (n_layers=3): C64-C128-C256-C512 then a 1-channel map, instance norm."""
    import torch.nn as nn
    layers = [nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True)]
    for cin, cout, st in ((64, 128, 2), (128, 256, 2), (256, 512, 1)):
        layers += [nn.Conv2d(cin, cout, 4, st, 1), nn.InstanceNorm2d(cout), nn.LeakyReLU(0.2, True)]
    layers += [nn.Conv2d(512, 1, 4, 1, 1)]
    return nn.Sequential(*layers).to(device)


def stack(planes, ids, key):
    return np.stack([planes[i][key] for i in ids])


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
    ap.add_argument("--w-anchor", type=float, default=0.05, help="L1 pull toward the frozen B0 output")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/m7")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reg-net", action="store_true", help="joint registration network (training only)")
    ap.add_argument("--b0", default="runs/b0_s{seed}.pt", help="starting checkpoint per seed (merge: the v2 B0)")
    ap.add_argument("--reg-max-px", type=float, default=4.0)
    ap.add_argument("--loss", choices=["pair", "p2p"], default="pair",
                    help="pair: this work's same-plane agreement loss; p2p: the pix2pix baseline (Isola et al., "
                         "CVPR 2017): L1 to the real plane (weight --p2p-l1) plus a conditional PatchGAN, no anchor")
    ap.add_argument("--p2p-l1", type=float, default=100.0)
    ap.add_argument("--exemplars", default=None, help="H-RX: training exemplars (build_exemplars.py); the exemplar "
                    "goes into the extra input channel that lab mode keeps at zero")
    ap.add_argument("--exemplars-test", default=None, help="H-RX: test exemplars (ref_retrieval.py preds)")
    ap.add_argument("--p2p-lr", type=float, default=2e-4, help="pix2pix's learning rate (G and D)")
    args = ap.parse_args(argv)

    import torch
    from usbg import render_lotus as R
    from sonospadepp import learned_echo as LE
    from sonospadepp import metrics as M
    from sonospadepp import paired_translator as PT
    from sonospadepp.variants import Translator

    dev = R.get_device()
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
    rad = M._sector_radius((128, 128))
    kern = LE.gauss_kernel(args.sigma, device=dev)
    t0 = time.time()
    train_all = load_planes(args.train_shards)
    test_all = load_planes(args.test_shards)
    prepare_regions(train_all, rad, args.sigma, t); prepare_regions(test_all, rad, args.sigma, t)
    print(f"{len(train_all)} training planes, {len(test_all)} test planes, prepared in {time.time() - t0:.0f}s", flush=True)
    use_ct = args.mode == "ct"
    tag = args.tag or args.mode
    os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.runs, "m7"), exist_ok=True)

    EX = EXI = EXT = None
    if args.exemplars:                                   # H-RX: retrieved, label-warped real image of another patient
        e = np.load(args.exemplars, allow_pickle=False)
        assert list(e["sessions"]) == [p["kid"] for p in train_all], "exemplars built on another plane order"
        EX, EXI = e["ex"], e["idx"]
        et = np.load(args.exemplars_test, allow_pickle=False)
        assert list(et["sessions"]) == [p["kid"] for p in test_all] and et["done"].all()
        EXT = et["gen"]

    def run(G, planes, ids, ex=None):
        if ex is None:
            c, s = PT.inputs(stack(planes, ids, "phys"), stack(planes, ids, "lab"), stack(planes, ids, "z"),
                             stack(planes, ids, "hu"), use_ct, dev)
        else:                                            # z channel 0, HU channel = exemplar (in [0, 1])
            e = np.stack([ex(i) for i in ids]).astype(np.float32)
            c, s = PT.inputs(stack(planes, ids, "phys"), stack(planes, ids, "lab"), np.zeros_like(e), e, True, dev)
        return PT.forward01(G, c, s)

    for seed in args.seeds:
        ck = args.b0.format(seed=seed)
        G0 = Translator(ck, device=dev).G
        for q in G0.parameters():
            q.requires_grad_(False)
        gen_out = np.zeros((len(test_all), 128, 128), np.float16)
        done = np.zeros(len(test_all), bool)
        log = {}
        for fold in args.folds:
            test = [i for i, p in enumerate(test_all) if LE.fold_of(p["kid"]) == fold]
            train = [i for i, p in enumerate(train_all) if LE.fold_of(p["kid"]) != fold and p["regions"]]
            torch.manual_seed(1000 * seed + fold); rng = np.random.default_rng(1000 * seed + fold)
            G = PT.build(ck, dev); G.train()
            ex_tr = ex_te = None
            if EX is not None:
                def ex_tr(i, _f=fold):
                    k = int(EXI[_f, i]); assert k >= 0, "training plane from the test fold"
                    return EX[k]
                ex_te = lambda i: EXT[i]
            params = list(G.parameters())
            if args.reg_net:
                Rn = reg_net(dev); params += list(Rn.parameters())
            p2p = args.loss == "p2p"
            opt = torch.optim.Adam(params, lr=args.p2p_lr if p2p else args.lr, betas=(0.5, 0.999))
            if p2p:
                D = patch_disc(1 + PT.N_LAB + 1, dev); D.train()
                opt_d = torch.optim.Adam(D.parameters(), lr=args.p2p_lr, betas=(0.5, 0.999))
                bce = torch.nn.BCEWithLogitsLoss()
            hist, t1 = [], time.time()
            for step in range(args.steps):
                bi = rng.choice(train, size=min(args.batch, len(train)), replace=False)
                fake = run(G, train_all, bi, ex_tr)
                if p2p:                                   # pix2pix: conditional PatchGAN + L1, fan-masked
                    vm = t(stack(train_all, bi, "valid").astype(np.float32))[:, None]
                    real_t = t(stack(train_all, bi, "real").astype(np.float32))[:, None]
                    c_in, s_in = PT.inputs(stack(train_all, bi, "phys"), stack(train_all, bi, "lab"),
                                           stack(train_all, bi, "z"), stack(train_all, bi, "hu"), False, dev)
                    cond = torch.cat([c_in[:, :1], s_in[:, :PT.N_LAB]], 1) * vm
                    f_in = torch.cat([cond, (fake[:, None] * 2 - 1) * vm], 1)
                    r_in = torch.cat([cond, (real_t * 2 - 1) * vm], 1)
                    pr, pf = D(r_in), D(f_in.detach())
                    loss_d = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf)))
                    opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                    pg = D(f_in)
                    l1 = ((fake[:, None] - real_t).abs() * vm).sum() / vm.sum().clamp(min=1)
                    loss = bce(pg, torch.ones_like(pg)) + args.p2p_l1 * l1
                    opt.zero_grad(); loss.backward(); opt.step()
                    hist.append((float(l1.detach()), float(loss_d.detach())))
                    continue
                with torch.no_grad():
                    ref = ((G0(t(stack(train_all, bi, "phys"))[:, None] * 2 - 1,
                               PT.inputs(stack(train_all, bi, "phys"), stack(train_all, bi, "lab"), stack(train_all, bi, "z"),
                                         stack(train_all, bi, "hu"), False, dev)[1][:, :PT.N_LAB])[:, 0] + 1) / 2).clamp(0, 1)
                if args.reg_net:
                    real_t = t(stack(train_all, bi, "real").astype(np.float32))
                    fake_s = warp(fake, Rn(torch.stack([fake, real_t], 1)), args.reg_max_px)
                else:
                    fake_s = fake
                score = agreement(LE.smooth_t(fake_s, kern), train_all, bi).mean()
                vmask = t(stack(train_all, bi, "valid").astype(np.float32))
                anchor = ((fake - ref).abs() * vmask).sum() / vmask.sum().clamp(min=1)
                loss = -score + args.w_anchor * anchor
                opt.zero_grad(); loss.backward(); opt.step()
                hist.append((float(score.detach()), float(anchor.detach())))
            G.eval()
            with torch.no_grad():
                held = []
                for j in range(0, len(test), 16):
                    ids = test[j:j + 16]
                    out = run(G, test_all, ids, ex_te)
                    gen_out[ids] = out.cpu().numpy().astype(np.float16); done[ids] = True
                    keep = [b for b, i in enumerate(ids) if test_all[i]["regions"]]
                    if keep:
                        sm = LE.smooth_t(out[keep], kern)
                        held += agreement(sm, test_all, [ids[b] for b in keep]).cpu().tolist()
            h = np.asarray(hist)
            log[fold] = {"train_r_last50": float(h[-50:, 0].mean()), "anchor_last50": float(h[-50:, 1].mean()),
                         "heldout_r": float(np.mean(held)), "minutes": round((time.time() - t1) / 60, 1)}
            torch.save({k: v.detach().cpu() for k, v in G.state_dict().items()},
                       os.path.join(args.runs, "m7", f"{tag}_s{seed}_f{fold}.pt"))
            print(f"seed {seed} fold {fold}: train r {log[fold]['train_r_last50']:+.3f}, held-out r {log[fold]['heldout_r']:+.3f}, "
                  f"anchor {log[fold]['anchor_last50']:.3f}, {log[fold]['minutes']} min", flush=True)
        json.dump(log, open(os.path.join(args.runs, "m7", f"{tag}_s{seed}.json"), "w"))
        phys = np.stack([np.round(p["phys"] * 255) for p in test_all]).astype(np.uint8)
        np.savez_compressed(os.path.join(args.out, f"preds_{tag}_s{seed}.npz"), phys=phys, gen=gen_out, done=done,
                            sessions=np.asarray([p["kid"] for p in test_all]))
        print(f"wrote {args.out}/preds_{tag}_s{seed}.npz ({int(done.sum())}/{len(test_all)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
