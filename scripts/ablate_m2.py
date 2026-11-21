#!/usr/bin/env python3
"""Which part of M2 helps on the same patient? Re-render every TRUSTED plane with M2 split apart,
from the raw scan-grid slices that trusted_prepare.py stores, and translate with M2's generators.

Arms (soft-tissue classes only, as in M2; air, bone, lung and background unchanged):
  m2z      impedance modulated by CT density, backscatter of the labels
  m2b      backscatter modulated by |z|, impedance of the labels
  m2bmean  backscatter scaled per tissue by the plane's MEAN M2 backscatter factor for that tissue:
           the tissue-level darkening of m2b without its within-tissue pattern
The speckle seed and every other render setting are those of the stored phys_m2 frames, which the
script first reproduces as a check. Output per arm and GAN seed: outputs/m2abl/preds_<arm>_s<seed>.npz
in the format of train_m4.py, for scripts/eval_trusted.py --pred.

  python -I scripts/ablate_m2.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import replace

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/trusted/pairs")
    ap.add_argument("--cache", default="data/trusted/pp_cache")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="outputs/m2abl")
    args = ap.parse_args(argv)

    from usbg import render_lotus as R
    from sonospadepp.labels import HU_MODULATED
    from sonospadepp.physics import maps_from_labels, maps_hu_modulated, render_from_maps
    from sonospadepp.variants import Translator
    from sonospadepp.warp import warp_frame
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_pairs import resize_img

    dev = R.get_device()
    post = lambda img: warp_frame({"p": resize_img(img, 128)}, {"p": "image"})["p"]
    arms = ("m2z", "m2b", "m2bmean")
    phys = {a: [] for a in arms}
    labs, sessions, check = [], [], []
    refs = {}
    t0 = time.time()
    for f in sorted(glob.glob(os.path.join(args.shards, "*.npz"))):
        d = np.load(f, allow_pickle=False)
        for j in range(len(d["sessions"])):
            kid = str(d["sessions"][j]); pid = kid[:-1]
            if pid not in refs:
                refs[pid] = np.load(os.path.join(args.cache, f"{pid}.npz"), allow_pickle=False)["ref"]
            ref = refs[pid]
            lab, hu, z = d["raw_lab"][j].astype(np.int64), d["raw_hu"][j].astype(np.float32), d["raw_z"][j].astype(np.float32)
            cfg = replace(R.RenderConfig(), seed=int(d["raw_seed"][j]))
            Z0, MU0, BS0 = maps_from_labels(lab, cfg)
            Zm, _, BSm = maps_hu_modulated(lab, hu, z, ref, cfg)
            # check: the full M2 from the stored raw slices reproduces the stored M2 frame
            full = post(render_from_maps(Zm, MU0, BSm, cfg, dev))
            check.append(float(np.abs(full.astype(np.float32) - d["phys_m2"][j].astype(np.float32)).mean()))
            mod = np.isin(lab, HU_MODULATED) & ~np.isnan(np.asarray(ref)[lab])
            fbs = np.where(mod, BSm / np.where(BS0 > 0, BS0, 1.0), 1.0)
            fmean = np.ones_like(fbs)
            for L in np.unique(lab[mod]):
                sel = mod & (lab == L)
                fmean[sel] = fbs[sel].mean()
            phys["m2z"].append(post(render_from_maps(Zm, MU0, BS0, cfg, dev)))
            phys["m2b"].append(post(render_from_maps(Z0, MU0, BSm, cfg, dev)))
            phys["m2bmean"].append(post(render_from_maps(Z0, MU0, (BS0 * fmean).astype(np.float32), cfg, dev)))
            labs.append(d["lab_base"][j]); sessions.append(kid)
    check = np.asarray(check)
    print(f"{len(sessions)} planes rendered in {time.time() - t0:.0f}s; full M2 vs stored phys_m2: "
          f"mean |diff| {check.mean():.3f}/255 (max {check.max():.3f})", flush=True)
    if check.mean() > 1.0:
        raise SystemExit("re-rendered M2 does not match the stored frames; check the raw slices")

    labs = np.stack(labs).astype(np.int64); sessions = np.asarray(sessions)
    os.makedirs(args.out, exist_ok=True)
    for seed in args.seeds:
        tr = Translator(os.path.join(args.runs, f"m2_s{seed}.pt"), device=dev)
        for a in arms:
            p = np.stack(phys[a])
            gen = np.concatenate([tr.batch(p[i:i + 32] / 255.0, labs[i:i + 32]) for i in range(0, len(p), 32)])
            np.savez_compressed(os.path.join(args.out, f"preds_{a}_s{seed}.npz"), phys=p, gen=gen.astype(np.float16),
                                done=np.ones(len(p), bool), sessions=sessions)
        print(f"seed {seed}: translated {len(arms)} arms with runs/m2_s{seed}.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
