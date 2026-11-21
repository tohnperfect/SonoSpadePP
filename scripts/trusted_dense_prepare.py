#!/usr/bin/env python3
"""Dense 128 px training planes from TRUSTED, with the deformable registration.

The test planes stay the seven native planes per kidney of trusted_hires_prepare.py (data/trusted/
pairs_def). For training a larger translator, every sweep angle through the kidney is used here:
candidate angles every 1 degree (as trusted_prepare.py), kept where the kidney cross-section is at
least half its largest, each cut and rendered as the 128 px planes of trusted_hires_prepare.py
(registration: landmark Kabsch, trimmed ICP if it raises Dice, deformable kidney-surface refinement).

Per kidney, data/trusted/pairs_dense/<kid>.npz holds phys_b0, lab_base, z, hu_img, real, kidney_us,
valid, kidney_ct (128 x 128, sector-warped), phi_deg and the render seeds.

  python -I scripts/trusted_dense_prepare.py --kids 206R 206L       # some kidneys (default: all)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

KINDS = {"phys_b0": "image", "lab_base": "label", "z": "float", "real": "image", "kidney_us": "label",
         "valid": "label", "kidney_ct": "label", "hu_img": "float"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kids", nargs="*", default=None)
    ap.add_argument("--cache", default="data/trusted/pp_cache")
    ap.add_argument("--rigid-meta", default="data/trusted_pairs_meta.json", help="the kidneys of the 128 px run")
    ap.add_argument("--out", default="data/trusted/pairs_dense")
    ap.add_argument("--step-deg", type=float, default=1.0)
    ap.add_argument("--min-area-frac", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=2000.0)
    args = ap.parse_args(argv)

    import nibabel as nib
    from scipy.ndimage import map_coordinates
    from usbg import render_lotus as R
    from usbg import volume as V
    from usbg import slicer as S
    from usbg.segmenter import align_label_to_image
    from sonospadepp import register as RG, trusted as TR, deform as DF, hires as H
    from sonospadepp.ctvol import load_pp
    from sonospadepp.physics import maps_from_labels, render_from_maps
    from sonospadepp.slicing import relative_hu as relative_hu_std
    from sonospadepp.warp import warp_frame
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_pairs import resize_img, resize_float
    from trusted_prepare import leading_air_fill

    dev = R.get_device()
    kids = sorted({p["kid"] for p in json.load(open(args.rigid_meta))["planes"]})
    os.makedirs(args.out, exist_ok=True)
    pts_std, _ = S.scan_grid_probe(); Lg, Sg = pts_std.shape[:2]
    entries = [e for e in TR.index() if e["kid"] in kids and (args.kids is None or e["kid"] in args.kids)]

    def world(T_wp, pts):
        return (pts.reshape(-1, 3) @ T_wp[:3, :3].T + T_wp[:3, 3]) * 1000.0      # mm

    for e in entries:
        out_d = os.path.join(args.out, f"{e['kid']}.npz")
        if os.path.exists(out_d):
            continue
        t0 = time.time()
        vol = load_pp(os.path.join(args.cache, f"{e['pid']}.npz"))
        us_n = nib.load(e["us_img"]); us_aff = us_n.affine
        us = np.asanyarray(us_n.dataobj)
        us_mask = np.asanyarray(nib.load(e["us_mask"]).dataobj) > 0
        ct_n = nib.load(e["ct_mask"]); ct_aff = ct_n.affine
        l_us, l_ct = np.loadtxt(e["us_ldk"]), np.loadtxt(e["ct_ldk"])
        ct_k = TR.ct_kidney_side(np.asanyarray(ct_n.dataobj), ct_aff, l_ct)
        R0, t0v = RG.kabsch(l_us, l_ct); T0 = RG.to_T(R0, t0v)
        dice0 = RG.mask_dice(us_mask, us_aff, ct_k, ct_aff, T0)
        T1, _ = RG.icp_rigid(RG.surface_points(us_mask, us_aff), RG.surface_points(ct_k, ct_aff), T0)
        dice1 = RG.mask_dice(us_mask, us_aff, ct_k, ct_aff, T1)
        T = T1 if dice1 >= dice0 else T0
        field, info = DF.fit_deform(us_mask, us_aff, ct_k, ct_aff, T, lam=args.lam)
        g = TR.fit_geometry(us)
        inv_us = np.linalg.inv(us_aff)

        def us_at(pw_mm, arr, order):
            vox = (pw_mm @ inv_us[:3, :3].T + inv_us[:3, 3]).T
            return map_coordinates(arr, vox, order=order, mode="constant", cval=0, output=np.float32)

        cand = np.arange(-(g.sweep_half - 3), g.sweep_half - 3 + 1e-6, args.step_deg)
        areas, cache = [], []
        for phi in cand:
            T_wp = TR.T_world_from_probe_m(us_aff, *TR.plane_pose_vox(g, phi))
            ps = world(T_wp, pts_std)
            ok = TR.fan_valid(g, ps @ inv_us[:3, :3].T + inv_us[:3, 3])
            km = us_at(ps, us_mask.view(np.uint8), 0) > 0
            areas.append(int((km & ok).sum())); cache.append((ps, ok, km))
        areas = np.asarray(areas)
        good = np.nonzero(areas >= args.min_area_frac * areas.max())[0]
        acc = {k: [] for k in KINDS}
        phis, seeds = [], []
        for n_pl, i in enumerate(good):
            ps, ok, km = cache[i]
            real = np.where(ok, us_at(ps, us, 1), 0.0).reshape(Lg, Sg)
            kus = km.reshape(Lg, Sg); ok = ok.reshape(Lg, Sg)
            lab_b, lab_p, hu = H.sample_ct(vol, DF.map_points(field, T, ps.astype(np.float32)))
            lab_b = lab_b.reshape(Lg, Sg); lab_p = lab_p.reshape(Lg, Sg); hu = hu.reshape(Lg, Sg)
            z = relative_hu_std(hu, lab_b, vol)
            kct = lab_b == V.KIDNEY
            lab_b, lab_p, hu, z, _ = leading_air_fill(lab_b, lab_p, hu, z)
            seed = int(e["pid"]) * 1000 + (500 if e["side"] == "R" else 0) + 100 + n_pl
            cfg = replace(R.RenderConfig(), seed=seed)
            sz = (128, 128)
            frame = {"phys_b0": resize_img(render_from_maps(*maps_from_labels(lab_b, cfg), cfg, dev), 128),
                     "lab_base": align_label_to_image(lab_b, sz).astype(np.uint8),
                     "z": resize_float(z.T, 128),
                     "real": np.clip(np.round(resize_float(real.T, 128)), 0, 255).astype(np.uint8),
                     "kidney_us": align_label_to_image(kus.astype(np.uint8), sz).astype(np.uint8),
                     "valid": align_label_to_image(ok.astype(np.uint8), sz).astype(np.uint8),
                     "kidney_ct": align_label_to_image(kct.astype(np.uint8), sz).astype(np.uint8),
                     "hu_img": resize_float(((np.clip(hu, -200, 300) + 200) / 500.0).T, 128)}
            w = warp_frame(frame, KINDS)
            for k in KINDS:
                acc[k].append(w[k])
            phis.append(float(cand[i])); seeds.append(seed)
        out = {k: np.stack(v) for k, v in acc.items()}
        out["z"] = out["z"].astype(np.float16); out["hu_img"] = out["hu_img"].astype(np.float16)
        np.savez_compressed(out_d, sessions=np.asarray([e["kid"]] * len(phis)), phi_deg=np.asarray(phis, np.float32),
                            seeds=np.asarray(seeds), **out,
                            meta=json.dumps({"dice_def": round(DF.dice_mapped(us_mask, us_aff, ct_k, ct_aff, T, field), 3),
                                             "jac_min": round(info["jac_min"], 3)}))
        print(f"{e['kid']}: {len(phis)} planes ({phis[0]:+.0f} to {phis[-1]:+.0f} deg), {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
