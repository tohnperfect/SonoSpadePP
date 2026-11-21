#!/usr/bin/env python3
"""Full-resolution planes with rigid and deformable registration, and the 128 px planes again with
deformable registration.

Per kidney (the planes of trusted_prepare.py: same sweep angles, same speckle seeds):
  1. rigid US -> CT as trusted_prepare.py (landmark Kabsch, then trimmed ICP if it raises Dice);
  2. deformable refinement (deform.fit_deform: kidney-mask signed distances only; the landmarks are
     left out, so their residual is an independent check);
  3. per plane, on a 0.3 mm Cartesian grid: the real US (trilinear), its kidney mask and the valid
     fan; for each registration the CT labels, HU and relative HU; on a 320 x 512 scan grid, for
     each registration, the labels / HU / relative HU that the full-resolution render needs;
  4. the 128 px products of trusted_prepare.py (renders, labels, warp to the sector geometry, raw
     160 x 512 slices) with the deformable registration, in the same shard format.

  python -I scripts/trusted_hires_prepare.py --kids 206R 206L       # some kidneys (default: all)
  python -I scripts/trusted_hires_prepare.py --merge                 # merge the 128 px shards
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
    ap.add_argument("--kids", nargs="*", default=None)
    ap.add_argument("--cache", default="data/trusted/pp_cache")
    ap.add_argument("--rigid-meta", default="data/trusted_pairs_meta.json", help="plane angles of the 128 px run")
    ap.add_argument("--hires", default="data/trusted/hires")
    ap.add_argument("--shards-def", default="data/trusted/pairs_def")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out-def", default="data/trusted_pairs_def.npz")
    ap.add_argument("--lines", type=int, default=320)
    ap.add_argument("--lam", type=float, default=2000.0)
    args = ap.parse_args(argv)
    KINDS = {"phys_b0": "image", "phys_m1": "image", "phys_m2": "image", "lab_base": "label",
             "lab_pp": "label", "z": "float", "real": "image", "kidney_us": "label", "valid": "label",
             "kidney_ct": "label", "hu_img": "float"}
    if args.merge:
        return merge(args, KINDS)

    import nibabel as nib
    from scipy.ndimage import map_coordinates
    from usbg import render_lotus as R
    from usbg import volume as V
    from usbg import slicer as S
    from usbg.segmenter import align_label_to_image
    from sonospadepp import register as RG, trusted as TR, deform as DF, hires as H
    from sonospadepp.ctvol import load_pp
    from sonospadepp.physics import maps_from_labels, maps_hu_modulated, render_from_maps
    from sonospadepp.slicing import relative_hu as relative_hu_std
    from sonospadepp.warp import warp_frame
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_pairs import resize_img, resize_float
    from trusted_prepare import leading_air_fill

    dev = R.get_device()
    rmeta = json.load(open(args.rigid_meta))
    phis = {}
    for p in rmeta["planes"]:
        phis.setdefault(p["kid"], []).append(p["phi_deg"])
    os.makedirs(args.hires, exist_ok=True); os.makedirs(args.shards_def, exist_ok=True)
    pts_std, _ = S.scan_grid_probe(); Lg, Sg = pts_std.shape[:2]
    pts_pol = H.polar_grid(args.lines, Sg)
    sp_pol = H.polar_spacing_mm(args.lines, Sg)
    pts_cart, fan_cart, rad_cart, th_cart = H.cart_grid()
    NZ, NX = fan_cart.shape
    entries = [e for e in TR.index() if e["kid"] in phis and (args.kids is None or e["kid"] in args.kids)]

    def world(T_wp, pts):
        return (pts.reshape(-1, 3) @ T_wp[:3, :3].T + T_wp[:3, 3]) * 1000.0      # mm

    for e in entries:
        out_h = os.path.join(args.hires, f"{e['kid']}.npz")
        out_d = os.path.join(args.shards_def, f"{e['kid']}.npz")
        if os.path.exists(out_h) and os.path.exists(out_d):
            continue
        t0 = time.time()
        vol = load_pp(os.path.join(args.cache, f"{e['pid']}.npz"))
        us_n = nib.load(e["us_img"]); us_aff = us_n.affine
        us = np.asanyarray(us_n.dataobj)
        us_mask = np.asanyarray(nib.load(e["us_mask"]).dataobj) > 0
        ct_n = nib.load(e["ct_mask"]); ct_aff = ct_n.affine
        l_us, l_ct = np.loadtxt(e["us_ldk"]), np.loadtxt(e["ct_ldk"])
        ct_k = TR.ct_kidney_side(np.asanyarray(ct_n.dataobj), ct_aff, l_ct)
        # rigid, exactly as trusted_prepare.py
        R0, t0v = RG.kabsch(l_us, l_ct); T0 = RG.to_T(R0, t0v)
        dice0 = RG.mask_dice(us_mask, us_aff, ct_k, ct_aff, T0)
        T1, _ = RG.icp_rigid(RG.surface_points(us_mask, us_aff), RG.surface_points(ct_k, ct_aff), T0)
        dice1 = RG.mask_dice(us_mask, us_aff, ct_k, ct_aff, T1)
        T = T1 if dice1 >= dice0 else T0
        # deformable refinement
        field, info = DF.fit_deform(us_mask, us_aff, ct_k, ct_aff, T, lam=args.lam)
        reg = {"rig": None, "def": field}
        dice = {k: DF.dice_mapped(us_mask, us_aff, ct_k, ct_aff, T, f) for k, f in reg.items()}
        ldk = {k: np.linalg.norm(DF.map_points(f, T, l_us.astype(np.float32)) - l_ct, axis=1).round(2).tolist()
               for k, f in reg.items()}
        g = TR.fit_geometry(us)
        inv_us = np.linalg.inv(us_aff)

        def us_at(pw_mm, arr, order):
            vox = (pw_mm @ inv_us[:3, :3].T + inv_us[:3, 3]).T
            return map_coordinates(arr, vox, order=order, mode="constant", cval=0, output=np.float32)

        H_acc = {k: [] for k in ("real", "kidney_us", "valid")}
        for k in ("rig", "def"):
            for n in ("cart_lab", "cart_hu", "cart_z", "pol_lab", "pol_hu", "pol_z"):
                H_acc[f"{n}_{k}"] = []
        D_acc = {k: [] for k in KINDS}
        raw = {"raw_lab": [], "raw_hu": [], "raw_z": [], "raw_seed": []}
        seeds = []
        for n_pl, phi in enumerate(phis[e["kid"]]):
            T_wp = TR.T_world_from_probe_m(us_aff, *TR.plane_pose_vox(g, phi))
            seed = int(e["pid"]) * 1000 + (500 if e["side"] == "R" else 0) + n_pl
            seeds.append(seed)
            # full resolution: Cartesian real US, kidney mask, valid fan
            pc = world(T_wp, pts_cart)
            H_acc["real"].append(np.clip(np.round(us_at(pc, us, 1)), 0, 255).astype(np.uint8).reshape(NZ, NX))
            H_acc["kidney_us"].append((us_at(pc, us_mask.view(np.uint8), 0) > 0).reshape(NZ, NX))
            pv = (pc @ inv_us[:3, :3].T + inv_us[:3, 3])
            H_acc["valid"].append(TR.fan_valid(g, pv).reshape(NZ, NX) & fan_cart)
            pp = world(T_wp, pts_pol)
            for k, f in reg.items():
                lab_b, _, hu = H.sample_ct(vol, DF.map_points(f, T, pc.astype(np.float32)))
                lab_b = lab_b.reshape(NZ, NX); hu = hu.reshape(NZ, NX)
                H_acc[f"cart_lab_{k}"].append(lab_b)
                H_acc[f"cart_hu_{k}"].append(np.clip(np.round(hu), -1024, 3071).astype(np.int16))
                H_acc[f"cart_z_{k}"].append(H.relative_hu(hu, lab_b, vol, (H.MM, H.MM)).astype(np.float16))
                lab_b, lab_p, hu = H.sample_ct(vol, DF.map_points(f, T, pp.astype(np.float32)))
                lab_b = lab_b.reshape(args.lines, Sg); lab_p = lab_p.reshape(args.lines, Sg); hu = hu.reshape(args.lines, Sg)
                z = H.relative_hu(hu, lab_b, vol, sp_pol)
                lab_b, lab_p, hu, z, _ = leading_air_fill(lab_b, lab_p, hu, z)
                H_acc[f"pol_lab_{k}"].append(lab_b.astype(np.uint8))
                H_acc[f"pol_hu_{k}"].append(np.clip(np.round(hu), -1024, 3071).astype(np.int16))
                H_acc[f"pol_z_{k}"].append(z.astype(np.float16))
            # 128 px products with the deformable registration (trusted_prepare.py, step 4-5)
            ps = world(T_wp, pts_std)
            real = us_at(ps, us, 1).reshape(Lg, Sg)
            kus = (us_at(ps, us_mask.view(np.uint8), 0) > 0).reshape(Lg, Sg)
            ok = TR.fan_valid(g, ps @ inv_us[:3, :3].T + inv_us[:3, 3]).reshape(Lg, Sg)
            real = np.where(ok, real, 0.0)
            lab_b, lab_p, hu = H.sample_ct(vol, DF.map_points(field, T, ps.astype(np.float32)))
            lab_b = lab_b.reshape(Lg, Sg); lab_p = lab_p.reshape(Lg, Sg); hu = hu.reshape(Lg, Sg)
            z = relative_hu_std(hu, lab_b, vol)
            kct = lab_b == V.KIDNEY
            lab_b, lab_p, hu, z, _ = leading_air_fill(lab_b, lab_p, hu, z)
            cfg = replace(R.RenderConfig(), seed=seed)
            raw["raw_lab"].append(lab_b.astype(np.uint8)); raw["raw_z"].append(z.astype(np.float16))
            raw["raw_hu"].append(np.clip(np.round(hu), -1024, 3071).astype(np.int16)); raw["raw_seed"].append(seed)
            sz = (128, 128)
            frame = {"phys_b0": resize_img(render_from_maps(*maps_from_labels(lab_b, cfg), cfg, dev), 128),
                     "phys_m1": resize_img(render_from_maps(*maps_from_labels(lab_p, cfg), cfg, dev), 128),
                     "phys_m2": resize_img(render_from_maps(*maps_hu_modulated(lab_b, hu, z, vol.ref, cfg), cfg, dev), 128),
                     "lab_base": align_label_to_image(lab_b, sz).astype(np.uint8),
                     "lab_pp": align_label_to_image(lab_p, sz).astype(np.uint8),
                     "z": resize_float(z.T, 128),
                     "real": np.clip(np.round(resize_float(real.T, 128)), 0, 255).astype(np.uint8),
                     "kidney_us": align_label_to_image(kus.astype(np.uint8), sz).astype(np.uint8),
                     "valid": align_label_to_image(ok.astype(np.uint8), sz).astype(np.uint8),
                     "kidney_ct": align_label_to_image(kct.astype(np.uint8), sz).astype(np.uint8),
                     "hu_img": resize_float(((np.clip(hu, -200, 300) + 200) / 500.0).T, 128)}
            w = warp_frame(frame, KINDS)
            for k in KINDS:
                D_acc[k].append(w[k])
        m = {"dice_rigid": round(dice["rig"], 3), "dice_def": round(dice["def"], 3), "ldk_rigid_mm": ldk["rig"],
             "ldk_def_mm": ldk["def"], "sdf_mse_before": round(info["sdf_mse_before"], 2),
             "sdf_mse_after": round(info["sdf_mse_after"], 2), "jac_min": round(info["jac_min"], 3),
             "grad_max": round(info["grad_max"], 3), "u_max_mm": round(float(np.abs(field.u).max()), 1),
             "lam": args.lam, "lines": args.lines, "mm": H.MM}
        sessions = np.asarray([e["kid"]] * len(seeds))
        np.savez_compressed(out_h, sessions=sessions, seeds=np.asarray(seeds), meta=json.dumps(m),
                            **{k: np.stack(v) for k, v in H_acc.items()})
        out = {k: np.stack(v) for k, v in D_acc.items()}
        out["z"] = out["z"].astype(np.float16); out["hu_img"] = out["hu_img"].astype(np.float16)
        out.update({k: np.asarray(v) for k, v in raw.items()})
        km = {"dice_landmarks": round(dice0, 3), "dice_icp": round(dice1, 3), "used": "icp" if dice1 >= dice0 else "landmarks",
              "dice_def": m["dice_def"], "ldk_def_mm": ldk["def"], "ldk_rigid_mm": ldk["rig"], "n_planes": len(seeds)}
        np.savez_compressed(out_d, sessions=sessions,
                            meta=json.dumps({"kidney": km, "planes": [{"kid": e["kid"], "phi_deg": p} for p in phis[e["kid"]]]}),
                            **out)
        print(f"{e['kid']}: Dice rigid {m['dice_rigid']:.3f} -> deformable {m['dice_def']:.3f}, landmarks "
              f"{np.mean(ldk['rig']):.1f} -> {np.mean(ldk['def']):.1f} mm, jac min {m['jac_min']:.2f}, "
              f"{len(seeds)} planes, {time.time() - t0:.0f}s", flush=True)
    return 0


def merge(args, KINDS):
    files = sorted(f for f in os.listdir(args.shards_def) if f.endswith(".npz"))
    parts = [np.load(os.path.join(args.shards_def, f), allow_pickle=False) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in list(KINDS) + ["sessions"]}
    kid_meta, planes_meta = {}, []
    for f, p in zip(files, parts):
        mm = json.loads(str(p["meta"]))
        kid_meta[f[:-4]] = mm["kidney"]; planes_meta += mm["planes"]
    sessions = out.pop("sessions")
    meta = {"planes": planes_meta, "kidneys": kid_meta, "size": 128, "geometry": "kaggle sector",
            "registration": "rigid + deformable (kidney-mask signed distances)"}
    np.savez_compressed(args.out_def, sessions=sessions, meta=json.dumps(meta), **out)
    with open(os.path.splitext(args.out_def)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {len(sessions)} frames from {len(kid_meta)} kidneys -> {args.out_def}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
