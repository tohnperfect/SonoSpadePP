#!/usr/bin/env python3
"""Cut matching CT and US planes for every annotated TRUSTED kidney and render each variant's inputs.

Per kidney:
  1. CT volume: TotalSegmentator labels -> base labels, M1 labels, smoothed HU (build_pp_volume),
     cached in data/trusted/pp_cache/<pid>.npz. Labels come from TotalSegmentator, as they would
     for a new patient; the TRUSTED kidney masks are used only to register and to score.
  2. Registration US -> CT: Kabsch on the 7 landmark pairs, then trimmed rigid ICP between the
     STAPLE kidney-mask surfaces (register.py). Kidney Dice and landmark-1 residual are recorded.
  3. Probe geometry: fit the 3D fan (trusted.fit_geometry). Native planes at several sweep angles
     through the kidney (kidney area >= 50% of its maximum over the sweep), at most --planes.
  4. Per plane, the virtual probe sits on the real array apex along the real central beam. The same
     scan points sample the real US (trilinear), its kidney mask, the CT labels/HU (slice_all), so
     synthetic and real are pixel-aligned. Where the virtual probe face lies outside the CT body
     (patient position and probe pressure differ between the scans), the leading air on each scan
     line becomes soft tissue (coupling), so the beam is not blocked before it reaches the body.
  5. Physics renders (default RenderConfig, no randomization, one speckle seed per plane), resize to
     128x128 and warp to the Kaggle sector geometry with one set of coordinates, as make_pairs.py.

  python scripts/trusted_prepare.py --out data/trusted_pairs.npz [--kids 206R 206L] [--planes 7]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np


def leading_air_fill(lab_b, lab_p, hu, z, soft_hu=40.0):
    """Scan lines are rows (lateral) x samples (depth). Samples before the first non-air label on a
    line become soft tissue. Returns the arrays and the mean filled depth (samples)."""
    from usbg import volume as V
    lab_b, lab_p, hu, z = lab_b.copy(), lab_p.copy(), hu.copy(), z.copy()
    body = lab_b != V.BACKGROUND
    first = np.where(body.any(1), body.argmax(1), lab_b.shape[1])
    gaps = []
    for l, f in enumerate(first):
        if f > 0:
            lab_b[l, :f] = V.SOFT_TISSUE; lab_p[l, :f] = V.SOFT_TISSUE
            hu[l, :f] = soft_hu; z[l, :f] = 0.0
        gaps.append(f)
    return lab_b, lab_p, hu, z, float(np.mean(gaps))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kids", nargs="*", default=None, help="kidney ids like 206R (default: all with CT labels)")
    ap.add_argument("--totalseg", default="data/trusted/totalseg")
    ap.add_argument("--cache", default="data/trusted/pp_cache")
    ap.add_argument("--planes", type=int, default=7)
    ap.add_argument("--min-area-frac", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--shards", default="data/trusted/pairs", help="one .npz per kidney; existing ones are kept")
    ap.add_argument("--out", default="data/trusted_pairs.npz", help="all shards merged")
    args = ap.parse_args(argv)

    import nibabel as nib
    from usbg import render_lotus as R
    from usbg import volume as V
    from usbg import slicer as S
    from usbg.segmenter import align_label_to_image
    from sonospadepp import register as RG
    from sonospadepp import trusted as TR
    from sonospadepp.ctvol import build_pp_volume, load_pp, save_pp
    from sonospadepp.physics import maps_from_labels, maps_hu_modulated, render_from_maps
    from sonospadepp.slicing import slice_all
    from sonospadepp.warp import warp_frame
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # run with python -I
    from make_pairs import resize_img, resize_float

    dev = R.get_device()
    KINDS = {"phys_b0": "image", "phys_m1": "image", "phys_m2": "image", "lab_base": "label",
             "lab_pp": "label", "z": "float", "real": "image", "kidney_us": "label", "valid": "label",
             "kidney_ct": "label", "hu_img": "float"}
    os.makedirs(args.shards, exist_ok=True)
    entries = [e for e in TR.index() if (args.kids is None or e["kid"] in args.kids)]
    pts_probe, _ = S.scan_grid_probe()
    Lg, Sg = pts_probe.shape[:2]
    for e in entries:
        shard = os.path.join(args.shards, f"{e['kid']}.npz")
        if os.path.exists(shard):
            with np.load(shard, allow_pickle=False) as old:
                if "raw_lab" in old.files:              # older shards lack the raw slices M4 trains on
                    continue
        seg_dir = os.path.join(args.totalseg, e["pid"])
        # TotalSegmentator "total" writes 117 masks at the end of a run: wait for all of them
        if len([f for f in os.listdir(seg_dir) if f.endswith(".nii.gz")] if os.path.isdir(seg_dir) else []) < 117:
            print(f"{e['kid']}: no TotalSegmentator labels yet, skipped", flush=True)
            continue
        acc = {k: [] for k in KINDS}
        sessions, planes_meta = [], []
        raw = {"raw_lab": [], "raw_hu": [], "raw_z": [], "raw_seed": []}
        t0 = time.time()
        cpath = os.path.join(args.cache, f"{e['pid']}.npz")
        if os.path.exists(cpath):
            vol = load_pp(cpath)
        else:
            vol = build_pp_volume(e["ct_img"], seg_dir)
            save_pp(vol, cpath)

        # --- registration US -> CT (world mm) -------------------------------------------------
        us_n = nib.load(e["us_img"]); us_aff = us_n.affine
        us = np.asanyarray(us_n.dataobj)
        us_mask = np.asanyarray(nib.load(e["us_mask"]).dataobj) > 0
        ct_n = nib.load(e["ct_mask"]); ct_aff = ct_n.affine
        l_us, l_ct = np.loadtxt(e["us_ldk"]), np.loadtxt(e["ct_ldk"])
        ct_k = TR.ct_kidney_side(np.asanyarray(ct_n.dataobj), ct_aff, l_ct)
        R0, t0v = RG.kabsch(l_us, l_ct)
        T0 = RG.to_T(R0, t0v)
        dice0 = RG.mask_dice(us_mask, us_aff, ct_k, ct_aff, T0)
        T1, rms = RG.icp_rigid(RG.surface_points(us_mask, us_aff), RG.surface_points(ct_k, ct_aff), T0)
        dice1 = RG.mask_dice(us_mask, us_aff, ct_k, ct_aff, T1)
        T_ct_from_us = T1 if dice1 >= dice0 else T0
        lres = RG.apply_T(T_ct_from_us, l_us) - l_ct
        T_us_from_ct_m = np.linalg.inv(T_ct_from_us); T_us_from_ct_m[:3, 3] /= 1000.0

        # --- geometry and plane choice -------------------------------------------------------
        g = TR.fit_geometry(us)
        inv_us = np.linalg.inv(us_aff)
        cand = np.arange(-(g.sweep_half - 3), g.sweep_half - 3 + 1e-6, 1.0)
        areas, poses = [], []
        for phi in cand:
            T_wp = TR.T_world_from_probe_m(us_aff, *TR.plane_pose_vox(g, phi))
            pw = (pts_probe.reshape(-1, 3) @ T_wp[:3, :3].T + T_wp[:3, 3])
            pv = (pw * 1000.0) @ inv_us[:3, :3].T + inv_us[:3, 3]
            ok = TR.fan_valid(g, pv)
            km = TR.sample_world(us_mask.astype(np.uint8), us_aff, pw, order=0) > 0
            areas.append(int((km & ok).sum())); poses.append((T_wp, pw, pv, ok))
        areas = np.asarray(areas)
        good = np.nonzero(areas >= args.min_area_frac * areas.max())[0]
        pick = good[np.unique(np.round(np.linspace(0, len(good) - 1, min(args.planes, len(good)))).astype(int))]

        gaps = []
        for n_pl, i in enumerate(pick):
            T_wp, pw, pv, ok = poses[i]
            real = TR.sample_world(us.astype(np.float32), us_aff, pw, order=1).reshape(Lg, Sg)
            kus = (TR.sample_world(us_mask.astype(np.uint8), us_aff, pw, order=0) > 0).reshape(Lg, Sg)
            ok = ok.reshape(Lg, Sg)
            real = np.where(ok, real, 0.0)
            lab_b, lab_p, hu, z = slice_all(vol, T_wp, T_us_from_ct_m)
            kct = lab_b == V.KIDNEY
            lab_b, lab_p, hu, z, gap = leading_air_fill(lab_b, lab_p, hu, z)
            gaps.append(gap)
            cfg = replace(R.RenderConfig(), seed=int(e["pid"]) * 1000 + (500 if e["side"] == "R" else 0) + n_pl)
            # the scan-grid slices themselves, for training a learned acoustic front end (M4)
            raw["raw_lab"].append(lab_b.astype(np.uint8)); raw["raw_z"].append(z.astype(np.float16))
            raw["raw_hu"].append(np.clip(np.round(hu), -1024, 3071).astype(np.int16))
            raw["raw_seed"].append(int(cfg.seed))
            img_b = render_from_maps(*maps_from_labels(lab_b, cfg), cfg, dev)
            img_p = render_from_maps(*maps_from_labels(lab_p, cfg), cfg, dev)
            img_h = render_from_maps(*maps_hu_modulated(lab_b, hu, z, vol.ref, cfg), cfg, dev)
            sz = (args.size, args.size)
            frame = {
                "phys_b0": resize_img(img_b, args.size), "phys_m1": resize_img(img_p, args.size),
                "phys_m2": resize_img(img_h, args.size),
                "lab_base": align_label_to_image(lab_b, sz).astype(np.uint8),
                "lab_pp": align_label_to_image(lab_p, sz).astype(np.uint8),
                "z": resize_float(z.T, args.size),
                "real": np.clip(np.round(resize_float(real.T, args.size)), 0, 255).astype(np.uint8),
                "kidney_us": align_label_to_image(kus.astype(np.uint8), sz).astype(np.uint8),
                "valid": align_label_to_image(ok.astype(np.uint8), sz).astype(np.uint8),
                "kidney_ct": align_label_to_image(kct.astype(np.uint8), sz).astype(np.uint8),
                "hu_img": resize_float(((np.clip(hu, -200, 300) + 200) / 500.0).T, args.size),
            }
            w = warp_frame(frame, KINDS)
            for k in KINDS:
                acc[k].append(w[k])
            sessions.append(e["kid"])
            planes_meta.append({"kid": e["kid"], "phi_deg": float(cand[i]), "kidney_px": int(areas[i])})
        m = {
            "dice_landmarks": round(dice0, 3), "dice_icp": round(dice1, 3), "used": "icp" if dice1 >= dice0 else "landmarks",
            "icp_rms_mm": round(rms, 2), "ldk1_residual_mm": round(float(np.linalg.norm(lres[0])), 2),
            "ldk_rms_mm": round(float(np.sqrt((lres ** 2).sum(1).mean())), 2),
            "n_planes": int(len(pick)), "mean_air_fill_samples": round(float(np.mean(gaps)), 1),
            "geometry": {k: round(float(v), 2) for k, v in g.__dict__.items()},
            "ref_hu": vol.meta.get("ref_hu"), "cyst_vox": vol.meta.get("cyst_vox")}
        print(f"{e['kid']}: Dice landmarks {m['dice_landmarks']:.3f} -> ICP {m['dice_icp']:.3f}, "
              f"ldk1 {m['ldk1_residual_mm']:.1f} mm, {m['n_planes']} planes, air fill "
              f"{m['mean_air_fill_samples']:.0f} samples, {time.time() - t0:.0f}s", flush=True)

        out = {k: np.stack(v) for k, v in acc.items()}
        out["z"] = out["z"].astype(np.float16); out["hu_img"] = out["hu_img"].astype(np.float16)
        out.update({k: np.asarray(v) for k, v in raw.items()})
        np.savez_compressed(shard, sessions=np.asarray(sessions),
                            meta=json.dumps({"kidney": m, "planes": planes_meta}), **out)

    # merge every shard present
    files = sorted(f for f in os.listdir(args.shards) if f.endswith(".npz"))
    parts = [np.load(os.path.join(args.shards, f), allow_pickle=False) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in list(KINDS) + ["sessions"]}
    kid_meta, planes_meta = {}, []
    for f, p in zip(files, parts):
        mm = json.loads(str(p["meta"]))
        kid_meta[f[:-4]] = mm["kidney"]; planes_meta += mm["planes"]
    sessions = out.pop("sessions")
    meta = {"planes": planes_meta, "kidneys": kid_meta, "size": args.size, "geometry": "kaggle sector",
            "render": "default RenderConfig, seed per plane"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, sessions=sessions, meta=json.dumps(meta), **out)
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {len(sessions)} frames from {len(kid_meta)} kidneys -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
