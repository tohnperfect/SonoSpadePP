#!/usr/bin/env python3
"""Thumbnails of the paper's overview figure (Fig. 1, written to figures/overview_*.pdf): one CT slice, the label slice, the
physics render and the translated frame of one held-out simulator frame.

The simulator frame is the first row of the qualitative figure (fig_variants_v2.py: same content set and the same
rule), translated by SonoSPADE, seed 0. The CT slice is the
axial slice of the same public CT through the centre of the kidney, shown in a soft-tissue window; it illustrates
the input and is not the probe plane. Public CT only; no real ultrasound image.
Each thumbnail is written as a one-image PDF. CPU only.

  .venv/bin/python -I scripts/v2/fig_overview_assets.py [--pp-root .] [--out-dir figures]
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _variants():
    spec = importlib.util.spec_from_file_location("fig_variants_v2", os.path.join(HERE, "fig_variants_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(path, img, gray=True):
    import matplotlib.pyplot as plt
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(1.0, 1.0 * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray" if gray else None, vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def ct_slice(case_dir, lo=-160.0, hi=240.0, margin=6):
    """Axial slice through the centre of the right kidney (left if absent): soft-tissue window, anterior up,
    isotropic pixels, cropped to a square around the body (the table is removed)."""
    import nibabel as nib
    from scipy import ndimage
    ct = nib.as_closest_canonical(nib.load(os.path.join(case_dir, "ct.nii.gz")))
    vol = np.asarray(ct.dataobj, dtype=np.float32)
    k = vol.shape[2] // 2
    for name in ("kidney_right", "kidney_left"):
        p = os.path.join(case_dir, "segmentations", name + ".nii.gz")
        if os.path.exists(p):
            seg = np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj) > 0
            if seg.any():
                k = int(np.round(np.nonzero(seg)[2].mean()))
                break
    dx, dy = (float(v) for v in ct.header.get_zooms()[:2])
    hu = ndimage.zoom(vol[:, :, k].T[::-1], (dy / dx, 1.0), order=1)           # rows: anterior to posterior
    body = ndimage.binary_opening(hu > -300, structure=np.ones((9, 9), bool))
    lab, n = ndimage.label(body)
    if n:
        body = lab == 1 + int(np.argmax(ndimage.sum(body, lab, range(1, n + 1))))
    body = ndimage.binary_fill_holes(body)
    ys, xs = np.flatnonzero(body.any(1)), np.flatnonzero(body.any(0))
    side = max(ys[-1] - ys[0], xs[-1] - xs[0]) + 1 + 2 * margin
    cy, cx = (ys[0] + ys[-1]) // 2, (xs[0] + xs[-1]) // 2
    out = np.zeros((side, side), np.float32)
    img = np.where(body, np.clip((hu - lo) / (hi - lo), 0, 1), 0.0)
    for r in range(side):
        y = cy - side // 2 + r
        if 0 <= y < img.shape[0]:
            x0 = cx - side // 2
            a, b = max(x0, 0), min(x0 + side, img.shape[1])
            out[r, a - x0:b - x0] = img[y, a:b]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp-root", default=os.environ.get("SONOSPADEPP_ROOT", "."))
    ap.add_argument("--pairs", default="data/dataset/pairs_spp_eval200.npz")
    ap.add_argument("--ct-root", default="data/raw/totalseg_small")
    ap.add_argument("--min-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    from sonospadepp.v2.evaldata import Stage, load_pairs

    V = _variants()
    phys, labs, sess = load_pairs(a.pairs, None)
    fan = phys > 0
    i = V.pick(labs, sess, fan, 1, a.min_frac)[0]
    out = Stage("sonospade", os.path.join(a.pp_root, f"runs/b0_s{a.seed}.pt"), device="cpu")(phys[i], labs[i])

    ys, xs = np.flatnonzero(fan[i].any(1)), np.flatnonzero(fan[i].any(0))      # the square window that holds the fan
    side = max(ys[-1] - ys[0], xs[-1] - xs[0]) + 3
    ya = int(np.clip(ys[0] - 1 - (side - (ys[-1] - ys[0] + 3)) // 2, 0, phys.shape[1] - side))
    xa = int(np.clip(xs[0] - 1 - (side - (xs[-1] - xs[0] + 3)) // 2, 0, phys.shape[2] - side))
    w = np.s_[ya:ya + side, xa:xa + side]

    os.makedirs(a.out_dir, exist_ok=True)
    save(os.path.join(a.out_dir, "overview_label.pdf"), V.label_rgb(labs[i])[w], gray=False)
    save(os.path.join(a.out_dir, "overview_phys.pdf"), phys[i][w])
    save(os.path.join(a.out_dir, "overview_out.pdf"), out[w])
    case = os.path.join(a.ct_root, str(sess[i]))
    pad = ct_slice(case)
    save(os.path.join(a.out_dir, "overview_ct.pdf"), pad)
    print("content frame", int(i), "session", str(sess[i]), "window", (ya, xa, int(side)),
          "CT slice shape", pad.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
