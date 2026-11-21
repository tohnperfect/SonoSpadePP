#!/usr/bin/env python3
"""Figure 3: qualitative comparison on held-out content, with the label-free baselines (figures/qual_baselines.pdf).

Rows: three content frames of the fixed evaluation set (data/dataset/pairs_med.npz, first 200 pairs), chosen by a
fixed rule: liver and kidney each at least 3% of the frame, distinct CT cases, largest min(liver, kidney) share.
Columns: the label map, the physics render, CUT, CycleGAN, S-CycleGAN, SonoSPADE (retrained, seed 0) and the full
model of this work (seed 0). A last row shows real evaluation frames of other people (not paired with the content),
the frames with the largest annotated liver share. Each image carries an enlarged inset (3x, nearest neighbour) of a
24 x 24 px crop centred deep inside the liver (the maximum of the liver's distance transform, the frame border
counting as a boundary), marked by a box.
Grayscale shown as is, [0, 1] mapped linearly (no gamma). CPU only.

  python -I scripts/v2/fig_qual_v2.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np

STAGES = [("Physics render", "physics", None), ("CUT", "cut", "data/v1_ckpt/exp_real_cut.pt"),
          ("CycleGAN", "cyclegan", "data/v1_ckpt/exp_real_cyclegan.pt"),
          ("S-CycleGAN", "cyclegan_sc", "data/v1_ckpt/exp_real_cyclegan_sc.pt"),
          ("SonoSPADE", "sonospade", "runs/v2/v1_repro/sonospade_s0.pt"),
          ("This work", "sonospade_v2", "runs/v2/wp3/gen_sususfm_s0.pt")]
LIVER, KIDNEY = 1, 4
CROP, ZOOM = 24, 3
BOX = "#eb6834"                                      # categorical slot 2 (validated palette), marks the crop


def pick(labs, sessions, n=3, min_frac=0.05, kidney=True):
    """Largest min(liver, kidney) share among frames with both at least min_frac (kidney=False: largest liver
    share), at most one frame per CT case."""
    cand = []
    for i, lab in enumerate(labs):
        fl, fk = (lab == LIVER).mean(), (lab == KIDNEY).mean()
        if kidney and fl >= min_frac and fk >= min_frac:
            cand.append((min(fl, fk), i))
        elif not kidney and fl >= min_frac:
            cand.append((fl, i))
    out, seen = [], set()
    for _, i in sorted(cand, reverse=True):
        s = sessions[i] if sessions is not None else i
        if s not in seen:
            out.append(i)
            seen.add(s)
        if len(out) == n:
            break
    return out


def crop_center(lab):
    from scipy import ndimage
    m = np.pad(lab == LIVER, 1)                       # the frame border counts as a boundary
    d = ndimage.distance_transform_edt(m)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(d), d.shape)
    h, w = lab.shape
    y0 = int(np.clip(y - CROP // 2, 0, h - CROP))
    x0 = int(np.clip(x - CROP // 2, 0, w - CROP))
    return y0, x0


def label_rgb(lab):
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    rgb = np.zeros(lab.shape + (3,), np.float32)
    for c in np.unique(lab):
        if c == 0:
            continue
        rgb[lab == c] = cmap(int(c) % 20)[:3]
    return rgb


def show(ax, img, y0, x0, gray=True):
    import matplotlib.patches as patches
    h, w = img.shape[:2]
    ax.imshow(img, cmap="gray" if gray else None, vmin=0, vmax=1, interpolation="nearest")
    ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), CROP, CROP, fill=False, ec=BOX, lw=0.8))
    if gray:                                         # inset in the lower-right corner
        z = np.kron(img[y0:y0 + CROP, x0:x0 + CROP], np.ones((ZOOM, ZOOM)))
        s = 0.36 * w                                   # inset spans about a third of the panel
        ax.imshow(z, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                  extent=(w - s - 0.5, w - 0.5, h - 0.5, h - s - 0.5))
        ax.add_patch(patches.Rectangle((w - s - 0.5, h - s - 0.5), s, s, fill=False, ec=BOX, lw=0.8))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--real-pairs", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--v2-label", default="This work", help="panel label of the SonoSPADE + USFM segmenter model")
    ap.add_argument("--out", default="figures/qual_baselines.pdf")
    a = ap.parse_args(argv)
    stages_def = STAGES[:-1] + [(a.v2_label,) + STAGES[-1][1:]]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sonospadepp.v2.evaldata import Stage, load_pairs

    phys, labs, sess = load_pairs(a.pairs, a.limit)
    real, rlabs, _ = load_pairs(a.real_pairs, None)
    rows = pick(labs, sess, min_frac=0.03)
    rrows = pick(rlabs, None, n=len(stages_def) + 1, min_frac=0.05, kidney=False)
    stages = [(t, Stage(k, c, device="cpu")) for t, k, c in stages_def]
    outs = {t: [st(phys[i], labs[i]) for i in rows] for t, st in stages}

    ncol = len(stages_def) + 1
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(rows) + 1, ncol, figsize=(7.0, 7.0 * (len(rows) + 1) / ncol + 0.25),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    for r, i in enumerate(rows):
        y0, x0 = crop_center(labs[i])
        show(axes[r, 0], label_rgb(labs[i]), y0, x0, gray=False)
        for c, (t, _) in enumerate(stages, start=1):
            show(axes[r, c], outs[t][r], y0, x0)
    for c, (t, _) in enumerate([("Label map", None)] + stages):
        axes[0, c].set_title(t, fontsize=7, pad=3)
    for c, j in enumerate(rrows[:ncol]):
        y0, x0 = crop_center(rlabs[j])
        show(axes[len(rows), c], real[j], y0, x0)
    for c in range(len(rrows), ncol):
        axes[len(rows), c].axis("off")
    axes[len(rows), 0].set_ylabel("real frames\n(other people)", fontsize=7)
    for r in range(len(rows)):
        axes[r, 0].set_ylabel(f"content {r + 1}", fontsize=7)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("content rows", rows, "sessions", [str(sess[i]) for i in rows], "real frames", rrows[:ncol])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
