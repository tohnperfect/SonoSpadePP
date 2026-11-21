#!/usr/bin/env python3
"""Qualitative figure of the paper (figures/qual.pdf): the variants of this work on held-out
simulator content, in place of the label-free baselines that fig_qual_v2.py shows.

Content: SonoSpadePP's held-out simulator set in the fan geometry that the paired models are trained on
(data/dataset/pairs_spp_eval200.npz, public CT only; the content of the tier-1 side-effect check).
Rows are chosen by the fixed rule of fig_qual_v2.py: liver and kidney each at least --min-frac of the fan, distinct
CT cases, largest min(liver, kidney) share. Columns, one lineage of models (seed 0 each): the label map, the physics
render, SonoSPADE (b0), tier 0 = SonoSPADE trained with the USFM segmenter (b0v2) and tier 1 = b0 fine-tuned on
paired planes of other patients (fold 0, as in the side-effect check). Tier 2 has no column: it needs a real image
of the patient, which the simulator CTs do not have (see fig_trusted_v2.py). The last column shows real evaluation
frames of other people (not paired with the content), the frames with the largest annotated liver share. Each image
carries an enlarged inset (3x, nearest neighbour) of a 24 x 24 px crop centred deep inside the liver.
Grayscale shown as is, [0, 1] mapped linearly (no gamma). CPU only.

  python -I scripts/v2/fig_variants_v2.py [--pp-root /path/to/working-copy]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

LIVER, KIDNEY = 1, 4
CROP, ZOOM = 24, 3
BOX = "#eb6834"                                      # categorical slot 2 (validated palette), marks the crop


def stages_def(root, seed):
    b0 = os.path.join(root, f"runs/b0_s{seed}.pt")
    return [("Physics render", "physics", None),
            ("SonoSPADE", "sonospade", b0),
            ("Tier 0 (USFM $S_{\\mathrm{US}}$)", "sonospade", os.path.join(root, f"runs/b0v2_s{seed}.pt")),
            ("Tier 1", "paired_lab", b0 + "::" + os.path.join(root, f"runs/m7/lab_s{seed}_f0.pt"))]


def pick(labs, sessions, fan, n, min_frac, kidney=True):
    """Largest min(liver, kidney) share of the fan among frames with both at least min_frac (kidney=False: largest
    liver share), at most one frame per CT case."""
    cand = []
    for i, lab in enumerate(labs):
        area = max(int(fan[i].sum()), 1)
        fl, fk = (lab == LIVER).sum() / area, (lab == KIDNEY).sum() / area
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
    return int(np.clip(y - CROP // 2, 0, h - CROP)), int(np.clip(x - CROP // 2, 0, w - CROP))


def label_rgb(lab):
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    rgb = np.zeros(lab.shape + (3,), np.float32)
    for c in np.unique(lab):
        if c:
            rgb[lab == c] = cmap(int(c) % 20)[:3]
    return rgb


def show(ax, img, y0, x0, win, gray=True, corner="right"):
    """img: the whole frame; win = (ya, yb, xa, xb), the part shown; (y0, x0): the crop, in frame coordinates."""
    import matplotlib.patches as patches
    ya, yb, xa, xb = win
    h, w = yb - ya, xb - xa
    ax.imshow(img[ya:yb, xa:xb], cmap="gray" if gray else None, vmin=0, vmax=1, interpolation="nearest")
    ax.add_patch(patches.Rectangle((x0 - xa - 0.5, y0 - ya - 0.5), CROP, CROP, fill=False, ec=BOX, lw=0.8))
    if gray:                                          # inset in a lower corner
        z = np.kron(img[y0:y0 + CROP, x0:x0 + CROP], np.ones((ZOOM, ZOOM)))
        s = 0.36 * w
        xl = w - s - 0.5 if corner == "right" else -0.5
        ax.imshow(z, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=(xl, xl + s, h - 0.5, h - s - 0.5))
        ax.add_patch(patches.Rectangle((xl, h - s - 0.5), s, s, fill=False, ec=BOX, lw=0.8))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp-root", default=os.environ.get("SONOSPADEPP_ROOT", "."),
                    help="working copy that holds the same-patient runs, outputs and TRUSTED data (default: this one)")
    ap.add_argument("--pairs", default="data/dataset/pairs_spp_eval200.npz")
    ap.add_argument("--real-pairs", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--min-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=float, default=7.0, help="figure width in inches")
    ap.add_argument("--out", default="figures/qual.pdf")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sonospadepp.v2.evaldata import Stage, load_pairs

    phys, labs, sess = load_pairs(a.pairs, None)
    real, rlabs, _ = load_pairs(a.real_pairs, None)
    fan = phys > 0
    rows = pick(labs, sess, fan, a.rows, a.min_frac)
    sdef = stages_def(a.pp_root, a.seed)
    ncol = len(sdef) + 2
    rrows = pick(rlabs, None, np.ones_like(rlabs, bool), len(rows), 0.05, kidney=False)
    stages = [(t, Stage(k, c, device="cpu")) for t, k, c in sdef]
    outs = {t: [st(phys[i], labs[i]) for i in rows] for t, st in stages}

    # the part of the frame that the fan of the shown rows covers
    any_fan = np.any(fan[rows], axis=0)
    ys, xs = np.flatnonzero(any_fan.any(1)), np.flatnonzero(any_fan.any(0))
    win = (max(ys[0] - 1, 0), min(ys[-1] + 2, phys.shape[1]), max(xs[0] - 1, 0), min(xs[-1] + 2, phys.shape[2]))
    side = max(win[1] - win[0], win[3] - win[2])      # a square window, as the real frames are square
    ya = int(np.clip(win[0] - (side - (win[1] - win[0])) // 2, 0, phys.shape[1] - side))
    xa = int(np.clip(win[2] - (side - (win[3] - win[2])) // 2, 0, phys.shape[2] - side))
    win = (ya, ya + side, xa, xa + side)
    hc, wc = side, side
    hr, wr = real.shape[1:]
    full = (0, hr, 0, wr)

    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    pw = a.width / ncol
    fig, axes = plt.subplots(len(rows), ncol, figsize=(a.width, pw * len(rows) * hc / wc + 0.25), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.04})
    for r, i in enumerate(rows):
        y0, x0 = crop_center(labs[i])
        corner = "left" if x0 + CROP / 2 > (win[2] + win[3]) / 2 else "right"
        show(axes[r, 0], label_rgb(labs[i]), y0, x0, win, gray=False)
        for c, (t, _) in enumerate(stages, start=1):
            show(axes[r, c], outs[t][r], y0, x0, win, corner=corner)
        j = rrows[r]
        show(axes[r, ncol - 1], real[j], *crop_center(rlabs[j]), full)
        axes[r, 0].set_ylabel(f"content {r + 1}", fontsize=7)
    for c, t in enumerate(["Label map"] + [t for t, _ in stages] + ["Real (other person)"]):
        axes[0, c].set_title(t, fontsize=7, pad=3)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("content rows", rows, "sessions", [str(sess[i]) for i in rows], "real frames", rrows,
          "window", win, "size_in", [round(float(v), 2) for v in fig.get_size_inches()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
