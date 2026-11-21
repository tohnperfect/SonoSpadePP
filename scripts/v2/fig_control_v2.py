#!/usr/bin/env python3
"""Figure 4: per-tissue control by label (figures/control.pdf).

The label-swap protocol of the locality metric (sonospadepp.v2.metrics.label_swap_locality): on one content frame
(the first frame Fig. 3 shows), the kidney region is relabeled as liver, the reference tissue, with the physics
render unchanged. Rows: SonoSPADE (retrained, seed 0) and the full model of this work (seed 0). Columns: output with
the original labels, output with the kidney relabeled, and the absolute difference (single-hue scale, shared by both
rows), with the kidney outline (solid) and the 8 px band excluded from "outside" (dashed). Panel titles give that
frame's locality with the band (mean change inside / mean change outside). CPU only.

  python -I scripts/v2/fig_control_v2.py [--frame 29]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

STAGES = [("SonoSPADE", "sonospade", "runs/v2/v1_repro/sonospade_s0.pt"),
          ("This work", "sonospade_v2", "runs/v2/wp3/gen_sususfm_s0.pt")]
LIVER, KIDNEY = 1, 4
OUTLINE = "#eb6834"                                   # categorical slot 2 (validated palette)
# single-hue sequential ramp (the validated blue ramp, light to dark) for the difference
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--frame", type=int, default=29)
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--v2-label", default="This work", help="panel label of the SonoSPADE + USFM segmenter model")
    ap.add_argument("--out", default="figures/control.pdf")
    ap.add_argument("--only-v1", action="store_true", help="one row: SonoSPADE only, without the USFM-segmenter model")
    a = ap.parse_args(argv)
    stages_def = STAGES[:1] if a.only_v1 else STAGES[:-1] + [(a.v2_label,) + STAGES[-1][1:]]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy import ndimage
    from sonospadepp.v2 import metrics as M
    from sonospadepp.v2.evaldata import Stage, load_pairs

    phys, labs, _ = load_pairs(a.pairs, a.frame + 1)
    img, lab = phys[a.frame], labs[a.frame]
    region = lab == KIDNEY
    sw = lab.copy()
    sw[region] = LIVER
    band = ndimage.binary_dilation(region, iterations=a.band) & ~region
    res = []
    for title, kind, ck in stages_def:
        st = Stage(kind, ck, device="cpu")
        base, alt = st(img, lab), st(img, sw)
        res.append((title, base, alt, np.abs(alt - base), M.locality_band(base, alt, region, band=a.band)["locality"]))
    vmax = float(np.percentile(np.concatenate([r[3].ravel() for r in res]), 99.5))
    cmap = LinearSegmentedColormap.from_list("blue_seq", RAMP)

    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(res), 4, figsize=(4.9, 1.62 * len(res) + 0.3), squeeze=False,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.22, "width_ratios": [1, 1, 1, 0.06]})
    for r, (title, base, alt, diff, loc) in enumerate(res):
        axes[r, 0].imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 1].imshow(alt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        im = axes[r, 2].imshow(diff, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        for c in range(3):
            axes[r, c].contour(region, levels=[0.5], colors=OUTLINE, linewidths=0.8)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
        axes[r, 2].contour(region | band, levels=[0.5], colors=OUTLINE, linewidths=0.6, linestyles="dashed")
        axes[r, 0].set_ylabel(title, fontsize=7)
        axes[r, 2].text(0.03, 0.97, f"locality {loc:.1f}", transform=axes[r, 2].transAxes, ha="left", va="top",
                        fontsize=6.5, color="#2b2b29")
        cb = fig.colorbar(im, cax=axes[r, 3])
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
    axes[0, 0].set_title("original labels", fontsize=7, pad=2)
    axes[0, 1].set_title("kidney relabeled", fontsize=7, pad=2)
    axes[0, 2].set_title("|difference|", fontsize=7, pad=2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    print({t: round(float(l), 2) for t, _, _, _, l in res})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
