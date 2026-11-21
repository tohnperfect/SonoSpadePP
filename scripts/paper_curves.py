#!/usr/bin/env python3
"""Paper Fig. 3: the learned HU-to-echogenicity curves of M4 (log backscatter factor vs relative HU),
for the tissues that dominate kidney planes; mean over the 5 folds x 3 seeds, band = min..max.
The HU-free control's per-tissue offsets are drawn as dashed horizontal lines.

  python scripts/paper_curves.py --out figures/curves.pdf
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/m4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="figures/curves.pdf")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from usbg import volume as V

    def load(mode):
        out = []
        for s in args.seeds:
            p = os.path.join(args.runs, f"{mode}_s{s}.json")
            if os.path.exists(p):
                out += list(json.load(open(p)).values())
        return out
    hu, const = load("hu"), load("const")
    if not hu:
        raise SystemExit("no M4 runs found")
    knots = np.asarray(hu[0]["knots"])
    zz = np.linspace(knots[0], knots[-1], 200)
    tissues = [(V.KIDNEY, "kidney"), (V.FAT, "fat"), (V.LIVER, "liver"), (V.MUSCLE, "muscle"),
               (V.SOFT_TISSUE, "other soft tissue"), (V.SPLEEN, "spleen")]
    colors = ["#1b6ca8", "#d1495b", "#2e8b57", "#8d6a9f", "#7a7a7a", "#e09f3e"]
    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    for (L, name), c in zip(tissues, colors):
        curves = np.stack([np.clip(st["a"][L] + np.interp(zz, knots, st["g"][L]), -2, 2) for st in hu])
        ax.plot(zz, curves.mean(0), color=c, lw=1.3, label=name)
        ax.fill_between(zz, curves.min(0), curves.max(0), color=c, alpha=0.10, lw=0)
        if const:
            ax.axhline(np.mean([st["a"][L] for st in const]), color=c, lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("relative HU z (robust SDs from the tissue median)", fontsize=7)
    ax.set_ylabel("log backscatter factor", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, handlelength=1.2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out); fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)
    print(f"wrote {args.out} from {len(hu)} learned fronts ({len(const)} controls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
