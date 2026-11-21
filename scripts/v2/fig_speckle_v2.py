#!/usr/bin/env python3
"""Figure: window-level log Nakagami m per tissue, real frames against texture stages (figures/speckle.pdf).

Same windows as the speckle endpoint of eval_v2.py (7 px, stride 3, erode 1, DR 60 dB, inside the real pool's fan;
real masks = the 60 annotated frames' masks, synthetic masks = the input labels of the first 200 content pairs).
A stage with several checkpoints (seeds) contributes the windows of all of them. Runs on the CPU, so it can share
the machine with a training job.

  python -I scripts/v2/fig_speckle_v2.py [--stage "This work=sonospade_v2:runs/.../a_s0.pt,runs/.../a_s1.pt"]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

TISSUES = ((1, "Liver"), (4, "Kidney"))      # the tissues with enough real windows
NK = dict(win=7, stride=3, erode=1)
# validated categorical slots 1-3 (dataviz reference palette, light mode); real frames in neutral ink
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
STYLES = ["-", "--", "-."]
INK, INK2 = "#2b2b29", "#6b6a64"


def parse_stage(s):
    label, rest = s.split("=", 1)
    kind, _, ckpts = rest.partition(":")
    return label, kind, [c for c in ckpts.split(",") if c] or [None]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--real-pairs", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--pool", default="data/real/kaggle_vitale/pool")
    ap.add_argument("--dr", type=float, default=60.0)
    ap.add_argument("--stage", action="append", default=None,
                    help='"Label=kind:ckpt1,ckpt2"; default: physics render and the three SonoSPADE retrains')
    ap.add_argument("--out", default="figures/speckle.pdf")
    ap.add_argument("--summary", default="results/v2/fig_speckle.json")
    a = ap.parse_args(argv)
    stages = [parse_stage(s) for s in (a.stage or [
        "Physics render=physics:",
        "SonoSPADE=sonospade:" + ",".join(f"runs/v2/v1_repro/sonospade_s{s}.pt" for s in range(3))])]
    if len(stages) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} stages (validated palette slots)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    from sonospadepp.v2 import metrics as M
    from sonospadepp.v2.evaldata import Stage, fan_mask, load_pairs, load_pool

    phys, labs, _ = load_pairs(a.pairs, a.limit)
    real, rlabs, _ = load_pairs(a.real_pairs, None)
    fan = fan_mask(load_pool(a.pool))

    def logm(frames, labels, t):
        w = [M.nakagami_windows(img, (lab == t) & fan, dr_db=a.dr, **NK)[0] for img, lab in zip(frames, labels)]
        m = np.concatenate(w) if w else np.empty(0)
        return np.log(m[np.isfinite(m) & (m > 0)])

    vals = {"Real": {name: logm(real, rlabs, t) for t, name in TISSUES}}
    for label, kind, ckpts in stages:
        per = {name: [] for _, name in TISSUES}
        for ck in ckpts:
            st = Stage(kind, ck, device="cpu")
            trans = [st(p, l) for p, l in zip(phys, labs)]
            for t, name in TISSUES:
                per[name].append(logm(trans, labs, t))
        vals[label] = {k: np.concatenate(v) for k, v in per.items()}

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(TISSUES), figsize=(5.4, 2.4), constrained_layout=True)
    summary = {}
    for ax, (_, name) in zip(axes, TISSUES):
        q = np.array([np.percentile(vals[k][name], [0.5, 99.5]) for k in vals])
        lo, hi = q[:, 0].min(), q[:, 1].max()
        x = np.linspace(lo, hi, 400)
        r = vals["Real"][name]
        kr = gaussian_kde(r)(x)
        ax.fill_between(x, kr, color=INK, alpha=0.12, lw=0)
        ax.plot(x, kr, color=INK, lw=2, label="Real frames")
        summary[name] = {"Real": {"n": int(len(r)), "median_logm": float(np.median(r))}}
        for i, (label, _, _) in enumerate(stages):
            v = vals[label][name]
            ax.plot(x, gaussian_kde(v)(x), color=COLORS[i], ls=STYLES[i], lw=2, label=label)
            summary[name][label] = {"n": int(len(v)), "median_logm": float(np.median(v)),
                                    "w1_logm": float(M.w1_1d(r, v))}
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)       # headroom for the window counts
        ax.axvline(0.0, color=INK2, lw=0.8, ls=":")
        ax.annotate("m = 1", (0.0, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    color=INK2, fontsize=7)
        ax.set_title(name, color=INK, fontsize=9, loc="left")
        ax.text(0.99, 0.97, f"real windows: {len(r)}\n" + "\n".join(
            f"{lb}: {len(vals[lb][name])}" for lb, _, _ in stages), transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=INK2, linespacing=1.1)
        ax.set_xlabel(r"window-level $\log m$")
        ax.set_yticks([])
        ax.grid(False)
    axes[0].set_ylabel("density")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="outside upper center", ncol=len(lb), frameon=False, fontsize=7.5)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200)
    with open(a.summary, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
