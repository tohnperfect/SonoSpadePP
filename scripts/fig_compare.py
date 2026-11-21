#!/usr/bin/env python3
"""Side-by-side figure: the same held-out poses through every variant, plus real Kaggle frames.

  python scripts/fig_compare.py --pairs data/pairs_eval.npz --seed 0 --out outputs/compare_panel.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs_eval.npz")
    ap.add_argument("--real-test", default="data/real_test.npz")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="outputs/compare_panel.png")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sonospadepp.labels import GAS
    from sonospadepp.variants import VARIANTS, Translator

    d = np.load(args.pairs, allow_pickle=False)
    lab_pp = d["lab_pp"]
    # frames that show what the CT adds: gas and CT vessels in view, one per case where possible
    score = (lab_pp == GAS).sum((1, 2)) + 3 * ((lab_pp == 3) & (d["lab_base"] == 1)).sum((1, 2))
    sess = d["sessions"]
    picks, seen = [], set()
    for i in np.argsort(-score):
        if sess[i] not in seen:
            picks.append(int(i)); seen.add(sess[i])
        if len(picks) == args.n:
            break
    for i in np.argsort(-score):                            # fewer cases than columns: fill by score
        if len(picks) == args.n:
            break
        if int(i) not in picks:
            picks.append(int(i))
    rows = [("base labels", d["lab_base"], "lab"), ("M1 labels", lab_pp, "lab"),
            ("relative HU tanh(z/3)", np.tanh(d["z"].astype(np.float32) / 3), "z"),
            ("physics (B0, M3 input)", d["phys_b0"] / 255.0, "img"),
            ("physics M1", d["phys_m1"] / 255.0, "img"), ("physics M2", d["phys_m2"] / 255.0, "img")]
    for v in ("b0", "m1", "m2", "m3"):
        ck = os.path.join(args.runs, f"{v}_s{args.seed}.pt")
        if not os.path.exists(ck):
            continue
        tr, cfg = Translator(ck), VARIANTS[v]
        out = np.zeros((len(sess), 128, 128), np.float32)
        out[picks] = tr.batch(d[cfg["phys"]][picks] / 255.0, d[cfg["lab"]][picks],
                              d["z"][picks].astype(np.float32) if cfg["hu"] else None)
        rows.append((cfg["title"], out, "img"))
    real = np.load(args.real_test, allow_pickle=False)["images"]
    fig, axes = plt.subplots(len(rows) + 1, len(picks), figsize=(1.9 * len(picks), 1.9 * (len(rows) + 1)),
                             squeeze=False)
    for r, (title, arr, kind) in enumerate(rows):
        for c, i in enumerate(picks):
            ax = axes[r][c]
            if kind == "lab":
                ax.imshow(arr[i], cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
            elif kind == "z":
                ax.imshow(arr[i], cmap="coolwarm", vmin=-1, vmax=1)
            else:
                ax.imshow(arr[i], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, fontsize=7)
            if r == 0:
                ax.set_title(str(sess[i]), fontsize=8)
    for c in range(len(picks)):
        ax = axes[-1][c]
        ax.imshow(real[c * 11 % len(real)], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("real Kaggle frames\n(other patients)", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
