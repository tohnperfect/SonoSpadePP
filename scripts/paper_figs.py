#!/usr/bin/env python3
"""Paper Fig. 2: the same inputs through every variant (GAN seed 0).

Left block: held-out TotalSegmentator poses (unpaired; last row = real frames of other people).
Right block: TRUSTED planes (paired; last row = the real ultrasound of the same patient and plane,
red = its kidney mask, cyan = the registered CT kidney).

  python scripts/paper_figs.py --pairs data/pairs_eval.npz --trusted data/trusted_pairs.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs_eval.npz")
    ap.add_argument("--real-test", default="data/real_test.npz")
    ap.add_argument("--trusted", default="data/trusted_pairs.npz")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--m4", default="outputs/m4")
    ap.add_argument("--out", default="figures/qualitative.pdf")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sonospadepp.labels import GAS
    from sonospadepp.variants import VARIANTS, Translator

    trs = {v: Translator(os.path.join(args.runs, f"{v}_s{args.seed}.pt")) for v in ("b0", "m1", "m2", "m3")}

    def translate(d, idx):
        out = {"render": d["phys_b0"][idx] / 255.0}
        for v, tr in trs.items():
            c = VARIANTS[v]
            out[v] = tr.batch(d[c["phys"]][idx] / 255.0, d[c["lab"]][idx],
                              d["z"][idx].astype(np.float32) if c["hu"] else None)
        return out

    # unpaired picks: frames showing CT-derived structure (gas, liver vessels), one per CT
    d = np.load(args.pairs, allow_pickle=False)
    score = (d["lab_pp"] == GAS).sum((1, 2)) + 3 * ((d["lab_pp"] == 3) & (d["lab_base"] == 1)).sum((1, 2))
    picks, seen = [], set()
    for i in np.argsort(-score):
        if d["sessions"][i] not in seen:
            picks.append(int(i)); seen.add(d["sessions"][i])
        if len(picks) == args.n:
            break
    left = translate(d, picks)
    real = np.load(args.real_test, allow_pickle=False)["images"]
    left["real"] = np.stack([real[(7 * c + 3) % len(real)] / 255.0 for c in range(len(picks))])

    # paired picks: largest kidney cross-section, one plane per kidney, different patients
    t = np.load(args.trusted, allow_pickle=False)
    kid = t["kidney_us"].sum((1, 2))
    tp, seen = [], set()
    for i in np.argsort(-kid):
        p = str(t["sessions"][i])[:-1]
        if p not in seen:
            tp.append(int(i)); seen.add(p)
        if len(tp) == args.n:
            break
    for i in np.argsort(-kid):                              # fewer patients than columns
        if len(tp) == args.n:
            break
        if int(i) not in tp and str(t["sessions"][i]) not in {str(t["sessions"][k]) for k in tp}:
            tp.append(int(i))
    for i in np.argsort(-kid):
        if len(tp) == args.n:
            break
        if int(i) not in tp:
            tp.append(int(i))
    right = translate(t, tp)
    right["real"] = t["real"][tp] / 255.0
    m4p = os.path.join(args.m4, f"preds_hu_s{args.seed}.npz")
    if os.path.exists(m4p):
        pr = np.load(m4p, allow_pickle=False)
        assert pr["sessions"].tolist() == t["sessions"].tolist()
        right["m4"] = pr["gen"][tp].astype(np.float32)

    rows = [("render", "physics render"), ("b0", "B0 labels only"), ("m1", "M1 sub-labels"),
            ("m2", "M2 HU physics"), ("m3", "M3 HU channel")] + ([("m4", "M4 learned HU-to-echo")] if "m4" in right else []) \
           + [("real", "real ultrasound")]
    nc = 2 * args.n
    fig, ax = plt.subplots(len(rows), nc, figsize=(0.95 * nc + 0.6, 0.95 * len(rows) + 0.3), squeeze=False)
    for r, (key, title) in enumerate(rows):
        for c in range(nc):
            a = ax[r][c]
            src, j = (left, c) if c < args.n else (right, c - args.n)
            if key not in src:                               # M4 exists only for TRUSTED planes
                a.set_facecolor("#eeeeee"); a.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=6, color="#777777")
            else:
                a.imshow(src[key][j], cmap="gray", vmin=0, vmax=1)
            if c >= args.n and key in src:
                i = tp[j]
                a.contour(t["kidney_us"][i] > 0, [0.5], colors="r", linewidths=0.5)
                a.contour(t["kidney_ct"][i] > 0, [0.5], colors="c", linewidths=0.5)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if c == 0:
                a.set_ylabel(title, fontsize=6)
    ax[0][0].set_title("held-out CTs (unpaired)", fontsize=6, loc="left")
    ax[0][args.n].set_title("TRUSTED planes (same patient)", fontsize=6, loc="left")
    ax[-1][0].set_xlabel("other people", fontsize=5)
    ax[-1][args.n].set_xlabel("same patient, same plane", fontsize=5)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.96, bottom=0.03, wspace=0.03, hspace=0.03)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=160)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
