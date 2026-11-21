#!/usr/bin/env python3
"""Aggregate outputs/compare.json over GAN seeds into a markdown table (mean +- SD across seeds).

  python scripts/summarize.py --results outputs/compare.json --out docs/RESULTS_TABLE.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

TITLES = {"phys_b0": "physics render, base labels", "phys_m1": "physics render, M1 labels",
          "phys_m2": "physics render, HU-modulated",
          "b0": "B0 SonoSPADE (labels + physics)", "m1": "M1 CT sub-labels",
          "m2": "M2 HU-modulated physics", "m3": "M3 HU as a SPADE channel"}
ORGANS = {1: "liver", 2: "gallbladder", 3: "vessel", 4: "kidney", 7: "spleen"}


def ms(vals, fmt):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return (fmt + " ± " + fmt).format(np.mean(vals), np.std(vals))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/compare.json")
    ap.add_argument("--out", default="docs/RESULTS_TABLE.md")
    args = ap.parse_args(argv)
    res = json.load(open(args.results))
    groups = defaultdict(list)
    for name, row in res["rows"].items():
        groups[row.get("variant", name)].append(row)
    order = [k for k in ("phys_b0", "phys_m1", "phys_m2", "b0", "m1", "m2", "m3") if k in groups]
    lines = [f"Held-out CT content: {res['n_eval']} frames from {len(res['cases'])} CTs "
             f"({', '.join(res['cases'])}); real reference: 60 labeled Kaggle ussimandsegm frames.",
             "Trained rows: mean ± SD over GAN seeds (n per row in the last column). TSTR per row is "
             "itself a mean over 3 segmenter seeds.", "",
             "| Row | organ W1 ↓ | heterogeneity gap ↓ | TSTR Dice ↑ | edge IoU ↑ | locality ↑ | HU sensitivity | n |",
             "|---|---|---|---|---|---|---|---|"]
    for k in order:
        rows = groups[k]
        g = lambda key: [r.get(key) for r in rows]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            TITLES.get(k, k), ms(g("organ_w1"), "{:.3f}"), ms(g("hetero_gap"), "{:.4f}"),
            ms(g("tstr_dice"), "{:.3f}"), ms(g("edge_iou"), "{:.3f}"),
            ms(g("locality"), "{:.1f}") if rows[0].get("kind") == "variant" else "n/a",
            ms(g("hu_sensitivity"), "{:.4f}") if rows[0].get("kind") == "variant" else "n/a", len(rows)))
    lines += ["", "Per-organ intensity W1 to real (lower is better):", "",
              "| Row | " + " | ".join(ORGANS.values()) + " |", "|---" * (len(ORGANS) + 1) + "|"]
    for k in order:
        rows = groups[k]
        cells = [ms([r["organ_w1_per"].get(str(c), r["organ_w1_per"].get(c)) for r in rows], "{:.3f}")
                 for c in ORGANS]
        lines.append(f"| {TITLES.get(k, k)} | " + " | ".join(cells) + " |")
    lines += ["", "Low-frequency heterogeneity (SD of 2-px-smoothed intensity inside the organ; real in "
              "brackets):", "", "| Row | liver | kidney | spleen |", "|---|---|---|---|"]
    for k in order:
        rows = groups[k]
        cells = []
        for c in (1, 4, 7):
            per = [r["hetero_per"].get(str(c), r["hetero_per"].get(c)) for r in rows]
            per = [p for p in per if p]
            cells.append(ms([p["fake"] for p in per], "{:.4f}") + (f" [{per[0]['real']:.4f}]" if per else ""))
        lines.append(f"| {TITLES.get(k, k)} | " + " | ".join(cells) + " |")
    lines += ["", "TSTR per-class Dice on the real frames:", "",
              "| Row | " + " | ".join(ORGANS.values()) + " |", "|---" * (len(ORGANS) + 1) + "|"]
    for k in order:
        rows = groups[k]
        cells = []
        for c in ORGANS:
            vals = [np.mean([run["per_class"].get(str(c), run["per_class"].get(c, np.nan))
                             for run in r["tstr_runs"]]) for r in rows]
            cells.append(ms(vals, "{:.3f}"))
        lines.append(f"| {TITLES.get(k, k)} | " + " | ".join(cells) + " |")
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
