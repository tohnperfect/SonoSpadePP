#!/usr/bin/env python3
"""Build results/v2/metrics.csv (all rows) and results/v2/summary.csv (mean +- SD over seeds) from
results/v2/rows/*.csv, and print the gate checks. Every number in the paper's tables comes from these files.

  python -I scripts/v2/make_v2_tables.py [--gate wp0]
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

# Published v1 values (SASHIMI 2026, Table 1, the SonoSpade v0.2.0 release's results_macros.tex)
PAPER_V1 = {"edge_iou": 0.41, "organ_w1": 0.119, "locality_band0": 26.3, "tstr_dice": 0.025}

KEY = [  # (metric, tissue, extractor, protocol prefix or None, direction)
    ("edge_iou", "all", "", "v1", "higher"),
    ("organ_w1", "all", "", "v1", "lower"),
    ("locality_band0", "all", "", "v1", "higher"),
    ("locality_band8", "all", "", "band8", "higher"),
    ("tstr_dice", "all", "", "v1 tstr_seed", "higher"),
    ("tissue_kid", "mean", "usfm", "crop16", "lower"),
    ("tissue_kid", "mean", "dinov2", "crop16", "lower"),
    ("tissue_kid", "mean", "echocare", "crop16", "lower"),
    ("frame_kid", "all", "dinov2", "fan", "lower"),
    ("frame_kid", "all", "dinov2", "raw", "lower"),
    ("nakagami_logm_w1", "mean", "", "", "lower"),
    ("nakagami_m_w1", "mean", "", "", "lower"),
    ("speckle_size_err", "mean", "", "", "lower"),
    ("diversity_lpips", "all", "", None, "higher"),
]


def load_rows(rows_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(rows_dir, "*.csv"))):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def select(rows, metric, tissue, extractor, proto, dr="60.0"):
    out = []
    for r in rows:
        if r["metric"] != metric or r["tissue"] != tissue or r["extractor"] != extractor:
            continue
        if r["dr_db"] not in ("", dr):
            continue
        p = r["protocol"]
        if proto is not None:
            if proto == "" and p != "":
                continue
            if proto and not p.startswith(proto):
                continue
        if "sensitivity" in p:
            continue
        out.append(r)
    return out


def mean_sd(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    m = math.fsum(vals) / len(vals)                     # fsum: the same bytes on every Python version
    sd = math.sqrt(math.fsum((v - m) * (v - m) for v in vals) / (len(vals) - 1)) if len(vals) > 1 else float("nan")
    return m, sd, len(vals)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", default="results/v2/rows")
    ap.add_argument("--out-dir", default="results/v2")
    ap.add_argument("--gate", choices=["", "wp0"], default="")
    args = ap.parse_args(argv)

    rows = load_rows(args.rows_dir)
    if not rows:
        raise SystemExit(f"no rows under {args.rows_dir}")
    fields = list(rows[0].keys())
    rows.sort(key=lambda r: (r["model"], int(r["seed"]), r["metric"], r["tissue"], r["extractor"], r["dr_db"],
                             r["protocol"]))
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    models = sorted({r["model"] for r in rows})
    summ = []
    for metric, tissue, ext, proto, direction in KEY:
        for m in models:
            sel = [r for r in select(rows, metric, tissue, ext, proto) if r["model"] == m]
            if not sel:
                continue
            per_seed = defaultdict(list)
            for r in sel:
                per_seed[r["seed"]].append(float(r["value"]))
            seed_means = [math.fsum(v) / len(v) for v in per_seed.values()]        # TSTR: average TSTR seeds first
            mu, sd, n = mean_sd(seed_means)
            allv = [float(r["value"]) for r in sel]
            mu_all, sd_all, n_all = mean_sd(allv)
            summ.append(dict(model=m, metric=metric, tissue=tissue, extractor=ext, mean=mu, sd=sd, n_seeds=n,
                             min=min(allv), max=max(allv), sd_all_runs=sd_all, n_runs=n_all, direction=direction))
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    print(f"{'model':16s} {'metric':16s} {'tissue':6s} {'extr':7s} {'mean':>10s} {'sd':>9s} {'n':>3s}")
    for s in summ:
        print(f"{s['model']:16s} {s['metric']:16s} {s['tissue']:6s} {s['extractor']:7s} {s['mean']:10.4f} "
              f"{s['sd']:9.4f} {s['n_seeds']:3d}")

    if args.gate == "wp0":
        print("\nWP0 gate: published v1 vs retrained v1 (seeds 0-2), pass = within mean +- 2 SD or min-max")
        ok_all = True
        for metric, paper in PAPER_V1.items():
            s = next((s for s in summ if s["model"] == "v1" and s["metric"] == metric), None)
            if s is None:
                print(f"  {metric:16s} missing"); ok_all = False; continue
            sd = s["sd_all_runs"] if metric == "tstr_dice" else s["sd"]
            lo, hi = s["mean"] - 2 * sd, s["mean"] + 2 * sd
            ok = (lo <= paper <= hi) or (s["min"] <= paper <= s["max"])
            ok_all &= ok
            print(f"  {metric:16s} paper {paper:8.3f}  retrained {s['mean']:8.3f} +- {sd:.3f} "
                  f"[{s['min']:.3f}, {s['max']:.3f}]  {'PASS' if ok else 'FAIL'}")
        print(f"  overall: {'PASS' if ok_all else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
