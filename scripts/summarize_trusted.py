#!/usr/bin/env python3
"""Aggregate outputs/trusted.json into markdown: per-kidney means over planes, then over kidneys;
trained rows mean +- SD across GAN seeds; paired Wilcoxon signed-rank tests against B0 (trained
rows, seed-averaged per kidney) or against the base-label physics render (physics rows).

  python scripts/summarize_trusted.py --results outputs/trusted.json --out docs/TRUSTED_TABLE.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

TITLES = {"phys_b0": "physics render, base labels", "phys_m1": "physics render, M1 labels",
          "phys_m2": "physics render, HU-modulated", "ct_hu": "CT intensity itself (reference)",
          "b0": "B0 SonoSPADE (labels + physics)", "m1": "M1 CT sub-labels",
          "m2": "M2 HU-modulated physics", "m3": "M3 HU as a SPADE channel",
          "m3_hu0": "M3 with its HU channel zeroed (ablation)",
          "phys_m4c": "physics render, learned per-tissue echo (no HU)", "phys_m4": "physics render, learned HU-to-echo",
          "m4c": "M4 control: learned per-tissue echo, no HU", "m4": "M4 learned HU-to-echo (same-patient training, CV)",
          "phys_m2z": "physics render, M2 impedance only", "phys_m2b": "physics render, M2 backscatter only",
          "phys_m2bmean": "physics render, M2 backscatter as a per-tissue mean",
          "m2z": "M2 ablation: impedance only (M2 generator)", "m2b": "M2 ablation: backscatter only (M2 generator)",
          "m2bmean": "M2 ablation: per-tissue mean backscatter only (M2 generator)",
          "phys_m4w": "physics render, M4 with 100x weaker penalties", "phys_m4m": "physics render, M4 started at M2's rule",
          "m4w": "M4 check: 100x weaker penalties", "m4m": "M4 check: started at M2's backscatter rule, weak penalties",
          "phys_m5": "physics render, M5 patient-specific HU-to-echo (sweep halves)",
          "phys_m5c": "physics render, M5c patient-specific echo, no HU (sweep halves)",
          "phys_m5k": "physics render, M5 fitted on the other kidney", "phys_m5kc": "physics render, M5c fitted on the other kidney",
          "m5": "M5 patient-specific HU-to-echo (fit on half the sweep, test on the other half)",
          "m5c": "M5c patient-specific echo, no HU (sweep halves)",
          "m5k": "M5 patient-specific HU-to-echo (fit on the other kidney)", "m5kc": "M5c patient-specific echo, no HU (other kidney)"}
METRICS = ("r_kidney_dt", "r_band_dt", "r_fan", "r_kidney")


def per_kidney(values, kid_ids, kidneys):
    v = np.asarray(values, np.float64)
    return np.array([np.nanmean(v[kid_ids == k]) if np.isfinite(v[kid_ids == k]).any() else np.nan
                     for k in kidneys])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/trusted.json")
    ap.add_argument("--out", default="docs/TRUSTED_TABLE.md")
    args = ap.parse_args(argv)
    from scipy.stats import wilcoxon

    res = json.load(open(args.results))
    kid_ids = np.asarray(res["kidney_id"])
    kidneys = res["kidneys"]
    groups = defaultdict(list)
    for name, row in res["rows"].items():
        groups[row.get("variant", name)].append(row)
    # group -> metric -> (n_seeds, n_kidneys)
    pk = {g: {m: np.stack([per_kidney(r[m], kid_ids, kidneys) for r in rows]) for m in METRICS}
          for g, rows in groups.items()}
    order = [g for g in ("ct_hu", "phys_b0", "phys_m1", "phys_m2", "phys_m2z", "phys_m2b", "phys_m2bmean", "phys_m4c",
                         "phys_m4", "phys_m4w", "phys_m4m", "b0", "m1", "m2", "m2z", "m2b", "m2bmean", "m3", "m3_hu0", "m4c", "m4",
                         "m4w", "m4m", "phys_m5c", "phys_m5", "phys_m5kc", "phys_m5k", "m5c", "m5", "m5kc", "m5k") if g in pk]

    def cell(g, m):
        a = pk[g][m]                                           # seeds x kidneys
        seed_means = np.nanmean(a, axis=1)
        if len(seed_means) == 1:
            return f"{seed_means[0]:+.3f}"
        return f"{np.mean(seed_means):+.3f} ± {np.std(seed_means):.3f}"

    def test(g, m):
        ref = "phys_b0" if g.startswith("phys_") else "b0"
        if g in (ref, "ct_hu") or ref not in pk:
            return ""
        x = np.nanmean(pk[g][m], axis=0); y = np.nanmean(pk[ref][m], axis=0)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6:
            return "n/a"
        diff = x[ok] - y[ok]
        p = wilcoxon(x[ok], y[ok]).pvalue
        return f"{np.median(diff):+.3f} (p={p:.3g}, {int((diff > 0).sum())}/{int(ok.sum())} up)"

    n_frames = res["n_frames"]
    lines = [f"TRUSTED same-patient planes: {n_frames} frames from {len(kidneys)} kidneys. "
             f"Pearson r between the smoothed synthetic and the smoothed real US of the same plane "
             f"(sigma {res['sigma']} px), averaged over each kidney's planes, then over kidneys. "
             "Trained rows: mean ± SD across GAN seeds.", "",
             "Kidney and band correlations are computed after removing a quadratic trend in depth inside "
             "the region (gain and attenuation along the beam); the last column is the raw kidney value.", "",
             "| Row | r kidney interior ↑ | r kidney + band ↑ | r whole fan ↑ | raw r kidney |", "|---|---|---|---|---|"]
    for g in order:
        lines.append(f"| {TITLES.get(g, g)} | " + " | ".join(cell(g, m) for m in METRICS) + " |")
    lines += ["", "Paired difference against B0 (trained rows) or the base-label physics render "
              "(physics rows): median over kidneys of the per-kidney difference, Wilcoxon "
              "signed-rank p, and how many kidneys improved.", "",
              "| Row | r kidney interior | r kidney + band | r whole fan | raw r kidney |", "|---|---|---|---|---|"]
    for g in order:
        t = [test(g, m) for m in METRICS]
        if any(t):
            lines.append(f"| {TITLES.get(g, g)} | " + " | ".join(t) + " |")
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
