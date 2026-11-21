#!/usr/bin/env python3
"""Summary of the one-image setting (M9): for every kidney the central native plane is the one real US
image given, and the other six native planes (354 in all) are scored. Every row is restricted to
those planes, including the rows that never see the given image (B0, M7, M8), so all rows are
compared on the same planes.

Reads outputs/trusted_def.json and outputs/trusted_m9.json (fixed registration, eval_trusted.py),
outputs/trusted_aligned.json and outputs/trusted_aligned_m9.json (alignment-tolerant, eval_aligned.py).
Writes docs/TRUSTED_M9.md and outputs/m9_summary.json.

  python -I scripts/summarize_m9.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_hires import per_kidney, wil  # noqa: E402

ROWS = [("b0", "SonoSPADE B0 (unpaired, Kaggle)"), ("b0t", "B0t: SonoSPADE recipe fine-tuned on TRUSTED"),
        ("m7c", "Approach 1 (M7c: paired training)"), ("m9ac", "Approach 2 (M9ac: + one-shot fine-tune on the image)"),
        ("copy", "aux: given image copied"), ("labwarp", "aux: given image warped by labels"),
        ("m7", "aux: approach 1 + HU (M7)"), ("m9a", "aux: approach 2 + HU (M9a)"),
        ("m8c", "aux: per-patient, half sweep (M8c)"), ("m8", "aux: M8c + HU (M8)"),
        ("m9cw", "aux: exemplar warp alone (M9c W)"), ("m9w", "aux: exemplar warp alone + HU (M9 W)"),
        ("m9c", "aux: exemplar network (M9c)"), ("m9", "aux: exemplar network + HU (M9)")]
PAIRS = [("m7c", "b0t", "approach 1 vs domain-matched SonoSPADE"), ("b0t", "b0", "scanner domain only"),
         ("m7c", "b0", "approach 1 vs SonoSPADE"), ("m9ac", "m7c", "approach 2 vs approach 1"),
         ("m9ac", "labwarp", "approach 2 vs the image warped by labels"), ("m9ac", "b0t", "approach 2 vs domain-matched SonoSPADE"),
         ("m7", "m7c", "HU, approach 1"), ("m9a", "m9ac", "HU, approach 2"), ("m8", "m8c", "HU, per-patient"),
         ("m9", "m9c", "HU, exemplar network"), ("m9c", "m7c", "exemplar network vs approach 1"),
         ("m9ac", "m9c", "one-shot fine-tune vs exemplar network"), ("m9ac", "m8c", "one image vs half the sweep")]
BINS = ((0, 7), (7, 13), (13, 90))


def load_groups(paths, keys):
    groups, kid_ids, kidneys = {}, None, None
    for p in paths:
        if not os.path.exists(p):
            continue
        res = json.load(open(p))
        if kid_ids is None:
            kid_ids, kidneys = np.asarray(res["kidney_id"]), res["kidneys"]
        assert list(res["kidney_id"]) == list(kid_ids), f"{p}: different planes"
        for name, row in res["rows"].items():
            g = row.get("variant", name.rsplit("_s", 1)[0])
            groups.setdefault(g, {}).setdefault(name, {k: np.asarray(row[k], float) for k in keys if k in row})
    return groups, kid_ids, kidneys


def main():
    missing = [p for p in ("data/trusted/pairs_def", "outputs/trusted_def.json", "outputs/trusted_m9.json", "outputs/trusted_aligned.json", "outputs/trusted_aligned_m9.json") if not os.path.exists(p)]
    if missing:   # the tracked result files are never rewritten from an incomplete set of inputs
        raise SystemExit("missing inputs (see the README, Reproduce, level 2): " + ", ".join(missing))
    from train_m7 import load_planes
    from train_m9 import native_refs
    planes = load_planes("data/trusted/pairs_def")
    refs = native_refs(planes, "data/trusted/pairs_def")
    target = np.zeros(len(planes), bool); dphi = np.full(len(planes), np.nan)
    for kid, (r, tg) in refs.items():
        target[tg] = True
        for i in tg:
            dphi[i] = abs(planes[i]["phi"] - planes[r]["phi"])
    out, lines = {"n_target_planes": int(target.sum()), "fixed": {}, "aligned": {}, "pairs": {}}, []
    settings = [("fixed", ["outputs/trusted_def.json", "outputs/trusted_m9.json"], ("r_kidney_dt", "r_band_dt")),
                ("aligned", ["outputs/trusted_aligned.json", "outputs/trusted_aligned_m9.json"], ("r_kidney_after", "r_band_after"))]
    for setting, paths, (kk, kb) in settings:
        groups, kid_ids, kidneys = load_groups(paths, (kk, kb))
        if not groups:
            continue
        pk = {}
        for g, rows in groups.items():
            mk = lambda key, m: np.stack([per_kidney(np.where(m, r[key], np.nan), kid_ids, kidneys) for r in rows.values()])
            pk[g] = {"kid": mk(kk, target), "band": mk(kb, target),
                     "bins": [mk(kk, target & (dphi >= lo) & (dphi < hi)) for lo, hi in BINS]}
        lines += ["", f"## {setting} registration" if setting == "fixed" else "## alignment-tolerant (smooth warp of at most 5 mm)", "",
                  "| row | seeds | r kidney | r kidney+band | " + " | ".join(f"kidney, {lo}-{hi} deg" for lo, hi in BINS) + " |",
                  "|---|---|---|---|" + "---|" * len(BINS)]
        for g, title in ROWS:
            if g not in pk:
                continue
            seed_k = np.nanmean(pk[g]["kid"], 1); seed_b = np.nanmean(pk[g]["band"], 1)
            bins = [float(np.nanmean(b)) for b in pk[g]["bins"]]
            out[setting][g] = {"kidney": float(seed_k.mean()), "band": float(seed_b.mean()), "kidney_sd_seeds": float(seed_k.std()),
                               "bins": bins, "n_seeds": int(len(seed_k))}
            lines.append(f"| {title} | {len(seed_k)} | {seed_k.mean():+.3f} | {seed_b.mean():+.3f} | "
                         + " | ".join(f"{b:+.3f}" for b in bins) + " |")
        lines += ["", "| comparison | what it tests | median per-kidney diff (kidney) | p | kidneys up | diff (band) | p |",
                  "|---|---|---|---|---|---|---|"]
        for a, b, what in PAIRS:
            if a not in pk or b not in pk:
                continue
            wk = wil(np.nanmean(pk[a]["kid"], 0), np.nanmean(pk[b]["kid"], 0))
            wb = wil(np.nanmean(pk[a]["band"], 0), np.nanmean(pk[b]["band"], 0))
            if not wk or not wb:
                continue
            out["pairs"][f"{setting}:{a}-{b}"] = {"kidney": wk, "band": wb}
            lines.append(f"| {a} - {b} | {what} | {wk[0]:+.3f} | {wk[1]:.2g} | {wk[2]}/{wk[3]} | {wb[0]:+.3f} | {wb[1]:.2g} |")
    json.dump(out, open("outputs/m9_summary.json", "w"), indent=1)
    head = ("# One real US image given (M9)\n\nFor every kidney the central native plane (4th of 7 by sweep angle) is the "
            f"one real US image given; the other {int(target.sum())} native planes are scored, for every row. Agreement: "
            "depth-detrended Pearson r of the 1.5 px smoothed images; per-kidney means, mean over the three seeds; "
            "comparisons by Wilcoxon signed-rank over kidneys. Angle bins: distance of the scored plane from the given plane.\n")
    open("docs/TRUSTED_M9.md", "w").write(head + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote docs/TRUSTED_M9.md and outputs/m9_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
