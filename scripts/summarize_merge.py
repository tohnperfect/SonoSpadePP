#!/usr/bin/env python3
"""Pass criteria of the same-patient test, applied mechanically to every measure (fixed before anything ran).

Planes: the 354 native planes other than each kidney's central (given) plane, for every row, as in
summarize_m9.py. Per-kidney means, mean over seeds, Wilcoxon signed-rank over kidneys. A comparison passes only
with p < 0.05 and a positive median difference; a tie fails.

Measures. Fixed registration (eval_trusted.py): depth-detrended r over the kidney (kidney dt) and the kidney + band
(band dt), plain r over the kidney (kidney) and over the whole common fan (fan). Alignment-tolerant (eval_aligned.py,
detrended only): kidney al, band al. Kidney W1: intensity W1 between the output and the real image inside the
kidney mask of the real image, per plane (lower is better; computed here from the prediction files).

Criteria:
  approach 1 vs B0t        pass on kidney dt, band dt, kidney, fan, kidney al, band al; and not worse on kidney W1
  approach 2 vs labwarp    pass in every angle bin on kidney dt, kidney, fan and kidney al
  approach 2 vs approach 1 the same
  v2 rows (when present)   v2 vs v1 on TRUSTED (reported either way), approach 1 from v2 vs B0t-v2 and vs approach 1
                           from v1, approach 2 from v2 vs labwarp and vs approach 1 from v2

Writes outputs/merge_summary.json and docs/TRUSTED_MERGE.md (numbers only, no images).

  python -I scripts/summarize_merge.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_hires import per_kidney, wil  # noqa: E402
from summarize_m9 import BINS, load_groups  # noqa: E402

FIXED = ["outputs/trusted_def.json", "outputs/trusted_m9.json", "outputs/trusted_merge_v2.json"]
ALIGNED = ["outputs/trusted_aligned.json", "outputs/trusted_aligned_m9.json", "outputs/trusted_aligned_merge_v2.json"]
FIXED_KEYS = {"kidney dt": "r_kidney_dt", "band dt": "r_band_dt", "kidney": "r_kidney", "fan": "r_fan"}
ALIGNED_KEYS = {"kidney al": "r_kidney_after", "band al": "r_band_after"}
PREDS = {"b0t": "outputs/b0t/preds_b0t_s{s}.npz", "m7c": "outputs/m7/preds_lab_s{s}.npz",
         "m9ac": "outputs/m9/preds_lab1_s{s}.npz", "labwarp": "outputs/m9/preds_labwarp.npz",
         "b0t_v2": "outputs/merge_v2/preds_b0t_v2_s{s}.npz", "m7c_v2": "outputs/merge_v2/preds_m7c_v2_s{s}.npz",
         "m9ac_v2": "outputs/merge_v2/preds_m9ac_v2_s{s}.npz", "p2p": "outputs/m7/preds_p2p_s{s}.npz",
         "phys_b0": "outputs/m7/preds_lab_s{s}.npz", "retrieve": "outputs/m9/preds_retrieve.npz",
         "m7x": "outputs/m7/preds_m7r_s{s}.npz"}   # hybrid; "m7r" is taken by an older run in trusted_def.json
PRED_KEY = {"phys_b0": "phys"}                 # the untranslated render, stored next to every prediction
CRITERIA = [
    ("approach 1 vs B0t", "m7c", "b0t", "a1"),
    ("approach 2 vs the label-warped given image", "m9ac", "labwarp", "a2"),
    ("approach 2 vs approach 1", "m9ac", "m7c", "a2"),
    ("v2 vs v1 on TRUSTED, no fine-tuning (reported either way)", "b0_v2", "b0", "report"),
    ("approach 1 from v2 vs B0t-v2", "m7c_v2", "b0t_v2", "a1"),
    ("approach 1 from v2 vs approach 1 from v1", "m7c_v2", "m7c", "a1"),
    ("approach 2 from v2 vs the label-warped given image", "m9ac_v2", "labwarp", "a2"),
    ("approach 2 from v2 vs approach 1 from v2", "m9ac_v2", "m7c_v2", "a2"),
    # added 2026-11-19 (paired-method checks): the pix2pix baseline is pre-registered; the
    # comparison with the untranslated render is a secondary analysis decided after the main results
    ("approach 1 vs the pix2pix baseline", "m7c", "p2p", "a1"),
    ("pix2pix baseline vs B0t (reported either way)", "p2p", "b0t", "report"),
    ("approach 1 vs the untranslated physics render (secondary)", "m7c", "phys_b0", "report"),
    # retrieved-exemplar warp (H-R), pre-registered 2026-11-19
    ("retrieved image vs approach 1", "retrieve", "m7c", "a1"),
    ("retrieved image vs SonoSPADE (reported)", "retrieve", "b0", "report"),
    ("retrieved image vs the untranslated render (reported)", "retrieve", "phys_b0", "report"),
    ("retrieved image vs the patient's own label-warped image (reported)", "retrieve", "labwarp", "report"),
    # hybrid, tier 1 with the retrieved exemplar as input (H-RX), pre-registered 2026-11-19
    ("hybrid (tier 1 + retrieved image) vs approach 1", "m7x", "m7c", "a1"),
    ("hybrid vs the retrieved image alone (reported)", "m7x", "retrieve", "report"),
    ("hybrid vs the patient's own label-warped image (reported)", "m7x", "labwarp", "report"),
]


def passes(w):
    return bool(w) and w[1] < 0.05 and w[0] > 0


def kidney_w1(pred_path, d, target, key="gen"):
    """Per-plane W1 between output and real intensities inside the real image's kidney mask (NaN elsewhere)."""
    pr = np.load(pred_path, allow_pickle=False)
    gen = pr[key].astype(np.float32)
    if key != "gen":
        gen = gen / 255.0
    real = d["real"].astype(np.float32) / 255.0
    mask = (d["kidney_us"] > 0) & (d["valid"] > 0)
    done = pr["done"].astype(bool)
    out = np.full(len(gen), np.nan)
    q = np.linspace(0.005, 0.995, 100)
    for i in np.flatnonzero(target & done):
        m = mask[i]
        if m.sum() >= 50:
            out[i] = float(np.mean(np.abs(np.quantile(gen[i][m], q) - np.quantile(real[i][m], q))))
    return out


def main():
    from train_m7 import load_planes
    from train_m9 import native_refs
    planes = load_planes("data/trusted/pairs_def")
    refs = native_refs(planes, "data/trusted/pairs_def")
    target = np.zeros(len(planes), bool)
    dphi = np.full(len(planes), np.nan)
    for _kid, (r, tg) in refs.items():
        target[tg] = True
        for i in tg:
            dphi[i] = abs(planes[i]["phi"] - planes[r]["phi"])
    bins = {"all": target} | {f"{lo}-{hi} deg": target & (dphi >= lo) & (dphi < hi) for lo, hi in BINS}

    pk = {}                                      # group -> measure -> bin -> (seeds, kidneys)
    kid_ids = kidneys = None
    for paths, keys in ((FIXED, FIXED_KEYS), (ALIGNED, ALIGNED_KEYS)):
        groups, ki, kd = load_groups(paths, tuple(keys.values()))
        if ki is not None:
            kid_ids, kidneys = ki, kd
        for g, rows in groups.items():
            for label, key in keys.items():
                if not all(key in r for r in rows.values()):
                    continue
                for bname, m in bins.items():
                    pk.setdefault(g, {}).setdefault(label, {})[bname] = np.stack(
                        [per_kidney(np.where(m, r[key], np.nan), ki, kd) for r in rows.values()])
    d = np.load("data/trusted_pairs_def.npz", allow_pickle=False)
    for g, pat in PREDS.items():
        seeds = [s for s in (0, 1, 2) if os.path.exists(pat.format(s=s))]
        if seeds:
            w = [per_kidney(kidney_w1(pat.format(s=s), d, target, PRED_KEY.get(g, "gen")), kid_ids, kidneys)
                 for s in (seeds[:1] if g in PRED_KEY else seeds)]
            pk.setdefault(g, {})["kidney W1"] = {"all": np.stack(w)}

    def comp(a, b, measure, bname="all"):
        if measure not in pk.get(a, {}) or measure not in pk.get(b, {}) or bname not in pk[a][measure]:
            return None
        return wil(np.nanmean(pk[a][measure][bname], 0), np.nanmean(pk[b][measure][bname], 0))

    out = {"rows": {}, "criteria": []}
    for g, ms in pk.items():
        out["rows"][g] = {m: float(np.nanmean(v["all"])) for m, v in ms.items()}
    lines = ["# Pass criteria of the same-patient test on TRUSTED", "",
             "Per-kidney means over the 354 non-given native planes, mean over seeds; Wilcoxon over kidneys; pass = "
             "p < 0.05 with a positive median difference (for W1, 'not worse' = not significantly higher).", ""]
    for title, a, b, kind in CRITERIA:
        if a not in pk or b not in pk:
            continue
        checks = []
        if kind in ("a1", "report"):
            for m in ("kidney dt", "band dt", "kidney", "fan", "kidney al", "band al"):
                checks.append((m, "all", comp(a, b, m)))
        else:
            for bname in [k for k in bins if k != "all"]:
                for m in ("kidney dt", "kidney", "fan", "kidney al"):
                    checks.append((m, bname, comp(a, b, m, bname)))
        rows = []
        ok_all = True
        for m, bname, w in checks:
            ok = passes(w)
            ok_all &= ok
            rows.append({"measure": m, "bin": bname, "median_diff": w[0] if w else None, "p": w[1] if w else None,
                         "up": f"{w[2]}/{w[3]}" if w else None, "pass": ok})
        w1 = None
        if kind == "a1":
            w = comp(a, b, "kidney W1")
            w1 = {"median_diff": w[0] if w else None, "p": w[1] if w else None,
                  "not_worse": not (bool(w) and w[1] < 0.05 and w[0] > 0)}
            ok_all &= w1["not_worse"] if w else False
        verdict = "reported" if kind == "report" else ("PASS" if ok_all else "FAIL")
        out["criteria"].append({"title": title, "a": a, "b": b, "checks": rows, "kidney_w1": w1, "verdict": verdict})
        lines += [f"## {title} ({a} - {b}): **{verdict}**", "", "| measure | bin | median diff | p | kidneys up | pass |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            md = "" if r["median_diff"] is None else f"{r['median_diff']:+.3f}"
            p = "" if r["p"] is None else f"{r['p']:.2g}"
            lines.append(f"| {r['measure']} | {r['bin']} | {md} | {p} | {r['up'] or ''} | {'yes' if r['pass'] else 'no'} |")
        if w1:
            md = "" if w1["median_diff"] is None else f"{w1['median_diff']:+.4f}"
            p = "" if w1["p"] is None else f"{w1['p']:.2g}"
            lines.append(f"| kidney W1 (lower is better) | all | {md} | {p} | | {'not worse' if w1['not_worse'] else 'worse'} |")
        lines.append("")
    lines += ["## Row means (all 354 planes)", "", "| row | " + " | ".join(
        ["kidney dt", "band dt", "kidney", "fan", "kidney al", "band al", "kidney W1"]) + " |", "|---|" + "---|" * 7]
    for g in sorted(out["rows"]):
        r = out["rows"][g]
        lines.append(f"| {g} | " + " | ".join(f"{r[m]:+.3f}" if m in r and np.isfinite(r[m]) else "" for m in
                                               ["kidney dt", "band dt", "kidney", "fan", "kidney al", "band al", "kidney W1"]) + " |")
    json.dump(out, open("outputs/merge_summary.json", "w"), indent=1)
    open("docs/TRUSTED_MERGE.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
