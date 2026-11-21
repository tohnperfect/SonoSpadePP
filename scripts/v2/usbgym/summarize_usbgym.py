#!/usr/bin/env python3
"""USB Gym closed-loop summary for v2 (protocol fixed before the runs).

Endpoint (USB Gym's definition, its scripts/collect_results_v2.py): per held-out patient, the mean
final translation error (final_pos, mm) of the RL policy trained with learned masks (arm "matched") minus that of
the policy trained with CT masks (arm "ctmask"); phase final, contact on, eval randomization off; pooled over the
three RL seeds and both targets. Six patients.

Per stage, USB Gym's rule: the gap "holds" if the patient-cluster bootstrap 95% CI (10,000 resamples, seed 0)
excludes 0 and at least 5 of 6 patients share the sign of the mean.

Primary comparison (one test, no correction): v2 against v1 retrained in the same environment (v1r), per-patient
gap difference, two-sided paired t-test p < 0.05 AND at least 5 of 6 patients with the sign of the mean
difference; a smaller gap for v2 is an improvement. Secondary (reported, Holm over the two): v2 against the
published v1 stage, v1r against the published v1 (a reproducibility reference). Also reported: the matched-arm
final error itself, and the published physics, CUT and CycleGAN gaps.

  python -I scripts/v2/usbgym/summarize_usbgym.py
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os

import numpy as np

GYM = os.environ.get("USBG_GYM_ROOT", "")          # a USB Gym checkout (https://github.com/tohnperfect/USBGym)
if not GYM:
    raise SystemExit("set USBG_GYM_ROOT to a USB Gym checkout")
SOURCES = {   # label -> (episodes_long.csv, stage name inside it)
    "physics": (f"{GYM}/outputs/v2/table4/episodes_long.csv", "physics"),
    "cut": (f"{GYM}/outputs/v2/table4/episodes_long.csv", "cut"),
    "cyclegan": (f"{GYM}/outputs/v2/table4/episodes_long.csv", "cyclegan"),
    "v1_published": (f"{GYM}/outputs/v2/table4/episodes_long.csv", "sonospade"),
    "v1r": (f"{GYM}/outputs/v2fm/table4_v1r/episodes_long.csv", "sonospade"),
    "v2": (f"{GYM}/outputs/v2fm/table4_v2/episodes_long.csv", "sonospade"),
}
BASE = dict(contact="on", eval_randomize="off", phase="final")


def per_patient(rows, **flt):
    acc, seeds = collections.defaultdict(list), set()
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in flt.items()):
            try:
                acc[r["patient"]].append(float(r["final_pos"]))
                seeds.add(r["seed"])
            except (TypeError, ValueError):
                pass
    return {p: float(np.mean(v)) for p, v in acc.items()}, sorted(seeds)


def boot_rule(d, nboot=10000, seed=0):
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if not len(d):
        return {"holds": False, "mean": math.nan, "ci": [math.nan, math.nan], "agree": 0, "n": 0}
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(1)
    m, ci = float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    agree = int((np.sign(d) == np.sign(m)).sum())
    return {"holds": bool((ci[0] > 0 or ci[1] < 0) and agree >= 5), "mean": m, "ci": ci, "agree": agree, "n": len(d)}


def paired(a, b):
    """a - b per patient (dicts): mean, paired t-test p, sign count."""
    from scipy.stats import ttest_rel
    ps = sorted(set(a) & set(b))
    x, y = np.array([a[p] for p in ps]), np.array([b[p] for p in ps])
    if len(ps) < 3:
        return {"n": len(ps), "mean": math.nan, "p": math.nan, "agree": 0, "per_patient": {}}
    d = x - y
    p = float(ttest_rel(x, y).pvalue) if np.any(d != 0) else 1.0
    return {"n": len(ps), "mean": float(d.mean()), "p": p, "agree": int((np.sign(d) == np.sign(d.mean())).sum()),
            "per_patient": {pp: float(v) for pp, v in zip(ps, d)}}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [math.nan] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    missing = [p for p in (sorted({p for p, _ in SOURCES.values()})) if not os.path.exists(p)]
    if missing:   # the tracked result files are never rewritten from an incomplete set of inputs
        raise SystemExit("missing inputs (see the README, Reproduce, level 2): " + ", ".join(missing))
    cache, stages = {}, {}
    for label, (path, st) in SOURCES.items():
        if not os.path.exists(path):
            continue
        rows = cache.setdefault(path, list(csv.DictReader(open(path))))
        m, sm = per_patient(rows, group="matched", arm="matched", stage=st, **BASE)
        c, sc = per_patient(rows, group="ctmask", arm="ctmask", stage=st, **BASE)
        gap = {p: m[p] - c[p] for p in sorted(set(m) & set(c))}
        stages[label] = {"matched": m, "ctmask": c, "gap": gap, "seeds_matched": sm, "seeds_ctmask": sc,
                         "rule": boot_rule(list(gap.values())),
                         "complete": len(sm) == 3 and len(sc) == 3 and len(gap) == 6}
    out = {"stages": {k: {kk: v[kk] for kk in ("gap", "rule", "complete", "seeds_matched", "seeds_ctmask")} |
                      {"gap_mean": float(np.mean(list(v["gap"].values()))) if v["gap"] else math.nan,
                       "matched_mean": float(np.mean(list(v["matched"].values()))) if v["matched"] else math.nan,
                       "ctmask_mean": float(np.mean(list(v["ctmask"].values()))) if v["ctmask"] else math.nan}
                      for k, v in stages.items()}, "comparisons": {}}
    if "v2" in stages and "v1r" in stages:
        pr = paired(stages["v2"]["gap"], stages["v1r"]["gap"])
        pr["verdict"] = ("v2 smaller gap" if pr["p"] < 0.05 and pr["agree"] >= 5 and pr["mean"] < 0 else
                         "v2 larger gap" if pr["p"] < 0.05 and pr["agree"] >= 5 and pr["mean"] > 0 else "no difference shown")
        out["comparisons"]["primary: v2 - v1r (gap)"] = pr
        out["comparisons"]["v2 - v1r (matched-arm final error)"] = paired(stages["v2"]["matched"], stages["v1r"]["matched"])
    sec = [(n, a, b) for n, a, b in (("secondary: v2 - v1_published (gap)", "v2", "v1_published"),
                                     ("secondary: v1r - v1_published (gap)", "v1r", "v1_published"))
           if a in stages and b in stages]
    res = [paired(stages[a]["gap"], stages[b]["gap"]) for _, a, b in sec]
    for (n, _, _), r, adj in zip(sec, res, holm([r["p"] for r in res]) if res else []):
        r["p_holm"] = adj
        out["comparisons"][n] = r
    os.makedirs("results/v2", exist_ok=True)
    json.dump(out, open("results/v2/usbgym.json", "w"), indent=1)
    with open("results/v2/usbgym.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "patient", "matched_final_mm", "ctmask_final_mm", "gap_mm"])
        for k, v in stages.items():
            for p in sorted(v["gap"]):
                w.writerow([k, p, f"{v['matched'][p]:.4f}", f"{v['ctmask'][p]:.4f}", f"{v['gap'][p]:.4f}"])
    for k, v in out["stages"].items():
        r = v["rule"]
        print(f"{k:13s} gap {v['gap_mean']:+.2f} mm  CI [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  sign {r['agree']}/{r['n']}  "
              f"holds {r['holds']}  matched {v['matched_mean']:.2f}  ct {v['ctmask_mean']:.2f}  complete {v['complete']}")
    for n, r in out["comparisons"].items():
        print(f"{n}: mean {r['mean']:+.2f} mm, p {r['p']:.3g}" + (f" (Holm {r['p_holm']:.3g})" if "p_holm" in r else "")
              + f", sign {r['agree']}/{r['n']}" + (f" -> {r['verdict']}" if "verdict" in r else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
