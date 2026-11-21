#!/usr/bin/env python3
"""T1G summary: tier 1 in the USB Gym closed loop (criteria fixed before any run).

Same endpoint and per-stage rule as summarize_usbgym.py. Primary (single test): t1g minus v1r per-patient gap
difference, two-sided paired t-test p < 0.05 and at least 5 of 6 patients with the sign of the mean. Secondary:
(a) t1g stays at the label-conditioned level: its bootstrap 95% CI upper bound is below the lowest CI lower bound
among the label-free stages (physics, CUT, CycleGAN; 2.15 mm when planned); (b) matched-arm final error, t1g - v1r.

  python -I scripts/v2/usbgym/summarize_t1g.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_usbgym as SU  # noqa: E402

SU.SOURCES["t1g"] = (f"{SU.GYM}/outputs/v2fm/table4_t1g/episodes_long.csv", "sonospade")


def main():
    missing = [p for p in (sorted({p for p, _ in SU.SOURCES.values()})) if not os.path.exists(p)]
    if missing:   # the tracked result files are never rewritten from an incomplete set of inputs
        raise SystemExit("missing inputs (see the README, Reproduce, level 2): " + ", ".join(missing))
    cache, st = {}, {}
    for label, (path, name) in SU.SOURCES.items():
        if not os.path.exists(path):
            continue
        rows = cache.setdefault(path, list(csv.DictReader(open(path))))
        m, sm = SU.per_patient(rows, group="matched", arm="matched", stage=name, **SU.BASE)
        c, sc = SU.per_patient(rows, group="ctmask", arm="ctmask", stage=name, **SU.BASE)
        gap = {p: m[p] - c[p] for p in sorted(set(m) & set(c))}
        st[label] = {"matched": m, "ctmask": c, "gap": gap, "rule": SU.boot_rule(list(gap.values())),
                     "complete": len(sm) == 3 and len(sc) == 3 and len(gap) == 6}
    out = {"stages": {k: {"gap": v["gap"], "rule": v["rule"], "complete": v["complete"],
                          "gap_mean": float(np.mean(list(v["gap"].values()))) if v["gap"] else float("nan"),
                          "matched_mean": float(np.mean(list(v["matched"].values()))) if v["matched"] else float("nan"),
                          "ctmask_mean": float(np.mean(list(v["ctmask"].values()))) if v["ctmask"] else float("nan")}
                      for k, v in st.items()}, "tests": {}}
    if "t1g" in st and "v1r" in st:
        pr = SU.paired(st["t1g"]["gap"], st["v1r"]["gap"])
        ok = pr["p"] < 0.05 and pr["agree"] >= 5
        pr["verdict"] = ("t1g smaller gap" if ok and pr["mean"] < 0 else "t1g larger gap" if ok and pr["mean"] > 0
                         else "no difference shown")
        out["tests"]["primary: t1g - v1r (gap)"] = pr
        free = [st[k]["rule"]["ci"][0] for k in ("physics", "cut", "cyclegan") if k in st]
        bound = min(free)
        hi = st["t1g"]["rule"]["ci"][1]
        out["tests"]["secondary a: t1g CI upper below the label-free CI lower bound"] = {
            "t1g_ci": st["t1g"]["rule"]["ci"], "bound": bound, "pass": bool(hi < bound)}
        out["tests"]["secondary b: t1g - v1r (matched-arm final error)"] = SU.paired(st["t1g"]["matched"], st["v1r"]["matched"])
    os.makedirs("results/v2", exist_ok=True)
    json.dump(out, open("results/v2/usbgym_t1g.json", "w"), indent=1)
    with open("results/v2/usbgym_t1g.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "patient", "matched_final_mm", "ctmask_final_mm", "gap_mm"])
        for k in ("v1r", "t1g"):
            for p in sorted(st.get(k, {}).get("gap", {})):
                w.writerow([k, p, f"{st[k]['matched'][p]:.4f}", f"{st[k]['ctmask'][p]:.4f}", f"{st[k]['gap'][p]:.4f}"])
    for k, v in out["stages"].items():
        r = v["rule"]
        print(f"{k:13s} gap {v['gap_mean']:+.2f} mm  CI [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  sign {r['agree']}/{r['n']}  "
              f"holds {r['holds']}  matched {v['matched_mean']:.2f}  ct {v['ctmask_mean']:.2f}  complete {v['complete']}")
    for n, r in out["tests"].items():
        print(n, json.dumps({k: v for k, v in r.items() if k != "per_patient"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
