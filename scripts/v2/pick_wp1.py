#!/usr/bin/env python3
"""Apply WP1's pre-registered selection rule to results/v2/selection.csv.

Feasible = locality (band 0 and band 8) and edge IoU at least 95% of v1sel's. Among feasible runs whose name
starts with --prefix, pick the lowest validation fan-masked KID with --extractor. Prints `NAME LAMBDA` or
`NONE` when no run is feasible.

  python -I scripts/v2/pick_wp1.py --prefix e1a_echocare_adv          # -> best lambda_fm_adv
  python -I scripts/v2/pick_wp1.py --prefix e1 --names e1a_echocare_adv0.5,e1b_echocare_tis1,e1c_...
"""
from __future__ import annotations

import argparse
import csv
import re


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/v2/selection.csv")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--names", default="", help="comma-separated exact names (overrides --prefix)")
    ap.add_argument("--extractor", default="usfm")
    ap.add_argument("--score", default=None, help="score column (default val_fan_kid_<extractor>)")
    ap.add_argument("--ref", default="v1sel")
    ap.add_argument("--frac", type=float, default=0.95)
    ap.add_argument("--require", action="append", default=[],
                    help="extra feasibility condition COLUMN:MIN (value must exceed MIN), e.g. diversity_lpips:0.02")
    a = ap.parse_args(argv)
    rows = {r["name"]: r for r in csv.DictReader(open(a.selection))}
    ref = rows[a.ref]
    keys = ("locality_band0", "locality_band8", "edge_iou")
    names = a.names.split(",") if a.names else [n for n in rows if n.startswith(a.prefix) and n != a.ref]
    best, verdicts = None, []
    for n in names:
        r = rows.get(n)
        if r is None:
            verdicts.append(f"{n}: missing")
            continue
        ok = all(float(r[k]) >= a.frac * float(ref[k]) for k in keys)
        for req in a.require:
            col, lo = req.split(":")
            ok = ok and r.get(col, "") not in ("", None) and float(r[col]) > float(lo)
        score = float(r[a.score or f"val_fan_kid_{a.extractor}"])
        verdicts.append(f"{n}: score {score:.3f} feasible {ok} " +
                        " ".join(f"{k} {float(r[k]):.3f}/{float(ref[k]):.3f}" for k in keys))
        if ok and (best is None or score < best[1]):
            best = (n, score)
    for v in verdicts:
        print("#", v)
    if best is None:
        print("NONE")
    else:
        lam = re.findall(r"(?:adv|tis|spk)([0-9.]+)", best[0])
        print(best[0], lam[-1] if lam else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
