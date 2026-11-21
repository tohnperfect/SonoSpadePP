#!/usr/bin/env python3
"""WP3 selection: the S_US initialization with the highest held-out simulator macro Dice (results/v2/sus.csv)."""
import csv
import sys

inits = sys.argv[1:] or ["usfm", "dinov2", "echocare"]
best = None
for r in csv.DictReader(open("results/v2/sus.csv")):
    if r["metric"] == "heldout_sim_dice" and r["tissue"] == "macro":
        print(f"# {r['name']}: held-out sim Dice {float(r['value']):.4f}")
        if r["name"] in inits and (best is None or float(r["value"]) > best[1]):
            best = (r["name"], float(r["value"]))
print(best[0] if best else "NONE")
