#!/usr/bin/env python3
"""WP-final plan, read from the gate files (no choice by hand): the full model has every tier-0 component whose
gate passed, at its selected setting; each leave-one-out ablation removes one of them. A model whose flags equal a
work package's gated model reuses that model's runs and rows (an alias); only new flag sets are trained.
Writes results/v2/final_model.json and prints one line per model to train: `KEY FLAGS...`.

Components:
  A1 foundation-model discriminator   gates results/v2/gate_wp1_*.json        (both settings failed: none)
  A3 foundation-model S_US            gate results/v2/gate_sus<pick>.json     -> --s-us runs/v2/wp3/s_us_<pick>.pt
  A2 style codes (INADE fallback)     gate results/v2/gate_wp2i_e2a.json      -> --style inade --lambda-ms <pick>
  A5 speckle statistics               no feasible setting (no gate file)

  python -I scripts/v2/plan_final.py
"""
from __future__ import annotations

import glob
import json
import os
import re


def _passed(path):
    return os.path.exists(path) and json.load(open(path)).get("gate") == "PASS"


def components():
    comp = {}
    for f in sorted(glob.glob("results/v2/gate_wp1_*.json")):
        if _passed(f):
            name = json.load(open(f))["candidate"].removeprefix("wp1_")
            m = re.fullmatch(r"e1([abc])_(\w+?)_(.*)", name)
            enc, rest = m.group(2), m.group(3)
            flags = [f"--fm-disc {enc}"]
            for kind, val in re.findall(r"(adv|tis)([0-9.]+)", rest):
                flags.append(f"--lambda-fm-{'adv' if kind == 'adv' else 'tissue'} {val}")
            comp["fm"] = {"flags": " ".join(flags), "alias": f"wp1_{name}"}
            break                                   # at most one A1 setting in the full model
    for f in sorted(glob.glob("results/v2/gate_sus*.json")):
        if _passed(f):
            init = json.load(open(f))["candidate"].removeprefix("sus")
            comp["sus"] = {"flags": f"--s-us runs/v2/wp3/s_us_{init}.pt", "alias": f"sus{init}"}
    if _passed("results/v2/gate_wp2i_e2a.json"):
        pick = open("logs/v2/wp2b_pick.log").read().split()[-1]          # "WP2 INADE pick: e2i_ms<L>"
        comp["style"] = {"flags": f"--style inade --lambda-ms {pick.removeprefix('e2i_ms')}", "alias": "wp2i_e2a"}
    return comp


def main():
    comp = components()
    plan = {"components": comp, "models": {}}
    keys = sorted(comp)

    def entry(subset, key):
        if not subset:
            return {"key": "v1", "flags": "", "train": False}
        if len(subset) == 1:
            c = comp[subset[0]]
            return {"key": c["alias"], "flags": c["flags"], "train": False}
        return {"key": key, "flags": " ".join(comp[k]["flags"] for k in subset), "train": True}

    plan["models"]["full"] = entry(keys, "v2")
    for k in keys:
        plan["models"][f"no_{k}"] = entry([x for x in keys if x != k], f"v2_no_{k}")
    os.makedirs("results/v2", exist_ok=True)
    with open("results/v2/final_model.json", "w") as fh:
        json.dump(plan, fh, indent=1)
    print(f"# components passed: {keys or 'none'}; full model = {plan['models']['full']['key']}")
    for name, m in plan["models"].items():
        if m["train"]:
            print(m["key"], m["flags"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
