#!/usr/bin/env python3
"""Apply the pre-registered gate test to a candidate's evaluation rows.

Realism: the candidate passes on a measure when its 3-seed mean is lower than the control's 3-seed mean AND every
one of its seeds is below the control's 3-seed mean. Measures: tissue-mean KID with USFM (16 px crops) or the
tissue-mean FD with USFM; either one passing counts ("improves per-tissue KID or FD over v1"). DINOv2 is reported.
Invariant 1: the candidate's 3-seed means of locality (band 0, band 8) and edge IoU are at least 95% of the
control's. The gate passes only when realism passes and invariant 1 holds. Latency is measured separately.

Writes results/v2/gate_<candidate>.json and prints one line per check, then `GATE PASS` or `GATE FAIL`.

  python -I scripts/v2/gate_v2.py --cand wp1_e1b_echocare_tis0.1 [--control v1]

--mode noharm (WP3, "does not hurt the generator"): realism passes when the tissue KID (USFM) and W1 log m 3-seed
means are not above the control's 3-seed mean + 1 SD; with --sus-name, the segmenter's pooled real macro Dice must
also exceed scratch's (results/v2/sus.csv).
  python -I scripts/v2/gate_v2.py --cand susechocare --mode noharm --sus-name echocare
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from sonospadepp.v2.hygiene import realism_extractor  # noqa: E402
FD_TISSUES = ("liver", "gallbladder", "vessel", "kidney")   # the tissues the fixed content set scores (section 3)


def _tables():
    spec = importlib.util.spec_from_file_location("make_v2_tables", os.path.join(HERE, "make_v2_tables.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_seed(T, rows, model, metric, tissue, ext, proto):
    d = defaultdict(list)
    for r in T.select(rows, metric, tissue, ext, proto):
        if r["model"] == model and math.isfinite(float(r["value"])):
            d[int(r["seed"])].append(float(r["value"]))
    return {s: sum(v) / len(v) for s, v in d.items()}


def tissue_mean_fd(T, rows, model, ext):
    per = [per_seed(T, rows, model, "tissue_fd", t, ext, "crop16") for t in FD_TISSUES]
    seeds = set.intersection(*(set(p) for p in per)) if all(per) else set()
    return {s: sum(p[s] for p in per) / len(per) for s in seeds}


def _sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else math.nan


def _mean(d):
    return sum(d.values()) / len(d) if d else math.nan


def wp2(a, T, rows, res):
    """WP2 gate: E2a realism no worse than the reference (3-seed mean within the
    reference's mean + 1 SD, tissue KID USFM and W1 log m); diversity 3-seed mean > 0.02 with every seed > 0;
    style-swap locality (8 px band) at least v1's label-swap locality (8 px band); label-swap locality (both)
    and edge IoU at least 95% of v1's."""
    import glob
    ref = a.ref
    if ref == "auto":
        passing = []
        for f in sorted(glob.glob("results/v2/gate_*.json")):
            g = json.load(open(f))
            cand = g.get("candidate", "")
            if g.get("gate") == "PASS" and (cand.startswith("wp1_") or cand.startswith("sus")):
                passing.append(cand)
        ext = realism_extractor(rows, a.cand, a.control, *passing)     # clean for every model compared
        best = None
        for cand in passing:
            k = _mean(per_seed(T, rows, cand, "tissue_kid", "mean", ext, "crop16"))
            if best is None or k < best[1]:
                best = (cand, k)
        ref = best[0] if best else a.control
    ext = realism_extractor(rows, a.cand, ref)
    res["reference"], res["realism_extractor"] = ref, ext
    print(f"realism reference: {ref}; extractor {ext} (not in either model's training loss)")
    ok_all = True
    for name, metric, ext, proto in ((f"tissue_kid_{ext}", "tissue_kid", ext, "crop16"),
                                     ("nakagami_logm_w1", "nakagami_logm_w1", "", "")):
        c = per_seed(T, rows, a.cand, metric, "mean", ext, proto)
        k = per_seed(T, rows, ref, metric, "mean", ext, proto)
        ok = len(c) >= 3 and len(k) >= 3 and _mean(c) <= _mean(k) + _sd(list(k.values()))
        lim = _mean(k) + _sd(list(k.values())) if len(k) > 1 else math.nan
        ok_all &= ok
        res["checks"][f"realism_{name}"] = {"pass": ok, "cand_mean": _mean(c), "limit": lim}
        print(f"realism {name}: cand {_mean(c):.3f} vs reference mean + SD {lim:.3f} -> {'holds' if ok else 'fails'}")
    div = per_seed(T, rows, a.sampled or a.cand, "diversity_lpips", "all", "", None)
    ok = len(div) >= 3 and _mean(div) > 0.02 and all(v > 0 for v in div.values())
    ok_all &= ok
    res["checks"]["diversity"] = {"pass": ok, "cand_mean": _mean(div), "seeds": {str(k): v for k, v in div.items()}}
    print(f"diversity LPIPS: {_mean(div):.4f} (seeds " + ", ".join(f"{v:.4f}" for _, v in sorted(div.items()))
          + f") -> {'pass' if ok else 'fail'}")
    ssl = per_seed(T, rows, a.cand, "style_swap_locality", "all", "", None)
    lsl = per_seed(T, rows, a.control, "locality_band8", "all", "", "band8")
    ok = bool(ssl) and bool(lsl) and _mean(ssl) >= _mean(lsl)
    ok_all &= ok
    res["checks"]["style_swap_locality"] = {"pass": ok, "cand_mean": _mean(ssl), "control_label_swap": _mean(lsl)}
    print(f"style-swap locality {_mean(ssl):.3f} vs v1 label-swap (8 px band) {_mean(lsl):.3f} -> "
          f"{'pass' if ok else 'fail'}")
    inv = True
    for metric, proto in (("locality_band0", "v1"), ("locality_band8", "band8"), ("edge_iou", "v1")):
        c = per_seed(T, rows, a.cand, metric, "all", "", proto)
        k = per_seed(T, rows, a.control, metric, "all", "", proto)
        ok = bool(c) and bool(k) and _mean(c) >= a.frac * _mean(k)
        inv &= ok
        res["checks"][f"invariant1_{metric}"] = {"pass": ok, "cand_mean": _mean(c), "control_mean": _mean(k)}
        print(f"invariant 1 {metric}: cand {_mean(c):.4f} vs {a.frac:.0%} of control {a.frac * _mean(k):.4f} -> "
              f"{'holds' if ok else 'violated'}")
    res["mode"] = "wp2"
    res["gate"] = "PASS" if ok_all and inv else "FAIL"
    out = a.out or f"results/v2/gate_{a.cand}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"GATE {res['gate']} (WP2) -> {out}")
    return 0


def noharm(a, T, rows, res):
    import csv
    ok_all = True
    ext = realism_extractor(rows, a.cand, a.control)
    res["realism_extractor"] = ext
    print(f"realism extractor: {ext} (not in either model's training loss)")
    for name, metric, ext, proto in ((f"tissue_kid_{ext}", "tissue_kid", ext, "crop16"),
                                     ("nakagami_logm_w1", "nakagami_logm_w1", "", "")):
        c = per_seed(T, rows, a.cand, metric, "mean", ext, proto)
        k = per_seed(T, rows, a.control, metric, "mean", ext, proto)
        if len(c) < 3 or len(k) < 3:
            ok, cm, km, lim = False, math.nan, math.nan, math.nan
        else:
            cm = sum(c.values()) / len(c)
            km = sum(k.values()) / len(k)
            lim = km + _sd(list(k.values()))
            ok = cm <= lim
        ok_all &= ok
        res["checks"][f"noharm_{name}"] = {"pass": ok, "cand_mean": cm, "control_mean": km, "limit": lim,
                                            "n_seeds": len(c)}
        print(f"no harm {name}: cand {cm:.3f} vs control mean + SD {lim:.3f} ({len(c)} seeds) -> "
              f"{'holds' if ok else 'violated'}")
    if a.sus_name:
        dice = {r["name"]: float(r["value"]) for r in csv.DictReader(open(a.sus_csv))
                if r["metric"] == "real_dice" and r["tissue"] == "macro"}
        cd, sd = dice.get(a.sus_name, math.nan), dice.get("scratch_v1", math.nan)
        ok = cd > sd
        ok_all &= ok
        res["checks"]["sus_real_dice"] = {"pass": ok, "cand": cd, "scratch": sd}
        print(f"S_US pooled real macro Dice: {a.sus_name} {cd:.4f} vs scratch {sd:.4f} -> {'pass' if ok else 'fail'}")
    inv = True
    for metric, proto in (("locality_band0", "v1"), ("locality_band8", "band8"), ("edge_iou", "v1")):
        c = per_seed(T, rows, a.cand, metric, "all", "", proto)
        k = per_seed(T, rows, a.control, metric, "all", "", proto)
        cm = sum(c.values()) / len(c) if c else math.nan
        km = sum(k.values()) / len(k) if k else math.nan
        ok = bool(c) and bool(k) and cm >= a.frac * km
        inv &= ok
        res["checks"][f"invariant1_{metric}"] = {"pass": ok, "cand_mean": cm, "control_mean": km}
        print(f"invariant 1 {metric}: cand {cm:.4f} vs {a.frac:.0%} of control {a.frac * km:.4f} -> "
              f"{'holds' if ok else 'violated'}")
    res["mode"] = "noharm"
    res["gate"] = "PASS" if ok_all and inv else "FAIL"
    out = a.out or f"results/v2/gate_{a.cand}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"GATE {res['gate']} (no-harm mode) -> {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--control", default="v1")
    ap.add_argument("--rows-dir", default="results/v2/rows")
    ap.add_argument("--frac", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["improve", "noharm", "wp2"], default="improve")
    ap.add_argument("--ref", default="auto", help="wp2: realism reference; auto = best gate-passing WP1/WP3 "
                                                  "model by tissue KID (results/v2/gate_*.json), else the control")
    ap.add_argument("--sampled", default=None, help="wp2: model key of the sampled-code rows (E2b)")
    ap.add_argument("--sus-name", default=None)
    ap.add_argument("--sus-csv", default="results/v2/sus.csv")
    a = ap.parse_args(argv)
    T = _tables()
    rows = T.load_rows(a.rows_dir)
    res = {"candidate": a.cand, "control": a.control, "checks": {}}

    def realism(name, cand, ctrl):
        if len(cand) < 3 or len(ctrl) < 3:
            res["checks"][name] = {"pass": False, "note": f"seeds cand {sorted(cand)} control {sorted(ctrl)}"}
            print(f"{name}: missing seeds (cand {sorted(cand)}, control {sorted(ctrl)}) -> fail")
            return False
        cm = sum(cand.values()) / len(cand)
        km = sum(ctrl.values()) / len(ctrl)
        ok = cm < km and all(v < km for v in cand.values())
        res["checks"][name] = {"pass": ok, "cand_mean": cm, "control_mean": km,
                               "cand_seeds": {str(k): v for k, v in sorted(cand.items())}}
        print(f"{name}: cand {cm:.3f} (seeds " + ", ".join(f"{v:.3f}" for _, v in sorted(cand.items())) +
              f") vs control {km:.3f} -> {'pass' if ok else 'fail'}")
        return ok

    if a.mode == "noharm":
        return noharm(a, T, rows, res)
    if a.mode == "wp2":
        return wp2(a, T, rows, res)
    ext = realism_extractor(rows, a.cand, a.control)
    res["realism_extractor"] = ext
    print(f"realism extractor: {ext} (not in either model's training loss)")
    kid = realism(f"tissue_kid_{ext}", per_seed(T, rows, a.cand, "tissue_kid", "mean", ext, "crop16"),
                  per_seed(T, rows, a.control, "tissue_kid", "mean", ext, "crop16"))
    fd = realism(f"tissue_fd_{ext}", tissue_mean_fd(T, rows, a.cand, ext), tissue_mean_fd(T, rows, a.control, ext))
    other = "dinov2" if ext == "usfm" else "usfm"
    if other not in __import__("sonospadepp.v2.hygiene", fromlist=["x"]).excluded_from_rows(rows, a.cand):
        realism(f"tissue_kid_{other} (reported, not gating)",
                per_seed(T, rows, a.cand, "tissue_kid", "mean", other, "crop16"),
                per_seed(T, rows, a.control, "tissue_kid", "mean", other, "crop16"))
    realism("nakagami_logm_w1 (primary endpoint 2, reported, not gating)",
            per_seed(T, rows, a.cand, "nakagami_logm_w1", "mean", "", ""),
            per_seed(T, rows, a.control, "nakagami_logm_w1", "mean", "", ""))

    inv = True
    for metric, proto in (("locality_band0", "v1"), ("locality_band8", "band8"), ("edge_iou", "v1")):
        c = per_seed(T, rows, a.cand, metric, "all", "", proto)
        k = per_seed(T, rows, a.control, metric, "all", "", proto)
        if not c or not k:
            ok, cm, km = False, math.nan, math.nan
        else:
            cm, km = sum(c.values()) / len(c), sum(k.values()) / len(k)
            ok = cm >= a.frac * km
        inv &= ok
        res["checks"][f"invariant1_{metric}"] = {"pass": ok, "cand_mean": cm, "control_mean": km}
        print(f"invariant 1 {metric}: cand {cm:.4f} vs {a.frac:.0%} of control {a.frac * km:.4f} -> "
              f"{'holds' if ok else 'violated'}")

    res["realism_pass"] = kid or fd
    res["invariant1"] = inv
    res["gate"] = "PASS" if (kid or fd) and inv else "FAIL"
    out = a.out or f"results/v2/gate_{a.cand}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"GATE {res['gate']} (realism {'pass' if kid or fd else 'fail'}, invariant 1 "
          f"{'holds' if inv else 'violated'}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
