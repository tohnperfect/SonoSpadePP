#!/usr/bin/env python3
"""Write results/trusted/results_macros.tex: every number the paper reports, computed from the result files.

  outputs/compare.json   unpaired comparison (eval_compare.py)
  outputs/trusted.json   same-patient TRUSTED comparison (eval_trusted.py)
  data/trusted_pairs_meta.json, runs/s_us_*.json

A missing input leaves its macros as \\na (a grey dash), so the paper always compiles.

  python scripts/paper_macros.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

ROWS = {"phys_b0": "PhysBase", "phys_m1": "PhysSub", "phys_m2": "PhysHU", "b0": "BZero", "m1": "MOne",
        "m2": "MTwo", "m3": "MThree", "m3_hu0": "MThreeNoHU", "ct_hu": "CTHU",
        "phys_m4c": "PhysLearnC", "phys_m4": "PhysLearn", "m4c": "MFourC", "m4": "MFour",
        "phys_m2z": "PhysHUZ", "phys_m2b": "PhysHUB", "phys_m2bmean": "PhysHUBm",
        "m2z": "MTwoZ", "m2b": "MTwoB", "m2bmean": "MTwoBm",
        "phys_m4w": "PhysLearnW", "phys_m4m": "PhysLearnM", "m4w": "MFourW", "m4m": "MFourM",
        "phys_m5": "PhysPat", "phys_m5c": "PhysPatC", "phys_m5k": "PhysPatK", "phys_m5kc": "PhysPatKC",
        "m5": "MFive", "m5c": "MFiveC", "m5k": "MFiveK", "m5kc": "MFiveKC"}
# head-to-head comparisons (row, reference): macros PD/PP/PUp/Pv <metric> <row code> Vs <ref code>
PAIRS = (("m5", "m5c"), ("m5k", "m5kc"), ("m5", "m4"), ("m5c", "m4c"), ("m5k", "m4"), ("m5kc", "m4c"),
         ("phys_m5", "phys_m5c"), ("phys_m5k", "phys_m5kc"))
M2_ARMS = (("m2z", "MTwoZ", "m2"), ("m2b", "MTwoB", "m2"), ("m2bmean", "MTwoBm", "m2"),
           ("phys_m2z", "PhysHUZ", "phys_m2"), ("phys_m2b", "PhysHUB", "phys_m2"), ("phys_m2bmean", "PhysHUBm", "phys_m2"))
OUT = "results/trusted/results_macros.tex"


def fmt(v, d=3, sign=False):
    if v is None or not np.isfinite(v):
        return r"\na"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") or "+-" in s else s.replace("+", "$+$")


def pm(vals, d=3, sign=False):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return r"\na"
    if len(vals) == 1:
        return fmt(vals[0], d, sign)
    return f"{fmt(float(np.mean(vals)), d, sign)}\\,$\\pm$\\,{float(np.std(vals)):.{d}f}"


def main():
    missing = [p for p in ("outputs/trusted.json", "data/trusted_pairs_meta.json") if not os.path.exists(p)]
    if missing:   # the tracked result files are never rewritten from an incomplete set of inputs
        raise SystemExit("missing inputs (see the README, Reproduce, level 2): " + ", ".join(missing))
    m = {}

    def put(name, val):
        m[name] = val

    # ---------------- unpaired ----------------
    if os.path.exists("outputs/compare.json"):
        res = json.load(open("outputs/compare.json"))
        g = defaultdict(list)
        for name, row in res["rows"].items():
            g[row.get("variant", name)].append(row)
        put("UnFrames", str(res["n_eval"])); put("UnCTs", str(len(res["cases"])))
        for key, code in ROWS.items():
            if key not in g:
                continue
            rows = g[key]
            put(f"UW{code}", pm([r["organ_w1"] for r in rows]))
            put(f"UHet{code}", pm([r["hetero_gap"] for r in rows], 4))
            put(f"UTstr{code}", pm([r["tstr_dice"] for r in rows]))
            put(f"UEdge{code}", pm([r["edge_iou"] for r in rows]))
            if "locality" in rows[0]:
                put(f"ULoc{code}", pm([r["locality"] for r in rows], 1))
            if rows[0].get("kind") == "variant":
                put(f"UHus{code}", pm([r["hu_sensitivity"] for r in rows], 4))
            for oid, oname in ((1, "Liver"), (2, "Gb"), (3, "Vessel"), (4, "Kidney")):
                vals = [r["organ_w1_per"].get(str(oid), {}).get("w1") if isinstance(r["organ_w1_per"].get(str(oid)), dict)
                        else r["organ_w1_per"].get(str(oid)) for r in rows]
                put(f"UW{oname}{code}", pm([v for v in vals if v is not None]))
            for oid, oname in ((1, "Liver"), (4, "Kidney")):
                per = [r["hetero_per"].get(str(oid)) for r in rows]
                put(f"UHetFake{oname}{code}", pm([p["fake"] for p in per if p], 3))
                if per and per[0]:
                    put(f"UHetReal{oname}", fmt(per[0]["real"], 3))
            # TSTR vessel class
            vv = [np.mean([t["per_class"].get("3", np.nan) for t in r["tstr_runs"]]) for r in rows if r["tstr_runs"]]
            put(f"UTstrVessel{code}", pm(vv))
    for v, path in (("BZero", "runs/s_us_phys_b0__lab_base.json"), ("MOne", "runs/s_us_phys_m1__lab_pp.json"),
                    ("MTwo", "runs/s_us_phys_m2__lab_base.json")):
        if os.path.exists(path):
            put(f"SusDice{v}", fmt(json.load(open(path))["heldout_sim_dice"], 3))

    # ---------------- paired (TRUSTED) ----------------
    if os.path.exists("outputs/trusted.json"):
        from scipy.stats import wilcoxon
        res = json.load(open("outputs/trusted.json"))
        kid_ids = np.asarray(res["kidney_id"]); kidneys = res["kidneys"]
        g = defaultdict(list)
        for name, row in res["rows"].items():
            g[row.get("variant", name)].append(row)
        put("PrFrames", str(res["n_frames"])); put("PrKidneys", str(len(kidneys)))
        put("PrPatients", str(len({k[:-1] for k in kidneys})))
        put("PrSigma", f"{res['sigma']:g}")

        def per_k(vals):
            v = np.asarray(vals, float)
            return np.array([np.nanmean(v[kid_ids == k]) if np.isfinite(v[kid_ids == k]).any() else np.nan
                             for k in kidneys])
        MT = (("r_kidney", "Kid"), ("r_band", "Band"), ("r_fan", "Fan"), ("r_kidney_dt", "KidD"), ("r_band_dt", "BandD"))
        pk = {key: {mt: np.stack([per_k(r[mt]) for r in rows]) for mt, _ in MT if mt in rows[0]}
              for key, rows in g.items()}
        for key, code in ROWS.items():
            if key not in pk:
                continue
            for mt, mc in MT:
                if mt not in pk[key]:
                    continue
                a = pk[key][mt]
                put(f"P{mc}{code}", pm(list(np.nanmean(a, axis=1)), 3, sign=True))
                put(f"PM{mc}{code}", fmt(float(np.mean(np.nanmean(a, axis=1))), 2))   # mean only, for prose
                ref = "phys_b0" if key.startswith("phys_") else "b0"
                if key in (ref, "ct_hu") or ref not in pk:
                    continue
                x = np.nanmean(a, axis=0); y = np.nanmean(pk[ref][mt], axis=0)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 6:
                    d = x[ok] - y[ok]
                    p = float(wilcoxon(x[ok], y[ok]).pvalue)
                    put(f"PD{mc}{code}", fmt(float(np.median(d)), 3, sign=True))
                    put(f"PP{mc}{code}", (f"{p:.2g}" if p >= 1e-3 else r"{<}\,0.001"))
                    put(f"PUp{mc}{code}", f"{int((d > 0).sum())}/{int(ok.sum())}")
    if os.path.exists("data/trusted_pairs_meta.json"):
        meta = json.load(open("data/trusted_pairs_meta.json"))["kidneys"].values()
        dice = [k["dice_icp"] if k["used"] == "icp" else k["dice_landmarks"] for k in meta]
        put("RegDiceMean", fmt(float(np.mean(dice)), 2)); put("RegDiceMin", fmt(float(np.min(dice)), 2))
        put("RegLdkMedian", fmt(float(np.median([k["ldk1_residual_mm"] for k in meta])), 1))
        put("RegPlanes", str(int(np.sum([k["n_planes"] for k in meta]))))

    # M3 against M3 with its channel zeroed (does the HU channel matter on the same patient?)
    if os.path.exists("outputs/trusted.json") and "m3" in pk and "m3_hu0" in pk:
        x = np.nanmean(pk["m3"]["r_kidney_dt"], axis=0); y = np.nanmean(pk["m3_hu0"]["r_kidney_dt"], axis=0)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 6:
            put("PDKidMThreeVsNoHU", fmt(float(np.median(x[ok] - y[ok])), 3, sign=True))
            p = float(wilcoxon(x[ok], y[ok]).pvalue)
            put("PPKidMThreeVsNoHU", f"{p:.2g}" if p >= 1e-3 else r"{<}\,0.001")
    # M4 against its HU-free control (what the HU adds once the mapping is learned)
    for key, code in (("m4", "MFour"), ("m4w", "MFourW"), ("m4m", "MFourM")):
        if not (os.path.exists("outputs/trusted.json") and key in pk and "m4c" in pk):
            continue
        for mt, mc in (("r_kidney_dt", "KidD"), ("r_band_dt", "BandD")):
            x = np.nanmean(pk[key][mt], axis=0); y = np.nanmean(pk["m4c"][mt], axis=0)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 6:
                p = float(wilcoxon(x[ok], y[ok]).pvalue)
                put(f"PD{mc}{code}VsC", fmt(float(np.median(x[ok] - y[ok])), 3, sign=True))
                put(f"PP{mc}{code}VsC", f"{p:.2g}" if p >= 1e-3 else r"{<}\,0.001")
                put(f"PUp{mc}{code}VsC", f"{int(((x - y)[ok] > 0).sum())}/{int(ok.sum())}")
    # the M2 ablation arms against full M2 (translated) or the full M2 render (physics)
    if os.path.exists("outputs/trusted.json"):
        for key, code, ref in M2_ARMS:
            if key not in pk or ref not in pk:
                continue
            for mt, mc in (("r_kidney_dt", "KidD"), ("r_band_dt", "BandD")):
                x = np.nanmean(pk[key][mt], axis=0); y = np.nanmean(pk[ref][mt], axis=0)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 6:
                    p = float(wilcoxon(x[ok], y[ok]).pvalue)
                    put(f"PD{mc}{code}VsMTwo", fmt(float(np.median(x[ok] - y[ok])), 3, sign=True))
                    put(f"PP{mc}{code}VsMTwo", f"{p:.2g}" if p >= 1e-3 else r"{<}\,0.001")
                    put(f"PUp{mc}{code}VsMTwo", f"{int(((x - y)[ok] > 0).sum())}/{int(ok.sum())}")
    for key, refk in PAIRS:
        if not (os.path.exists("outputs/trusted.json") and key in pk and refk in pk):
            continue
        for mt, mc in (("r_kidney_dt", "KidD"), ("r_band_dt", "BandD")):
            x = np.nanmean(pk[key][mt], axis=0); y = np.nanmean(pk[refk][mt], axis=0)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 6:
                p = float(wilcoxon(x[ok], y[ok]).pvalue)
                tag = f"{mc}{ROWS[key]}Vs{ROWS[refk]}"
                put(f"PD{tag}", fmt(float(np.median(x[ok] - y[ok])), 3, sign=True))
                put(f"PP{tag}", f"{p:.2g}" if p >= 1e-3 else r"{<}\,0.001")
                put(f"PUp{tag}", f"{int(((x - y)[ok] > 0).sum())}/{int(ok.sum())}")
                put(f"PN{tag}", str(int(ok.sum())))
    # M5: how much each patient fit raised the agreement on the planes it was fitted to
    for mode, split, code in (("hu", "halves", "MFive"), ("const", "halves", "MFiveC"),
                              ("hu", "kidney", "MFiveK"), ("const", "kidney", "MFiveKC")):
        gains = []
        for sd in (0, 1, 2):
            fp = f"outputs/m5/fits_{mode}_{split}_s{sd}.json"
            if os.path.exists(fp):
                gains += [f["fit_r_end"] - f["fit_r_start"] for f in json.load(open(fp))]
        if gains:
            put(f"FitGain{code}", fmt(float(np.mean(gains)), 3, sign=True))
    # how much the HU curve raises the training fit over the HU-free control (same folds and seeds)
    for tag, code in (("hu", "MFour"), ("huweak", "MFourW"), ("hum2", "MFourM")):
        fits = []
        for sd in (0, 1, 2):
            ph, pc = f"runs/m4/{tag}_s{sd}.json", f"runs/m4/const_s{sd}.json"
            if os.path.exists(ph) and os.path.exists(pc):
                H, C = json.load(open(ph)), json.load(open(pc))
                fits += [H[f]["train_r_last20"] - C[f]["train_r_last20"] for f in H if f in C]
        if fits:
            put(f"TrainFit{code}VsCMax", fmt(float(np.max(np.abs(fits))), 4))
            put(f"TrainFit{code}VsCMean", fmt(float(np.mean(fits)), 4, sign=True))
    # robustness: the same comparison among the better-registered half of the kidneys
    if os.path.exists("outputs/trusted.json") and os.path.exists("data/trusted_pairs_meta.json"):
        km = json.load(open("data/trusted_pairs_meta.json"))["kidneys"]
        dice = np.array([(km[k]["dice_icp"] if km[k]["used"] == "icp" else km[k]["dice_landmarks"])
                         if k in km else np.nan for k in kidneys])
        hi = dice >= np.nanmedian(dice)
        put("RegDiceMedian", fmt(float(np.nanmedian(dice)), 2)); put("PrKidneysHi", str(int(hi.sum())))
        for key, code in ROWS.items():
            if key not in pk or key in ("b0", "ct_hu") or key.startswith("phys_") or "b0" not in pk:
                continue
            x = np.nanmean(pk[key]["r_kidney_dt"], axis=0)[hi]; y = np.nanmean(pk["b0"]["r_kidney_dt"], axis=0)[hi]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 6:
                p = float(wilcoxon(x[ok], y[ok]).pvalue)
                put(f"PDKidHi{code}", fmt(float(np.median(x[ok] - y[ok])), 3, sign=True))
                put(f"PPKidHi{code}", f"{p:.2g}" if p >= 1e-3 else r"{<}\,0.001")
    # every macro the paper uses exists, pending ones as \na
    expected = ["PrFrames", "PrKidneys", "PrPatients", "PrSigma", "RegDiceMean", "RegDiceMin", "RegLdkMedian",
                "RegPlanes", "PDKidMThreeVsNoHU", "PPKidMThreeVsNoHU", "RegDiceMedian", "PrKidneysHi",
                "TrainFitMFourVsCMax", "TrainFitMFourWVsCMax", "TrainFitMFourMVsCMax",
                "TrainFitMFourVsCMean", "TrainFitMFourWVsCMean", "TrainFitMFourMVsCMean",
                "FitGainMFive", "FitGainMFiveC", "FitGainMFiveK", "FitGainMFiveKC"]
    for code in ("MOne", "MTwo", "MThree", "MThreeNoHU", "MFourC", "MFour"):
        expected += [f"PDKidHi{code}", f"PPKidHi{code}"]
    for mc in ("KidD", "BandD"):
        for code in ("MFour", "MFourW", "MFourM"):
            expected += [f"PD{mc}{code}VsC", f"PP{mc}{code}VsC", f"PUp{mc}{code}VsC"]
    for key, refk in PAIRS:
        for mc in ("KidD", "BandD"):
            tag = f"{mc}{ROWS[key]}Vs{ROWS[refk]}"
            expected += [f"PD{tag}", f"PP{tag}", f"PUp{tag}", f"PN{tag}"]
    for _, code, _ in M2_ARMS:
        for mc in ("KidD", "BandD"):
            expected += [f"PD{mc}{code}VsMTwo", f"PP{mc}{code}VsMTwo", f"PUp{mc}{code}VsMTwo"]
    for code in ("CTHU", "PhysBase", "PhysSub", "PhysHU", "BZero", "MOne", "MTwo", "MThree", "MThreeNoHU",
                 "PhysLearnC", "PhysLearn", "MFourC", "MFour", "PhysHUZ", "PhysHUB", "PhysHUBm",
                 "MTwoZ", "MTwoB", "MTwoBm", "PhysLearnW", "PhysLearnM", "MFourW", "MFourM",
                 "PhysPat", "PhysPatC", "PhysPatK", "PhysPatKC", "MFive", "MFiveC", "MFiveK", "MFiveKC"):
        for mc in ("Kid", "Band", "Fan", "KidD", "BandD"):
            expected += [f"P{mc}{code}", f"PM{mc}{code}", f"PD{mc}{code}", f"PP{mc}{code}", f"PUp{mc}{code}"]
    for k in expected:
        m.setdefault(k, r"\na")
    # prose form of every p value: "$p=0.0088$" or "$p<0.001$" (the PP macros are table cells)
    for k in [k for k in m if k.startswith("PP")]:
        v = m[k]
        m["Pv" + k[2:]] = v if v == r"\na" else ("$p<0.001$" if "<" in v else f"$p={v}$")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("% generated by scripts/paper_macros.py from the result files; do not edit by hand\n")
        for k in sorted(m):
            fh.write(f"\\newcommand{{\\{k}}}{{{m[k]}}}\n")
    print(f"wrote {len(m)} macros -> {OUT}")


if __name__ == "__main__":
    main()
