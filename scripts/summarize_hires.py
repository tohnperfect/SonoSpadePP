#!/usr/bin/env python3
"""Summarize the full-resolution / deformable-registration experiment.

Reads outputs/trusted_hires.json (eval_hires.py), outputs/trusted_def.json (eval_trusted.py on the
128 px planes with deformable registration), the registration records in data/trusted/hires/*.npz
and the per-patient fit logs in runs/hr. Writes docs/TRUSTED_HIRES.md and results/trusted/results_macros_hr.tex
(macros prefixed H, so they never collide with results_macros.tex).

  python scripts/summarize_hires.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_macros import fmt, pm          # noqa: E402

HR_ROWS = {"ct_hu": "CTHU", "phys_b0": "PhysBase", "phys_m2": "PhysHU", "phys_m2b": "PhysHUB",
           "m4chr": "MFourC", "m4hr": "MFour", "m5chr": "MFiveC", "m5hr": "MFive", "m5kchr": "MFiveKC", "m5khr": "MFiveK"}
HR_TITLES = {"ct_hu": "CT slice itself", "phys_b0": "render, base labels", "phys_m2": "render, M2",
             "phys_m2b": "render, M2 backscatter only", "m4chr": "M4c at full resolution (no HU)",
             "m4hr": "M4 at full resolution", "m5chr": "M5c at full resolution, sweep halves (no HU)",
             "m5hr": "M5 at full resolution, sweep halves", "m5kchr": "M5c at full resolution, other kidney (no HU)",
             "m5khr": "M5 at full resolution, other kidney"}
HR_PAIRS = (("phys_m2", "phys_b0"), ("phys_m2b", "phys_b0"), ("m4hr", "m4chr"), ("m5hr", "m5chr"), ("m5khr", "m5kchr"),
            ("m4chr", "phys_b0"), ("m5chr", "m4chr"), ("m5hr", "m4hr"))
D_ROWS = {"b0": "BZero", "b0t": "BZeroT", "m1": "MOne", "m2": "MTwo", "m3": "MThree", "m3_hu0": "MThreeNoHU", "m4c": "MFourC", "m4": "MFour",
          "m5c": "MFiveC", "m5": "MFive", "m5kc": "MFiveKC", "m5k": "MFiveK", "m6c": "MSixC", "m6": "MSix",
          "m7c": "MSevenC", "m7": "MSeven", "m8c": "MEightC", "m8": "MEight", "m7rc": "MSevenRC", "m7r": "MSevenR",
          "ct_hu": "CTHU", "phys_b0": "PhysBase", "phys_m2": "PhysHU"}
REG = {"rig": "Rig", "def": "Def"}
SIG = {1.0: "One", 2.0: "Two"}


def per_kidney(vals, kid_ids, kidneys):
    v = np.asarray(vals, float)
    return np.array([np.nanmean(v[kid_ids == k]) if np.isfinite(v[kid_ids == k]).any() else np.nan for k in kidneys])


def wil(x, y):
    from scipy.stats import wilcoxon
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6:
        return None
    d = x[ok] - y[ok]
    p = float(wilcoxon(x[ok], y[ok]).pvalue) if np.any(d != 0) else 1.0
    return float(np.median(d)), p, int((d > 0).sum()), int(ok.sum())


def ptex(p):
    return "$p<0.001$" if p < 1e-3 else f"$p={p:.2g}$"


def main():
    missing = [p for p in ("data/trusted/hires", "outputs/trusted_def.json", "outputs/trusted_hires.json") if not os.path.exists(p)]
    if missing:   # the tracked result files are never rewritten from an incomplete set of inputs
        raise SystemExit("missing inputs (see the README, Reproduce, level 2): " + ", ".join(missing))
    m, lines = {}, []
    put = m.__setitem__
    # --- registration ----------------------------------------------------------------------------
    metas = [json.loads(str(np.load(f, allow_pickle=False)["meta"])) for f in sorted(glob.glob("data/trusted/hires/*.npz"))]
    if metas:
        dr = np.array([x["dice_rigid"] for x in metas]); dd = np.array([x["dice_def"] for x in metas])
        lr = np.array([np.mean(x["ldk_rigid_mm"]) for x in metas]); ld = np.array([np.mean(x["ldk_def_mm"]) for x in metas])
        l1r = np.array([x["ldk_rigid_mm"][0] for x in metas]); l1d = np.array([x["ldk_def_mm"][0] for x in metas])
        put("HRegN", str(len(metas)))
        put("HRegDiceRig", fmt(dr.mean(), 2)); put("HRegDiceDef", fmt(dd.mean(), 2))
        put("HRegDiceRigMin", fmt(dr.min(), 2)); put("HRegDiceDefMin", fmt(dd.min(), 2))
        put("HRegLdkRig", fmt(np.median(lr), 1)); put("HRegLdkDef", fmt(np.median(ld), 1))
        put("HRegLdkOneRig", fmt(np.median(l1r), 1)); put("HRegLdkOneDef", fmt(np.median(l1d), 1))
        w = wil(-ld, -lr)
        if w:
            put("HRegLdkImproved", f"{w[2]}/{w[3]}"); put("HRegLdkP", ptex(w[1]))
        put("HRegJacMin", fmt(min(x["jac_min"] for x in metas), 2))
        put("HRegUMax", fmt(float(np.median([x["u_max_mm"] for x in metas])), 1))
        put("HRegLam", f"{metas[0]['lam']:g}")
        lines += ["## Registration", "",
                  f"{len(metas)} kidneys. Kidney Dice rigid {dr.mean():.3f} (min {dr.min():.3f}) -> deformable {dd.mean():.3f} "
                  f"(min {dd.min():.3f}). Mean landmark residual per kidney, median over kidneys: rigid {np.median(lr):.1f} mm -> "
                  f"deformable {np.median(ld):.1f} mm (renal pelvis {np.median(l1r):.1f} -> {np.median(l1d):.1f} mm); the "
                  "landmarks are not used by the deformable step. Min Jacobian determinant over all fields "
                  f"{min(x['jac_min'] for x in metas):.2f}; median max displacement {np.median([x['u_max_mm'] for x in metas]):.1f} mm.", ""]
    # --- full resolution -------------------------------------------------------------------------
    if os.path.exists("outputs/trusted_hires.json"):
        res = json.load(open("outputs/trusted_hires.json"))
        kid_ids = np.asarray(res["kidney_id"]); kidneys = res["kidneys"]
        pk = {}
        for key, row in res["rows"].items():
            pk[(row["variant"], row["reg"], float(row["sigma_mm"]))] = {
                mt: per_kidney(row[mt], kid_ids, kidneys) for mt in ("r_kidney_dt", "r_band_dt", "r_fan")}
        lines += ["## Full resolution (0.3 mm, renders only)", "",
                  "Mean over kidneys of the per-kidney mean r (depth-detrended kidney interior / kidney + band; raw fan).", "",
                  "| row | reg | sigma | r kidney | r kidney+band | r fan |", "|---|---|---|---|---|---|"]
        for name, code in HR_ROWS.items():
            for reg in ("rig", "def"):
                for sg in (1.0, 2.0):
                    if (name, reg, sg) not in pk:
                        continue
                    a = pk[(name, reg, sg)]
                    for mt, mc in (("r_kidney_dt", "Kid"), ("r_band_dt", "Band"), ("r_fan", "Fan")):
                        put(f"H{mc}{code}{REG[reg]}{SIG[sg]}", fmt(float(np.nanmean(a[mt])), 3, sign=True))
                    lines.append(f"| {HR_TITLES[name]} | {reg} | {sg:g} mm | {np.nanmean(a['r_kidney_dt']):+.3f} | "
                                 f"{np.nanmean(a['r_band_dt']):+.3f} | {np.nanmean(a['r_fan']):+.3f} |")
        lines += ["", "Paired differences (median over kidneys, Wilcoxon p, kidneys improved):", "",
                  "| comparison | reg | sigma | kidney | kidney+band |", "|---|---|---|---|---|"]
        for a_, b_ in HR_PAIRS:
            for reg in ("rig", "def"):
                for sg in (1.0, 2.0):
                    if (a_, reg, sg) not in pk or (b_, reg, sg) not in pk:
                        continue
                    cells = []
                    for mt, mc in (("r_kidney_dt", "Kid"), ("r_band_dt", "Band")):
                        w = wil(pk[(a_, reg, sg)][mt], pk[(b_, reg, sg)][mt])
                        if w is None:
                            cells.append("n/a"); continue
                        tag = f"H{mc}{HR_ROWS[a_]}Vs{HR_ROWS[b_]}{REG[reg]}{SIG[sg]}"
                        put(f"HD{tag[1:]}", fmt(w[0], 3, sign=True)); put(f"Hv{tag[1:]}", ptex(w[1]))
                        put(f"HUp{tag[1:]}", f"{w[2]}/{w[3]}")
                        cells.append(f"{w[0]:+.3f} (p={w[1]:.2g}, {w[2]}/{w[3]})")
                    lines.append(f"| {HR_TITLES[a_]} - {HR_TITLES[b_]} | {reg} | {sg:g} mm | " + " | ".join(cells) + " |")
        # registration effect on the same row
        lines += ["", "Deformable minus rigid, same row (sigma 1 mm):", "", "| row | kidney | kidney+band |", "|---|---|---|"]
        for name, code in HR_ROWS.items():
            if (name, "rig", 1.0) not in pk or (name, "def", 1.0) not in pk:
                continue
            cells = []
            for mt, mc in (("r_kidney_dt", "Kid"), ("r_band_dt", "Band")):
                w = wil(pk[(name, "def", 1.0)][mt], pk[(name, "rig", 1.0)][mt])
                if w:
                    put(f"HD{mc}{code}DefVsRig", fmt(w[0], 3, sign=True)); put(f"Hv{mc}{code}DefVsRig", ptex(w[1]))
                    cells.append(f"{w[0]:+.3f} (p={w[1]:.2g}, {w[2]}/{w[3]})")
                else:
                    cells.append("n/a")
            lines.append(f"| {HR_TITLES[name]} | " + " | ".join(cells) + " |")
        # per-patient fit gains
        for scheme, mode, code in (("halves", "hu", "MFive"), ("halves", "const", "MFiveC"), ("kidney", "hu", "MFiveK"),
                                   ("kidney", "const", "MFiveKC")):
            fp = f"runs/hr/fits_{scheme}_{mode}_def.json"
            if os.path.exists(fp):
                g = [f["fit_r_end"] - f["fit_r_start"] for f in json.load(open(fp))]
                put(f"HFitGain{code}Def", fmt(float(np.mean(g)), 3, sign=True))
    # --- 128 px with deformable registration -----------------------------------------------------
    if os.path.exists("outputs/trusted_def.json"):
        res = json.load(open("outputs/trusted_def.json"))
        kid_ids = np.asarray(res["kidney_id"]); kidneys = res["kidneys"]
        groups = {}
        for name, row in res["rows"].items():
            groups.setdefault(row.get("variant", name), []).append(row)
        pk = {g: {mt: np.stack([per_kidney(r[mt], kid_ids, kidneys) for r in rows]) for mt in ("r_kidney_dt", "r_band_dt")}
              for g, rows in groups.items()}
        lines += ["", "## 128 px pipeline with deformable registration", "",
                  "| row | r kidney | r kidney+band | vs B0 (kidney) |", "|---|---|---|---|"]
        for g, code in D_ROWS.items():
            if g not in pk:
                continue
            for mt, mc in (("r_kidney_dt", "KidD"), ("r_band_dt", "BandD")):
                put(f"HDef{mc}{code}", pm(list(np.nanmean(pk[g][mt], axis=1)), 3, sign=True))
                put(f"HDefM{mc}{code}", fmt(float(np.mean(np.nanmean(pk[g][mt], axis=1))), 2))
            ref = "phys_b0" if g.startswith("phys") else "b0"
            cmp = ""
            if g not in (ref, "ct_hu") and ref in pk:
                w = wil(np.nanmean(pk[g]["r_kidney_dt"], 0), np.nanmean(pk[ref]["r_kidney_dt"], 0))
                if w:
                    put(f"HDefDKidD{code}", fmt(w[0], 3, sign=True)); put(f"HDefvKidD{code}", ptex(w[1]))
                    put(f"HDefUpKidD{code}", f"{w[2]}/{w[3]}")
                    cmp = f"{w[0]:+.3f} (p={w[1]:.2g}, {w[2]}/{w[3]})"
            lines.append(f"| {g} | {pm(list(np.nanmean(pk[g]['r_kidney_dt'], axis=1)), 3, sign=True)} | "
                         f"{pm(list(np.nanmean(pk[g]['r_band_dt'], axis=1)), 3, sign=True)} | {cmp} |".replace("$", ""))
        if "m4" in pk and "m4c" in pk:
            for mt, mc in (("r_kidney_dt", "KidD"), ("r_band_dt", "BandD")):
                w = wil(np.nanmean(pk["m4"][mt], 0), np.nanmean(pk["m4c"][mt], 0))
                if w:
                    put(f"HDefD{mc}MFourVsC", fmt(w[0], 3, sign=True)); put(f"HDefv{mc}MFourVsC", ptex(w[1]))
                    lines.append(f"\nM4 - M4c ({mc}): {w[0]:+.3f} (p={w[1]:.2g}, {w[2]}/{w[3]})")
        for a_, b_ in (("m6", "m6c"), ("m6c", "m4c"), ("m6", "m4"), ("m5", "m5c"), ("m5k", "m5kc"),
                       ("m7", "m7c"), ("m7c", "m6c"), ("m7", "m6"), ("m8", "m8c"), ("m8c", "m7c"), ("m8", "m7"),
                       ("m7r", "m7rc"), ("m7rc", "m7c"), ("m7c", "b0t"), ("b0t", "b0")):
            if a_ in pk and b_ in pk:
                for mt, mc in (("r_kidney_dt", "KidD"), ("r_band_dt", "BandD")):
                    w = wil(np.nanmean(pk[a_][mt], 0), np.nanmean(pk[b_][mt], 0))
                    if w:
                        code = f"{D_ROWS.get(a_, a_)}Vs{D_ROWS.get(b_, b_)}"
                        put(f"HDefD{mc}{code}", fmt(w[0], 3 if abs(w[0]) >= 0.01 else 4, sign=True)); put(f"HDefv{mc}{code}", ptex(w[1]))
                        put(f"HDefUp{mc}{code}", f"{w[2]}/{w[3]}")
                        lines.append(f"\n{a_} - {b_} ({mc}): {w[0]:+.4f} (p={w[1]:.2g}, {w[2]}/{w[3]})")
        if "m3" in pk and "m3_hu0" in pk:
            w = wil(np.nanmean(pk["m3"]["r_kidney_dt"], 0), np.nanmean(pk["m3_hu0"]["r_kidney_dt"], 0))
            if w:
                put("HDefDKidDMThreeVsNoHU", fmt(w[0], 3, sign=True)); put("HDefvKidDMThreeVsNoHU", ptex(w[1]))
    if os.path.exists("outputs/m6_hu_use.json"):                      # scripts/m6_hu_use.py
        u = json.load(open("outputs/m6_hu_use.json"))
        put("HMSixHUChange", fmt(u["hu_change"], 2)); put("HMSixSD", fmt(u["sd_m6"], 2)); put("HMSixCSD", fmt(u["sd_m6c"], 2))
        put("HMSixTrainR", fmt(u["train_r_m6"], 3)); put("HMSixCTrainR", fmt(u["train_r_m6c"], 3))
        put("HMSixParams", f"{round(u['n_params_ct'] / 1000)}k")
        lines += ["", f"M6 HU use: mean |change of log factor| when z and HU are flattened per class {u['hu_change']:.3f} "
                  f"(spread of log factor: M6 {u['sd_m6']:.2f}, M6c {u['sd_m6c']:.2f}); training r M6 {u['train_r_m6']:.3f}, "
                  f"M6c {u['train_r_m6c']:.3f}"]
    # --- alignment-tolerant agreement (scripts/eval_aligned.py) ----------------------------------
    if os.path.exists("outputs/trusted_aligned.json"):
        al = json.load(open("outputs/trusted_aligned.json"))
        kid_ids = np.asarray(al["kidney_id"]); kidneys = al["kidneys"]
        AL = {}
        for row in al["rows"].values():
            for key in ("r_kidney_before", "r_band_before", "r_kidney_after", "r_band_after"):
                AL.setdefault(row["variant"], {}).setdefault(key, []).append(per_kidney(row[key], kid_ids, kidneys))
        AL = {v: {k: np.nanmean(np.stack(x), 0) for k, x in d.items()} for v, d in AL.items()}
        lines += ["", f"## Alignment-tolerant agreement (smooth warp of each output, {al['grid']}x{al['grid']} grid, "
                  f"at most {al['max_px']:g} px)", "",
                  "| row | kidney before | kidney after | band before | band after |", "|---|---|---|---|---|"]
        for v, d in AL.items():
            code = D_ROWS.get(v, v)
            for key, mc in (("r_kidney_before", "KidBefore"), ("r_kidney_after", "KidAfter"),
                            ("r_band_before", "BandBefore"), ("r_band_after", "BandAfter")):
                put(f"HAl{mc}{code}", fmt(float(np.nanmean(d[key])), 3, sign=True))
            lines.append(f"| {v} | {np.nanmean(d['r_kidney_before']):+.3f} | {np.nanmean(d['r_kidney_after']):+.3f} | "
                         f"{np.nanmean(d['r_band_before']):+.3f} | {np.nanmean(d['r_band_after']):+.3f} |")
        for a_, b_ in (("m6", "m6c"), ("m7", "m7c"), ("m8", "m8c"), ("m7c", "b0"), ("m8c", "m7c"), ("m7r", "m7rc"),
                       ("m7rc", "m7c"), ("m7c", "b0t"), ("b0t", "b0")):
            if a_ in AL and b_ in AL:
                for key, mc in (("r_kidney_after", "Kid"), ("r_band_after", "Band")):
                    w = wil(AL[a_][key], AL[b_][key])
                    if w:
                        code = f"{D_ROWS.get(a_, a_)}Vs{D_ROWS.get(b_, b_)}"
                        put(f"HAlD{mc}{code}", fmt(w[0], 3 if abs(w[0]) >= 0.01 else 4, sign=True))
                        put(f"HAlv{mc}{code}", ptex(w[1])); put(f"HAlUp{mc}{code}", f"{w[2]}/{w[3]}")
                        lines.append(f"\naligned {a_} - {b_} ({mc}): {w[0]:+.4f} (p={w[1]:.2g}, {w[2]}/{w[3]})")
    # --- paper Table 3: 128 px vs 0.3 mm, rigid vs deformable (kidney interior, detrended, sigma 1 mm at 0.3 mm)
    src = {"RigLo": ("outputs/trusted.json", None), "DefLo": ("outputs/trusted_def.json", None),
           "RigHi": ("outputs/trusted_hires.json", "rig"), "DefHi": ("outputs/trusted_hires.json", "def")}
    rows_t3 = {"CT": ("ct_hu", "ct_hu"), "Base": ("phys_b0", "phys_b0"), "MTwoR": ("phys_m2", "phys_m2"),
               "BZero": ("b0", None), "MTwo": ("m2", None), "MFourC": ("m4c", "m4chr"), "MFour": ("m4", "m4hr"),
               "MFiveC": ("m5c", "m5chr"), "MFive": ("m5", "m5hr"), "MSixC": ("m6c", None), "MSix": ("m6", None)}
    cache = {}

    def col(colname, variant, mt="r_kidney_dt"):
        path, reg = src[colname]
        if not os.path.exists(path):
            return None
        if path not in cache:
            cache[path] = json.load(open(path))
        res = cache[path]
        kid = np.asarray(res["kidney_id"]); ks = res["kidneys"]
        rows = [r for n, r in res["rows"].items() if r.get("variant", n) == variant
                and (reg is None or (r.get("reg") == reg and float(r.get("sigma_mm")) == 1.0))]
        if not rows:
            return None
        return np.nanmean(np.stack([per_kidney(r[mt], kid, ks) for r in rows]), 0)

    for rc, (lo, hi) in rows_t3.items():
        for cn in src:
            v = col(cn, lo if cn.endswith("Lo") else hi) if (lo if cn.endswith("Lo") else hi) else None
            put(f"HT{rc}{cn}", r"\na" if v is None or not np.isfinite(v).any() else fmt(float(np.nanmean(v)), 3, sign=True))
    for rc, (a_lo, a_hi, b_lo, b_hi) in {"MFour": ("m4", "m4hr", "m4c", "m4chr"), "MFive": ("m5", "m5hr", "m5c", "m5chr"),
                                         "MSix": ("m6", None, "m6c", None)}.items():
        for cn in src:
            x = col(cn, a_lo if cn.endswith("Lo") else a_hi); y = col(cn, b_lo if cn.endswith("Lo") else b_hi)
            w = wil(x, y) if x is not None and y is not None else None
            put(f"HTD{rc}{cn}", r"\na" if w is None else ("0.0000" if round(w[0], 4) == 0 else fmt(w[0], 4, sign=True)))
            put(f"HTP{rc}{cn}", r"\na" if w is None else (r"{<}\,0.001" if w[1] < 1e-3 else f"{w[1]:.2g}"))
            put(f"HTUp{rc}{cn}", r"\na" if w is None else f"{w[2]}/{w[3]}")
    for rc, (a_lo, a_hi, b_lo, b_hi) in {"MFourBand": ("m4", "m4hr", "m4c", "m4chr")}.items():
        for cn in src:
            x = col(cn, a_lo if cn.endswith("Lo") else a_hi, "r_band_dt"); y = col(cn, b_lo if cn.endswith("Lo") else b_hi, "r_band_dt")
            w = wil(x, y) if x is not None and y is not None else None
            put(f"HTD{rc}{cn}", r"\na" if w is None else ("0.0000" if round(w[0], 4) == 0 else fmt(w[0], 4, sign=True)))
            put(f"HTv{rc}{cn}", r"\na" if w is None else ptex(w[1]))
    # deformable minus rigid at 128 px, translated rows
    for v, code in (("b0", "BZero"), ("m2", "MTwo"), ("m4c", "MFourC"), ("m4", "MFour")):
        x, y = col("DefLo", v), col("RigLo", v)
        w = wil(x, y) if x is not None and y is not None else None
        if w:
            put(f"HDefVsRig{code}", fmt(w[0], 3, sign=True)); put(f"HDefVsRigv{code}", ptex(w[1]))
    os.makedirs("docs", exist_ok=True)
    open("docs/TRUSTED_HIRES.md", "w").write("# Same-patient test at full resolution and with deformable registration\n\n"
                                            + "\n".join(lines) + "\n")
    with open("results/trusted/results_macros_hr.tex", "w") as fh:
        fh.write("% generated by scripts/summarize_hires.py; do not edit by hand\n")
        for k in sorted(m):
            fh.write(f"\\newcommand{{\\{k}}}{{{m[k]}}}\n")
    print(f"wrote docs/TRUSTED_HIRES.md and {len(m)} macros -> results/trusted/results_macros_hr.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
