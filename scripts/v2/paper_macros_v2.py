#!/usr/bin/env python3
"""Write results/paper_macros/results_macros.tex: every number the paper reports, as LaTeX macros.

Sources (no hand-typed numbers in the paper):
  results/v2/rows/*.csv     per-seed metric rows (eval_v2.py), summarized here as in make_v2_tables.py
  results/v2/latency.csv    bench_latency_v2.py
  results/v2/latency_endtoend.csv    bench_endtoend.py
  results/v2/sus.csv        eval_sus_v2.py (segmenter Dice)
  results/v2/selection.csv  WP1 selection runs (select_v2.py) -> \\RSel... macros and the generated tab_selection.tex
  --spp-summary             outputs/m9_summary.json (same-patient TRUSTED scores, summarize_m9.py)
  --merge-summary           outputs/merge_summary.json (same-patient pass criteria, summarize_merge.py)
  --spp-macros-dir          results/trusted (generated macro files of the TRUSTED data and registration facts)

A value that does not exist yet is written as \\na (a gray dash), so the draft always builds.

  python -I scripts/v2/paper_macros_v2.py [--out results/paper_macros/results_macros.tex]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# model key in the rows -> macro stem (letters only: LaTeX macro names cannot hold digits)
MODELS = {
    "physics": "Phys", "v1": "VOne", "v1_paper": "VOnePaper", "cut_paper": "CUT", "cyclegan_paper": "Cyc",
    "cyclegan_sc_paper": "SCyc",
    # filled as the work packages finish (\na until then)
    "v2": "VTwo", "v2_no_fm": "VTwoNoFM", "v2_no_sus": "VTwoNoSUS", "v2_no_style": "VTwoNoStyle",
    "v2_no_speckle": "VTwoNoSpk",
    # tier-0 component models: WP1 gated settings, WP3 segmenter inits, WP2 SEAN (E2a, E2b)
    "wp1_e1b_echocare_tis0.1": "WOneEcho", "wp1_e1b_dinov2_tis0.1": "WOneDino",
    "sususfm": "SusUsfm", "susdinov2": "SusDino", "susechocare": "SusEcho",
    "wp2_e2a": "SeanA", "wp2_e2b": "SeanB",
}
# (metric, tissue, extractor, protocol prefix, macro stem, decimals)
METRICS = [
    ("edge_iou", "all", "", "v1", "Edge", 3),
    ("organ_w1", "all", "", "v1", "OrganW", 3),
    ("locality_band0", "all", "", "v1", "Loc", 1),
    ("locality_band8", "all", "", "band8", "LocBand", 1),
    ("tstr_dice", "all", "", "v1 tstr_seed", "Tstr", 3),
    ("tissue_kid", "mean", "usfm", "crop16", "TKidUSFM", 0),
    ("tissue_kid", "mean", "dinov2", "crop16", "TKidDINO", 0),
    ("frame_kid", "all", "dinov2", "fan", "FKidFan", 0),
    ("frame_kid", "all", "dinov2", "raw", "FKidRaw", 0),
    ("nakagami_logm_w1", "mean", "", "", "LogM", 2),
    ("nakagami_m_w1", "mean", "", "", "RawM", 2),
    ("speckle_size_err", "mean", "", "", "SpkSize", 2),
    ("diversity_lpips", "all", "", None, "Div", 3),
]
SPP_ROWS = {"b0": "BZero", "b0t": "BZeroT", "m7c": "ApOne", "m9ac": "ApTwo", "labwarp": "LabWarp", "copy": "Copy",
            "m7": "ApOneHU", "m9a": "ApTwoHU",
            # rows the merged evaluation adds
            "b0t_v2": "BZeroTVTwo", "m7c_v2": "ApOneVTwo", "style_v2": "ApTwoStyle"}
SPP_PAIRS = {"m7c-b0t": "ApOneVsBZeroT", "b0t-b0": "BZeroTVsBZero", "m9ac-m7c": "ApTwoVsApOne",
             "m9ac-labwarp": "ApTwoVsLabWarp", "m7-m7c": "HUApOne", "m9a-m9ac": "HUApTwo"}


def _load_tables_module():
    spec = importlib.util.spec_from_file_location("make_v2_tables", os.path.join(HERE, "make_v2_tables.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt(x, d):
    if x is None or not math.isfinite(x):
        return r"\na"
    t = f"{x:.{d}f}"
    return (r"\ensuremath{-}" + t[1:]) if t.startswith("-") and float(t) != 0 else t.lstrip("-")   # a real minus


def fmt_p(p):
    if p is None or not math.isfinite(p):
        return r"\na"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"
    return f"{p:.2g}"


SEL_ENC = {"echocare": ("Echo", "EchoCare"), "usfm": ("Usfm", "USFM"), "dinov2": ("Dino", "DINOv2")}
SEL_W = {"0.1": "Pt", "0.25": "Qtr", "0.3": "PtThree", "0.5": "Half", "1": "One", "2": "Two"}
SEL_COLS = (("val_fan_kid_usfm", "Kid", 2), ("val_fan_kid_dinov2", "KidDino", 0), ("edge_iou", "Edge", 3),
            ("locality_band0", "LocA", 1), ("locality_band8", "LocB", 1))
SEL_KEYS = ("locality_band0", "locality_band8", "edge_iou")       # feasibility, as in pick_wp1.py


def _sel_parse(name):
    """selection run name -> (macro stem, encoder label, term label, weight label) or None."""
    import re
    if name == "v1sel":
        return "Ref", "", "SonoSPADE (reference)", ""
    m = re.fullmatch(r"e1([abc])_(\w+?)_(.+)", name)
    if not m or m.group(2) not in SEL_ENC:
        return None
    arm, (estem, elabel), rest = m.group(1), SEL_ENC[m.group(2)], m.group(3)
    w = re.findall(r"(adv|tis)([0-9.]+)", rest)
    if arm == "c":
        return f"{estem}Both", elabel, "both terms", ", ".join(v for _, v in w)
    if not w or w[0][1] not in SEL_W:
        return None
    term = "adversarial" if arm == "a" else "feature matching"
    return f"{estem}{'Adv' if arm == 'a' else 'Tis'}{SEL_W[w[0][1]]}", elabel, term, w[0][1]


def write_selection(path, table_path, emit):
    """WP1 selection runs: \\RSel<stem><metric> macros and a generated appendix table."""
    rows = list(csv.DictReader(open(path))) if path and os.path.exists(path) else []
    ref = next((r for r in rows if r["name"] == "v1sel"), None)
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else math.nan
    for k, mstem, d in (("edge_iou", "Edge", 3), ("locality_band0", "LocA", 1), ("locality_band8", "LocB", 1)):
        emit(f"RSel{mstem}Thr", fmt(0.95 * f(ref, k), d) if ref else r"\na")
    lines = []
    for r in rows:
        parsed = _sel_parse(r["name"])
        if parsed is None:
            continue
        stem, enc, term, w = parsed
        for k, mstem, d in SEL_COLS:
            emit(f"RSel{stem}{mstem}", fmt(f(r, k), d))
        ok = ref is not None and all(f(r, k) >= 0.95 * f(ref, k) for k in SEL_KEYS)
        cells = [enc, term, w] + [fmt(f(r, k), d) if math.isfinite(f(r, k)) else "--" for k, _, d in SEL_COLS]
        cells.append("" if stem == "Ref" else ("yes" if ok else "no"))
        lines.append(" & ".join(cells) + r"\\" + (r"\midrule" if stem == "Ref" else ""))
    out = [r"% Generated by scripts/v2/paper_macros_v2.py from results/v2/selection.csv. Do not edit by hand.",
           r"\begin{tabular}{@{}lllcccccc@{}}", r"\toprule",
           r"Encoder & Term & Weight & KID, USFM $\downarrow$ & KID, DINOv2 $\downarrow$ & Edge IoU $\uparrow$ & "
           r"Locality $\uparrow$ & Locality, 8\,px band $\uparrow$ & Criteria met\\", r"\midrule"]
    out += lines or [r"\multicolumn{9}{c}{\na}\\"]
    out += [r"\bottomrule", r"\end{tabular}"]
    if table_path:
        os.makedirs(os.path.dirname(os.path.abspath(table_path)), exist_ok=True)
        open(table_path, "w").write("\n".join(out) + "\n")


def write_selection_extra(emit):
    r"""WP2 INADE (results/v2/selection_wp2.csv) and WP5 (selection_wp5.csv) selection runs: \RSelIn<w><metric>,
    \RSelSpk<stem><metric> (no generated table; the text cites them)."""
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else math.nan
    if os.path.exists("results/v2/selection_wp2.csv"):
        for r in csv.DictReader(open("results/v2/selection_wp2.csv")):
            if r["name"].startswith("e2i_ms"):
                w = SEL_W.get(r["name"].removeprefix("e2i_ms"))
                if w:
                    for k, m, d in (("val_fan_kid_usfm", "Kid", 1), ("diversity_lpips", "Div", 3), ("edge_iou", "Edge", 3),
                                    ("locality_band0", "LocA", 1), ("locality_band8", "LocB", 1)):
                        emit(f"RSelIn{w}{m}", fmt(f(r, k), d))
    if os.path.exists("results/v2/selection_wp5.csv"):
        stems = {"v1sel": "Ref", "e5b_noise": "Noise"}
        for r in csv.DictReader(open("results/v2/selection_wp5.csv")):
            n = r["name"]
            st = stems.get(n) or (f"Loss{SEL_W.get(n.removeprefix('e5a_spk'), '')}" if n.startswith("e5a_spk") else
                                  ("Both" if n.startswith("e5c_") else None))
            if st:
                for k, m, d in (("val_fan_logm_w1", "LogM", 3), ("val_fan_speckle_size_err", "Size", 3),
                                ("val_fan_kid_usfm", "Kid", 1), ("edge_iou", "Edge", 3), ("locality_band0", "LocA", 1),
                                ("locality_band8", "LocB", 1)):
                    emit(f"RSelSpk{st}{m}", fmt(f(r, k), d))


MERGE_CRIT = {("m7c", "b0t"): "ApOneVsBZeroT", ("m9ac", "labwarp"): "ApTwoVsLabWarp", ("m9ac", "m7c"): "ApTwoVsApOne",
              ("b0_v2", "b0"): "BZeroVTwoVsBZero", ("m7c_v2", "b0t_v2"): "ApOneVTwoVsBZeroTVTwo",
              ("m7c_v2", "m7c"): "ApOneVTwoVsApOne", ("m9ac_v2", "labwarp"): "ApTwoVTwoVsLabWarp",
              ("m9ac_v2", "m7c_v2"): "ApTwoVTwoVsApOneVTwo", ("m7c", "p2p"): "ApOneVsPix",
              ("p2p", "b0t"): "PixVsBZeroT", ("m7c", "phys_b0"): "ApOneVsPhys", ("retrieve", "m7c"): "RetVsApOne", ("m7x", "m7c"): "HybVsApOne",
              ("m7x", "retrieve"): "HybVsRet"}
MERGE_MEAS = {"kidney dt": "KidDt", "band dt": "BandDt", "kidney": "Kid", "fan": "Fan", "kidney al": "KidAl",
              "band al": "BandAl", "kidney W1": "WOne"}
MERGE_BIN = {"all": "", "0-7 deg": "BinA", "7-13 deg": "BinB", "13-90 deg": "BinC"}
MERGE_ROWS = {"b0": "BZero", "b0t": "BZeroT", "m7c": "ApOne", "m9ac": "ApTwo", "labwarp": "LabWarp", "copy": "Copy",
              "b0_v2": "BZeroVTwo", "b0t_v2": "BZeroTVTwo", "m7c_v2": "ApOneVTwo", "m9ac_v2": "ApTwoVTwo",
              "p2p": "Pix", "phys_b0": "Phys", "retrieve": "Ret", "m7x": "Hyb"}


def write_merge(path, emit):
    r"""The same-patient pass criteria (scripts/summarize_merge.py): \RMg<criterion>Verdict, \RMg<criterion><measure>
    <bin>{D,P,Up}, \RMg<criterion>WOne{D,P}, and row means \RMgRow<row><measure>. Missing values print as \na."""
    m = json.load(open(path)) if path and os.path.exists(path) else {"criteria": [], "rows": {}}
    have = {(c["a"], c["b"]): c for c in m["criteria"]}
    for (a, b), cs in MERGE_CRIT.items():
        c = have.get((a, b))
        emit(f"RMg{cs}Verdict", c["verdict"].lower() if c else r"\na")
        checks = {(k["measure"], k["bin"]): k for k in (c["checks"] if c else [])}
        for meas, ms in MERGE_MEAS.items():
            if meas == "kidney W1":
                continue
            for bn, bs in MERGE_BIN.items():
                k = checks.get((meas, bn))
                emit(f"RMg{cs}{ms}{bs}D", fmt(k["median_diff"], 3) if k and k["median_diff"] is not None else r"\na")
                emit(f"RMg{cs}{ms}{bs}P", fmt_p(k["p"]) if k and k["p"] is not None else r"\na")
                emit(f"RMg{cs}{ms}{bs}Up", k["up"] if k and k["up"] else r"\na")
        w1 = (c or {}).get("kidney_w1") or {}
        emit(f"RMg{cs}WOneD", fmt(w1["median_diff"], 3) if w1.get("median_diff") is not None else r"\na")
        emit(f"RMg{cs}WOneP", fmt_p(w1["p"]) if w1.get("p") is not None else r"\na")
    for row, rs in MERGE_ROWS.items():
        r = m["rows"].get(row, {})
        for meas, ms in MERGE_MEAS.items():
            v = r.get(meas)
            emit(f"RMgRow{rs}{ms}", fmt(v, 3) if v is not None else r"\na")


USBGYM_STAGES = {"physics": "Phys", "cut": "Cut", "cyclegan": "CycGan", "v1_published": "VOnePub", "v1r": "VOneR",
                 "v2": "VTwo"}
USBGYM_CMP = {"primary: v2 - v1r (gap)": "Prim", "v2 - v1r (matched-arm final error)": "Matched",
              "secondary: v2 - v1_published (gap)": "SecVTwoPub", "secondary: v1r - v1_published (gap)": "SecVOneRPub"}


def _p2(p):
    """fmt_p, but two decimals from 0.1 up (0.10, not 0.1)."""
    return f"{p:.2f}" if p is not None and math.isfinite(p) and p >= 0.1 else fmt_p(p)


def write_usbgym(path, emit):
    r"""USB Gym closed loop (scripts/v2/usbgym/summarize_usbgym.py): per stage \RUg<stage>{Gap,Lo,Hi,Agree,Matched,Ct}
    (mm, the learned-mask gap with its patient-cluster bootstrap CI), per comparison \RUg<cmp>{D,P,PHolm,Agree}."""
    d = json.load(open(path)) if path and os.path.exists(path) else {"stages": {}, "comparisons": {}}
    for st, ss in USBGYM_STAGES.items():
        r = d["stages"].get(st)
        rule = (r or {}).get("rule") or {}
        emit(f"RUg{ss}Gap", fmt(rule.get("mean"), 2) if r else r"\na")
        emit(f"RUg{ss}Lo", fmt(rule["ci"][0], 2) if rule.get("ci") else r"\na")
        emit(f"RUg{ss}Hi", fmt(rule["ci"][1], 2) if rule.get("ci") else r"\na")
        emit(f"RUg{ss}Agree", rule["agree"] if rule.get("agree") is not None else r"\na")
        emit(f"RUg{ss}Matched", fmt(r["matched_mean"], 2) if r else r"\na")
        emit(f"RUg{ss}Ct", fmt(r["ctmask_mean"], 2) if r else r"\na")
    for cmp, cs in USBGYM_CMP.items():
        c = d["comparisons"].get(cmp)
        emit(f"RUg{cs}D", fmt(c["mean"], 2) if c else r"\na")
        emit(f"RUg{cs}P", _p2(c["p"]) if c else r"\na")
        emit(f"RUg{cs}PHolm", _p2(c["p_holm"]) if c and "p_holm" in c else r"\na")
        emit(f"RUg{cs}Agree", c["agree"] if c else r"\na")



def write_usbgym_t1g(path, gate, ref, emit):
    r"""Tier 1 in the USB Gym closed loop (scripts/v2/usbgym/summarize_t1g.py): the t1g stage
    \RUgTOneG{Gap,Lo,Hi,Agree,Matched,Ct}; the primary t1g - v1r gap difference \RUgTOneG{D,P,DAgree}; the learned-mask
    final error difference \RUgTOneG{MatchedD,MatchedP,MatchedAgree}; the fine-tune's held-out TRUSTED r (gate G2)
    \RUgTOneG{R,StartR} and the paper's tier-1 model of the same seed and fold \RUgTOneGRefR."""
    d = json.load(open(path)) if path and os.path.exists(path) else {"stages": {}, "tests": {}}
    r = d["stages"].get("t1g")
    rule = (r or {}).get("rule") or {}
    emit("RUgTOneGGap", fmt(rule.get("mean"), 2) if r else r"\na")
    emit("RUgTOneGLo", fmt(rule["ci"][0], 2) if rule.get("ci") else r"\na")
    emit("RUgTOneGHi", fmt(rule["ci"][1], 2) if rule.get("ci") else r"\na")
    emit("RUgTOneGAgree", rule["agree"] if rule.get("agree") is not None else r"\na")
    emit("RUgTOneGMatched", fmt(r["matched_mean"], 2) if r else r"\na")
    emit("RUgTOneGCt", fmt(r["ctmask_mean"], 2) if r else r"\na")
    for key, stem in (("primary: t1g - v1r (gap)", ""), ("secondary b: t1g - v1r (matched-arm final error)", "Matched")):
        c = d["tests"].get(key)
        emit(f"RUgTOneG{stem}D", fmt(c["mean"], 2) if c else r"\na")
        emit(f"RUgTOneG{stem}P", _p2(c["p"]) if c else r"\na")
        emit(f"RUgTOneG{stem or 'D'}Agree", c["agree"] if c else r"\na")
    g = json.load(open(gate)) if gate and os.path.exists(gate) else {}
    emit("RUgTOneGR", fmt(g.get("G2_t1g_heldout_r"), 2))
    emit("RUgTOneGStartR", fmt(g.get("G2_v1r_heldout_r"), 2))
    rr = json.load(open(ref)).get("0", {}).get("heldout_r") if ref and os.path.exists(ref) else None
    emit("RUgTOneGRefR", fmt(rr, 2))

NOHARM_CHECKS = {"noharm_tissue_kid_usfm": "Kid", "noharm_nakagami_logm_w1": "LogM", "invariant1_locality_band0": "LocA",
                 "invariant1_locality_band8": "LocB", "invariant1_edge_iou": "Edge"}


def write_noharm(path, emit, tag):
    r"""A no-harm gate (gate_v2.py --mode noharm): \RNh<tag><check>{Cand,Ctrl,Pct} (Pct: candidate change in percent
    of the control mean) and \RNh<tag>Gate."""
    d = json.load(open(path)) if path and os.path.exists(path) else {"checks": {}}
    emit(f"RNh{tag}Gate", d.get("gate", r"\na").lower())
    for ck, cs in NOHARM_CHECKS.items():
        c = d["checks"].get(ck)
        cm, km = (c or {}).get("cand_mean"), (c or {}).get("control_mean")
        nd = 3 if cs == "Edge" else (0 if cs == "Kid" else 2)
        emit(f"RNh{tag}{cs}Cand", fmt(cm, nd) if cm is not None else r"\na")
        emit(f"RNh{tag}{cs}Ctrl", fmt(km, nd) if km is not None else r"\na")
        emit(f"RNh{tag}{cs}Pct", fmt(100 * (cm / km - 1), 0) if cm is not None and km else r"\na")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", default="results/v2/rows")
    ap.add_argument("--latency", default="results/v2/latency.csv")
    ap.add_argument("--latency-ete", default="results/v2/latency_endtoend.csv")
    ap.add_argument("--sus", default="results/v2/sus.csv")
    ap.add_argument("--spp-summary", default="outputs/m9_summary.json")
    ap.add_argument("--spp-macros-dir", default="results/trusted",
                    help="folder with results_macros.tex and results_macros_hr.tex (scripts/paper_macros.py, "
                         "scripts/summarize_hires.py)")
    ap.add_argument("--selection", default="results/v2/selection.csv")
    ap.add_argument("--sel-table", default="results/paper_macros/tab_selection.tex")
    ap.add_argument("--merge-summary", default="outputs/merge_summary.json")
    ap.add_argument("--usbgym", default="results/v2/usbgym.json")
    ap.add_argument("--usbgym-t1g", default="results/v2/usbgym_t1g.json")
    ap.add_argument("--gate-t1g", default="results/v2/gate_t1g.json")
    ap.add_argument("--t1g-ref", default="results/trusted/tier1_lab_s0.json",
                    help="fold records of the tier-1 model, seed 0 (runs/m7/lab_s0.json of train_m7.py)")
    ap.add_argument("--noharm-tier1", default="results/v2/gate_tier1.json")
    ap.add_argument("--noharm-tier1-fan", default="results/v2/gate_tier1_fan.json")
    ap.add_argument("--out", default="results/paper_macros/results_macros.tex")
    a = ap.parse_args(argv)

    T = _load_tables_module()
    rows = T.load_rows(a.rows_dir)
    out = ["% Generated by scripts/v2/paper_macros_v2.py. Do not edit by hand.",
           r"\providecommand{\na}{\textcolor{gray}{--}}",
           r"\providecommand{\excl}{\textcolor{gray}{excl.}}"]

    def emit(name, val):
        out.append(rf"\newcommand{{\{name}}}{{{val}}}")

    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
    from sonospadepp.v2.hygiene import excluded_from_rows
    models = dict(MODELS)
    if os.path.exists("results/v2/final_model.json"):         # the full model (VTwo) is what WP-final composed
        plan = json.load(open("results/v2/final_model.json"))["models"]
        models = {k: v for k, v in models.items() if v != "VTwo"}
        models = {**models, plan["full"]["key"]: models.get(plan["full"]["key"], "VTwo")}
        pairs = [(k, v) for k, v in models.items()] + ([(plan["full"]["key"], "VTwo")]
                                                         if models[plan["full"]["key"]] != "VTwo" else [])
    else:
        pairs = list(models.items())
    for metric, tissue, ext, proto, stem, d in METRICS:
        for key, mstem in pairs:
            if ext and ext in excluded_from_rows(rows, key):     # metric hygiene: never shown for comparison
                emit(f"R{stem}{mstem}", r"\excl")
                emit(f"R{stem}{mstem}Sd", r"\excl")
                emit(f"R{stem}{mstem}N", r"\na")
                continue
            sel = [r for r in T.select(rows, metric, tissue, ext, proto) if r["model"] == key]
            per_seed = defaultdict(list)
            for r in sel:
                per_seed[r["seed"]].append(float(r["value"]))
            mu, sd, n = T.mean_sd([sum(v) / len(v) for v in per_seed.values()]) if sel else (math.nan, math.nan, 0)
            emit(f"R{stem}{mstem}", fmt(mu, d))
            emit(f"R{stem}{mstem}Sd", fmt(sd, d if d else 0))
            emit(f"R{stem}{mstem}N", str(n) if n else r"\na")

    # latency: mean over every checkpoint of a model family, per device
    lat = defaultdict(list)
    if os.path.exists(a.latency):
        for r in csv.DictReader(open(a.latency)):
            fam = "VOne" if r["stage"].startswith("v1") else ("VTwo" if r["stage"].startswith("v2") else None)
            if fam:
                lat[(fam, r["device"])].append((float(r["mean_ms"]), float(r["fps"])))
    for fam in ("VOne", "VTwo"):
        for dev, dstem in (("cpu", "CPU"), ("mps", "MPS")):
            v = lat.get((fam, dev), [])
            ms = sum(x[0] for x in v) / len(v) if v else math.nan
            fps = sum(x[1] for x in v) / len(v) if v else math.nan
            emit(f"RLat{fam}{dstem}", fmt(ms, 1))
            emit(f"RFps{fam}{dstem}", fmt(fps, 0))
    same = {}                                                  # the ratio: v1 and v2 timed in one process
    if os.path.exists(a.latency):
        for r in csv.DictReader(open(a.latency)):
            if r["stage"] in ("v1_final", "v2_final") and r["device"] == "cpu":
                same[r["stage"]] = float(r["mean_ms"])
    ratio = same["v2_final"] / same["v1_final"] if len(same) == 2 else math.nan
    emit("RLatRatioCPU", fmt(ratio, 2))
    # end-to-end frame (bench_endtoend.py): label slice, physics render, sector warp and translator through USB Gym's
    # frame path; \RLatEte{Phys,Full,Tr}{CPU,MPS} (ms) and \RFpsEte{Phys,Full}{CPU,MPS}
    ete = {}
    if os.path.exists(a.latency_ete):
        for r in csv.DictReader(open(a.latency_ete)):
            ete[(r["stage"], r["device"])] = r
    for dev, dstem in (("cpu", "CPU"), ("mps", "MPS")):
        ph, fu = ete.get(("physics", dev)), ete.get(("sonospade", dev))
        emit(f"RLatEtePhys{dstem}", fmt(float(ph["mean_ms"]), 1) if ph else r"\na")
        emit(f"RFpsEtePhys{dstem}", fmt(float(ph["fps"]), 0) if ph else r"\na")
        emit(f"RLatEteFull{dstem}", fmt(float(fu["mean_ms"]), 1) if fu else r"\na")
        emit(f"RFpsEteFull{dstem}", fmt(float(fu["fps"]), 0) if fu else r"\na")
        emit(f"RLatEteTr{dstem}", fmt(float(fu["translator_ms"]), 1) if fu else r"\na")
    ort = ete.get(("sonospade_ort", "cpu"))                 # the same weights through ONNX Runtime, CPU only
    emit("RLatEteOrtCPU", fmt(float(ort["mean_ms"]), 1) if ort else r"\na")
    emit("RFpsEteOrtCPU", fmt(float(ort["fps"]), 0) if ort else r"\na")
    emit("RLatEteTrOrtCPU", fmt(float(ort["translator_ms"]), 1) if ort else r"\na")
    emit("REteOrtThreads", ort["threads"] if ort else r"\na")
    emit("REteOrtMaxDiff", fmt_p(float(ort["max_diff_vs_pytorch"])) if ort else r"\na")

    # segmenter Dice (v1's released S_US)
    sus = {}
    if os.path.exists(a.sus):
        for r in csv.DictReader(open(a.sus)):
            sus[(r["name"], r["metric"], r["tissue"])] = float(r["value"])
    # real_dice is pooled over the 60 frames (the workshop paper's definition); per-frame is secondary
    emit("RSusVOneSim", fmt(sus.get(("scratch_v1", "heldout_sim_dice", "macro"), math.nan), 3))
    emit("RSusVOneReal", fmt(sus.get(("scratch_v1", "real_dice", "macro"), math.nan), 3))
    emit("RSusVOneRealLiver", fmt(sus.get(("scratch_v1", "real_dice", "liver"), math.nan), 3))
    for name, st in (("usfm", "Usfm"), ("dinov2", "Dino"), ("echocare", "Echo")):        # WP3 initializations
        emit(f"RSus{st}Sim", fmt(sus.get((name, "heldout_sim_dice", "macro"), math.nan), 3))
        emit(f"RSus{st}Real", fmt(sus.get((name, "real_dice", "macro"), math.nan), 3))
    # the leak through the segmenter's encoder (found 2026-11-17): the excluded USFM tissue KID of the
    # USFM-S_US model, reported ONLY as the evidence of that leak, never as a result
    leak = [r for r in T.select(rows, "tissue_kid", "mean", "usfm", "crop16") if r["model"] == "sususfm"]
    ps = defaultdict(list)
    for r in leak:
        ps[r["seed"]].append(float(r["value"]))
    lm = T.mean_sd([sum(v) / len(v) for v in ps.values()])[0] if ps else math.nan
    emit("RLeakTKidUSFMSusUsfm", fmt(lm, 0))
    leak = [r for r in T.select(rows, "tissue_kid", "mean", "dinov2", "crop16") if r["model"] == "susdinov2"]
    emit("RLeakTKidDINOSusDino", fmt(sum(float(r["value"]) for r in leak) / len(leak), 0) if leak else r"\na")

    # same-patient TRUSTED scores (scripts/summarize_m9.py); rows absent from the summary print as \na
    spp = json.load(open(a.spp_summary)) if a.spp_summary and os.path.exists(a.spp_summary) else {}
    emit("RTrPlanes", str(spp.get("n_target_planes", r"\na")))
    for setting, sstem in (("fixed", "Fix"), ("aligned", "Al")):
        block = spp.get(setting, {})
        for key, rstem in SPP_ROWS.items():
            r = block.get(key, {})
            emit(f"RTr{sstem}Kid{rstem}", fmt(r.get("kidney", math.nan), 3))
            emit(f"RTr{sstem}Band{rstem}", fmt(r.get("band", math.nan), 3))
            for i, b in enumerate(("A", "B", "C")):
                bins = r.get("bins") or [math.nan] * 3
                emit(f"RTr{sstem}Bin{b}{rstem}", fmt(bins[i], 3))
        for key, pstem in SPP_PAIRS.items():
            p = spp.get("pairs", {}).get(f"{setting}:{key}")
            k = p["kidney"] if p else [math.nan, math.nan, None, None]
            emit(f"RTr{sstem}D{pstem}", fmt(k[0], 3))
            emit(f"RTr{sstem}P{pstem}", fmt_p(k[1]))
            emit(f"RTr{sstem}Up{pstem}", f"{k[2]}/{k[3]}" if k[2] is not None else r"\na")

    # TRUSTED data and registration facts, re-exported from the generated macro files of the same-patient study
    import re
    spp_tex = {}
    for f in ("results_macros.tex", "results_macros_hr.tex"):
        p = os.path.join(a.spp_macros_dir, f) if a.spp_macros_dir else ""
        if p and os.path.exists(p):
            for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(p).read()):
                spp_tex[m.group(1)] = m.group(2)
    for src, dst in (("PrFrames", "RTrNative"), ("PrKidneys", "RTrKidneys"), ("PrPatients", "RTrPatients"),
                     ("PrSigma", "RTrSigma"), ("RegDiceMean", "RTrRegDiceRig"), ("HRegDiceDef", "RTrRegDiceDef"),
                     ("HRegLdkRig", "RTrRegLdkRig"), ("HRegLdkDef", "RTrRegLdkDef")):
        emit(dst, spp_tex.get(src, r"\na"))

    write_selection(a.selection, a.sel_table, emit)
    write_selection_extra(emit)
    write_merge(a.merge_summary, emit)
    write_usbgym(a.usbgym, emit)
    write_usbgym_t1g(a.usbgym_t1g, a.gate_t1g, a.t1g_ref, emit)
    write_noharm(a.noharm_tier1, emit, "TOne")
    write_noharm(a.noharm_tier1_fan, emit, "TOneFan")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"wrote {a.out}: {len(out) - 2} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
