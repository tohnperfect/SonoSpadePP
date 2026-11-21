#!/usr/bin/env python3
"""Same-patient examples (figures/trusted.pdf): the real plane beside the rows of Table I, on held-out
TRUSTED planes, so that every variant of this work is seen against the patient's own ultrasound.

Rows are chosen by a fixed rule, not by eye: the kidneys at the lower and the upper quartile of tier 1's detrended
kidney correlation (per-kidney mean over the scored planes and the three seeds, as Table I averages), and in each
the scored plane nearest 10 degrees from the given plane (the middle of the 7 to 13 degree bin). Columns: the real
plane, the physics render, SonoSPADE, tier 0 (SonoSPADE trained with the USFM segmenter), SonoSPADE's unpaired
TRUSTED control, tier 1, pix2pix, tier 2 (one-shot fine-tune) and the given image warped through the label maps;
seed 0 where a model is trained. SonoSPADE and tier 0 are the checkpoints of fig_variants_v2.py, run here as
eval_trusted.py runs them; the other columns are the stored held-out predictions. Each panel prints that plane's
detrended kidney correlation with the real plane, read from the result files behind Table I, and the two generated
columns are checked against them (--tol). The real kidney mask is outlined.

The inputs live in the working copy that ran the same-patient study (--pp-root). TRUSTED's Data Usage Agreement allows small portions of the
data to clarify a publication; the figure file itself stays out of git (.gitignore). CPU only; needs
torch and the sonospadepp package (the .venv of this checkout).

  .venv/bin/python -I scripts/v2/fig_trusted_v2.py [--pp-root /path/to/working-copy] [--out figures/trusted.pdf]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# (title, result-file variant, source): "phys" = the render in the pairs file, *.pt = a checkpoint run on the render
# and the labels as eval_trusted.py does, *.npz = stored held-out predictions
COLS = [("Physics render", "phys_b0", "phys"),
        ("SonoSPADE", "b0", "runs/b0_s0.pt"),
        ("Tier 0", "b0_v2", "runs/b0v2_s0.pt"),
        ("Unpaired control", "b0t", "outputs/b0t/preds_b0t_s0.npz"),
        ("Tier 1", "m7c", "outputs/m7/preds_lab_s0.npz"),
        ("pix2pix", "p2p", "outputs/m7/preds_p2p_s0.npz"),
        ("Tier 2", "m9ac", "outputs/m9/preds_lab1_s0.npz"),
        ("Warped image", "labwarp", "outputs/m9/preds_labwarp.npz")]
SCORES = ["outputs/trusted_def.json", "outputs/trusted_m9.json", "outputs/trusted_merge_v2.json"]
OUTLINE = "#eb6834"                                   # the outline color of Fig. 4


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp-root", default=os.environ.get("SONOSPADEPP_ROOT", "."),
                    help="working copy that holds the same-patient runs, outputs and TRUSTED data (default: this one)")
    ap.add_argument("--angle", type=float, default=10.0, help="target angle from the given plane, degrees")
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.25, 0.75])
    ap.add_argument("--width", type=float, default=7.16, help="figure width in inches (the text width)")
    ap.add_argument("--tol", type=float, default=0.01, help="largest allowed difference between a regenerated "
                    "panel's score and the stored one")
    ap.add_argument("--out", default="figures/trusted.pdf")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = a.pp_root
    d = np.load(os.path.join(root, "data/trusted_pairs_def.npz"), allow_pickle=False)
    planes = json.loads(str(d["meta"]))["planes"]
    kid = np.array([p["kid"] for p in planes])
    phi = np.array([p["phi_deg"] for p in planes], float)
    real = d["real"].astype(np.float32) / 255.0
    kmask = (d["kidney_us"] > 0) & (d["valid"] > 0)

    # the given plane of each kidney is the middle one by sweep angle (train_m9.native_refs); the others are scored
    given, dphi = {}, np.full(len(planes), np.nan)
    for k in dict.fromkeys(kid):
        ids = sorted(np.flatnonzero(kid == k), key=lambda i: phi[i])
        given[k] = ids[len(ids) // 2]
        for i in ids:
            if i != given[k]:
                dphi[i] = abs(phi[i] - phi[given[k]])
    scored = np.isfinite(dphi)

    rows, sigma = {}, None                            # result-file row name -> per-plane detrended kidney r
    for p in SCORES:
        p = os.path.join(root, p)
        if os.path.exists(p):
            res = json.load(open(p))
            assert list(res["kidney_id"]) == list(kid), f"{p}: different planes"
            assert sigma in (None, res["sigma"]), f"{p}: different smoothing"
            sigma = res["sigma"]
            for name, row in res["rows"].items():
                if "r_kidney_dt" in row:
                    rows[name] = (row.get("variant", name.rsplit("_s", 1)[0]), np.asarray(row["r_kidney_dt"], float))

    def group(g):
        return {n: v for n, (var, v) in rows.items() if var == g}

    def seed0(g):
        gr = group(g)
        if len(gr) == 1:
            return next(iter(gr.values()))
        return next(v for n, v in gr.items() if n.endswith("_s0"))

    # per-kidney mean of tier 1 over the scored planes, then over seeds, as Table I averages
    kids = list(dict.fromkeys(kid))
    t1 = np.array([np.mean([np.nanmean(np.where(scored, v, np.nan)[kid == k]) for v in group("m7c").values()])
                   for k in kids])
    order = np.argsort(t1)
    picks = []
    for q in a.quantiles:
        k = kids[order[int(round(q * (len(kids) - 1)))]]
        ids = np.flatnonzero((kid == k) & scored)
        i = int(ids[np.argmin(np.abs(dphi[ids] - a.angle))])
        picks.append((k, i, float(t1[kids.index(k)])))

    pick_ids = [i for _k, i, _t in picks]
    gens = []                                         # per column: ({plane: image}, per-plane stored score)
    for _title, g, path in COLS:
        sc = seed0(g)
        if path == "phys":
            imgs = {i: d["phys_b0"][i].astype(np.float32) / 255.0 for i in pick_ids}
        elif path.endswith(".pt"):
            from sonospadepp import metrics as M
            from sonospadepp.variants import Translator
            tr = Translator(os.path.join(root, path), device="cpu")
            out = tr.batch(d["phys_b0"][pick_ids].astype(np.float32) / 255.0, d["lab_base"][pick_ids].astype(np.int64))
            imgs = {i: np.asarray(o, np.float32) for i, o in zip(pick_ids, out)}
            for i in pick_ids:                        # the regenerated frame must be the one the table scored
                got = M.paired_scores(imgs[i], real[i], d["kidney_us"][i].astype(bool), d["valid"][i].astype(bool),
                                      sigma=sigma)["r_kidney_dt"]
                assert abs(got - sc[i]) <= a.tol, f"{path} plane {i}: r {got:.4f}, stored {sc[i]:.4f}"
        else:
            pr = np.load(os.path.join(root, path), allow_pickle=False)
            assert all(pr["done"][i] for i in pick_ids), f"{path}: a picked plane was not predicted"
            imgs = {i: pr["gen"][i].astype(np.float32) for i in pick_ids}
        gens.append((imgs, sc))

    valid = np.any([d["valid"][i] > 0 for _k, i, _t in picks], axis=0)
    ys, xs = np.flatnonzero(valid.any(1)), np.flatnonzero(valid.any(0))
    y0, y1, x0, x1 = max(ys[0] - 1, 0), min(ys[-1] + 2, 128), max(xs[0] - 1, 0), min(xs[-1] + 2, 128)
    h, w = y1 - y0, x1 - x0

    fs = 6.5
    plt.rcParams.update({"font.size": fs, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    ncol, nrow = len(COLS) + 1, len(picks)
    fw = a.width
    pw = fw / ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(fw, nrow * (pw * h / w + 0.15) + 0.20), squeeze=False)
    titles = ["Real plane"] + [c[0] for c in COLS]
    summary = []
    for r, (k, i, t) in enumerate(picks):
        imgs = [(real[i], None)] + [(g[i], float(s[i])) for g, s in gens]
        for c, (img, rv) in enumerate(imgs):
            ax = axes[r][c]
            ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.contour(kmask[i][y0:y1, x0:x1].astype(float), levels=[0.5], colors=[OUTLINE], linewidths=0.55)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=fs, pad=2.5, linespacing=0.95)
            ax.set_xlabel("" if rv is None else f"r = {rv:.2f}".replace("-", "\u2212"), fontsize=fs, labelpad=1.5)
        summary.append({"kidney": k, "plane": i, "angle_from_given_deg": round(float(dphi[i]), 2),
                        "tier1_kidney_mean_r": round(t, 3),
                        "r_kidney_dt_seed0": {COLS[c][1]: round(float(gens[c][1][i]), 3) for c in range(len(COLS))}})
    fig.subplots_adjust(left=0.003, right=0.997, top=0.93, bottom=0.06, wspace=0.04, hspace=0.12)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(json.dumps({"n_kidneys": len(kids), "n_scored_planes": int(scored.sum()), "picks": summary}, indent=1))
    print("written:", a.out)


if __name__ == "__main__":
    main()
