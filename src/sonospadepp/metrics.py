"""metrics.py: the comparison metrics. All are computed on held-out CT content (cases never used in
training) against the real Kaggle ussimandsegm frames, with every variant scored on the SAME 13-class
base labels.

  organ_w1        per-organ intensity Wasserstein-1 to the 60 labeled real frames (lower = closer),
                  mean over the organs the real labels contain (liver, gallbladder, vessel, kidney,
                  spleen); from usbg.texture_eval
  hetero_gap      low-frequency heterogeneity inside an organ: the SD of the 2-px Gaussian-smoothed
                  intensity in the (eroded) organ mask, pooled over frames, as |fake - real|. Speckle
                  averages out at that scale; vessels, gas and lesions do not. This is the metric aimed
                  at the structure CT adds
  tstr_dice       a segmenter trained on the translated frames + base labels, tested on the 60 real
                  frames (mean over seeds); from usbg.texture_eval
  edge_iou        structure kept from the variant's own physics input
  locality        label-swap in-region / out-region change (per-tissue control); usbg.texture_eval
  hu_sensitivity  mean |T(x, l, z) - T(x, l, 0)| inside the body: how much the output depends on the
                  HU channel (M3 only; 0 by construction for the others)
"""
from __future__ import annotations

import numpy as np
from usbg import texture_eval as TE

REAL_ORGANS = (1, 2, 3, 4, 7)          # liver, gallbladder, vessel, kidney, spleen in the real labels
HETERO_ORGANS = (1, 4, 7)              # organs large enough for a low-frequency SD


def organ_w1(trans, labels, real_imgs, real_labs):
    w1 = TE.intensity_w1_per_class(trans, labels, real_imgs, real_labs)
    per = {int(k): float(v) for k, v in w1.items() if int(k) in REAL_ORGANS}
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def _hetero(imgs, labs, c, sigma=2.0, erode=2, min_px=150):
    from scipy import ndimage
    vals = []
    for im, lb in zip(imgs, labs):
        m = np.asarray(lb) == c
        if erode:
            m = ndimage.binary_erosion(m, iterations=erode)
        if m.sum() < min_px:
            continue
        lp = ndimage.gaussian_filter(np.asarray(im, np.float32), sigma)
        vals.append(float(lp[m].std()))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def hetero_gap(trans, labels, real_imgs, real_labs):
    per = {}
    for c in HETERO_ORGANS:
        f, nf = _hetero(trans, labels, c)
        r, nr = _hetero(real_imgs, real_labs, c)
        if nf and nr:
            per[c] = {"fake": f, "real": r, "gap": abs(f - r), "n_fake": nf, "n_real": nr}
    gaps = [v["gap"] for v in per.values()]
    return (float(np.mean(gaps)) if gaps else float("nan")), per


def tstr(trans, labels, real_imgs, real_labs, seeds=(0, 1, 2), iters=500, device=None):
    runs = []
    for s in seeds:
        d, per = TE.tstr_dice(list(trans), list(labels), list(real_imgs), list(real_labs),
                              device=device, iters=iters, seed=int(s))
        runs.append({"seed": int(s), "dice": float(d), "per_class": {int(k): float(v) for k, v in per.items()}})
    ds = [r["dice"] for r in runs]
    return float(np.mean(ds)), float(np.std(ds)), runs


def edge_iou(phys, trans):
    return float(np.mean([TE.edge_iou(p, t) for p, t in zip(phys, trans)]))


def locality(translate_aligned, phys, labels, n_classes, n=40):
    sens = TE.summarize_per_class([
        {c: v["locality"] for c, v in TE.label_swap_sensitivity(
            translate_aligned, im, lb, n_classes=n_classes).items()}
        for im, lb in zip(phys[:n], labels[:n])])
    return (float(np.mean(list(sens.values()))) if sens else float("nan")), sens


def hu_sensitivity(translator, phys, labels, zs, n=60):
    diffs = []
    for im, lb, z in zip(phys[:n], labels[:n], zs[:n]):
        body = np.asarray(lb) != 0
        if body.sum() < 100:
            continue
        a = translator(im, lb, z)
        b = translator(im, lb, np.zeros_like(z))
        diffs.append(float(np.abs(a - b)[body].mean()))
    return float(np.mean(diffs)) if diffs else float("nan")


__all__ = ["organ_w1", "hetero_gap", "tstr", "edge_iou", "locality", "hu_sensitivity", "paired_scores", "pearson_in",
           "REAL_ORGANS", "HETERO_ORGANS"]


# ---- paired (same-patient) metrics for TRUSTED ------------------------------------------------

def smooth(img, sigma=1.5):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(img, np.float32), sigma)


def pearson_in(a, b, region, min_px=50):
    """Pearson r between a and b over the pixels in `region` (nan if fewer than min_px)."""
    m = np.asarray(region, bool)
    if m.sum() < min_px:
        return float("nan")
    x = a[m].astype(np.float64); y = b[m].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def paired_scores(fake, real, kidney, valid, sigma=1.5, band_px=4):
    """Same-plane agreement between one synthetic frame and the real US frame of the same patient.

    fake, real : (H,W) float in [0,1], pixel-aligned.  kidney : real-US kidney mask (bool).
    valid      : pixels where both images have data (inside both fans).
    Returns r over the kidney interior (structure inside the organ: sinus, medulla, cysts), over
    the kidney plus a band around it (organ outline and its neighbours), and over the whole valid
    fan. All on images smoothed with sigma px, so speckle averages out and only structure counts.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    f, r = smooth(fake, sigma), smooth(real, sigma)
    k = np.asarray(kidney, bool) & valid
    inner = binary_erosion(k, iterations=1) & valid
    ring = binary_dilation(k, iterations=band_px) & valid
    rad = _sector_radius(f.shape)
    return {"r_kidney": pearson_in(f, r, inner), "r_band": pearson_in(f, r, ring),
            "r_fan": pearson_in(f, r, valid),
            "r_kidney_dt": pearson_in(_detrend(f, rad, inner), _detrend(r, rad, inner), inner),
            "r_band_dt": pearson_in(_detrend(f, rad, ring), _detrend(r, rad, ring), ring)}


def _sector_radius(shape):
    """Distance from the sector apex (Kaggle geometry, warp.DEFAULT): depth along each beam."""
    from .warp import DEFAULT
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.sqrt((xx - DEFAULT["apex_x"]) ** 2 + (yy - DEFAULT["apex_y"]) ** 2)


def _detrend(img, rad, region, deg=2):
    """Remove a polynomial in depth fitted inside `region`, so gain and attenuation trends along
    the beam do not count as structure. Outside the region the image is returned unchanged."""
    m = np.asarray(region, bool)
    if m.sum() < 3 * (deg + 1):
        return img
    c = np.polyfit(rad[m], img[m], deg)
    out = img.copy()
    out[m] = img[m] - np.polyval(c, rad[m])
    return out
