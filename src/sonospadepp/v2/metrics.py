"""Per-tissue realism, speckle and control metrics for SonoSPADE v2.

Every per-tissue metric runs inside a tissue mask: the real masks come from the annotated evaluation
frames, the fake masks from the generator's input label. Frames are float [0, 1], labels are canonical
ids (usbg.volume). The v1 metrics (organ intensity W1, edge IoU, label-swap locality, TSTR Dice) are NOT
re-implemented here: callers use usbg.texture_eval directly, so the v1 definitions stay exact.

New here:
  * envelope            approximate amplitude from a log-compressed frame, A = 10 ** ((x - 1) * DR / 20)
  * nakagami_windows    window-level Nakagami m = E[A^2]^2 / Var(A^2) and Omega = E[A^2] inside an eroded mask
  * speckle_acov_sum / fwhm_from_acov   depth-binned axial and lateral speckle size (FWHM of the normalized
                        autocovariance of A)
  * w1_1d               1D Wasserstein-1 between two samples (exact, by quantiles of the pooled sizes)
  * tissue_crops        fixed-size crops centred inside an eroded tissue mask (for per-tissue KID / FD)
  * kid, fd             KID (unbiased polynomial-kernel MMD^2, v1's estimator) and Frechet distance
  * locality_band       label-swap / style-swap locality with an excluded boundary band of N px
  * bootstrap_ci        percentile bootstrap over frames
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

DR_DEFAULT = 60.0          # dB, dynamic range assumed when undoing log compression


# --------------------------------------------------------------------------------------------------
# envelope and masks
# --------------------------------------------------------------------------------------------------
def envelope(x01, dr_db: float = DR_DEFAULT):
    """Approximate amplitude from a log-compressed B-mode in [0, 1]: x = 1 + 20 log10(A) / DR, so
    A = 10 ** ((x - 1) * DR / 20), A in (0, 1]. Works on numpy arrays (and torch tensors via `**`)."""
    return 10.0 ** ((x01 - 1.0) * (dr_db / 20.0))


def eroded(mask, iters: int):
    m = np.asarray(mask, bool)
    return ndimage.binary_erosion(m, iterations=iters) if iters > 0 else m


# --------------------------------------------------------------------------------------------------
# Nakagami window statistics
# --------------------------------------------------------------------------------------------------
def nakagami_windows(img01, mask, win: int = 9, stride: int = 4, dr_db: float = DR_DEFAULT,
                     erode: int = 2, min_cover: float = 1.0):
    """Window-level Nakagami parameters inside an eroded tissue mask.

    Windows of win x win px are centred on a stride grid; a window counts when at least `min_cover` of
    its pixels lie in the eroded mask, and its moments use only those pixels. Per window, with I = A^2:
    Omega = E[I], m = E[I]^2 / Var(I) (the moment estimator; m = 1 for Rayleigh speckle).
    Returns (m, log_omega) as 1D float64 arrays (empty when no window qualifies)."""
    if win % 2 == 0:
        raise ValueError("win must be odd")
    A = envelope(np.asarray(img01, np.float64), dr_db)
    inten = A * A
    m = eroded(mask, erode).astype(np.float64)
    if m.sum() == 0:
        return np.empty(0), np.empty(0)
    area = float(win * win)
    s_m = ndimage.uniform_filter(m, win, mode="constant") * area
    s_i = ndimage.uniform_filter(inten * m, win, mode="constant") * area
    s_i2 = ndimage.uniform_filter(inten * inten * m, win, mode="constant") * area
    h = win // 2
    H, W = m.shape
    rows = np.arange(h, H - h, stride)
    cols = np.arange(h, W - h, stride)
    if rows.size == 0 or cols.size == 0:
        return np.empty(0), np.empty(0)
    R, C = np.meshgrid(rows, cols, indexing="ij")
    n = s_m[R, C]
    ok = n >= min_cover * area - 1e-6
    if not ok.any():
        return np.empty(0), np.empty(0)
    n = n[ok]
    mu = s_i[R, C][ok] / n
    ex2 = s_i2[R, C][ok] / n
    var = ex2 - mu * mu
    good = (var > 1e-12 * np.maximum(mu * mu, 1e-30)) & (mu > 0)
    mu, var = mu[good], var[good]
    return mu * mu / var, np.log(mu)


# --------------------------------------------------------------------------------------------------
# speckle size (autocovariance FWHM, depth-binned)
# --------------------------------------------------------------------------------------------------
def _patch_acov(p):
    """Unbiased linear autocovariance of a patch at non-negative lags, (h, w): the zero-padded
    autocorrelation of the mean-removed patch divided by the number of overlapping pixel pairs per lag."""
    a = p - p.mean()
    h, w = a.shape
    F = np.fft.rfft2(a, s=(2 * h, 2 * w))
    ac = np.fft.irfft2(np.abs(F) ** 2, s=(2 * h, 2 * w))[:h, :w]
    pairs = (h - np.arange(h))[:, None] * (w - np.arange(w))[None, :]
    return ac / pairs


def speckle_acov_sum(img01, mask, patch: int = 12, stride: int = 6, depth_bins: int = 3,
                     dr_db: float = DR_DEFAULT, erode: int = 2):
    """Summed autocovariance of A over patches fully inside the eroded mask, per depth bin.

    Depth is the row axis (the axial direction of the linear render; for the curvilinear real frames this
    is an approximation, noted in the paper). Returns (acov_sum (depth_bins, patch, patch), counts)."""
    A = envelope(np.asarray(img01, np.float64), dr_db)
    m = eroded(mask, erode)
    H, W = m.shape
    acc = np.zeros((depth_bins, patch, patch))
    cnt = np.zeros(depth_bins, np.int64)
    if not m.any():
        return acc, cnt
    edges = np.linspace(0, H, depth_bins + 1)
    for r0 in range(0, H - patch + 1, stride):
        for c0 in range(0, W - patch + 1, stride):
            if not m[r0:r0 + patch, c0:c0 + patch].all():
                continue
            b = min(int(np.searchsorted(edges, r0 + patch / 2.0, side="right")) - 1, depth_bins - 1)
            acc[b] += _patch_acov(A[r0:r0 + patch, c0:c0 + patch])
            cnt[b] += 1
    return acc, cnt


def fwhm_from_acov(acov):
    """Axial (row-lag) and lateral (column-lag) full width at half maximum, in px, of a summed
    autocovariance (normalized by its zero lag here). NaN when the profile never drops to 0.5 within the
    patch (speckle coarser than the patch) or the acov is empty."""
    if acov[0, 0] <= 0:
        return float("nan"), float("nan")
    a = acov / acov[0, 0]

    def width(prof):
        below = np.nonzero(prof <= 0.5)[0]
        if below.size == 0:
            return float("nan")
        k = int(below[0])
        k0 = (k - 1) + (prof[k - 1] - 0.5) / (prof[k - 1] - prof[k])
        return 2.0 * float(k0)

    return width(a[:, 0]), width(a[0, :])


# --------------------------------------------------------------------------------------------------
# distribution distances
# --------------------------------------------------------------------------------------------------
def w1_1d(a, b, n_q: int = 256) -> float:
    """1D Wasserstein-1 between two samples, by matching n_q quantiles (exact for equal-size sorted
    samples as n_q grows). NaN if either sample is empty."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    q = (np.arange(n_q) + 0.5) / n_q
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def kid(fr, fg, subset: int = 100, n_subsets: int = 100, seed: int = 0):
    """Unbiased polynomial-kernel MMD^2 (KID), averaged over random subsets; v1's estimator
    (scripts/v1/eval_fid_kid.py), so v1 and v2 numbers are on one scale. Returns (mean, std)."""
    fr = np.asarray(fr, np.float64)
    fg = np.asarray(fg, np.float64)
    rng = np.random.default_rng(seed)
    d = fr.shape[1]
    n = min(subset, fr.shape[0], fg.shape[0])
    if n < 2:
        return float("nan"), float("nan")

    def poly(a, b):
        return (a @ b.T / d + 1.0) ** 3

    vals = []
    for _ in range(n_subsets):
        r = fr[rng.choice(fr.shape[0], n, replace=False)]
        g = fg[rng.choice(fg.shape[0], n, replace=False)]
        krr, kgg, krg = poly(r, r), poly(g, g), poly(r, g)
        np.fill_diagonal(krr, 0.0)
        np.fill_diagonal(kgg, 0.0)
        vals.append(krr.sum() / (n * (n - 1)) + kgg.sum() / (n * (n - 1)) - 2.0 * krg.mean())
    return float(np.mean(vals)), float(np.std(vals))


def fd(fr, fg, eps: float = 1e-6) -> float:
    """Frechet distance between Gaussian fits of two feature sets (v1's FID formula, any features)."""
    from scipy import linalg
    fr = np.asarray(fr, np.float64)
    fg = np.asarray(fg, np.float64)
    if fr.shape[0] < 2 or fg.shape[0] < 2:
        return float("nan")
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cr, cg = np.cov(fr, rowvar=False), np.cov(fg, rowvar=False)
    off = np.eye(cr.shape[0]) * eps
    covmean = linalg.sqrtm((cr + off) @ (cg + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    return float(diff @ diff + np.trace(cr + cg - 2.0 * covmean))


# --------------------------------------------------------------------------------------------------
# tissue crops
# --------------------------------------------------------------------------------------------------
def tissue_crops(img01, mask, size: int = 32, max_crops: int = 8, erode: int = 2,
                 min_cover: float = 0.5, rng=None):
    """Up to max_crops size x size crops whose centre lies in the eroded mask and whose window holds at
    least min_cover tissue pixels. Context outside the tissue is kept (an ultrasound crop is not
    meaningful with a blanked background). Returns (n, size, size) float32."""
    rng = rng if rng is not None else np.random.default_rng(0)
    img = np.asarray(img01, np.float32)
    m = np.asarray(mask, bool)
    H, W = m.shape
    h = size // 2
    cover = ndimage.uniform_filter(m.astype(np.float64), size, mode="constant")
    em = eroded(m, erode)
    cand = np.argwhere(em)
    if cand.size == 0:
        return np.empty((0, size, size), np.float32)
    r, c = cand[:, 0], cand[:, 1]
    inside = (r - h >= 0) & (r - h + size <= H) & (c - h >= 0) & (c - h + size <= W)
    cand = cand[inside]
    if cand.size == 0:
        return np.empty((0, size, size), np.float32)
    cand = cand[cover[cand[:, 0], cand[:, 1]] >= min_cover]
    if cand.size == 0:
        return np.empty((0, size, size), np.float32)
    pick = cand[rng.choice(len(cand), min(max_crops, len(cand)), replace=False)]
    return np.stack([img[r0 - h:r0 - h + size, c0 - h:c0 - h + size] for r0, c0 in pick]).astype(np.float32)


# --------------------------------------------------------------------------------------------------
# locality with a boundary band
# --------------------------------------------------------------------------------------------------
def locality_band(base, alt, region, band: int = 8) -> dict:
    """Mean |alt - base| inside `region` over the mean outside the region dilated by `band` px.
    band = 0 is v1's label-swap locality (texture_eval.label_swap_sensitivity) exactly."""
    diff = np.abs(np.asarray(alt, np.float32) - np.asarray(base, np.float32))
    m = np.asarray(region, bool)
    far = ~ndimage.binary_dilation(m, iterations=band) if band > 0 else ~m
    if not m.any() or not far.any():
        return {"in_region": float("nan"), "out_region": float("nan"), "locality": float("nan")}
    i, o = float(diff[m].mean()), float(diff[far].mean())
    return {"in_region": i, "out_region": o, "locality": i / (o + 1e-6)}


def label_swap_locality(translate_aligned, phys_img01, aligned_label, n_classes: int = 13,
                        ref_class: int = 1, min_px: int = 200, band: int = 8) -> dict:
    """v1's label-swap protocol (relabel each region to the reference tissue, liver) scored with
    locality_band. With band = 0 it equals usbg.texture_eval.label_swap_sensitivity."""
    img = np.asarray(phys_img01, np.float32)
    lab = np.asarray(aligned_label)
    base = np.asarray(translate_aligned(img, lab), np.float32)
    out = {}
    for c in range(n_classes):
        m = lab == c
        if c == ref_class or int(m.sum()) < min_px or not (~m).any():
            continue
        sw = lab.copy()
        sw[m] = ref_class
        out[c] = locality_band(base, translate_aligned(img, sw), m, band=band)
    return out


def style_swap_locality(translate_styled, phys_img01, aligned_label, style_a: dict, style_b: dict,
                        tissue: int, band: int = 8, min_px: int = 200) -> dict:
    """Style-swap locality (WP2): change only tissue k's style code and compare the change inside k with
    the change outside k's dilated band. translate_styled(img, lab, style) -> img01."""
    lab = np.asarray(aligned_label)
    m = lab == tissue
    if int(m.sum()) < min_px or not (~m).any():
        return {"in_region": float("nan"), "out_region": float("nan"), "locality": float("nan")}
    img = np.asarray(phys_img01, np.float32)
    base = translate_styled(img, lab, style_a)
    alt = translate_styled(img, lab, style_b)
    return locality_band(base, alt, m, band=band)


# --------------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------------
def bootstrap_ci(stat_fn, n_items: int, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap: stat_fn(index_array) -> float, resampling n_items with replacement.
    Returns (lo, hi); NaN when the statistic is undefined on most resamples."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = stat_fn(rng.integers(0, n_items, n_items))
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
