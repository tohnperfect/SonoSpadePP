"""hires.py: the same-patient test at full resolution.

The 128x128 pipeline resizes every render and real frame to 128 px (about 1.25 mm per pixel
laterally) and warps it to the sector geometry the translators were trained on. Here nothing is
resized: the real ultrasound is sampled from the 3D volume on a Cartesian grid at its own voxel size
(0.3 mm) in the native plane, and the physics render is computed on a scan grid with twice the
lines (320 x 512; lateral point spread scaled so the beam width in mm is unchanged) and scan-
converted to the same grid. Only renders are compared: the translators exist at 128 px only.

Geometry (probe frame, metres): convex array, apex at z = -0.04, scan lines from -30 to +30 deg,
radius 0.04 .. 0.16 (usbg's scan grid with more lines).
"""
from __future__ import annotations

import numpy as np

APEX_Z, R0, HALF_DEG = -0.04, 0.04, 30.0
MM = 0.3                                  # Cartesian pixel (mm): the TRUSTED US voxel size


def depth_m():
    from usbg import render_lotus as R
    return R.IMG_DEPTH_M


def polar_grid(L=320, S=512):
    """Scan points (L, S, 3) in the probe frame (m), as usbg.slicer.scan_grid_probe with L lines."""
    th = np.radians(np.linspace(-HALF_DEG, HALF_DEG, L))[:, None]
    r = R0 + np.linspace(0.0, depth_m(), S)[None, :]
    return np.stack([r * np.sin(th), np.zeros_like(r * th), APEX_Z + r * np.cos(th)], -1).astype(np.float64)


def cart_grid(mm=MM):
    """Cartesian in-plane grid: points (NZ, NX, 3) in the probe frame (m), fan mask, radius from the
    apex (m), and the fractional (sample, line) index of each pixel in a polar render."""
    s = mm / 1000.0
    x = np.arange(-(R0 + depth_m()) * np.sin(np.radians(HALF_DEG)), (R0 + depth_m()) * np.sin(np.radians(HALF_DEG)) + 1e-9, s)
    z = np.arange(APEX_Z + R0 * np.cos(np.radians(HALF_DEG)), APEX_Z + R0 + depth_m() + 1e-9, s)
    X, Z = np.meshgrid(x, z)
    pts = np.stack([X, np.zeros_like(X), Z], -1)
    rad = np.hypot(X, Z - APEX_Z)
    th = np.degrees(np.arctan2(X, Z - APEX_Z))
    fan = (rad >= R0) & (rad <= R0 + depth_m()) & (np.abs(th) <= HALF_DEG)
    return pts, fan, rad.astype(np.float32), th.astype(np.float32)


def polar_index(rad, th, L, S):
    """Fractional (sample, line) of each Cartesian pixel in an (S, L) render."""
    s_idx = (rad - R0) / depth_m() * (S - 1)
    l_idx = (th + HALF_DEG) / (2 * HALF_DEG) * (L - 1)
    return s_idx.astype(np.float32), l_idx.astype(np.float32)


def scan_convert(img_SL, s_idx, l_idx, fan):
    """(S, L) render -> Cartesian image (bilinear), 0 outside the fan."""
    from scipy.ndimage import map_coordinates
    out = map_coordinates(np.asarray(img_SL, np.float32), [s_idx.ravel(), l_idx.ravel()], order=1, mode="nearest")
    return np.where(fan, out.reshape(fan.shape), 0.0).astype(np.float32)


class ScanConvertT:
    """Differentiable scan conversion (torch grid_sample), (B, S, L) -> (B, NZ, NX)."""

    def __init__(self, s_idx, l_idx, fan, S, L, device="cpu"):
        import torch
        g = np.stack([l_idx / (L - 1) * 2 - 1, s_idx / (S - 1) * 2 - 1], -1).astype(np.float32)
        self.grid = torch.as_tensor(g, device=device)[None]
        self.fan = torch.as_tensor(fan.astype(np.float32), device=device)

    def __call__(self, img_BSL):
        import torch.nn.functional as F
        x = img_BSL[:, None]
        out = F.grid_sample(x, self.grid.expand(x.shape[0], -1, -1, -1), mode="bilinear", padding_mode="border",
                            align_corners=True)
        return out[:, 0] * self.fan


# ------------------------------------------------------------------------------- CT sampling ----
def sample_ct(vol, q_ct_mm):
    """CT labels (nearest; 0 outside), M1 labels and smoothed HU (trilinear; -1000 outside) at
    CT-world points (mm, (N,3))."""
    from scipy.ndimage import map_coordinates
    inv = np.linalg.inv(np.asarray(vol.affine_m, np.float64))
    p = np.asarray(q_ct_mm, np.float64).reshape(-1, 3) / 1000.0
    vox = (p @ inv[:3, :3].T + inv[:3, 3]).T
    lab_b = map_coordinates(vol.labels_base, vox, order=0, mode="constant", cval=0).astype(np.uint8)
    lab_p = map_coordinates(vol.labels_pp, vox, order=0, mode="constant", cval=0).astype(np.uint8)
    hu = map_coordinates(vol.hu.astype(np.float32), vox, order=1, mode="constant", cval=-1000.0)
    return lab_b, lab_p, hu.astype(np.float32)


def relative_hu(hu, lab, vol, spacing_mm, edge_mm=(1.0, 3.0)):
    """slicing.relative_hu with an explicit pixel spacing (mm) for the boundary ramp."""
    from scipy.ndimage import distance_transform_edt
    lab = np.asarray(lab).astype(np.int64)
    ref = np.nan_to_num(vol.ref, nan=0.0)[lab]
    sd = np.nan_to_num(vol.sd, nan=1.0)[lab]
    has = ~np.isnan(vol.ref)[lab]
    edge = np.zeros(lab.shape, bool)
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]; edge[:-1, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]; edge[:, :-1] |= lab[:, 1:] != lab[:, :-1]
    ramp = np.ones(lab.shape, np.float32) if not edge.any() else np.clip(
        (distance_transform_edt(~edge, sampling=spacing_mm) - edge_mm[0]) / (edge_mm[1] - edge_mm[0]), 0, 1)
    return (np.where(has, (hu - ref) / sd, 0.0) * ramp).astype(np.float32)


def polar_spacing_mm(L, S):
    """(lateral at mid depth, axial) spacing of the (L, S) scan grid in mm."""
    r_mid = R0 + depth_m() / 2
    return float(r_mid * np.radians(2 * HALF_DEG) / (L - 1) * 1000), float(depth_m() / (S - 1) * 1000)


# ---------------------------------------------------------------------------------- metric ------
def regions(kidney, valid, mm=MM, erode_mm=1.25, band_mm=5.0):
    """Kidney interior (eroded by erode_mm) and kidney + band (band_mm), both within valid, as the
    128-px metric (1 px erosion, 4 px band at about 1.25 mm per px)."""
    from scipy.ndimage import distance_transform_edt
    k = np.asarray(kidney, bool) & valid
    if not k.any():
        return k, k
    inside = distance_transform_edt(k, sampling=mm)
    outside = distance_transform_edt(~k, sampling=mm)
    return (inside > erode_mm) & valid, (outside <= band_mm) & valid


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def _detrended(img, rad, m, deg=2):
    rr = rad[m].astype(np.float64)
    c = np.polyfit((rr - rr.mean()) / (rr.std() + 1e-9), img[m].astype(np.float64), deg)
    return img[m] - np.polyval(c, (rr - rr.mean()) / (rr.std() + 1e-9))


def scores(fake, real, kidney, valid, rad, sigma_mm=1.0, mm=MM, min_px=200):
    """Same-plane agreement at full resolution: Pearson r after Gaussian smoothing (sigma_mm), over
    the kidney interior and kidney + band after removing a quadratic in depth, and over the whole
    valid fan without detrending."""
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(np.asarray(fake, np.float32), sigma_mm / mm)
    r = gaussian_filter(np.asarray(real, np.float32), sigma_mm / mm)
    inner, ring = regions(kidney, valid, mm)
    out = {"r_fan": _pearson(f[valid], r[valid]) if valid.sum() >= min_px else float("nan")}
    for name, m in (("kidney", inner), ("band", ring)):
        if m.sum() < min_px:
            out[f"r_{name}"] = out[f"r_{name}_dt"] = float("nan")
            continue
        out[f"r_{name}"] = _pearson(f[m], r[m])
        out[f"r_{name}_dt"] = _pearson(_detrended(f, rad, m), _detrended(r, rad, m))
    return out


__all__ = ["polar_grid", "cart_grid", "polar_index", "scan_convert", "ScanConvertT", "sample_ct", "relative_hu",
           "polar_spacing_mm", "regions", "scores", "MM"]
