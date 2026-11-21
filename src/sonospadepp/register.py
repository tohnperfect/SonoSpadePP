"""register.py: rigid US -> CT kidney registration from landmarks, refined on the mask surfaces.

TRUSTED gives, per kidney, 7 landmarks and a STAPLE mask in both the 3D US and the CT. The paper
initializes registration from landmarks and refines it; its best method reached a 4.47 mm target
error. Here the goal is only a good alignment to cut matching planes, so the GT masks may be used:

  1. Kabsch fit of the 7 landmark pairs (rigid, least squares);
  2. trimmed rigid ICP from US mask-surface points to CT mask-surface points.

Kidneys move and deform little with breathing, so rigid is the paper's own transform model.
All coordinates are physical millimetres in each image's own world frame.
"""
from __future__ import annotations

import numpy as np


def kabsch(src: np.ndarray, dst: np.ndarray):
    """Rigid (R, t) minimizing sum ||R src_i + t - dst_i||^2 (no scaling, no reflection)."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cd - R @ cs


def to_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R; T[:3, 3] = t
    return T


def apply_T(T, pts):
    pts = np.asarray(pts, np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def surface_points(mask: np.ndarray, affine_mm: np.ndarray, max_pts: int = 20000, seed: int = 0):
    """Physical (mm) coordinates of the mask's boundary voxels, subsampled to max_pts."""
    from scipy.ndimage import binary_erosion
    m = np.asarray(mask) > 0
    surf = m & ~binary_erosion(m)
    ijk = np.argwhere(surf).astype(np.float64)
    if len(ijk) > max_pts:
        ijk = ijk[np.random.default_rng(seed).choice(len(ijk), max_pts, replace=False)]
    return apply_T(np.asarray(affine_mm, np.float64), ijk)


def icp_rigid(src_pts, dst_pts, T0, iters: int = 60, trim: float = 0.9, tol: float = 1e-4):
    """Trimmed point-to-point rigid ICP from src to dst, starting at T0. Returns (T, rms_mm)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst_pts)
    T = np.asarray(T0, np.float64).copy()
    prev = np.inf
    rms = np.nan
    for _ in range(iters):
        moved = apply_T(T, src_pts)
        d, idx = tree.query(moved)
        keep = d <= np.quantile(d, trim)
        R, t = kabsch(src_pts[keep], dst_pts[idx[keep]])
        T = to_T(R, t)
        rms = float(np.sqrt(np.mean(d[keep] ** 2)))
        if abs(prev - rms) < tol:
            break
        prev = rms
    return T, rms


def mask_dice(us_mask, us_aff_mm, ct_mask, ct_aff_mm, T_ct_from_us, step: int = 3) -> float:
    """Dice between the US mask and the CT mask resampled into US voxels (every `step` voxel)."""
    from scipy.ndimage import map_coordinates
    um = np.asarray(us_mask)[::step, ::step, ::step] > 0
    ijk = np.argwhere(np.ones(um.shape, bool)).astype(np.float64) * step
    phys_us = apply_T(np.asarray(us_aff_mm, np.float64), ijk)
    phys_ct = apply_T(T_ct_from_us, phys_us)
    vox_ct = apply_T(np.linalg.inv(np.asarray(ct_aff_mm, np.float64)), phys_ct)
    cm = map_coordinates((np.asarray(ct_mask) > 0).astype(np.float32), vox_ct.T, order=0,
                         mode="constant", cval=0.0).reshape(um.shape) > 0.5
    inter = float((um & cm).sum())
    return 2.0 * inter / max(1.0, float(um.sum() + cm.sum()))


__all__ = ["kabsch", "to_T", "apply_T", "surface_points", "icp_rigid", "mask_dice"]
