"""warp.py: linear render -> curvilinear sector, matching the real Kaggle frames.

Same geometry as SonoSpade's scripts/curvi_warp.py (measured on the Kaggle ussimandsegm test frames
at 128x128), extended to warp any number of aligned channels with ONE set of coordinates: images and
the relative-HU channel bilinear, label maps nearest. Outside the sector: image 0, label 0, z 0.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

DEFAULT = dict(apex_x=68.0, apex_y=-28.0, half_angle_deg=23.0, r_min=36.0, r_max=128.0)


def sector_coords(H, W, p=DEFAULT, rot_deg=0.0, tx=0.0, ty=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ax, ay = p["apex_x"] + tx * W, p["apex_y"] + ty * H
    dx, dy = xx - ax, yy - ay
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.degrees(np.arctan2(dx, dy)) - rot_deg
    half = p["half_angle_deg"]
    mask = (np.abs(theta) <= half) & (r >= p["r_min"]) & (r <= p["r_max"])
    src_col = (theta + half) / (2.0 * half) * (W - 1)
    src_row = (r - p["r_min"]) / (p["r_max"] - p["r_min"]) * (H - 1)
    return np.stack([src_row.ravel(), src_col.ravel()]), mask


def warp(arr, coords, mask, order):
    H, W = mask.shape
    out = map_coordinates(np.asarray(arr, np.float32), coords, order=order, mode="constant", cval=0.0)
    return out.reshape(H, W) * mask


def warp_frame(channels: dict, kinds: dict, rot_deg=0.0, tx=0.0, ty=0.0, p=DEFAULT) -> dict:
    """channels: name -> (H,W) array; kinds: name -> 'image' | 'label' | 'float'. Returns warped
    arrays with the input dtypes (images are uint8 0..255 in and out)."""
    any_arr = next(iter(channels.values()))
    H, W = any_arr.shape
    coords, mask = sector_coords(H, W, p, rot_deg, tx, ty)
    out = {}
    for k, a in channels.items():
        kind = kinds[k]
        if kind == "label":
            out[k] = warp(a, coords, mask, 0).astype(a.dtype)
        elif kind == "image":
            out[k] = np.clip(np.round(warp(a, coords, mask, 1)), 0, 255).astype(np.uint8)
        else:
            out[k] = warp(a, coords, mask, 1).astype(np.float32)
    return out


__all__ = ["DEFAULT", "sector_coords", "warp_frame"]
