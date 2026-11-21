"""slicing.py: one probe pose -> aligned base labels, M1 labels and the relative-HU slice.

Labels use usbg.slicer (nearest neighbour, identical to the release). The HU slice samples the
smoothed CT at the same scan points with trilinear interpolation, then becomes a per-pixel z-score
relative to the tissue the pixel belongs to (base label), which is phase-robust:

    z = (HU - ref[label]) / sd[label],   0 where the label has no reference (air, bone, lung)

then faded to 0 within ~1-3 mm of a label boundary, where CT partial volume would otherwise
read as a deviation (edge_ramp). M2's impedance uses the raw HU, so real interfaces still reflect.
"""
from __future__ import annotations

import numpy as np
from usbg import slicer as S
from usbg.contracts_bridge import volume_from_probe

from .ctvol import PPVolume


def slice_hu(vol: PPVolume, T_world_from_probe, T_world_from_volume) -> np.ndarray:
    """Smoothed HU at the scan grid, float32 (N_LINES, N_SAMPLES); outside the CT -> -1000 (air)."""
    from scipy.ndimage import map_coordinates
    pts_probe, _ = S.scan_grid_probe()
    L, Sm = pts_probe.shape[:2]
    T = np.asarray(volume_from_probe(T_world_from_probe, T_world_from_volume), np.float64)
    flat = pts_probe.reshape(-1, 3).astype(np.float64)
    pts_vol = np.einsum("ij,nj->ni", T[:3, :3], flat) + T[:3, 3]
    inv = np.linalg.inv(np.asarray(vol.affine_m, np.float64))
    vox = np.einsum("ij,nj->ni", inv[:3, :3], pts_vol) + inv[:3, 3]
    hu = map_coordinates(vol.hu.astype(np.float32), vox.T, order=1, mode="constant", cval=-1000.0)
    return hu.reshape(L, Sm).astype(np.float32)


EDGE_MM = (1.0, 3.0)        # z fades from 0 (<= 1 mm from a label boundary) to full (>= 3 mm)


def _pixel_mm():
    """(lateral, axial) scan-grid spacing in mm at mid depth."""
    p, depth = S.scan_grid_probe()
    mid = p.shape[1] // 2
    return float(np.linalg.norm(p[1, mid] - p[0, mid]) * 1000), float((depth[1] - depth[0]) * 1000)


def edge_ramp(base_label_slice: np.ndarray) -> np.ndarray:
    """Weight in [0,1] that is 0 at label boundaries and 1 from EDGE_MM[1] mm inside a region.

    At a boundary the smoothed CT mixes two tissues (partial volume), so a z-score against either
    tissue's reference reads as a false deviation and would draw an outline around every organ."""
    from scipy.ndimage import distance_transform_edt
    lab = np.asarray(base_label_slice)
    edge = np.zeros(lab.shape, bool)
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]; edge[:-1, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]; edge[:, :-1] |= lab[:, 1:] != lab[:, :-1]
    if not edge.any():
        return np.ones(lab.shape, np.float32)
    dist = distance_transform_edt(~edge, sampling=_pixel_mm())
    return np.clip((dist - EDGE_MM[0]) / (EDGE_MM[1] - EDGE_MM[0]), 0.0, 1.0).astype(np.float32)


def relative_hu(hu_slice: np.ndarray, base_label_slice: np.ndarray, vol: PPVolume) -> np.ndarray:
    """Per-pixel z-score of HU relative to its own tissue (0 where the tissue has no reference),
    faded to 0 near label boundaries (edge_ramp)."""
    lab = np.asarray(base_label_slice).astype(np.int64)
    ref = np.nan_to_num(vol.ref, nan=0.0)[lab]
    sd = np.nan_to_num(vol.sd, nan=1.0)[lab]
    has = ~np.isnan(vol.ref)[lab]
    z = np.where(has, (hu_slice - ref) / sd, 0.0) * edge_ramp(lab)
    return z.astype(np.float32)


def slice_all(vol: PPVolume, T_world_from_probe, T_world_from_volume):
    """(lab_base, lab_pp, hu, z), all (N_LINES, N_SAMPLES), for one pose."""
    lab_base = S.slice_at_pose(vol.labeled("base"), T_world_from_probe, T_world_from_volume)
    lab_pp = S.slice_at_pose(vol.labeled("pp"), T_world_from_probe, T_world_from_volume)
    hu = slice_hu(vol, T_world_from_probe, T_world_from_volume)
    return lab_base, lab_pp, hu, relative_hu(hu, lab_base, vol)


__all__ = ["slice_hu", "relative_hu", "slice_all"]
