"""trusted.py: TRUSTED (paired kidney 3D US + CT) loading, probe geometry and matched planes.

The 3D US volumes are scan-converted sweeps of a mechanically driven convex probe (Acuson S3000,
7CF1), 0.3 mm isotropic. Measured on the data (fan edges are straight lines in the data mask):
  * voxel axis 1 is depth; the probe face is at the low-index side;
  * voxel axis 2 is the mechanical sweep (elevation): in an axis1 x axis2 plane the data is a fan
    whose apex is the motor axis, about 18 mm behind the face, half-angle about 37 deg;
  * voxel axis 0 is the array (lateral) direction: in the central sweep plane the data is a fan
    whose apex is the array's centre of curvature, further behind the face.
Screen overlays (text, scale bars, a thumbnail) are burned into some voxels outside the fan; the
fitted fan geometry is used to ignore them.

A "native plane" at sweep angle phi is the central lateral-depth plane rotated by phi about the
motor axis. The virtual probe used for rendering is placed in that plane with its fan apex on the
real array's apex and its axis on the real central beam, so synthetic and real beams coincide.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

ROOT = "data/trusted/raw/TRUSTED_dataset_for_nsd"


# ---------------------------------------------------------------- index ------------------------

def index(root: str = ROOT):
    """List the annotated US kidneys: dicts with pid, side, kid and all file paths."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "US_DATA/US_images/*_imgUS.nii.gz"))):
        kid = os.path.basename(p).split("_")[0]                 # e.g. 206R
        pid, side = kid[:-1], kid[-1]
        out.append(dict(
            kid=kid, pid=pid, side=side, us_img=p,
            us_mask=os.path.join(root, f"US_DATA/US_masks/GT_estimated_masksUS/{kid}_maskUS.nii.gz"),
            us_ldk=os.path.join(root, f"US_DATA/US_landmarks/GT_estimated_ldksUS/{kid}_ldkUS.txt"),
            ct_img=os.path.join(root, f"CT_DATA/CT_images/{pid}_imgCT.nii.gz"),
            ct_mask=os.path.join(root, f"CT_DATA/CT_masks/GT_estimated_masksCT/{pid}_maskCT.nii.gz"),
            ct_ldk=os.path.join(root, f"CT_DATA/CT_landmarks_inimg/GT_estimated_ldksCT/{kid}_ldkCT.txt")))
    return out


def ct_kidney_side(ct_mask: np.ndarray, ct_aff: np.ndarray, ldk_ct: np.ndarray) -> np.ndarray:
    """The CT mask holds both kidneys: keep the connected component nearest the side's landmarks."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(ct_mask > 0)
    if n <= 1:
        return lab > 0
    vox = (np.linalg.inv(ct_aff) @ np.c_[ldk_ct, np.ones(len(ldk_ct))].T)[:3].T
    cents = ndi.center_of_mass(lab > 0, lab, range(1, n + 1))
    sizes = ndi.sum(lab > 0, lab, range(1, n + 1))
    best, bd = 1, np.inf
    for i, (c, s) in enumerate(zip(cents, sizes), start=1):
        if s < 1000:
            continue
        d = np.linalg.norm(vox.mean(0) - np.asarray(c))
        if d < bd:
            best, bd = i, d
    return lab == best


# ---------------------------------------------------------------- fan geometry -----------------

def _longest_runs(P: np.ndarray, j_lo: int, j_hi: int):
    """P (lat, depth) bool; per depth column in [j_lo, j_hi), the longest True run (j, lo, hi)."""
    out = []
    for j in range(max(0, j_lo), min(P.shape[1], j_hi)):
        col = P[:, j].astype(np.int8)
        if col.sum() < 20:
            continue
        d = np.diff(np.r_[0, col, 0])
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        k = int(np.argmax(en - st))
        out.append((j, st[k], en[k]))
    return np.asarray(out, float)


def _fit_fan(P: np.ndarray, j_lo: int, j_hi: int):
    """Fit straight fan edges lat = a + b*depth on both sides; return apex (lat, depth), half-angle
    (deg) and the mean absolute edge residual (px)."""
    E = _longest_runs(P, j_lo, j_hi)
    j, lo, hi = E.T
    pl, pr = np.polyfit(j, lo, 1), np.polyfit(j, hi, 1)
    jd = (pr[1] - pl[1]) / (pl[0] - pr[0])
    res = float(np.mean(np.abs(np.polyval(pl, j) - lo)) + np.mean(np.abs(np.polyval(pr, j) - hi))) / 2
    half = float(np.degrees(np.arctan(abs(pl[0])) + np.arctan(abs(pr[0]))) / 2)
    return float(np.polyval(pl, jd)), float(jd), half, res


@dataclass
class FanGeom:
    """In voxel units of the US volume (axis0 lateral, axis1 depth, axis2 sweep)."""
    motor_d: float      # depth (axis1) of the motor axis
    motor_e: float      # sweep coordinate (axis2) of the motor axis
    sweep_half: float   # deg
    lat_c: float        # lateral (axis0) coordinate of the array apex
    lat_apex_d: float   # depth (axis1) of the array apex in the central plane
    lat_half: float     # deg
    face_d: float       # depth (axis1) of the face at the centre
    far_d: float        # deepest data (axis1)
    res_sweep: float
    res_lat: float


def fit_geometry(us: np.ndarray) -> FanGeom:
    from scipy import ndimage as ndi
    D = ndi.binary_opening(us > 0, iterations=1)
    # the fan is the largest connected blob (overlays are small and separate)
    s = 4
    Ds = D[::s, ::s, ::s]
    lab, n = ndi.label(Ds)
    main = lab == (1 + int(np.argmax(ndi.sum(Ds, lab, range(1, n + 1)))))
    ii, jj, kk = np.nonzero(main)
    e0 = int(np.median(kk) * s)
    d_lo, d_hi = int(jj.min() * s), int(jj.max() * s)
    span = d_hi - d_lo

    def lateral(e):
        P = D[:, :, int(round(e))]
        lat_c, lat_d, lat_half, res = _fit_fan(P, int(d_lo + 0.25 * span), int(d_lo + 0.6 * span))
        col = np.nonzero(P[int(round(lat_c))])[0]
        face = float(col[col > lat_d].min()) if (col > lat_d).any() else float(d_lo)
        return lat_c, lat_d, lat_half, res, face

    lat_c, lat_d, lat_half, res_l, face_d = lateral(e0)
    # sweep fan in axis1 x axis2 planes near the lateral centre, between 30% and 80% of the depth
    a, b = int(face_d + 0.3 * (d_hi - face_d)), int(face_d + 0.8 * (d_hi - face_d))
    fits = [_fit_fan(D[int(round(c)), :, :].T, a, b) for c in (lat_c - 40, lat_c, lat_c + 40)]
    motor_e = float(np.median([f[0] for f in fits])); motor_d = float(np.median([f[1] for f in fits]))
    sweep_half = float(np.median([f[2] for f in fits])); res_s = float(np.median([f[3] for f in fits]))
    lat_c, lat_d, lat_half, res_l, face_d = lateral(motor_e)
    return FanGeom(motor_d, motor_e, sweep_half, lat_c, lat_d, lat_half, face_d, float(d_hi), res_s, res_l)


# ---------------------------------------------------------------- planes ----------------------

def plane_pose_vox(g: FanGeom, phi_deg: float, apex_offset_m: float = 0.04, sp_mm: float = 0.3):
    """Virtual-probe pose for the native plane at sweep angle phi, as (origin, x_lat, y_elev, z_depth)
    in US VOXEL coordinates (unit vectors in voxel space; voxels are isotropic). The probe frame's
    fan apex (at z = -apex_offset_m in the probe frame) sits on the real array apex."""
    ph = np.radians(phi_deg)
    M = np.array([0.0, g.motor_d, g.motor_e])                      # a point on the motor axis
    apex0 = np.array([g.lat_c, g.lat_apex_d, g.motor_e])
    # rotation about axis0 (lateral) by phi: depth (axis1) tilts toward the sweep axis (axis2)
    Rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
    apex = M + Rx @ (apex0 - M)
    z = Rx @ np.array([0.0, 1.0, 0.0])
    y = Rx @ np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)                                             # right-handed: x = y cross z
    origin = apex + z * (apex_offset_m * 1000.0 / sp_mm)
    return origin, x, y, z


def T_world_from_probe_m(us_aff_mm: np.ndarray, origin_vox, x, y, z) -> np.ndarray:
    """Probe -> US physical frame (metres), with world = the US NIfTI's RAS frame."""
    A = np.asarray(us_aff_mm, np.float64)
    o = (A @ np.r_[origin_vox, 1.0])[:3] / 1000.0
    cols = []
    for v in (x, y, z):
        w = A[:3, :3] @ v
        cols.append(w / np.linalg.norm(w))
    R = np.stack(cols, 1)
    if np.linalg.det(R) < 0:                       # the NIfTI affine flips handedness: mirror x
        R[:, 0] *= -1
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = o
    return T


def sample_world(vol: np.ndarray, aff_mm: np.ndarray, pts_world_m: np.ndarray, order: int, cval=0.0):
    from scipy.ndimage import map_coordinates
    inv = np.linalg.inv(np.asarray(aff_mm, np.float64))
    p = np.asarray(pts_world_m, np.float64).reshape(-1, 3) * 1000.0
    vox = p @ inv[:3, :3].T + inv[:3, 3]
    return map_coordinates(vol, vox.T, order=order, mode="constant", cval=cval)


def fan_valid(g: FanGeom, pts_vox: np.ndarray, margin_deg: float = 1.0) -> np.ndarray:
    """Inside the real 3D fan (sweep angle and lateral angle within range, between face and far)."""
    p = np.asarray(pts_vox, np.float64).reshape(-1, 3)
    de, dd = p[:, 2] - g.motor_e, p[:, 1] - g.motor_d
    phi = np.degrees(np.arctan2(de, dd))
    ok_s = np.abs(phi) <= g.sweep_half - margin_deg
    # undo the sweep, then check the lateral angle about the array apex
    ph = np.radians(phi)
    r = np.sqrt(de ** 2 + dd ** 2)
    d_unswept = g.motor_d + r                                     # depth along the rotated axis
    lat = p[:, 0] - g.lat_c
    ang = np.degrees(np.arctan2(lat, d_unswept - g.lat_apex_d))
    ok_l = np.abs(ang) <= g.lat_half - margin_deg
    rad = np.sqrt(lat ** 2 + (d_unswept - g.lat_apex_d) ** 2)
    ok_r = (rad >= (g.face_d - g.lat_apex_d) + 3) & (d_unswept <= g.far_d - 5)
    del ph
    return ok_s & ok_l & ok_r


__all__ = ["ROOT", "index", "ct_kidney_side", "FanGeom", "fit_geometry", "plane_pose_vox",
           "T_world_from_probe_m", "sample_world", "fan_valid"]
