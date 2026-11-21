"""ctvol.py: a CT label volume that keeps the Hounsfield units (HU) instead of discarding them.

SonoSPADE's volume builder (usbg.volume.load_labeled_ct) reads HU once, to fill air, fat and soft
tissue and to find bone, then stores labels only. This builder keeps three things per case:

  labels_base  the 13 canonical labels, built the same way EXCEPT the bone rule (below)
  labels_pp    the M1 labels: labels_base plus structure found from HU (gas, vessels, cysts)
  hu           the CT smoothed with a 1-voxel Gaussian (int16), for M2 and M3
  ref, sd      per-label tissue median HU and robust spread, so M2/M3 use HU RELATIVE to the
               organ it sits in (contrast phase shifts absolute HU by 100+ between scans)

Bone rule. The release overlays BONE on any non-background voxel at or above 300 HU, after the
organ masks. On arterial-phase CT that catches contrast-filled vessels and renal cortex, so an aorta
renders as a bright interface with a shadow. Here voxels inside vessel, kidney or spleen masks are
exempt; explicit bone masks still win. This applies to EVERY variant (it is a label fix, not a
texture method), so the comparison does not credit M1 with it.

M1 sub-labels, all from HU inside existing TotalSegmentator masks:
  GAS     GI-mask voxels below -200 HU (smoothed), with air acoustics
  VESSEL  liver or spleen voxels, away from a `rim`-voxel shell at the organ edge (partial volume),
          whose HU differs from the organ median by more than `vessel_z` robust standard
          deviations (intrahepatic vessels in either contrast phase, and focal lesions), cleaned
          of 3D components under `min_vessel_vox` voxels
  VESSEL  TotalSegmentator kidney-cyst masks (anechoic fluid)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from usbg import volume as V

from .labels import GAS, HU_MODULATED, N_PP

HU_TISSUE = (-200.0, 300.0)          # tissue window for the per-label reference statistics
BONE_EXEMPT = (V.VESSEL, V.KIDNEY, V.SPLEEN)
SD_FLOOR = 10.0                      # HU; keeps z finite for very uniform organs


@dataclass
class PPVolume:
    labels_base: np.ndarray          # (I,J,K) uint8, 13 classes
    labels_pp: np.ndarray            # (I,J,K) uint8, 14 classes (M1)
    hu: np.ndarray                   # (I,J,K) int16, smoothed HU
    ref: np.ndarray                  # (N_PP,) float32 per-label tissue median HU (nan if absent)
    sd: np.ndarray                   # (N_PP,) float32 per-label robust SD (HU)
    affine_m: np.ndarray
    spacing_m: np.ndarray
    origin_m: np.ndarray
    meta: dict

    def labeled(self, which: str = "base") -> V.LabeledVolume:
        """A usbg LabeledVolume view (for usbg.slicer and usbg.placement)."""
        lab = self.labels_base if which == "base" else self.labels_pp
        return V.LabeledVolume(labels=lab, affine_m=self.affine_m, spacing_m=self.spacing_m,
                               origin_m=self.origin_m, meta=self.meta)


def _load_mask(seg_dir, stem, shape):
    import nibabel as nib
    p = os.path.join(seg_dir, f"{stem}.nii.gz")
    if not os.path.exists(p):
        return None
    m = np.asanyarray(nib.as_closest_canonical(nib.load(p)).dataobj) > 0.5
    return m if m.shape == shape else None


def _robust_stats(values: np.ndarray):
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(1.4826 * mad, SD_FLOOR)


def _remove_small(mask: np.ndarray, min_vox: int) -> np.ndarray:
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_vox
    keep[0] = False
    return keep[lab]


def build_pp_volume(ct_path: str, seg_dir: str, *, smooth_sigma: float = 1.0,
                    vessel_z: float = 2.5, rim: int = 2, min_vessel_vox: int = 20) -> PPVolume:
    import nibabel as nib
    from scipy import ndimage

    ct = nib.as_closest_canonical(nib.load(ct_path))
    hu_raw = np.asanyarray(ct.dataobj).astype(np.float32)
    shape = hu_raw.shape

    # base labels: HU fill, organ overlay (same order as the release), then bone with exemptions
    labels = np.full(shape, V.SOFT_TISSUE, dtype=np.uint8)
    labels[hu_raw < V.HU_AIR] = V.BACKGROUND
    labels[(hu_raw >= V.HU_AIR) & (hu_raw < V.HU_FAT)] = V.FAT
    by_label: dict[int, np.ndarray] = {}
    present = []
    for stem, lab in V.TOTALSEG_MAP.items():
        m = _load_mask(seg_dir, stem, shape)
        if m is None or not m.any():
            continue
        by_label.setdefault(lab, np.zeros(shape, bool))
        by_label[lab] |= m
        present.append(stem)
    for lab in V._ORGAN_OVERLAY:
        if lab in by_label:
            labels[by_label[lab]] = lab
    bone_masks = np.zeros(shape, bool)
    for fn in sorted(os.listdir(seg_dir)):
        stem = fn[:-7] if fn.endswith(".nii.gz") else os.path.splitext(fn)[0]
        if stem.startswith(V.BONE_PREFIXES):
            m = _load_mask(seg_dir, stem, shape)
            if m is not None:
                bone_masks |= m
    hu_bone = (hu_raw >= V.HU_BONE) & (labels != V.BACKGROUND)
    exempt = np.isin(labels, BONE_EXEMPT)
    old_bone = bone_masks | hu_bone                       # the release rule
    new_bone = bone_masks | (hu_bone & ~exempt)
    labels[new_bone] = V.BONE
    labels_base = labels

    # smoothed HU and per-label reference statistics (tissue window only, so gas does not drag the
    # GI median and contrast does not matter: everything downstream is relative to the organ)
    hu_s = ndimage.gaussian_filter(hu_raw, smooth_sigma) if smooth_sigma > 0 else hu_raw
    ref = np.full(N_PP, np.nan, np.float32)
    sd = np.full(N_PP, np.nan, np.float32)
    tissue = (hu_s >= HU_TISSUE[0]) & (hu_s < HU_TISSUE[1])
    for L in HU_MODULATED:
        sel = (labels_base == L) & tissue
        if sel.sum() >= 100:
            ref[L], sd[L] = _robust_stats(hu_s[sel])

    # M1 sub-labels
    labels_pp = labels_base.copy()
    gi = labels_base == V.GI
    gas = gi & (hu_s < V.HU_AIR)
    labels_pp[gas] = GAS
    vessel = np.zeros(shape, bool)
    for organ in (V.LIVER, V.SPLEEN):
        if np.isnan(ref[organ]):
            continue
        # skip a `rim`-voxel shell at the organ edge: there the smoothed HU mixes with the fat or
        # air outside (partial volume) and would read as a false low-HU "vessel" lining the organ
        org = ndimage.binary_erosion(labels_base == organ, iterations=rim) if rim else labels_base == organ
        z = np.abs(hu_s - ref[organ]) / sd[organ]
        vessel |= org & tissue & (z > vessel_z)
    vessel = _remove_small(vessel, min_vessel_vox)
    labels_pp[vessel] = V.VESSEL
    cyst = np.zeros(shape, bool)
    for stem in ("kidney_cyst_left", "kidney_cyst_right"):
        m = _load_mask(seg_dir, stem, shape)
        if m is not None:
            cyst |= m
    cyst &= np.isin(labels_base, (V.KIDNEY, V.SOFT_TISSUE, V.FAT))
    labels_pp[cyst] = V.VESSEL

    def frac(a, b):
        nb = int(b.sum())
        return float((a & b).sum() / nb) if nb else 0.0

    liver = labels_base == V.LIVER
    meta = {
        "source": "totalsegmentator", "ct_path": os.path.abspath(ct_path),
        "structures_present": sorted(present), "shape": list(shape),
        "smooth_sigma": smooth_sigma, "vessel_z": vessel_z, "rim": rim, "min_vessel_vox": min_vessel_vox,
        "bone_rule_exempted_vox": int((old_bone & ~new_bone).sum()),
        "gas_frac_of_gi": frac(gas, gi),
        "vessel_frac_of_liver": frac(vessel, liver),
        "cyst_vox": int(cyst.sum()),
        "ref_hu": {V.USBG_LABEL_NAMES[L]: (None if np.isnan(ref[L]) else round(float(ref[L]), 1))
                   for L in HU_MODULATED},
    }
    affine_m = V._affine_to_meters(ct.affine)
    return PPVolume(labels_base=labels_base, labels_pp=labels_pp,
                    hu=np.clip(np.round(hu_s), -1024, 3071).astype(np.int16), ref=ref, sd=sd,
                    affine_m=affine_m, spacing_m=np.abs(np.diag(affine_m)[:3]).astype(np.float64),
                    origin_m=affine_m[:3, 3].astype(np.float64), meta=meta)


def save_pp(vol: PPVolume, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, labels_base=vol.labels_base, labels_pp=vol.labels_pp, hu=vol.hu,
                        ref=vol.ref, sd=vol.sd, affine_m=vol.affine_m, spacing_m=vol.spacing_m,
                        origin_m=vol.origin_m, meta=json.dumps(vol.meta))
    return path


def load_pp(path: str) -> PPVolume:
    d = np.load(path, allow_pickle=False)
    return PPVolume(labels_base=d["labels_base"], labels_pp=d["labels_pp"], hu=d["hu"],
                    ref=d["ref"], sd=d["sd"], affine_m=d["affine_m"], spacing_m=d["spacing_m"],
                    origin_m=d["origin_m"], meta=json.loads(str(d["meta"])))


__all__ = ["PPVolume", "build_pp_volume", "save_pp", "load_pp", "BONE_EXEMPT"]
