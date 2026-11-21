"""Label schemes and acoustic tables for SonoSpadePP.

The base scheme is SonoSPADE's 13 canonical labels (usbg.volume). Method M1 adds one label, GAS
(bowel or stomach gas found from CT Hounsfield units inside a GI mask). GAS takes the air acoustics
the release gives LUNG, so a gas pocket renders as a bright interface with a shadow behind it.

Every variant is scored against the 13-class base labels, so a frame's evaluation target never
depends on which method rendered it.
"""
from __future__ import annotations

import numpy as np
from usbg import volume as V
from usbg.segmenter import GROUPED_MAP, N_GROUPED

N_BASE = V.N_LABELS                     # 13 canonical labels, 0..12
GAS = N_BASE                            # 13: M1 sub-label (gas inside the GI tract)
N_PP = N_BASE + 1                       # 14 labels in the M1 scheme

# canonical id -> (Z MRayl, att dB/cm/MHz, mu_0, mu_1, sigma_0); GAS copies LUNG (air)
ACOUSTIC_PP: dict[int, tuple] = dict(V.ACOUSTIC_LOTUS)
ACOUSTIC_PP[GAS] = V.ACOUSTIC_LOTUS[V.LUNG]

# labels whose acoustics M2 modulates by CT intensity (soft tissues and fluids; not air or bone)
HU_MODULATED = (V.LIVER, V.GALLBLADDER, V.VESSEL, V.KIDNEY, V.GI, V.PANCREAS, V.SPLEEN,
                V.MUSCLE, V.FAT, V.SOFT_TISSUE)


def grouped_map(n_labels: int) -> np.ndarray:
    """Canonical id -> grouped S_US class (SonoSPADE's 9-class scheme). In the 14-label M1 scheme,
    GAS joins the air group with LUNG: the two share acoustics, so B-mode cannot separate them."""
    a = np.arange(n_labels, dtype=np.int64)
    for k, v in GROUPED_MAP.items():
        a[k] = v
    if n_labels > GAS:
        a[GAS] = GROUPED_MAP[V.LUNG]
    return a


def to_base(label_pp: np.ndarray) -> np.ndarray:
    """Map an M1 label map back to the base scheme (GAS -> GI)."""
    out = np.asarray(label_pp).copy()
    out[out == GAS] = V.GI
    return out


__all__ = ["N_BASE", "N_PP", "GAS", "ACOUSTIC_PP", "HU_MODULATED", "grouped_map", "to_base",
           "N_GROUPED"]
