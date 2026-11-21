"""SonoSPADE v2: more realistic per-tissue ultrasound texture at v1's control and speed.

v1 (the published SonoSPADE, package `usbg`, pinned at v0.2.0) is never modified: v2 extends it. Large
models act only at training time (critics, pseudo-labelers, feature extractors), so inference stays the
v1 generator plus small additions.

Modules:
  metrics   per-tissue realism, speckle and control metrics
  features  frozen foundation-model feature extractors (DINOv2, USFM, EchoCare) for metrics and D_fm
  splits    the fixed data splits (real pool train / validation by sequence, evaluation frames)
"""
