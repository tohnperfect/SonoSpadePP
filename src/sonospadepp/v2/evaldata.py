"""Shared evaluation inputs for v2: content pairs, the 60 annotated real frames, the real-pool fan mask,
and translator loading for every texture stage (physics, v1 baselines, v1 SonoSPADE, v2)."""
from __future__ import annotations

import glob
import os

import numpy as np
from scipy import ndimage

REAL_ORGANS = (1, 2, 3, 4, 7)          # liver, gallbladder, vessel, kidney, spleen: annotated in the real set


def load_pairs(path, limit=None):
    from usbg.dataset import load_pairs_npz
    imgs, labs, sess, _meta = load_pairs_npz(path)
    imgs = imgs.astype(np.float32) / 255.0
    labs = labs.astype(np.int64)
    if limit:
        imgs, labs, sess = imgs[:limit], labs[:limit], sess[:limit]
    return imgs, labs, sess


def load_pool(pool_dir, indices=None):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(pool_dir, "*.png")))
    if indices is not None:
        files = [files[i] for i in indices]
    return np.stack([np.asarray(Image.open(f).convert("L"), np.float32) / 255.0 for f in files])


def fan_mask(pool_imgs, frac: float = 0.5, thr: float = 0.02, erode: int = 2) -> np.ndarray:
    """The ultrasound sector of the real pool: pixels nonzero in at least `frac` of the pool frames,
    holes filled (dark anechoic content is not geometry), eroded by `erode` px. Built from the TRAINING
    pool only, never from the evaluation frames."""
    nz = (np.asarray(pool_imgs) > thr).mean(0) >= frac
    nz = ndimage.binary_fill_holes(nz)
    return ndimage.binary_erosion(nz, iterations=erode) if erode else nz


class Stage:
    """A texture stage with one call: translate(phys01 HxW, label HxW) -> HxW float [0, 1]."""

    def __init__(self, kind: str, ckpt: str | None = None, device=None):
        self.kind, self.ckpt = kind, ckpt
        self.label_aware = False
        if kind == "physics":
            self._f = lambda img, lab: np.asarray(img, np.float32)
        elif kind == "sonospade":
            from usbg import texture_gan as TG
            tr = TG.SonoSPADETranslator(ckpt, device=device)
            self._f, self.label_aware = tr.translate_aligned, True
        elif kind in ("cut", "cyclegan", "cyclegan_sc"):
            from usbg import texture_gan as TG
            tr = TG.TextureTranslator(ckpt, device=device)
            self._f = lambda img, lab: tr.translate(img)
        elif kind == "sonospade_v2":
            from sonospadepp.v2.translator import SonoSPADEv2Translator
            tr = SonoSPADEv2Translator(ckpt, device=device)
            self.translator = tr
            self._f, self.label_aware = tr.translate_aligned, True
        elif kind == "paired_lab":                    # a tier-1 (train_m7 --mode lab) fold model: "B0.pt::M7.pt"
            import torch
            from sonospadepp import paired_translator as PT
            b0, m7 = ckpt.split("::")
            dev = device or "cpu"
            G = PT.build(b0, dev)
            G.load_state_dict(torch.load(m7, map_location=dev, weights_only=True))
            G.eval()

            def f(img, lab):
                z = np.zeros((1,) + np.shape(img), np.float32)
                c, s = PT.inputs(np.asarray(img, np.float32)[None], np.asarray(lab)[None], z, z, False, dev)
                with torch.no_grad():
                    return PT.forward01(G, c, s)[0].cpu().numpy()
            self._f, self.label_aware = f, True
        else:
            raise ValueError(kind)

    def __call__(self, img, lab):
        return np.asarray(self._f(img, lab), np.float32)

    def fixed_noise(self, seed=0):
        """Context: injected noise (A5) fixed across calls, so paired comparisons (label-swap and style-swap
        locality) measure the label or style effect, not two noise draws. No-op for other stages."""
        import contextlib
        tr = getattr(self, "translator", None)

        @contextlib.contextmanager
        def ctx():
            old = getattr(tr, "noise_seed", None)
            if tr is not None:
                tr.noise_seed = seed
            try:
                yield
            finally:
                if tr is not None:
                    tr.noise_seed = old
        return ctx()
