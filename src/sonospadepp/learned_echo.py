"""learned_echo.py: M4, a learned HU-to-echogenicity front end trained on same-patient planes.

M1-M3 either fix how CT intensity changes the image by hand (M1 rules, M2 acoustics) or leave it to
an unpaired generator that has no reason to use it (M3). M4 learns it from TRUSTED, where the real
ultrasound of the same patient and plane is known:

    BS(p) = BS_label(p) * exp( a_l + g_l(z(p)) ),   l = label(p), only for the soft-tissue classes

a_l is a per-tissue echogenicity offset and g_l a piecewise-linear curve over the relative HU z
(knots on [-4, 4]). Impedance and attenuation stay those of the labels, so with a = g = 0 the render
is exactly the base render. The render is differentiable in BS (physics.render_from_parts), and so
are the resize, the sector warp and the frozen B0 generator, so (a, g) are fitted by gradient
descent on the same-plane agreement used for evaluation, with patients split into TRUSTED's own
five folds. The HU-free control ('const') learns a_l only.
"""
from __future__ import annotations

import numpy as np

from .labels import HU_MODULATED, N_PP
from .warp import sector_coords

# TRUSTED README: 5-fold cross-validation splits by kidney (both kidneys of a patient share a fold)
US_FOLDS = [
    ['263R', '263L', '794R', '794L', '592R', '206R', '206L', '579R', '579L', '915L', '250R', '250L'],
    ['561R', '418R', '418L', '636R', '258R', '258L', '283L', '610L', '517R', '517L'],
    ['443R', '506R', '506L', '641R', '641L', '398R', '711L', '532R', '532L', '371R', '721L', '735R', '680L'],
    ['755R', '755L', '510R', '510L', '701R', '701L', '948R', '948L', '329R', '329L', '704L', '399R'],
    ['861R', '861L', '200R', '220R', '716R', '348R', '348L', '466R', '466L', '738R', '314R', '314L'],
]


def fold_of(kid: str) -> int:
    for i, f in enumerate(US_FOLDS):
        if kid in f:
            return i
    raise KeyError(kid)


def _torch():
    import torch
    return torch


class LearnedEcho:
    """Parameters a (N_PP,), g (N_PP, K). Use .params() with any torch optimizer."""

    def __init__(self, hu_dependent=True, knots=9, z_max=4.0, clamp=2.0, device="cpu"):
        torch = _torch()
        self.K, self.z_max, self.clamp, self.hu = knots, z_max, clamp, hu_dependent
        self.a = torch.zeros(N_PP, device=device, requires_grad=True)
        self.g = torch.zeros(N_PP, knots, device=device, requires_grad=hu_dependent)
        mod = np.zeros(N_PP, bool); mod[list(HU_MODULATED)] = True
        self.mod = torch.as_tensor(mod, device=device)

    def params(self):
        return [self.a] + ([self.g] if self.hu else [])

    def init_m2_rule(self, bs_k=0.35):
        """Start g at M2's hand-set rule, log factor -bs_k min(|z|, 4), on every modulated class.
        The knots include z = 0, so the piecewise-linear curve represents the rule exactly."""
        torch = _torch()
        knots = torch.linspace(-self.z_max, self.z_max, self.K, device=self.g.device)
        with torch.no_grad():
            self.g.copy_(torch.where(self.mod[:, None], -bs_k * knots.abs().clamp(max=4.0)[None], torch.zeros_like(self.g)))

    def log_factor(self, lab, z):
        """lab (..) long, z (..) float tensors -> log backscatter factor (..)."""
        torch = _torch()
        lf = self.a[lab]
        if self.hu:
            u = (z.clamp(-self.z_max, self.z_max) + self.z_max) / (2 * self.z_max) * (self.K - 1)
            i0 = u.floor().clamp(0, self.K - 2).long()
            w = u - i0
            lf = lf + self.g[lab, i0] * (1 - w) + self.g[lab, i0 + 1] * w
        lf = lf.clamp(-self.clamp, self.clamp)
        return torch.where(self.mod[lab], lf, torch.zeros_like(lf))

    def factor_numpy(self, lab, z):
        torch = _torch()
        with torch.no_grad():
            lf = self.log_factor(torch.as_tensor(np.asarray(lab, np.int64), device=self.a.device),
                                 torch.as_tensor(np.asarray(z, np.float32), device=self.a.device))
        return np.exp(lf.cpu().numpy()).astype(np.float32)

    def regularizer(self, w_smooth=1e-2, w_mag=1e-3):
        m = self.mod
        reg = w_mag * (self.a[m] ** 2).sum()
        if self.hu:
            g = self.g[m]
            reg = reg + w_smooth * ((g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]) ** 2).sum() + w_mag * (g ** 2).sum()
        return reg

    def state(self):
        return {"a": self.a.detach().cpu().numpy().tolist(), "g": self.g.detach().cpu().numpy().tolist(),
                "knots": np.linspace(-self.z_max, self.z_max, self.K).tolist(), "hu_dependent": self.hu}

    @classmethod
    def from_state(cls, st, device="cpu"):
        torch = _torch()
        m = cls(hu_dependent=st["hu_dependent"], knots=len(st["knots"]), z_max=st["knots"][-1], device=device)
        with torch.no_grad():
            m.a.copy_(torch.tensor(st["a"])); m.g.copy_(torch.tensor(st["g"]))
        return m


class SectorPost:
    """Differentiable twin of make_pairs: (S, L) render -> 128x128 resize -> Kaggle sector warp."""

    def __init__(self, size=128, device="cpu"):
        torch = _torch()
        coords, mask = sector_coords(size, size)
        rows, cols = coords[0].reshape(size, size), coords[1].reshape(size, size)
        grid = np.stack([cols / (size - 1) * 2 - 1, rows / (size - 1) * 2 - 1], -1).astype(np.float32)
        self.grid = torch.as_tensor(grid, device=device)[None]
        self.mask = torch.as_tensor(mask.astype(np.float32), device=device)
        self.size = size

    def __call__(self, img_sl):
        torch = _torch()
        import torch.nn.functional as F
        x = img_sl[:, None] if img_sl.dim() == 3 else img_sl[None, None]
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        w = F.grid_sample(x, self.grid.expand(x.shape[0], -1, -1, -1), mode="bilinear",
                          padding_mode="zeros", align_corners=True)
        return w[:, 0] * self.mask


def gauss_kernel(sigma=1.5, truncate=4.0, device="cpu"):
    torch = _torch()
    r = int(truncate * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2); k /= k.sum()
    return torch.as_tensor(k, device=device)


def smooth_t(img, k):
    """Separable Gaussian on (B, H, W), reflect padding (scipy's default up to the edge pixel)."""
    import torch.nn.functional as F
    r = (len(k) - 1) // 2
    x = F.pad(img[:, None], (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[:, 0]


def region_design(region, rad):
    """Indices of a region and the normalized quadratic-in-depth design matrix used to detrend it."""
    idx = np.flatnonzero(region)
    r = rad.ravel()[idx]
    rn = (r - r.mean()) / (r.std() + 1e-6)
    X = np.stack([np.ones_like(rn), rn, rn ** 2], 1).astype(np.float64)
    return idx, X


def detrended_resid_np(img, idx, X):
    y = img.ravel()[idx].astype(np.float64)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return (y - X @ beta).astype(np.float32)


def detrended_r_t(img, idx_t, X_t, XtX_inv_t, real_res_t):
    """Pearson r between the detrended region of img (H, W) and a precomputed real residual."""
    torch = _torch()
    y = img.reshape(-1)[idx_t]
    beta = XtX_inv_t @ (X_t.t() @ y)
    res = y - X_t @ beta
    a = res - res.mean(); b = real_res_t - real_res_t.mean()
    return (a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum() + 1e-12)


__all__ = ["US_FOLDS", "fold_of", "LearnedEcho", "SectorPost", "gauss_kernel", "smooth_t", "region_design",
           "detrended_resid_np", "detrended_r_t"]
