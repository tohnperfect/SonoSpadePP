"""exemplar.py: M9, the translator given ONE real ultrasound image of the same patient (exemplar-based
translation, after CoCosNet: Zhang et al., CVPR 2020).

Setting: for a new patient, the CT and one real US image at a known plane (registered to the CT) are
available; every other plane of the patient is to be synthesised. M9 learns, across the training
patients, to carry the appearance of that one image to the other planes:

  correspondence   an encoder maps the target plane's CT inputs (one-hot labels, and the CT channels
                   tanh(z/3) and HU unless they are zeroed) and the reference plane's CT inputs to
                   features at 32 x 32. A softmax over reference positions of
                   scale * cosine(feature_t, feature_r) - lam * distance^2 (a locality prior: the
                   planes share the probe's fan geometry) gives, per target position, a soft match;
                   its expected reference position is a flow that warps the full-resolution reference
                   US onto the target plane (W), with the match confidence (max probability) as C.
  translation      the generator of M7 with W and C as two more channels in its content path and in
                   every SPADE block (zero-initialised, so the untrained M9 is B0).

The CT-free control (M9c) zeroes the CT channels in both parts, so its correspondence is found from
the label maps alone. With scale small at the start the warp is close to the identity, i.e. W starts
as the reference image copied to the target plane.
"""
from __future__ import annotations

import math

import numpy as np

from . import paired_translator as PT

N_REF = 2          # warped reference W and its confidence C


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def corr_module(device, in_ch=PT.N_LAB + PT.N_CT, feat=64, res=32, sigma_cells=3.0):
    torch, nn, F = _torch()

    class Corr(nn.Module):
        def __init__(self):
            super().__init__()
            c = [in_ch, 32, 64, 64]
            layers = []
            for i, co in enumerate(c[1:]):
                layers += [nn.Conv2d(c[i], co, 3, stride=1 if i == 0 else 2, padding=1), nn.InstanceNorm2d(co), nn.LeakyReLU(0.2)]
            layers += [nn.Conv2d(64, feat, 3, padding=1)]
            self.enc = nn.Sequential(*layers)
            self.log_scale = nn.Parameter(torch.tensor(0.0))                     # scale = 1 at start
            self.log_lam = nn.Parameter(torch.tensor(math.log(1.0 / (2 * sigma_cells ** 2))))
            ys, xs = torch.meshgrid(torch.arange(res, dtype=torch.float32), torch.arange(res, dtype=torch.float32), indexing="ij")
            pos = torch.stack([xs.flatten(), ys.flatten()], 1)                   # (N, 2) in cells
            self.register_buffer("d2", ((pos[:, None] - pos[None]) ** 2).sum(-1))
            self.register_buffer("grid", pos / (res - 1) * 2 - 1)                # (N, 2) in [-1, 1], (x, y)
            self.res = res

        def forward(self, x_t, x_r, valid_r, ref_img):
            """x_t, x_r (B, in_ch, 128, 128); valid_r (B, 128, 128) bool/float; ref_img (B, 128, 128) [0,1]
            -> W (B, 128, 128) the reference warped to the target, C (B, 128, 128) confidence in [0, 1]."""
            B, _, H, Wd = x_t.shape
            ft = F.normalize(self.enc(x_t), dim=1).flatten(2)                 # (B, C, N)
            fr = F.normalize(self.enc(x_r), dim=1).flatten(2)
            logits = self.log_scale.exp() * torch.einsum("bcn,bcm->bnm", ft, fr) - self.log_lam.exp() * self.d2[None]
            vr = F.adaptive_avg_pool2d(valid_r.float()[:, None], self.res).flatten(1) > 0.5   # (B, N)
            logits = logits.masked_fill(~vr[:, None, :], -1e4)
            P = logits.softmax(-1)
            flow = (P @ self.grid).view(B, self.res, self.res, 2).permute(0, 3, 1, 2)          # (B, 2, r, r)
            flow = F.interpolate(flow, size=(H, Wd), mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
            W = F.grid_sample(ref_img[:, None], flow, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
            C = F.interpolate(P.max(-1).values.view(B, 1, self.res, self.res), size=(H, Wd), mode="bilinear",
                              align_corners=True)[:, 0]
            return W, C

    return Corr().to(device)


def ct_inputs(lab, z, hu01, use_ct, device):
    """(B,H,W) arrays -> (B, N_LAB + N_CT, H, W): one-hot labels and the CT channels (zeros if not use_ct)."""
    torch, _, F = _torch()
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    ct = torch.stack([torch.tanh(t(z) / 3.0), t(hu01) * 2 - 1], 1)
    if not use_ct:
        ct = torch.zeros_like(ct)
    lab_t = torch.as_tensor(np.ascontiguousarray(lab), dtype=torch.long, device=device).clamp(0, PT.N_LAB - 1)
    return torch.cat([F.one_hot(lab_t, PT.N_LAB).permute(0, 3, 1, 2).float(), ct], 1)


def gen_inputs(phys01, x_ct, W, C, device):
    """Generator inputs: content [render, CT(2), W, C], conditioning [one-hot, CT(2), W, C]."""
    torch, _, _ = _torch()
    p = torch.as_tensor(np.ascontiguousarray(phys01), dtype=torch.float32, device=device)[:, None] * 2 - 1
    ref = torch.stack([W * 2 - 1, C * 2 - 1], 1)
    return torch.cat([p, x_ct[:, PT.N_LAB:], ref], 1), torch.cat([x_ct, ref], 1)


__all__ = ["corr_module", "ct_inputs", "gen_inputs", "N_REF"]
