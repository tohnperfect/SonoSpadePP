"""spatial_echo.py: M6, a spatial CT-to-echogenicity network.

M4 maps each pixel's relative HU through a per-tissue curve: no context, one number in, one out.
M6 lets a small convolutional network see the CT around each pixel (about 1 to 2 cm) and predict
the log backscatter factor there:

    log f(p) = clip( a_l(p) + CNN(x)(p), -2, 2 ),   on the soft-tissue classes only (as M4)

with x the one-hot tissue labels plus, for M6, the relative HU z (/4) and the HU scaled from
[-200, 300] to [0, 1]. The control M6c is the same network on the labels alone: it can learn
context (the renal sinus inside the kidney outline, the distance to an interface) but not CT
intensity, so M6 minus M6c is what the intensity adds once spatial context is allowed. The last
layer starts at zero, so an untrained model is the per-tissue offsets a_l (zero: the base render).

The network runs on the scan grid (lines x samples, about 0.65 x 0.23 mm at mid depth); kernels
are 5 lines x 9 samples, dilated 1, 2 and 4, for a receptive field of 29 lines x 57 samples.
"""
from __future__ import annotations

import numpy as np

from .labels import HU_MODULATED, N_PP

N_IN_LAB = 13


def _nn():
    import torch.nn as nn
    return nn


class SpatialEcho:
    """Wrapper holding the torch module, the per-tissue offsets and the input encoding."""

    def __init__(self, use_ct=True, width=32, clamp=2.0, device="cpu"):
        import torch
        nn = _nn()
        self.use_ct, self.clamp, self.device = use_ct, clamp, device
        cin = N_IN_LAB + (2 if use_ct else 0)
        layers, c = [], cin
        for d in (1, 2, 4):
            layers += [nn.Conv2d(c, width, (5, 9), dilation=d, padding=(2 * d, 4 * d)), nn.ReLU()]
            c = width
        last = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last).to(device)
        self.a = torch.zeros(N_PP, device=device, requires_grad=True)
        mod = np.zeros(N_PP, bool); mod[list(HU_MODULATED)] = True
        self.mod = torch.as_tensor(mod, device=device)

    def param_groups(self, lr_net=1e-3, lr_a=0.03, wd=1e-4):
        return [{"params": self.net.parameters(), "lr": lr_net, "weight_decay": wd}, {"params": [self.a], "lr": lr_a}]

    def encode(self, lab_t, z_t, hu_t):
        """lab (B,L,S) long, z/hu (B,L,S) float -> network input (B,C,L,S)."""
        import torch
        import torch.nn.functional as F
        x = F.one_hot(lab_t.clamp(0, N_IN_LAB - 1), N_IN_LAB).permute(0, 3, 1, 2).float()
        if self.use_ct:
            x = torch.cat([x, (z_t / 4.0).clamp(-2, 2)[:, None], ((hu_t.clamp(-200, 300) + 200) / 500.0)[:, None]], 1)
        return x

    def log_factor(self, lab_t, z_t, hu_t):
        """(B,L,S) log backscatter factor and the CNN term (for its penalty)."""
        import torch
        cnn = self.net(self.encode(lab_t, z_t, hu_t))[:, 0]
        lf = (self.a[lab_t] + cnn).clamp(-self.clamp, self.clamp)
        return torch.where(self.mod[lab_t], lf, torch.zeros_like(lf)), cnn

    def factor_numpy(self, lab, z, hu):
        import torch
        t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)[None]
        with torch.no_grad():
            lf, _ = self.log_factor(t(lab, torch.long), t(z, torch.float32), t(hu, torch.float32))
        return np.exp(lf[0].cpu().numpy()).astype(np.float32)

    def state(self):
        import torch
        return {"net": {k: v.detach().cpu() for k, v in self.net.state_dict().items()}, "a": self.a.detach().cpu(),
                "use_ct": self.use_ct}

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


__all__ = ["SpatialEcho"]
