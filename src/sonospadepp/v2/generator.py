"""The v2 generator: v1's dual-input SPADE generator (usbg.texture_gan._spade_generator, built from the
conditioning channel count) with optional additions that start with NO effect, so a v2 generator initialized
from a v1 state dict reproduces v1 exactly. v1's module names are kept for v1's parts (stem, down, res, up,
up_blocks, out), so v1 checkpoints load with only the new keys missing.

  noise_inject (A5)  StyleGAN-style per-pixel Gaussian noise after each upsampling block (64 and 128 px at the
                     default size), scaled per tissue and per channel: scale = onehot(label) @ S_k, with S_k
                     initialized to zero. At inference the noise is fresh per call (seedable).
  gain_sigma (gain field, added 2026-11-19)  the output keeps the physics render's fine structure and the network
                     changes only its low-frequency part: out = phys + blur(net(phys, seg) - phys), with a fixed
                     Gaussian blur of gain_sigma px (reflect padding). In the log-compressed domain an additive
                     smooth field is a smooth multiplicative gain on the envelope, which leaves window-level
                     Nakagami m of the render's speckle unchanged up to the variation of the gain within a window.
"""
from __future__ import annotations


def build_generator(n_cond, ngf=32, n_blocks=6, spade_hidden=64, noise_inject=False, n_label=13, gain_sigma=0.0):
    """n_cond: conditioning channels seen by the SPADE label path (13 tissue one-hot + any extra channels).
    n_label: the tissue one-hot channels (first n_label of the conditioning), which key the noise scales, so
    widening n_cond for extra channels does not change the noise parameters."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from usbg import texture_gan as TG

    base = TG._spade_generator(n_cond, ngf=ngf, n_blocks=n_blocks, spade_hidden=spade_hidden)
    if not noise_inject and not gain_sigma:
        return base

    kern = None
    if gain_sigma:
        r = int(max(1, round(3 * gain_sigma)))
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        kern = torch.exp(-0.5 * (t / float(gain_sigma)) ** 2)
        kern = kern / kern.sum()

    up_channels = []
    c = base.feat_channels[1]
    for _ in base.up_blocks:
        c //= 2
        up_channels.append(c)

    class SPADEGenV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem, self.down, self.res = base.stem, base.down, base.res
            self.up, self.up_blocks, self.out = base.up, base.up_blocks, base.out
            self.feat_channels = base.feat_channels
            if noise_inject:
                self.noise_scale = nn.ParameterList([nn.Parameter(torch.zeros(n_label, ch)) for ch in up_channels])
            self.noise_gen = None                     # optional torch.Generator for reproducible inference
            self.gain_sigma = float(gain_sigma or 0.0)
            if kern is not None:
                self.register_buffer("gain_kernel", kern, persistent=False)

        def _blur(self, d):
            k = self.gain_kernel.to(d.dtype)
            r = (k.numel() - 1) // 2
            d = F.conv2d(F.pad(d, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
            return F.conv2d(F.pad(d, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))

        def _noise(self, h, seg, j):
            s = F.interpolate(seg[:, :n_label], size=h.shape[2:], mode="nearest")  # tissue one-hot only
            scale = torch.einsum("bkhw,kc->bchw", s, self.noise_scale[j])
            if self.noise_gen is not None:
                z = torch.randn(h.shape[0], 1, *h.shape[2:], generator=self.noise_gen).to(h.device)
            else:
                z = torch.randn(h.shape[0], 1, *h.shape[2:], device=h.device)
            return h + scale * z

        def forward(self, phys, seg):
            h = self.down(self.stem(phys))
            for b in self.res:
                h = b(h, seg)
            for j, (up, blk) in enumerate(zip(self.up, self.up_blocks)):
                h = blk(up(h), seg)
                if noise_inject:
                    h = self._noise(h, seg, j)
            out = self.out(h)
            if self.gain_sigma:
                out = torch.clamp(phys + self._blur(out - phys), -1.0, 1.0)
            return out

        def forward_features(self, phys, seg):
            return base.__class__.forward_features(self, phys, seg)

    return SPADEGenV2()
