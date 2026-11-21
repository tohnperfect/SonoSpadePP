"""A2 (WP2): per-region style codes (SEAN-style) for per-tissue appearance control and diversity.

* StyleEncoder E_s: a small CNN on a frame in [0, 1] -> feature map at H/4; region-wise average pooling under a
  label map (S_US's grouped scheme) gives one code per tissue: mean and log-variance (d dims each).
* SEANNorm: wraps a v1 SPADE normalization, keeping its submodules and names (param_free, shared, to_gamma,
  to_beta), and adds a style branch: gamma = (1 - a_g) * gamma_label + a_g * gamma_style (beta likewise), where
  gamma_style is a 3x3 conv of the style map (each pixel carries its tissue's code). a_g and a_b start at 0, so
  a converted generator is exactly v1 until training moves them.
* The conditioning tensor passed to the generator is [one-hot label (K channels); any extra label-like channels
  (e.g. SonoSpadePP's CT channels); style map (d channels)]. The SPADE residual blocks pass it to their norms
  unchanged, and each SEANNorm splits it at its `shared` conv's input width, read at run time, so widening that
  conv for extra channels (SonoSpadePP's paired_translator.build) keeps working.
* Training (unpaired): each fake takes its codes either from a real exemplar (E_s on a real frame and its S_US
  pseudo-label, reparameterized) or from the prior N(0, I), half and half; a KL term keeps the exemplar codes
  near the prior; latent regression (E_s's mean code of the fake under its input label must equal the code used,
  L1, target detached) keeps the codes meaningful and rules out a constant encoder. Tissues absent from the
  exemplar fall back to the style bank mean.
* INADE (fallback, INADENorm): per-class modulation distributions with instance-level noise codes, no
  encoder; diversity kept by a mode-seeking term in training (trainer).
* Inference: no encoder. A per-tissue Gaussian style bank (mean and diagonal SD of real-pool codes) gives
  style=None -> bank mean (deterministic) and 'sample' -> a draw.
"""
from __future__ import annotations


class StyleEncoder:
    def __new__(cls, n_groups, dim=16, width=64):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim, self.n_groups = dim, n_groups
                self.net = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, width, 3, stride=2, padding=1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(width, width, 3, stride=2, padding=1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(width, 2 * dim, 1))

            def forward(self, x01, lab, min_frac=0.02):
                """x01 (B,1,H,W) in [0,1]; lab (B,H,W) long in the grouped scheme.
                Returns mu, logvar (B, K, d) and present (B, K) bool."""
                f = self.net(x01)
                oh = F.one_hot(lab.clamp(0, self.n_groups - 1), self.n_groups).permute(0, 3, 1, 2).float()
                w = F.interpolate(oh, size=f.shape[-2:], mode="area")                      # (B,K,h,w)
                s = w.sum(dim=(2, 3))                                                       # (B,K)
                pooled = torch.einsum("bkhw,bchw->bkc", w, f) / s.clamp(min=1e-6)[..., None]
                mu, logvar = pooled[..., :self.dim], pooled[..., self.dim:].clamp(-8, 4)
                present = s >= min_frac * f.shape[-1] * f.shape[-2]
                return mu, logvar, present

        return Enc()


class SEANNorm:
    def __new__(cls, spade, style_dim, n_cond):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Norm(nn.Module):
            def __init__(self):
                super().__init__()
                self.param_free, self.shared = spade.param_free, spade.shared
                self.to_gamma, self.to_beta = spade.to_gamma, spade.to_beta
                c = spade.to_gamma.out_channels
                self.style_gamma = nn.Conv2d(style_dim, c, 3, padding=1)
                self.style_beta = nn.Conv2d(style_dim, c, 3, padding=1)
                self.alpha_g = nn.Parameter(torch.zeros(1))
                self.alpha_b = nn.Parameter(torch.zeros(1))
                self.n_cond = n_cond

            def forward(self, x, cond):
                normed = self.param_free(x)
                cond = F.interpolate(cond, size=x.shape[2:], mode="nearest")
                k = self.shared[0].in_channels          # split where the label part ends, read at run time,
                seg, st = cond[:, :k].contiguous(), cond[:, k:].contiguous()   # so channel widening still works
                a = self.shared(seg)
                g, b = self.to_gamma(a), self.to_beta(a)
                if st.shape[1]:
                    g = (1 - self.alpha_g) * g + self.alpha_g * self.style_gamma(st)
                    b = (1 - self.alpha_b) * b + self.alpha_b * self.style_beta(st)
                return normed * (1 + g) + b

        return Norm()


def convert_to_sean(gen, style_dim, n_cond):
    """In place: every SPADE norm of the residual and upsampling blocks becomes a SEANNorm (v1 names kept)."""
    for blk in list(gen.res) + list(gen.up_blocks):
        for name in ("norm0", "norm1", "norm_s"):
            if hasattr(blk, name):
                setattr(blk, name, SEANNorm(getattr(blk, name), style_dim, n_cond))
    gen.style_dim = style_dim
    return gen


class INADENorm:
    """INADE-style instance-adaptive modulation (Tan et al., 2021) on a v1 SPADE norm (WP2 fallback). Per tissue
    class k and channel: a learned mean mu[k] and scale sigma[k] of the modulation; every tissue instance draws a
    noise code z (d dims, carried by the style map) that a 1x1 projection P maps to the channels:
        gamma = gamma_label + mu_g[k] + sigma_g[k] * (P_g z),   beta likewise.
    mu starts at 0 and sigma at sigma0 > 0, so the noise acts from the first step (SEAN's blend weights started at
    0 and stayed there); z = 0 gives each class's mean appearance. No encoder: the codes are N(0, I) by design."""

    def __new__(cls, spade, style_dim, n_label, sigma0=0.05):
        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Norm(nn.Module):
            def __init__(self):
                super().__init__()
                self.param_free, self.shared = spade.param_free, spade.shared
                self.to_gamma, self.to_beta = spade.to_gamma, spade.to_beta
                c = spade.to_gamma.out_channels
                self.proj_g = nn.Conv2d(style_dim, c, 1, bias=False)
                self.proj_b = nn.Conv2d(style_dim, c, 1, bias=False)
                self.mu_g = nn.Parameter(torch.zeros(n_label, c))
                self.mu_b = nn.Parameter(torch.zeros(n_label, c))
                self.logsig_g = nn.Parameter(torch.full((n_label, c), math.log(sigma0)))
                self.logsig_b = nn.Parameter(torch.full((n_label, c), math.log(sigma0)))
                self.n_label = n_label

            def forward(self, x, cond):
                normed = self.param_free(x)
                cond = F.interpolate(cond, size=x.shape[2:], mode="nearest")
                k = self.shared[0].in_channels
                seg, st = cond[:, :k].contiguous(), cond[:, k:].contiguous()
                a = self.shared(seg)
                g, b = self.to_gamma(a), self.to_beta(a)
                if st.shape[1]:
                    oh = seg[:, :self.n_label]
                    g = g + torch.einsum("bkhw,kc->bchw", oh, self.mu_g) \
                        + torch.einsum("bkhw,kc->bchw", oh, self.logsig_g.exp()) * self.proj_g(st)
                    b = b + torch.einsum("bkhw,kc->bchw", oh, self.mu_b) \
                        + torch.einsum("bkhw,kc->bchw", oh, self.logsig_b.exp()) * self.proj_b(st)
                return normed * (1 + g) + b

        return Norm()


def convert_to_inade(gen, style_dim, n_label, sigma0=0.05):
    """In place: every SPADE norm of the residual and upsampling blocks becomes an INADENorm (v1 names kept)."""
    for blk in list(gen.res) + list(gen.up_blocks):
        for name in ("norm0", "norm1", "norm_s"):
            if hasattr(blk, name):
                setattr(blk, name, INADENorm(getattr(blk, name), style_dim, n_label, sigma0))
    gen.style_dim = style_dim
    return gen


def convert_style(gen, kind, style_dim, n_label, sigma0=0.05):
    if kind == "sean":
        return convert_to_sean(gen, style_dim, n_label)
    if kind == "inade":
        return convert_to_inade(gen, style_dim, n_label, sigma0)
    return gen


def style_map(codes, lab_grouped, n_groups):
    """codes (B, K, d), lab_grouped (B, H, W) -> (B, d, H, W): each pixel gets its tissue's code."""
    import torch
    import torch.nn.functional as F
    oh = F.one_hot(lab_grouped.clamp(0, n_groups - 1), n_groups).permute(0, 3, 1, 2).float()
    return torch.einsum("bkhw,bkd->bdhw", oh, codes)


class StyleBank:
    """Per-tissue Gaussian (mean, diagonal SD) of real-pool codes, accumulated as running sums."""

    def __init__(self, n_groups, dim):
        import torch
        self.n, self.d = n_groups, dim
        self.s1 = torch.zeros(n_groups, dim)
        self.s2 = torch.zeros(n_groups, dim)
        self.cnt = torch.zeros(n_groups)

    def add(self, mu, present):
        mu, present = mu.detach().cpu(), present.detach().cpu()
        for k in range(self.n):
            sel = mu[present[:, k], k]
            if len(sel):
                self.s1[k] += sel.sum(0)
                self.s2[k] += (sel ** 2).sum(0)
                self.cnt[k] += len(sel)

    def mean(self):
        import torch
        return self.s1 / self.cnt.clamp(min=1)[:, None]

    def sd(self):
        import torch
        m = self.mean()
        return (self.s2 / self.cnt.clamp(min=1)[:, None] - m ** 2).clamp(min=1e-8).sqrt()

    def state(self):
        return {"mean": self.mean(), "sd": self.sd(), "count": self.cnt.clone()}
