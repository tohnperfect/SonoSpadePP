"""DiffAugment (Zhao et al., NeurIPS 2020) for single-channel frames in [0, 1]: differentiable brightness,
contrast, translation and cutout, applied identically in kind to real and fake inputs of a discriminator so a
small real pool is not memorized. Policy string as in the reference code: 'color,translation,cutout'."""
from __future__ import annotations


def _brightness(x):
    import torch
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.5   # +-0.25 in [0, 1] units


def _contrast(x):
    import torch
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + m


def _translation(x, ratio=0.125):
    import torch
    import torch.nn.functional as F
    B, _, H, W = x.shape
    sh, sw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    ty = torch.randint(-sh, sh + 1, (B, 1, 1), device=x.device)
    tx = torch.randint(-sw, sw + 1, (B, 1, 1), device=x.device)
    gy, gx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
    gy = torch.clamp(gy[None] + ty + 1, 0, H + 1)
    gx = torch.clamp(gx[None] + tx + 1, 0, W + 1)
    xp = F.pad(x, [1, 1, 1, 1])
    idx = (gy * (W + 2) + gx).view(B, 1, -1)
    return torch.gather(xp.view(B, 1, -1), 2, idx).view(B, 1, H, W)


def _cutout(x, ratio=0.5):
    import torch
    B, _, H, W = x.shape
    ch, cw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    oy = torch.randint(0, H + (1 - ch % 2), (B, 1, 1), device=x.device)
    ox = torch.randint(0, W + (1 - cw % 2), (B, 1, 1), device=x.device)
    gy, gx = torch.meshgrid(torch.arange(ch, device=x.device), torch.arange(cw, device=x.device), indexing="ij")
    gy = torch.clamp(gy[None] + oy - ch // 2, 0, H - 1)
    gx = torch.clamp(gx[None] + ox - cw // 2, 0, W - 1)
    mask = torch.ones(B, H, W, device=x.device, dtype=x.dtype)
    mask[torch.arange(B, device=x.device)[:, None, None], gy, gx] = 0
    return x * mask[:, None]


def diff_augment(x, policy: str = "color,translation,cutout"):
    """x: (B, 1, H, W) in [0, 1]. Returns an augmented tensor (gradients flow through)."""
    if not policy:
        return x
    for p in policy.split(","):
        if p == "color":
            x = _contrast(_brightness(x))
        elif p == "translation":
            x = _translation(x)
        elif p == "cutout":
            x = _cutout(x)
        elif p:
            raise ValueError(f"unknown DiffAugment op {p!r}")
    return x
