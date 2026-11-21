"""A1 (WP1): a foundation-model feature discriminator D_fm and per-tissue masked feature matching.

D_fm follows Projected GAN / vision-aided GAN: a FROZEN pretrained encoder gives token maps at several depths
(shallow ones keep more local texture), and small trainable heads (1x1 conv, 3x3 conv, 1x1 conv, spectral
norm) output a real / fake logit per token, per depth. Loss: non-saturating (softplus), the two-class analogue
of the per-pixel cross-entropy the v1 OASIS discriminator uses.

Per-tissue masked feature matching L_fm_tissue: for each tissue k and depth, the label mask is averaged onto
the token grid (a token counts for k when at least half of it is k and inside the fan); the masked mean and
diagonal variance of the fake tokens (masked by the INPUT label) are matched to an EMA bank of the same
statistics on the real pool (masked by S_US pseudo-labels). Tissues are in S_US's (grouped) class scheme;
background is skipped; a tissue is skipped in a batch with fewer than `min_cells` tokens.
"""
from __future__ import annotations


def _head(dim, width=256):
    import torch.nn as nn
    sn = nn.utils.spectral_norm
    return nn.Sequential(sn(nn.Conv2d(dim, width, 1)), nn.LeakyReLU(0.2, True),
                         sn(nn.Conv2d(width, width, 3, padding=1)), nn.LeakyReLU(0.2, True),
                         sn(nn.Conv2d(width, 1, 1)))


class FMHeads:
    def __new__(cls, dims, width=256):
        """dims: channel count of each feature map (one head per map)."""
        import torch.nn as nn

        class Heads(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = nn.ModuleList([_head(d, width) for d in dims])

            def forward(self, feats):
                return [h(f) for h, f in zip(self.heads, feats)]

        return Heads()


def ns_d_loss(logits_real, logits_fake):
    import torch.nn.functional as F
    return sum(F.softplus(-r).mean() + F.softplus(f).mean() for r, f in zip(logits_real, logits_fake)) / len(logits_real)


def ns_g_loss(logits_fake):
    import torch.nn.functional as F
    return sum(F.softplus(-f).mean() for f in logits_fake) / len(logits_fake)


def token_weights(lab, n_classes, grid_hw, fan=None, thresh=0.5):
    """lab (B, H, W) long in S_US's scheme -> (B, n_classes, h, w) {0, 1} token membership."""
    import torch
    import torch.nn.functional as F
    oh = F.one_hot(lab.clamp(0, n_classes - 1), n_classes).permute(0, 3, 1, 2).float()
    if fan is not None:
        oh = oh * fan
    h, w = (int(x) for x in grid_hw)
    if oh.device.type == "mps" and (oh.shape[-2] % h or oh.shape[-1] % w):
        # MPS has no adaptive average pooling for non-divisible sizes (e.g. 128 px labels on a ViT's 14 x 14 grid);
        # the CPU result is the same operation
        frac = F.interpolate(oh.cpu(), size=(h, w), mode="area").to(oh.device)
    else:
        frac = F.interpolate(oh, size=(h, w), mode="area")
    return (frac >= thresh).float()


def masked_stats(feat, w):
    """feat (B, D, h, w), w (B, h, w) {0,1} -> (mean (D,), var (D,), count)."""
    n = w.sum()
    if n < 1:
        return None, None, n
    wf = w[:, None]
    mu = (feat * wf).sum(dim=(0, 2, 3)) / n
    var = (((feat - mu[None, :, None, None]) ** 2) * wf).sum(dim=(0, 2, 3)) / n
    return mu, var, n


class TissueBank:
    """EMA of the real pool's per-(depth, tissue) token mean and variance."""

    def __init__(self, momentum=0.99):
        self.m = momentum
        self.mu, self.var, self.updates = {}, {}, {}

    def update(self, key, mu, var):
        import torch
        mu, var = mu.detach(), var.detach()
        if key not in self.mu:
            self.mu[key], self.var[key], self.updates[key] = mu.clone(), var.clone(), 1
        else:
            self.mu[key].mul_(self.m).add_(mu, alpha=1 - self.m)
            self.var[key].mul_(self.m).add_(var, alpha=1 - self.m)
            self.updates[key] += 1
        return torch.as_tensor(self.updates[key])

    def state(self):
        return {"mu": {f"{k[0]}_{k[1]}": v.cpu() for k, v in self.mu.items()},
                "var": {f"{k[0]}_{k[1]}": v.cpu() for k, v in self.var.items()},
                "updates": {f"{k[0]}_{k[1]}": int(v) for k, v in self.updates.items()}}


def fm_tissue_loss(feats_fake, w_fake, bank, n_classes, min_cells=4, min_updates=10, eps=1e-4):
    """Sum over depths of the mean over tissues present in the fake batch (and in the bank) of
    mean_d (mu_f - mu_r)^2 / (var_r + eps) + mean_d (log(var_f + eps) - log(var_r + eps))^2."""
    import torch
    total, terms = 0.0, 0
    for li, f in enumerate(feats_fake):
        for k in range(1, n_classes):
            key = (li, k)
            if key not in bank.mu or bank.updates[key] < min_updates:
                continue
            mu, var, n = masked_stats(f, w_fake[li][:, k])
            if mu is None or n < min_cells:
                continue
            mr, vr = bank.mu[key], bank.var[key]
            total = total + ((mu - mr) ** 2 / (vr + eps)).mean() + \
                ((torch.log(var + eps) - torch.log(vr + eps)) ** 2).mean()
            terms += 1
    return (total / terms if terms else None), terms
