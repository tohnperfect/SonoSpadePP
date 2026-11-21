"""The four compared variants, the M3 trainer, and one translator class for all of them.

  b0  SonoSPADE as released: physics from base labels, SPADE conditioned on 13 one-hot labels
  m1  CT sub-labels: physics and conditioning from the 14-label M1 map (gas, CT vessels, cysts)
  m2  HU-modulated physics: physics from base labels with per-pixel CT modulation, 13 labels
  m3  HU channel: b0's physics, SPADE conditioned on 13 one-hot labels + 1 relative-HU channel

Every variant trains with the same poses, render seeds, real pool, iterations and loss weights, and
is evaluated against the same 13-class base labels.
"""
from __future__ import annotations

import numpy as np
from usbg import texture_gan as TG

from .labels import N_BASE, N_PP

VARIANTS = {
    "b0": dict(phys="phys_b0", lab="lab_base", n_lab=N_BASE, hu=False,
               title="SonoSPADE (labels + physics)"),
    "m1": dict(phys="phys_m1", lab="lab_pp", n_lab=N_PP, hu=False,
               title="M1: CT-derived sub-labels"),
    "m2": dict(phys="phys_m2", lab="lab_base", n_lab=N_BASE, hu=False,
               title="M2: HU-modulated physics"),
    "m3": dict(phys="phys_b0", lab="lab_base", n_lab=N_BASE, hu=True,
               title="M3: HU as a SPADE channel"),
}


def s_us_key(variant: str) -> str:
    v = VARIANTS[variant]
    return f"{v['phys']}__{v['lab']}"


def hu_channel(z):
    """Relative-HU conditioning channel in (-1, 1): tanh(z / 3)."""
    return np.tanh(np.asarray(z, np.float32) / 3.0)


def cond_tensor(lab, z, n_lab, use_hu, device):
    """(B,H,W) labels (+ (B,H,W) z) -> (B, n_lab [+1], H, W) float conditioning tensor."""
    import torch
    oh = TG.onehot_labels(np.asarray(lab), n_lab, device)
    if not use_hu:
        return oh
    h = torch.from_numpy(hu_channel(z)).to(device)
    if h.dim() == 2:
        h = h[None]
    return torch.cat([oh, h[:, None]], dim=1)


class HUSonoSPADE(TG.SonoSPADE):
    """M3 trainer: SonoSPADE whose SPADE input is [one-hot labels, relative-HU channel].

    The losses are the release's, unchanged; the only difference is that the target tissue map for
    the OASIS and segmentation-consistency losses is read from the one-hot block, never from the
    HU channel."""

    def __init__(self, device, n_lab=N_BASE, **kw):
        super().__init__(device, n_classes=n_lab + 1, **kw)
        self.n_lab = n_lab

    def generator_losses(self, phys, cond, fake):
        from usbg.segmenter import seg_ce_dice_loss
        target_lab = cond[:, :self.n_lab].argmax(dim=1)
        seg_target = self._seg_target(target_lab)
        loss_g_adv = self._balanced_ce(self.D(fake), seg_target)
        fk = self.G.forward_features(fake, cond)
        fq = self.G.forward_features(phys, cond)
        q, ids = self.H(fk); k, _ = self.H(fq, patch_ids=ids)
        loss_nce = TG._patchnce_loss(q, k, self.tau)
        seg_logits = self.S.net(self._to01(fake))
        loss_seg = seg_ce_dice_loss(seg_logits, seg_target, n_classes=self.n_seg,
                                    dice_weight=self.lambda_seg_dice)
        out = {"adv": loss_g_adv, "nce": loss_nce, "seg": loss_seg}
        if self.lambda_tc and self.lambda_tc > 0:
            dy, dx = 2, 2
            shifted = self.G(self._shift(phys, dy, dx), self._shift(cond, dy, dx))
            out["tc"] = (shifted - self._shift(fake, dy, dx)).abs().mean()
        return out

    def save(self, path):
        import torch
        super().save(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state.update({"n_lab": self.n_lab, "hu_channel": True})
        torch.save(state, path)


class Translator:
    """Inference for any variant checkpoint: (physics img01, aligned label [, z]) -> img01."""

    def __init__(self, ckpt_path, device=None):
        import torch
        from usbg.render_lotus import get_device
        self.device = device or get_device()
        st = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.n_cond = int(st["n_classes"])
        self.use_hu = bool(st.get("hu_channel", False))
        self.n_lab = int(st.get("n_lab", self.n_cond))
        self.G = TG._spade_generator(self.n_cond, ngf=st.get("ngf", 32), n_blocks=st.get("n_blocks", 6),
                                     spade_hidden=st.get("spade_hidden", 64)).to(self.device)
        self.G.load_state_dict(st["G_spade"])
        self.G.eval()

    def batch(self, phys01, lab, z=None):
        """phys01 (B,H,W) float, lab (B,H,W) int, z (B,H,W) float or None -> (B,H,W) float [0,1]."""
        import torch
        phys01 = np.asarray(phys01, np.float32)
        if self.use_hu and z is None:
            z = np.zeros_like(phys01)
        with torch.no_grad():
            a = torch.from_numpy(phys01[:, None] * 2 - 1).to(self.device)
            c = cond_tensor(lab, z, self.n_lab, self.use_hu, self.device)
            out = self.G(a, c)[:, 0].cpu().numpy()
        return np.clip((out + 1) / 2, 0, 1).astype(np.float32)

    def __call__(self, phys01, lab, z=None):
        zz = None if z is None else np.asarray(z)[None]
        return self.batch(np.asarray(phys01)[None], np.asarray(lab)[None], zz)[0]


__all__ = ["VARIANTS", "s_us_key", "hu_channel", "cond_tensor", "HUSonoSPADE", "Translator"]
