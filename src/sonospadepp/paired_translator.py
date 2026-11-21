"""paired_translator.py: M7, the whole SPADE translator fine-tuned on same-patient pairs, with the CT image.

M4 to M6 only change the backscatter of the physics render and keep the B0 generator frozen. M7
makes the generator itself learn from the paired data and gives it the CT image directly, in both
of its paths:

  content path   the stem convolution sees [render, tanh(z/3), HU] instead of the render alone;
  conditioning   every SPADE block is modulated by [one-hot labels, tanh(z/3), HU] instead of the
                 labels alone (HU scaled from [-200, 300] to [-1, 1]).

The new input channels start with zero weights, so the untrained M7 is exactly B0. All generator
weights are then fine-tuned with the same-plane agreement loss. The control M7c has the identical
architecture and training, with both CT channels set to zero: it can re-learn texture, contrast
and context from the labels and the render, but never sees the CT intensity.
"""
from __future__ import annotations

import numpy as np

N_LAB = 13
N_CT = 2


def build(ckpt_path, device, n_extra=N_CT):
    """B0 checkpoint -> generator with n_extra extra channels (default the N_CT CT channels) in the stem
    and in every SPADE block, initialised so that its output equals B0's."""
    import torch
    import torch.nn as nn
    from usbg import texture_gan as TG
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert int(st["n_classes"]) == N_LAB and not st.get("hu_channel", False), "expects a B0 checkpoint"
    kw = dict(ngf=st.get("ngf", 32), n_blocks=st.get("n_blocks", 6), spade_hidden=st.get("spade_hidden", 64))
    G = TG._spade_generator(N_LAB + n_extra, **kw)
    old = st["G_spade"]
    new = G.state_dict()
    for k, v in old.items():
        if new[k].shape == v.shape:
            new[k] = v.clone()
        else:                                            # a SPADE shared conv: (hidden, 13 + n_extra, k, k)
            w = torch.zeros_like(new[k]); w[:, :v.shape[1]] = v; new[k] = w
    G.load_state_dict(new)
    conv = G.stem[1]                                     # Conv2d(1, ngf, 7) -> Conv2d(1 + n_extra, ngf, 7)
    c2 = nn.Conv2d(1 + n_extra, conv.out_channels, conv.kernel_size)
    with torch.no_grad():
        c2.weight.zero_(); c2.weight[:, :1] = conv.weight; c2.bias.copy_(conv.bias)
    G.stem[1] = c2
    return G.to(device)


def inputs(phys01, lab, z, hu01, use_ct, device):
    """Batch tensors: phys01 (B,H,W) in [0,1], lab (B,H,W) int, z (B,H,W) relative HU, hu01 (B,H,W) in
    [0,1] -> content (B,1+N_CT,H,W) and conditioning (B,N_LAB+N_CT,H,W)."""
    import torch
    import torch.nn.functional as F
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    p = t(phys01)[:, None] * 2 - 1
    ct = torch.stack([torch.tanh(t(z) / 3.0), t(hu01) * 2 - 1], 1)
    if not use_ct:
        ct = torch.zeros_like(ct)
    lab_t = torch.as_tensor(np.ascontiguousarray(lab), dtype=torch.long, device=device).clamp(0, N_LAB - 1)
    oh = F.one_hot(lab_t, N_LAB).permute(0, 3, 1, 2).float()
    return torch.cat([p, ct], 1), torch.cat([oh, ct], 1)


def forward01(G, content, cond):
    return ((G(content, cond)[:, 0] + 1) / 2).clamp(0, 1)


__all__ = ["build", "inputs", "forward01", "N_LAB", "N_CT"]
