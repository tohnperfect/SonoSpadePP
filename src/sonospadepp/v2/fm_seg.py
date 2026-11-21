"""A3 (WP3): S_US initialized from a foundation model.

FMSegmenter mirrors usbg.segmenter.Segmenter's interface (.net, .n_classes, .group_map, save / load), so the
v1 trainer uses it unchanged: .net maps (B, 1, H, W) in [0, 1] to (B, n_classes, H, W) logits.

Network: the foundation encoder (fine-tuned with a lower learning rate, or frozen) gives token maps at four
depths; a light hybrid head fuses them (1x1 conv to 96 channels each, sum, two 3x3 convs) and upsamples to the
input size through two skip stages from a small full-resolution CNN stem (32 and 64 px features), so thin
structures that a 16 px token grid cannot resolve still have a path (an LR-ASPP / UPerNet-style compromise
sized for 128 px frames).
"""
from __future__ import annotations

import numpy as np


class _Net:
    def __new__(cls, enc_name, n_classes, layers, device):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from .features import FMEncoder

        class FMSegNet(nn.Module):
            def __init__(self):
                super().__init__()
                fe = FMEncoder(enc_name, "cpu")
                self.encoder = fe.model
                self.enc_name, self.size, self.layers = enc_name, fe.size, list(layers)
                self.swin = fe.pool == "swin"                     # EchoCare: Swin stages at several resolutions
                self.register_buffer("mean", fe.mean.clone(), persistent=False)
                self.register_buffer("std", fe.std.clone(), persistent=False)
                self.lat = nn.ModuleList([nn.Conv2d(c, 96, 1) for c in fe.layer_dims(self.layers)])
                self.fuse = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(True),
                                          nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(True))
                self.stem1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                                           nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),
                                           nn.ReLU(True))                       # H/2
                self.stem2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),
                                           nn.ReLU(True))                       # H/4
                self.dec2 = nn.Sequential(nn.Conv2d(96 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
                self.dec1 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True))
                self.cls = nn.Conv2d(32, n_classes, 1)

            def forward(self, x01):
                H, W = x01.shape[-2:]
                x = x01.clamp(0, 1).repeat(1, 3, 1, 1)
                x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
                if self.swin:
                    outs = self.encoder((x - self.mean) / self.std, normalize=True)
                    feats = [outs[i] for i in self.layers]
                else:
                    feats = self.encoder.forward_intermediates((x - self.mean) / self.std, indices=self.layers,
                                                               output_fmt="NCHW", intermediates_only=True)
                hw = feats[0].shape[-2:]                           # the finest of the chosen maps
                f = sum(F.interpolate(l(t), size=hw, mode="bilinear", align_corners=False)
                        if t.shape[-2:] != hw else l(t) for l, t in zip(self.lat, feats))
                f = self.fuse(f)
                s1 = self.stem1(x01)
                s2 = self.stem2(s1)
                f = F.interpolate(f, size=s2.shape[-2:], mode="bilinear", align_corners=False)
                f = self.dec2(torch.cat([f, s2], 1))
                f = F.interpolate(f, size=s1.shape[-2:], mode="bilinear", align_corners=False)
                f = self.dec1(torch.cat([f, s1], 1))
                return F.interpolate(self.cls(f), size=(H, W), mode="bilinear", align_corners=False)

        return FMSegNet().to(device)


class FMSegmenter:
    def __init__(self, enc_name, device="cpu", n_classes=9, group_map=None, layers=None):
        self.device = device
        if layers is None:                                   # ViT blocks 3/6/9/12; EchoCare's Swin stages 1-4
            layers = (1, 2, 3, 4) if enc_name == "echocare" else (2, 5, 8, 11)
        self.enc_name, self.layers = enc_name, tuple(layers)
        self.n_classes = n_classes
        self.group_map = None if group_map is None else np.asarray(group_map, np.int64)
        self.net = _Net(enc_name, n_classes, layers, device)

    def param_groups(self, lr, enc_lr_mult=0.1, freeze_encoder=False):
        enc = [p for p in self.net.encoder.parameters()]
        rest = [p for n, p in self.net.named_parameters() if not n.startswith("encoder.")]
        for p in enc:
            p.requires_grad_(not freeze_encoder)
        groups = [{"params": rest, "lr": lr}]
        if not freeze_encoder:
            groups.append({"params": enc, "lr": lr * enc_lr_mult})
        return groups

    def predict(self, img01):
        import torch
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(img01, np.float32))[None, None].to(self.device)
            return self.net(x)[0].argmax(0).cpu().numpy()

    def save(self, path):
        import torch
        torch.save({"kind": "fm_segmenter", "enc_name": self.enc_name, "layers": list(self.layers),
                    "n_classes": self.n_classes,
                    "group_map": None if self.group_map is None else self.group_map.tolist(),
                    "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        import torch
        st = torch.load(path, map_location="cpu", weights_only=True)
        seg = cls(st["enc_name"], device=device, n_classes=st["n_classes"], group_map=st["group_map"],
                  layers=st["layers"])
        seg.net.load_state_dict(st["state_dict"])
        seg.net.to(device)
        return seg


def load_any_segmenter(path, device="cpu"):
    """A v1 S_US (usbg.segmenter.Segmenter) or a v2 FMSegmenter, by checkpoint kind."""
    import torch
    st = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(st, dict) and st.get("kind") == "fm_segmenter":
        return FMSegmenter.load(path, device=device)
    from usbg import segmenter as SEG
    return SEG.Segmenter.load(path, device=device)
