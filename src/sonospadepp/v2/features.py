"""Frozen foundation-model encoders for v2: per-tissue KID / FD features (metrics) and the D_fm branch (WP1).

All encoders take single-channel B-mode tensors in [0, 1], (B, 1, H, W), repeat them to 3 channels,
resize bilinearly to the encoder's input size and normalize with the encoder's own statistics.

  dinov2    DINOv2 ViT-B/14 (Oquab et al. 2024), timm `vit_base_patch14_dinov2.lvd142m` (Apache-2.0),
            run at 224 px (position embedding interpolated by timm). Global feature: the CLS token.
  usfm      USFM (Jiao et al., Med. Image Anal. 2024), `USFM_latest.pth` from the openmedlab/USFM README (CC BY-NC 4.0,
            research use). A BEiT-style ViT-B/16 at 224 px with a shared relative position bias and layer
            scale, loaded into timm's Beit. Preprocessing follows the repo's data pipeline (RGB, ImageNet
            mean/std). Global feature: mean of the final-norm patch tokens (USFM's config sets mean pooling).
  echocare  EchoCare (arXiv 2509.11752), weights from Hugging Face CAIR-HKISI/EchoCare (a torch zip checkpoint
            named model.safetensors, renamed echocare_encoder.pth; no license stated). MONAI SwinTransformer,
            patch 2, window 8, depths 2-2-18-2, at 256 px; preprocessing from utils/data_utils.py of the repo: RGB,
            ToTensor to [0, 1], no mean/std normalization. TRAINING ONLY (decided 2026-11-16): never an
            evaluation extractor (license unclear, pretraining overlap with the Kaggle set undetermined).

Weights are downloaded by hand into data/fm/ (or $SONOSPADE_FM_DIR) and never committed: USFM as
data/fm/USFM_latest.pth (its SHA-256 is USFM_SHA256 below), EchoCare as data/fm/echocare/echocare_encoder.pth;
timm downloads DINOv2 on first use. The README lists the sources and their terms.
"""
from __future__ import annotations

import os

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FM_DIR = os.environ.get("SONOSPADE_FM_DIR", "data/fm")
USFM_SHA256 = "d5fdab3edd140e4ca61471bb4087f91cd7ff2ce270db71b9cab30feda881bd17"


def _build(name: str):
    import timm
    import torch
    if name == "dinov2":
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224,
                              num_classes=0)
        return m, 224, "cls"
    if name == "usfm":
        from timm.models.beit import Beit
        m = Beit(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 qkv_bias=True, use_abs_pos_emb=False, use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=True, init_values=0.1, num_classes=0, global_pool="token")
        sd = torch.load(os.path.join(FM_DIR, "USFM_latest.pth"), map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if k != "mask_token"}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"USFM load mismatch: missing {missing}, unexpected {unexpected}")
        return m, 224, "mean"
    if name == "echocare":
        from monai.networks.nets.swin_unetr import SwinTransformer
        m = SwinTransformer(in_chans=3, embed_dim=128, window_size=(8, 8), patch_size=(2, 2), depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32), mlp_ratio=4.0, qkv_bias=True, spatial_dims=2, use_v2=True)
        sd = torch.load(os.path.join(FM_DIR, "echocare", "echocare_encoder.pth"), map_location="cpu",
                        weights_only=True)
        m.load_state_dict({k: v for k, v in sd.items() if k != "mask_token"}, strict=True)
        return m, 256, "swin"
    raise ValueError(f"unknown encoder {name!r}")


class FMEncoder:
    """A frozen encoder (eval mode, no grad on its weights). Gradients still flow to the INPUT, so the
    same object serves the D_fm branch, where the generator is trained through it."""

    def __init__(self, name: str, device="cpu"):
        import torch
        self.name = name
        self.device = torch.device(device)
        self.model, self.size, self.pool = _build(name)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        mean, std = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) if self.pool == "swin" else (IMAGENET_MEAN, IMAGENET_STD)
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.dim = 2048 if self.pool == "swin" else self.model.num_features

    def layer_dims(self, indices):
        """Channel count of each map returned by intermediates(indices)."""
        if self.pool == "swin":
            return [128 * 2 ** i for i in indices]
        return [self.dim] * len(indices)

    def prep(self, x01):
        import torch.nn.functional as F
        x = x01.clamp(0, 1).repeat(1, 3, 1, 1)
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def tokens(self, x01):
        """Final-norm token sequence (B, 1 + N, D) (ViT encoders only)."""
        return self.model.forward_features(self.prep(x01))

    def global_feat(self, x01):
        if self.pool == "swin":
            return self.model(self.prep(x01), normalize=True)[-1].mean(dim=(2, 3))
        t = self.tokens(x01)
        return t[:, 0] if self.pool == "cls" else t[:, 1:].mean(1)

    def intermediates(self, x01, indices):
        """Feature maps (B, C, h, w) at the given depths, for the D_fm heads (WP1): ViT blocks for DINOv2 and
        USFM, Swin stages (0 = 128 ch at H/2 ... 4 = 2048 ch at H/32) for EchoCare."""
        if self.pool == "swin":
            outs = self.model(self.prep(x01), normalize=True)
            return [outs[i] for i in indices]
        return self.model.forward_intermediates(self.prep(x01), indices=indices, output_fmt="NCHW",
                                                intermediates_only=True)


def extract_global(enc: FMEncoder, imgs01, batch: int = 32) -> np.ndarray:
    """(N, h, w) float [0, 1] -> (N, D) float64 global features."""
    import torch
    imgs01 = np.asarray(imgs01, np.float32)
    if len(imgs01) == 0:
        return np.empty((0, enc.dim))
    out = []
    with torch.no_grad():
        for j in range(0, len(imgs01), batch):
            x = torch.from_numpy(imgs01[j:j + batch][:, None]).to(enc.device)
            out.append(enc.global_feat(x).float().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)
