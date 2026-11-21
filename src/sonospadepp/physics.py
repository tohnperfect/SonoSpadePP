"""physics.py: the SonoSPADE physics renderer, driven by per-pixel acoustic maps.

`render_from_maps` is the body of usbg.render_lotus.render_bmode_physics (SonoSpade v0.2.0) with the
per-label map lookup lifted out, so a caller can hand it per-pixel impedance, attenuation and
backscatter. tests/test_physics.py checks it is bit-identical to the release on plain label maps.

Three ways to build the maps:
  maps_from_labels(lab, cfg)          the release lookup (B0, M3), extended to the 14-label M1 scheme
  maps_from_labels(lab_pp, cfg)       M1: same lookup, the GAS label carries air acoustics
  maps_hu_modulated(lab, hu, z, ...)  M2: label acoustics modulated per pixel by CT intensity

M2 modulation (soft tissues and fluids only; air, bone, lung and background are left alone):
  impedance    Z = Z_label * (1 + HU/1000) / (1 + ref_label/1000)
               CT number tracks density and Z = density * sound speed, so a gas pocket inside the
               bowel drops Z to a few percent and reflects strongly, and a vessel wall makes a weak
               interface. Contrast agent raises HU without raising density, a known error.
  backscatter  BS = BS_label * exp(-bs_k * min(|z|, z_max))
               pixels unlike their organ's parenchyma (vessels, cysts, lesions, gas) scatter less,
               whatever the sign of the HU difference, so the rule holds in any contrast phase.
"""
from __future__ import annotations

import numpy as np
from usbg import render_lotus as R
from usbg import volume as V

from .labels import ACOUSTIC_PP, GAS, HU_MODULATED, N_PP

Z_FACTOR_RANGE = (0.02, 3.0)


def _tables(cfg):
    """Per-label (Z, MU, BS) with cfg scales, multiplied in float32 exactly as the release does."""
    Z0 = np.zeros(N_PP, np.float32); MU0 = np.zeros(N_PP, np.float32); BS0 = np.zeros(N_PP, np.float32)
    zs = np.ones(N_PP, np.float32); ms = np.ones(N_PP, np.float32); bs = np.ones(N_PP, np.float32)
    for L, (z, a, m0, _m1, _s0) in ACOUSTIC_PP.items():
        src = V.LUNG if L == GAS else L                   # GAS takes LUNG's domain randomization
        Z0[L], MU0[L], BS0[L] = z, a, m0
        zs[L] = cfg.z_scale.get(src, 1.0); ms[L] = cfg.mu_scale.get(src, 1.0)
        bs[L] = cfg.bs_scale.get(src, 1.0)
    return Z0 * zs, MU0 * ms, BS0 * bs


def maps_from_labels(label_slice: np.ndarray, cfg):
    """Per-pixel (Z, MU, BS) float32 numpy maps from a 13- or 14-label slice."""
    lab = np.asarray(label_slice).astype(np.int64)
    Z0, MU0, BS0 = _tables(cfg)
    return Z0[lab], MU0[lab], BS0[lab]


def maps_hu_modulated(label_slice, hu_slice, z_slice, ref, cfg, *, bs_k: float = 0.35,
                      z_max: float = 4.0):
    """M2: label maps modulated by CT intensity. ref = per-label tissue median HU (PPVolume.ref)."""
    lab = np.asarray(label_slice).astype(np.int64)
    Z, MU, BS = maps_from_labels(lab, cfg)
    mod = np.isin(lab, HU_MODULATED) & ~np.isnan(np.asarray(ref)[lab])
    r = np.nan_to_num(np.asarray(ref, np.float32), nan=0.0)[lab]
    fz = (1.0 + np.asarray(hu_slice, np.float32) / 1000.0) / (1.0 + r / 1000.0)
    fz = np.clip(fz, *Z_FACTOR_RANGE)
    fbs = np.exp(-bs_k * np.minimum(np.abs(np.asarray(z_slice, np.float32)), z_max))
    Z = np.where(mod, Z * fz, Z).astype(np.float32)
    BS = np.where(mod, BS * fbs, BS).astype(np.float32)
    return Z, MU, BS


def render_from_maps(Z, MU, BS, cfg=None, device=None) -> np.ndarray:
    """Release physics renderer on per-pixel maps (N_LINES, N_SAMPLES). Returns float32
    (N_SAMPLES, N_LINES) in [0,1], deterministic given (maps, cfg, cfg.seed)."""
    import torch

    cfg = cfg or R.RenderConfig()
    device = device or R.get_device()
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=device)
    Z, MU, BS = t(Z), t(MU), t(BS)
    L, S = Z.shape
    dz = R.IMG_DEPTH_M / S
    dz_cm = dz * 100.0
    depth = np.linspace(0.0, R.IMG_DEPTH_M, S, dtype=np.float32)
    depth_t = torch.as_tensor(depth, device=device)[None, :]

    eps = 1e-6
    Rc = torch.zeros_like(Z)
    dZ = Z[:, 1:] - Z[:, :-1]
    Rc[:, 1:] = (dZ / (Z[:, 1:] + Z[:, :-1] + eps)) ** 2
    Rc = torch.clamp(Rc * cfg.shadow_gain, 0.0, 0.999)
    T_cum = torch.cumprod(1.0 - Rc, dim=1)
    A_cum = torch.exp(-2.0 * R.ATTEN_COEF * torch.cumsum(MU * dz_cm, dim=1))
    shadow = T_cum * A_cum

    g = torch.Generator().manual_seed(int(cfg.seed))
    w_re = torch.randn(L, S, generator=g).to(device)
    w_im = torch.randn(L, S, generator=g).to(device)
    kz = R._gauss1d_kernel(cfg.psf_sigma_ax, device)
    env_re = R._lateral_depthvarying(R._conv_axis(w_re, kz, 1), cfg, depth, device)
    env_im = R._lateral_depthvarying(R._conv_axis(w_im, kz, 1), cfg, depth, device)
    env = torch.sqrt(env_re ** 2 + env_im ** 2 + eps)

    scat = BS * cfg.speckle_strength * env * shadow
    spec = R.REFL_GAIN * Rc * shadow
    echo = R._conv_axis(scat + spec, kz, 1)
    echo = R._lateral_depthvarying(echo, cfg, depth, device)

    depth_cm = depth_t * 100.0
    tgc = torch.exp(2.0 * R.ATTEN_COEF * R.TGC_REF_MU * cfg.tgc_slope * depth_cm)
    echo = echo * tgc * cfg.gain
    echo = echo + cfg.noise_sigma * torch.randn(L, S, generator=g).to(device)
    echo = torch.clamp(echo, min=0.0)

    db = 20.0 * torch.log10(echo + 1e-4)
    top = torch.quantile(db.flatten(), R.TOP_PCTILE)
    img01 = torch.clamp((db - (top - cfg.dyn_range_db)) / cfg.dyn_range_db, 0.0, 1.0)
    img01 = img01 ** R.GAMMA
    return img01.t().contiguous().cpu().numpy().astype(np.float32)


def render_parts(Z, MU, cfg=None, device=None):
    """The parts of render_from_maps that do not depend on backscatter, for a fixed (Z, MU, seed):

        echo = A(BS * P) * tgc * gain + E,   A = axial PSF then depth-varying lateral PSF (linear)

    P = speckle envelope x transmission x attenuation, E = the specular echo through A, gain
    compensation and gain, plus the additive noise. With these, render_from_parts is differentiable
    in BS and equals render_from_maps (tests/test_physics.py)."""
    import torch

    cfg = cfg or R.RenderConfig()
    device = device or R.get_device()
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=device)
    Z, MU = t(Z), t(MU)
    L, S = Z.shape
    dz_cm = R.IMG_DEPTH_M / S * 100.0
    depth = np.linspace(0.0, R.IMG_DEPTH_M, S, dtype=np.float32)
    depth_t = torch.as_tensor(depth, device=device)[None, :]
    eps = 1e-6
    Rc = torch.zeros_like(Z)
    Rc[:, 1:] = ((Z[:, 1:] - Z[:, :-1]) / (Z[:, 1:] + Z[:, :-1] + eps)) ** 2
    Rc = torch.clamp(Rc * cfg.shadow_gain, 0.0, 0.999)
    shadow = torch.cumprod(1.0 - Rc, dim=1) * torch.exp(-2.0 * R.ATTEN_COEF * torch.cumsum(MU * dz_cm, dim=1))
    g = torch.Generator().manual_seed(int(cfg.seed))
    w_re = torch.randn(L, S, generator=g).to(device)
    w_im = torch.randn(L, S, generator=g).to(device)
    kz = R._gauss1d_kernel(cfg.psf_sigma_ax, device)
    env_re = R._lateral_depthvarying(R._conv_axis(w_re, kz, 1), cfg, depth, device)
    env_im = R._lateral_depthvarying(R._conv_axis(w_im, kz, 1), cfg, depth, device)
    env = torch.sqrt(env_re ** 2 + env_im ** 2 + eps)
    P = cfg.speckle_strength * env * shadow
    tg = torch.exp(2.0 * R.ATTEN_COEF * R.TGC_REF_MU * cfg.tgc_slope * depth_t * 100.0) * cfg.gain
    spec = R._lateral_depthvarying(R._conv_axis(R.REFL_GAIN * Rc * shadow, kz, 1), cfg, depth, device)
    E = spec * tg + cfg.noise_sigma * torch.randn(L, S, generator=g).to(device)
    return {"P": P, "E": E, "tg": tg, "kz": kz, "depth": depth, "cfg": cfg, "device": device}


def render_from_parts(BS, parts):
    """Differentiable render from a torch backscatter map (L, S) and render_parts(...). Returns the
    image (S, L) in [0,1] as a torch tensor."""
    import torch

    cfg, device = parts["cfg"], parts["device"]
    scat = R._lateral_depthvarying(R._conv_axis(BS * parts["P"], parts["kz"], 1), cfg, parts["depth"], device)
    echo = torch.clamp(scat * parts["tg"] + parts["E"], min=0.0)
    db = 20.0 * torch.log10(echo + 1e-4)
    top = torch.quantile(db.flatten(), R.TOP_PCTILE)
    img01 = torch.clamp((db - (top - cfg.dyn_range_db)) / cfg.dyn_range_db, 0.0, 1.0)
    return (img01 ** R.GAMMA).t()


__all__ = ["maps_from_labels", "maps_hu_modulated", "render_from_maps", "render_parts", "render_from_parts"]
