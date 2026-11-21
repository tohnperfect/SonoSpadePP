"""render_from_maps must reproduce SonoSpade's physics renderer exactly on plain label maps, so any
difference between b0 and m1/m2 comes from the CT maps, not from the copy."""
import numpy as np
import pytest
from usbg import render_lotus as R
from usbg import volume as V

from sonospadepp.labels import GAS
from sonospadepp.physics import maps_from_labels, maps_hu_modulated, render_from_maps


def _label_slice():
    lab = np.full((64, 96), V.SOFT_TISSUE, np.uint8)
    lab[:, :8] = V.BACKGROUND
    lab[10:40, 20:70] = V.LIVER
    lab[25:30, 30:60] = V.VESSEL
    lab[45:60, 40:90] = V.GI
    lab[5:12, 80:90] = V.BONE
    return lab


@pytest.mark.parametrize("seed", [0, 7])
def test_render_from_maps_matches_release(seed):
    lab = _label_slice()
    cfg = R.domain_randomize(R.RenderConfig(), np.random.default_rng(seed))
    ref = R.render_bmode_physics(lab, cfg, device="cpu")
    out = render_from_maps(*maps_from_labels(lab, cfg), cfg, device="cpu")
    assert np.array_equal(ref, out)


def test_gas_label_uses_air_acoustics():
    lab = _label_slice()
    lab_pp = lab.copy()
    lab_pp[50:55, 50:70] = GAS
    cfg = R.RenderConfig()
    Z, _, _ = maps_from_labels(lab_pp, cfg)
    assert np.allclose(Z[lab_pp == GAS], V.ACOUSTIC_LOTUS[V.LUNG][0])


def test_hu_modulation_is_identity_at_reference():
    lab = _label_slice()
    cfg = R.RenderConfig()
    ref = np.full(14, np.nan, np.float32)
    for L in (V.LIVER, V.VESSEL, V.GI, V.SOFT_TISSUE):
        ref[L] = 50.0
    hu = np.full(lab.shape, 50.0, np.float32)
    z = np.zeros(lab.shape, np.float32)
    a = maps_from_labels(lab, cfg)
    b = maps_hu_modulated(lab, hu, z, ref, cfg)
    for x, y in zip(a, b):
        assert np.allclose(x, y)


def test_hu_modulation_gas_reflects():
    lab = _label_slice()
    cfg = R.RenderConfig()
    ref = np.full(14, np.nan, np.float32); ref[V.GI] = 30.0
    hu = np.full(lab.shape, 30.0, np.float32); hu[50:55, 50:70] = -900.0
    z = (hu - 30.0) / 15.0
    Z, _, BS = maps_hu_modulated(lab, hu, z, ref, cfg)
    assert Z[52, 60] < 0.2 * V.ACOUSTIC_LOTUS[V.GI][0]
    assert BS[52, 60] < BS[47, 45]


def test_render_from_parts_matches_render_from_maps():
    """The split renderer (fixed parts + differentiable backscatter) equals the full renderer."""
    import torch
    from dataclasses import replace
    from usbg import render_lotus as R
    from sonospadepp.physics import maps_from_labels, render_from_maps, render_parts, render_from_parts
    rng = np.random.default_rng(3)
    lab = np.full((160, 512), 12, np.uint8)
    lab[:, 200:330] = 4; lab[40:90, 250:300] = 9; lab[:, :20] = 9
    cfg = replace(R.RenderConfig(), seed=11)
    dev = torch.device("cpu")
    Z, MU, BS = maps_from_labels(lab, cfg)
    BS = BS * rng.uniform(0.6, 1.4, BS.shape).astype(np.float32)
    ref = render_from_maps(Z, MU, BS, cfg, dev)
    parts = render_parts(Z, MU, cfg, dev)
    bs_t = torch.tensor(BS, requires_grad=True)
    out = render_from_parts(bs_t, parts)
    assert np.allclose(out.detach().numpy(), ref, atol=1e-4)
    out.mean().backward()
    assert bs_t.grad is not None and torch.isfinite(bs_t.grad).all() and bs_t.grad.abs().sum() > 0
