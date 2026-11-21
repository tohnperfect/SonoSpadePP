"""INADE (WP2 fallback): with zero codes a converted generator reproduces the unconverted one exactly (mu starts at
0, so the class mean is SPADE's modulation), and non-zero codes change the output inside the coded tissue only
through the modulation (CPU, tiny generator)."""
import pytest

torch = pytest.importorskip("torch")


def _gen():
    from sonospadepp.v2.generator import build_generator
    torch.manual_seed(0)
    return build_generator(13, ngf=8, n_blocks=1, spade_hidden=8).eval()


def test_zero_codes_reproduce_spade_and_codes_act():
    from sonospadepp.v2.style import convert_to_inade
    g0 = _gen()
    g1 = _gen()
    g1.load_state_dict(g0.state_dict())
    convert_to_inade(g1, 4, 13)
    g1.eval()
    phys = torch.rand(1, 1, 32, 32) * 2 - 1
    lab = torch.zeros(1, 13, 32, 32)
    lab[:, 1, :, :16] = 1
    lab[:, 4, :, 16:] = 1
    with torch.no_grad():
        ref = g0(phys, lab)
        same = g1(phys, torch.cat([lab, torch.zeros(1, 4, 32, 32)], 1))
        diff = g1(phys, torch.cat([lab, torch.randn(1, 4, 1, 1).expand(1, 4, 32, 32)], 1))
    assert torch.allclose(ref, same, atol=1e-6)
    assert (diff - ref).abs().max() > 1e-4
