"""Gain-field generator: the network changes only the low-frequency part of the render, and a plain generator's
state dict loads into it unchanged (no new persistent keys)."""
import torch

from sonospadepp.v2.generator import build_generator


def _hf_energy(d):
    """Mean squared response of a 3x3 Laplacian (high-frequency energy)."""
    lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    return float(torch.nn.functional.conv2d(d, lap).pow(2).mean())


def test_gain_field_keeps_render_fine_structure():
    torch.manual_seed(0)
    plain = build_generator(13).eval()
    gain = build_generator(13, gain_sigma=4.0).eval()
    gain.load_state_dict(plain.state_dict(), strict=True)       # same parameters, no extra persistent keys
    phys = torch.rand(2, 1, 128, 128) * 1.6 - 0.8
    seg = torch.nn.functional.one_hot(torch.randint(0, 13, (2, 128, 128)), 13).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        d_plain = plain(phys, seg) - phys
        d_gain = gain(phys, seg) - phys
    assert _hf_energy(d_gain) < 0.01 * _hf_energy(d_plain)


def test_gain_off_is_the_plain_generator():
    g = build_generator(13, gain_sigma=0.0)
    assert not hasattr(g, "gain_sigma")
