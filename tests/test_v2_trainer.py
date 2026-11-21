"""SonoSPADEv2 keeps v1 exactly when its flags are off, and the A1 additions train end to end."""
import os

import numpy as np
import pytest
import torch
from usbg import segmenter as SEG
from usbg import texture_gan as TG

from sonospadepp.v2.trainer import SonoSPADEv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_PAPER_CKPT = os.path.join(ROOT, "data/v1_ckpt/exp_real_sonospade.pt")
USFM = os.path.join(ROOT, "data/fm/USFM_latest.pth")
KW = dict(n_classes=13, ngf=8, ndf=8, n_blocks=1, spade_hidden=8, d_depth=2, lambda_tc=1.0, lambda_lm=1.0,
          r1_gamma=1.0)


def _s_us():
    torch.manual_seed(123)
    return SEG.Segmenter(device="cpu", n_classes=SEG.N_GROUPED, base=8, depth=2,
                         group_map=SEG.grouped_map_array())


def _batch(seed, n=2, hw=32):
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 13, (n, 4, 4))
    blocks[:, 0, 0] = 2                                    # every grouped class the toy S_US may predict
    blocks[:, 0, 1] = 1
    lab = np.kron(blocks, np.ones((hw // 4, hw // 4), np.int64))     # blocky, so tokens are mostly one class
    phys = torch.from_numpy(rng.uniform(-1, 1, (n, 1, hw, hw)).astype(np.float32))
    real = torch.from_numpy(rng.uniform(-1, 1, (n, 1, hw, hw)).astype(np.float32))
    return phys, TG.onehot_labels(lab, 13, "cpu"), real


def _run(cls, steps=3, **extra):
    s_us = _s_us()
    torch.manual_seed(0)
    gan = cls("cpu", s_us=s_us, **KW, **extra)
    torch.manual_seed(1)
    outs = [gan.step(*_batch(i)) for i in range(steps)]
    return gan, outs


def test_v2_with_flags_off_is_bitwise_v1():
    g1, o1 = _run(TG.SonoSPADE)
    g2, o2 = _run(SonoSPADEv2, v2={"fm_disc": "none"}, fan=np.ones((32, 32)))
    assert o1 == o2
    for a, b in zip(g1.G.state_dict().values(), g2.G.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(g1.D.state_dict().values(), g2.D.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(g1.G_ema.state_dict().values(), g2.G_ema.state_dict().values()):
        assert torch.equal(a, b)


def test_diffaug_keeps_shape_and_gradient():
    from sonospadepp.v2.diffaug import diff_augment
    x = torch.rand(3, 1, 32, 32, requires_grad=True)
    y = diff_augment(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.skipif(not os.path.exists(USFM), reason="USFM weights not fetched")
@pytest.mark.parametrize("adv,tis", [(0.5, 0.0), (0.0, 1.0), (0.5, 1.0)])
def test_a1_step_trains_heads_and_bank(adv, tis):
    v2 = {"fm_disc": "usfm", "lambda_fm_adv": adv, "lambda_fm_tissue": tis, "fm_layers": [2, 5],
          "fm_min_updates": 1, "fm_min_cells": 1}
    gan, outs = _run(SonoSPADEv2, steps=3, v2=v2, fan=np.ones((32, 32)))
    last = outs[-1]
    assert all(np.isfinite(v) for v in last.values())
    if adv:
        assert "dfm" in last and "fm_adv" in last
    if tis:
        assert len(gan.bank.mu) > 0 and "fm_tissue" in last


def test_checkpoint_keeps_v1_keys_and_loads_in_both_translators(tmp_path):
    from sonospadepp.v2.translator import SonoSPADEv2Translator
    gan, _ = _run(SonoSPADEv2, steps=1, v2={"fm_disc": "none"})
    p = str(tmp_path / "v2.pt")
    gan.save(p, run_config={"note": "test"})
    st = torch.load(p, map_location="cpu", weights_only=True)
    for k in ("G_spade", "G_raw", "n_classes", "ngf", "n_blocks", "spade_hidden"):
        assert k in st
    assert st["variant"] == "sonospade_v2" and st["n_classes"] == 13
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    lab = np.random.default_rng(1).integers(0, 13, (32, 32))
    a = TG.SonoSPADETranslator(p, device="cpu").translate_aligned(img, lab)
    b = SonoSPADEv2Translator(p, device="cpu").translate_aligned(img, lab)
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not os.path.exists(V1_PAPER_CKPT), reason="v1 paper checkpoint not linked")
def test_v2_translator_reproduces_v1_on_released_checkpoint():
    from sonospadepp.v2.translator import SonoSPADEv2Translator
    img = np.random.default_rng(0).uniform(0, 1, (128, 128)).astype(np.float32)
    lab = np.random.default_rng(1).integers(0, 13, (128, 128))
    a = TG.SonoSPADETranslator(V1_PAPER_CKPT, device="cpu").translate_aligned(img, lab)
    b = SonoSPADEv2Translator(V1_PAPER_CKPT, device="cpu").translate_aligned(img, lab)
    np.testing.assert_array_equal(a, b)


def test_style_and_speckle_paths_train_and_roundtrip(tmp_path):
    from sonospadepp.v2.translator import SonoSPADEv2Translator
    v2 = {"fm_disc": "none", "style": "sean", "style_dim": 4, "lambda_speckle": 1.0, "noise_inject": True}
    gan, outs = _run(SonoSPADEv2, steps=3, v2=v2, fan=np.ones((32, 32)))
    last = outs[-1]
    assert all(np.isfinite(v) for v in last.values()) and "style_rec" in last
    gan.build_style_bank(np.random.default_rng(0).uniform(0, 1, (6, 32, 32)))
    p = str(tmp_path / "v2s.pt")
    gan.save(p)
    tr = SonoSPADEv2Translator(p, device="cpu")
    assert tr.has_style
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    lab = np.random.default_rng(1).integers(0, 13, (32, 32))
    a = tr.translate_aligned(img, lab)
    b = tr.translate_aligned(img, lab, style={1: "sample"})
    assert a.shape == b.shape == (32, 32)
    codes = tr.encode_styles(img, lab)
    assert len(codes) > 0


def test_sean_and_noise_generator_start_as_v1():
    from sonospadepp.v2.generator import build_generator
    from sonospadepp.v2.style import convert_to_sean
    torch.manual_seed(0)
    v1 = TG._spade_generator(13, ngf=8, n_blocks=1, spade_hidden=8)
    v2 = convert_to_sean(build_generator(13, ngf=8, n_blocks=1, spade_hidden=8, noise_inject=True), 4, 13)
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert not unexpected
    assert all(("style_" in k) or ("alpha_" in k) or k.startswith("noise_scale") for k in missing)
    phys = torch.rand(2, 1, 32, 32) * 2 - 1
    seg = TG.onehot_labels(np.random.default_rng(0).integers(0, 13, (2, 32, 32)), 13)
    cond = torch.cat([seg, torch.randn(2, 4, 32, 32)], 1)
    with torch.no_grad():
        # alpha = 0 and noise scale = 0: v1's function (noise alone is bit-exact; the SEAN path differs only by
        # float-level kernel differences, max ~7e-7, on the sliced conditioning tensor)
        assert torch.allclose(v1(phys, seg), v2(phys, cond), atol=1e-5, rtol=0)


def test_v2_generator_survives_sonospadepp_channel_widening():
    """Emulates SonoSpadePP's paired_translator.build on a SEAN + noise v2 generator: rebuild with 2 extra
    conditioning channels and 2 extra stem inputs, copy weights, zero-fill the new input columns. With the extra
    channels present, the widened generator must reproduce the original output."""
    import torch.nn as nn
    from sonospadepp.v2.generator import build_generator
    from sonospadepp.v2.style import convert_to_sean
    torch.manual_seed(0)

    def make(n_cond):
        return convert_to_sean(build_generator(n_cond, ngf=8, n_blocks=1, spade_hidden=8, noise_inject=True), 4, n_cond)

    g13 = make(13)
    with torch.no_grad():                                  # trained-looking: style and noise switched on
        for m in g13.modules():
            if hasattr(m, "alpha_g"):
                m.alpha_g.fill_(0.3); m.alpha_b.fill_(0.2)
        for p in g13.noise_scale:
            p.normal_(0, 0.1)
    g15 = make(15)
    new = g15.state_dict()
    for k, v in g13.state_dict().items():
        if new[k].shape == v.shape:
            new[k] = v.clone()
        else:
            w = torch.zeros_like(new[k]); w[:, :v.shape[1]] = v; new[k] = w
    g15.load_state_dict(new)
    conv = g15.stem[1]
    c2 = nn.Conv2d(3, conv.out_channels, conv.kernel_size)
    with torch.no_grad():
        c2.weight.zero_(); c2.weight[:, :1] = conv.weight; c2.bias.copy_(conv.bias)
    g15.stem[1] = c2
    phys = torch.rand(2, 1, 32, 32) * 2 - 1
    oh = TG.onehot_labels(np.random.default_rng(0).integers(0, 13, (2, 32, 32)), 13)
    st = torch.randn(2, 4, 32, 32)
    extra = torch.randn(2, 2, 32, 32)
    g13.noise_gen = torch.Generator().manual_seed(0)
    g15.noise_gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        a = g13(phys, torch.cat([oh, st], 1))
        b = g15(torch.cat([phys, extra], 1), torch.cat([oh, extra, st], 1))
    assert torch.allclose(a, b, atol=1e-5, rtol=0)
